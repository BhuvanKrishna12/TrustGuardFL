"""
test_sampling_mv.py
-------------------------------------------------------------------
Checks the revision-2 training-data passes on multi-variant files built from
synthetic raw hours (two "days"), against the round-1 code run on real v3 and
v2 pipeline outputs of the same raw data:

 1. per-file cached stats: design CN == train_file_stats(v3 files),
    design LF == train_file_stats(v2 files);
 2. uniform sampler: the corrected view of the ONE union sample is exactly
    uniform_sample(v3 files) (same rows, same order); the legacy view is an
    exact capped sample of the v2 files (every bucket has min(cap, bucket
    size) rows, all rows exist in v2 with v2 labels and v2 feature values);
 3. legacy loader: legacy_sample_view on the mv files equals
    federated_simulation.load_data on the v3 files (corrected) and on the v2
    files (legacy), row for row, including the re-chunking after the filter;
 4. both samplers resume after a stop between files with identical results.

Usage: python3 experiments/test_sampling_mv.py [WORK_DIR]
"""

import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(HERE, "reference_r1"))

import federated_simulation as fs  # noqa: E402
import make_synthetic_raw as msr  # noqa: E402
import opensky_fl_ids_pipeline as pl  # noqa: E402
import sampling  # noqa: E402
import sampling_r1  # noqa: E402
import variants as V  # noqa: E402

quiet = lambda *a, **k: None  # noqa: E731
BUDGET, CHUNK = 60_000, 7_000


def build(work):
    files = {"v2": [], "v3": [], "mv": []}
    for date in ("2019-01-07", "2019-04-01"):
        raw = msr.write_day(os.path.join(work, "raw"), date, hours=3, n_aircraft=140, seed=3)
        for ver in files:
            out = os.path.join(work, f"{ver}_opensky_{date}_labeled.csv")
            with contextlib.redirect_stdout(io.StringIO()):
                pl.main(raw, chunk_files=1, out_path=out, pipeline_version=ver)
            files[ver].append(out)
    return files


def eq(a, b, cols, what):
    assert len(a) == len(b), f"{what}: {len(a)} vs {len(b)} rows"
    for c in cols:
        x, y = a[c].reset_index(drop=True), b[c].reset_index(drop=True)
        if c == "icao24":       # the revision-2 samplers store the address as an integer
            x = pd.Series(sampling.icao_to_u32(x)) if not pd.api.types.is_integer_dtype(x) else x
            y = pd.Series(sampling.icao_to_u32(y)) if not pd.api.types.is_integer_dtype(y) else y
        if pd.api.types.is_float_dtype(x) or pd.api.types.is_float_dtype(y):
            ok = np.isclose(x.to_numpy(dtype="float64"), y.to_numpy(dtype="float64"),
                            rtol=0, atol=0, equal_nan=True)
        else:
            ok = x.astype(str).to_numpy() == y.astype(str).to_numpy()
        assert ok.all(), f"{what}: column {c} differs in {int((~ok).sum())} rows"


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)
    cache = os.path.join(work, "cache")
    os.makedirs(cache, exist_ok=True)
    F = build(work)
    cols = fs.FEATURE_COLS + ["region_cell", "attack_type", "icao24", "time", "lastposupdate"]

    # 1. stats
    st, icao = sampling.train_file_stats_cached(F["mv"], cache, chunksize=5000, log=quiet)
    st2, icao2 = sampling.train_file_stats_cached(F["mv"], cache, log=quiet)      # from cache
    assert st["designs"] == st2["designs"] and icao == icao2
    r3, i3 = sampling_r1.train_file_stats(F["v3"], log=quiet)
    r2, _ = sampling_r1.train_file_stats(F["v2"], log=quiet)
    assert st["total_rows"] == r3["total_rows"] and st["class_counts"] == r3["class_counts"]
    assert st["regions"] == r3["regions"] and icao == i3
    lf = st["designs"]["LF"]
    assert lf["total_rows"] == r2["total_rows"] and lf["class_counts"] == r2["class_counts"]
    assert lf["regions"] == r2["regions"]
    assert set(st["designs"]) == set(V.DESIGNS)
    print(f"[1] stats: CN == v3 files ({st['total_rows']:,} rows), LF == v2 files "
          f"({lf['total_rows']:,} rows), per-file cache reused")

    # 2. uniform sampler
    union, info = sampling.uniform_sample_mv(F["mv"], BUDGET, st, chunksize=CHUNK, log=quiet)
    ref, rinfo = sampling_r1.uniform_sample(F["v3"], BUDGET, r3["regions"], sorted(r3["class_counts"]),
                                            chunksize=CHUNK, log=quiet)
    corr = sampling.sample_view(union, info, V.parse_variant("corrected"))
    eq(corr, ref, cols + ["_src"], "uniform corrected view vs round-1 uniform_sample")
    assert info["cap_per_region_class"] == rinfo["cap_per_region_class"]
    st3, _ = sampling.train_file_stats_cached(F["v3"], cache, log=quiet)
    plain, pinfo = sampling.uniform_sample_mv(F["v3"], BUDGET, st3, chunksize=CHUNK, log=quiet)
    eq(sampling.sample_view(plain, pinfo, frozenset()), ref, cols + ["_src"], "plain v3 file")
    cap = info["caps_by_design"]["LF"]
    leg = sampling.sample_view(union, info, V.parse_variant("legacy"))
    v2 = pd.concat([pd.read_csv(p, usecols=cols, low_memory=False) for p in F["v2"]], ignore_index=True)
    want = v2.groupby(["region_cell", "attack_type"]).size().clip(upper=cap)
    got = leg.groupby(["region_cell", "attack_type"], observed=True).size()
    assert want.sort_index().to_dict() == got.sort_index().to_dict(), "legacy view bucket sizes"
    assert (want == cap).sum() > 3, "no capped bucket in the test data"
    key = ["icao24", "time", "lastposupdate", "attack_type"]
    v2["icao24"] = sampling.icao_to_u32(v2["icao24"])
    m = leg.merge(v2.drop_duplicates(key), on=key, how="left", suffixes=("", "_v2"), indicator=True)
    assert (m["_merge"] == "both").all(), "legacy view has rows that are not in the v2 files"
    for c in ("track_message_count", "squawk_changed", "implied_speed_mps", "is_physically_implausible"):
        a, b = m[c], m[c + "_v2"]
        ok = (np.isclose(a.astype("float64"), b.astype("float64"), rtol=1e-6, equal_nan=True)
              if pd.api.types.is_float_dtype(a) else a.astype(str) == b.astype(str))
        assert np.all(ok), f"legacy view feature {c} != v2 value"
    # every variant's view is a full capped sample of its design
    for spec in ("loo:C3", "add:C3", "prefloor", "loo:C1", "add:C5"):
        ls = V.parse_variant(spec)
        d = V.design_of(ls)
        vv = sampling.sample_view(union, info, ls)
        assert vv.groupby(["region_cell", "attack_type"], observed=True).size().max() <= info["caps_by_design"][d]
    print(f"[2] uniform sampler: corrected view == round-1 uniform_sample ({len(ref):,} rows, same order); "
          f"legacy view is an exact capped sample of the v2 files ({len(leg):,} rows); "
          f"union {len(union):,} rows")

    # 3. legacy loader
    for spec, ver in (("corrected", "v3"), ("legacy", "v2")):
        ls = V.parse_variant(spec)
        mine, linfo = sampling.legacy_sample_view(F["mv"], BUDGET, st, ls, chunksize=CHUNK, log=quiet)
        mine = sampling.features_view(mine, ls)
        with contextlib.redirect_stdout(io.StringIO()):
            ref_l = fs.load_data(F[ver], max_total_rows=BUDGET, chunksize=CHUNK)
        eq(mine, ref_l, cols, f"legacy loader, {spec} view vs load_data({ver} files)")
        print(f"[3] legacy loader: {spec} view == load_data({ver} files) ({len(ref_l):,} rows)")

    # 4. resume after a stop between files
    class Stop:
        def __init__(self, n):
            self.n, self.calls = n, 0

        def __call__(self):
            self.calls += 1
            return self.calls > self.n
    n_chunks_file1 = int(np.ceil(st["per_file"][0]["rows"] / CHUNK))
    sp = os.path.join(work, "u.state.pkl")
    try:
        sampling.uniform_sample_mv(F["mv"], BUDGET, st, chunksize=CHUNK, log=quiet, state_path=sp,
                                   stop_check=Stop(n_chunks_file1 + 2))
        raise AssertionError("stop was not raised")
    except sampling.StopRequested:
        pass
    assert os.path.exists(sp)
    msgs = []
    union2, info2 = sampling.uniform_sample_mv(F["mv"], BUDGET, st, chunksize=CHUNK,
                                               log=msgs.append, state_path=sp)
    assert any("resuming after 1 finished file" in m for m in msgs), msgs
    eq(union2, union, list(union.columns), "uniform sampler resumed vs uninterrupted")
    lp = os.path.join(work, "l.state.pkl")
    ls = V.parse_variant("legacy")
    full, _ = sampling.legacy_sample_view(F["mv"], BUDGET, st, ls, chunksize=CHUNK, log=quiet)
    try:
        sampling.legacy_sample_view(F["mv"], BUDGET, st, ls, chunksize=CHUNK, log=quiet, state_path=lp,
                                    stop_check=Stop(n_chunks_file1 + 2))
        raise AssertionError("stop was not raised")
    except sampling.StopRequested:
        pass
    res, _ = sampling.legacy_sample_view(F["mv"], BUDGET, st, ls, chunksize=CHUNK, log=quiet, state_path=lp)
    eq(res, full, list(full.columns), "legacy loader resumed vs uninterrupted")
    print("[4] both samplers: stop after file 1, resume -> identical to an uninterrupted pass")
    print("SAMPLING OK")


if __name__ == "__main__":
    main()
