"""
test_variants.py
-------------------------------------------------------------------
Proves, on synthetic RAW hourly files, that the multi-variant pipeline output
contains pipeline v3 and pipeline v2 exactly:

 1. the non-legacy columns of the mv file are the v3 file, byte for byte;
 2. the legacy view (all *_legacy columns, legacy_filter_drop rows removed)
    is the v2 file: same rows in the same order, same labels, same features;
 3. apply_view('legacy') keep-mask == ~legacy_filter_drop, 'corrected' keeps all;
 4. the C4-legacy reconstruction (rate features missing when dt < 1 s) equals a
    pipeline run with the pre-floor denominator rule patched in, for both the
    v3-style ('loo:C4') and the v2-style ('prefloor') pipeline, including the
    interaction with the v2 row filter;
 5. loo:C3 and add:C5 equal a v3 pipeline with the v2 filter appended, and
    loo:C5 / add:C3 equal the corresponding hybrids (labels and masks).
Also prints the byte overhead of the mv file relative to v3.

Usage: python3 experiments/test_variants.py [WORK_DIR]
"""

import filecmp
import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import make_synthetic_raw as msr  # noqa: E402
import opensky_fl_ids_pipeline as pl  # noqa: E402
import variants as V  # noqa: E402

FEATS = ["implied_speed_mps", "speed_consistency_delta", "vrate_consistency_delta",
         "inter_arrival_s", "turn_rate_deg_s", "track_message_count",
         "is_physically_implausible", "squawk_changed", "icao24_valid_format", "is_stale",
         "region_avg_inter_arrival", "is_low_coverage_region", "is_climbing_or_descending"]


def run(paths, out, version, chunk_files=1, seed=42):
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        pl.main(paths, chunk_files=chunk_files, out_path=out, seed=seed, pipeline_version=version)
    return pd.read_csv(out, low_memory=False)


def same_frames(a, b, what, cols=None):
    """Same rows in the same order; exact for non-floats, 1e-9 relative for floats."""
    cols = cols or list(b.columns)
    assert len(a) == len(b), f"{what}: {len(a)} vs {len(b)} rows"
    for c in cols:
        x, y = a[c].reset_index(drop=True), b[c].reset_index(drop=True)
        if pd.api.types.is_float_dtype(y) or pd.api.types.is_float_dtype(x):
            xv, yv = x.to_numpy(dtype="float64"), y.to_numpy(dtype="float64")
            ok = np.isclose(xv, yv, rtol=1e-9, atol=1e-12, equal_nan=True)
            assert ok.all(), f"{what}: column {c}: {int((~ok).sum())} rows differ"
        else:
            xs, ys = x.astype(str).to_numpy(), y.astype(str).to_numpy()
            assert (xs == ys).all(), f"{what}: column {c}: {int((xs != ys).sum())} rows differ"


def view_frame(mv, spec):
    """Variant view of the mv frame as it would be written by a pipeline that
    had exactly that set of legacy behaviours."""
    ls = V.parse_variant(spec)
    v, keep = V.apply_view(mv, ls)
    return v.loc[keep].drop(columns=V.MV_EXTRA_COLS).reset_index(drop=True), keep


def patched_prefloor(fn):
    """Runs fn with the pre-floor denominator rule in add_tier_a_features."""
    orig = pl.add_tier_a_features

    def tier_a(df):
        df = orig(df)
        sub = df["inter_arrival_s"].abs() < 1.0
        for c in V.RATE_FEATURES:
            df[c] = df[c].where(~sub, np.nan)
        return df
    pl.add_tier_a_features = tier_a
    try:
        return fn()
    finally:
        pl.add_tier_a_features = orig


def patched_finalize(fn, filter_on):
    """Runs fn with the v2 benign-only filter forced on/off whatever the version
    (to build the hybrid pipelines 'v3 + filter' and 'v2 - filter')."""
    orig = pl._finalize_and_write

    def fin(piece, out_path, write_header, pipeline_version=pl.DEFAULT_PIPELINE_VERSION):
        if not filter_on:
            return orig(piece, out_path, write_header, pipeline_version="v3")
        return orig(piece, out_path, write_header, pipeline_version="v2")
    pl._finalize_and_write = fin
    try:
        return fn()
    finally:
        pl._finalize_and_write = orig


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)
    raw = msr.write_day(os.path.join(work, "raw"), "2019-01-07", hours=3, n_aircraft=140, seed=5)
    n_checks = 0
    for chunk_files, seed in ((1, 42), (2, 42), (1, 7)):
        tag = f"cf{chunk_files}_s{seed}"
        o = lambda name: os.path.join(work, f"{name}_{tag}.csv")  # noqa: E731
        v2 = run(raw, o("v2"), "v2", chunk_files, seed)
        v3 = run(raw, o("v3"), "v3", chunk_files, seed)
        mv = run(raw, o("mv"), "mv", chunk_files, seed)

        # 1. corrected columns == v3 file, byte for byte
        mv.drop(columns=V.MV_EXTRA_COLS).to_csv(o("mv_as_v3"), index=False)
        v3.to_csv(o("v3_rt"), index=False)
        assert filecmp.cmp(o("mv_as_v3"), o("v3_rt"), shallow=False), "mv corrected columns != v3"
        assert list(mv.columns) == list(v3.columns) + V.MV_EXTRA_COLS
        c, keep_c = view_frame(mv, "corrected")
        assert keep_c.all()
        same_frames(c, v3, "corrected view vs v3")

        # 2./3. legacy view == v2 file
        leg, keep = view_frame(mv, "legacy")
        drop = V._as_bool(mv["legacy_filter_drop"])
        assert (keep == ~drop).all(), "apply_view keep mask != ~legacy_filter_drop"
        assert drop.sum() > 0, "synthetic data has no legacy-filtered rows"
        assert list(leg.columns) == list(v2.columns)
        same_frames(leg, v2, "legacy view vs v2")
        # literal reading of the requirement, without apply_view
        lit = mv.loc[~drop].copy()
        lit["track_message_count"] = lit["track_message_count_legacy"]
        lit["squawk_changed"] = lit["squawk_changed_legacy"]
        lit["attack_type"] = lit["attack_type_legacy"]
        lit["label"] = lit["label_legacy"]
        same_frames(lit.drop(columns=V.MV_EXTRA_COLS), v2, "literal legacy columns vs v2")
        nfl = int(((mv["attack_type_legacy"] == "flooding") & (mv["attack_type"] == "benign")).sum())
        assert nfl > 0

        # 4. C4 (pre-floor rule)
        v3p = patched_prefloor(lambda: run(raw, o("v3p"), "v3", chunk_files, seed))
        v2p = patched_prefloor(lambda: run(raw, o("v2p"), "v2", chunk_files, seed))
        a, _ = view_frame(mv, "loo:C4")
        same_frames(a, v3p, "loo:C4 vs v3 without floor")
        b, keep_p = view_frame(mv, "prefloor")
        same_frames(b, v2p, "prefloor vs v2 without floor")
        n_sub = int(V.subsecond_mask(mv).sum())
        assert n_sub > 0 and (keep_p != keep).any(), "C4 x C3 interaction not exercised"

        # 5. hybrids of the filter and the labels
        v3f = patched_finalize(lambda: run(raw, o("v3f"), "v3", chunk_files, seed), True)
        a, keep_cf = view_frame(mv, "loo:C3")
        same_frames(a, v3f, "loo:C3 vs v3 + filter")
        assert (keep_cf != keep).any(), "C3 x C5 interaction not exercised"
        v2n = patched_finalize(lambda: run(raw, o("v2n"), "v2", chunk_files, seed), False)
        a, _ = view_frame(mv, "add:C3")
        same_frames(a, v2n, "add:C3 vs v2 - filter")
        # add:C5 = v3 labels and filter, v2 count and squawk; loo:C5 = v2 labels on v3
        a, _ = view_frame(mv, "add:C5")
        same_frames(a, v3f, "add:C5 labels/rows vs v3 + filter",
                    cols=[c for c in v3f.columns if c not in ("track_message_count", "squawk_changed")])
        a5, _ = view_frame(mv, "loo:C5")
        same_frames(a5, v2n, "loo:C5 labels vs v2 - filter", cols=["label", "attack_type", "time", "icao24"])
        same_frames(a5, v3, "loo:C5 features vs v3", cols=FEATS)
        a1, _ = view_frame(mv, "loo:C1")
        same_frames(a1, v3, "loo:C1 vs v3", cols=[c for c in v3.columns if c != "track_message_count"])
        same_frames(a1, v2n, "loo:C1 count vs v2 - filter", cols=["track_message_count"])
        a2, _ = view_frame(mv, "loo:C2")
        same_frames(a2, v2n, "loo:C2 squawk vs v2 - filter", cols=["squawk_changed"])
        # design bookkeeping agrees with apply_view
        dl = V.design_labels_and_alive(mv)
        for spec in ("corrected", "legacy", "prefloor", "loo:C3", "loo:C4", "loo:C5", "add:C3",
                     "add:C5", "set:C3+C4"):
            ls = V.parse_variant(spec)
            vv, kk = V.apply_view(mv, ls)
            at, alive = dl[V.design_of(ls)]
            assert (alive == kk).all() and (at == vv["attack_type"].astype(str).to_numpy()).all(), spec
        n_checks += 1
        b3, bm = os.path.getsize(o("v3")), os.path.getsize(o("mv"))
        print(f"[{tag}] v3 {len(v3):,} rows, v2 {len(v2):,} rows, legacy-filtered {int(drop.sum()):,}, "
              f"flooding originals {nfl:,}, sub-second rows {n_sub:,}; "
              f"mv/v3 bytes {bm / b3:.4f} ({(bm - b3) / len(mv):.1f} extra bytes per row)")
    assert V.variant_name(V.parse_variant("add:C4")) == "legacy"
    assert V.variant_name(V.parse_variant("set:C1+C2+C3+C4+C5")) == "prefloor"
    print(f"VARIANTS OK: corrected == v3 (byte-identical), legacy == v2, "
          f"C4 / hybrid reconstructions exact ({n_checks} configurations)")


if __name__ == "__main__":
    main()
