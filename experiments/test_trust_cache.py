#!/usr/bin/env python3
"""
test_trust_cache.py
-------------------------------------------------------------------
Cache-only ("trust cache") equivalence test. On small synthetic multi-variant
data:
  1. build the cache WITH the labelled training files present (a --quick run of
     every remaining model x sampler x design into out-dir A), and write the
     input manifest;
  2. DELETE the labelled training files;
  3. re-run the same grid in cache-only mode (files absent) into out-dir B,
     sharing the same cache;
  4. assert every run's metrics, s*, selection and holdout match A exactly.

Also checks that a missing cache artifact fails with a clear message, not a
FileNotFoundError on the absent labelled file.

  python3 experiments/test_trust_cache.py WORK_DIR
"""
import contextlib
import io
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import make_synthetic_raw as msr  # noqa: E402
import opensky_fl_ids_pipeline as pl  # noqa: E402
import run_experiments as R  # noqa: E402

# the remaining grid, reduced to one or two seeds per (model, sampler, variant)
# but covering every design and both samplers
GRID = [
    dict(variant="corrected", models="lgbm,central_cnn,fed_cnn,central_mlp,fed_mlp"),
    dict(variant="legacy", models="lgbm,central_cnn"),       # design LF
    dict(variant="loo:C3", models="lgbm"),                   # design CF
    dict(variant="loo:C5", models="lgbm"),                   # design LN
    dict(variant="add:C3", models="lgbm"),                   # design LN
    dict(variant="prefloor", models="lgbm"),                 # design LP
    dict(variant="corrected", models="lgbm", legacy_sampler=True),   # legacy_CN
    dict(variant="legacy", models="lgbm", legacy_sampler=True),      # legacy_LF
    dict(variant="loo:C3", models="fed_cnn"),                # federated ablation path
]


def build_inputs(work):
    train, holdout = [], None
    for date in ("2019-01-07", "2019-04-01", "2019-12-09"):
        raw = msr.write_day(os.path.join(work, "raw"), date, hours=3, n_aircraft=150, seed=5)
        out = os.path.join(work, f"opensky_{date}_labeled.csv")
        with contextlib.redirect_stdout(io.StringIO()):
            pl.main(raw, chunk_files=1, out_path=out, pipeline_version="mv")
        if date == "2019-12-09":
            holdout = out
        else:
            train.append(out)
    return train, holdout


def spec_dir(spec):
    return spec["variant"].replace(":", "_") + ("_ls" if spec.get("legacy_sampler") else "")


def argv_for(spec, train, holdout, out, cache, trust):
    out = os.path.join(out, spec_dir(spec))
    av = ["--train-files", *train, "--holdout", holdout, "--out-dir", out,
          "--cache-dir", cache, "--variant", spec["variant"], "--protocol", "v2",
          "--models", spec["models"], "--feature-sets", "full", "--seeds", "1",
          "--device", "cpu", "--quick", "--max-total-rows", "60000",
          "--min-client-rows", "800"]
    if spec.get("legacy_sampler"):
        av.append("--legacy-sampler")
    if trust:
        av.append("--trust-cache")
    return av


def runs_metrics(out):
    got = {}
    for spec in GRID:
        rdir = os.path.join(out, spec_dir(spec), "runs")
        if not os.path.isdir(rdir):
            continue
        for f in sorted(os.listdir(rdir)):
            if f.endswith(".json") and "__" in f and f.startswith(("lgbm", "central", "fed")):
                got[spec_dir(spec) + "/" + f] = json.load(open(os.path.join(rdir, f)))
    return got


def comparable(r):
    """The parts of a run that must be identical with or without the files."""
    return {
        "s_star": r["s_star"],
        "classes": r["classes"],
        "train_sample_rows": r["data"]["train_sample_rows"],
        "holdout_rows": r["holdout"]["rows"],
        "metrics_1": r["metrics"]["1.00"]["pooled"],
        "metrics_s0": r["metrics"]["0.00"]["pooled"],
        "metrics_sstar": r["metrics"][f"{r['s_star']:.2f}"]["pooled"],
        "pr_auc": r.get("pr_auc"),
        "val_macro_f1": r["selection"]["val_macro_f1_at_s_star"],
    }


def assert_equal(a, b, what):
    ja = json.dumps(a, sort_keys=True, default=float)
    jb = json.dumps(b, sort_keys=True, default=float)
    if ja == jb:
        return
    # allow a tiny float drift
    def close(x, y):
        if isinstance(x, dict):
            return all(close(x[k], y[k]) for k in x)
        if isinstance(x, list):
            return len(x) == len(y) and all(close(u, v) for u, v in zip(x, y))
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            return np.isclose(x, y, rtol=1e-9, atol=1e-9)
        return x == y
    assert close(a, b), f"{what} differs:\n  with files: {ja}\n  cache-only: {jb}"


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)
    cache = os.path.join(work, "cache")
    outA, outB = os.path.join(work, "A"), os.path.join(work, "B")
    train, holdout = build_inputs(work)

    # 1. with files present
    for spec in GRID:
        R.LOG_FH = None
        with contextlib.redirect_stdout(io.StringIO()):
            rc = R.main(argv_for(spec, train, holdout, outA, cache, trust=False))
        assert rc == 0, f"with-files run failed: {spec}"
    mA = runs_metrics(outA)

    # manifest, then delete the labelled training files
    from make_input_manifest import main as mani
    with contextlib.redirect_stdout(io.StringIO()):
        mani(["--train-files", *train, "--holdout", holdout, "--cache-dir", cache])
    for p in train:
        os.remove(p)
    assert all(not os.path.exists(p) for p in train)

    # 2. cache-only
    for spec in GRID:
        R.LOG_FH = None
        with contextlib.redirect_stdout(io.StringIO()):
            rc = R.main(argv_for(spec, train, holdout, outB, cache, trust=True))
        assert rc == 0, f"cache-only run failed: {spec}"
    mB = runs_metrics(outB)

    # 3. compare
    assert set(mA) == set(mB), f"different run sets:\n  A {sorted(mA)}\n  B {sorted(mB)}"
    assert mA, "no runs produced"
    for k in sorted(mA):
        assert_equal(comparable(mA[k]), comparable(mB[k]), k)

    # 4. clear error when a cache artifact is missing
    import glob
    missing_cache = os.path.join(work, "cache_missing")
    os.makedirs(missing_cache)
    # copy only the manifest, not the sample/stats caches
    import shutil
    shutil.copy(os.path.join(cache, "input_manifest.json"),
                os.path.join(missing_cache, "input_manifest.json"))
    R.LOG_FH = None
    err = ""
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            R.main(argv_for(GRID[0], train, holdout, os.path.join(work, "C"),
                            missing_cache, trust=True))
        raised = False
    except (SystemExit, FileNotFoundError) as e:
        raised, err = True, str(e)
    assert raised and ("cache" in err.lower()), f"expected a clear cache error, got: {err!r}"

    print(f"TRUST-CACHE TEST OK: {len(mA)} runs identical with and without the labelled "
          f"files (files deleted); missing-cache run failed with a clear message")


if __name__ == "__main__":
    main()
