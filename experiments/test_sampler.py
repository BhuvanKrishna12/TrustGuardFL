"""
test_sampler.py
-------------------------------------------------------------------
Unit test for the training-sample sampler. Four synthetic labelled files (one
UTC day each). For buckets that hit the per-(region, class) cap, a uniform
sampler keeps each file's rows in proportion to that file's share of the
bucket's raw rows (the expected share is computed exactly from the raw files).
The legacy federated_simulation.load_data re-draws the bucket after every
chunk, so its sample is dominated by the last file.

Usage: python3 experiments/test_sampler.py [WORK_DIR]
Exit code 0 when the uniform sampler is near uniform and the legacy one is not.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import federated_simulation as fs  # noqa: E402
import make_synthetic_data as msd  # noqa: E402
import sampling  # noqa: E402

DATES = ["2019-01-07", "2019-04-01", "2019-06-24", "2019-09-30"]
ROWS, BUDGET, CHUNK = 30_000, 50_000, 5_000


def capped_buckets(paths, cap):
    """Buckets with more than `cap` raw rows, and the expected share of each
    file among the sampled rows of those buckets under uniform sampling
    (every capped bucket contributes exactly `cap` rows)."""
    raw = pd.concat([pd.read_csv(p, usecols=["region_cell", "attack_type"]).assign(_f=i)
                     for i, p in enumerate(paths)])
    raw["region_cell"] = raw["region_cell"].astype(str)
    n = raw.groupby(["region_cell", "attack_type", "_f"]).size().unstack(fill_value=0)
    tot = n.sum(axis=1)
    capped = n[tot > cap]
    share = capped.div(capped.sum(axis=1), axis=0).mean(axis=0)
    return set(capped.index), {DATES[i]: float(share.get(i, 0.0)) for i in range(len(paths))}


def shares(df, capped):
    key = list(zip(df["region_cell"].astype(str), df["attack_type"].astype(str)))
    m = np.array([k in capped for k in key])
    day = sampling.day_of(df)[m]
    return {d: float((day == d).mean()) for d in DATES}


def main():
    work = sys.argv[1] if len(sys.argv) > 1 else tempfile.mkdtemp()
    os.makedirs(work, exist_ok=True)
    paths = []
    for i, d in enumerate(DATES):
        p = os.path.join(work, f"day{i + 1}.csv")
        if not os.path.exists(p):
            msd.make(ROWS, [d], 100 + i).to_csv(p, index=False)
        paths.append(p)

    stats, _ = sampling.train_file_stats(paths, chunksize=CHUNK, log=lambda *a: None)
    cap = max(500, BUDGET // (stats["n_regions"] * 6))
    uni, info = sampling.uniform_sample(paths, BUDGET, stats["regions"], sorted(stats["class_counts"]),
                                        seed=42, chunksize=CHUNK, log=lambda *a: None,
                                        compact_every=20_000)
    leg = fs.load_data(paths, max_total_rows=BUDGET, chunksize=CHUNK)
    capped, expected = capped_buckets(paths, cap)
    su, sl = shares(uni, capped), shares(leg, capped)
    print(f"cap {cap} rows per (region, class); {len(capped)} capped buckets")
    print("share of each training day among sampled rows of capped buckets:")
    print(f"{'day':<12}{'expected':>10}{'uniform':>10}{'legacy':>10}")
    for d in DATES:
        print(f"{d:<12}{expected[d]:>10.3f}{su[d]:>10.3f}{sl[d]:>10.3f}")
    comp = sampling.sample_composition(uni, paths, sorted(stats["class_counts"]))
    print("uniform sample rows per source file:",
          {k: v["rows"] for k, v in comp["per_file"].items()})

    ok_uniform = max(abs(su[d] - expected[d]) for d in DATES) < 0.03
    ok_legacy_biased = sl[DATES[-1]] - expected[DATES[-1]] > 0.2
    assert info["raw_rows_streamed"] == stats["total_rows"] == ROWS * len(DATES)
    print(f"uniform within 0.03 of expected: {ok_uniform}; legacy last day over-represented "
          f"by more than 0.2: {ok_legacy_biased}")
    if not (ok_uniform and ok_legacy_biased):
        sys.exit(1)
    print("SAMPLER TEST OK")


if __name__ == "__main__":
    main()
