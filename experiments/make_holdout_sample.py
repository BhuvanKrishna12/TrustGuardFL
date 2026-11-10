"""
make_holdout_sample.py
-------------------------------------------------------------------
Builds the held-out test sample: a 1% uniform sample of the given labelled
day files, drawn exactly as before (pandas sample(frac=0.01, random_state=42)
on every 2,000,000-row chunk of each file, files in the order given). The
natural class mix is preserved; nothing is used for training or selection.

Usage:
  python3 experiments/make_holdout_sample.py \
      /mnt/d/fl-ids-sweep/labeled_v3/opensky_2019-12-09_labeled.csv \
      /mnt/d/fl-ids-sweep/labeled_v3/opensky_2019-12-16_labeled.csv \
      /mnt/d/fl-ids-sweep/labeled_v3/opensky_2019-12-23_labeled.csv \
      /mnt/d/fl-ids-sweep/labeled_v3/opensky_2019-12-30_labeled.csv \
      --out /mnt/d/fl-ids-sweep/holdout_sample_v3.csv
"""

import argparse
import os
import time

import pandas as pd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="Labelled day CSVs (test days)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunksize", type=int, default=2_000_000)
    args = ap.parse_args()

    tmp = args.out + ".part"
    first = True
    total_in = total_out = 0
    counts = {}
    for p in args.paths:
        t0 = time.time()
        n_in = n_out = 0
        for ch in pd.read_csv(p, chunksize=args.chunksize, low_memory=False):
            s = ch.sample(frac=args.frac, random_state=args.seed)
            s.to_csv(tmp, mode="w" if first else "a", header=first, index=False)
            first = False
            n_in += len(ch)
            n_out += len(s)
            for k, v in s["attack_type"].astype(str).value_counts().items():
                counts[k] = counts.get(k, 0) + int(v)
        total_in += n_in
        total_out += n_out
        print(f"{os.path.basename(p)}: {n_in:,} rows -> {n_out:,} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    os.replace(tmp, args.out)
    print(f"wrote {args.out}: {total_out:,} rows from {total_in:,}")
    for k in sorted(counts):
        print(f"  {k:<16}{counts[k]:>10,}  {counts[k] / max(1, total_out):.4f}")


if __name__ == "__main__":
    main()
