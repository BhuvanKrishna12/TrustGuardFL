"""
make_class_sample_csv.py
-------------------------------------------------------------------
Builds a small, balanced sample CSV from a labeled holdout file
(one produced by opensky_fl_ids_pipeline.py), by taking N rows of
EACH distinct value found in the `attack_type` column -- so the
sample naturally includes `benign` plus whatever attack types are
actually present (gps_spoof, velocity_spoof, replay, ghost_aircraft,
flooding, or any future category), without hardcoding that list.

This is a quick, easy-to-eyeball test input for batch_inference_demo.py:
it preserves ALL original columns unmodified, so it works as a
drop-in `--input-csv` regardless of which FEATURE_COLS a given model
checkpoint needs.

Rows are grouped by class in the output (simple concat, no shuffle)
so they're easy to scan by eye. If you'd rather have the rows in
random order, shuffle the final DataFrame before writing it out,
e.g.: sample_df = sample_df.sample(frac=1, random_state=args.seed)

Usage:
    python3 make_class_sample_csv.py --input-csv holdout_jan14_6h.csv --n-per-class 10
-------------------------------------------------------------------
"""
import argparse
import os

import pandas as pd


def main():
    p = argparse.ArgumentParser(
        description="Build a small sample CSV with N rows of each distinct "
                    "attack_type value, preserving all columns."
    )
    p.add_argument("--input-csv", required=True,
                   help="Labeled holdout CSV to sample from (must have an "
                        "'attack_type' column).")
    p.add_argument("--output-csv", default=None,
                   help="Where to write the sample. Default: "
                        "'<input-csv stem>_sample.csv' next to the input file.")
    p.add_argument("--n-per-class", type=int, default=10,
                   help="Number of rows to sample per distinct attack_type "
                        "value (default 10).")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for reproducible sampling (default 42, "
                        "matching this project's convention elsewhere).")
    args = p.parse_args()

    if args.output_csv:
        output_csv = args.output_csv
    else:
        base, _ext = os.path.splitext(args.input_csv)
        output_csv = f"{base}_sample.csv"

    print(f"Reading input CSV: {args.input_csv} ...")
    df = pd.read_csv(args.input_csv)
    n_total = len(df)
    print(f"Loaded {n_total:,} rows.")

    if "attack_type" not in df.columns:
        raise ValueError(
            f"Input CSV '{args.input_csv}' has no 'attack_type' column. "
            f"This script groups strictly by 'attack_type' and will not "
            f"fall back to 'label' or guess another column -- re-run "
            f"against a file produced by opensky_fl_ids_pipeline.py that "
            f"retains 'attack_type'."
        )

    classes = sorted(df["attack_type"].unique().tolist())
    print(f"Found {len(classes)} distinct attack_type value(s): {classes}")

    sampled_groups = []
    summary_rows = []
    for cls in classes:
        group = df[df["attack_type"] == cls]
        group_size = len(group)
        n_take = min(args.n_per_class, group_size)
        if n_take < args.n_per_class:
            print(f"  WARNING: class '{cls}' has only {group_size:,} row(s) -- "
                  f"requested {args.n_per_class:,}, got {n_take:,}.")
        sampled = group.sample(n=n_take, random_state=args.seed)
        sampled_groups.append(sampled)
        summary_rows.append((cls, args.n_per_class, n_take))

    # Grouped-by-class concat, in the order classes were found above.
    # (See module docstring for how to shuffle instead.)
    sample_df = pd.concat(sampled_groups, axis=0)
    sample_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 60)
    print("SAMPLE SUMMARY")
    print("=" * 60)
    print(f"{'attack_type':<20}{'requested':>12}{'obtained':>12}")
    print("-" * 44)
    for cls, requested, obtained in summary_rows:
        print(f"{cls:<20}{requested:>12,}{obtained:>12,}")
    print("-" * 44)
    print(f"{'TOTAL':<20}{args.n_per_class * len(classes):>12,}{len(sample_df):>12,}")
    print(f"\nOutput written to: {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
