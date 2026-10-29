"""
prior_correct_cnn.py
-------------------------------------------------------------------
Applies the class-prior correction found with lgbm_baseline.py to an
already-trained federated CNN/MLP checkpoint. No retraining.

Why: training caps every class to similar counts and uses "balanced"
class weights, so the model's probabilities assume attacks are as common
as normal traffic. Multiplying each predicted probability by the real
class mix of the TRAINING days (taken from the LightGBM report's
"real_prior", never from the holdout) moves the decision back toward the
real-world base rate, which cuts false alarms on benign traffic.

Reports three versions on the fixed holdout, in the same format as
lgbm_baseline.py: as trained, full correction, half (sqrt) correction.

Usage:
  python3 prior_correct_cnn.py --model-path runs/A1_4days_10M.pt \
      --holdout holdout_sample.csv \
      --prior-json runs/L1_4days_10M.json \
      --out-prefix runs/A1_4days_10M_prior
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch

import batch_inference_demo as bid
from lgbm_baseline import evaluate


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True, help="Federated CNN/MLP .pt checkpoint")
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--prior-json", required=True,
                    help="lgbm_baseline.py JSON report containing 'real_prior' "
                         "(must come from the same TRAINING days as the model)")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--batch-size", type=int, default=65536)
    args = ap.parse_args()

    fh = open(args.out_prefix + ".txt", "w")
    bid.load_model(args.model_path)
    classes = np.array(bid.CLASS_NAMES)
    cls_index = {c: i for i, c in enumerate(classes)}

    prior_map = json.load(open(args.prior_json))["real_prior"]
    missing = [c for c in classes if c not in prior_map]
    if missing:
        raise ValueError(f"Prior JSON has no entry for classes {missing}")
    real_prior = np.array([prior_map[c] for c in classes])
    bid_log = ", ".join(f"{c}={p:.3f}" for c, p in zip(classes, real_prior))
    print(f"Real class mix used for correction: {bid_log}")
    fh.write(f"Model: {args.model_path}\nReal class mix used for correction: {bid_log}\n")

    df = pd.read_csv(args.holdout, usecols=bid.FEATURE_COLS + ["attack_type"], low_memory=False)
    df = df[df["attack_type"].astype(str).isin(cls_index)].reset_index(drop=True)
    df[bid.FEATURE_COLS] = df[bid.FEATURE_COLS].fillna(-1)
    y = df["attack_type"].astype(str).map(cls_index).values
    print(f"Holdout rows: {len(df):,}")

    probs = np.zeros((len(df), len(classes)), dtype=np.float32)
    for lo in range(0, len(df), args.batch_size):
        hi = min(lo + args.batch_size, len(df))
        _, p, valid = bid.predict_batch(df.iloc[lo:hi])
        if not valid.all():
            raise ValueError("NaN features remain after fill; unexpected")
        probs[lo:hi] = p
        print(f"  scored {hi:,}/{len(df):,}", end="\r")
    print()

    results = {
        "model_path": args.model_path,
        "real_prior": dict(zip(classes.tolist(), real_prior.tolist())),
        "holdout_plain": evaluate("HOLDOUT: model as trained", y, probs.argmax(1), classes, fh),
        "holdout_prior_corrected": evaluate(
            "HOLDOUT: prior-corrected to real class mix",
            y, (probs * real_prior[None, :]).argmax(1), classes, fh),
        "holdout_half_corrected": evaluate(
            "HOLDOUT: half prior correction (sqrt of real mix)",
            y, (probs * np.sqrt(real_prior)[None, :]).argmax(1), classes, fh),
    }
    with open(args.out_prefix + ".json", "w") as jf:
        json.dump(results, jf, indent=2)
    print(f"\nReport: {args.out_prefix}.txt  Metrics: {args.out_prefix}.json")
    fh.close()


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
