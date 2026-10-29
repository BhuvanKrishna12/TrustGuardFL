"""
prior_sweep.py
-------------------------------------------------------------------
Trade-off sweep for the class-prior correction on a trained CNN/MLP.

The correction multiplies each predicted class probability by
    (real share of that class in the training days) ** strength
strength 0   = no correction (model as trained, assumes balanced classes)
strength 1   = full correction (assumes attacks are as rare as in the data)
values between = partial correction (0.5 = the "half" setting)

For each strength it reports, on the fixed holdout:
  - false alarms: benign messages flagged as any attack
  - missed attacks: attack messages predicted as benign
  - wrong attack type: attack messages flagged, but as the wrong attack
  - average attack recall (5 attack classes), macro F1, accuracy
  - recall per attack class
so the cost of each setting is visible in both directions: fewer false
alarms vs more missed attacks.

The model is scored once; every strength reuses the same probabilities.

Usage:
  python3 prior_sweep.py --model-path runs/A2_4days_20M.pt \
      --holdout holdout_sample.csv --prior-json runs/L1_4days_10M.json \
      --out-prefix runs/A2_sweep
Outputs: <out-prefix>.csv (one row per strength), <out-prefix>.txt (tables),
         <out-prefix>.png (trade-off chart)
"""

import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

import batch_inference_demo as bid


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--prior-json", required=True,
                    help="lgbm_baseline.py JSON with 'real_prior' from the same training days")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--strengths", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="Comma-separated correction strengths (exponents on the real class mix)")
    ap.add_argument("--batch-size", type=int, default=65536)
    args = ap.parse_args()

    bid.load_model(args.model_path)
    classes = np.array(bid.CLASS_NAMES)
    cls_index = {c: i for i, c in enumerate(classes)}
    benign = cls_index["benign"]
    attack_idx = [i for i in range(len(classes)) if i != benign]

    prior_map = json.load(open(args.prior_json))["real_prior"]
    real_prior = np.array([prior_map[c] for c in classes])

    df = pd.read_csv(args.holdout, usecols=bid.FEATURE_COLS + ["attack_type"], low_memory=False)
    df = df[df["attack_type"].astype(str).isin(cls_index)].reset_index(drop=True)
    df[bid.FEATURE_COLS] = df[bid.FEATURE_COLS].fillna(-1)
    y = df["attack_type"].astype(str).map(cls_index).values
    print(f"Holdout rows: {len(df):,}")

    probs = np.zeros((len(df), len(classes)), dtype=np.float64)
    for lo in range(0, len(df), args.batch_size):
        hi = min(lo + args.batch_size, len(df))
        _, p, valid = bid.predict_batch(df.iloc[lo:hi])
        if not valid.all():
            raise ValueError("NaN features remain after fill; unexpected")
        probs[lo:hi] = p
        print(f"  scored {hi:,}/{len(df):,}", end="\r")
    print()
    del df

    is_benign = y == benign
    is_attack = ~is_benign
    n_benign, n_attack = int(is_benign.sum()), int(is_attack.sum())

    rows = []
    for s in [float(v) for v in args.strengths.split(",")]:
        pred = (probs * (real_prior ** s)[None, :]).argmax(1)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y, pred, labels=np.arange(len(classes)), zero_division=0)
        false_alarms = int((pred[is_benign] != benign).sum())
        missed = int((pred[is_attack] == benign).sum())
        wrong_type = int(((pred != benign) & (pred != y) & is_attack).sum())
        row = {
            "strength": s,
            "accuracy": float((pred == y).mean()),
            "macro_f1": float(f1.mean()),
            "avg_attack_recall": float(rec[attack_idx].mean()),
            "false_alarms": false_alarms,
            "false_alarm_rate": false_alarms / n_benign,
            "missed_attacks": missed,
            "missed_attack_rate": missed / n_attack,
            "wrong_attack_type": wrong_type,
        }
        for i in attack_idx:
            row[f"recall_{classes[i]}"] = float(rec[i])
        for i in attack_idx:
            row[f"precision_{classes[i]}"] = float(prec[i])
        rows.append(row)

    res = pd.DataFrame(rows)
    res.to_csv(args.out_prefix + ".csv", index=False)

    short = {"flooding": "flood", "ghost_aircraft": "ghost", "gps_spoof": "gps",
             "replay": "replay", "velocity_spoof": "veloc"}
    lines = [f"Model: {args.model_path}",
             f"Holdout: {n_benign:,} benign rows, {n_attack:,} attack rows",
             "",
             "Strength 0 = no correction, 0.5 = half, 1 = full.",
             "",
             f"{'strength':>8} {'false alarms':>18} {'missed attacks':>18} {'wrong type':>11} "
             f"{'atk recall':>10} {'macro F1':>9} {'accuracy':>9}"]
    for r in rows:
        lines.append(
            f"{r['strength']:>8.2f} {r['false_alarms']:>10,} ({r['false_alarm_rate']:>5.1%})"
            f" {r['missed_attacks']:>10,} ({r['missed_attack_rate']:>5.1%})"
            f" {r['wrong_attack_type']:>11,} {r['avg_attack_recall']:>10.3f}"
            f" {r['macro_f1']:>9.3f} {r['accuracy']:>9.3f}")
    lines += ["", "Recall per attack class:",
              f"{'strength':>8} " + " ".join(f"{short.get(classes[i], classes[i]):>7}" for i in attack_idx)]
    for r in rows:
        lines.append(f"{r['strength']:>8.2f} " +
                     " ".join(f"{r['recall_' + classes[i]]:>7.3f}" for i in attack_idx))
    lines += ["", "Precision per attack class:",
              f"{'strength':>8} " + " ".join(f"{short.get(classes[i], classes[i]):>7}" for i in attack_idx)]
    for r in rows:
        lines.append(f"{r['strength']:>8.2f} " +
                     " ".join(f"{r['precision_' + classes[i]]:>7.3f}" for i in attack_idx))
    text = "\n".join(lines)
    print("\n" + text)
    open(args.out_prefix + ".txt", "w").write(text + "\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        ax = axes[0]
        ax.plot(res["strength"], res["false_alarm_rate"] * 100, "o-", label="benign falsely flagged (%)")
        ax.plot(res["strength"], res["missed_attack_rate"] * 100, "s-", label="attacks missed (%)")
        ax.set_xlabel("correction strength (0 = none, 1 = full)")
        ax.set_ylabel("%")
        ax.set_title("False alarms vs missed attacks", loc="left", fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        ax = axes[1]
        for i in attack_idx:
            ax.plot(res["strength"], res[f"recall_{classes[i]}"], "o-", ms=3, label=classes[i])
        ax.plot(res["strength"], res["macro_f1"], "k--", lw=2, label="macro F1")
        ax.set_xlabel("correction strength (0 = none, 1 = full)")
        ax.set_title("Recall per attack class, and macro F1", loc="left", fontsize=10)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.out_prefix + ".png", dpi=150)
        print(f"\nChart: {args.out_prefix}.png")
    except Exception as e:
        print(f"(chart skipped: {e})")
    print(f"Table: {args.out_prefix}.csv  Report: {args.out_prefix}.txt")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
