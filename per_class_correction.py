"""
per_class_correction.py
-------------------------------------------------------------------
Per-class version of the class-prior correction, tuned honestly.

The uniform correction multiplies every class probability by
(real class share) ** strength, with one strength for all classes.
velocity_spoof pays most of its cost, so this script lets selected
classes have their OWN strength while every other class keeps the base
strength (default: full correction).

Honest tuning: the holdout days are split in two.
  * TUNE days (default: the first half of the holdout dates) are used
    only to choose the per-class strengths.
  * EVAL days (the remaining dates) are used only to report results.
The headline numbers therefore never come from data the setting was
chosen on.

Selection rule (pick one):
  --objective macro_f1             highest macro F1 on the tune days (default)
  --max-false-alarm-rate 0.02      highest average attack recall whose false
                                   alarm rate on the tune days stays at or
                                   below this limit

Usage:
  python3 per_class_correction.py --model-path runs/A2_4days_20M.pt \
      --holdout holdout_sample.csv --prior-json runs/L1_4days_10M.json \
      --classes velocity_spoof,replay --out-prefix runs/A2_perclass
"""

import argparse
import itertools
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

import batch_inference_demo as bid


def metrics(y, pred, classes, benign):
    labels = np.arange(len(classes))
    prec, rec, f1, _ = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    atk = [i for i in labels if i != benign]
    is_b = y == benign
    fa = int((pred[is_b] != benign).sum())
    missed = int((pred[~is_b] == benign).sum())
    flagged_attacks = int((~is_b).sum()) - missed
    alerts = flagged_attacks + fa
    return {
        "accuracy": float((pred == y).mean()),
        "macro_f1": float(f1.mean()),
        "avg_attack_recall": float(rec[atk].mean()),
        "false_alarms": fa,
        "false_alarm_rate": fa / max(1, int(is_b.sum())),
        "missed_attacks": missed,
        "missed_attack_rate": missed / max(1, int((~is_b).sum())),
        "alerts_real_share": flagged_attacks / max(1, alerts),
        "recall": {classes[i]: float(rec[i]) for i in labels},
        "precision": {classes[i]: float(prec[i]) for i in labels},
        "f1": {classes[i]: float(f1[i]) for i in labels},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--prior-json", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--classes", default="velocity_spoof",
                    help="Comma-separated classes that get their own strength (max 3)")
    ap.add_argument("--base-strength", type=float, default=1.0,
                    help="Strength for every other class (1 = full correction)")
    ap.add_argument("--grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--tune-dates", default=None,
                    help="Comma-separated YYYY-MM-DD dates for tuning (default: first half of holdout dates)")
    ap.add_argument("--objective", default="macro_f1", choices=["macro_f1"])
    ap.add_argument("--max-false-alarm-rate", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=65536)
    args = ap.parse_args()

    bid.load_model(args.model_path)
    classes = np.array(bid.CLASS_NAMES)
    cls_index = {c: i for i, c in enumerate(classes)}
    benign = cls_index["benign"]
    special = [c.strip() for c in args.classes.split(",") if c.strip()]
    for c in special:
        if c not in cls_index:
            raise ValueError(f"Unknown class '{c}'. Classes: {list(classes)}")
    if len(special) > 3:
        raise ValueError("At most 3 classes (the grid grows quickly)")
    grid = [float(g) for g in args.grid.split(",")]

    prior_map = json.load(open(args.prior_json))["real_prior"]
    prior = np.array([prior_map[c] for c in classes])

    # ---- holdout with a day label ----
    # Day comes from lastposupdate where available: the replay attack shifts
    # 'time' but not 'lastposupdate', so this keeps replayed rows on their day.
    cols = bid.FEATURE_COLS + ["attack_type", "time", "lastposupdate"]
    header = pd.read_csv(args.holdout, nrows=0).columns
    df = pd.read_csv(args.holdout, usecols=[c for c in cols if c in header], low_memory=False)
    df = df[df["attack_type"].astype(str).isin(cls_index)].reset_index(drop=True)
    ts = df["lastposupdate"] if "lastposupdate" in df else df["time"]
    ts = pd.to_numeric(ts, errors="coerce").fillna(pd.to_numeric(df["time"], errors="coerce"))
    day = pd.to_datetime(ts, unit="s", utc=True).dt.strftime("%Y-%m-%d").values
    df[bid.FEATURE_COLS] = df[bid.FEATURE_COLS].fillna(-1)
    y = df["attack_type"].astype(str).map(cls_index).values

    day_counts = pd.Series(day).value_counts().sort_index()
    main_days = [d for d, n in day_counts.items() if n >= 0.05 * len(df)]
    if args.tune_dates:
        tune_days = [d.strip() for d in args.tune_dates.split(",")]
    else:
        tune_days = main_days[: len(main_days) // 2]
    eval_days = [d for d in main_days if d not in tune_days]
    is_tune = np.isin(day, tune_days)
    is_eval = np.isin(day, eval_days)
    if is_tune.sum() == 0 or is_eval.sum() == 0:
        raise ValueError(f"Could not split holdout by day. Days found: {day_counts.to_dict()}")

    probs = np.zeros((len(df), len(classes)), dtype=np.float64)
    for lo in range(0, len(df), args.batch_size):
        hi = min(lo + args.batch_size, len(df))
        _, p, valid = bid.predict_batch(df.iloc[lo:hi])
        probs[lo:hi] = p
        print(f"  scored {hi:,}/{len(df):,}", end="\r")
    print()
    del df

    def weights(strengths):
        s = np.full(len(classes), args.base_strength)
        for c, v in strengths.items():
            s[cls_index[c]] = v
        return prior ** s

    def run(mask, strengths):
        pred = (probs[mask] * weights(strengths)[None, :]).argmax(1)
        return metrics(y[mask], pred, classes, benign)

    out = []
    out.append(f"Model: {args.model_path}")
    out.append(f"Days in holdout: " + ", ".join(f"{d} ({n:,})" for d, n in day_counts.items()))
    out.append(f"TUNE days: {', '.join(tune_days)} ({int(is_tune.sum()):,} rows)")
    out.append(f"EVAL days: {', '.join(eval_days)} ({int(is_eval.sum()):,} rows)")
    out.append(f"Classes with their own strength: {', '.join(special)}; "
               f"all others at strength {args.base_strength}")
    rule = (f"highest attack recall with false alarm rate <= {args.max_false_alarm_rate:.1%}"
            if args.max_false_alarm_rate is not None else "highest macro F1")
    out.append(f"Selection rule (applied to TUNE days only): {rule}")

    # ---- grid search on tune days ----
    results = []
    for combo in itertools.product(grid, repeat=len(special)):
        st = dict(zip(special, combo))
        m = run(is_tune, st)
        results.append((st, m))

    if args.max_false_alarm_rate is not None:
        ok = [r for r in results if r[1]["false_alarm_rate"] <= args.max_false_alarm_rate]
        if not ok:
            raise ValueError("No setting meets the false-alarm limit on the tune days; raise the limit")
        best_st, best_m = max(ok, key=lambda r: (r[1]["avg_attack_recall"], r[1]["macro_f1"]))
    else:
        best_st, best_m = max(results, key=lambda r: r[1]["macro_f1"])

    hdr = " ".join(f"{c[:12]:>12}" for c in special)
    out += ["", "TUNE days, best 10 settings by the selection rule:",
            f"{hdr} {'macro F1':>9} {'atk recall':>10} {'false alarm':>11} {'missed':>8}"
            + "".join(f" {('rec ' + c[:8]):>13}" for c in special)]
    key = ((lambda r: (r[1]["false_alarm_rate"] <= args.max_false_alarm_rate,
                       r[1]["avg_attack_recall"]))
           if args.max_false_alarm_rate is not None else (lambda r: r[1]["macro_f1"]))
    for st, m in sorted(results, key=key, reverse=True)[:10]:
        out.append(" ".join(f"{st[c]:>12.1f}" for c in special)
                   + f" {m['macro_f1']:>9.3f} {m['avg_attack_recall']:>10.3f}"
                   f" {m['false_alarm_rate']:>11.1%} {m['missed_attack_rate']:>8.1%}"
                   + "".join(f" {m['recall'][c]:>13.3f}" for c in special))
    out.append("Chosen: " + ", ".join(f"{c} strength {v:.1f}" for c, v in best_st.items()))

    # ---- report on eval days ----
    uniform_full = {c: args.base_strength for c in special}
    configs = [
        ("no correction", {c: 0.0 for c in classes}),
        ("uniform half", {c: 0.5 for c in classes}),
        ("uniform full", {c: args.base_strength for c in classes}),
        ("per-class (chosen)", best_st),
    ]
    evals = {}
    for name, st in configs:
        if name in ("no correction", "uniform half"):
            s = np.array([st[c] for c in classes])
            pred = (probs[is_eval] * (prior ** s)[None, :]).argmax(1)
            evals[name] = metrics(y[is_eval], pred, classes, benign)
        else:
            evals[name] = run(is_eval, st)

    out += ["", "=" * 78, "EVAL days only (not used for choosing):", "=" * 78,
            f"{'setting':<20} {'accuracy':>8} {'macro F1':>9} {'atk recall':>10} "
            f"{'false alarms':>18} {'missed':>8} {'alerts real':>11}"]
    for name, m in evals.items():
        out.append(f"{name:<20} {m['accuracy']:>8.3f} {m['macro_f1']:>9.3f} {m['avg_attack_recall']:>10.3f}"
                   f" {m['false_alarms']:>9,} ({m['false_alarm_rate']:>5.1%})"
                   f" {m['missed_attack_rate']:>8.1%} {m['alerts_real_share']:>11.1%}")
    out += ["", "Per class on EVAL days (recall / precision):",
            f"{'class':<16}" + "".join(f"{n:>22}" for n in evals)]
    for c in classes:
        out.append(f"{c:<16}" + "".join(
            f"{m['recall'][c]:>13.3f} / {m['precision'][c]:.3f}" for m in evals.values()))

    text = "\n".join(out)
    print("\n" + text)
    open(args.out_prefix + ".txt", "w").write(text + "\n")
    json.dump({"tune_days": tune_days, "eval_days": eval_days, "chosen": best_st,
               "base_strength": args.base_strength, "rule": rule,
               "tune_metrics_chosen": best_m, "eval": evals},
              open(args.out_prefix + ".json", "w"), indent=2)
    print(f"\nReport: {args.out_prefix}.txt  Metrics: {args.out_prefix}.json")


if __name__ == "__main__":
    torch.set_num_threads(4)
    main()
