"""
prevalence_sweep.py
-------------------------------------------------------------------
How do the headline metrics change if attacks are rarer (or more common)
than in the holdout?

Some metrics do NOT depend on the benign/attack mix of the test set:
    per-class recall, false alarm rate (benign flagged), average attack recall.
Others DO, because they mix rows of different classes together:
    accuracy, per-class precision, macro F1, "share of alerts that are real".

This script scores the holdout once, then re-weights the per-class results to
any assumed attack prevalence. Within a class, the rows are exactly the
holdout's rows; only the class mix is re-weighted, so nothing is resampled.

Two ways to treat the prior correction (probability x class_share ** strength):
  fixed    the correction uses the TRAINING-days class mix from --prior-json
           (what you report today). The decision rule stays fixed while the
           assumed deployment prevalence changes. This shows how a rule tuned
           to a 20% attack mix behaves when attacks are rarer.
  adapted  the correction uses the scenario's own class mix. This shows what
           you could achieve by setting the correction to the deployment
           prevalence (fewer false alarms, more missed attacks).

Attack share is split across the five attack types equally by default
(--attack-split equal), or in the holdout's own proportions (--attack-split holdout).

Usage:
  python3 prevalence_sweep.py --model-path runs/A2_4days_20M.pt \
      --holdout holdout_sample.csv --prior-json runs/L1_4days_10M.json \
      --out-prefix runs/A2_prevalence

Outputs <out-prefix>.txt (readable), <out-prefix>.csv (all numbers).
-------------------------------------------------------------------
"""

import argparse
import json

import numpy as np
import pandas as pd


def class_rates(y, pred, k):
    """R[a, p] = P(predicted p | actual a). Rows sum to 1."""
    counts = np.bincount(y * k + pred, minlength=k * k).reshape(k, k).astype(np.float64)
    return counts / np.maximum(counts.sum(axis=1, keepdims=True), 1.0)


def scenario_weights(p_att, benign, k, attack_split):
    """Class mix: benign = 1 - p_att, attacks share p_att by attack_split (len k, zero at benign)."""
    w = attack_split * p_att
    w[benign] = 1.0 - p_att
    return w


def mix_metrics(R, w, benign):
    """Metrics for confusion rates R under class mix w."""
    k = R.shape[0]
    atk = [i for i in range(k) if i != benign]
    recall = np.diag(R)
    pred_mass = (w[:, None] * R).sum(axis=0)                      # P(predicted p)
    precision = np.where(pred_mass > 0, w * recall / np.maximum(pred_mass, 1e-300), 0.0)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / np.maximum(precision + recall, 1e-300), 0.0)
    real_alerts = sum(w[a] * (1.0 - R[a, benign]) for a in atk)
    false_alerts = w[benign] * (1.0 - R[benign, benign])
    atk_w = np.array([w[a] for a in atk])
    missed = float(sum(w[a] * R[a, benign] for a in atk) / atk_w.sum()) if atk_w.sum() > 0 else 0.0
    return {
        "accuracy": float((w * recall).sum()),
        "false_alarm_rate": float(1.0 - R[benign, benign]),
        "missed_attack_rate": missed,
        "avg_attack_recall": float(np.mean([recall[a] for a in atk])),
        "alert_precision": float(real_alerts / max(real_alerts + false_alerts, 1e-300)),
        "macro_f1": float(f1.mean()),
        "recall": recall,
        "precision": precision,
    }


def run_sweep(probs, y, classes, benign, train_prior, strengths, prevalences, attack_split, modes):
    """Return list of result dicts, one per (mode, strength, scenario)."""
    k = len(classes)
    rows = []
    holdout_share = np.bincount(y, minlength=k) / len(y)
    scenarios = [("holdout as-is", None)] + [(f"{p:g}", p) for p in prevalences]
    fixed_rates = {}
    for s in strengths:
        if "fixed" in modes:
            pred = (probs * (train_prior ** s)[None, :]).argmax(1)
            fixed_rates[s] = class_rates(y, pred, k)
    for name, p_att in scenarios:
        if p_att is None:
            w = holdout_share.copy()
        else:
            w = scenario_weights(p_att, benign, k, attack_split)
        for s in strengths:
            for mode in modes:
                if mode == "fixed":
                    R = fixed_rates[s]
                else:
                    pred = (probs * (np.maximum(w, 1e-12) ** s)[None, :]).argmax(1)
                    R = class_rates(y, pred, k)
                m = mix_metrics(R, w, benign)
                m.update({"mode": mode, "strength": s, "scenario": name,
                          "attack_share": float(1 - w[benign])})
                rows.append(m)
    return rows


def format_report(rows, classes, strengths, modes, header_lines):
    out = list(header_lines)
    atk_names = [c for c in classes if c != "benign"]
    for mode in modes:
        out += ["", "=" * 96,
                f"MODE: {mode}  ("
                + ("correction uses the TRAINING-days class mix" if mode == "fixed"
                   else "correction uses each scenario's own class mix") + ")",
                "=" * 96]
        for s in strengths:
            out += ["", f"strength {s:g}",
                    f"{'attack share':<24} {'accuracy':>8} {'false alarm':>11} {'missed atk':>10} "
                    f"{'atk recall':>10} {'alert prec':>10} {'macro F1':>8}"]
            for r in [r for r in rows if r["mode"] == mode and r["strength"] == s]:
                out.append(f"{r['scenario'] + ' (' + format(r['attack_share'], '.4g') + ')':<24} "
                           f"{r['accuracy']:>8.3f} {r['false_alarm_rate']:>11.2%} {r['missed_attack_rate']:>10.2%} "
                           f"{r['avg_attack_recall']:>10.3f} {r['alert_precision']:>10.2%} {r['macro_f1']:>8.3f}")
    out += ["", "-" * 96,
            "Columns that do not move with the attack share: false alarm, atk recall.",
            "(In 'adapted' mode they move, because the decision rule itself is re-set per scenario.)",
            "alert prec = share of raised alerts that are real attacks.", ""]
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--holdout", required=True)
    ap.add_argument("--prior-json", required=True)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--strengths", default="0,0.5,1.0",
                    help="Correction strengths to report (0 none, 0.5 half, 1 full)")
    ap.add_argument("--prevalences", default="0.1,0.02,0.005,0.0005",
                    help="Assumed TOTAL attack share of traffic, comma-separated")
    ap.add_argument("--attack-split", default="equal", choices=["equal", "holdout"],
                    help="How the attack share is divided among attack types")
    ap.add_argument("--modes", default="fixed,adapted")
    ap.add_argument("--batch-size", type=int, default=65536)
    args = ap.parse_args()

    import torch
    import batch_inference_demo as bid   # the sweep version (same one per_class_correction.py uses)

    bid.load_model(args.model_path)
    classes = np.array(bid.CLASS_NAMES)
    cls_index = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    benign = cls_index["benign"]
    strengths = [float(s) for s in args.strengths.split(",")]
    prevalences = [float(p) for p in args.prevalences.split(",")]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    prior_map = json.load(open(args.prior_json))["real_prior"]
    train_prior = np.array([prior_map[c] for c in classes])

    cols = bid.FEATURE_COLS + ["attack_type"]
    df = pd.read_csv(args.holdout, usecols=cols, low_memory=False)
    df = df[df["attack_type"].astype(str).isin(cls_index)].reset_index(drop=True)
    df[bid.FEATURE_COLS] = df[bid.FEATURE_COLS].fillna(-1)
    y = df["attack_type"].astype(str).map(cls_index).values

    probs = np.zeros((len(df), k), dtype=np.float64)
    for lo in range(0, len(df), args.batch_size):
        hi = min(lo + args.batch_size, len(df))
        _, p, _ = bid.predict_batch(df.iloc[lo:hi])
        probs[lo:hi] = p
        print(f"  scored {hi:,}/{len(df):,}", end="\r")
    print()
    del df

    holdout_share = np.bincount(y, minlength=k) / len(y)
    if args.attack_split == "equal":
        split = np.array([0.0 if i == benign else 1.0 for i in range(k)])
    else:
        split = holdout_share.copy()
        split[benign] = 0.0
    split = split / split.sum()

    rows = run_sweep(probs, y, classes, benign, train_prior, strengths, prevalences, split, modes)

    header = [
        f"Model: {args.model_path}",
        f"Holdout rows: {len(y):,}; class mix: "
        + ", ".join(f"{c} {holdout_share[i]:.1%}" for i, c in enumerate(classes)),
        f"Training-days mix used by the 'fixed' correction: "
        + ", ".join(f"{c} {train_prior[i]:.1%}" for i, c in enumerate(classes)),
        f"Assumed attack shares: {', '.join(f'{p:g}' for p in prevalences)}; "
        f"attack share split: {args.attack_split}",
    ]
    # recall per class does not depend on the mix: report it once per strength (fixed rule)
    k_rates = {s: class_rates(y, (probs * (train_prior ** s)[None, :]).argmax(1), k) for s in strengths}
    header += ["", "Per-class recall (mix-independent, 'fixed' rule):",
               f"{'strength':<9}" + "".join(f"{c:>16}" for c in classes)]
    for s in strengths:
        header.append(f"{s:<9g}" + "".join(f"{v:>16.3f}" for v in np.diag(k_rates[s])))

    text = format_report(rows, classes, strengths, modes, header)
    print("\n" + text)
    open(args.out_prefix + ".txt", "w").write(text + "\n")

    flat = []
    for r in rows:
        d = {kk: v for kk, v in r.items() if kk not in ("recall", "precision")}
        for i, c in enumerate(classes):
            d[f"recall_{c}"] = float(r["recall"][i])
            d[f"precision_{c}"] = float(r["precision"][i])
        flat.append(d)
    pd.DataFrame(flat).to_csv(args.out_prefix + ".csv", index=False)
    print(f"Report: {args.out_prefix}.txt   Numbers: {args.out_prefix}.csv")


if __name__ == "__main__":
    main()
