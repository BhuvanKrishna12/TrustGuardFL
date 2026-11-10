"""
lgbm_baseline.py
-------------------------------------------------------------------
Centralized LightGBM baseline for FL-IDS.

Purpose: a diagnostic, not a replacement for the federated CNN.
It answers two questions on the SAME data the federated runs use:

  1. Is the model family the bottleneck?  Gradient-boosted trees are
     usually the strongest choice for tabular features like these. If
     LightGBM (trained centrally, with no federation handicap) scores far
     better on the fixed holdout than the federated CNN, a better model
     family is worth pursuing. If it scores about the same, the limit is
     in the data/features/class balance, not the CNN.

  2. Is class balancing causing the false alarms?  Training samples are
     capped equally per (region, class) and weighted "balanced", so the
     model learns as if attacks were as common as normal traffic. This
     script also reports a PRIOR-CORRECTED version: the same trained model,
     with predicted probabilities re-weighted to the real class mix of the
     TRAINING days (never the holdout). If false alarms drop sharply while
     attack recall holds, that confirms the cause and the same correction
     can be applied to the CNN.

Training rows are sampled with federated_simulation.load_data(), the exact
same function and seed the federated runs use, so with the same files and
--max-total-rows the training set is identical to the matching CNN run.

Usage:
  python3 lgbm_baseline.py TRAIN_CSV [TRAIN_CSV ...] \
      --holdout holdout_sample.csv --max-total-rows 10000000 \
      --out-prefix runs/L1_4days_10M

Outputs: <out-prefix>.txt (full report), <out-prefix>.json (metrics),
         <out-prefix>_sample.pkl (cached training sample, reused on rerun).
"""

import argparse
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from federated_simulation import FEATURE_COLS, RNG_SEED, load_data, prep_features


def log(msg, fh):
    print(msg, flush=True)
    fh.write(msg + "\n")
    fh.flush()


def real_class_prior(paths, classes, chunksize=2_000_000):
    """Class proportions across ALL raw rows of the training files.
    Reads only the attack_type column, so it is cheap relative to a full load."""
    counts = pd.Series(0, index=classes, dtype="int64")
    for p in paths:
        for ch in pd.read_csv(p, usecols=["attack_type"], chunksize=chunksize):
            vc = ch["attack_type"].astype(str).value_counts()
            counts = counts.add(vc.reindex(classes, fill_value=0), fill_value=0)
    return (counts / counts.sum()).values


def evaluate(name, y_true, y_pred, classes, fh):
    labels = np.arange(len(classes))
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    acc = float((y_true == y_pred).mean())
    benign_idx = list(classes).index("benign") if "benign" in classes else None

    log(f"\n=== {name} ===", fh)
    log(f"Overall accuracy: {acc:.4f}", fh)
    log(f"{'Class':<18}{'Precision':>11}{'Recall':>10}{'F1':>8}{'Support':>12}", fh)
    for i, c in enumerate(classes):
        log(f"{c:<18}{prec[i]:>11.3f}{rec[i]:>10.3f}{f1[i]:>8.3f}{sup[i]:>12,}", fh)
    log(f"Average recall (6 classes): {rec.mean():.4f}", fh)
    log(f"Macro F1:                   {f1.mean():.4f}", fh)

    out = {
        "accuracy": acc,
        "avg_recall": float(rec.mean()),
        "macro_f1": float(f1.mean()),
        "per_class": {c: {"precision": float(prec[i]), "recall": float(rec[i]),
                          "f1": float(f1[i]), "support": int(sup[i])}
                      for i, c in enumerate(classes)},
    }
    if benign_idx is not None:
        mask = y_true == benign_idx
        false_alarms = int((y_pred[mask] != benign_idx).sum())
        out["benign_false_alarms"] = false_alarms
        out["benign_false_alarm_rate"] = false_alarms / max(1, int(mask.sum()))
        log(f"Benign rows flagged as attack: {false_alarms:,} "
            f"({out['benign_false_alarm_rate']:.1%})", fh)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    log("\nConfusion matrix (rows = actual, columns = predicted):", fh)
    log(" " * 14 + "".join(f"{c[:10]:>12}" for c in classes), fh)
    for i, c in enumerate(classes):
        log(f"{c[:13]:<14}" + "".join(f"{v:>12,}" for v in cm[i]), fh)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="Labeled training CSV(s)")
    ap.add_argument("--holdout", required=True, help="Fixed holdout CSV (e.g. holdout_sample.csv)")
    ap.add_argument("--max-total-rows", type=int, default=10_000_000)
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--n-estimators", type=int, default=1000,
                    help="Max boosting rounds; early stopping usually ends sooner")
    ap.add_argument("--learning-rate", type=float, default=0.1)
    ap.add_argument("--num-leaves", type=int, default=127)
    ap.add_argument("--threads", type=int, default=4,
                    help="CPU threads; keep modest if a GPU run is going at the same time")
    ap.add_argument("--no-prior-correction", action="store_true",
                    help="Skip the extra pass that reads the training files' real class mix")
    args = ap.parse_args()

    import lightgbm as lgb

    os.makedirs(os.path.dirname(os.path.abspath(args.out_prefix)), exist_ok=True)
    fh = open(args.out_prefix + ".txt", "w")
    t0 = time.time()

    # ---- training sample (cached) ----
    cache = args.out_prefix + "_sample.pkl"
    if os.path.exists(cache):
        log(f"Loading cached training sample: {cache}", fh)
        df = pd.read_pickle(cache)
    else:
        log(f"Sampling training data with federated_simulation.load_data "
            f"(max_total_rows={args.max_total_rows:,}, seed={RNG_SEED}) ...", fh)
        df = load_data(args.paths, max_total_rows=args.max_total_rows)
        df.to_pickle(cache)
        log(f"Cached sample -> {cache}", fh)
    log(f"Training sample: {len(df):,} rows, {df['region_cell'].nunique():,} regions", fh)

    classes = np.array(sorted(df["attack_type"].astype(str).unique()))
    cls_index = {c: i for i, c in enumerate(classes)}
    X = prep_features(df)
    y = df["attack_type"].astype(str).map(cls_index).values
    del df

    train_prior = np.bincount(y, minlength=len(classes)) / len(y)
    log("Training-sample class mix: " +
        ", ".join(f"{c}={p:.3f}" for c, p in zip(classes, train_prior)), fh)

    # "balanced" weights, same formula federated_simulation uses
    counts = np.bincount(y, minlength=len(classes))
    cw = len(y) / (len(classes) * counts)
    w = cw[y]

    X_tr, X_va, y_tr, y_va, w_tr, w_va = train_test_split(
        X, y, w, test_size=0.1, stratify=y, random_state=RNG_SEED)

    # ---- train ----
    log(f"\nTraining LightGBM (leaves={args.num_leaves}, lr={args.learning_rate}, "
        f"max rounds={args.n_estimators}, threads={args.threads}) ...", fh)
    model = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=args.n_estimators,
        learning_rate=args.learning_rate, num_leaves=args.num_leaves,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
        n_jobs=args.threads, random_state=RNG_SEED, verbose=-1)
    t_train = time.time()
    model.fit(X_tr, y_tr, sample_weight=w_tr,
              eval_set=[(X_va, y_va)], eval_sample_weight=[w_va],
              callbacks=[lgb.early_stopping(30, verbose=False),
                         lgb.log_evaluation(50)])
    log(f"Trained in {(time.time() - t_train) / 60:.1f} min, "
        f"best iteration {model.best_iteration_}", fh)

    val_pred = model.predict(X_va)
    log(f"Internal validation accuracy (10% of training sample): "
        f"{(val_pred == y_va).mean():.4f}", fh)

    # ---- holdout ----
    log(f"\nReading holdout: {args.holdout}", fh)
    hd = pd.read_csv(args.holdout, usecols=FEATURE_COLS + ["attack_type"], low_memory=False)
    hd = hd[hd["attack_type"].astype(str).isin(cls_index)]
    Xh = prep_features(hd)
    yh = hd["attack_type"].astype(str).map(cls_index).values
    log(f"Holdout rows: {len(hd):,}", fh)
    del hd

    proba = model.predict_proba(Xh)
    results = {
        "args": vars(args),
        "train_rows": int(len(y)),
        "best_iteration": int(model.best_iteration_),
        "holdout_plain": evaluate("HOLDOUT: LightGBM (balanced training, as-is)",
                                  yh, proba.argmax(1), classes, fh),
    }

    # ---- prior correction (uses training files only, never the holdout) ----
    if not args.no_prior_correction:
        log("\nReading real class mix of the TRAINING files (attack_type column only) ...", fh)
        real_prior = real_class_prior(args.paths, classes)
        log("Real class mix in training days: " +
            ", ".join(f"{c}={p:.3f}" for c, p in zip(classes, real_prior)), fh)
        # The model was trained under balanced weighting, i.e. an effectively
        # uniform class prior. Re-weight its probabilities by the real prior.
        adj = proba * real_prior[None, :]
        results["real_prior"] = dict(zip(classes.tolist(), real_prior.tolist()))
        results["holdout_prior_corrected"] = evaluate(
            "HOLDOUT: LightGBM, prior-corrected to real class mix",
            yh, adj.argmax(1), classes, fh)
        # Halfway correction (square root of the prior): shows the trade-off
        # curve between false alarms and attack recall, since the full
        # correction can over-shoot when classes overlap.
        adj_half = proba * np.sqrt(real_prior)[None, :]
        results["holdout_half_corrected"] = evaluate(
            "HOLDOUT: LightGBM, half prior correction (sqrt of real mix)",
            yh, adj_half.argmax(1), classes, fh)

    # ---- feature importance ----
    gain = model.booster_.feature_importance(importance_type="gain")
    order = np.argsort(gain)[::-1]
    log("\n=== Feature importance (share of total gain) ===", fh)
    total = gain.sum()
    for i in order:
        log(f"  {X.columns[i]:<28}{gain[i] / total:>8.1%}", fh)
    results["feature_importance_gain"] = {X.columns[i]: float(gain[i] / total) for i in order}

    results["minutes_total"] = (time.time() - t0) / 60
    with open(args.out_prefix + ".json", "w") as jf:
        json.dump(results, jf, indent=2)
    log(f"\nDone in {results['minutes_total']:.1f} min. "
        f"Report: {args.out_prefix}.txt  Metrics: {args.out_prefix}.json", fh)
    fh.close()


if __name__ == "__main__":
    main()
