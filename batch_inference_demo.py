"""
batch_inference_demo.py
-------------------------------------------------------------------
Whole-file batch inference demo. Loads your trained model (same
checkpoint format as inference_server.py) and runs it over EVERY row
of a real CSV file, reporting predictions as they naturally come out.

How this differs from inference_server.py:
  inference_server.py is the LIVE, single-row, attack-injection demo --
  a browser dashboard where you click a button, sample one row at a
  time (optionally injecting a synthetic attack on top of it), and
  watch a live prediction stream in.

  This script is the FILE-DRIVEN, whole-dataset demo -- no injection,
  no synthetic tampering, no sampling. You point it at a real CSV
  (typically a holdout day's file the model never trained on, produced
  by opensky_fl_ids_pipeline.py) and it runs inference on ALL rows,
  vectorized in batches for speed, then reports how the model actually
  did (or, for unlabeled data, what it predicted) -- the "does this
  work on a whole real file" story, rather than the "watch it react
  live" story.

Model loading, feature selection, bool-casting, NaN-guarding, and
scaling all follow the EXACT same path as inference_server.py's
load_model()/predict() -- that logic has already been debugged and
validated, so this script deliberately does not diverge from it.

Only supports row-independent models (mlp/cnn) -- a GRU model needs a
real chronological window per prediction, not row-independent
inference, and isn't supported here (same restriction, same reason,
as inference_server.py).

Usage:
    python3 batch_inference_demo.py --model-path best_model.pt \
        --input-csv holdout_jan14_6h.csv

    python3 batch_inference_demo.py --model-path best_model.pt \
        --input-csv unlabeled_day.csv --output-csv predictions.csv \
        --batch-size 8192
-------------------------------------------------------------------
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd
import torch

import federated_simulation as fs

# Same bool-like columns inference_server.py's predict() casts to float
# before scaling -- kept as a single constant here so batch casting
# stays byte-for-byte consistent with the single-row path.
BOOL_COLS = [
    "is_physically_implausible",
    "squawk_changed",
    "icao24_valid_format",
    "is_stale",
    "is_low_coverage_region",
    "is_climbing_or_descending",
]

# Columns to carry through into the output CSV as identifying context,
# if present in the input file -- purely for readability, never used
# as model input.
CONTEXT_COLS = ["icao24", "callsign", "time", "lastposupdate", "lastcontact"]

MODEL = None
SCALER = None
FEATURE_COLS = None
CLASS_NAMES = None


def load_model(model_path):
    """Loads a checkpoint EXACTLY the way inference_server.py's
    load_model() does -- same payload keys, same GRU rejection, same
    paired-scaler path convention -- so this script never drifts from
    the one already-debugged loading path."""
    global MODEL, SCALER, FEATURE_COLS, CLASS_NAMES
    payload = torch.load(model_path, weights_only=False)
    if payload["model_type"] == "gru":
        raise ValueError(
            "This batch demo only supports row-independent models (mlp/cnn) -- "
            "a GRU model needs a real chronological window of several messages per "
            "prediction, not a single independent row, and would need a different "
            "batch-building approach (grouped, ordered windows per aircraft) to "
            "run meaningfully. Re-run with an mlp or cnn checkpoint."
        )
    FEATURE_COLS = payload["feature_cols"]
    CLASS_NAMES = payload["class_names"]
    MODEL = fs.build_model(
        payload["model_type"], payload["n_features"], len(CLASS_NAMES),
        mlp_hidden_sizes=payload["mlp_hidden_sizes"], cnn_channels=payload["cnn_channels"],
        device=torch.device("cpu"),
    )
    MODEL.load_state_dict(payload["model_state_dict"])
    MODEL.eval()

    preprocess_path = model_path.rsplit(".", 1)[0] + "_preprocess.pkl"
    preprocess = joblib.load(preprocess_path)
    SCALER = preprocess["scaler"]
    print(f"Loaded {payload['model_type'].upper()} model, best_round={payload['best_round']}, "
          f"best_test_acc={payload['best_test_acc']:.4f}")
    print(f"Classes: {list(CLASS_NAMES)}")


def load_input_csv(path):
    """Reads the input CSV, checking that every required FEATURE_COLS
    column is actually present (failing loudly rather than guessing or
    fabricating values), and detects whether ground-truth columns
    (label/attack_type) are present. Missing feature values are filled
    with -1, matching prep_features() at training time and
    inference_server.py's load_holdout_data() exactly -- this project's
    established finding is that -1 is the correct fill value, not
    dropping."""
    header = pd.read_csv(path, nrows=0).columns.tolist()
    missing_features = [c for c in FEATURE_COLS if c not in header]
    if missing_features:
        raise ValueError(
            f"Input CSV '{path}' is missing required feature column(s): {missing_features}. "
            f"This model was trained with FEATURE_COLS={FEATURE_COLS}; the input file must "
            f"contain all of them (e.g. run it through opensky_fl_ids_pipeline.py first). "
            f"Refusing to guess or fabricate missing values."
        )

    has_ground_truth = ("attack_type" in header) or ("label" in header)
    context_cols_present = [c for c in CONTEXT_COLS if c in header]
    usecols = list(dict.fromkeys(
        FEATURE_COLS + context_cols_present
        + (["attack_type"] if "attack_type" in header else [])
        + (["label"] if "label" in header else [])
    ))
    df = pd.read_csv(path, usecols=usecols)
    return df, has_ground_truth, context_cols_present


def predict_batch(X: pd.DataFrame):
    """Vectorized version of inference_server.py's predict() -- same
    feature-selection, bool-casting, NaN-guard, and scaling steps,
    applied to a full batch matrix instead of a single row.

    Returns (pred_idx, probs, valid_mask): pred_idx and probs are
    arrays for VALID rows only (length == valid_mask.sum()); rows with
    any NaN feature after fill are excluded and reported by the
    caller, same spirit as predict()'s NaN-guard raising rather than
    silently proceeding."""
    Xf = X[FEATURE_COLS].copy()
    for c in BOOL_COLS:
        Xf[c] = Xf[c].astype(float)

    valid_mask = ~Xf.isna().any(axis=1).to_numpy()
    if not valid_mask.all():
        Xf = Xf.loc[valid_mask]

    if len(Xf) == 0:
        return np.array([], dtype=int), np.zeros((0, len(CLASS_NAMES))), valid_mask

    X_scaled = SCALER.transform(Xf)
    with torch.no_grad():
        logits = MODEL(torch.tensor(X_scaled, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
    pred_idx = probs.argmax(axis=1)
    return pred_idx, probs, valid_mask


def run_inference(df: pd.DataFrame, batch_size: int):
    """Runs predict_batch() over the full DataFrame in chunks of
    batch_size, vectorized per chunk (not row-by-row) for speed on
    large files. Returns a results DataFrame aligned to df's index,
    with a 'skipped' column marking rows excluded for NaN features."""
    n = len(df)
    pred_class = np.full(n, "", dtype=object)
    pred_is_attack = np.zeros(n, dtype=bool)
    confidence = np.full(n, np.nan)
    all_probs = np.full((n, len(CLASS_NAMES)), np.nan)
    skipped = np.zeros(n, dtype=bool)

    n_batches = (n + batch_size - 1) // batch_size
    for b in range(n_batches):
        lo, hi = b * batch_size, min((b + 1) * batch_size, n)
        chunk = df.iloc[lo:hi]
        pred_idx, probs, valid_mask = predict_batch(chunk)

        skipped[lo:hi] = ~valid_mask
        valid_positions = np.where(valid_mask)[0]
        for j, pos in enumerate(valid_positions):
            global_i = lo + pos
            cls = CLASS_NAMES[pred_idx[j]]
            pred_class[global_i] = cls
            pred_is_attack[global_i] = (cls != "benign")
            confidence[global_i] = probs[j].max()
            all_probs[global_i] = probs[j]

        done = hi
        print(f"  Processed {done:,}/{n:,} rows ({done/n:.0%})", end="\r")
    print()

    return pd.DataFrame({
        "predicted_class": pred_class,
        "predicted_is_attack": pred_is_attack,
        "confidence": confidence,
        "skipped": skipped,
    }), all_probs


def print_labeled_summary(results, ground_truth):
    """Prints accuracy, per-class precision/recall, and a confusion
    matrix (predicted vs actual attack_type) -- the 'does it actually
    work on a whole real file' story."""
    valid = ~results["skipped"].to_numpy()
    y_true = ground_truth.to_numpy()[valid]
    y_pred = results["predicted_class"].to_numpy()[valid]

    n_valid = valid.sum()
    overall_acc = (y_true == y_pred).mean() if n_valid > 0 else float("nan")

    all_labels = sorted(set(y_true) | set(y_pred))

    print("\n" + "=" * 60)
    print("ACCURACY SUMMARY (labeled file -- real ground truth)")
    print("=" * 60)
    print(f"Rows evaluated: {n_valid:,}")
    print(f"Overall accuracy: {overall_acc:.4f} ({overall_acc:.1%})")

    print(f"\n{'Class':<20}{'Precision':>12}{'Recall':>12}{'Support':>10}")
    print("-" * 54)
    for cls in all_labels:
        true_mask = (y_true == cls)
        pred_mask = (y_pred == cls)
        support = int(true_mask.sum())
        tp = int((true_mask & pred_mask).sum())
        recall = tp / support if support > 0 else float("nan")
        pred_count = int(pred_mask.sum())
        precision = tp / pred_count if pred_count > 0 else float("nan")
        prec_str = f"{precision:.3f}" if not np.isnan(precision) else "n/a"
        rec_str = f"{recall:.3f}" if not np.isnan(recall) else "n/a"
        print(f"{cls:<20}{prec_str:>12}{rec_str:>12}{support:>10,}")

    print("\nConfusion matrix (rows = actual, columns = predicted):")
    header = "".join(f"{c[:10]:>12}" for c in all_labels)
    print(f"{'':<14}{header}")
    for actual in all_labels:
        row = []
        for predicted in all_labels:
            count = int(((y_true == actual) & (y_pred == predicted)).sum())
            row.append(f"{count:>12,}")
        print(f"{actual[:12]:<14}{''.join(row)}")


def print_unlabeled_summary(results, n_total):
    """No ground truth available -- just report the distribution of
    predicted classes, with no accuracy claim."""
    valid_results = results[~results["skipped"]]
    print("\n" + "=" * 60)
    print("PREDICTION SUMMARY (unlabeled file -- no accuracy available)")
    print("=" * 60)
    print(f"Rows predicted: {len(valid_results):,} / {n_total:,}")
    print(f"\n{'Predicted class':<20}{'Count':>10}{'Share':>10}")
    print("-" * 40)
    counts = valid_results["predicted_class"].value_counts()
    for cls, count in counts.items():
        print(f"{cls:<20}{count:>10,}{count/len(valid_results):>9.1%}")


def main():
    p = argparse.ArgumentParser(
        description="Run whole-file batch inference with a trained FL-IDS model "
                    "on real (non-injected) data and report the results."
    )
    p.add_argument("--model-path", required=True,
                   help="Trained checkpoint (.pt), same format as inference_server.py "
                        "expects. Must be an mlp or cnn model, not gru.")
    p.add_argument("--input-csv", required=True,
                   help="CSV file of real data to run inference on -- ideally a labeled "
                        "file from opensky_fl_ids_pipeline.py (e.g. a holdout day the "
                        "model never trained on), but an unlabeled file also works.")
    p.add_argument("--output-csv", default=None,
                   help="Where to write per-row predictions. Default: "
                        "'<input-csv stem>_predictions.csv' next to the input file.")
    p.add_argument("--batch-size", type=int, default=4096,
                   help="Rows per vectorized forward pass (default 4096). Larger batches "
                        "are faster but use more memory; this has no effect on results, "
                        "only speed.")
    args = p.parse_args()

    if args.output_csv:
        output_csv = args.output_csv
    else:
        base, _ext = os.path.splitext(args.input_csv)
        output_csv = f"{base}_predictions.csv"

    load_model(args.model_path)

    print(f"\nReading input CSV: {args.input_csv} ...")
    df, has_ground_truth, context_cols_present = load_input_csv(args.input_csv)
    n_total = len(df)
    print(f"Loaded {n_total:,} rows.")

    n_incomplete = df[FEATURE_COLS].isna().any(axis=1).sum()
    if n_incomplete > 0:
        print(f"  Filling {n_incomplete:,} rows' missing feature values with -1 "
              f"({n_incomplete/n_total:.1%} of loaded data) -- matches training's own "
              f"prep_features() convention, does not drop real data.")
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(-1)

    ground_truth = None
    if has_ground_truth:
        if "attack_type" in df.columns:
            ground_truth = df["attack_type"]
        else:
            # Only 'label' (0/1) present, no class name -- map to
            # binary benign/attack so the summary still means something.
            ground_truth = df["label"].map({0: "benign", 1: "attack"})
        print(f"Ground-truth column found -- will report accuracy against real labels.")
    else:
        print("No 'label'/'attack_type' column found -- treating as unlabeled data "
              "(predictions only, no accuracy claim).")

    print(f"\nRunning inference in batches of {args.batch_size:,} ...")
    results, all_probs = run_inference(df, args.batch_size)

    n_skipped = int(results["skipped"].sum())
    if n_skipped > 0:
        print(f"\nWARNING: skipped {n_skipped:,} row(s) ({n_skipped/n_total:.1%}) that still "
              f"contained NaN feature values after the -1 fill (e.g. a feature column "
              f"entirely absent of usable data for those rows) -- these were EXCLUDED from "
              f"predictions and accuracy, not fabricated.")

    # Build output CSV
    out = pd.DataFrame(index=df.index)
    for c in context_cols_present:
        out[c] = df[c]
    if has_ground_truth:
        out["actual_attack_type"] = ground_truth
    out["predicted_class"] = results["predicted_class"]
    out["predicted_is_attack"] = results["predicted_is_attack"]
    out["confidence"] = results["confidence"].round(4)
    if has_ground_truth:
        valid = ~results["skipped"]
        correct = pd.Series(np.nan, index=df.index, dtype=object)
        correct.loc[valid] = (ground_truth.loc[valid].to_numpy()
                               == results.loc[valid, "predicted_class"].to_numpy())
        out["correct"] = correct
    out["skipped_nan_features"] = results["skipped"]

    out.to_csv(output_csv, index=False)

    # Console summary
    if has_ground_truth:
        print_labeled_summary(results, ground_truth)
    else:
        print_unlabeled_summary(results, n_total)

    print("\n" + "=" * 60)
    print(f"Rows processed: {n_total:,}  |  Skipped (NaN features): {n_skipped:,}")
    print(f"Predictions written to: {output_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
