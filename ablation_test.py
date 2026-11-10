"""
ablation_test.py
-------------------------------------------------------------------
Tests whether the model detects flooding via the genuine inter_arrival_s
timing signal, or via the -1 fillna shortcut (implied_speed_mps,
speed_consistency_delta, vrate_consistency_delta, turn_rate_deg_s all
being exactly -1, a pattern that only occurs for flooding duplicates,
since their near-zero time delta makes these rate-based features
uncomputable upstream in the pipeline).

Method: take REAL flooding duplicate rows (gap < 1s) from a held-out
file. For each one, keep inter_arrival_s untouched (the genuine timing
signal) but replace the 4 kinematic features with a "typical" value
(the scaler's own fitted mean -- i.e. what an unremarkable, average row
looks like) instead of -1. Compare the model's flooding confidence
before and after this single substitution.

  - If confidence COLLAPSES after removing the -1 pattern: the model was
    leaning on the shortcut, not genuinely reading the timing signal.
  - If confidence STAYS HIGH: inter_arrival_s itself carries real weight
    too, independent of the -1 artifact.

Usage:
    python3 ablation_test.py --model-path best_model.pt \
        --holdout-data-path holdout_jan14_6h.csv --n-samples 300
-------------------------------------------------------------------
"""
import argparse

import joblib
import numpy as np
import pandas as pd
import torch

import federated_simulation as fs

KINEMATIC_COLS = ["implied_speed_mps", "speed_consistency_delta",
                   "vrate_consistency_delta", "turn_rate_deg_s"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--holdout-data-path", required=True)
    p.add_argument("--n-samples", type=int, default=300)
    args = p.parse_args()

    payload = torch.load(args.model_path, weights_only=False)
    if payload["model_type"] == "gru":
        raise ValueError("Ablation test only supports row-independent models (mlp/cnn).")
    feature_cols = payload["feature_cols"]
    class_names = list(payload["class_names"])
    flooding_idx = class_names.index("flooding")

    model = fs.build_model(payload["model_type"], payload["n_features"], len(class_names),
                            mlp_hidden_sizes=payload["mlp_hidden_sizes"],
                            cnn_channels=payload["cnn_channels"], device=torch.device("cpu"))
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    preprocess_path = args.model_path.rsplit(".", 1)[0] + "_preprocess.pkl"
    scaler = joblib.load(preprocess_path)["scaler"]

    # "Typical" real-unit value for each kinematic column = the scaler's
    # own fitted mean -- literally the average row the model trained on,
    # in real units, for that specific feature.
    mean_by_col = dict(zip(feature_cols, scaler.mean_))

    print(f"Loaded {payload['model_type'].upper()} model, best_test_acc={payload['best_test_acc']:.4f}")
    print(f"'Typical' substitution values: " +
          ", ".join(f"{c}={mean_by_col[c]:.2f}" for c in KINEMATIC_COLS))

    df = pd.read_csv(args.holdout_data_path, usecols=feature_cols + ["attack_type"])
    flooding = df[df["attack_type"] == "flooding"]
    duplicates = flooding[flooding["inter_arrival_s"] < 1].copy()
    duplicates[feature_cols] = duplicates[feature_cols].fillna(-1)
    # Only rows that actually show the -1 pattern are a valid test case --
    # confirms we're testing the real phenomenon, not incidental rows.
    has_shortcut = (duplicates[KINEMATIC_COLS] == -1).all(axis=1)
    duplicates = duplicates[has_shortcut]
    print(f"\nFound {len(duplicates):,} real flooding duplicates with the -1 pattern "
          f"(out of {len(flooding):,} total flooding rows).")

    n = min(args.n_samples, len(duplicates))
    sample = duplicates.sample(n=n, random_state=1)

    def predict_flooding_conf(row_dict):
        X = pd.DataFrame([row_dict])[feature_cols].copy()
        bool_cols = ["is_physically_implausible", "squawk_changed", "icao24_valid_format",
                     "is_stale", "is_low_coverage_region", "is_climbing_or_descending"]
        for c in bool_cols:
            X[c] = X[c].astype(float)
        X_scaled = scaler.transform(X)
        with torch.no_grad():
            probs = torch.softmax(model(torch.tensor(X_scaled, dtype=torch.float32)), dim=1)[0]
        return probs[flooding_idx].item(), class_names[probs.argmax().item()]

    before_confs, after_confs = [], []
    before_correct, after_correct = 0, 0

    for _, row in sample.iterrows():
        original = row.to_dict()
        conf_before, pred_before = predict_flooding_conf(original)
        before_confs.append(conf_before)
        if pred_before == "flooding":
            before_correct += 1

        ablated = dict(original)
        for c in KINEMATIC_COLS:
            ablated[c] = mean_by_col[c]  # replace -1 with a "typical" value; inter_arrival_s untouched
        conf_after, pred_after = predict_flooding_conf(ablated)
        after_confs.append(conf_after)
        if pred_after == "flooding":
            after_correct += 1

    before_confs = np.array(before_confs)
    after_confs = np.array(after_confs)

    print(f"\n{'='*70}")
    print(f"RESULTS -- {n} real flooding duplicates, -1 pattern removed one row at a time")
    print(f"{'='*70}")
    print(f"BEFORE (real -1 values):  mean flooding confidence = {before_confs.mean():.4f}   "
          f"correctly predicted flooding: {before_correct}/{n} ({before_correct/n:.1%})")
    print(f"AFTER  (-1 -> typical):   mean flooding confidence = {after_confs.mean():.4f}   "
          f"correctly predicted flooding: {after_correct}/{n} ({after_correct/n:.1%})")
    print(f"\nConfidence drop: {before_confs.mean() - after_confs.mean():.4f} "
          f"({(1 - after_confs.mean()/max(before_confs.mean(),1e-9)):.1%} relative decrease)")

    if after_correct / n < 0.3 and before_correct / n > 0.7:
        print("\nVERDICT: model relies heavily on the -1 shortcut, not genuine timing signal.")
    elif after_correct / n > 0.6:
        print("\nVERDICT: model still detects flooding well via inter_arrival_s alone -- "
              "the -1 pattern is NOT the primary signal.")
    else:
        print("\nVERDICT: mixed -- both the -1 pattern and inter_arrival_s appear to contribute.")


if __name__ == "__main__":
    main()
