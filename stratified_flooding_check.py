"""
stratified_flooding_check.py
-------------------------------------------------------------------
Necessary follow-up to ablation_test.py's finding. ablation_test.py
showed that of 1,143,664 total flooding-labeled rows in the holdout
file, 980,371 (85.7%) show a specific artifact: the 4 kinematic
columns (implied_speed_mps, speed_consistency_delta,
vrate_consistency_delta, turn_rate_deg_s) are all exactly -1 after
the standard fillna(-1) convention, because flooding's near-duplicate
timestamps make these rate-based features mathematically
uncomputable upstream. Swapping that -1 pattern for "typical"
(scaler-mean) values, while leaving the genuine inter_arrival_s
timing signal untouched, collapsed flooding confidence from 0.8922 to
0.0000 and accuracy from 100% to 0% across 300 sampled rows -- i.e.
the model's flooding detection appeared to rely almost entirely on
the -1 sentinel pattern, not on genuinely reading the timing signal.

That leaves 1,143,664 - 980,371 = 163,293 real flooding rows that do
NOT show the -1 pattern (at least one of the 4 kinematic columns has
a real, non -1 computed value after the fill). This script answers
the open question those rows raise: does the model detect flooding
at all on THIS subset, where the shortcut pattern isn't available?

Method: load the real holdout file, filter to attack_type ==
"flooding", fill NaNs with -1 (this project's established
convention), and split into:
  Group A (has_shortcut=True)  -- the "artifact" rows, all 4
                                   kinematic columns exactly -1.
  Group B (has_shortcut=False) -- the rows without the pattern, the
                                   ones this follow-up is about.
Run real vectorized batch inference (same pattern as
batch_inference_demo.py's predict_batch) on both groups separately
and report accuracy at predicting "flooding", mean softmax
probability assigned to the true "flooding" class (comparable to
ablation_test.py's "mean flooding confidence"), and, for Group B,
a breakdown of what it gets misclassified as when it isn't predicted
as flooding.

  - If Group B accuracy/confidence is still high: there IS a genuine
    secondary signal beyond the -1 artifact -- the earlier finding is
    "mostly shortcut, some real signal".
  - If Group B does badly too: flooding detection barely works at all
    without the artifact, a stronger and more important finding than
    the original ablation result alone.

Usage:
    python3 stratified_flooding_check.py --model-path best_model.pt \
        --holdout-data-path holdout_jan14_6h.csv
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

# Same bool-like columns inference_server.py's predict() / batch_inference_demo.py's
# predict_batch() cast to float before scaling -- kept identical here so this
# script never drifts from the one already-debugged path.
BOOL_COLS = [
    "is_physically_implausible",
    "squawk_changed",
    "icao24_valid_format",
    "is_stale",
    "is_low_coverage_region",
    "is_climbing_or_descending",
]


def load_model(model_path):
    """Loads a checkpoint EXACTLY the way inference_server.py's and
    batch_inference_demo.py's load_model() do -- same payload keys, same
    GRU rejection, same paired _preprocess.pkl scaler loading."""
    payload = torch.load(model_path, weights_only=False)
    if payload["model_type"] == "gru":
        raise ValueError(
            "This check only supports row-independent models (mlp/cnn) -- a GRU "
            "model needs a real chronological window per prediction, not "
            "row-independent inference. Re-run with an mlp or cnn checkpoint."
        )
    feature_cols = payload["feature_cols"]
    class_names = list(payload["class_names"])

    model = fs.build_model(
        payload["model_type"], payload["n_features"], len(class_names),
        mlp_hidden_sizes=payload["mlp_hidden_sizes"], cnn_channels=payload["cnn_channels"],
        device=torch.device("cpu"),
    )
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    preprocess_path = model_path.rsplit(".", 1)[0] + "_preprocess.pkl"
    scaler = joblib.load(preprocess_path)["scaler"]

    print(f"Loaded {payload['model_type'].upper()} model, best_test_acc={payload['best_test_acc']:.4f}")
    print(f"Classes: {class_names}")
    return model, scaler, feature_cols, class_names


def predict_batch(X: pd.DataFrame, model, scaler, feature_cols, class_names):
    """Vectorized prediction, same bool-casting / scaling / forward-pass
    pattern as batch_inference_demo.py's predict_batch(). Assumes the
    caller has already fillna(-1)'d the feature columns, so no NaN-guard
    row-exclusion is needed here (unlike batch_inference_demo.py, which
    handles arbitrary unfilled input)."""
    Xf = X[feature_cols].copy()
    for c in BOOL_COLS:
        Xf[c] = Xf[c].astype(float)
    X_scaled = scaler.transform(Xf)
    with torch.no_grad():
        logits = model(torch.tensor(X_scaled, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1).numpy()
    pred_idx = probs.argmax(axis=1)
    pred_class = np.array(class_names)[pred_idx]
    return pred_class, probs


def summarize_group(name, df, model, scaler, feature_cols, class_names, flooding_idx):
    n = len(df)
    if n == 0:
        print(f"\n{name}: 0 rows -- skipping.")
        return {"n": 0, "accuracy": float("nan"), "mean_flooding_conf": float("nan"),
                "pred_class": np.array([]), "n": 0}

    pred_class, probs = predict_batch(df, model, scaler, feature_cols, class_names)
    correct = (pred_class == "flooding")
    accuracy = correct.mean()
    mean_flooding_conf = probs[:, flooding_idx].mean()

    print(f"\n{name}: {n:,} rows")
    print(f"  Accuracy (predicted as flooding): {correct.sum():,}/{n:,} ({accuracy:.1%})")
    print(f"  Mean softmax probability assigned to true class 'flooding': {mean_flooding_conf:.4f}")

    return {
        "n": n,
        "accuracy": accuracy,
        "mean_flooding_conf": mean_flooding_conf,
        "pred_class": pred_class,
    }


def main():
    p = argparse.ArgumentParser(
        description="Stratified follow-up to ablation_test.py: checks whether the model "
                    "detects flooding on real flooding rows that do NOT show the -1 "
                    "kinematic-column shortcut pattern."
    )
    p.add_argument("--model-path", required=True)
    p.add_argument("--holdout-data-path", required=True)
    p.add_argument("--seed", type=int, default=1,
                   help="Random seed (default 1, matching ablation_test.py's random_state=1 "
                        "convention). Not used for sampling here (this script evaluates full "
                        "groups, not a sample), kept for consistency/reproducibility.")
    args = p.parse_args()

    np.random.seed(args.seed)

    model, scaler, feature_cols, class_names = load_model(args.model_path)
    flooding_idx = class_names.index("flooding")

    df = pd.read_csv(args.holdout_data_path, usecols=feature_cols + ["attack_type"])
    flooding = df[df["attack_type"] == "flooding"].copy()
    flooding[feature_cols] = flooding[feature_cols].fillna(-1)
    total_flooding = len(flooding)

    has_shortcut = (flooding[KINEMATIC_COLS] == -1).all(axis=1)
    group_a = flooding[has_shortcut]   # artifact rows
    group_b = flooding[~has_shortcut]  # rows without the pattern -- the subject of this check

    print(f"\nTotal flooding rows: {total_flooding:,}")
    print(f"Group A (has -1 shortcut pattern): {len(group_a):,} ({len(group_a)/total_flooding:.1%})")
    print(f"Group B (no -1 shortcut pattern):  {len(group_b):,} ({len(group_b)/total_flooding:.1%})")

    print(f"\n{'='*70}")
    print("PER-GROUP RESULTS")
    print(f"{'='*70}")

    res_a = summarize_group("Group A (artifact rows)", group_a, model, scaler,
                             feature_cols, class_names, flooding_idx)
    res_b = summarize_group("Group B (no artifact)", group_b, model, scaler,
                             feature_cols, class_names, flooding_idx)

    if res_b["n"] > 0:
        misclassified_mask = res_b["pred_class"] != "flooding"
        n_misclassified = int(misclassified_mask.sum())
        print(f"\nGroup B misclassification breakdown "
              f"({n_misclassified:,}/{res_b['n']:,} rows not predicted as flooding):")
        if n_misclassified > 0:
            counts = pd.Series(res_b["pred_class"][misclassified_mask]).value_counts()
            for cls, count in counts.items():
                print(f"  predicted {cls:<15} {count:>8,}  ({count/n_misclassified:.1%} of misclassified, "
                      f"{count/res_b['n']:.1%} of group B)")
        else:
            print("  (none -- every Group B row was predicted as flooding)")

    print(f"\n{'='*70}")
    print("FINAL COMPARISON")
    print(f"{'='*70}")
    print(f"{'':<28}{'Group A (artifact)':>22}{'Group B (no artifact)':>24}")
    print(f"{'Rows':<28}{res_a['n']:>22,}{res_b['n']:>24,}")
    a_acc_str = f"{res_a['accuracy']:.1%}" if res_a["n"] > 0 else "n/a"
    b_acc_str = f"{res_b['accuracy']:.1%}" if res_b["n"] > 0 else "n/a"
    print(f"{'Accuracy (as flooding)':<28}{a_acc_str:>22}{b_acc_str:>24}")
    a_conf_str = f"{res_a['mean_flooding_conf']:.4f}" if res_a["n"] > 0 else "n/a"
    b_conf_str = f"{res_b['mean_flooding_conf']:.4f}" if res_b["n"] > 0 else "n/a"
    print(f"{'Mean flooding confidence':<28}{a_conf_str:>22}{b_conf_str:>24}")

    if res_b["n"] == 0:
        print("\nVERDICT: no Group B rows found in this holdout file -- cannot assess "
              "whether a genuine secondary signal exists.")
    elif res_b["accuracy"] > 0.6:
        print("\nVERDICT: the model still detects flooding well on rows WITHOUT the -1 "
              "artifact -- a genuine secondary signal exists beyond the shortcut pattern.")
    elif res_b["accuracy"] < 0.3:
        print("\nVERDICT: flooding detection collapses on rows without the -1 artifact -- "
              "the model's flooding detection depends almost entirely on the shortcut "
              "pattern, with little to no genuine secondary signal.")
    else:
        print("\nVERDICT: mixed -- the model retains some real flooding signal without the "
              "-1 artifact, but noticeably weaker than on artifact rows.")


if __name__ == "__main__":
    main()
