"""
adsb_feature_importance.py
-------------------------------------------------------------------
Computes REAL feature importance against your actual ADS-B attack
labels -- both overall (multiclass) and PER ATTACK TYPE (one-vs-benign),
so you get a direct answer to "which features actually flag THIS
specific attack" rather than one blended ranking across all five
attack types combined.

Usage:
    python3 adsb_feature_importance.py \
        --adsb-path ~/fl-ids-project/states_2017-06-05-00_labeled.csv \
        --sample-size 60000
-------------------------------------------------------------------
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import gaf_vs_raw_cross_domain as mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--adsb-path", required=True)
    p.add_argument("--sample-size", type=int, default=60000,
                    help="Rows per class to pull via the streaming loader's reservoir "
                         "(passed as target_rows). Higher = more reliable MI estimates, "
                         "slower to compute.")
    p.add_argument("--adsb-chunksize", type=int, default=50000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = mod.load_adsb_opensky(args.adsb_path, target_rows=args.sample_size,
                                chunksize=args.adsb_chunksize, seed=args.seed)
    print(f"\nLoaded {len(df):,} rows:")
    print(df["label"].value_counts())

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    print(f"\n{len(numeric_cols)} numeric candidate features: {numeric_cols}")

    X = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())  # diagnostic-only fill, not the leakage-safe pipeline fill

    attack_types = sorted(a for a in df["label"].unique() if a != "benign")
    print(f"Attack types found: {attack_types}")

    # ---------------- Overall multiclass importance (context) ----------------
    y_all = LabelEncoder().fit_transform(df["label"].astype(str))
    print("\nComputing overall multiclass mutual information...")
    mi_overall = mutual_info_classif(X, y_all, random_state=args.seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_all, test_size=0.3, random_state=args.seed, stratify=y_all)
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, random_state=args.seed,
                                 n_jobs=-1, class_weight="balanced")
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"RF multiclass test accuracy (sanity check only): {rf_acc:.3f}")

    overall = pd.DataFrame({
        "feature": numeric_cols,
        "mi_overall": mi_overall,
        "rf_importance_overall": rf.feature_importances_,
    }).sort_values("mi_overall", ascending=False).reset_index(drop=True)
    overall["rank_overall"] = overall.index + 1

    # ---------------- Per-attack-type importance (the actual answer) ----------------
    per_attack_mi = {}
    for atk in attack_types:
        mask = df["label"].isin(["benign", atk])
        y_bin = (df.loc[mask, "label"] == atk).astype(int).to_numpy()
        X_bin = X.loc[mask]
        print(f"Computing mutual information: {atk} vs benign ({mask.sum():,} rows)...")
        per_attack_mi[atk] = mutual_info_classif(X_bin, y_bin, random_state=args.seed)

    per_attack_df = pd.DataFrame(per_attack_mi, index=numeric_cols)
    per_attack_df.index.name = "feature"

    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")

    print("\n" + "=" * 100)
    print("OVERALL FEATURE IMPORTANCE (multiclass, all attack types blended together)")
    print("=" * 100)
    print(overall.to_string(index=False))

    print("\n" + "=" * 100)
    print("PER-ATTACK-TYPE MUTUAL INFORMATION (one-vs-benign) -- full matrix")
    print("=" * 100)
    print(per_attack_df.to_string())

    print("\n" + "=" * 100)
    print("ANSWER: TOP 3 MOST DISCRIMINATIVE FEATURES, PER ATTACK TYPE")
    print("=" * 100)
    for atk in attack_types:
        top3 = per_attack_df[atk].sort_values(ascending=False).head(3)
        print(f"\n{atk}:")
        for feat, score in top3.items():
            print(f"    {feat:<30s} MI = {score:.4f}")


if __name__ == "__main__":
    main()
