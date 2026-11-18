"""
make_fixture.py: build a small FAKE project root for testing the FL-IDS demo.

Everything is synthetic. Nothing here comes from real OpenSky data. The fixture
mirrors the real input layout from demo/CONTRACT.md so every demo component can be
exercised end to end without the user's 300 MB pickle or 688k-row holdout.

CLI:
    python3 demo/tools/make_fixture.py                        # -> /tmp/fl-demo-fixture
    python3 demo/tools/make_fixture.py --root /some/dir       # another location
    python3 demo/tools/make_fixture.py --sample-rows 60000 --holdout-rows 40000 --seed 7
    python3 demo/tools/make_fixture.py --skip-model           # data only, no torch needed

Flags:
    --root PATH          fixture root (default /tmp/fl-demo-fixture), created if missing
    --sample-rows N      rows in models/sample_3m.pkl (default 60000)
    --holdout-rows N     rows in the holdout CSV (default 40000)
    --seed N             RNG seed (default 42)
    --train-epochs N     epochs for the tiny detection model (default 4)
    --skip-model         do not train/save models/fixed4_cnn.pt

Files written under ROOT:
    models/sample_3m.pkl
        {"meta": {...}, "df": DataFrame}, the same format as
        federated_simulation.load_data_cached. Columns: region_cell, attack_type,
        icao24, time, lastposupdate + the 13 FEATURE_COLS. Boolean features are
        Python bools (bool dtype, or object dtype with NaN for one column), some
        kinematic NaNs, all six classes, the six demo cells plus four other cells.
    opensky_data/labeled_fixed/holdout_feb04_1pct.csv
        ~79% benign, flooding ~13%, the rest spread over the other attacks.
        Booleans as literal "True"/"False", occasional blanks and a rare "-" in
        numeric columns, plus extra columns (lat, lon, callsign) the demo ignores.
        Cell 5_-14 (C6) is deliberately short of velocity_spoof and replay rows
        so the pool top-up/borrowing rule gets exercised.
    models/fixed4_cnn.pt + models/fixed4_cnn_preprocess.pkl
        tiny CNN (channels 32,64) built with federated_simulation.build_model and
        trained for a few epochs, saved in the real payload format.
    results/lgbm_feb04.json
        {"real_prior": {...}} plus a couple of unrelated keys.

Then, for example:
    python3 demo/prepare_demo_data.py --sample /tmp/fl-demo-fixture/models/sample_3m.pkl \
        --holdout /tmp/fl-demo-fixture/opensky_data/labeled_fixed/holdout_feb04_1pct.csv \
        --out /tmp/fl-demo-fixture/demo/data
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_DIR = Path(__file__).resolve().parent.parent
ROOT = DEMO_DIR.parent
sys.path.insert(0, str(ROOT))

import federated_simulation as fs  # noqa: E402
sys.path.insert(0, str(DEMO_DIR))
import fs_compat  # noqa: E402  (demo-side wrapper around fs.prep_features)

CLASSES = ["benign", "flooding", "ghost_aircraft", "gps_spoof", "replay", "velocity_spoof"]
DEMO_CELLS = ["4_-10", "5_1", "3_-9", "0_10", "-4_14", "5_-14"]
OTHER_CELLS = ["4_0", "3_-12", "5_0", "2_7"]
BOOL_COLS = ["is_physically_implausible", "squawk_changed", "icao24_valid_format",
             "is_stale", "is_low_coverage_region", "is_climbing_or_descending"]
KINEMATIC = ["implied_speed_mps", "speed_consistency_delta", "vrate_consistency_delta",
             "turn_rate_deg_s"]
DAY_START = 1549238400  # 2019-02-04 00:00:00 UTC

REAL_PRIOR = {"benign": 0.7942, "flooding": 0.1318, "ghost_aircraft": 0.0215,
              "gps_spoof": 0.0201, "replay": 0.0187, "velocity_spoof": 0.0137}

# sample: per-(region, class) capping in the real loader makes it far more balanced
SAMPLE_MIX = {"benign": 0.40, "flooding": 0.14, "ghost_aircraft": 0.11,
              "gps_spoof": 0.12, "replay": 0.11, "velocity_spoof": 0.12}


def make_rows(n, cls, cell, rng):
    """Synthetic feature rows for one class in one cell. Classes are separated by
    a few strong signals so a briefly trained model clearly detects them, with
    some overlap (replay is the hardest, like in the real project)."""
    low_cov = cell in ("5_-14", "-4_14", "0_10")
    region_iat = {True: 9.0, False: 2.5}[low_cov] * rng.uniform(0.9, 1.1)

    d = {}
    d["implied_speed_mps"] = rng.normal(210, 45, n).clip(40, 320)
    d["speed_consistency_delta"] = np.abs(rng.normal(0, 3, n))
    d["vrate_consistency_delta"] = np.abs(rng.normal(0, 1.5, n))
    d["inter_arrival_s"] = rng.exponential(region_iat, n) + 0.3
    d["turn_rate_deg_s"] = np.abs(rng.normal(0, 0.6, n))
    d["track_message_count"] = rng.integers(5, 600, n).astype(float)
    d["is_physically_implausible"] = rng.random(n) < 0.005
    d["squawk_changed"] = rng.random(n) < 0.01
    d["icao24_valid_format"] = rng.random(n) > 0.001
    d["is_stale"] = rng.random(n) < 0.03
    d["region_avg_inter_arrival"] = np.full(n, region_iat) + rng.normal(0, 0.05, n)
    d["is_low_coverage_region"] = np.full(n, low_cov)
    d["is_climbing_or_descending"] = rng.random(n) < 0.3

    if cls == "flooding":
        d["inter_arrival_s"] = rng.uniform(0.0005, 0.05, n)
        d["track_message_count"] = rng.integers(800, 5000, n).astype(float)
    elif cls == "ghost_aircraft":
        for c in KINEMATIC:
            d[c] = np.full(n, np.nan)
        d["track_message_count"] = np.ones(n)
        d["icao24_valid_format"] = rng.random(n) > 0.35
        d["inter_arrival_s"] = np.full(n, np.nan)
    elif cls == "gps_spoof":
        d["implied_speed_mps"] = rng.uniform(600, 6000, n)
        d["is_physically_implausible"] = rng.random(n) < 0.9
        d["speed_consistency_delta"] = rng.uniform(300, 5000, n)
        d["turn_rate_deg_s"] = rng.uniform(5, 60, n)
    elif cls == "replay":
        # only about half of replay rows carry a clear timing signal: replay
        # stays the hardest class, as in the real project
        neg = rng.random(n) < 0.55
        d["inter_arrival_s"] = np.where(neg, -rng.exponential(30, n), d["inter_arrival_s"])
        d["is_stale"] = rng.random(n) < 0.4
        d["speed_consistency_delta"] = np.abs(rng.normal(0, 8, n))
    elif cls == "velocity_spoof":
        strong = rng.random(n) < 0.85
        d["speed_consistency_delta"] = np.where(strong, rng.uniform(60, 400, n),
                                                np.abs(rng.normal(0, 10, n)))
        d["vrate_consistency_delta"] = np.where(strong, rng.uniform(15, 80, n),
                                                np.abs(rng.normal(0, 4, n)))
    df = pd.DataFrame(d)

    # sprinkle realistic missing kinematics on non-ghost rows (first message of a track)
    if cls != "ghost_aircraft":
        m = rng.random(n) < 0.02
        df.loc[m, ["implied_speed_mps", "turn_rate_deg_s"]] = np.nan

    df["region_cell"] = cell
    df["attack_type"] = cls
    df["icao24"] = [f"{x:06x}" for x in rng.integers(0x100000, 0xFFFFFF, n)]
    df["time"] = np.sort(rng.integers(DAY_START, DAY_START + 86400, n))
    df["lastposupdate"] = df["time"] - rng.uniform(0, 3, n).round(3)
    return df


def build_sample(n_rows, rng):
    weights = {c: 1.0 for c in DEMO_CELLS} | {c: 0.4 for c in OTHER_CELLS}
    tot_w = sum(weights.values())
    parts = []
    for cell, w in weights.items():
        n_cell = int(n_rows * w / tot_w)
        for cls, frac in SAMPLE_MIX.items():
            parts.append(make_rows(max(1, int(n_cell * frac)), cls, cell, rng))
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    cols = ["region_cell", "attack_type", "icao24", "time", "lastposupdate"] + fs.FEATURE_COLS
    df = df[cols]
    for c in BOOL_COLS:
        df[c] = df[c].astype(bool)
    # one boolean column with missing values, as the real CSV loader produces:
    # object dtype holding Python True/False and NaN
    s = df["squawk_changed"].astype(object)
    s[rng.random(len(s)) < 0.01] = np.nan
    df["squawk_changed"] = s
    return df


def build_holdout(n_rows, rng):
    weights = {c: 1.0 for c in DEMO_CELLS} | {c: 0.6 for c in OTHER_CELLS}
    tot_w = sum(weights.values())
    parts = []
    for cell, w in weights.items():
        n_cell = int(n_rows * w / tot_w)
        for cls, frac in REAL_PRIOR.items():
            n = int(n_cell * frac)
            if cell == "5_-14" and cls in ("velocity_spoof", "replay"):
                n = 12  # thin on purpose: exercises the pool borrowing rule
            parts.append(make_rows(max(1, n), cls, cell, rng))
    df = pd.concat(parts, ignore_index=True)
    df = df.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    a, b = (df["region_cell"].str.split("_", expand=True).astype(int).values.T)
    df["lat"] = (10 * a + rng.uniform(0, 10, len(df))).round(4)
    df["lon"] = (10 * b + rng.uniform(0, 10, len(df))).round(4)
    df["callsign"] = [f"FX{x:04d}" for x in rng.integers(0, 9999, len(df))]

    out = pd.DataFrame()
    for c in ["time", "icao24", "lat", "lon", "callsign", "lastposupdate"] + fs.FEATURE_COLS \
            + ["region_cell", "attack_type"]:
        col = df[c]
        if c in BOOL_COLS:
            col = col.map({True: "True", False: "False"})
        out[c] = col
    # occasional blanks and junk text in numeric columns, like the real export
    for c in ["implied_speed_mps", "vrate_consistency_delta", "turn_rate_deg_s"]:
        s = out[c].astype(object)
        s[rng.random(len(s)) < 0.003] = np.nan
        s[rng.random(len(s)) < 0.0005] = "-"
        out[c] = s
    return out


def train_tiny_model(sample, models_dir, epochs, rng_seed):
    import joblib
    import torch
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    torch.manual_seed(rng_seed)
    X = fs_compat.prep_features(fs, sample)
    le = LabelEncoder()
    y = le.fit_transform(sample["attack_type"].astype(str))
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X).astype(np.float32)
    counts = np.bincount(y, minlength=len(le.classes_))
    cw = torch.tensor(len(y) / (len(le.classes_) * counts), dtype=torch.float32)

    cnn_channels = (32, 64)
    model = fs.build_model("cnn", len(fs.FEATURE_COLS), len(le.classes_),
                           cnn_channels=cnn_channels, device=torch.device("cpu"))
    fs.DEVICE = torch.device("cpu")
    perm = np.random.default_rng(rng_seed).permutation(len(y))
    cut = int(0.8 * len(y))
    tr, te = perm[:cut], perm[cut:]
    fs.local_train(model, Xs[tr], y[tr], epochs, cw)
    acc, per_class = fs.evaluate(model, Xs[te], y[te], le.classes_)
    print(f"  tiny CNN test acc {acc:.4f}; per-class recall "
          + ", ".join(f"{k} {v:.2f}" for k, v in per_class.items()))

    payload = {
        "model_state_dict": model.state_dict(),
        "model_type": "cnn",
        "n_features": len(fs.FEATURE_COLS),
        "feature_cols": fs.FEATURE_COLS,
        "class_names": [str(c) for c in le.classes_],
        "mlp_hidden_sizes": (64, 32),
        "cnn_channels": cnn_channels,
        "seq_window_size": fs.SEQ_WINDOW_SIZE,
        "best_round": 1,
        "n_rounds": 1,
        "best_test_acc": float(acc),
        "final_test_acc": float(acc),
    }
    torch.save(payload, models_dir / "fixed4_cnn.pt")
    joblib.dump({"scaler": scaler, "label_encoder": le},
                models_dir / "fixed4_cnn_preprocess.pkl")


def main():
    p = argparse.ArgumentParser(description="Build a synthetic FL-IDS fixture project root.")
    p.add_argument("--root", default="/tmp/fl-demo-fixture")
    p.add_argument("--sample-rows", type=int, default=60_000)
    p.add_argument("--holdout-rows", type=int, default=40_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-epochs", type=int, default=4)
    p.add_argument("--skip-model", action="store_true")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.root).expanduser().resolve()
    models = root / "models"
    results = root / "results"
    hold_dir = root / "opensky_data" / "labeled_fixed"
    for d in (models, results, hold_dir):
        d.mkdir(parents=True, exist_ok=True)

    sample = build_sample(args.sample_rows, rng)
    meta = {"input_files": [{"path": "synthetic", "size_bytes": 0, "mtime": 0.0}],
            "max_total_rows": args.sample_rows, "max_rows_per_region_class": None,
            "load_chunksize": 200_000, "preserve_aircraft_groups": False,
            "rng_seed": args.seed, "fixture": True}
    pd.to_pickle({"meta": meta, "df": sample}, models / "sample_3m.pkl")
    print(f"Wrote {models / 'sample_3m.pkl'} ({len(sample):,} rows)")

    hold = build_holdout(args.holdout_rows, rng)
    hold_path = hold_dir / "holdout_feb04_1pct.csv"
    hold.to_csv(hold_path, index=False)
    benign_share = (hold["attack_type"] == "benign").mean()
    print(f"Wrote {hold_path} ({len(hold):,} rows, {benign_share:.1%} benign)")

    with open(results / "lgbm_feb04.json", "w") as f:
        json.dump({"model": "lightgbm (synthetic fixture)", "real_prior": REAL_PRIOR,
                   "holdout_rows": len(hold)}, f, indent=2)
    print(f"Wrote {results / 'lgbm_feb04.json'}")

    if not args.skip_model:
        train_tiny_model(sample, models, args.train_epochs, args.seed)
        print(f"Wrote {models / 'fixed4_cnn.pt'} + fixed4_cnn_preprocess.pkl")
    print(f"\nFixture ready at {root}")


if __name__ == "__main__":
    main()
