"""
inference_server.py
-------------------------------------------------------------------
Backend for the live attack-injection demo. Loads your trained model
(--save-model-path output from federated_simulation.py) and serves it
for real-time inference.

Two data sources, in order of preference:
  1. --holdout-data-path: real, labeled traffic from a day the model
     NEVER TRAINED ON (process a few hours through
     opensky_fl_ids_pipeline.py separately). Rows are sampled directly
     from this real, already-correctly-labeled data -- no synthetic
     generation, no approximated attack injection. This is strictly
     better than synthetic generation: real data has genuine feature
     correlations a per-feature-independent synthetic sampler can't
     reproduce, and using an untrained-on day makes this an honest
     generalization check, not just a demo of the model reproducing its
     own training distribution.
  2. Falls back to synthetic generation (sampling near the scaler's
     mean and inverse-transforming, plus hand-defined attack signatures)
     if no holdout file is given -- useful for a quick test without
     waiting for real data to be processed, but real held-out data is
     the better choice whenever it's available.

Usage:
    python3 inference_server.py --model-path best_model.pt \
        --holdout-data-path holdout_jan14_6h.csv

    Then open http://localhost:8000 in a browser.

Only supports row-independent models (mlp/cnn) -- a GRU model needs a
real chronological window, not a single row, and isn't supported by
this simple single-row UI.
-------------------------------------------------------------------
"""
import argparse
import random

import joblib
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, Response
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import federated_simulation as fs

app = FastAPI()

MODEL = None
SCALER = None
LABEL_ENCODER = None
FEATURE_COLS = None
CLASS_NAMES = None
HOLDOUT_BY_CLASS = None  # dict: class_name -> DataFrame of real rows, or None if unavailable

# Dedicated, genuinely-random generator for LIVE demo sampling -- deliberately
# separate from federated_simulation.py's np.random.seed(42), which is set at
# import time for TRAINING reproducibility. pandas' .sample() without an
# explicit random_state silently draws from that same pinned global numpy
# stream, which made every fresh server restart replay the EXACT SAME
# sequence of "random" draws (confirmed empirically: identical values in
# identical positions across separate process restarts -- not a coincidence,
# a fully deterministic replay). Using our own unseeded Generator here
# decouples live demo randomness from training's reproducibility seed.
_DEMO_RNG = np.random.default_rng()

# Fallback synthetic attack signatures, used ONLY if no --holdout-data-path
# is given. Mirrors the same per-attack feature effects verified
# empirically throughout the project (validate_labeled_dataset.py's
# EXPECTED_SIGNATURES).
ATTACK_INJECTIONS = {
    "gps_spoof": lambda row: {**row, "implied_speed_mps": random.uniform(8000, 20000)},
    "velocity_spoof": lambda row: {**row, "speed_consistency_delta": random.uniform(400, 1200)},
    "replay": lambda row: {**row, "inter_arrival_s": random.uniform(0.05, 1.5)},
    "flooding": lambda row: {**row, "inter_arrival_s": random.uniform(0.01, 0.3)},
    "ghost_aircraft": lambda row: {**row, "track_message_count": random.uniform(1, 2)},
}


def load_holdout_data(path, max_rows_per_class=5000):
    """
    Loads a labeled CSV (from opensky_fl_ids_pipeline.py, ideally from a
    day the model never trained on) and splits it into per-class row
    pools for direct real-data sampling. Capped per class purely to keep
    memory modest for a demo -- a live dashboard doesn't need more than
    a few thousand real examples per class to feel varied.

    Missing feature values are filled with -1, matching EXACTLY what
    federated_simulation.py's prep_features() does at TRAINING time --
    NOT dropped, which was found to be a real, serious bug: real
    flooding DUPLICATE rows have 100% NaN in implied_speed_mps,
    speed_consistency_delta, vrate_consistency_delta, and
    turn_rate_deg_s (the pipeline treats their near-zero time deltas as
    too unreliable to compute a rate from, by design). Dropping those
    rows silently deleted 85.7% of ALL real flooding data -- every
    single duplicate -- leaving the demo's flooding pool almost entirely
    composed of unrepresentative "original" rows. The model was actually
    TRAINED with -1 in these slots for exactly these rows, so filling
    with -1 here (not dropping) is what makes the demo consistent with
    what the model actually learned, and also happens to still fix the
    original JSON-serialization crash NaN caused (ghost_aircraft rows,
    which are missing these same features for a different, genuine
    reason -- a fabricated ID has no previous sighting to compute a
    delta against).
    """
    df = pd.read_csv(path, usecols=FEATURE_COLS + ["attack_type"])
    n_incomplete = df[FEATURE_COLS].isna().any(axis=1).sum()
    if n_incomplete > 0:
        print(f"  Filling {n_incomplete:,} holdout rows' missing feature values with -1 "
              f"({n_incomplete/len(df):.1%} of loaded data) -- matches training's own "
              f"prep_features() convention exactly, does not drop real attack data.")
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(-1)

    pools = {}
    for cname in list(CLASS_NAMES) + (["benign"] if "benign" not in CLASS_NAMES else []):
        subset = df[df["attack_type"] == cname]
        if len(subset) > max_rows_per_class:
            subset = subset.sample(n=max_rows_per_class, random_state=42)
        pools[cname] = subset.reset_index(drop=True)
        print(f"  Loaded {len(pools[cname]):,} real '{cname}' rows from holdout data")
    return pools


def sample_real_row(attack_type):
    """Pulls one real row (as a plain dict of NATIVE Python types -- not
    numpy scalars, which aren't JSON-serializable and crash FastAPI's
    encoder) from the holdout pool for the requested class -- 'none'
    maps to 'benign'."""
    cname = "benign" if attack_type == "none" else attack_type
    pool = HOLDOUT_BY_CLASS.get(cname)
    if pool is None or len(pool) == 0:
        return None
    row = pool.sample(n=1, random_state=_DEMO_RNG).iloc[0]
    result = {}
    for c in FEATURE_COLS:
        v = row[c]
        if isinstance(v, (np.bool_, bool)):
            result[c] = bool(v)
        elif isinstance(v, (np.integer, np.floating)):
            result[c] = v.item()
        else:
            result[c] = v
    return result


def generate_benign_row():
    """
    SYNTHETIC FALLBACK, used only when no --holdout-data-path is given.
    Generates a genuinely realistic 'benign' row by sampling NEAR ZERO in
    the model's own SCALED (z-score) space, then inverse-transforming
    back to real units via the ACTUAL fitted scaler -- rather than
    guessing raw-unit ranges by hand.

    Why this matters: a StandardScaler's mean (z=0) IS, by definition,
    the average of whatever the model actually trained on. Sampling near
    z=0 and inverse-transforming guarantees "benign" rows match the real
    training distribution's actual center and spread, regardless of
    whether hand-guessed raw-unit ranges happened to be close or not --
    this replaces an earlier version that used hand-guessed ranges and
    was found, empirically, to make the model see "benign" samples as
    statistical outliers (12/13 misclassified as attacks) because those
    guessed ranges didn't actually match the real fitted distribution.
    """
    z = _DEMO_RNG.normal(0, 0.4, size=len(FEATURE_COLS))  # modest spread around the real mean
    raw = SCALER.inverse_transform(z.reshape(1, -1))[0]
    row = dict(zip(FEATURE_COLS, raw))

    # Boolean-like columns don't have a meaningful "z-score" -- after
    # inverse-transform they're continuous approximations of 0/1, so
    # threshold them back to true booleans rather than leaving them as
    # arbitrary floats the model was never trained to see for these
    # specific columns.
    bool_cols = ["is_physically_implausible", "squawk_changed", "icao24_valid_format",
                 "is_stale", "is_low_coverage_region", "is_climbing_or_descending"]
    for c in bool_cols:
        row[c] = bool(row[c] > 0.5)
    return row


def predict(row: dict):
    """Runs a real forward pass through the loaded model, using the
    SAME scaler fit during training -- required for correct predictions,
    same as verified when the save/load mechanism was first tested."""
    X = pd.DataFrame([row])[FEATURE_COLS].copy()
    bool_cols = ["is_physically_implausible", "squawk_changed", "icao24_valid_format",
                 "is_stale", "is_low_coverage_region", "is_climbing_or_descending"]
    for c in bool_cols:
        X[c] = X[c].astype(float)
    if X.isna().any().any():
        # Second layer of defense against the exact crash a missing-feature
        # row causes (found via a real server crash on ghost_aircraft) --
        # the loader should already exclude these, but this guarantees
        # the crash can never recur even from a future data source that
        # isn't pre-filtered.
        raise ValueError(f"Row contains NaN in one or more features, cannot predict: "
                          f"{X.isna().sum().to_dict()}")
    X_scaled = SCALER.transform(X)
    with torch.no_grad():
        logits = MODEL(torch.tensor(X_scaled, dtype=torch.float32))
        probs = torch.softmax(logits, dim=1)[0].tolist()
    pred_idx = int(np.argmax(probs))
    return {
        "predicted_class": CLASS_NAMES[pred_idx],
        "is_attack": CLASS_NAMES[pred_idx] != "benign",
        "probabilities": {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(probs)},
    }


class InjectRequest(BaseModel):
    attack_type: str  # one of ATTACK_INJECTIONS keys, or "none" for a clean row


@app.get("/api/stream_sample")
def stream_sample(attack_type: str = "none", response: Response = Response()):
    """Returns one row -- REAL held-out data if --holdout-data-path was
    given, synthetic otherwise -- and the model's actual prediction on
    it. This is the endpoint the frontend polls to drive the live
    stream.

    Explicit no-cache headers are REQUIRED here: the frontend polls this
    exact URL (same query string) every 900ms, and without these
    headers a browser can silently cache the first response and keep
    serving it forever -- which looks EXACTLY like "the server is stuck
    returning the same value", even when the server itself is working
    correctly. This was a real, separate suspect alongside the RNG
    seeding bug when live values appeared frozen."""
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    row = None
    data_source = "synthetic"
    if HOLDOUT_BY_CLASS is not None:
        row = sample_real_row(attack_type)
        if row is not None:
            data_source = "real_holdout"
    if row is None:
        # Either no holdout data loaded, or this class had zero real
        # rows in the holdout file -- fall back to synthetic.
        row = generate_benign_row()
        if attack_type != "none" and attack_type in ATTACK_INJECTIONS:
            row = ATTACK_INJECTIONS[attack_type](row)
    result = predict(row)
    result["injected_attack_type"] = attack_type
    result["data_source"] = data_source
    result["row"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}
    return result


@app.get("/api/model_info")
def model_info():
    return {
        "class_names": list(CLASS_NAMES),
        "available_injections": list(ATTACK_INJECTIONS.keys()),
    }


@app.get("/")
def index():
    return HTMLResponse(open("index.html").read())


def load_model(model_path):
    global MODEL, SCALER, LABEL_ENCODER, FEATURE_COLS, CLASS_NAMES
    payload = torch.load(model_path, weights_only=False)
    if payload["model_type"] == "gru":
        raise ValueError(
            "This demo server only supports row-independent models (mlp/cnn) -- "
            "a GRU model needs a real chronological window of several messages per "
            "prediction, not a single injected row, and would need a different UI "
            "(a mini rolling history per client) to demo meaningfully."
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
    global SCALER
    SCALER = preprocess["scaler"]
    print(f"Loaded {payload['model_type'].upper()} model, best_round={payload['best_round']}, "
          f"best_test_acc={payload['best_test_acc']:.4f}")


if __name__ == "__main__":
    import uvicorn
    p = argparse.ArgumentParser()
    p.add_argument("--model-path", required=True)
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--holdout-data-path", type=str, default=None,
                    help="Labeled CSV from opensky_fl_ids_pipeline.py, ideally processed from "
                         "a day the model never trained on. Rows are sampled directly from this "
                         "REAL data per class instead of synthetic generation -- strictly "
                         "preferred whenever available, since real data preserves genuine "
                         "feature correlations a synthetic per-feature sampler cannot, and using "
                         "an untrained-on day makes this an honest generalization check.")
    p.add_argument("--holdout-max-per-class", type=int, default=5000,
                    help="Cap on real rows loaded per class from the holdout file (default 5000) "
                         "-- kept modest since a live demo doesn't need more for variety.")
    args = p.parse_args()
    load_model(args.model_path)
    if args.holdout_data_path:
        print(f"\nLoading real holdout data from {args.holdout_data_path} ...")
        HOLDOUT_BY_CLASS = load_holdout_data(args.holdout_data_path, args.holdout_max_per_class)
    else:
        print("\nNo --holdout-data-path given -- using synthetic data generation (fallback).")
        HOLDOUT_BY_CLASS = None
    uvicorn.run(app, host="0.0.0.0", port=args.port)
