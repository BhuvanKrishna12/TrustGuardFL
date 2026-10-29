"""
run_experiments.py
-------------------------------------------------------------------
Experiment harness for the journal revision of the FL-IDS paper.

For every (model, feature set, seed) it trains one model, selects its
checkpoint on a validation split carved from TRAINING data only, scores the
whole held-out test file ONCE, saves the class probabilities, and computes
metrics under class-prior correction for every strength s:

    score(k) = P_model(k) * pi(k) ** s,   prediction = argmax_k score(k)

pi is the natural class mix of ALL rows of the training files (one chunked
stats pass, which also records raw row counts and the set of training aircraft).
The holdout is never used for any selection.

Training sample: uniform per-(region_cell, attack_type) bottom-k sampler
(sampling.py), same cap formula and columns as federated_simulation.load_data;
--legacy-sampler uses load_data itself (biased towards late chunks and files).

Extra evaluations per run (all from the saved probabilities, so they also work
with --holdout-only): per test day; the subset of test rows whose icao24 never
appears in the training files ("unseen aircraft"); and test-row reweighting to
an assumed deployment attack prevalence rho (0.1% and 1% by default), with the
correction using the training prior (mismatched) or the true deployment prior
(oracle).

Models
  fed_cnn      FedAvg exactly as federated_simulation.main (region_cell
               clients, 80/20 per-client train/validation split, pooled
               scaler and balanced class weights, full participation,
               fresh Adam per client per round, best round by pooled
               validation accuracy). Uses federated_simulation.local_train
               and federated_average directly.
  central_cnn  Same architecture, scaler, class weights, optimizer, batch
               size; trained on the pooled union of the clients' training
               splits, early stopping on the pooled validation splits.
  lgbm         LightGBM multiclass, params of lgbm_baseline.py, balanced
               sample weights, early stopping on the pooled validation rows.

Layout of --out-dir
  cache/        training-file stats (json + icao24 list), training sample
                (pickle/parquet + json), holdout arrays (npz). Built once, reused.
  runs/         one JSON per run: <model>__<featureset>__seed<k>.json
  probs/        holdout class probabilities per run (npz, float16)
  models/       saved checkpoints (.pt + scaler .pkl, LightGBM .txt)
  holdout_labels.npz   holdout labels, day of each row, class names
  run_experiments.log  progress log (tail -f friendly)

See experiments/README.md for the commands used on the real data.
"""

import argparse
import contextlib
import copy
import gc
import gzip
import hashlib
import io
import json
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

import federated_simulation as fs  # noqa: E402
import sampling  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

FEATURE_SETS = {
    "full": list(fs.FEATURE_COLS),
    "core": ["implied_speed_mps", "speed_consistency_delta", "vrate_consistency_delta",
             "inter_arrival_s", "turn_rate_deg_s", "region_avg_inter_arrival", "is_stale"],
}
MODELS = ("fed_cnn", "central_cnn", "lgbm")
RATE_FEATURES = ["implied_speed_mps", "speed_consistency_delta",
                 "vrate_consistency_delta", "turn_rate_deg_s"]
SAMPLE_SEED = 42           # the training sample is drawn once with this seed
TRAIN_FRACTION = 0.8       # per-client train / validation split, as in build_clients
MIN_DAY_SHARE = 0.05       # a holdout date is a "test day" if it holds >= 5% of rows
DEFAULT_STRENGTHS = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
DEFAULT_RHOS = (0.001, 0.01)  # assumed deployment attack prevalences (test-row reweighting)

LOG_FH = None


def log(msg=""):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if LOG_FH is not None:
        LOG_FH.write(line + "\n")
        LOG_FH.flush()


class _Tee(io.TextIOBase):
    """Copies writes to the real stdout and to a buffer (to capture what
    federated_simulation.load_data prints, e.g. the raw row count)."""

    def __init__(self, real):
        self.real, self.buf = real, io.StringIO()

    def write(self, s):
        self.real.write(s)
        self.real.flush()
        self.buf.write(s)
        if LOG_FH is not None:
            LOG_FH.write(s)
            LOG_FH.flush()
        return len(s)

    def flush(self):
        self.real.flush()


def file_key(paths, *extra):
    """Cache key from file basenames + sizes (not absolute paths, so a cache
    copied to another machine with the same files is still recognised)."""
    parts = [(os.path.basename(p), os.path.getsize(p)) for p in paths]
    blob = json.dumps([parts, list(extra)], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def run_id(model, fset, seed):
    return f"{model}__{fset}__seed{seed}"


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


# ---------------------------------------------------------------------------
# data: training-file statistics, training sample, holdout
# ---------------------------------------------------------------------------

def load_train_stats(paths, cache_dir):
    """Raw row counts per file and class (natural prior), regions, and the set
    of training icao24 addresses. One chunked pass, cached."""
    key = file_key(paths, "stats-v1")
    jpath = os.path.join(cache_dir, f"train_stats_{key}.json")
    ipath = os.path.join(cache_dir, f"train_icao24_{key}.txt.gz")
    if os.path.exists(jpath) and os.path.exists(ipath):
        log(f"Training-file stats: loading cache {jpath}")
        stats = json.load(open(jpath))
        with gzip.open(ipath, "rt") as f:
            icao = set(f.read().split())
        return stats, icao, jpath
    log("Training-file stats: one chunked pass over region_cell, attack_type, icao24 of every "
        "training file (raw rows, natural prior, training aircraft)")
    t0 = time.time()
    stats, icao = sampling.train_file_stats(paths, log=log)
    stats["files"] = [os.path.abspath(p) for p in paths]
    stats["minutes"] = (time.time() - t0) / 60
    with gzip.open(ipath + ".tmp", "wt") as f:
        f.write("\n".join(sorted(icao)))
    os.replace(ipath + ".tmp", ipath)
    atomic_json(stats, jpath)
    log(f"  {stats['total_rows']:,} raw rows, {stats['n_regions']:,} regions, "
        f"{stats['n_aircraft']:,} distinct icao24 ({stats['minutes']:.1f} min)")
    return stats, icao, jpath


def load_training_sample(paths, max_total_rows, cache_dir, sampler, stats):
    """Training sample, drawn once with seed 42 and cached.
    sampler='uniform' (default): sampling.uniform_sample, uniform within each
    (region_cell, attack_type) bucket. sampler='legacy': federated_simulation.load_data."""
    key = file_key(paths, max_total_rows, SAMPLE_SEED, sampler)
    meta_path = os.path.join(cache_dir, f"train_sample_{sampler}_{key}.json")
    try:
        import pyarrow  # noqa: F401
        data_path = os.path.join(cache_dir, f"train_sample_{sampler}_{key}.parquet")
    except ImportError:
        data_path = os.path.join(cache_dir, f"train_sample_{sampler}_{key}.pkl")

    if os.path.exists(data_path) and os.path.exists(meta_path):
        log(f"Training sample ({sampler}): loading cache {data_path}")
        df = pd.read_parquet(data_path) if data_path.endswith(".parquet") else pd.read_pickle(data_path)
        meta = json.load(open(meta_path))
        log(f"  {len(df):,} rows")
        return df, meta

    t0 = time.time()
    classes_all = sorted(stats["class_counts"])
    if sampler == "uniform":
        log(f"Training sample: uniform per-(region, class) sampler over {len(paths)} file(s) "
            f"(max_total_rows={max_total_rows:,}, seed={SAMPLE_SEED}); slow one-off step")
        df, sinfo = sampling.uniform_sample(paths, max_total_rows, stats["regions"], classes_all,
                                            seed=SAMPLE_SEED, log=log)
        hit_cap = False
    else:
        log(f"Training sample: LEGACY federated_simulation.load_data over {len(paths)} file(s) "
            f"(max_total_rows={max_total_rows:,}, seed={SAMPLE_SEED}); slow one-off step")
        tee = _Tee(sys.stdout)
        with contextlib.redirect_stdout(tee):
            df = fs.load_data(paths, max_total_rows=max_total_rows, seed=SAMPLE_SEED)
        m = re.search(r"Streamed ([\d,]+) raw rows", tee.buf.getvalue())
        m2 = re.search(r"cap to ([\d,]+) rows", tee.buf.getvalue())
        hit_cap = "Reached global cap" in tee.buf.getvalue()
        sinfo = {"sampler": "legacy_load_data",
                 "raw_rows_streamed": int(m.group(1).replace(",", "")) if m else None,
                 "cap_per_region_class": int(m2.group(1).replace(",", "")) if m2 else None,
                 "seed": SAMPLE_SEED}

    classes = sorted(df["attack_type"].astype(str).unique())
    comp = sampling.sample_composition(df, paths, classes)
    n_icao = int(df["icao24"].nunique()) if "icao24" in df else None
    keep = [c for c in fs.FEATURE_COLS + ["region_cell", "attack_type"] if c in df.columns]
    df = df[keep].copy()
    df["attack_type"] = df["attack_type"].astype(str).astype("category")
    df["region_cell"] = df["region_cell"].astype(str).astype("category")
    df = df.reset_index(drop=True)
    gc.collect()

    meta = dict(sinfo)
    meta.update({
        "files": [os.path.abspath(p) for p in paths],
        "file_sizes": [os.path.getsize(p) for p in paths],
        "max_total_rows": max_total_rows,
        "rows": int(len(df)),
        "train_raw_rows_total": stats["total_rows"],
        "hit_global_cap_early": hit_cap,
        "n_regions_in_files": stats["n_regions"],
        "n_aircraft_in_sample": n_icao,
        "composition": comp,
        "load_minutes": (time.time() - t0) / 60,
        "data_path": data_path,
    })
    if data_path.endswith(".parquet"):
        df.to_parquet(data_path)
    else:
        df.to_pickle(data_path)
    atomic_json(meta, meta_path)
    log(f"  cached {len(df):,} rows -> {data_path} ({meta['load_minutes']:.1f} min)")
    for d, v in comp["per_day"].items():
        log(f"    sample rows from {d}: {v['rows']:,}")
    return df, meta


def natural_prior(stats, classes, prior_json=None):
    """Natural class mix of ALL rows of the training files (from the stats pass).
    Never computed from the holdout."""
    if prior_json:
        d = json.load(open(prior_json))
        d = d.get("real_prior", d.get("prior", d))
        prior = np.array([float(d[c]) for c in classes])
        log(f"Natural prior: taken from {prior_json}")
        return prior / prior.sum(), os.path.abspath(prior_json)
    counts = stats["class_counts"]
    tot = sum(counts.get(c, 0) for c in classes)
    prior = np.array([counts.get(c, 0) / tot for c in classes])
    if (prior == 0).any():
        log(f"  WARNING: zero natural prior for {list(np.array(classes)[prior == 0])}")
    log("Natural prior pi = " + ", ".join(f"{c}={p:.4f}" for c, p in zip(classes, prior)))
    return prior, "training-file stats pass"


def load_holdout(path, classes, cache_dir, train_icao, stats_key, chunksize=500_000):
    """Holdout arrays: prepped 13 features (float32), labels, UTC day from
    lastposupdate (fallback time), and whether the row's icao24 never occurs in
    the training files. Also rule-baseline counts and blank-feature shares.
    Cached as npz."""
    st = os.stat(path)
    key = hashlib.sha1(json.dumps([os.path.basename(path), st.st_size, int(st.st_mtime),
                                   list(classes), stats_key, "holdout-v2"]).encode()).hexdigest()[:12]
    cpath = os.path.join(cache_dir, f"holdout_{key}.npz")
    if os.path.exists(cpath):
        log(f"Holdout: loading cache {cpath}")
        z = np.load(cpath, allow_pickle=False)
        info = json.loads(str(z["info"]))
        return z["X"], z["y"], z["day"], z["unseen"], info

    log(f"Holdout: reading {path}")
    header = pd.read_csv(path, nrows=0).columns
    want = fs.FEATURE_COLS + ["attack_type", "time", "lastposupdate", "icao24"]
    usecols = [c for c in want if c in header]
    missing = set(fs.FEATURE_COLS + ["attack_type"]) - set(usecols)
    if missing:
        raise ValueError(f"holdout is missing columns {missing}")
    has_icao = "icao24" in usecols
    cls_index = {c: i for i, c in enumerate(classes)}
    Xs, ys, days, blanks, floods, unseens, icaos = [], [], [], [], [], [], []
    dropped = 0
    for ch in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        at = ch["attack_type"].astype(str)
        ok = at.isin(cls_index).values
        dropped += int((~ok).sum())
        ch = ch[ok]
        at = at[ok]
        raw = ch[RATE_FEATURES].apply(pd.to_numeric, errors="coerce")
        blanks.append((raw.isna() | (raw == -1)).all(axis=1).values)
        ia = pd.to_numeric(ch["inter_arrival_s"], errors="coerce")
        floods.append((ia < 1).fillna(False).values.astype(bool))
        if has_icao:
            ic = ch["icao24"].astype(str)
            unseens.append(~ic.isin(train_icao).values)
            icaos.append(ic.unique())
        else:
            unseens.append(np.zeros(len(ch), bool))
        days.append(sampling.day_of(ch))
        Xs.append(fs.prep_features(ch).astype(np.float32).values)
        ys.append(at.map(cls_index).values.astype(np.int8))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    day = np.concatenate(days)
    blank = np.concatenate(blanks)
    flood = np.concatenate(floods)
    unseen = np.concatenate(unseens)
    K = len(classes)

    def fires(mask):
        return {c: int((mask & (y == i)).sum()) for i, c in enumerate(classes)}

    support = {c: int((y == i).sum()) for i, c in enumerate(classes)}
    info = {"path": os.path.abspath(path), "rows": int(len(y)),
            "rows_dropped_unknown_class": dropped,
            "day_source": "lastposupdate (fallback time), UTC date",
            "support": support,
            "blank_rate_features": RATE_FEATURES,
            "blank_share_by_class": {c: (fires(blank)[c] / support[c] if support[c] else None)
                                     for c in classes},
            "rules": {
                "ghost_aircraft": {"rule": "all four rate features missing (NaN or -1)",
                                   "fires_by_class": fires(blank)},
                "flooding": {"rule": "inter_arrival_s < 1 s (missing values do not fire)",
                             "fires_by_class": fires(flood)},
            }}
    if has_icao:
        test_icao = set(np.concatenate(icaos).tolist()) if icaos else set()
        info["overlap"] = {
            "has_icao24": True,
            "test_aircraft": len(test_icao),
            "test_aircraft_seen_in_training": len(test_icao & train_icao),
            "share_aircraft_seen": len(test_icao & train_icao) / max(1, len(test_icao)),
            "share_rows_seen": float(1 - unseen.mean()) if len(unseen) else None,
            "unseen_rows": int(unseen.sum()),
            "unseen_support": {c: int((unseen & (y == i)).sum()) for i, c in enumerate(classes)},
            "train_aircraft": len(train_icao),
        }
    else:
        info["overlap"] = {"has_icao24": False}
    np.savez(cpath, X=X, y=y, day=np.asarray(day, dtype="U10"), unseen=unseen,
             info=np.array(json.dumps(info)))
    log(f"  {len(y):,} rows ({dropped:,} dropped with unknown class), "
        f"{int(unseen.sum()):,} rows from aircraft never seen in training -> cached {cpath}")
    _ = K
    return X, y, day, unseen, info


def holdout_days(day):
    vals, counts = np.unique(day, return_counts=True)
    all_counts = {str(v): int(c) for v, c in zip(vals, counts)}
    main = sorted(str(v) for v, c in zip(vals, counts) if c >= MIN_DAY_SHARE * len(day))
    return main, all_counts



# ---------------------------------------------------------------------------
# features, clients
# ---------------------------------------------------------------------------

def prep_subset(df, cols):
    """federated_simulation.prep_features (bool cast, fillna(-1)) restricted to cols.
    prep_features is column-wise, so subsetting afterwards is identical."""
    return fs.prep_features(df)[cols]


def make_clients(regions, min_rows, seed):
    """Index-level copy of federated_simulation.build_clients (non-sequence path):
    groups rows by region_cell, drops regions below min_rows, then per region
    rng = default_rng(seed); permutation of the region's row indices; first 80%
    train, rest validation. With seed=42 this reproduces build_clients exactly
    (checked in experiments/check_equivalence.py)."""
    groups = pd.DataFrame({"region_cell": regions}).groupby("region_cell", observed=True).groups
    clients, dropped = [], []
    for region, idx in groups.items():
        idx = np.array(idx)
        if len(idx) < min_rows:
            dropped.append((str(region), int(len(idx))))
            continue
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(idx)
        split = int(len(shuffled) * TRAIN_FRACTION)
        clients.append({"region": str(region), "train_idx": shuffled[:split],
                        "val_idx": shuffled[split:]})
    return clients, dropped


def client_summary(clients, y, classes):
    out = []
    K = len(classes)
    for c in clients:
        lab = np.bincount(y[np.concatenate([c["train_idx"], c["val_idx"]])], minlength=K)
        out.append({"region": c["region"], "n_train": int(len(c["train_idx"])),
                    "n_val": int(len(c["val_idx"])),
                    "label_counts": {classes[k]: int(lab[k]) for k in range(K)}})
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def confusion(y, pred, K):
    return np.bincount(y.astype(np.int64) * K + pred.astype(np.int64),
                       minlength=K * K).reshape(K, K)


def metrics_from_cm(cm, classes):
    """All metrics from a (possibly row-weighted) confusion matrix
    (rows = actual, columns = predicted)."""
    cm = np.asarray(cm, float)
    K = len(classes)
    tp = np.diag(cm)
    sup = cm.sum(1)
    predn = cm.sum(0)
    with np.errstate(divide="ignore", invalid="ignore"):
        rec = np.where(sup > 0, tp / np.where(sup > 0, sup, 1), 0.0)
        prec = np.where(predn > 0, tp / np.where(predn > 0, predn, 1), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / np.where(prec + rec > 0, prec + rec, 1), 0.0)
    b = list(classes).index("benign")
    atk = [k for k in range(K) if k != b]
    n_b = cm[b].sum()
    n_a = cm[atk].sum()
    fa = n_b - cm[b, b]
    missed = cm[atk, b].sum()
    alerts = cm[:, atk].sum()
    real_alerts = cm[np.ix_(atk, atk)].sum()
    is_int = np.allclose(cm, np.round(cm))
    conv = (lambda v: int(round(v))) if is_int else float
    return {
        "n": conv(cm.sum()),
        "accuracy": float(tp.sum() / max(cm.sum(), 1e-12)),
        "macro_f1": float(f1.mean()),
        "false_alarm_rate": float(fa / max(n_b, 1e-12)),
        "missed_attack_rate": float(missed / max(n_a, 1e-12)),
        "avg_attack_recall": float(rec[atk].mean()),
        "alert_real_share": float(real_alerts / alerts) if alerts > 0 else None,  # no alerts
        "false_alarms": conv(fa),
        "missed_attacks": conv(missed),
        "per_class": {classes[k]: {"precision": float(prec[k]), "recall": float(rec[k]),
                                   "f1": float(f1[k]), "support": conv(sup[k])}
                      for k in range(K)},
        "confusion_matrix": [[conv(v) for v in row] for row in cm],
    }


def metrics_from_pred(y, pred, classes):
    return metrics_from_cm(confusion(y, pred, len(classes)), classes)


def deployment_prior(y, classes, rho):
    """Benign share 1 - rho; the five attack classes keep their test proportions
    and share rho between them."""
    K = len(classes)
    b = list(classes).index("benign")
    n = np.bincount(y.astype(np.int64), minlength=K).astype(float)
    atk = [k for k in range(K) if k != b]
    target = np.zeros(K)
    target[b] = 1 - rho
    if n[atk].sum() > 0:
        target[atk] = rho * n[atk] / n[atk].sum()
    return target


def row_weights(y, classes, target):
    """Importance weight per TRUE class so that the weighted test mix equals target.
    Weights depend only on the true class, so weighting rows equals scaling the
    rows of the confusion matrix."""
    K = len(classes)
    n = np.bincount(y.astype(np.int64), minlength=K).astype(float)
    share = n / max(n.sum(), 1)
    return np.where(share > 0, target / np.where(share > 0, share, 1), 0.0)


def evaluate_strengths(proba16, y, day, main_days, prior, strengths, classes,
                       unseen=None, rhos=DEFAULT_RHOS):
    """proba16: float16 probabilities exactly as saved, so offline recomputation
    from the npz gives identical numbers.

    Per strength s: pooled metrics, per test day, unseen-aircraft subset, and
    deployment-prevalence reweighting for each rho (total attack share):
      mismatched: correction with the training prior pi (what a deployment would do)
      oracle:     correction with the true deployment prior
    Reweighting changes only the class mix of the test rows (no resampling, no
    retraining), so recall and FAR are unchanged; precision, F1, accuracy and the
    share of alerts that are real attacks change."""
    p = proba16.astype(np.float32)
    K = len(classes)
    out = {}
    day_masks = {d: day == d for d in main_days}
    has_unseen = unseen is not None and unseen.any()
    dep = {rho: deployment_prior(y, classes, rho) for rho in rhos}
    wts = {rho: row_weights(y, classes, dep[rho]) for rho in rhos}
    for s in strengths:
        pred = (p * (prior.astype(np.float32) ** np.float32(s))[None, :]).argmax(1)
        cm = confusion(y, pred, K)
        entry = {"pooled": metrics_from_cm(cm, classes), "days": {}}
        for d, m in day_masks.items():
            entry["days"][d] = metrics_from_pred(y[m], pred[m], classes)
        if has_unseen:
            entry["unseen"] = metrics_from_pred(y[unseen], pred[unseen], classes)
        entry["prevalence"] = {}
        for rho in rhos:
            w = wts[rho]
            pred_o = (p * (dep[rho].astype(np.float32) ** np.float32(s))[None, :]).argmax(1)
            cm_o = confusion(y, pred_o, K)
            entry["prevalence"][f"{rho:g}"] = {
                "deployment_prior": {c: float(v) for c, v in zip(classes, dep[rho])},
                "mismatched": metrics_from_cm(cm * w[:, None], classes),
                "oracle": metrics_from_cm(cm_o * w[:, None], classes),
            }
        out[f"{s:.2f}"] = entry
    return out



# ---------------------------------------------------------------------------
# torch helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_logits(model, X, batch=65536):
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    for lo in range(0, len(X), batch):
        xb = torch.as_tensor(X[lo:lo + batch], dtype=torch.float32, device=dev)
        outs.append(model(xb).float().cpu())
    return torch.cat(outs).numpy() if outs else np.zeros((0, 0), np.float32)


def accuracy(model, X, y, batch=65536):
    return float((predict_logits(model, X, batch).argmax(1) == y).mean())


def softmax_np(logits):
    t = torch.from_numpy(logits)
    return torch.softmax(t, dim=1).numpy()


def seed_everything(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# trainers
# ---------------------------------------------------------------------------

def train_fed_cnn(X, y, clients, class_weights, n_classes, args, seed, tag):
    """The FedAvg loop of federated_simulation.main, driven with a seed."""
    n_features = X.shape[1]
    cw = torch.tensor(class_weights, dtype=torch.float32).to(fs.DEVICE)
    cdata = {}
    for c in clients:
        cdata[c["region"]] = dict(X_train=X[c["train_idx"]], y_train=y[c["train_idx"]])
    Xv = np.concatenate([X[c["val_idx"]] for c in clients])
    yv = np.concatenate([y[c["val_idx"]] for c in clients])

    seed_everything(seed)
    model = fs.build_model("cnn", n_features, n_classes, cnn_channels=args.cnn_channels)
    payload = fs.model_size_bytes(model)
    n_params = sum(p.numel() for p in model.parameters())
    names = list(cdata.keys())
    rng = np.random.default_rng(seed)
    best_acc, best_round, best_state = -1.0, 0, None
    hist = {"round": [], "train_acc": [], "val_acc": [], "seconds": []}
    total_bytes = 0
    t0 = time.time()
    log(f"  [{tag}] FedAvg: {args.rounds} rounds x {args.local_epochs} local epochs, "
        f"{len(names)} clients, model {payload:,} bytes ({n_params:,} params)")
    for rnd in range(1, args.rounds + 1):
        tr = time.time()
        n_part = max(1, int(len(names) * fs.CLIENT_FRACTION))
        participants = rng.choice(names, size=n_part, replace=False)
        states, weights, taccs = [], [], []
        for cname in participants:
            local = copy.deepcopy(model).to(fs.DEVICE)
            c = cdata[cname]
            st, ta = fs.local_train(local, c["X_train"], c["y_train"], args.local_epochs, cw)
            states.append(st)
            weights.append(len(c["X_train"]))
            taccs.append(ta)
            total_bytes += 2 * payload
        model.load_state_dict(fs.federated_average(states, weights))
        tacc = sum(a * w for a, w in zip(taccs, weights)) / sum(weights)
        vacc = accuracy(model, Xv, yv)
        if vacc > best_acc:
            best_acc, best_round = vacc, rnd
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        hist["round"].append(rnd)
        hist["train_acc"].append(float(tacc))
        hist["val_acc"].append(vacc)
        hist["seconds"].append(time.time() - tr)
        log(f"  [{tag}] round {rnd:3d}/{args.rounds}  train acc {tacc:.4f}  val acc {vacc:.4f}  "
            f"({time.time() - tr:.0f}s, elapsed {(time.time() - t0) / 60:.1f} min)")
    model.load_state_dict(best_state)
    steps = args.rounds * args.local_epochs * sum(
        int(np.ceil(len(c["X_train"]) / fs.BATCH_SIZE)) for c in cdata.values())
    info = {
        "rounds": args.rounds, "local_epochs": args.local_epochs,
        "client_fraction": fs.CLIENT_FRACTION, "batch_size": fs.BATCH_SIZE, "lr": fs.LR,
        "cnn_channels": list(args.cnn_channels), "n_params": int(n_params),
        "per_round_val_acc": hist["val_acc"], "per_round_train_acc": hist["train_acc"],
        "per_round_seconds": hist["seconds"],
        "best_round": best_round, "best_val_acc": best_acc,
        "final_val_acc": hist["val_acc"][-1],
        "model_bytes": int(payload),
        "comm_bytes": int(2 * args.rounds * len(names) * payload),
        "comm_bytes_counted": int(total_bytes),
        "n_clients": len(names),
        "gradient_steps": int(steps),
        "train_seconds": time.time() - t0,
    }
    return model, info


def train_central_cnn(X, y, clients, class_weights, n_classes, args, seed, tag):
    n_features = X.shape[1]
    tr_idx = np.concatenate([c["train_idx"] for c in clients])
    va_idx = np.concatenate([c["val_idx"] for c in clients])
    Xv, yv = X[va_idx], y[va_idx]
    dev = fs.DEVICE
    Xt = torch.as_tensor(X[tr_idx], dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y[tr_idx], dtype=torch.long, device=dev)
    N = len(tr_idx)

    seed_everything(seed)
    model = fs.build_model("cnn", n_features, n_classes, cnn_channels=args.cnn_channels)
    opt = torch.optim.Adam(model.parameters(), lr=fs.LR)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(dev))
    gen = torch.Generator().manual_seed(seed)
    bs = fs.BATCH_SIZE
    best_acc, best_epoch, best_state, bad = -1.0, 0, None, 0
    hist = {"val_acc": [], "train_loss": [], "seconds": []}
    t0 = time.time()
    log(f"  [{tag}] central CNN: {N:,} train rows, {len(va_idx):,} val rows, "
        f"max {args.central_max_epochs} epochs, patience {args.central_patience}")
    epochs_run = 0
    for ep in range(1, args.central_max_epochs + 1):
        te = time.time()
        model.train()
        perm = torch.randperm(N, generator=gen).to(dev)
        tot, nb = 0.0, 0
        for lo in range(0, N, bs):
            bi = perm[lo:lo + bs]
            opt.zero_grad()
            loss = crit(model(Xt[bi]), yt[bi])
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        vacc = accuracy(model, Xv, yv)
        epochs_run = ep
        hist["val_acc"].append(vacc)
        hist["train_loss"].append(tot / max(1, nb))
        hist["seconds"].append(time.time() - te)
        if vacc > best_acc:
            best_acc, best_epoch, bad = vacc, ep, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"  [{tag}] epoch {ep:3d}/{args.central_max_epochs}  loss {tot / max(1, nb):.4f}  "
            f"val acc {vacc:.4f}  ({time.time() - te:.0f}s)")
        if bad >= args.central_patience:
            log(f"  [{tag}] early stop: no val improvement for {bad} epochs")
            break
    model.load_state_dict(best_state)
    del Xt, yt
    steps_per_epoch = int(np.ceil(N / bs))
    fed_steps = args.rounds * args.local_epochs * sum(
        int(np.ceil(len(c["train_idx"]) / bs)) for c in clients)
    info = {
        "max_epochs": args.central_max_epochs, "patience": args.central_patience,
        "epochs_run": epochs_run, "best_epoch": best_epoch, "best_val_acc": best_acc,
        "per_epoch_val_acc": hist["val_acc"], "per_epoch_train_loss": hist["train_loss"],
        "per_epoch_seconds": hist["seconds"],
        "batch_size": bs, "lr": fs.LR, "cnn_channels": list(args.cnn_channels),
        "model_bytes": int(fs.model_size_bytes(model)),
        "gradient_steps": int(epochs_run * steps_per_epoch),
        "epoch_budget": {
            "central_max_epochs": args.central_max_epochs,
            "central_epochs_run": epochs_run,
            "central_best_epoch": best_epoch,
            "fedavg_rounds": args.rounds,
            "fedavg_local_epochs": args.local_epochs,
            "fedavg_passes_over_training_rows": args.rounds * args.local_epochs,
            "fedavg_gradient_steps": int(fed_steps),
            "central_gradient_steps": int(epochs_run * steps_per_epoch),
            "note": (f"FedAvg makes {args.rounds} rounds x {args.local_epochs} local epochs = "
                     f"{args.rounds * args.local_epochs} passes over every client's training rows "
                     f"(models averaged after every {args.local_epochs} local epochs). The centralized CNN "
                     f"sees the same rows for at most {args.central_max_epochs} epochs with early "
                     f"stopping (patience {args.central_patience}) on pooled validation accuracy; "
                     f"it ran {epochs_run}. Both keep the checkpoint with the best validation "
                     f"accuracy."),
        },
        "train_seconds": time.time() - t0,
    }
    return model, info


def train_lgbm(Xraw, y, clients, class_weights, n_classes, args, seed, tag):
    import warnings
    import lightgbm as lgb
    warnings.filterwarnings("ignore", message=".*eval_set.*deprecated.*")
    tr_idx = np.concatenate([c["train_idx"] for c in clients])
    va_idx = np.concatenate([c["val_idx"] for c in clients])
    w = np.asarray(class_weights)[y]
    log(f"  [{tag}] LightGBM: {len(tr_idx):,} train rows, {len(va_idx):,} val rows, "
        f"leaves={args.lgbm_num_leaves}, lr={args.lgbm_learning_rate}, "
        f"max trees={args.lgbm_n_estimators}, threads={args.lgbm_threads}")
    t0 = time.time()
    model = lgb.LGBMClassifier(
        objective="multiclass", n_estimators=args.lgbm_n_estimators,
        learning_rate=args.lgbm_learning_rate, num_leaves=args.lgbm_num_leaves,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
        n_jobs=args.lgbm_threads, random_state=seed, verbose=-1)
    model.fit(Xraw[tr_idx], y[tr_idx], sample_weight=w[tr_idx],
              eval_set=[(Xraw[va_idx], y[va_idx])], eval_sample_weight=[w[va_idx]],
              callbacks=[lgb.early_stopping(args.lgbm_early_stopping, verbose=False),
                         lgb.log_evaluation(50)])
    best_it = int(model.best_iteration_ or model.n_estimators)
    val_acc = float((model.predict(Xraw[va_idx]) == y[va_idx]).mean())
    gain = model.booster_.feature_importance(importance_type="gain")
    info = {
        "n_estimators": args.lgbm_n_estimators, "learning_rate": args.lgbm_learning_rate,
        "num_leaves": args.lgbm_num_leaves, "subsample": 0.8, "subsample_freq": 1,
        "colsample_bytree": 0.9, "early_stopping_rounds": args.lgbm_early_stopping,
        "early_stopping_metric": "multi_logloss (balanced sample weights) on pooled validation",
        "threads": args.lgbm_threads, "random_state": seed,
        "best_iteration": best_it, "iterations_run": int(model.booster_.current_iteration()),
        "gradient_steps": int(model.booster_.current_iteration()),
        "gradient_steps_note": "boosting iterations run (each fits one tree per class)",
        "best_val_acc": val_acc,
        "feature_importance_gain_share": None,  # filled by caller (needs names)
        "_gain": gain.tolist(),
        "train_seconds": time.time() - t0,
    }
    log(f"  [{tag}] best iteration {best_it}, val acc {val_acc:.4f} "
        f"({(time.time() - t0) / 60:.1f} min)")
    return model, info


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-files", nargs="+", default=None, help="Labelled training CSVs")
    ap.add_argument("--holdout", default=None, help="Labelled held-out test CSV")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-total-rows", type=int, default=10_000_000)
    ap.add_argument("--legacy-sampler", action="store_true",
                    help="Draw the training sample with federated_simulation.load_data (biased "
                         "towards the last chunks of the last file) instead of the uniform sampler")
    ap.add_argument("--seeds", default=None, help="default 1,2,3,4,5 (1,2 with --quick)")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--feature-sets", default="full,core")
    ap.add_argument("--strengths", default=None,
                    help=f"default {DEFAULT_STRENGTHS} (0,0.5,1 with --quick)")
    ap.add_argument("--rhos", default=",".join(f"{r:g}" for r in DEFAULT_RHOS),
                    help="Assumed deployment attack prevalences for test-row reweighting")
    ap.add_argument("--min-client-rows", type=int, default=3000)
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0 ...")
    ap.add_argument("--rounds", type=int, default=50, help="FedAvg rounds")
    ap.add_argument("--local-epochs", type=int, default=2)
    ap.add_argument("--cnn-channels", default="32,64")
    ap.add_argument("--central-max-epochs", type=int, default=20)
    ap.add_argument("--central-patience", type=int, default=3)
    ap.add_argument("--lgbm-n-estimators", type=int, default=1000)
    ap.add_argument("--lgbm-learning-rate", type=float, default=0.1)
    ap.add_argument("--lgbm-num-leaves", type=int, default=127)
    ap.add_argument("--lgbm-early-stopping", type=int, default=30)
    ap.add_argument("--lgbm-threads", type=int, default=4)
    ap.add_argument("--prior-json", default=None,
                    help="Optional JSON with the natural training prior ('real_prior' or "
                         "'prior' mapping); default: class counts of all training-file rows")
    ap.add_argument("--holdout-only", action="store_true",
                    help="Do not train: recompute metrics for existing runs from saved "
                         "probabilities (or re-predict from saved models if probabilities "
                         "are missing or --repredict is given)")
    ap.add_argument("--repredict", action="store_true",
                    help="With --holdout-only: re-score --holdout with the saved models")
    ap.add_argument("--force", action="store_true", help="Retrain runs whose JSON exists")
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: rounds=2, central epochs=2, 20 trees, seeds 1,2, "
                         "strengths 0,0.5,1")
    args = ap.parse_args(argv)
    if args.quick:
        args.rounds, args.central_max_epochs, args.lgbm_n_estimators = 2, 2, 20
        args.seeds = args.seeds or "1,2"
        args.strengths = args.strengths or "0,0.5,1"
    args.seeds = args.seeds or "1,2,3,4,5"
    args.strengths = args.strengths or DEFAULT_STRENGTHS
    args.sampler = "legacy" if args.legacy_sampler else "uniform"
    args.seed_list = [int(s) for s in args.seeds.split(",") if s.strip()]
    args.model_list = [m.strip() for m in args.models.split(",") if m.strip()]
    args.fs_list = [f.strip() for f in args.feature_sets.split(",") if f.strip()]
    args.strength_list = [float(s) for s in args.strengths.split(",") if s.strip()]
    args.rho_list = [float(r) for r in args.rhos.split(",") if r.strip()]
    args.cnn_channels = tuple(int(c) for c in args.cnn_channels.split(","))
    for m in args.model_list:
        if m not in MODELS:
            ap.error(f"unknown model {m}; choose from {MODELS}")
    for f in args.fs_list:
        if f not in FEATURE_SETS:
            ap.error(f"unknown feature set {f}; choose from {list(FEATURE_SETS)}")
    if not args.holdout_only and (not args.train_files or not args.holdout):
        ap.error("--train-files and --holdout are required unless --holdout-only")
    return args


def setup_device(name):
    if name == "auto":
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        dev = torch.device(name)
    fs.DEVICE = dev  # local_train / evaluate / build_model read this module global
    return dev


def versions():
    import platform
    import sklearn
    v = {"python": platform.python_version(), "torch": torch.__version__,
         "numpy": np.__version__, "pandas": pd.__version__, "sklearn": sklearn.__version__}
    try:
        import lightgbm
        v["lightgbm"] = lightgbm.__version__
    except ImportError:
        v["lightgbm"] = None
    v["cuda"] = torch.version.cuda if torch.cuda.is_available() else None
    return v


def save_probs(path, proba):
    np.savez_compressed(path, proba=proba.astype(np.float16))


def write_holdout_labels(out_dir, y, day, unseen, classes, info, main_days, day_counts):
    path = os.path.join(out_dir, "holdout_labels.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if "unseen" in z.files and len(z["y"]) == len(y) and np.array_equal(z["y"], y):
            return path
        log(f"NOTE: rewriting {path} (holdout changed or older format)")
    meta = dict(info, main_days=main_days, day_counts=day_counts)
    np.savez_compressed(path, y=y, day=np.asarray(day, dtype="U10"), unseen=unseen,
                        classes=np.array(classes), info=np.array(json.dumps(meta)))
    return path


def holdout_section(info, y, classes, main_days, day_counts):
    return {"path": info["path"], "rows": int(len(y)),
            "main_days": main_days, "day_counts": day_counts,
            "support": {c: int((y == i).sum()) for i, c in enumerate(classes)},
            "blank_share_by_class": info.get("blank_share_by_class"),
            "rules": info.get("rules"),
            "overlap": info.get("overlap"),
            "never_used_for_selection": True}


SELECTION_RULE = {
    "fed_cnn": "best FedAvg round by unweighted accuracy on the pooled validation rows, "
               "argmax of the raw softmax (s=0)",
    "central_cnn": "best epoch by unweighted accuracy on the pooled validation rows, argmax of "
                   "the raw softmax (s=0); early stop after `patience` epochs without improvement",
    "lgbm": "early stopping on multi_logloss of the pooled validation rows weighted with the "
            "balanced class weights",
}


def main(argv=None):
    global LOG_FH
    args = parse_args(argv)
    out = os.path.abspath(args.out_dir)
    for sub in ("cache", "runs", "probs", "models"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    LOG_FH = open(os.path.join(out, "run_experiments.log"), "a")
    log("=" * 70)
    log("run_experiments.py " + " ".join(sys.argv[1:] if argv is None else argv))
    dev = setup_device(args.device)
    log(f"device={dev}  versions={versions()}  threads={torch.get_num_threads()}")
    if args.holdout_only:
        return holdout_only(args, out)
    return train_all(args, out)


def train_all(args, out):
    todo = [(f, s, m) for f in args.fs_list for s in args.seed_list for m in args.model_list
            if args.force or not os.path.exists(os.path.join(out, "runs", run_id(m, f, s) + ".json"))]
    n_total = len(args.fs_list) * len(args.seed_list) * len(args.model_list)
    log(f"{n_total} run(s) requested, {n_total - len(todo)} already done, {len(todo)} to do")
    if not todo:
        return 0

    cache = os.path.join(out, "cache")
    stats, train_icao, stats_path = load_train_stats(args.train_files, cache)
    stats_key = os.path.basename(stats_path)
    df, sample_meta = load_training_sample(args.train_files, args.max_total_rows, cache,
                                           args.sampler, stats)
    classes = sorted(df["attack_type"].astype(str).unique())
    y_all = df["attack_type"].astype(str).map({c: i for i, c in enumerate(classes)}).values.astype(np.int64)
    K = len(classes)
    if "benign" not in classes:
        raise ValueError("no 'benign' class in training data")
    log(f"Classes: {classes}")
    prior, prior_src = natural_prior(stats, classes, prior_json=args.prior_json)
    Xh_all, yh, dayh, unseen, hinfo = load_holdout(args.holdout, classes, cache, train_icao, stats_key)
    del train_icao
    main_days, day_counts = holdout_days(dayh)
    log(f"Holdout days (>= {MIN_DAY_SHARE:.0%} of rows): {main_days}; all: {day_counts}")
    write_holdout_labels(out, yh, dayh, unseen, classes, hinfo, main_days, day_counts)

    # class weights: balanced, from all loaded labels (as federated_simulation.main)
    counts = np.bincount(y_all, minlength=K)
    class_weights = len(y_all) / (K * counts)
    log("Class weights (balanced): " + ", ".join(f"{c}={w:.3f}" for c, w in zip(classes, class_weights)))
    regions = df["region_cell"].astype(str).values
    vers = versions()

    for fset in args.fs_list:
        jobs = [(s, m) for f, s, m in todo if f == fset]
        if not jobs:
            continue
        cols = FEATURE_SETS[fset]
        log(f"--- feature set '{fset}' ({len(cols)} features) ---")
        Xraw_df = prep_subset(df, cols)
        scaler = StandardScaler()
        X = scaler.fit_transform(Xraw_df).astype(np.float32)   # scaler on pooled sample, as main()
        Xraw = Xraw_df.values.astype(np.float32)
        del Xraw_df
        gc.collect()
        col_idx = [fs.FEATURE_COLS.index(c) for c in cols]
        Xh_raw = np.ascontiguousarray(Xh_all[:, col_idx])
        Xh = scaler.transform(pd.DataFrame(Xh_raw.astype(np.float64), columns=cols)).astype(np.float32)

        for seed in sorted(set(s for s, _ in jobs)):
            clients, dropped = make_clients(regions, args.min_client_rows, seed)
            if len(clients) < 2:
                raise SystemExit("fewer than 2 usable clients; lower --min-client-rows")
            csum = client_summary(clients, y_all, classes)
            atomic_json({"seed": seed, "min_client_rows": args.min_client_rows,
                         "n_clients": len(clients), "dropped_regions": dropped, "clients": csum},
                        os.path.join(out, "runs", f"clients__seed{seed}.json"))
            n_tr = sum(len(c["train_idx"]) for c in clients)
            n_va = sum(len(c["val_idx"]) for c in clients)
            log(f"seed {seed}: {len(clients)} clients ({len(dropped)} regions dropped), "
                f"{n_tr:,} train rows, {n_va:,} validation rows")
            for s2, model_name in jobs:
                if s2 != seed:
                    continue
                rid = run_id(model_name, fset, seed)
                tag = rid
                log(f">>> {rid}")
                t0 = time.time()
                if model_name == "fed_cnn":
                    model, info = train_fed_cnn(X, y_all, clients, class_weights, K, args, seed, tag)
                elif model_name == "central_cnn":
                    model, info = train_central_cnn(X, y_all, clients, class_weights, K, args, seed, tag)
                else:
                    model, info = train_lgbm(Xraw, y_all, clients, class_weights, K, args, seed, tag)
                info["selection_rule"] = SELECTION_RULE[model_name]

                tp = time.time()
                if model_name == "lgbm":
                    proba = model.predict_proba(Xh_raw)
                    gain = np.array(info.pop("_gain"), float)
                    info["feature_importance_gain_share"] = {
                        c: float(g / max(gain.sum(), 1e-12)) for c, g in zip(cols, gain)}
                    model.booster_.save_model(os.path.join(out, "models", rid + ".txt"))
                else:
                    proba = softmax_np(predict_logits(model, Xh))
                    torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                                "model_type": "cnn", "n_features": len(cols), "feature_cols": cols,
                                "class_names": classes, "cnn_channels": list(args.cnn_channels)},
                               os.path.join(out, "models", rid + ".pt"))
                    import joblib
                    joblib.dump(scaler, os.path.join(out, "models", rid + "_scaler.pkl"))
                proba16 = proba.astype(np.float16)
                save_probs(os.path.join(out, "probs", rid + ".npz"), proba16)
                predict_s = time.time() - tp
                metrics = evaluate_strengths(proba16, yh, dayh, main_days, prior,
                                             args.strength_list, classes, unseen, args.rho_list)
                res = {
                    "run_id": rid, "model": model_name, "feature_set": fset, "features": cols,
                    "seed": seed, "classes": classes, "versions": vers,
                    "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()
                               if k not in ("seed_list", "model_list", "fs_list")},
                    "data": {
                        "train_files": [os.path.abspath(p) for p in args.train_files],
                        "sampler": sample_meta.get("sampler"),
                        "cap_per_region_class": sample_meta.get("cap_per_region_class"),
                        "max_total_rows": args.max_total_rows,
                        "train_sample_rows": int(len(y_all)),
                        "train_raw_rows": stats["total_rows"],
                        "train_raw_rows_streamed": sample_meta.get("raw_rows_streamed"),
                        "train_raw_rows_per_file": {os.path.basename(f["file"]): f["rows"]
                                                    for f in stats["per_file"]},
                        "train_raw_aircraft": stats.get("n_aircraft"),
                        "train_sample_aircraft": sample_meta.get("n_aircraft_in_sample"),
                        "train_sample_composition": sample_meta.get("composition"),
                        "train_stats_cache": stats_path,
                        "train_sample_class_counts": {c: int(n) for c, n in zip(classes, counts)},
                        "class_weights": {c: float(w) for c, w in zip(classes, class_weights)},
                        "natural_prior": {c: float(p) for c, p in zip(classes, prior)},
                        "natural_prior_source": prior_src,
                        "natural_class_counts": {c: int(stats["class_counts"].get(c, 0)) for c in classes},
                        "n_regions_in_sample": int(len(set(regions))),
                        "min_client_rows": args.min_client_rows,
                        "n_clients": len(clients), "n_dropped_regions": len(dropped),
                        "train_split_rows": int(n_tr), "val_rows": int(n_va),
                        "val_note": "validation = per-client 20% held-out split of the training sample",
                    },
                    "training": info,
                    "holdout": holdout_section(hinfo, yh, classes, main_days, day_counts),
                    "strengths": args.strength_list,
                    "rhos": args.rho_list,
                    "metrics": metrics,
                    "predict_seconds": predict_s,
                    "wall_seconds": time.time() - t0,
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "device": str(fs.DEVICE),
                }
                if model_name == "fed_cnn":
                    res["clients"] = csum
                atomic_json(res, os.path.join(out, "runs", rid + ".json"))
                m1 = metrics[f"{args.strength_list[-1]:.2f}"]["pooled"]
                m0 = metrics[f"{args.strength_list[0]:.2f}"]["pooled"]
                log(f"<<< {rid} done in {(time.time() - t0) / 60:.1f} min | "
                    f"s={args.strength_list[0]:g}: acc {m0['accuracy']:.4f} F1 {m0['macro_f1']:.4f} "
                    f"FAR {m0['false_alarm_rate']:.4f} | s={args.strength_list[-1]:g}: acc "
                    f"{m1['accuracy']:.4f} F1 {m1['macro_f1']:.4f} FAR {m1['false_alarm_rate']:.4f} "
                    f"atk rec {m1['avg_attack_recall']:.4f}")
                del model, proba
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del X, Xraw, Xh, Xh_raw
        gc.collect()
    log("all requested runs finished")
    return 0


def holdout_only(args, out):
    """Recompute metrics for existing runs without retraining."""
    lab_path = os.path.join(out, "holdout_labels.npz")
    Xh_all = None
    need_labels = not os.path.exists(lab_path) or "unseen" not in np.load(lab_path).files
    if (args.repredict or need_labels) and not args.holdout:
        raise SystemExit("--holdout is needed (no current holdout_labels.npz, or --repredict)")
    done = 0
    for fset in args.fs_list:
        for seed in args.seed_list:
            for m in args.model_list:
                rid = run_id(m, fset, seed)
                jpath = os.path.join(out, "runs", rid + ".json")
                if not os.path.exists(jpath):
                    log(f"skip {rid}: no run JSON")
                    continue
                res = json.load(open(jpath))
                classes = res["classes"]
                prior = np.array([res["data"]["natural_prior"][c] for c in classes])
                ppath = os.path.join(out, "probs", rid + ".npz")
                need_rescore = args.repredict or not os.path.exists(ppath)
                if Xh_all is None and (need_rescore or need_labels):
                    spath = res["data"].get("train_stats_cache")
                    ipath = (spath or "").replace("train_stats_", "train_icao24_").replace(".json", ".txt.gz")
                    if not os.path.exists(ipath):
                        raise SystemExit(f"training icao24 cache {ipath} not found (needed for the "
                                         f"unseen-aircraft subset)")
                    with gzip.open(ipath, "rt") as f:
                        train_icao = set(f.read().split())
                    Xh_all, yh, dayh, unseen, hinfo = load_holdout(
                        args.holdout, classes, os.path.join(out, "cache"), train_icao,
                        os.path.basename(spath))
                    del train_icao
                    main_days, day_counts = holdout_days(dayh)
                    write_holdout_labels(out, yh, dayh, unseen, classes, hinfo, main_days, day_counts)
                    need_labels = False
                if need_rescore:
                    if not args.holdout:
                        raise SystemExit(f"{rid}: no saved probabilities; pass --holdout to re-score")
                    cols = res["features"]
                    Xr = np.ascontiguousarray(Xh_all[:, [fs.FEATURE_COLS.index(c) for c in cols]])
                    if m == "lgbm":
                        import lightgbm as lgb
                        booster = lgb.Booster(model_file=os.path.join(out, "models", rid + ".txt"))
                        proba = booster.predict(Xr)
                    else:
                        import joblib
                        scaler = joblib.load(os.path.join(out, "models", rid + "_scaler.pkl"))
                        payload = torch.load(os.path.join(out, "models", rid + ".pt"),
                                             map_location=fs.DEVICE)
                        model = fs.build_model("cnn", payload["n_features"], len(classes),
                                               cnn_channels=tuple(payload["cnn_channels"]))
                        model.load_state_dict(payload["model_state_dict"])
                        Xs = scaler.transform(pd.DataFrame(Xr.astype(np.float64), columns=cols))
                        proba = softmax_np(predict_logits(model, Xs.astype(np.float32)))
                    save_probs(ppath, proba)
                    log(f"{rid}: re-scored holdout with saved model")
                z = np.load(lab_path, allow_pickle=False)
                yh_l, day_l, unseen_l = z["y"], z["day"], z["unseen"]
                hmeta = json.loads(str(z["info"]))
                proba16 = np.load(ppath)["proba"]
                if len(proba16) != len(yh_l):
                    raise SystemExit(f"{rid}: probabilities ({len(proba16)}) and holdout labels "
                                     f"({len(yh_l)}) differ; use --repredict --holdout ...")
                res["metrics"] = evaluate_strengths(proba16, yh_l, day_l, hmeta["main_days"], prior,
                                                    args.strength_list, classes, unseen_l,
                                                    args.rho_list)
                res["strengths"] = args.strength_list
                res["rhos"] = args.rho_list
                res["holdout"] = holdout_section(hmeta, yh_l, classes, hmeta["main_days"],
                                                 hmeta["day_counts"])
                res["holdout_only_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                atomic_json(res, jpath)
                done += 1
                m1 = res["metrics"][f"{args.strength_list[-1]:.2f}"]["pooled"]
                log(f"{rid}: metrics recomputed for {len(args.strength_list)} strengths "
                    f"(s={args.strength_list[-1]:g}: F1 {m1['macro_f1']:.4f})")
    log(f"holdout-only: {done} run(s) updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
