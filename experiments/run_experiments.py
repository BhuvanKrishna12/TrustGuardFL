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

Revision 2 additions (see experiments/README.md, "Revision 2"):
  --variant   correction variant of multi-variant files (variants.py): the model
              is trained on that variant and scored on (i) the variant's own test
              sample and (ii) the fully corrected test sample.
  --protocol  v2 (default): aircraft-disjoint validation split, scaler and class
              weights fitted on training rows only, correction strength s* chosen
              on the prior-reweighted validation set BEFORE any test scoring
              (experiments/SELECTION_RULE.md). v1: the round-1 behaviour, exactly.
  models      central_mlp / fed_mlp (MLP with the CNN's parameter count),
              --fedprox-mu for the federated models.
  resume      every federated round and every centralized epoch is checkpointed
              (checkpoints/); a rerun continues and gives the same result as an
              uninterrupted run. --stop-flag FILE: when FILE exists the run stops
              after the current round / epoch / run with exit code 75.

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
import variants as V  # noqa: E402
from sampling import StopRequested  # noqa: E402

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------

FEATURE_SETS = {
    "full": list(fs.FEATURE_COLS),
    "core": ["implied_speed_mps", "speed_consistency_delta", "vrate_consistency_delta",
             "inter_arrival_s", "turn_rate_deg_s", "region_avg_inter_arrival", "is_stale"],
}
MODELS = ("fed_cnn", "central_cnn", "lgbm", "central_mlp", "fed_mlp")
DEFAULT_MODELS = ("fed_cnn", "central_cnn", "lgbm")
ARCH = {"fed_cnn": "cnn", "central_cnn": "cnn", "central_mlp": "mlp", "fed_mlp": "mlp"}
# MLP baseline: 13 -> 190 -> 44 -> 6 has 11,334 parameters, exactly the CNN's count
# on the full feature set ((n+7)(h2+14) factorisation of the parameter formula).
DEFAULT_MLP_HIDDEN = "190,44"
SELECTION_GRID = [round(i / 10, 1) for i in range(11)]   # pre-registered grid for s*
VAL_AIRCRAFT_SHARE = 0.2
EXIT_STOPPED = 75
STOP_FLAG = None
RATE_FEATURES = ["implied_speed_mps", "speed_consistency_delta",
                 "vrate_consistency_delta", "turn_rate_deg_s"]
SAMPLE_SEED = 42           # the training sample is drawn once with this seed
TRAIN_FRACTION = 0.8       # per-client train / validation split, as in build_clients
MIN_DAY_SHARE = 0.05       # a holdout date is a "test day" if it holds >= 5% of rows
DEFAULT_STRENGTHS = "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"
DEFAULT_RHOS = (0.001, 0.01)  # assumed deployment attack prevalences (test-row reweighting)

LOG_FH = None


def stop_requested():
    return bool(STOP_FLAG) and os.path.exists(STOP_FLAG)


def check_stop(where):
    if stop_requested():
        raise StopRequested(where)


# Test hooks (experiments/test_fedprox_resume.py): after N checkpointed units
# either raise the STOP flag (graceful path) or SIGKILL this process in the
# middle of the next unit (power-loss path). Never set in real runs.
_TEST = {"stop_after": None, "kill_after": None, "units": 0}


def _test_hook(stage):
    if stage == "ckpt":
        _TEST["units"] += 1
        if _TEST["stop_after"] is not None and _TEST["units"] >= _TEST["stop_after"] and STOP_FLAG:
            open(STOP_FLAG, "w").write("test\n")
    elif stage == "mid" and _TEST["kill_after"] is not None and _TEST["units"] >= _TEST["kill_after"]:
        import signal
        if LOG_FH is not None:
            LOG_FH.flush()
        os.kill(os.getpid(), signal.SIGKILL)


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
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_npz(path, **arrays):
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def atomic_torch_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
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
    of training icao24 addresses. One chunked pass per file, each file cached
    on its own (a stopped pass continues with the next unfinished file). For
    multi-variant files the counts are kept per sampling design."""
    key = file_key(paths, "stats-v2")
    jpath = os.path.join(cache_dir, f"train_stats_{key}.json")
    log("Training-file stats: region_cell, attack_type, icao24 of every training file "
        "(raw rows, natural prior, training aircraft); cached per file")
    stats, icao = sampling.train_file_stats_cached(paths, cache_dir, log=log,
                                                   stop_check=stop_requested)
    stats["files"] = [os.path.abspath(p) for p in paths]
    if not os.path.exists(jpath):
        atomic_json(stats, jpath)
    log(f"  {stats['total_rows']:,} raw rows, {stats['n_regions']:,} regions, "
        f"{stats['n_aircraft']:,} distinct icao24"
        + ("  [multi-variant files]" if stats.get("mv") else ""))
    return stats, icao, jpath


def _slim_sample(df):
    """Dtypes that keep the cached sample small."""
    df = df.reset_index(drop=True)
    for c in ("attack_type", "attack_type_legacy", "region_cell", "icao24"):
        if c in df.columns and not pd.api.types.is_integer_dtype(df[c]):
            df[c] = df[c].astype(str).astype("category")
    return df


def load_training_sample(paths, max_total_rows, cache_dir, sampler, stats, legacy_set=frozenset()):
    """Training sample of one variant, drawn with seed 42.

    uniform: ONE cached union sample for all variants (sampling.uniform_sample_mv);
             the variant's rows / labels / features are selected in memory.
    legacy:  federated_simulation.load_data semantics on the variant's design
             view of the files (sampling.legacy_sample_view), one cache per design.
    Returns (df with FEATURE_COLS, region_cell, attack_type and icao24 if
    available, meta)."""
    ls = frozenset(legacy_set)
    design = V.design_of(ls)
    if design not in stats["designs"]:
        raise SystemExit(f"--variant {V.variant_name(ls)} needs multi-variant training files "
                         f"(opensky_fl_ids_pipeline.py --pipeline-version mv / reprocess_mv.py)")
    tag = sampler if sampler == "uniform" else f"legacy_{design}"
    key = file_key(paths, max_total_rows, SAMPLE_SEED, tag, "r2")
    meta_path = os.path.join(cache_dir, f"train_sample_{tag}_{key}.json")
    data_path = os.path.join(cache_dir, f"train_sample_{tag}_{key}.pkl")
    state_path = os.path.join(cache_dir, f"train_sample_{tag}_{key}.state.pkl")

    if os.path.exists(data_path) and os.path.exists(meta_path):
        log(f"Training sample ({tag}): loading cache {data_path}")
        raw = pd.read_pickle(data_path)
        base = json.load(open(meta_path))
    else:
        t0 = time.time()
        if sampler == "uniform":
            log(f"Training sample: uniform per-(region, class) sampler over {len(paths)} file(s) "
                f"(max_total_rows={max_total_rows:,}, seed={SAMPLE_SEED}); slow one-off step, "
                f"resumable per file")
            raw, base = sampling.uniform_sample_mv(paths, max_total_rows, stats, seed=SAMPLE_SEED,
                                                   log=log, state_path=state_path,
                                                   stop_check=stop_requested)
            base["hit_global_cap_early"] = False
        else:
            log(f"Training sample: LEGACY load_data semantics over {len(paths)} file(s), design "
                f"{design} (max_total_rows={max_total_rows:,}, seed={SAMPLE_SEED}); slow one-off "
                f"step, resumable per file")
            raw, base = sampling.legacy_sample_view(paths, max_total_rows, stats, ls,
                                                    seed=SAMPLE_SEED, log=log,
                                                    state_path=state_path,
                                                    stop_check=stop_requested)
        raw = _slim_sample(raw)
        base["load_minutes"] = (time.time() - t0) / 60
        base["rows_cached"] = int(len(raw))
        tmp = data_path + ".tmp"
        raw.to_pickle(tmp)
        os.replace(tmp, data_path)
        atomic_json(base, meta_path)          # written last: marks the cache as complete
        if os.path.exists(state_path):
            os.remove(state_path)
        log(f"  cached {len(raw):,} rows -> {data_path} ({base['load_minutes']:.1f} min)")

    if sampler == "uniform":
        df = sampling.sample_view(raw, base, ls)
    else:
        df = sampling.features_view(raw, ls) if ls else raw
    del raw
    classes = sorted(df["attack_type"].astype(str).unique())
    comp = sampling.sample_composition(df, paths, classes)
    n_icao = int(df["icao24"].nunique()) if "icao24" in df else None
    keep = [c for c in fs.FEATURE_COLS + ["region_cell", "attack_type", "icao24"] if c in df.columns]
    df = df[keep].copy()
    df["attack_type"] = df["attack_type"].astype(str).astype("category")
    df["region_cell"] = df["region_cell"].astype(str).astype("category")
    df = df.reset_index(drop=True)
    gc.collect()

    dstats = stats["designs"][design]
    meta = dict(base)
    meta.update({
        "files": [os.path.abspath(p) for p in paths],
        "file_sizes": [os.path.getsize(p) for p in paths],
        "max_total_rows": max_total_rows,
        "rows": int(len(df)),
        "design": design,
        "cap_per_region_class": base.get("caps_by_design", {}).get(design,
                                                                    base.get("cap_per_region_class")),
        "train_raw_rows_total": dstats["total_rows"],
        "n_regions_in_files": dstats["n_regions"],
        "n_aircraft_in_sample": n_icao,
        "composition": comp,
        "data_path": data_path,
    })
    log(f"  variant {V.variant_name(ls)} (design {design}): {len(df):,} sample rows")
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
    counts = stats["class_counts"]   # caller passes the variant's design stats
    tot = sum(counts.get(c, 0) for c in classes)
    prior = np.array([counts.get(c, 0) / tot for c in classes])
    if (prior == 0).any():
        log(f"  WARNING: zero natural prior for {list(np.array(classes)[prior == 0])}")
    log("Natural prior pi = " + ", ".join(f"{c}={p:.4f}" for c, p in zip(classes, prior)))
    return prior, "training-file stats pass"


def load_holdout(path, classes, cache_dir, train_icao, stats_key, chunksize=500_000,
                 legacy_set=frozenset()):
    """Holdout arrays of one variant: prepped 13 features (float32), labels, UTC
    day from lastposupdate (fallback time), whether the row's icao24 never
    occurs in the training files, region cell and a `keep` mask is applied
    (rows removed by the variant's legacy filter are not returned; `rowpos`
    gives the position of every returned row in the test-sample file, so
    variants can be aligned row by row). Also rule-baseline counts and
    blank-feature shares. Cached as npz. Returns a dict."""
    ls = frozenset(legacy_set)
    st = os.stat(path)
    key = hashlib.sha1(json.dumps([os.path.basename(path), st.st_size, int(st.st_mtime),
                                   list(classes), stats_key, "holdout-v3",
                                   V.variant_name(ls)]).encode()).hexdigest()[:12]
    cpath = os.path.join(cache_dir, f"holdout_{key}.npz")
    if os.path.exists(cpath):
        log(f"Holdout ({V.variant_name(ls)}): loading cache {cpath}")
        z = np.load(cpath, allow_pickle=False)
        return {"X": z["X"], "y": z["y"], "day": z["day"], "unseen": z["unseen"],
                "region": z["region"], "rowpos": z["rowpos"], "info": json.loads(str(z["info"]))}

    log(f"Holdout ({V.variant_name(ls)}): reading {path}")
    header = pd.read_csv(path, nrows=0).columns
    mv = V.is_mv_columns(header)
    if ls and not mv:
        raise SystemExit(f"--variant {V.variant_name(ls)} needs a multi-variant test sample")
    want = fs.FEATURE_COLS + ["attack_type", "time", "lastposupdate", "icao24", "region_cell"]
    if mv:
        want = want + V.MV_READ_COLS
    usecols = [c for c in want if c in header]
    missing = set(fs.FEATURE_COLS + ["attack_type"]) - set(usecols)
    if missing:
        raise ValueError(f"holdout is missing columns {missing}")
    has_icao = "icao24" in usecols
    cls_index = {c: i for i, c in enumerate(classes)}
    Xs, ys, days, blanks, floods, unseens, icaos, regs, poss = [], [], [], [], [], [], [], [], []
    dropped = 0
    n_file = n_masked = 0
    for ch in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
        pos = np.arange(n_file, n_file + len(ch), dtype=np.int64)
        n_file += len(ch)
        if ls:
            ch, keep = V.apply_view(ch, ls)
            n_masked += int((~keep).sum())
            ch, pos = ch[keep], pos[keep]
        at = ch["attack_type"].astype(str)
        ok = at.isin(cls_index).values
        dropped += int((~ok).sum())
        ch = ch[ok]
        at = at[ok]
        poss.append(pos[ok])
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
        regs.append(ch["region_cell"].astype(str).to_numpy() if "region_cell" in ch
                    else np.full(len(ch), "unknown"))
        days.append(sampling.day_of(ch))
        Xs.append(fs.prep_features(ch).astype(np.float32).values)
        ys.append(at.map(cls_index).values.astype(np.int8))
    X = np.concatenate(Xs)
    y = np.concatenate(ys)
    day = np.concatenate(days)
    blank = np.concatenate(blanks)
    flood = np.concatenate(floods)
    unseen = np.concatenate(unseens)
    region = np.concatenate(regs).astype("U16")
    rowpos = np.concatenate(poss)

    def fires(mask):
        return {c: int((mask & (y == i)).sum()) for i, c in enumerate(classes)}

    support = {c: int((y == i).sum()) for i, c in enumerate(classes)}
    info = {"path": os.path.abspath(path), "rows": int(len(y)),
            "rows_in_file": int(n_file), "multi_variant_file": bool(mv),
            "variant": V.variant_name(ls),
            "rows_masked_by_legacy_filter": int(n_masked),
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
    atomic_npz(cpath, X=X, y=y, day=np.asarray(day, dtype="U10"), unseen=unseen, region=region,
               rowpos=rowpos, info=np.array(json.dumps(info)))
    log(f"  {len(y):,} rows ({dropped:,} dropped with unknown class, {n_masked:,} masked by the "
        f"variant's legacy filter), {int(unseen.sum()):,} rows from aircraft never seen in "
        f"training -> cached {cpath}")
    return {"X": X, "y": y, "day": day, "unseen": unseen, "region": region, "rowpos": rowpos,
            "info": info}


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


def aircraft_in_validation(icao, seed, share=VAL_AIRCRAFT_SHARE):
    """Seeded hash of the icao24 address -> True for about `share` of the
    aircraft. blake2b, so the split does not depend on Python / numpy versions."""
    codes, uniq = pd.factorize(pd.Series(icao).astype(str), sort=False)
    u = np.empty(len(uniq), np.float64)
    for i, a in enumerate(uniq):
        h = hashlib.blake2b(f"{seed}|{a}".encode(), digest_size=8).digest()
        u[i] = int.from_bytes(h, "big") / 2.0 ** 64
    return (u < share)[codes], int(len(uniq)), int((u < share).sum())


def make_clients_v2(regions, icao, y, classes, min_rows, seed):
    """Protocol v2 clients: the same region clients as make_clients (regions
    with at least min_rows sampled rows), but the validation rows are ALL rows
    of the aircraft whose seeded address hash falls in the validation share
    (about 20% of aircraft), in every client. No aircraft has rows on both
    sides. Ghost rows carry a fabricated address that occurs once, so each
    ghost row is its own one-message aircraft and goes to validation with
    probability 0.2 on its own (there is no other row it could leak to or
    from; the address of the aircraft it was taken from is not in the data)."""
    is_val, n_air, n_val_air = aircraft_in_validation(icao, seed)
    groups = pd.DataFrame({"region_cell": regions}).groupby("region_cell", observed=True).groups
    K = len(classes)
    clients, dropped, lost = [], [], []
    for region, idx in groups.items():
        idx = np.sort(np.array(idx))
        if len(idx) < min_rows:
            dropped.append((str(region), int(len(idx))))
            continue
        tr, va = idx[~is_val[idx]], idx[is_val[idx]]
        if len(tr) == 0:
            dropped.append((str(region), int(len(idx))))
            continue
        clients.append({"region": str(region), "train_idx": tr, "val_idx": va})
        call = np.bincount(y[idx], minlength=K) > 0
        ctr = np.bincount(y[tr], minlength=K) > 0
        cva = np.bincount(y[va], minlength=K) > 0
        if (call & ~ctr).any() or (call & ~cva).any():
            lost.append({"region": str(region),
                         "missing_in_train": [classes[k] for k in np.flatnonzero(call & ~ctr)],
                         "missing_in_val": [classes[k] for k in np.flatnonzero(call & ~cva)]})
    n_tr = sum(len(c["train_idx"]) for c in clients)
    n_va = sum(len(c["val_idx"]) for c in clients)
    ytr = np.concatenate([y[c["train_idx"]] for c in clients]) if clients else np.zeros(0, int)
    yva = np.concatenate([y[c["val_idx"]] for c in clients]) if clients else np.zeros(0, int)
    report = {
        "split": "aircraft-disjoint (seeded blake2b hash of icao24)",
        "val_aircraft_share_target": VAL_AIRCRAFT_SHARE,
        "aircraft_in_sample": n_air, "aircraft_in_validation": n_val_air,
        "train_rows": int(n_tr), "val_rows": int(n_va),
        "val_row_share": float(n_va / max(1, n_tr + n_va)),
        "train_rows_by_class": {classes[k]: int(v) for k, v in enumerate(np.bincount(ytr, minlength=K))},
        "val_rows_by_class": {classes[k]: int(v) for k, v in enumerate(np.bincount(yva, minlength=K))},
        "clients_losing_a_class_in_train": int(sum(1 for c in lost if c["missing_in_train"])),
        "clients_losing_a_class_in_val": int(sum(1 for c in lost if c["missing_in_val"])),
        "clients_with_empty_validation": int(sum(1 for c in clients if len(c["val_idx"]) == 0)),
        "class_loss_detail": lost[:400],
        "ghost_rows": "each fabricated address is a one-row aircraft; split by its own hash",
    }
    return clients, dropped, report


# ---------------------------------------------------------------------------
# trainers (checkpointed; a resumed run equals an uninterrupted one)
# ---------------------------------------------------------------------------

def build_net(arch, n_features, n_classes, args):
    if arch == "cnn":
        return fs.build_model("cnn", n_features, n_classes, cnn_channels=args.cnn_channels)
    return fs.build_model("mlp", n_features, n_classes, mlp_hidden_sizes=args.mlp_hidden)


def _rng_snapshot():
    snap = {"torch": torch.get_rng_state(), "numpy": np.random.get_state()}
    if torch.cuda.is_available():
        snap["cuda"] = torch.cuda.get_rng_state_all()
    return snap


def _rng_restore(snap):
    torch.set_rng_state(snap["torch"])
    np.random.set_state(snap["numpy"])
    if torch.cuda.is_available() and snap.get("cuda") is not None:
        torch.cuda.set_rng_state_all(snap["cuda"])


def _cpu_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _load_ckpt(path, signature):
    if not path or not os.path.exists(path):
        return None
    try:
        ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:  # truncated file cannot happen with os.replace, but be safe
        log(f"  checkpoint {path} unreadable ({e}); starting this run from the beginning")
        return None
    if ck.get("signature") != signature:
        log(f"  checkpoint {path} belongs to another configuration; ignoring it")
        return None
    return ck


def local_train_prox(model, X_train, y_train, epochs, class_weights, mu):
    """federated_simulation.local_train with the FedProx proximal term
    (mu / 2) * ||w - w_global||^2 added to every batch loss. Same data loader,
    optimizer and RNG use as local_train; with mu = 0 the added term has zero
    value and zero gradient, so the result equals FedAvg bit for bit
    (experiments/test_fedprox_resume.py)."""
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=fs.LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    global_params = [p.detach().clone() for p in model.parameters()]
    ds = torch.utils.data.TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                                        torch.tensor(y_train, dtype=torch.long))
    loader = torch.utils.data.DataLoader(ds, batch_size=fs.BATCH_SIZE, shuffle=True)
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(fs.DEVICE), yb.to(fs.DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
            loss = loss + (mu / 2.0) * prox
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_train, dtype=torch.float32).to(fs.DEVICE)
        yb = torch.tensor(y_train, dtype=torch.long).to(fs.DEVICE)
        train_acc = (model(xb).argmax(dim=1) == yb).float().mean().item()
    return model.state_dict(), train_acc


def train_fed(X, y, clients, class_weights, n_classes, args, seed, tag, arch="cnn",
              ckpt_path=None, signature=None):
    """The FedAvg loop of federated_simulation.main, driven with a seed.
    Checkpoint after EVERY round: global weights, best-so-far weights and
    selection state, history, and all RNG states."""
    n_features = X.shape[1]
    mu = getattr(args, "fedprox_mu", None)
    cw = torch.tensor(class_weights, dtype=torch.float32).to(fs.DEVICE)
    cdata = {}
    for c in clients:
        cdata[c["region"]] = dict(X_train=X[c["train_idx"]], y_train=y[c["train_idx"]])
    Xv = np.concatenate([X[c["val_idx"]] for c in clients])
    yv = np.concatenate([y[c["val_idx"]] for c in clients])

    seed_everything(seed)
    model = build_net(arch, n_features, n_classes, args)
    payload = fs.model_size_bytes(model)
    n_params = sum(p.numel() for p in model.parameters())
    names = list(cdata.keys())
    rng = np.random.default_rng(seed)
    best_acc, best_round, best_state = -1.0, 0, None
    hist = {"round": [], "train_acc": [], "val_acc": [], "seconds": []}
    total_bytes = 0
    start_round, elapsed0, resumes = 1, 0.0, 0
    ck = _load_ckpt(ckpt_path, signature)
    if ck is not None:
        model.load_state_dict(ck["model"])
        best_acc, best_round, best_state = ck["best_acc"], ck["best_round"], ck["best_state"]
        hist, total_bytes = ck["hist"], ck["total_bytes"]
        rng.bit_generator.state = ck["np_generator"]
        _rng_restore(ck["rng"])
        start_round, elapsed0, resumes = ck["round"] + 1, ck["elapsed"], ck.get("resumes", 0) + 1
        log(f"  [{tag}] resuming after round {ck['round']} (checkpoint {ckpt_path})")
    t0 = time.time()
    log(f"  [{tag}] Fed{'Prox mu=%g' % mu if mu is not None else 'Avg'} ({arch}): {args.rounds} rounds x "
        f"{args.local_epochs} local epochs, {len(names)} clients, model {payload:,} bytes "
        f"({n_params:,} params)")
    for rnd in range(start_round, args.rounds + 1):
        tr = time.time()
        n_part = max(1, int(len(names) * fs.CLIENT_FRACTION))
        participants = rng.choice(names, size=n_part, replace=False)
        states, weights, taccs = [], [], []
        for ci, cname in enumerate(participants):
            if ci == len(participants) // 2:
                _test_hook("mid")
            local = copy.deepcopy(model).to(fs.DEVICE)
            c = cdata[cname]
            if mu is None:
                st, ta = fs.local_train(local, c["X_train"], c["y_train"], args.local_epochs, cw)
            else:
                st, ta = local_train_prox(local, c["X_train"], c["y_train"], args.local_epochs, cw, mu)
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
            f"({time.time() - tr:.0f}s, elapsed {(elapsed0 + time.time() - t0) / 60:.1f} min)")
        if ckpt_path:
            atomic_torch_save({
                "signature": signature, "round": rnd, "model": _cpu_state(model),
                "best_state": {k: v.cpu() for k, v in best_state.items()},
                "best_acc": best_acc, "best_round": best_round, "hist": hist,
                "total_bytes": total_bytes, "np_generator": rng.bit_generator.state,
                "rng": _rng_snapshot(), "elapsed": elapsed0 + time.time() - t0,
                "resumes": resumes}, ckpt_path)
            _test_hook("ckpt")
            if rnd < args.rounds:
                check_stop(f"{tag} after round {rnd}")
    model.load_state_dict(best_state)
    steps = args.rounds * args.local_epochs * sum(
        int(np.ceil(len(c["X_train"]) / fs.BATCH_SIZE)) for c in cdata.values())
    info = {
        "architecture": arch, "algorithm": "fedavg" if mu is None else "fedprox",
        "fedprox_mu": mu,
        "rounds": args.rounds, "local_epochs": args.local_epochs,
        "client_fraction": fs.CLIENT_FRACTION, "batch_size": fs.BATCH_SIZE, "lr": fs.LR,
        "cnn_channels": list(args.cnn_channels) if arch == "cnn" else None,
        "mlp_hidden": list(args.mlp_hidden) if arch == "mlp" else None,
        "n_params": int(n_params),
        "per_round_val_acc": hist["val_acc"], "per_round_train_acc": hist["train_acc"],
        "per_round_seconds": hist["seconds"],
        "best_round": best_round, "best_val_acc": best_acc,
        "final_val_acc": hist["val_acc"][-1],
        "model_bytes": int(payload),
        "comm_bytes": int(2 * args.rounds * len(names) * payload),
        "comm_bytes_counted": int(total_bytes),
        "n_clients": len(names),
        "gradient_steps": int(steps),
        "train_seconds": elapsed0 + time.time() - t0,
        "resumed_times": resumes,
    }
    if arch == "cnn":
        info.pop("mlp_hidden")
    return model, info


def train_fed_cnn(X, y, clients, class_weights, n_classes, args, seed, tag):
    """Round-1 entry point (used by check_equivalence.py)."""
    if not hasattr(args, "mlp_hidden"):
        args.mlp_hidden = tuple(int(v) for v in DEFAULT_MLP_HIDDEN.split(","))
    return train_fed(X, y, clients, class_weights, n_classes, args, seed, tag, arch="cnn")


def train_central(X, y, clients, class_weights, n_classes, args, seed, tag, arch="cnn",
                  ckpt_path=None, signature=None):
    """Centralized training on the pooled client training rows; checkpoint after
    every epoch (model, optimizer, shuffling generator, early-stopping state)."""
    n_features = X.shape[1]
    tr_idx = np.concatenate([c["train_idx"] for c in clients])
    va_idx = np.concatenate([c["val_idx"] for c in clients])
    Xv, yv = X[va_idx], y[va_idx]
    dev = fs.DEVICE
    Xt = torch.as_tensor(X[tr_idx], dtype=torch.float32, device=dev)
    yt = torch.as_tensor(y[tr_idx], dtype=torch.long, device=dev)
    N = len(tr_idx)

    seed_everything(seed)
    model = build_net(arch, n_features, n_classes, args)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=fs.LR)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32).to(dev))
    gen = torch.Generator().manual_seed(seed)
    bs = fs.BATCH_SIZE
    best_acc, best_epoch, best_state, bad = -1.0, 0, None, 0
    hist = {"val_acc": [], "train_loss": [], "seconds": []}
    start_ep, elapsed0, resumes, finished = 1, 0.0, 0, False
    ck = _load_ckpt(ckpt_path, signature)
    if ck is not None:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        gen.set_state(ck["gen"])
        _rng_restore(ck["rng"])
        best_acc, best_epoch, best_state, bad = ck["best_acc"], ck["best_epoch"], ck["best_state"], ck["bad"]
        best_state = {k: v.to(dev) for k, v in best_state.items()}
        hist = ck["hist"]
        start_ep, elapsed0, resumes = ck["epoch"] + 1, ck["elapsed"], ck.get("resumes", 0) + 1
        finished = ck.get("finished", False)
        log(f"  [{tag}] resuming after epoch {ck['epoch']} (checkpoint {ckpt_path})")
    t0 = time.time()
    log(f"  [{tag}] central {arch.upper()}: {N:,} train rows, {len(va_idx):,} val rows, "
        f"max {args.central_max_epochs} epochs, patience {args.central_patience}, "
        f"{n_params:,} params")
    epochs_run = start_ep - 1
    for ep in range(start_ep, args.central_max_epochs + 1):
        if finished:
            break
        te = time.time()
        model.train()
        perm = torch.randperm(N, generator=gen).to(dev)
        tot, nb = 0.0, 0
        half = (N // bs // 2) * bs
        for lo in range(0, N, bs):
            if lo == half:
                _test_hook("mid")
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
        finished = bad >= args.central_patience or ep == args.central_max_epochs
        if ckpt_path:
            atomic_torch_save({
                "signature": signature, "epoch": ep, "model": _cpu_state(model),
                "opt": opt.state_dict(), "gen": gen.get_state(), "rng": _rng_snapshot(),
                "best_state": {k: v.cpu() for k, v in best_state.items()},
                "best_acc": best_acc, "best_epoch": best_epoch, "bad": bad, "hist": hist,
                "elapsed": elapsed0 + time.time() - t0, "resumes": resumes,
                "finished": finished}, ckpt_path)
            _test_hook("ckpt")
        if bad >= args.central_patience:
            log(f"  [{tag}] early stop: no val improvement for {bad} epochs")
            break
        if ckpt_path and not finished:
            check_stop(f"{tag} after epoch {ep}")
    model.load_state_dict(best_state)
    del Xt, yt
    steps_per_epoch = int(np.ceil(N / bs))
    fed_steps = args.rounds * args.local_epochs * sum(
        int(np.ceil(len(c["train_idx"]) / bs)) for c in clients)
    info = {
        "architecture": arch, "n_params": int(n_params),
        "max_epochs": args.central_max_epochs, "patience": args.central_patience,
        "epochs_run": epochs_run, "best_epoch": best_epoch, "best_val_acc": best_acc,
        "per_epoch_val_acc": hist["val_acc"], "per_epoch_train_loss": hist["train_loss"],
        "per_epoch_seconds": hist["seconds"],
        "batch_size": bs, "lr": fs.LR,
        "cnn_channels": list(args.cnn_channels) if arch == "cnn" else None,
        "mlp_hidden": list(args.mlp_hidden) if arch == "mlp" else None,
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
                     f"(models averaged after every {args.local_epochs} local epochs). The centralized "
                     f"model sees the same rows for at most {args.central_max_epochs} epochs with early "
                     f"stopping (patience {args.central_patience}) on pooled validation accuracy; "
                     f"it ran {epochs_run}. Both keep the checkpoint with the best validation "
                     f"accuracy."),
        },
        "train_seconds": elapsed0 + time.time() - t0,
        "resumed_times": resumes,
    }
    if arch == "cnn":
        info.pop("mlp_hidden")
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
# validation-based choice of s, PR-AUC, per-client test metrics
# ---------------------------------------------------------------------------

def select_strength(val_proba16, yv, prior, classes, grid=SELECTION_GRID):
    """Pre-registered rule (experiments/SELECTION_RULE.md): s* = argmax over the
    grid of macro F1 on the validation rows reweighted to the natural
    training-day class prior; ties go to the smaller s. The validation sample
    is capped per (region, class), so each row gets the weight
    prior[class] / validation_share[class] (weights depend on the true class
    only, i.e. the rows of the confusion matrix are rescaled)."""
    K = len(classes)
    p = val_proba16.astype(np.float32)
    w = row_weights(yv, classes, np.asarray(prior, float))
    table = {}
    best_s, best_f1 = None, -1.0
    for s in grid:
        pred = (p * (prior.astype(np.float32) ** np.float32(s))[None, :]).argmax(1)
        m = metrics_from_cm(confusion(yv, pred, K) * w[:, None], classes)
        table[f"{s:.2f}"] = {"macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                             "false_alarm_rate": m["false_alarm_rate"],
                             "avg_attack_recall": m["avg_attack_recall"]}
        if m["macro_f1"] > best_f1:      # strict '>' : ties keep the smaller s
            best_s, best_f1 = s, m["macro_f1"]
    pred = (p * (prior.astype(np.float32) ** np.float32(best_s))[None, :]).argmax(1)
    cm = confusion(yv, pred, K)
    return {
        "rule": "argmax over s in {0,0.1,...,1} of macro F1 on the validation rows reweighted to "
                "the natural training-day class prior; ties to the smaller s; fixed before any "
                "test scoring",
        "grid": list(grid), "s_star": float(best_s),
        "val_macro_f1_at_s_star": float(best_f1),
        "val_by_s": table,
        "val_rows": int(len(yv)),
        "val_class_counts": {c: int((yv == i).sum()) for i, c in enumerate(classes)},
        "val_class_weight_to_prior": {c: float(v) for c, v in zip(classes, w)},
        "val_metrics_at_s_star_reweighted": metrics_from_cm(cm * w[:, None], classes),
        "val_metrics_at_s_star_unweighted": metrics_from_cm(cm, classes),
    }


def pr_auc(proba16, y, classes):
    """One-vs-rest average precision per class from the saved probabilities
    (ranking by the class probability; a per-class constant factor such as
    pi**s does not change a one-vs-rest ranking), and the macro mean."""
    from sklearn.metrics import average_precision_score
    p = proba16.astype(np.float32)
    out = {}
    for k, c in enumerate(classes):
        pos = (y == k)
        out[c] = float(average_precision_score(pos, p[:, k])) if pos.any() and (~pos).any() else None
    vals = [v for v in out.values() if v is not None]
    return {"per_class": out, "macro": float(np.mean(vals)) if vals else None,
            "attack_vs_benign": (float(average_precision_score(
                y != list(classes).index("benign"),
                1.0 - p[:, list(classes).index("benign")]))
                if (y != list(classes).index("benign")).any() else None)}


def per_client_metrics(proba16, y, region, prior, s, classes, min_rows=200):
    """Test metrics per region cell (the FL clients) at strength s."""
    K = len(classes)
    pred = (proba16.astype(np.float32) * (prior.astype(np.float32) ** np.float32(s))[None, :]).argmax(1)
    out = {}
    df = pd.DataFrame({"r": region, "i": np.arange(len(y))})
    for r, idx in df.groupby("r").indices.items():
        if len(idx) < min_rows:
            continue
        m = metrics_from_cm(confusion(y[idx], pred[idx], K), classes)
        out[str(r)] = {"n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                       "false_alarm_rate": m["false_alarm_rate"],
                       "avg_attack_recall": m["avg_attack_recall"],
                       "false_alarms": m["false_alarms"]}
    return out


def alert_counts(metrics, keys):
    """False-alert counts at the listed strengths (from the confusion matrices)."""
    out = {}
    for name, s in keys.items():
        m = metrics.get(f"{s:.2f}", {}).get("pooled")
        if m is None:
            continue
        cm = np.array(m["confusion_matrix"], float)
        b = [i for i, c in enumerate(m["per_class"]) if c == "benign"][0]
        alerts = cm.sum() - cm[:, b].sum()
        out[name] = {"s": float(s), "benign_rows": int(cm[b].sum()),
                     "false_alerts": int(m["false_alarms"]),
                     "false_alerts_per_million_benign": float(m["false_alarms"] / max(cm[b].sum(), 1) * 1e6),
                     "alerts_total": int(alerts),
                     "true_alerts": int(alerts - m["false_alarms"]),
                     "missed_attacks": int(m["missed_attacks"])}
    return out


def test_summary(metrics, s_star, strengths):
    """Test metrics at s*, 0, 0.5, 1 and (observation only) the test-optimal s."""
    def brief(s):
        m = metrics.get(f"{s:.2f}", {}).get("pooled")
        return None if m is None else {k: m[k] for k in (
            "accuracy", "macro_f1", "false_alarm_rate", "missed_attack_rate",
            "avg_attack_recall", "alert_real_share", "false_alarms", "missed_attacks")}
    best = max(strengths, key=lambda s: (metrics[f"{s:.2f}"]["pooled"]["macro_f1"], -s))
    return {"s_star": brief(s_star), "s0": brief(0.0), "s05": brief(0.5), "s1": brief(1.0),
            "test_optimal_s_observation_only": {
                "s": float(best), "macro_f1": metrics[f"{best:.2f}"]["pooled"]["macro_f1"],
                "note": "reported as an observation; never used to choose anything"}}


# ---------------------------------------------------------------------------
# main driver
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-files", nargs="+", default=None, help="Labelled training CSVs")
    ap.add_argument("--holdout", default=None, help="Labelled held-out test CSV")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=None,
                    help="Folder for the one-off passes (stats, training sample, test arrays); "
                         "default <out-dir>/cache. Share it between out-dirs to do them once.")
    ap.add_argument("--variant", default="corrected",
                    help="corrected | legacy | prefloor | loo:Cx | add:Cx | set:C1+C3 "
                         "(needs multi-variant files unless 'corrected'); see variants.py")
    ap.add_argument("--protocol", choices=("v1", "v2"), default="v2",
                    help="v2: aircraft-disjoint validation, scaler / class weights on training "
                         "rows only, s* chosen on validation. v1: round-1 behaviour")
    ap.add_argument("--max-total-rows", type=int, default=10_000_000)
    ap.add_argument("--legacy-sampler", action="store_true",
                    help="Draw the training sample with federated_simulation.load_data semantics "
                         "(biased towards the last chunks of the last file) instead of the "
                         "uniform sampler")
    ap.add_argument("--seeds", default=None, help="default 1,2,3,4,5 (1,2 with --quick)")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help=f"comma list from {MODELS}")
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
    ap.add_argument("--mlp-hidden", default=DEFAULT_MLP_HIDDEN,
                    help="hidden sizes of the MLP baseline (default 190,44: 11,334 parameters "
                         "on 13 features, the CNN's count)")
    ap.add_argument("--fedprox-mu", type=float, default=None,
                    help="FedProx proximal coefficient for the federated models "
                         "(omit for FedAvg; 0 runs the FedProx code path and equals FedAvg)")
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
    ap.add_argument("--stop-flag", default=os.environ.get("REV2_STOP_FLAG"),
                    help="When this file exists, stop after the current round / epoch / run "
                         f"(exit code {EXIT_STOPPED}); rerun the same command to continue")
    ap.add_argument("--no-checkpoints", action="store_true")
    ap.add_argument("--prepare-only", action="store_true",
                    help="Only do the one-off passes for this variant / sampler (file stats, "
                         "training sample, test arrays) and exit")
    ap.add_argument("--test-stop-after-units", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--test-kill-after-units", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--quick", action="store_true",
                    help="Smoke test: rounds=2, central epochs=2, 20 trees, seeds 1,2, "
                         "strengths 0,0.5,1")
    args = ap.parse_args(argv)
    if args.quick:
        if args.rounds == 50:
            args.rounds = 2
        if args.central_max_epochs == 20:
            args.central_max_epochs = 2
        if args.lgbm_n_estimators == 1000:
            args.lgbm_n_estimators = 20
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
    args.mlp_hidden = tuple(int(c) for c in args.mlp_hidden.split(","))
    try:
        args.legacy_set = V.parse_variant(args.variant)
    except ValueError as e:
        ap.error(str(e))
    args.variant = V.variant_name(args.legacy_set)
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
    atomic_npz(path, proba=proba.astype(np.float16))


def write_holdout_labels(out_dir, y, day, unseen, classes, info, main_days, day_counts,
                         name="holdout_labels.npz", region=None, rowpos=None):
    path = os.path.join(out_dir, name)
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        if "unseen" in z.files and len(z["y"]) == len(y) and np.array_equal(z["y"], y):
            return path
        log(f"NOTE: rewriting {path} (holdout changed or older format)")
    meta = dict(info, main_days=main_days, day_counts=day_counts)
    extra = {}
    if region is not None:
        extra["region"] = region
    if rowpos is not None:
        extra["rowpos"] = rowpos
    atomic_npz(path, y=y, day=np.asarray(day, dtype="U10"), unseen=unseen,
               classes=np.array(classes), info=np.array(json.dumps(meta)), **extra)
    return path


def holdout_section(info, y, classes, main_days, day_counts):
    return {"path": info["path"], "rows": int(len(y)),
            "main_days": main_days, "day_counts": day_counts,
            "support": {c: int((y == i).sum()) for i, c in enumerate(classes)},
            "blank_share_by_class": info.get("blank_share_by_class"),
            "rules": info.get("rules"),
            "overlap": info.get("overlap"),
            "variant": info.get("variant"),
            "rows_in_file": info.get("rows_in_file"),
            "rows_masked_by_legacy_filter": info.get("rows_masked_by_legacy_filter"),
            "never_used_for_selection": True}


SELECTION_RULE = {
    "fed_cnn": "best FedAvg round by unweighted accuracy on the pooled validation rows, "
               "argmax of the raw softmax (s=0)",
    "central_cnn": "best epoch by unweighted accuracy on the pooled validation rows, argmax of "
                   "the raw softmax (s=0); early stop after `patience` epochs without improvement",
    "lgbm": "early stopping on multi_logloss of the pooled validation rows weighted with the "
            "balanced class weights",
}
SELECTION_RULE["fed_mlp"] = SELECTION_RULE["fed_cnn"]
SELECTION_RULE["central_mlp"] = SELECTION_RULE["central_cnn"]


def main(argv=None):
    global LOG_FH, STOP_FLAG
    args = parse_args(argv)
    out = os.path.abspath(args.out_dir)
    for sub in ("runs", "probs", "models", "checkpoints"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    args.cache_dir = os.path.abspath(args.cache_dir or os.path.join(out, "cache"))
    os.makedirs(args.cache_dir, exist_ok=True)
    STOP_FLAG = args.stop_flag
    _TEST["stop_after"], _TEST["kill_after"] = args.test_stop_after_units, args.test_kill_after_units
    LOG_FH = open(os.path.join(out, "run_experiments.log"), "a")
    log("=" * 70)
    log("run_experiments.py " + " ".join(sys.argv[1:] if argv is None else argv))
    dev = setup_device(args.device)
    log(f"device={dev}  versions={versions()}  threads={torch.get_num_threads()}")
    try:
        if args.holdout_only:
            return holdout_only(args, out)
        return train_all(args, out)
    except StopRequested as e:
        log(f"STOPPED on request ({e}). Finished work is saved; rerun the same command to continue.")
        return EXIT_STOPPED


def _fit_scaler_v2(Xraw, tr_idx):
    """StandardScaler fitted on the training rows only (float64 statistics)."""
    scaler = StandardScaler()
    step = 2_000_000
    for lo in range(0, len(tr_idx), step):
        scaler.partial_fit(Xraw[tr_idx[lo:lo + step]].astype(np.float64))
    return scaler


def _transform_blocks(scaler, Xraw, step=2_000_000):
    out = np.empty(Xraw.shape, np.float32)
    for lo in range(0, len(Xraw), step):
        out[lo:lo + step] = scaler.transform(Xraw[lo:lo + step].astype(np.float64)).astype(np.float32)
    return out


def train_all(args, out):
    todo = [(f, s, m) for f in args.fs_list for s in args.seed_list for m in args.model_list
            if args.force or not os.path.exists(os.path.join(out, "runs", run_id(m, f, s) + ".json"))]
    n_total = len(args.fs_list) * len(args.seed_list) * len(args.model_list)
    log(f"{n_total} run(s) requested, {n_total - len(todo)} already done, {len(todo)} to do")
    if not todo and not args.prepare_only:
        return 0
    check_stop("before loading data")
    ls = args.legacy_set
    design = V.design_of(ls)
    v2 = args.protocol == "v2"
    log(f"variant={args.variant} (legacy corrections: {sorted(ls) or 'none'}; design {design})  "
        f"protocol={args.protocol}  sampler={args.sampler}")

    cache = args.cache_dir
    stats, train_icao, stats_path = load_train_stats(args.train_files, cache)
    stats_key = os.path.basename(stats_path)
    df, sample_meta = load_training_sample(args.train_files, args.max_total_rows, cache,
                                           args.sampler, stats, ls)
    dstats = stats["designs"][design]
    classes = sorted(df["attack_type"].astype(str).unique())
    y_all = df["attack_type"].astype(str).map({c: i for i, c in enumerate(classes)}).values.astype(np.int64)
    K = len(classes)
    if "benign" not in classes:
        raise ValueError("no 'benign' class in training data")
    log(f"Classes: {classes}")
    prior, prior_src = natural_prior(dstats, classes, prior_json=args.prior_json)
    H = load_holdout(args.holdout, classes, cache, train_icao, stats_key, legacy_set=ls)
    Xh_all, yh, dayh, unseen, hinfo = H["X"], H["y"], H["day"], H["unseen"], H["info"]
    main_days, day_counts = holdout_days(dayh)
    log(f"Holdout days (>= {MIN_DAY_SHARE:.0%} of rows): {main_days}; all: {day_counts}")
    write_holdout_labels(out, yh, dayh, unseen, classes, hinfo, main_days, day_counts,
                         region=H["region"], rowpos=H["rowpos"])
    HC = None
    if ls:      # second test set: the fully corrected test sample (all rows)
        HC = load_holdout(args.holdout, classes, cache, train_icao, stats_key, legacy_set=frozenset())
        HC["main_days"], HC["day_counts"] = holdout_days(HC["day"])
        write_holdout_labels(out, HC["y"], HC["day"], HC["unseen"], classes, HC["info"],
                             HC["main_days"], HC["day_counts"], name="holdout_labels_corrected.npz",
                             region=HC["region"], rowpos=HC["rowpos"])
    del train_icao
    if args.prepare_only:
        log(f"prepare-only: stats, training sample ({args.sampler}, design {design}) and test "
            f"arrays are cached in {cache}")
        return 0

    regions = df["region_cell"].astype(str).values
    icao = None
    if "icao24" in df.columns:      # stored as the integer address by the samplers
        ic = df["icao24"]
        icao = (np.char.mod("%06x", ic.to_numpy()) if pd.api.types.is_integer_dtype(ic)
                else ic.astype(str).values)
    if v2 and icao is None:
        raise SystemExit("--protocol v2 needs the icao24 column in the training files")
    # v1: balanced class weights from all loaded labels (as federated_simulation.main)
    counts_all = np.bincount(y_all, minlength=K)
    class_weights_v1 = len(y_all) / (K * counts_all)
    vers = versions()
    eval_strengths = sorted(set(args.strength_list) | {0.0, 0.5, 1.0})

    for fset in args.fs_list:
        jobs = [(s, m) for f, s, m in todo if f == fset]
        if not jobs:
            continue
        cols = FEATURE_SETS[fset]
        log(f"--- feature set '{fset}' ({len(cols)} features) ---")
        Xraw_df = prep_subset(df, cols)
        col_idx = [fs.FEATURE_COLS.index(c) for c in cols]
        Xh_raw = np.ascontiguousarray(Xh_all[:, col_idx])
        XhC_raw = np.ascontiguousarray(HC["X"][:, col_idx]) if HC is not None else None
        if not v2:
            scaler = StandardScaler()
            X = scaler.fit_transform(Xraw_df).astype(np.float32)   # scaler on pooled sample, as main()
        Xraw = Xraw_df.values.astype(np.float32)
        del Xraw_df
        gc.collect()

        for seed in sorted(set(s for s, _ in jobs)):
            split_report = None
            if v2:
                clients, dropped, split_report = make_clients_v2(regions, icao, y_all, classes,
                                                                 args.min_client_rows, seed)
            else:
                clients, dropped = make_clients(regions, args.min_client_rows, seed)
            if len(clients) < 2:
                raise SystemExit("fewer than 2 usable clients; lower --min-client-rows")
            csum = client_summary(clients, y_all, classes)
            atomic_json({"seed": seed, "min_client_rows": args.min_client_rows,
                         "protocol": args.protocol, "variant": args.variant,
                         "n_clients": len(clients), "dropped_regions": dropped, "clients": csum,
                         "split_report": split_report},
                        os.path.join(out, "runs", f"clients__seed{seed}.json"))
            n_tr = sum(len(c["train_idx"]) for c in clients)
            n_va = sum(len(c["val_idx"]) for c in clients)
            log(f"seed {seed}: {len(clients)} clients ({len(dropped)} regions dropped), "
                f"{n_tr:,} train rows, {n_va:,} validation rows")
            tr_all = np.concatenate([c["train_idx"] for c in clients])
            va_all = np.concatenate([c["val_idx"] for c in clients])
            if n_va == 0:
                raise SystemExit("validation split is empty")
            if v2:
                log(f"  aircraft-disjoint split: {split_report['aircraft_in_validation']:,} of "
                    f"{split_report['aircraft_in_sample']:,} aircraft in validation "
                    f"({split_report['val_row_share']:.1%} of rows); clients losing a class: "
                    f"train {split_report['clients_losing_a_class_in_train']}, "
                    f"val {split_report['clients_losing_a_class_in_val']}")
                counts = np.bincount(y_all[tr_all], minlength=K)
                class_weights = len(tr_all) / (K * np.maximum(counts, 1))
                scaler = _fit_scaler_v2(Xraw, tr_all)
                X = _transform_blocks(scaler, Xraw)
                Xh = _transform_blocks(scaler, Xh_raw)
                XhC = _transform_blocks(scaler, XhC_raw) if HC is not None else None
            else:
                counts, class_weights = counts_all, class_weights_v1
                Xh = scaler.transform(pd.DataFrame(Xh_raw.astype(np.float64), columns=cols)).astype(np.float32)
                XhC = (scaler.transform(pd.DataFrame(XhC_raw.astype(np.float64), columns=cols)).astype(np.float32)
                       if HC is not None else None)
            log("Class weights (balanced): " + ", ".join(f"{c}={w:.3f}" for c, w in zip(classes, class_weights)))
            yv = y_all[va_all]

            for s2, model_name in jobs:
                if s2 != seed:
                    continue
                check_stop("between runs")
                rid = run_id(model_name, fset, seed)
                tag = rid
                log(f">>> {rid}")
                t0 = time.time()
                ckpt = None if args.no_checkpoints else os.path.join(out, "checkpoints", rid + ".ckpt")
                signature = json.dumps({
                    "rid": rid, "variant": args.variant, "protocol": args.protocol,
                    "sampler": args.sampler, "rows": int(len(y_all)), "n_tr": int(n_tr),
                    "rounds": args.rounds, "local_epochs": args.local_epochs,
                    "central_max_epochs": args.central_max_epochs,
                    "patience": args.central_patience, "cnn": list(args.cnn_channels),
                    "mlp": list(args.mlp_hidden), "mu": args.fedprox_mu,
                    "min_client_rows": args.min_client_rows, "features": cols}, sort_keys=True)
                if model_name in ("fed_cnn", "fed_mlp"):
                    model, info = train_fed(X, y_all, clients, class_weights, K, args, seed, tag,
                                            arch=ARCH[model_name], ckpt_path=ckpt, signature=signature)
                elif model_name in ("central_cnn", "central_mlp"):
                    model, info = train_central(X, y_all, clients, class_weights, K, args, seed, tag,
                                                arch=ARCH[model_name], ckpt_path=ckpt,
                                                signature=signature)
                else:
                    model, info = train_lgbm(Xraw, y_all, clients, class_weights, K, args, seed, tag)
                info["selection_rule"] = SELECTION_RULE[model_name]

                # ---- validation probabilities and the choice of s (before any test scoring)
                if model_name == "lgbm":
                    predict = lambda A_raw, A_scaled: model.predict_proba(A_raw)  # noqa: E731
                else:
                    predict = lambda A_raw, A_scaled: softmax_np(predict_logits(model, A_scaled))  # noqa: E731
                val_proba16 = predict(Xraw[va_all], X[va_all]).astype(np.float16)
                selection = select_strength(val_proba16, yv, prior, classes)
                selection["protocol"] = args.protocol
                selection["validation_split"] = (
                    "aircraft-disjoint (protocol v2)" if v2
                    else "random 80/20 rows within clients (protocol v1)")
                selection["fixed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                s_star = selection["s_star"]
                wv = np.array([selection["val_class_weight_to_prior"][c] for c in classes])
                atomic_npz(os.path.join(out, "probs", rid + "__val.npz"), proba=val_proba16,
                           y=yv.astype(np.int8), class_weight_to_prior=wv, prior=prior,
                           classes=np.array(classes))
                atomic_json({"run_id": rid, "variant": args.variant, **selection},
                            os.path.join(out, "runs", rid + ".selection"))
                log(f"  [{tag}] s* = {s_star:g} (validation macro F1 "
                    f"{selection['val_macro_f1_at_s_star']:.4f}, prior-reweighted); now scoring the test sample")

                # ---- test scoring
                tp = time.time()
                if model_name == "lgbm":
                    gain = np.array(info.pop("_gain"), float)
                    info["feature_importance_gain_share"] = {
                        c: float(g / max(gain.sum(), 1e-12)) for c, g in zip(cols, gain)}
                    model.booster_.save_model(os.path.join(out, "models", rid + ".txt.tmp"))
                    os.replace(os.path.join(out, "models", rid + ".txt.tmp"),
                               os.path.join(out, "models", rid + ".txt"))
                else:
                    atomic_torch_save(
                        {"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                         "model_type": ARCH[model_name], "n_features": len(cols),
                         "feature_cols": cols, "class_names": classes,
                         "cnn_channels": list(args.cnn_channels),
                         "mlp_hidden": list(args.mlp_hidden)},
                        os.path.join(out, "models", rid + ".pt"))
                    import joblib
                    joblib.dump(scaler, os.path.join(out, "models", rid + "_scaler.pkl.tmp"))
                    os.replace(os.path.join(out, "models", rid + "_scaler.pkl.tmp"),
                               os.path.join(out, "models", rid + "_scaler.pkl"))
                proba16 = predict(Xh_raw, Xh).astype(np.float16)
                save_probs(os.path.join(out, "probs", rid + ".npz"), proba16)
                strengths_run = sorted(set(eval_strengths) | {s_star})
                metrics = evaluate_strengths(proba16, yh, dayh, main_days, prior,
                                             strengths_run, classes, unseen, args.rho_list)
                extra = {}
                if HC is not None:
                    probaC16 = predict(XhC_raw, XhC).astype(np.float16)
                    save_probs(os.path.join(out, "probs", rid + "__corrtest.npz"), probaC16)
                    extra["metrics_corrected_test"] = evaluate_strengths(
                        probaC16, HC["y"], HC["day"], HC["main_days"], prior, strengths_run, classes,
                        HC["unseen"], args.rho_list)
                    extra["holdout_corrected"] = holdout_section(HC["info"], HC["y"], classes,
                                                                 HC["main_days"], HC["day_counts"])
                    extra["pr_auc_corrected_test"] = pr_auc(probaC16, HC["y"], classes)
                    extra["test_summary_corrected_test"] = test_summary(
                        extra["metrics_corrected_test"], s_star, strengths_run)
                    extra["false_alerts_corrected_test"] = alert_counts(
                        extra["metrics_corrected_test"], {"s_star": s_star, "s0": 0.0, "s05": 0.5, "s1": 1.0})
                    del probaC16
                predict_s = time.time() - tp
                res = {
                    "run_id": rid, "model": model_name, "feature_set": fset, "features": cols,
                    "seed": seed, "classes": classes, "versions": vers,
                    "variant": args.variant, "variant_legacy_corrections": sorted(ls),
                    "protocol": args.protocol, "sampler": args.sampler,
                    "fedprox_mu": args.fedprox_mu if model_name.startswith("fed_") else None,
                    "s_star": s_star, "selection": selection,
                    "config": {k: (list(v) if isinstance(v, (tuple, frozenset)) else v)
                               for k, v in vars(args).items()
                               if k not in ("seed_list", "model_list", "fs_list", "legacy_set",
                                            "test_stop_after_units", "test_kill_after_units")},
                    "data": {
                        "train_files": [os.path.abspath(p) for p in args.train_files],
                        "sampler": sample_meta.get("sampler"),
                        "cap_per_region_class": sample_meta.get("cap_per_region_class"),
                        "max_total_rows": args.max_total_rows,
                        "train_sample_rows": int(len(y_all)),
                        "train_raw_rows": dstats["total_rows"],
                        "train_raw_rows_streamed": sample_meta.get("raw_rows_streamed"),
                        "train_raw_rows_per_file": {os.path.basename(f["file"]): f["rows"]
                                                    for f in dstats["per_file"]},
                        "train_raw_aircraft": stats.get("n_aircraft"),
                        "train_sample_aircraft": sample_meta.get("n_aircraft_in_sample"),
                        "train_sample_composition": sample_meta.get("composition"),
                        "train_stats_cache": stats_path,
                        "train_sample_class_counts": {c: int(n) for c, n in zip(classes, counts_all)},
                        "class_weights": {c: float(w) for c, w in zip(classes, class_weights)},
                        "class_weights_fitted_on": "training rows only" if v2 else "all sampled rows",
                        "scaler_fitted_on": "training rows only" if v2 else "all sampled rows",
                        "natural_prior": {c: float(p) for c, p in zip(classes, prior)},
                        "natural_prior_source": prior_src,
                        "natural_class_counts": {c: int(dstats["class_counts"].get(c, 0)) for c in classes},
                        "n_regions_in_sample": int(len(set(regions))),
                        "min_client_rows": args.min_client_rows,
                        "n_clients": len(clients), "n_dropped_regions": len(dropped),
                        "train_split_rows": int(n_tr), "val_rows": int(n_va),
                        "val_note": ("validation = all sampled rows of ~20% of aircraft (seeded hash of "
                                     "icao24), per client" if v2 else
                                     "validation = per-client 20% held-out split of the training sample"),
                        "split_report": ({k: v for k, v in split_report.items() if k != "class_loss_detail"}
                                         if split_report else None),
                    },
                    "training": info,
                    "holdout": holdout_section(hinfo, yh, classes, main_days, day_counts),
                    "strengths": strengths_run,
                    "rhos": args.rho_list,
                    "metrics": metrics,
                    "test_summary": test_summary(metrics, s_star, strengths_run),
                    "pr_auc": pr_auc(proba16, yh, classes),
                    "false_alerts": alert_counts(metrics, {"s_star": s_star, "s0": 0.0, "s05": 0.5, "s1": 1.0}),
                    "per_client_test": {"s": s_star, "min_rows": 200,
                                        "clients": per_client_metrics(proba16, yh, H["region"], prior,
                                                                      s_star, classes)},
                    "predict_seconds": predict_s,
                    "wall_seconds": info["train_seconds"] + (time.time() - tp),
                    "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "device": str(fs.DEVICE),
                }
                res.update(extra)
                if model_name.startswith("fed_"):
                    res["clients"] = csum
                atomic_json(res, os.path.join(out, "runs", rid + ".json"))
                json.load(open(os.path.join(out, "runs", rid + ".json")))   # verify it parses
                if ckpt and os.path.exists(ckpt):
                    os.remove(ckpt)
                m1 = metrics["1.00"]["pooled"]
                m0 = metrics["0.00"]["pooled"]
                ms = metrics[f"{s_star:.2f}"]["pooled"]
                log(f"<<< {rid} done in {res['wall_seconds'] / 60:.1f} min | "
                    f"s=0: acc {m0['accuracy']:.4f} F1 {m0['macro_f1']:.4f} "
                    f"FAR {m0['false_alarm_rate']:.4f} | s=1: acc "
                    f"{m1['accuracy']:.4f} F1 {m1['macro_f1']:.4f} FAR {m1['false_alarm_rate']:.4f} "
                    f"atk rec {m1['avg_attack_recall']:.4f} | s*={s_star:g}: F1 {ms['macro_f1']:.4f}")
                del model, proba16, val_proba16
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            if v2:
                del X, Xh, XhC
                gc.collect()
        if not v2:
            del X
        del Xraw, Xh_raw
        gc.collect()
    log("all requested runs finished")
    return 0


def holdout_only(args, out):
    """Recompute metrics for existing runs without retraining (the run's own
    test sample; `metrics_corrected_test` is recomputed too when its saved
    probabilities and labels exist)."""
    lab_path = os.path.join(out, "holdout_labels.npz")
    H = None
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
                ls = frozenset(res.get("variant_legacy_corrections", []))
                prior = np.array([res["data"]["natural_prior"][c] for c in classes])
                ppath = os.path.join(out, "probs", rid + ".npz")
                need_rescore = args.repredict or not os.path.exists(ppath)
                if H is None and (need_rescore or need_labels):
                    spath = res["data"].get("train_stats_cache")
                    train_icao = set()
                    for f in res["data"]["train_files"]:
                        ip = os.path.join(args.cache_dir,
                                          f"filestats_{os.path.basename(f)}.{os.path.getsize(f)}.icao24.txt.gz"
                                          ) if os.path.exists(f) else None
                        if not ip or not os.path.exists(ip):
                            raise SystemExit(f"training icao24 cache for {f} not found in "
                                             f"{args.cache_dir} (needed for the unseen-aircraft subset)")
                        with gzip.open(ip, "rt") as fh:
                            train_icao |= set(fh.read().split())
                    H = load_holdout(args.holdout, classes, args.cache_dir, train_icao,
                                     os.path.basename(spath), legacy_set=ls)
                    del train_icao
                    main_days, day_counts = holdout_days(H["day"])
                    write_holdout_labels(out, H["y"], H["day"], H["unseen"], classes, H["info"],
                                         main_days, day_counts, region=H["region"], rowpos=H["rowpos"])
                    need_labels = False
                if need_rescore:
                    if not args.holdout:
                        raise SystemExit(f"{rid}: no saved probabilities; pass --holdout to re-score")
                    cols = res["features"]
                    Xr = np.ascontiguousarray(H["X"][:, [fs.FEATURE_COLS.index(c) for c in cols]])
                    if m == "lgbm":
                        import lightgbm as lgb
                        booster = lgb.Booster(model_file=os.path.join(out, "models", rid + ".txt"))
                        proba = booster.predict(Xr)
                    else:
                        import joblib
                        scaler = joblib.load(os.path.join(out, "models", rid + "_scaler.pkl"))
                        payload = torch.load(os.path.join(out, "models", rid + ".pt"),
                                             map_location=fs.DEVICE)
                        if payload.get("model_type", "cnn") == "mlp":
                            model = fs.build_model("mlp", payload["n_features"], len(classes),
                                                   mlp_hidden_sizes=tuple(payload["mlp_hidden"]))
                        else:
                            model = fs.build_model("cnn", payload["n_features"], len(classes),
                                                   cnn_channels=tuple(payload["cnn_channels"]))
                        model.load_state_dict(payload["model_state_dict"])
                        if res.get("protocol", "v1") == "v2":
                            Xs = scaler.transform(Xr.astype(np.float64))
                        else:
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
                strengths = sorted(set(args.strength_list) | {0.0, 0.5, 1.0}
                                   | ({res["s_star"]} if res.get("s_star") is not None else set()))
                res["metrics"] = evaluate_strengths(proba16, yh_l, day_l, hmeta["main_days"], prior,
                                                    strengths, classes, unseen_l, args.rho_list)
                res["pr_auc"] = pr_auc(proba16, yh_l, classes)
                if res.get("s_star") is not None:
                    res["test_summary"] = test_summary(res["metrics"], res["s_star"], strengths)
                cl, cp = (os.path.join(out, "holdout_labels_corrected.npz"),
                          os.path.join(out, "probs", rid + "__corrtest.npz"))
                if os.path.exists(cl) and os.path.exists(cp) and not args.repredict:
                    zc = np.load(cl, allow_pickle=False)
                    cmeta = json.loads(str(zc["info"]))
                    res["metrics_corrected_test"] = evaluate_strengths(
                        np.load(cp)["proba"], zc["y"], zc["day"], cmeta["main_days"], prior, strengths,
                        classes, zc["unseen"], args.rho_list)
                res["strengths"] = strengths
                res["rhos"] = args.rho_list
                res["holdout"] = holdout_section(hmeta, yh_l, classes, hmeta["main_days"],
                                                 hmeta["day_counts"])
                res["holdout_only_updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
                atomic_json(res, jpath)
                done += 1
                m1 = res["metrics"]["1.00"]["pooled"]
                log(f"{rid}: metrics recomputed for {len(strengths)} strengths "
                    f"(s=1: F1 {m1['macro_f1']:.4f})")
    log(f"holdout-only: {done} run(s) updated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
