"""
sampling.py
-------------------------------------------------------------------
Training-data passes used by run_experiments.py.

train_file_stats(paths)
    One chunked pass over region_cell, attack_type and icao24 of every
    training file: raw row counts per file and per class (the natural prior),
    distinct region cells, and the set of training aircraft addresses (for
    the test-set overlap analysis).

uniform_sample(paths, max_total_rows, seed)
    Uniform per-(region_cell, attack_type) sampler. Same cap formula, columns
    and dtypes as federated_simulation.load_data, but every row of a bucket
    has the same chance of being kept regardless of which file or chunk it
    came from (bottom-k on seeded uniform random keys, streamed).

Why not load_data: for a capped bucket, load_data re-draws sample(n=cap) from
(kept rows + new chunk) after every chunk, so a row from an early chunk must
survive every later re-draw. Rows from early chunks and early files decay
geometrically and the sample of a capped bucket is dominated by the last
chunks of the last file. It also stops reading once the global row budget is
reached, skipping later files. experiments/test_sampler.py demonstrates both
samplers on synthetic files.
"""

import time

import numpy as np
import pandas as pd

import federated_simulation as fs

N_CLASSES_GUESS = 6     # as load_data: used only for cap sizing


def day_of(df):
    """UTC date of lastposupdate (fallback time) as YYYY-MM-DD strings."""
    if "lastposupdate" in df:
        ts = pd.to_numeric(df["lastposupdate"], errors="coerce")
        if "time" in df:
            ts = ts.fillna(pd.to_numeric(df["time"], errors="coerce"))
    elif "time" in df:
        ts = pd.to_numeric(df["time"], errors="coerce")
    else:
        return np.full(len(df), "unknown", dtype="U10")
    d = pd.to_datetime(ts, unit="s", utc=True).dt.strftime("%Y-%m-%d").fillna("unknown")
    return np.asarray(d.to_numpy(dtype=str), dtype="U10")


def train_file_stats(paths, chunksize=2_000_000, log=print):
    per_file = []
    class_counts = {}
    regions = set()
    icao = set()
    for p in paths:
        t0 = time.time()
        header = pd.read_csv(p, nrows=0).columns
        cols = [c for c in ("region_cell", "attack_type", "icao24") if c in header]
        n = 0
        fc = {}
        for ch in pd.read_csv(p, usecols=cols, chunksize=chunksize, low_memory=False):
            n += len(ch)
            vc = ch["attack_type"].astype(str).value_counts()
            for k, v in vc.items():
                fc[k] = fc.get(k, 0) + int(v)
            regions.update(ch["region_cell"].astype(str).unique())
            if "icao24" in ch:
                icao.update(ch["icao24"].dropna().astype(str).unique())
        for k, v in fc.items():
            class_counts[k] = class_counts.get(k, 0) + v
        per_file.append({"file": p, "rows": n, "class_counts": fc})
        log(f"  stats: {p.split('/')[-1]}: {n:,} rows ({(time.time() - t0) / 60:.1f} min)")
    return {
        "per_file": per_file,
        "total_rows": int(sum(f["rows"] for f in per_file)),
        "class_counts": class_counts,
        "regions": sorted(regions),
        "n_regions": len(regions),
        "n_aircraft": len(icao),
    }, icao


def uniform_sample(paths, max_total_rows, regions, classes, seed=42, chunksize=200_000,
                   log=print, compact_every=None):
    """Bottom-k per (region_cell, attack_type) bucket: each row gets a uniform
    random key from one seeded stream; a bucket keeps its `cap` smallest keys.
    Exact: a row rejected against a bucket's current k-th smallest key can never
    be among the final k smallest. Returns (df, info); df has load_data's
    columns plus `_src` (index of the source file)."""
    n_regions = max(1, len(regions))
    cap = max(500, max_total_rows // (n_regions * N_CLASSES_GUESS))
    log(f"  uniform sampler: {n_regions:,} regions, cap {cap:,} rows per (region, class) "
        f"(targeting a global total near {max_total_rows:,})")
    usecols = fs.FEATURE_COLS + ["region_cell", "attack_type", "icao24", "time", "lastposupdate"]
    reg_index = pd.Index([str(r) for r in regions])
    cls_index = pd.Index([str(c) for c in classes])
    K = len(cls_index)
    thr = np.full(len(reg_index) * K, np.inf)
    rng = np.random.default_rng(seed)
    kept = None
    buf, buf_rows = [], 0
    compact_every = compact_every or max(1_000_000, max_total_rows // 2)
    seen = 0

    def compact(kept, buf):
        big = pd.concat(([kept] if kept is not None else []) + buf, ignore_index=True)
        order = np.lexsort((big["_k"].values, big["_b"].values))
        big = big.iloc[order].reset_index(drop=True)
        b = big["_b"].values
        _, starts, inv = np.unique(b, return_index=True, return_inverse=True)
        rank = np.arange(len(b)) - starts[inv]
        big = big[rank < cap].reset_index(drop=True)
        b = big["_b"].values
        ub, st, cnt = np.unique(b, return_index=True, return_counts=True)
        full = cnt >= cap
        last = st + cnt - 1
        thr[ub[full]] = big["_k"].values[last[full]]
        return big

    for fi, p in enumerate(paths):
        t0 = time.time()
        header = pd.read_csv(p, nrows=0).columns
        cols = [c for c in usecols if c in header]
        missing = set(usecols) - set(cols)
        if missing:
            log(f"  WARNING: {p} is missing expected columns: {missing}")
        for ch in pd.read_csv(p, usecols=cols, chunksize=chunksize, low_memory=False):
            seen += len(ch)
            keys = rng.random(len(ch))
            ri = reg_index.get_indexer(ch["region_cell"].astype(str))
            ci = cls_index.get_indexer(ch["attack_type"].astype(str))
            if (ri < 0).any() or (ci < 0).any():
                raise ValueError("region or class not seen in the stats pass; stale stats cache?")
            bucket = ri * K + ci
            m = keys < thr[bucket]
            if not m.any():
                continue
            ch = ch[m].copy()
            num = [c for c in fs.FEATURE_COLS if c in ch.columns
                   and pd.api.types.is_numeric_dtype(ch[c])]
            if num:
                ch[num] = ch[num].astype("float32")
            ch = ch.drop(columns=["region_cell", "attack_type"])
            ch["_b"] = bucket[m].astype(np.int32)
            ch["_k"] = keys[m]
            ch["_src"] = np.int8(fi)
            buf.append(ch)
            buf_rows += len(ch)
            if buf_rows >= compact_every:
                kept = compact(kept, buf)
                buf, buf_rows = [], 0
        log(f"  uniform sampler: {p.split('/')[-1]} done ({(time.time() - t0) / 60:.1f} min, "
            f"{seen:,} raw rows so far)")
    if buf or kept is None:
        kept = compact(kept, buf)
    b = kept["_b"].values
    kept.insert(0, "attack_type", np.asarray(cls_index)[b % K])
    kept.insert(0, "region_cell", np.asarray(reg_index)[b // K])
    kept = kept.drop(columns=["_b", "_k"])
    info = {"sampler": "uniform_bottom_k", "cap_per_region_class": int(cap),
            "raw_rows_streamed": int(seen), "seed": seed}
    log(f"  uniform sampler: {seen:,} raw rows -> {len(kept):,} rows")
    return kept, info


def sample_composition(df, paths, classes):
    """Row counts of the final sample per source file and per training day,
    overall and per class."""
    out = {}
    at = df["attack_type"].astype(str).values
    if "_src" in df:
        src = df["_src"].values
        out["per_file"] = {}
        for i, p in enumerate(paths):
            m = src == i
            out["per_file"][p.split("/")[-1]] = {
                "rows": int(m.sum()),
                "by_class": {c: int((m & (at == c)).sum()) for c in classes}}
    day = day_of(df)
    out["per_day"] = {}
    for d in sorted(set(day.tolist())):
        m = day == d
        out["per_day"][d] = {"rows": int(m.sum()),
                             "by_class": {c: int((m & (at == c)).sum()) for c in classes}}
    return out
