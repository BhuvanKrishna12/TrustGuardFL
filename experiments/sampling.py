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


# ===========================================================================
# Revision 2: per-file cached passes, multi-variant files, stop/resume
# ===========================================================================

import gzip      # noqa: E402
import json      # noqa: E402
import os        # noqa: E402
import pickle    # noqa: E402

import variants as V  # noqa: E402


class StopRequested(Exception):
    """Raised when the STOP flag is seen; finished units are already on disk."""


def atomic_pickle(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=4)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _file_sig(p):
    return [os.path.basename(p), os.path.getsize(p)]


def file_is_mv(path):
    return V.is_mv_columns(pd.read_csv(path, nrows=0).columns)


def _stats_one_file(p, chunksize, stop_check):
    header = pd.read_csv(p, nrows=0).columns
    mv = V.is_mv_columns(header)
    cols = [c for c in ("region_cell", "attack_type", "icao24") if c in header]
    if mv:
        cols += ["attack_type_legacy", "is_physically_implausible", "inter_arrival_s"]
    designs = V.DESIGNS if mv else ("CN",)
    cc = {d: {} for d in designs}
    regs = {d: set() for d in designs}
    rows = {d: 0 for d in designs}
    icao = set()
    n = 0
    for ch in pd.read_csv(p, usecols=cols, chunksize=chunksize, low_memory=False):
        if stop_check and stop_check():
            raise StopRequested("stats pass")
        n += len(ch)
        reg = ch["region_cell"].astype(str).to_numpy()
        if "icao24" in ch:
            icao.update(ch["icao24"].dropna().astype(str).unique())
        if mv:
            dl = V.design_labels_and_alive(ch)
        else:
            dl = {"CN": (ch["attack_type"].astype(str).to_numpy(), np.ones(len(ch), bool))}
        for d in designs:
            at, alive = dl[d]
            rows[d] += int(alive.sum())
            for k, v in pd.Series(at[alive]).value_counts().items():
                cc[d][k] = cc[d].get(k, 0) + int(v)
            regs[d].update(np.unique(reg[alive]).tolist())
    return {"file": p, "sig": _file_sig(p), "mv": bool(mv), "rows": n,
            "class_counts": cc["CN"],
            "designs": {d: {"rows": rows[d], "class_counts": cc[d], "regions": sorted(regs[d])}
                        for d in designs}}, icao


def train_file_stats_cached(paths, cache_dir, chunksize=2_000_000, log=print, stop_check=None):
    """train_file_stats with one cache entry per input file (resumable), and, for
    multi-variant files, the class counts / regions / row counts of every
    sampling design (variants.DESIGNS). For the corrected design ("CN") the
    returned numbers are those of train_file_stats on the v3 file."""
    per_file, icao = [], set()
    for p in paths:
        key = f"{os.path.basename(p)}.{os.path.getsize(p)}"
        jp = os.path.join(cache_dir, f"filestats_{key}.json")
        ip = os.path.join(cache_dir, f"filestats_{key}.icao24.txt.gz")
        if os.path.exists(jp) and os.path.exists(ip):
            st = json.load(open(jp))
            with gzip.open(ip, "rt") as f:
                ic = set(f.read().split())
        else:
            t0 = time.time()
            st, ic = _stats_one_file(p, chunksize, stop_check)
            st["minutes"] = (time.time() - t0) / 60
            with gzip.open(ip + ".tmp", "wt") as f:
                f.write("\n".join(sorted(ic)))
            os.replace(ip + ".tmp", ip)
            atomic_json(st, jp)          # the json is written last: it marks the file as done
            log(f"  stats: {os.path.basename(p)}: {st['rows']:,} rows "
                f"({st['minutes']:.1f} min){' [multi-variant]' if st['mv'] else ''}")
        st["file"] = p
        per_file.append(st)
        icao |= ic
    designs = sorted(set.intersection(*[set(f["designs"]) for f in per_file]))
    out_d = {}
    for d in designs:
        cc, regs = {}, set()
        for f in per_file:
            for k, v in f["designs"][d]["class_counts"].items():
                cc[k] = cc.get(k, 0) + v
            regs.update(f["designs"][d]["regions"])
        out_d[d] = {"total_rows": int(sum(f["designs"][d]["rows"] for f in per_file)),
                    "class_counts": cc, "regions": sorted(regs), "n_regions": len(regs),
                    "per_file": [{"file": f["file"], "rows": f["designs"][d]["rows"],
                                  "class_counts": f["designs"][d]["class_counts"]}
                                 for f in per_file]}
    base = out_d["CN"]
    stats = {"per_file": base["per_file"], "total_rows": base["total_rows"],
             "class_counts": base["class_counts"], "regions": base["regions"],
             "n_regions": base["n_regions"], "n_aircraft": len(icao),
             "mv": all(f["mv"] for f in per_file), "designs": out_d}
    return stats, icao


_DROP_BIT = {"N": 1, "F": 2, "P": 4}


def uniform_sample_mv(paths, max_total_rows, stats, seed=42, chunksize=200_000, log=print,
                      compact_every=None, state_path=None, stop_check=None):
    """uniform_sample for every sampling design at once, resumable per file.

    One seeded key per row (the same stream as uniform_sample). A row is kept
    if it is among the `cap` smallest keys of its (region, class) bucket in AT
    LEAST ONE design, where a design fixes the labels (corrected / legacy) and
    the rows that exist (all / without the legacy-filtered rows, with or
    without the floor). The returned frame is the union; column `_dm` is a
    bit mask over stats['designs'] order (sorted names) telling in which
    designs the row is sampled. So every variant gets an exact uniform capped
    sample of ITS OWN rows and labels; nothing is approximated. For design
    "CN" (and for a plain, non-multi-variant file) the selected rows and their
    order are exactly those of uniform_sample.

    state_path: after each input file the sampler state (kept rows, thresholds,
    RNG state) is written atomically; a rerun continues after the last
    finished file."""
    designs = sorted(stats["designs"])
    mv = bool(stats.get("mv")) and len(designs) > 1
    if not mv:
        designs = ["CN"]
    regions = stats["regions"]
    classes = sorted(set().union(*[set(stats["designs"][d]["class_counts"]) for d in designs]))
    reg_index = pd.Index([str(r) for r in regions])
    cls_index = pd.Index([str(c) for c in classes])
    K = len(cls_index)
    caps = {d: max(500, max_total_rows // (max(1, stats["designs"][d]["n_regions"]) * N_CLASSES_GUESS))
            for d in designs}
    log(f"  uniform sampler: {len(reg_index):,} regions, cap {caps['CN']:,} rows per (region, class) "
        f"(targeting a global total near {max_total_rows:,})"
        + (f"; {len(designs)} sampling designs in one pass" if mv else ""))
    usecols = fs.FEATURE_COLS + ["region_cell", "attack_type", "icao24", "time", "lastposupdate"]
    if mv:
        usecols = usecols + V.MV_READ_COLS
    thr = {d: np.full(len(reg_index) * K, np.inf) for d in designs}
    rng = np.random.default_rng(seed)
    kept = None
    seen = 0
    start_file = 0
    sig = [[_file_sig(p) for p in paths], max_total_rows, seed, designs, caps]
    if state_path and os.path.exists(state_path):
        with open(state_path, "rb") as f:
            st = pickle.load(f)
        if st.get("sig") == json.loads(json.dumps(sig)):
            kept, thr, seen, start_file = st["kept"], st["thr"], st["seen"], st["next_file"]
            rng.bit_generator.state = st["rng_state"]
            log(f"  uniform sampler: resuming after {start_file} finished file(s) "
                f"({seen:,} raw rows already streamed)")
        else:
            log("  uniform sampler: saved state belongs to other inputs, starting over")
    compact_every = compact_every or max(1_000_000, max_total_rows // 2)

    for fi, p in enumerate(paths):
        if fi < start_file:
            continue
        t0 = time.time()
        header = pd.read_csv(p, nrows=0).columns
        cols = [c for c in usecols if c in header]
        missing = set(usecols) - set(cols)
        if missing:
            log(f"  WARNING: {p} is missing expected columns: {missing}")
        buf, buf_rows = [], 0
        for ch in pd.read_csv(p, usecols=cols, chunksize=chunksize, low_memory=False):
            if stop_check and stop_check():
                raise StopRequested("uniform sampling pass (the current file restarts on resume)")
            seen += len(ch)
            keys = rng.random(len(ch))
            ri = reg_index.get_indexer(ch["region_cell"].astype(str))
            ci = cls_index.get_indexer(ch["attack_type"].astype(str))
            if (ri < 0).any() or (ci < 0).any():
                raise ValueError("region or class not seen in the stats pass; stale stats cache?")
            bC = ri * K + ci
            if mv:
                ciL = cls_index.get_indexer(ch["attack_type_legacy"].astype(str))
                if (ciL < 0).any():
                    raise ValueError("legacy class not seen in the stats pass; stale stats cache?")
                bL = ri * K + ciL
                dl = V.design_labels_and_alive(ch)
                al = (np.uint8(1) + np.uint8(2) * dl["CF"][1] + np.uint8(4) * dl["CP"][1]).astype(np.uint8)
                alL = (np.uint8(1) + np.uint8(2) * dl["LF"][1] + np.uint8(4) * dl["LP"][1]).astype(np.uint8)
                sub = V.subsecond_mask(ch)
                m = np.zeros(len(ch), bool)
                for d in designs:
                    b = bC if d[0] == "C" else bL
                    a = ((al if d[0] == "C" else alL) & _DROP_BIT[d[1]]) > 0
                    m |= a & (keys < thr[d][b])
            else:
                m = keys < thr["CN"][bC]
            if not m.any():
                continue
            ch = ch[m].copy()
            num = [c for c in fs.FEATURE_COLS + ["track_message_count_legacy"] if c in ch.columns
                   and pd.api.types.is_numeric_dtype(ch[c])]
            if num:
                ch[num] = ch[num].astype("float32")
            ch = ch.drop(columns=["region_cell", "attack_type"])
            ch["_bC"] = bC[m].astype(np.int32)
            if mv:
                ch = ch.drop(columns=["attack_type_legacy"])
                ch["_bL"] = bL[m].astype(np.int32)
                # alive bits: a row can be alive under corrected labels and dropped under
                # legacy ones or the reverse, so both sets are kept (low / high nibble)
                ch["_al"] = (al[m] | (alL[m] << 4)).astype(np.uint8)
                ch["_sub"] = sub[m]
            else:
                ch["_al"] = np.uint8(1)
            ch["_k"] = keys[m]
            ch["_src"] = np.int8(fi)
            buf.append(ch)
            buf_rows += len(ch)
            if buf_rows >= compact_every:
                kept = _compact_mv(kept, buf, designs, caps, thr, mv)
                buf, buf_rows = [], 0
        if buf or kept is None:
            kept = _compact_mv(kept, buf, designs, caps, thr, mv)
        log(f"  uniform sampler: {p.split('/')[-1]} done ({(time.time() - t0) / 60:.1f} min, "
            f"{seen:,} raw rows so far)")
        if state_path:
            atomic_pickle({"sig": json.loads(json.dumps(sig)), "kept": kept, "thr": thr, "seen": seen,
                           "next_file": fi + 1, "rng_state": rng.bit_generator.state}, state_path)
    bC = kept["_bC"].to_numpy()
    kept.insert(0, "attack_type", np.asarray(cls_index)[bC % K])
    kept.insert(0, "region_cell", np.asarray(reg_index)[bC // K])
    if mv:
        kept["attack_type_legacy"] = np.asarray(cls_index)[kept["_bL"].to_numpy() % K]
    kept = kept.drop(columns=[c for c in ("_bC", "_bL", "_k", "_al") if c in kept.columns])
    info = {"sampler": "uniform_bottom_k", "cap_per_region_class": int(caps["CN"]),
            "caps_by_design": {d: int(c) for d, c in caps.items()},
            "designs": designs, "raw_rows_streamed": int(seen), "seed": seed}
    log(f"  uniform sampler: {seen:,} raw rows -> {len(kept):,} rows"
        + (" (union over designs)" if mv else ""))
    return kept, info


def _compact_mv(kept, buf, designs, caps, thr, mv):
    big = pd.concat(([kept] if kept is not None else []) + buf, ignore_index=True)
    k = big["_k"].to_numpy()
    al = big["_al"].to_numpy()
    dm = np.zeros(len(big), np.uint8)
    for bit, d in enumerate(designs):
        b_all = big["_bC" if d[0] == "C" else "_bL"].to_numpy()
        a = al if d[0] == "C" else (al >> 4)
        idx = np.flatnonzero((a & _DROP_BIT[d[1]]) > 0)
        order = idx[np.lexsort((k[idx], b_all[idx]))]
        b = b_all[order]
        ub, starts, inv, cnt = np.unique(b, return_index=True, return_inverse=True,
                                         return_counts=True)
        rank = np.arange(len(b)) - starts[inv]
        dm[order[rank < caps[d]]] |= np.uint8(1 << bit)
        full = cnt >= caps[d]
        last = starts + np.minimum(cnt, caps[d]) - 1
        thr[d][ub[full]] = k[order][last[full]]
    big["_dm"] = dm
    big = big[dm > 0]
    order = np.lexsort((big["_k"].to_numpy(), big["_bC"].to_numpy()))
    return big.iloc[order].reset_index(drop=True)


def features_view(df, legacy_set):
    """Feature columns of a sampled frame under a variant (C1, C2, C4). Labels
    and row membership are handled by the caller (they depend on the design)."""
    ls = frozenset(legacy_set)
    out = df
    if "C1" in ls:
        out = out.assign(track_message_count=df["track_message_count_legacy"].to_numpy())
    if "C2" in ls:
        out = out.assign(squawk_changed=df["squawk_changed_legacy"].to_numpy())
    if "C4" in ls:
        sub = df["_sub"].to_numpy().astype(bool)
        repl = {}
        for c in V.RATE_FEATURES:
            col = df[c].to_numpy(dtype="float32", copy=True)
            col[sub] = np.nan
            repl[c] = col
        repl["is_physically_implausible"] = V._as_bool(df["is_physically_implausible"]) & ~sub
        out = out.assign(**repl)
    return out


def sample_view(union, info, legacy_set):
    """Variant view of the union sample returned by uniform_sample_mv: the rows
    sampled in the variant's design, with the variant's labels and features."""
    ls = frozenset(legacy_set)
    design = V.design_of(ls)
    designs = info["designs"]
    if design not in designs:
        raise ValueError(f"variant needs sampling design {design}, but the training files are "
                         f"not multi-variant files")
    bit = np.uint8(1 << designs.index(design))
    m = (union["_dm"].to_numpy() & bit) > 0
    v = union[m]
    if design[0] == "L":
        v = v.assign(attack_type=v["attack_type_legacy"].to_numpy())
    v = features_view(v, ls)
    return v.reset_index(drop=True)


def legacy_sample_view(paths, max_total_rows, stats, legacy_set=frozenset(), seed=42,
                       chunksize=200_000, log=print, state_path=None, stop_check=None):
    """federated_simulation.load_data (the legacy loader) applied to a variant's
    DESIGN view of the files: the variant's labels, and the legacy-filtered
    rows removed BEFORE chunking (rows are re-chunked to `chunksize`), so the
    result equals load_data run on the file that variant's pipeline would have
    written (checked in test_sampling_mv.py against load_data on real v2 / v3
    pipeline outputs). All feature versions are kept. Resumable per file."""
    ls = frozenset(legacy_set)
    design = V.design_of(ls)
    if design not in stats["designs"]:
        raise ValueError(f"variant needs design {design}; training files are not multi-variant")
    mv = bool(stats.get("mv"))
    usecols = fs.FEATURE_COLS + ["region_cell", "attack_type", "icao24", "time", "lastposupdate"]
    extra = V.MV_READ_COLS if mv else []
    n_regions = max(1, stats["designs"][design]["n_regions"])
    cap = max(500, max_total_rows // (n_regions * N_CLASSES_GUESS))
    log(f"  legacy loader (design {design}): {n_regions:,} regions, cap {cap:,} per (region, class)")
    reservoirs, total_seen, hit, start_file = {}, 0, False, 0
    sig = json.loads(json.dumps([[_file_sig(p) for p in paths], max_total_rows, seed, design]))
    if state_path and os.path.exists(state_path):
        with open(state_path, "rb") as f:
            st = pickle.load(f)
        if st.get("sig") == sig:
            reservoirs, total_seen, hit, start_file = (st["reservoirs"], st["seen"], st["hit"],
                                                       st["next_file"])
            log(f"  legacy loader: resuming after {start_file} finished file(s)")

    def consume(chunk):
        nonlocal total_seen, hit
        total_seen += len(chunk)
        num_cols = [c for c in fs.FEATURE_COLS + ["track_message_count_legacy"]
                    if c in chunk.columns and pd.api.types.is_numeric_dtype(chunk[c])]
        if num_cols:
            chunk[num_cols] = chunk[num_cols].astype("float32")
        for (region, atk), group in chunk.groupby(["region_cell", "attack_type"]):
            key = (region, atk)
            reservoirs.setdefault(key, [])
            reservoirs[key].append(group)
            combined = pd.concat(reservoirs[key], ignore_index=True)
            if len(combined) > cap:
                combined = combined.sample(n=cap, random_state=seed)
            reservoirs[key] = [combined]
        if sum(len(v[0]) for v in reservoirs.values()) >= max_total_rows:
            log(f"  Reached global cap of {max_total_rows:,} rows after {total_seen:,} raw rows "
                f"seen -- stopping early (remaining files/chunks skipped).")
            hit = True

    for fi, path in enumerate(paths):
        if fi < start_file:
            continue
        if hit:
            break
        header = pd.read_csv(path, nrows=0).columns
        cols = [c for c in usecols + extra if c in header]
        pend = []
        pend_rows = 0
        for chunk in pd.read_csv(path, usecols=cols, chunksize=chunksize, low_memory=False):
            if stop_check and stop_check():
                raise StopRequested("legacy sampling pass (the current file restarts on resume)")
            if mv:
                at, alive = V.design_labels_and_alive(chunk)[design]
                chunk["_sub"] = V.subsecond_mask(chunk)
                if design[0] == "L":
                    chunk["attack_type"] = at
                if design[1] != "N":
                    chunk = chunk[alive]
            if design[1] == "N" or not mv:
                consume(chunk)
            else:
                pend.append(chunk)
                pend_rows += len(chunk)
                while pend_rows >= chunksize and not hit:
                    allp = pd.concat(pend, ignore_index=True)
                    consume(allp.iloc[:chunksize].copy())
                    rest = allp.iloc[chunksize:]
                    pend, pend_rows = [rest], len(rest)
            if hit:
                break
        if pend_rows and not hit:
            consume(pd.concat(pend, ignore_index=True))
        if state_path:
            atomic_pickle({"sig": sig, "reservoirs": reservoirs, "seen": total_seen, "hit": hit,
                           "next_file": fi + 1}, state_path)
    df = pd.concat([pd.concat(parts, ignore_index=True) for parts in reservoirs.values()],
                   ignore_index=True)
    log(f"  Streamed {total_seen:,} raw rows -> sampled down to {len(df):,} rows "
        f"across {n_regions:,} regions")
    info = {"sampler": "legacy_load_data", "design": design, "raw_rows_streamed": int(total_seen),
            "cap_per_region_class": int(cap), "seed": seed, "hit_global_cap_early": bool(hit)}
    return df, info
