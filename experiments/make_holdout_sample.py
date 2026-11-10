"""
make_holdout_sample.py
-------------------------------------------------------------------
Builds the held-out test sample from labelled day files. Nothing is used for
training or selection; the natural class mix is preserved.

--method pandas (default, round 1): 1% uniform sample drawn exactly as before
    (pandas sample(frac=0.01, random_state=42) on every 2,000,000-row chunk
    of each file, files in the order given).

--method hash (revision 2): a row is selected when a seeded hash of its
    stable row key (icao24, time, lat, lon, lastposupdate) falls below
    --frac. The key does not depend on labels or engineered features, so
      * the same rows are selected whatever the chunking, and
      * the same rows are selected from a v2, a v3 and a multi-variant file
        of the same day (rows that only exist in v3 are simply present or
        not). For a multi-variant file every column is kept, so each variant
        of the test sample is a column selection / row mask of ONE file
        (rows removed by the legacy filter are masked at load time).
    Written per input file to a temp name and renamed, then concatenated;
    a rerun skips finished files. The output is verified by row count.

Usage:
  python3 experiments/make_holdout_sample.py --method hash \
      ~/rev2_work/labeled_mv/opensky_2019-12-{09,16,23,30}_labeled.csv \
      --out ~/rev2_work/holdout_sample_mv.csv
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

KEY_COLS = ["icao24", "time", "lat", "lon", "lastposupdate"]
_M1, _M2, _G = np.uint64(0xBF58476D1CE4E5B9), np.uint64(0x94D049BB133111EB), np.uint64(0x9E3779B97F4A7C15)


def _mix(h):
    """splitmix64 finaliser on a uint64 array (wraps modulo 2**64)."""
    with np.errstate(over="ignore"):
        h = (h ^ (h >> np.uint64(30))) * _M1
        h = (h ^ (h >> np.uint64(27))) * _M2
        return h ^ (h >> np.uint64(31))


def row_uniform(df, seed):
    """Deterministic uniform number in [0, 1) per row from its stable key.
    Uses blake2b for the address and the IEEE bit patterns of the numeric
    key columns: independent of pandas / numpy versions and of chunking."""
    codes, uniq = pd.factorize(df["icao24"].astype(str), sort=False)
    hu = np.array([int.from_bytes(hashlib.blake2b(f"{seed}|{a}".encode(), digest_size=8).digest(),
                                  "big") for a in uniq], dtype=np.uint64)
    h = hu[codes] if len(uniq) else np.zeros(0, np.uint64)
    for c in KEY_COLS[1:]:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64, copy=True)
        v[np.isnan(v)] = -1.0
        v += 0.0          # -0.0 -> 0.0
        with np.errstate(over="ignore"):
            h = _mix((h + _G) ^ v.view(np.uint64))
    return (h >> np.uint64(11)).astype(np.float64) * (1.0 / (1 << 53))


def count_lines(path, bufsize=1 << 24):
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                return n
            n += b.count(b"\n")


def hash_sample(args):
    stop = args.stop_flag
    parts, total_in, total_out, counts = [], 0, 0, {}
    for i, p in enumerate(args.paths):
        part = f"{args.out}.part{i}"
        meta_p = part + ".json"
        sig = [os.path.basename(p), os.path.getsize(p), args.frac, args.seed]
        if os.path.exists(part) and os.path.exists(meta_p) and json.load(open(meta_p))["sig"] == sig:
            m = json.load(open(meta_p))
            print(f"{os.path.basename(p)}: already sampled ({m['rows_out']:,} rows)", flush=True)
        else:
            t0 = time.time()
            n_in = n_out = 0
            c = {}
            first = True
            tmp = part + ".tmp"
            for ch in pd.read_csv(p, chunksize=args.chunksize, low_memory=False):
                if stop and os.path.exists(stop):
                    print("STOPPED on request; rerun to continue", flush=True)
                    return 75
                s = ch[row_uniform(ch, args.seed) < args.frac]
                s.to_csv(tmp, mode="w" if first else "a", header=first, index=False)
                first = False
                n_in += len(ch)
                n_out += len(s)
                for k, v in s["attack_type"].astype(str).value_counts().items():
                    c[k] = c.get(k, 0) + int(v)
            if count_lines(tmp) != n_out + 1:
                raise RuntimeError(f"{tmp}: row count check failed")
            os.replace(tmp, part)
            m = {"sig": sig, "rows_in": n_in, "rows_out": n_out, "counts": c}
            with open(meta_p + ".tmp", "w") as f:
                json.dump(m, f)
            os.replace(meta_p + ".tmp", meta_p)
            print(f"{os.path.basename(p)}: {n_in:,} rows -> {n_out:,} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
        parts.append(part)
        total_in += m["rows_in"]
        total_out += m["rows_out"]
        for k, v in m["counts"].items():
            counts[k] = counts.get(k, 0) + v
    tmp = args.out + ".tmp"
    with open(tmp, "wb") as out:
        for k, part in enumerate(parts):
            with open(part, "rb") as f:
                header = f.readline()
                if k == 0:
                    out.write(header)
                shutil.copyfileobj(f, out, 1 << 24)
        out.flush()
        os.fsync(out.fileno())
    if count_lines(tmp) != total_out + 1:
        raise RuntimeError(f"{tmp}: row count check failed")
    os.replace(tmp, args.out)
    with open(args.out + ".done.json.tmp", "w") as f:
        json.dump({"rows": total_out, "rows_in": total_in, "bytes": os.path.getsize(args.out),
                   "frac": args.frac, "seed": args.seed, "method": "hash", "key": KEY_COLS,
                   "files": [os.path.basename(p) for p in args.paths], "counts": counts}, f, indent=1)
    os.replace(args.out + ".done.json.tmp", args.out + ".done.json")
    for part in parts:
        os.remove(part)
        os.remove(part + ".json")
    report(args.out, total_in, total_out, counts)
    return 0


def report(out, total_in, total_out, counts):
    print(f"wrote {out}: {total_out:,} rows from {total_in:,}")
    for k in sorted(counts):
        print(f"  {k:<16}{counts[k]:>10,}  {counts[k] / max(1, total_out):.4f}")


def pandas_sample(args):
    tmp = args.out + ".part"
    first = True
    total_in = total_out = 0
    counts = {}
    for p in args.paths:
        t0 = time.time()
        n_in = n_out = 0
        for ch in pd.read_csv(p, chunksize=args.chunksize, low_memory=False):
            s = ch.sample(frac=args.frac, random_state=args.seed)
            s.to_csv(tmp, mode="w" if first else "a", header=first, index=False)
            first = False
            n_in += len(ch)
            n_out += len(s)
            for k, v in s["attack_type"].astype(str).value_counts().items():
                counts[k] = counts.get(k, 0) + int(v)
        total_in += n_in
        total_out += n_out
        print(f"{os.path.basename(p)}: {n_in:,} rows -> {n_out:,} "
              f"({(time.time() - t0) / 60:.1f} min)", flush=True)
    os.replace(tmp, args.out)
    report(args.out, total_in, total_out, counts)
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="Labelled day CSVs (test days)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--method", choices=("pandas", "hash"), default="pandas")
    ap.add_argument("--frac", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunksize", type=int, default=None,
                    help="default 2,000,000 (pandas method, part of its definition) or 500,000 (hash)")
    ap.add_argument("--stop-flag", default=os.environ.get("REV2_STOP_FLAG"))
    args = ap.parse_args()
    if args.chunksize is None:
        args.chunksize = 2_000_000 if args.method == "pandas" else 500_000
    return hash_sample(args) if args.method == "hash" else pandas_sample(args)


if __name__ == "__main__":
    sys.exit(main())
