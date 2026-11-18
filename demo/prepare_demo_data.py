"""
prepare_demo_data.py: build the per-client data files for the FL-IDS live demo.

Reads the cached training sample (models/sample_3m.pkl) and the labeled holdout
CSV, keeps only the six demo region cells, and writes into demo/data/:

    clients.json         client ids, cells, names, centre lat/lon, row counts
    C1_train.csv ...     that cell's rows from the cached sample (all classes),
                         used by train_demo.py for Flower training
    C1_stream.csv ...    that cell's BENIGN holdout rows sorted by time
                         (the normal traffic the server streams)
    C1_pool.csv ...      that cell's ATTACK holdout rows (injection pool),
                         topped up from the other cells when a type has < 50 rows

Feature values are written already numeric (booleans as 1/0, text numerics
coerced) but missing values are left EMPTY, not filled with -1: the server
runs federated_simulation.prep_features itself. See demo/CONTRACT.md.

Usage (run from the project root):
    python3 demo/prepare_demo_data.py
    python3 demo/prepare_demo_data.py --holdout ~/opensky_data/labeled_fixed/holdout_feb04_1pct.csv
    python3 demo/prepare_demo_data.py --cells "4_-10,5_1,3_-9,0_10,-4_14,5_-14" \
        --names "US Midwest,Central Europe,US Southeast,Malay Peninsula,SE Australia,Gulf of Alaska"
    python3 demo/prepare_demo_data.py --sample /tmp/fl-demo-fixture/models/sample_3m.pkl \
        --holdout /tmp/fl-demo-fixture/opensky_data/labeled_fixed/holdout_feb04_1pct.csv \
        --out /tmp/fl-demo-fixture/demo/data
"""

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
sys.path.insert(0, str(ROOT))

import federated_simulation as fs  # noqa: E402  (needs the sys.path insert above)

FEATURE_COLS = fs.FEATURE_COLS
BOOL_COLS = ["is_physically_implausible", "squawk_changed", "icao24_valid_format",
             "is_stale", "is_low_coverage_region", "is_climbing_or_descending"]
CLASSES = ["benign", "flooding", "ghost_aircraft", "gps_spoof", "replay", "velocity_spoof"]
ATTACK_TYPES = CLASSES[1:]

DEFAULT_CELLS = ["4_-10", "5_1", "3_-9", "0_10", "-4_14", "5_-14"]
DEFAULT_NAMES = {
    "4_-10": "US Midwest",
    "5_1": "Central Europe",
    "3_-9": "US Southeast",
    "0_10": "Malay Peninsula",
    "-4_14": "SE Australia",
    "5_-14": "Gulf of Alaska",
}

MIN_POOL_PER_TYPE = 50      # top up a cell's pool to this many rows per attack type
WARN_STREAM_ROWS = 2_000    # below this a client runs out of benign traffic quickly
WARN_TRAIN_ROWS = 3_000     # same threshold as federated_simulation.MIN_CLIENT_ROWS
SEED = 42

TRAIN_COLS = FEATURE_COLS + ["attack_type"]
STREAM_COLS = ["time", "icao24"] + FEATURE_COLS + ["attack_type"]


def cell_centre(cell):
    """Cell 'a_b' covers lat [10a, 10a+10), lon [10b, 10b+10)."""
    a, b = (int(x) for x in cell.split("_"))
    return 10 * a + 5, 10 * b + 5


def to_numeric_features(df):
    """Same conversion semantics as fs.prep_features (bool strings to 1/0,
    text numerics coerced, bad values become NaN) but WITHOUT the -1 fill and
    without the float32 cast, so missing values stay empty in the CSV.
    Boolean columns come back as nullable Int8 so they are written as 1/0."""
    out = df.copy()
    bool_map = {"True": 1, "true": 1, "False": 0, "false": 0, True: 1, False: 0}
    for col in FEATURE_COLS:
        s = out[col]
        if col in BOOL_COLS:
            if not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
                s = s.astype(object).replace(bool_map)
            s = pd.to_numeric(s, errors="coerce")
            # anything that is not cleanly 0/1 after coercion is kept as is
            # numerically; only exact 0/1 columns become Int8
            if s.dropna().isin([0, 1]).all():
                s = s.astype("Int8")
        else:
            s = pd.to_numeric(s, errors="coerce")
        out[col] = s
    if "time" in out.columns:
        t = pd.to_numeric(out["time"], errors="coerce")
        if len(t) and t.isna().mean() > 0.5:
            # time stored as date text, not unix seconds: convert instead of blanking it
            dt = pd.to_datetime(out["time"], utc=True, errors="coerce", format="mixed")
            t = (dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
        # unix seconds: write as integers when they are whole numbers
        if t.notna().all() and (t % 1 == 0).all():
            t = t.astype("int64")
        out["time"] = t
    return out


def load_sample(path, cells):
    """Load the cached sample pickle, keep the six cells, drop the rest right away.
    Also returns per-cell row counts of the WHOLE sample for the suggestions table."""
    print(f"Loading cached sample {path} ...")
    payload = pd.read_pickle(path)
    if isinstance(payload, dict) and "df" in payload:
        df = payload["df"]
    elif isinstance(payload, pd.DataFrame):
        df = payload
    else:
        sys.exit(f"ERROR: {path} is not a {{'meta', 'df'}} sample cache or a DataFrame.")
    del payload
    missing = [c for c in ["region_cell", "attack_type"] + FEATURE_COLS if c not in df.columns]
    if missing:
        sys.exit(f"ERROR: sample is missing columns: {missing}")
    region = df["region_cell"].astype(str)
    all_counts = region.value_counts()
    keep = region.isin(cells).to_numpy()
    cols = [c for c in TRAIN_COLS + ["region_cell"] if c in df.columns]
    sub = df.loc[keep, cols].copy()
    n_total = len(df)
    del df, region, keep
    gc.collect()
    sub["region_cell"] = sub["region_cell"].astype(str)
    sub["attack_type"] = sub["attack_type"].astype(str)
    print(f"  {n_total:,} rows in sample, {len(sub):,} in the selected cells")
    return sub, all_counts


def load_holdout(path, cells, chunksize):
    """Stream the holdout as strings, keep only the selected cells.
    Returns the kept rows and per-cell (total, benign) counts over the whole file."""
    print(f"Reading holdout {path} in chunks of {chunksize:,} ...")
    header = pd.read_csv(path, nrows=0)
    want = ["time", "icao24", "region_cell", "attack_type"] + FEATURE_COLS
    missing = [c for c in want if c not in header.columns]
    if missing:
        sys.exit(f"ERROR: holdout is missing columns: {missing}")
    kept, total_counts, benign_counts = [], None, None
    n_rows = 0
    for chunk in pd.read_csv(path, usecols=want, dtype=str, chunksize=chunksize,
                             keep_default_na=True):
        n_rows += len(chunk)
        vc = chunk["region_cell"].value_counts()
        bc = chunk.loc[chunk["attack_type"] == "benign", "region_cell"].value_counts()
        total_counts = vc if total_counts is None else total_counts.add(vc, fill_value=0)
        benign_counts = bc if benign_counts is None else benign_counts.add(bc, fill_value=0)
        sel = chunk[chunk["region_cell"].isin(cells)]
        if len(sel):
            kept.append(sel)
    hold = pd.concat(kept, ignore_index=True) if kept else pd.DataFrame(columns=want)
    print(f"  {n_rows:,} holdout rows read, {len(hold):,} in the selected cells")
    counts = pd.DataFrame({"total": total_counts, "benign": benign_counts}).fillna(0).astype(int)
    return hold, counts


def main():
    p = argparse.ArgumentParser(description="Build per-client demo data files.")
    p.add_argument("--sample", default=str(ROOT / "models" / "sample_3m.pkl"),
                   help="cached training sample pickle (default models/sample_3m.pkl)")
    p.add_argument("--holdout",
                   default="~/opensky_data/labeled_fixed/holdout_feb04_1pct.csv",
                   help="labeled holdout CSV")
    p.add_argument("--cells", default=",".join(DEFAULT_CELLS),
                   help="comma-separated region cells, in client order C1..Cn")
    p.add_argument("--names", default=None,
                   help="optional comma-separated display names, same order as --cells")
    p.add_argument("--out", default=str(DEMO_DIR / "data"), help="output directory")
    p.add_argument("--chunksize", type=int, default=100_000, help="holdout rows per chunk")
    args = p.parse_args()

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    if len(set(cells)) != len(cells):
        sys.exit("ERROR: --cells contains duplicates")
    for c in cells:
        try:
            cell_centre(c)
        except ValueError:
            sys.exit(f"ERROR: bad cell {c!r}, expected 'a_b' with integers, e.g. 4_-10")
    if args.names:
        names = [n.strip() for n in args.names.split(",")]
        if len(names) != len(cells):
            sys.exit("ERROR: --names must have the same number of entries as --cells")
    else:
        names = [DEFAULT_NAMES.get(c, f"Cell {c}") for c in cells]

    sample_path = Path(args.sample).expanduser().resolve()
    holdout_path = Path(args.holdout).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    for pth in (sample_path, holdout_path):
        if not pth.is_file():
            sys.exit(f"ERROR: not found: {pth}")
    out.mkdir(parents=True, exist_ok=True)

    train_all, sample_counts = load_sample(sample_path, cells)
    hold_all, hold_counts = load_holdout(holdout_path, cells, args.chunksize)

    unknown = sorted(set(train_all["attack_type"]) - set(CLASSES))
    if unknown:
        print(f"  WARNING: sample has unexpected attack_type values {unknown}, dropping them")
        train_all = train_all[train_all["attack_type"].isin(CLASSES)]
    hold_all = hold_all[hold_all["attack_type"].isin(CLASSES)]

    train_all = to_numeric_features(train_all)
    hold_all = to_numeric_features(hold_all)

    attacks_all = hold_all[hold_all["attack_type"] != "benign"]
    rng = np.random.default_rng(SEED)

    clients_meta, summary, problems, fatal = [], [], [], []
    for i, (cell, name) in enumerate(zip(cells, names), start=1):
        cid = f"C{i}"
        lat, lon = cell_centre(cell)

        tr = train_all[train_all["region_cell"] == cell][TRAIN_COLS]
        tr.to_csv(out / f"{cid}_train.csv", index=False)

        h = hold_all[hold_all["region_cell"] == cell]
        stream = h[h["attack_type"] == "benign"].sort_values("time", kind="stable",
                                                             na_position="last")
        stream[STREAM_COLS].to_csv(out / f"{cid}_stream.csv", index=False)

        pool = h[h["attack_type"] != "benign"]
        parts, borrowed = [pool], {}
        for atk in ATTACK_TYPES:
            have = int((pool["attack_type"] == atk).sum())
            if have >= MIN_POOL_PER_TYPE:
                continue
            others = attacks_all[(attacks_all["attack_type"] == atk)
                                 & (attacks_all["region_cell"] != cell)]
            need = min(MIN_POOL_PER_TYPE - have, len(others))
            if need > 0:
                pick = rng.choice(len(others), size=need, replace=False)
                parts.append(others.iloc[np.sort(pick)])
                borrowed[atk] = int(need)
        pool = pd.concat(parts).sort_values("time", kind="stable", na_position="last")
        pool[STREAM_COLS].to_csv(out / f"{cid}_pool.csv", index=False)

        pool_rows = {atk: int((pool["attack_type"] == atk).sum()) for atk in ATTACK_TYPES}
        train_by_class = {c: int((tr["attack_type"] == c).sum()) for c in CLASSES}
        clients_meta.append({
            "id": cid, "cell": cell, "name": name, "lat": lat, "lon": lon,
            "train_rows": int(len(tr)), "stream_rows": int(len(stream)),
            "pool_rows": pool_rows, "pool_borrowed": borrowed,
        })
        summary.append((cid, cell, name, train_by_class, len(tr), len(stream), pool_rows, borrowed))
        if len(stream) == 0 or len(tr) == 0:
            fatal.append(f"{cid} ({cell}, {name}): {len(tr):,} train rows, {len(stream):,} stream rows")
        if len(stream) < WARN_STREAM_ROWS:
            problems.append(f"{cid} ({cell}, {name}): only {len(stream):,} stream rows "
                            f"(want >= {WARN_STREAM_ROWS:,})")
        if len(tr) < WARN_TRAIN_ROWS:
            problems.append(f"{cid} ({cell}, {name}): only {len(tr):,} train rows "
                            f"(want >= {WARN_TRAIN_ROWS:,})")
        for atk in ATTACK_TYPES:
            if pool_rows[atk] == 0:
                problems.append(f"{cid} ({cell}): pool has NO {atk} rows even after borrowing")

    meta = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source": {"sample": str(sample_path), "holdout": str(holdout_path)},
        "clients": clients_meta,
    }
    if fatal:
        # no clients.json, so start_demo.sh does not skip the prep step next time
        (out / "clients.json").unlink(missing_ok=True)
    else:
        with open(out / "clients.json", "w") as f:
            json.dump(meta, f, indent=2)

    # ---- summary table -------------------------------------------------------
    short = {"benign": "benign", "flooding": "flood", "ghost_aircraft": "ghost",
             "gps_spoof": "gps", "replay": "replay", "velocity_spoof": "veloc"}
    print(f"\nWrote {len(cells)} clients to {out}\n")
    print("TRAIN rows per class (from the cached sample)")
    head = f"  {'id':<3} {'cell':<7} {'name':<18}" + "".join(f"{short[c]:>8}" for c in CLASSES)
    print(head + f"{'total':>9}")
    for cid, cell, name, by_cls, n_tr, _, _, _ in summary:
        print(f"  {cid:<3} {cell:<7} {name[:18]:<18}"
              + "".join(f"{by_cls[c]:>8,}" for c in CLASSES) + f"{n_tr:>9,}")
    print("\nSTREAM (benign holdout) and POOL (attack holdout) rows")
    print(f"  {'id':<3} {'cell':<7} {'stream':>8}" + "".join(f"{short[a]:>8}" for a in ATTACK_TYPES)
          + "   borrowed")
    for cid, cell, _, _, _, n_st, pool_rows, borrowed in summary:
        b = ", ".join(f"{short[k]} +{v}" for k, v in borrowed.items()) or "-"
        print(f"  {cid:<3} {cell:<7} {n_st:>8,}"
              + "".join(f"{pool_rows[a]:>8,}" for a in ATTACK_TYPES) + f"   {b}")

    if problems:
        bar = "!" * 78
        print(f"\n{bar}\nWARNING: some demo cells look too thin for a convincing demo:")
        for msg in problems:
            print(f"  - {msg}")
        print("Consider replacing them via --cells. Densest cells in the holdout:")
        print(bar)
    else:
        print("\nAll cells have enough rows. For reference, densest cells in the holdout:")

    top = hold_counts.sort_values("total", ascending=False).head(10)
    print(f"  {'cell':<8} {'holdout':>9} {'benign':>9} {'sample':>9}  centre (lat, lon)")
    for cell, row in top.iterrows():
        try:
            lat, lon = cell_centre(str(cell))
            centre = f"({lat}, {lon})"
        except ValueError:
            centre = "?"
        mark = "  <- in use" if cell in cells else ""
        print(f"  {str(cell):<8} {row['total']:>9,} {row['benign']:>9,} "
              f"{int(sample_counts.get(cell, 0)):>9,}  {centre}{mark}")

    if fatal:
        print("\nERROR: these cells have NO rows, so the demo cannot run with them:")
        for msg in fatal:
            print(f"  - {msg}")
        print("Pick replacements from the table above and pass them with --cells "
              "(or CELLS=... for start_demo.sh).")
        sys.exit(1)


if __name__ == "__main__":
    main()
