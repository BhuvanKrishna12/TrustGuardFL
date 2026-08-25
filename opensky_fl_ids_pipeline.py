"""
OpenSky FL-IDS Pipeline
=======================
Loads one or more raw OpenSky state-vector CSVs, cleans them, engineers
Tier A/B/C features, and injects synthetic attack samples for
training/evaluating an intrusion detection model.

Multiple files are concatenated BEFORE feature engineering (not after),
so per-aircraft kinematic features (inter_arrival_s, implied_speed_mps,
etc.) stay continuous across hour/day boundaries instead of resetting
at each file's edge.

Usage:
    python opensky_fl_ids_pipeline.py states_2017-06-05-00.csv
    python opensky_fl_ids_pipeline.py states_2017-06-05-*.csv   (multi-file, via shell glob)

Output:
    <name>_labeled.csv  -- cleaned data + engineered features + attack labels
"""

import gc
import sys
import numpy as np
import pandas as pd

RNG_SEED = 42

# Columns that MUST stay full-precision. These are Unix timestamps
# (~1.5 billion+) and float32 only carries ~7 significant digits -- it
# literally cannot represent numbers this large exactly. Downcasting these
# to float32 would silently corrupt every diff-based feature in the
# pipeline (inter_arrival_s, is_stale, the replay attack's time-shift
# logic, etc). Deliberately left OUT of READ_DTYPES below so pandas keeps
# its default float64 for them.
TIMESTAMP_COLS = ["time", "lastposupdate", "lastcontact"]

# Everything else numeric (lat/lon/velocity/heading/vertrate/altitudes) is
# well within float32's precision range for this use case -- reading these
# directly as float32 (rather than reading float64 and downcasting after)
# means pandas never allocates the float64 version at all, which is the
# actual memory win across 120 files.
FLOAT32_COLS = [
    "lat", "lon", "velocity", "heading", "vertrate",
    "baroaltitude", "geoaltitude",
]
READ_DTYPES = {c: "float32" for c in FLOAT32_COLS}

# callsign is dropped entirely -- it's Tier C (lowest-weighted, per
# add_tier_c_features docstring) and only fed callsign_valid_format, a
# weak signal. It's also a Python object/string column, exactly the kind
# that dominates memory overhead in a dataset this size. Skipping it at
# read time (rather than loading then discarding) means it's never
# materialized in memory at all. If you ever want it back, remove it from
# DROP_COLS and restore the three call sites flagged "callsign:" below.
DROP_COLS = ["callsign"]

# Fastest civil aircraft (e.g. Concorde-era supersonic) tops out well under
# this; used as a physical plausibility ceiling for implied ground speed.
# Anything above this in a row NOT labeled as an attack is almost certainly
# sensor noise (duplicate/near-duplicate receiver reports, GPS glitches)
# rather than a real aircraft -- see main() for how this is used.
MAX_PLAUSIBLE_SPEED_MPS = 400

# Threshold for treating an aircraft as climbing/descending rather than in
# stable cruise. ~2.5 m/s (~500 ft/min) is a common rule-of-thumb cutoff.
# Used to contextualize vrate_consistency_delta: legitimate climb/descent
# phases naturally show larger broadcast-vs-GPS-derived disagreement than
# stable cruise, which otherwise looks similar to velocity_spoof.
CLIMB_DESCENT_VERTRATE_MPS = 2.5

# --- Attack severity randomization ---------------------------------------
# Previously each attack type used a single fixed magnitude range applied
# identically to every injected instance (e.g. gps_spoof always shifted
# position by uniform(-2, 2) degrees). That meant every example of a given
# attack type was roughly equally easy to detect -- a model trained on it
# learns to catch obvious corruption but never sees anything resembling a
# careful/stealthy attacker. These constants introduce a genuine severity
# spread per row: some injected attacks are now deliberately subtle
# (barely distinguishable from sensor noise) through severe (the old
# fixed behavior), so a downstream model has to learn the actual boundary
# rather than a single obvious threshold.
GPS_SPOOF_SEVERITY_DEG = {"subtle": 0.05, "moderate": 0.5, "severe": 2.0}
GPS_SPOOF_SEVERITY_PROBS = [0.34, 0.33, 0.33]
VELOCITY_SPOOF_MULTIPLIER_RANGE = (1.1, 8.0)   # was fixed at uniform(2, 5)
VELOCITY_SPOOF_VERTRATE_BOOST_RANGE = (5.0, 40.0)  # was fixed at uniform(20, 40)
REPLAY_SHIFT_RANGE_S = (2, 900)                # was fixed at uniform(60, 600)
FLOODING_DUP_COUNT_RANGE = (2, 11)             # rng.integers high is exclusive; was always exactly 5
FLOODING_OFFSET_RANGE_S = (0.01, 0.5)          # was fixed at 0.1

# Above this gap, treat consecutive sightings of the same icao24 as
# separate sessions (e.g. different collection days) rather than one
# continuous flight -- see add_tier_a_features for why this matters when
# combining multiple non-consecutive days. 1 hour is comfortably above
# any real single-region coverage gap (oceanic corridors observed ~170s
# max) and comfortably below the gap between separate collection days.
SESSION_BREAK_SECONDS = 3600


# ---------------------------------------------------------------------------
# 1. LOADING & CLEANING
# ---------------------------------------------------------------------------

def _read_one_csv(path: str) -> pd.DataFrame:
    """
    Read a single file with lat/lon/velocity/etc as float32 from the start
    (see READ_DTYPES). This is the fast path and is what runs for
    well-formed OpenSky files.

    Falls back to the old read-as-default + to_numeric(errors="coerce")
    path ONLY if the fast path chokes on genuinely malformed values (e.g.
    a stray non-numeric string in a column read expects to be all-numeric)
    -- pandas' C parser raises rather than silently coercing when a dtype
    is specified up front, so we can't just always use the fast path
    blindly.
    """
    try:
        return pd.read_csv(path, dtype=READ_DTYPES,
                            usecols=lambda c: c not in DROP_COLS)
    except (ValueError, TypeError):
        print(f"  Note: {path} has non-numeric junk in a numeric column, "
              f"falling back to coerce-and-downcast for this file")
        df = pd.read_csv(path, usecols=lambda c: c not in DROP_COLS)
        for col in FLOAT32_COLS:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float32")
        return df


def load_and_clean(paths) -> pd.DataFrame:
    if isinstance(paths, str):
        paths = [paths]

    frames = [_read_one_csv(p) for p in paths]
    df = pd.concat(frames, ignore_index=True) if len(frames) > 1 else frames[0]

    # frames is now redundant -- each frame's data lives inside df after
    # concat. Holding onto the list (and thus 120 separate DataFrame
    # objects) for the rest of the function was pure waste; drop the
    # references and force collection so that memory is actually freed
    # before feature engineering, not just eligible for it eventually.
    frames.clear()
    gc.collect()

    # Hourly files can occasionally overlap by a row or two at the
    # boundary (OpenSky's extraction windows aren't perfectly disjoint) --
    # drop exact duplicates so the same message isn't double-counted.
    if len(paths) > 1:
        before = len(df)
        df = df.drop_duplicates()
        if before != len(df):
            print(f"  Dropped {before - len(df):,} exact-duplicate rows "
                  f"from overlapping file boundaries")

    # callsign: dropped at load time (see DROP_COLS) -- no strip step needed
    # Booleans arrive as strings "True"/"False" -> real bool
    for col in ["onground", "alert", "spi"]:
        df[col] = df[col].astype(str).str.strip().map({"True": True, "False": False})

    # Numeric coercion for the timestamp columns ONLY -- lat/lon/velocity/
    # etc were already read (and dtype-enforced) as float32 above, so
    # re-coercing them here would just allocate a redundant float64 copy
    # before immediately discarding it.
    for col in TIMESTAMP_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows with no position fix at all -- can't build kinematic
    # features without lat/lon.
    df = df.dropna(subset=["lat", "lon", "time"]).copy()

    # Sort so per-aircraft diffs are chronological
    df = df.sort_values(["icao24", "time"]).reset_index(drop=True)

    # Flag OpenSky's own staleness indicator from the docs:
    # state vectors persist up to 300s after last real contact.
    df["is_stale"] = (df["time"] - df["lastcontact"]) > 15

    return df


# ---------------------------------------------------------------------------
# 2. TIER A -- kinematic / physics-consistency features
#    (protocol-agnostic; these are what attack injection will violate)
# ---------------------------------------------------------------------------

EARTH_RADIUS_M = 6_371_000


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres between consecutive fixes."""
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def add_tier_a_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    PERFORMANCE NOTE: this used to be g = df.groupby("icao24") followed by
    four g[col].shift(1) calls. groupby() has to build a full group index
    (hash every icao24, bucket every row) before it can do anything, and it
    rebuilds that index from scratch on EVERY .shift() call unless you keep
    the same groupby object -- with 50M+ rows and tens of thousands of
    unique icao24 values, this was the single biggest cost in the pipeline,
    and it was paid once at load time AND once per attack type during
    recompute (see inject_attack), so ~6x per chunk.

    The fix: df is guaranteed sorted by ["icao24", "time"] every time this
    function is called (load_and_clean does it once; the attack-injection
    recompute path re-sorts only the small affected subset before calling
    this). Once sorted that way, a plain whole-column .shift(1) already
    gives the correct "previous row" for every row EXCEPT the first row of
    each aircraft's block -- there it leaks in the last row of the
    PREVIOUS aircraft. `new_group` finds and masks exactly those rows.
    Same result as groupby().shift(1), no group index ever built.
    """
    new_group = (df["icao24"] != df["icao24"].shift(1)).to_numpy(copy=True)
    new_group[0] = True  # first row overall has no predecessor

    def shifted(col):
        s = df[col].shift(1)
        arr = s.to_numpy(copy=True)
        arr[new_group] = np.nan
        return pd.Series(arr, index=df.index)

    df["prev_lat"] = shifted("lat")
    df["prev_lon"] = shifted("lon")
    df["prev_time"] = shifted("time")
    df["prev_baroalt"] = shifted("baroaltitude")

    dt = df["time"] - df["prev_time"]
    # Guard against near-zero dt (duplicate/near-duplicate timestamps are
    # common in crowdsourced ADS-B and produce physically absurd implied
    # speeds when divided by a sub-second interval). Anything under 1s
    # is treated as unreliable for a kinematic estimate.
    #
    # ALSO guard against multi-day session breaks: when combining several
    # non-consecutive days (e.g. multiple Mondays' worth of data), the
    # same icao24 can legitimately reappear days/weeks later. That's a
    # real absence, not a single continuous kinematic gap -- treating it
    # as one would compute a technically-tiny "implied speed" (huge
    # distance / huge time) that's meaningless, and would make
    # inter_arrival_s show a multi-day outlier that isn't comparable to
    # real within-day coverage gaps (max ~hours). SESSION_BREAK_SECONDS
    # is set well above any realistic single-region coverage gap but well
    # below the gap between separate collection days.
    is_session_break = dt.abs() > SESSION_BREAK_SECONDS
    dt_safe = dt.where((dt.abs() >= 1) & (~is_session_break), np.nan)

    dist_m = haversine_m(df["prev_lat"], df["prev_lon"], df["lat"], df["lon"])
    df["implied_speed_mps"] = dist_m / dt_safe  # Tier A: position-derived speed

    # Consistency check: does implied speed agree with broadcast velocity?
    df["speed_consistency_delta"] = (df["implied_speed_mps"] - df["velocity"]).abs()

    # Altitude-rate consistency: does vertrate agree with baroaltitude change?
    implied_vrate = (df["baroaltitude"] - df["prev_baroalt"]) / dt_safe
    df["vrate_consistency_delta"] = (implied_vrate - df["vertrate"]).abs()

    # Inter-arrival time per aircraft (flooding / gap detection). Session
    # breaks are set to NaN here too -- a multi-day absence isn't a
    # "coverage gap" in the same sense is_low_coverage_region models, and
    # would otherwise appear as an extreme, misleading outlier.
    df["inter_arrival_s"] = dt.where(~is_session_break, np.nan)

    # Turn rate from heading, wrapped to [-180, 180]
    prev_heading = shifted("heading")
    dh = (df["heading"] - prev_heading + 180) % 360 - 180
    df["turn_rate_deg_s"] = dh / dt_safe

    df.drop(columns=["prev_lat", "prev_lon", "prev_time", "prev_baroalt"], inplace=True)
    return df


# ---------------------------------------------------------------------------
# 3. TIER B -- sensor / context features (drive FL client partitioning)
# ---------------------------------------------------------------------------

def compute_region_cell(lat: pd.Series, lon: pd.Series, cellsize: int = 10) -> pd.Series:
    return (lat // cellsize).astype(int).astype(str) + "_" + (lon // cellsize).astype(int).astype(str)


def add_tier_b_features(df: pd.DataFrame, region_baseline: pd.Series = None):
    """
    If region_baseline is None, this is the initial call on clean data:
    compute region_cell, then compute + return a per-region average
    inter_arrival_s baseline (this is what "normal coverage density" looks
    like for that region, before any attacks are injected).

    If region_baseline is provided (post-injection recompute for attacks
    that move an aircraft's position, e.g. gps_spoof), region_cell is
    recomputed from the new lat/lon, and the ORIGINAL clean baseline is
    reused via lookup -- so a spoofed row is judged against the coverage
    density of the region it's now claiming to be in, using a baseline
    that isn't itself corrupted by attack data.
    """
    # This sample export doesn't include the `sensors` array (that's a
    # Trino-only column) -- if/when you pull from Trino, explode `sensors`
    # here to get per-receiver client IDs. For now, bucket by lat/lon grid
    # cell as a stand-in "virtual sensor region" for FL partitioning.
    df["region_cell"] = compute_region_cell(df["lat"], df["lon"])
    new_group = (df["icao24"] != df["icao24"].shift(1)).to_numpy(copy=True)
    new_group[0] = True
    prev_squawk = df["squawk"].shift(1).to_numpy(copy=True)
    prev_squawk[new_group] = np.nan
    df["squawk_changed"] = pd.Series(prev_squawk, index=df.index) != df["squawk"]

    if region_baseline is None:
        region_baseline = df.groupby("region_cell")["inter_arrival_s"].mean()

    fallback = region_baseline.median()
    df["region_avg_inter_arrival"] = df["region_cell"].map(region_baseline).fillna(fallback)
    # Regions in the sparser quarter (by average inter-arrival time) are
    # naturally low-coverage -- e.g. oceanic corridors tracked by
    # wide-area/satellite reception rather than dense terrestrial
    # receivers. A large gap there is normal, not evidence of replay.
    sparse_cutoff = region_baseline.quantile(0.75)
    df["is_low_coverage_region"] = df["region_avg_inter_arrival"] > sparse_cutoff

    return df, region_baseline


# ---------------------------------------------------------------------------
# 4. TIER C -- identity / metadata features (weight lowest in final model)
# ---------------------------------------------------------------------------

SPECIAL_SQUAWKS = {"7500": "hijack", "7600": "radio_failure", "7700": "emergency"}


def add_tier_c_features(df: pd.DataFrame) -> pd.DataFrame:
    df["squawk_str"] = df["squawk"].apply(
        lambda x: str(int(x)).zfill(4) if pd.notna(x) else None
    )
    df["special_squawk"] = df["squawk_str"].map(SPECIAL_SQUAWKS).fillna("none")

    # icao24 should be a 6-hex-digit string; flag anything malformed
    df["icao24_valid_format"] = df["icao24"].astype(str).str.match(r"^[0-9a-f]{6}$")

    # callsign_valid_format removed -- callsign column dropped at load
    # time (see DROP_COLS in section 1)

    # How many times does this icao24 appear in THIS dataset? Ghost aircraft
    # (fabricated icao24s) are almost always singletons -- this is the real
    # detectable signal for them, since kinematic features like
    # inter_arrival_s are NaN for a row with no prior sighting to compare
    # against.
    df["track_message_count"] = df["icao24"].map(df["icao24"].value_counts())

    return df


# ---------------------------------------------------------------------------
# 5. ATTACK INJECTION
# ---------------------------------------------------------------------------

def apply_attack_to_subset(combined: pd.DataFrame, idx, attack_type: str, rng) -> tuple:
    """
    Mutates `combined` IN PLACE at the given (already-selected, DISJOINT
    across attack types) row positions `idx`, and sets label/attack_type
    for exactly those rows.

    This replaces the old inject_attack(), which took a full copy of the
    whole chunk per attack type and returned a full-size variant -- six
    full-chunk-size pieces (benign + 5 variants) were then written to
    disk per chunk, so the output file ended up ~6x the size of the input
    regardless of how small `rate` was. Operating in place on ONE shared
    dataframe, at disjoint row slices, means the chunk stays ~1x input
    size (plus the small number of new rows flooding legitimately adds).

    Returns (affected_icao, new_rows):
      affected_icao -- set of icao24 values whose Tier A kinematic
        features need recomputing afterward (only relevant for attack
        types that touch time/lat/lon/velocity/vertrate/icao24).
      new_rows -- a small DataFrame of newly duplicated rows to append
        once at the end (flooding only), or None for every other type.
    """
    n = len(idx)
    if n == 0:
        return set(), None

    orig_icao_for_ghost = None

    if attack_type == "gps_spoof":
        # Randomized severity per row instead of a single fixed range --
        # subtle shifts (~5km) are deliberately hard to distinguish from
        # sensor noise, moderate (~50km) and severe (~200km+) escalate to
        # the old always-obvious behavior. rng.uniform() returns float64
        # by default; lat/lon are float32 (see READ_DTYPES) so the noise
        # must be cast down to match.
        severity = rng.choice(list(GPS_SPOOF_SEVERITY_DEG.keys()), n, p=GPS_SPOOF_SEVERITY_PROBS)
        max_shift_deg = np.array([GPS_SPOOF_SEVERITY_DEG[s] for s in severity])
        combined.loc[idx, "lat"] += (rng.uniform(-1, 1, n) * max_shift_deg).astype("float32")
        combined.loc[idx, "lon"] += (rng.uniform(-1, 1, n) * max_shift_deg).astype("float32")

    elif attack_type == "velocity_spoof":
        combined.loc[idx, "velocity"] *= rng.uniform(*VELOCITY_SPOOF_MULTIPLIER_RANGE, n).astype("float32")
        combined.loc[idx, "vertrate"] = (
            combined.loc[idx, "vertrate"].fillna(0)
            + rng.uniform(*VELOCITY_SPOOF_VERTRATE_BOOST_RANGE, n).astype("float32")
        )

    elif attack_type == "replay":
        shift_s = rng.integers(*REPLAY_SHIFT_RANGE_S, n)
        combined.loc[idx, "time"] = combined.loc[idx, "time"] - shift_s
        combined.loc[idx, "lastcontact"] = combined.loc[idx, "lastcontact"] - shift_s

    elif attack_type == "ghost_aircraft":
        # Capture the REAL icao24 before overwriting -- the recompute
        # step needs to know which original aircraft group just lost a
        # row (its remaining rows' Tier A deltas shift), and this is the
        # only point where that original value is still available.
        orig_icao_for_ghost = combined.loc[idx, "icao24"].copy()
        fake_ids = [f"{rng.integers(0xf00000, 0xffffff):06x}" for _ in range(n)]
        combined.loc[idx, "icao24"] = fake_ids

    elif attack_type == "flooding":
        pass  # duplication happens below, after labeling

    elif attack_type == "signal_loss":
        raise ValueError(
            "signal_loss removes rows rather than labeling them and isn't "
            "compatible with the disjoint-slice model -- keep it out of "
            "ATTACK_TYPES (see the comment there) until it gets its own path."
        )

    else:
        raise ValueError(f"Unknown attack_type: {attack_type}")

    combined.loc[idx, "label"] = 1
    combined.loc[idx, "attack_type"] = attack_type

    new_rows = None
    if attack_type == "flooding":
        # Duplicate the selected (already-labeled) rows a RANDOM number of
        # times (2-10, previously always exactly 5) with a RANDOM time
        # offset per row (previously always exactly 0.1s) -- a slow
        # trickle of duplicates looks very different from a rapid burst,
        # and treating every flooding instance identically meant a model
        # only ever had to learn one specific pattern.
        flood_rows = combined.loc[idx]
        dup_counts = rng.integers(*FLOODING_DUP_COUNT_RANGE, n)
        offsets = rng.uniform(*FLOODING_OFFSET_RANGE_S, n)
        max_dup = int(dup_counts.max()) if n > 0 else 0
        copies = []
        for i in range(1, max_dup + 1):
            mask = dup_counts >= i
            if not mask.any():
                continue
            c = flood_rows.loc[mask].copy()
            c["time"] = c["time"] + i * offsets[mask]
            copies.append(c)
        new_rows = pd.concat(copies, ignore_index=True) if copies else None

    if attack_type == "ghost_aircraft":
        affected_icao = set(orig_icao_for_ghost.unique()) | set(combined.loc[idx, "icao24"].unique())
    else:
        affected_icao = set(combined.loc[idx, "icao24"].unique())

    return affected_icao, new_rows


# ---------------------------------------------------------------------------
# 6. MAIN (chunked -- see note below on why)
# ---------------------------------------------------------------------------

# How many consecutive hourly files to process together as one chunk.
# Bigger chunks preserve more cross-file kinematic continuity (see
# SESSION_BREAK_SECONDS) but cost more memory. Sized for ~2.1M rows/file
# (this dataset's average) so a chunk's fully-engineered DataFrame stays a
# few GB, leaving headroom under a 12GB WSL cap even while one attack-type
# variant is being built alongside it. If you still see `Killed` with this
# setting, lower it (e.g. to 3) before anything else -- that's the single
# knob that trades memory for a few more passes over the data.
CHUNK_FILES = 2

ATTACK_TYPES = ["gps_spoof", "velocity_spoof", "replay", "ghost_aircraft", "flooding"]


def _finalize_and_write(piece: pd.DataFrame, out_path: str, write_header: bool) -> tuple:
    """
    Apply the row-wise-only outlier handling / flight-phase flag, then
    write straight to disk. Returns (n_rows, n_attack, per_type_counts)
    -- per_type_counts feeds the early-stop check in main().
    """
    piece = piece.copy()
    piece["is_physically_implausible"] = (
        piece["implied_speed_mps"] > MAX_PLAUSIBLE_SPEED_MPS
    )
    noisy_benign_mask = (piece["label"] == 0) & piece["is_physically_implausible"]
    piece = piece.loc[~noisy_benign_mask].reset_index(drop=True)
    piece["is_climbing_or_descending"] = piece["vertrate"].abs() > CLIMB_DESCENT_VERTRATE_MPS

    n_rows = len(piece)
    n_attack = int(piece["label"].sum())
    type_counts = piece["attack_type"].value_counts().to_dict()
    piece.to_csv(out_path, mode="w" if write_header else "a",
                 header=write_header, index=False)
    return n_rows, n_attack, type_counts


def process_chunk(paths: list, chunk_label: str, out_path: str, write_header: bool,
                   rate: float = 0.02, seed: int = RNG_SEED) -> tuple:
    """
    Run clean -> feature-engineer -> inject on ONE chunk of files, then
    write ONE combined dataframe to disk (not six full-chunk-size pieces).

    Each attack type is assigned a DISJOINT `rate`-fraction slice of the
    chunk's rows (5 types x 2% = 10% of rows become attacks, 90% stay
    benign) and mutated in place on a single shared dataframe. Tier A
    kinematic features, track_message_count, and gps_spoof's region
    context are then recomputed ONCE over the union of every icao24
    touched by any attack type -- not once per attack type. Net effect:
    peak memory and output size both stay close to 1x the engineered
    chunk (plus flooding's small number of genuinely new duplicate rows),
    instead of the ~6x a full-copy-per-attack-type approach produces.
    """
    df = load_and_clean(paths)
    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    print(f"  [{chunk_label}] {len(df):,} rows after cleaning ({mem_mb:,.0f} MB)")

    df = add_tier_a_features(df)
    df, region_baseline = add_tier_b_features(df)
    df = add_tier_c_features(df)

    df["label"] = 0
    df["attack_type"] = "benign"

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(df.index.to_numpy())
    per_type_n = int(len(df) * rate)

    assignments = {}
    cursor = 0
    for atk in ATTACK_TYPES:
        assignments[atk] = shuffled[cursor:cursor + per_type_n]
        cursor += per_type_n

    all_affected_icao = set()
    gf_affected_icao = set()  # ghost_aircraft/flooding specifically -- these change
                               # an aircraft's TOTAL row count, so track_message_count
                               # must be recomputed for every row of that icao24,
                               # regardless of which attack type any individual row
                               # of theirs happens to carry.
    all_new_rows = []
    for atk in ATTACK_TYPES:
        affected_icao, new_rows = apply_attack_to_subset(df, assignments[atk], atk, rng)
        all_affected_icao |= affected_icao
        if atk in ("ghost_aircraft", "flooding"):
            gf_affected_icao |= affected_icao
        if new_rows is not None:
            all_new_rows.append(new_rows)

    if all_new_rows:
        df = pd.concat([df] + all_new_rows, ignore_index=True)

    if all_affected_icao:
        tier_a_cols = [
            "implied_speed_mps", "speed_consistency_delta",
            "vrate_consistency_delta", "inter_arrival_s", "turn_rate_deg_s",
        ]
        affected_mask = df["icao24"].isin(all_affected_icao)
        subset = df.loc[affected_mask].drop(columns=tier_a_cols)
        subset = subset.sort_values(["icao24", "time"])
        subset = add_tier_a_features(subset)
        df.loc[subset.index, tier_a_cols] = subset[tier_a_cols].values

    if gf_affected_icao:
        # track_message_count only actually changes for ghost_aircraft
        # (new icao24s, essentially always singletons) and flooding
        # (duplicated rows inflate an existing icao24's count) -- recompute
        # for EVERY row of an affected icao24 (icao24-only mask, no
        # attack_type filter), since a single aircraft's rows can be split
        # across several different attack-type assignments and all of them
        # share the same true row count.
        gf_mask = df["icao24"].isin(gf_affected_icao)
        counts = df.loc[gf_mask, "icao24"].value_counts()
        df.loc[gf_mask, "track_message_count"] = df.loc[gf_mask, "icao24"].map(counts)

    gps_idx = assignments["gps_spoof"]
    if len(gps_idx) > 0:
        df.loc[gps_idx, "region_cell"] = compute_region_cell(
            df.loc[gps_idx, "lat"], df.loc[gps_idx, "lon"]
        )
        fallback = region_baseline.median()
        df.loc[gps_idx, "region_avg_inter_arrival"] = (
            df.loc[gps_idx, "region_cell"].map(region_baseline).fillna(fallback)
        )
        sparse_cutoff = region_baseline.quantile(0.75)
        df.loc[gps_idx, "is_low_coverage_region"] = (
            df.loc[gps_idx, "region_avg_inter_arrival"] > sparse_cutoff
        )

    n_rows, n_attack, type_counts = _finalize_and_write(df, out_path, write_header)
    del df
    gc.collect()

    return n_rows, n_attack, type_counts


def main(paths, chunk_files=CHUNK_FILES, rate=0.02, min_rows_per_class=None,
         out_path=None, seed=RNG_SEED):
    if isinstance(paths, str):
        paths = [paths]
    paths = sorted(paths)  # filenames sort chronologically (states_YYYY-MM-DD-HH.csv)

    chunks = [paths[i:i + chunk_files] for i in range(0, len(paths), chunk_files)]
    if out_path is None:
        out_path = (paths[0].rsplit(".", 1)[0] + "_labeled.csv" if len(paths) == 1
                    else "combined_labeled.csv")

    print(f"Processing {len(paths)} file(s) in {len(chunks)} chunk(s) "
          f"of up to {chunk_files} file(s) each ...")
    if min_rows_per_class:
        print(f"Early-stop enabled: will stop once every attack class has "
              f">= {min_rows_per_class:,} rows (checked after each chunk).")

    total_rows = 0
    total_attack_rows = 0
    write_header = True
    running_type_counts = {atk: 0 for atk in ATTACK_TYPES}

    for i, chunk_paths in enumerate(chunks, 1):
        label = f"chunk {i}/{len(chunks)}"
        n_rows, n_attack, type_counts = process_chunk(
            chunk_paths, label, out_path, write_header, rate=rate, seed=seed
        )
        write_header = False
        total_rows += n_rows
        total_attack_rows += n_attack
        for atk in ATTACK_TYPES:
            running_type_counts[atk] += type_counts.get(atk, 0)

        counts_str = ", ".join(f"{atk}={running_type_counts[atk]:,}" for atk in ATTACK_TYPES)
        print(f"  [{label}] -> {n_rows:,} rows, {n_attack:,} attack rows "
              f"({n_attack / n_rows:.2%}) | running totals: {counts_str}")

        if min_rows_per_class:
            if all(running_type_counts[atk] >= min_rows_per_class for atk in ATTACK_TYPES):
                print(f"\nEarly stop: every attack class has reached "
                      f">= {min_rows_per_class:,} rows after {i}/{len(chunks)} chunks "
                      f"-- skipping the remaining {len(chunks) - i} chunk(s).")
                break

    print(f"\nFinal dataset: {total_rows:,} rows, {total_attack_rows:,} attack rows "
          f"({total_attack_rows / total_rows:.2%})")
    for atk in ATTACK_TYPES:
        print(f"  {atk}: {running_type_counts[atk]:,} rows")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="OpenSky FL-IDS pipeline: clean, engineer features, inject attacks.")
    p.add_argument("inputs", nargs="+",
                    help="Input CSV(s), e.g. states_2017-06-05-*.csv (shell glob)")
    p.add_argument("--chunk-files", type=int, default=CHUNK_FILES,
                    help=f"Hourly files processed together per chunk (default {CHUNK_FILES}). "
                         f"Lower this first if you still see the process get Killed.")
    p.add_argument("--rate", type=float, default=0.02,
                    help="Fraction of each chunk's rows assigned to EACH attack type "
                         "(default 0.02 = 2%% per type, 10%% total attack rows per chunk).")
    p.add_argument("--min-rows-per-class", type=int, default=None,
                    help="Stop processing further chunks once every attack class has "
                         "at least this many rows. Default: process all input files.")
    p.add_argument("--output", default=None,
                    help="Output CSV path. Default: <first_input>_labeled.csv, or "
                         "combined_labeled.csv for multiple inputs.")
    p.add_argument("--seed", type=int, default=RNG_SEED)
    args = p.parse_args()

    main(args.inputs, chunk_files=args.chunk_files, rate=args.rate,
         min_rows_per_class=args.min_rows_per_class, out_path=args.output, seed=args.seed)
