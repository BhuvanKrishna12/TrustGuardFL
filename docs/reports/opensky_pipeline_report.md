# `opensky_fl_ids_pipeline.py` — Pipeline Report

**Purpose:** turns raw OpenSky Network state-vector CSVs into a labeled,
feature-engineered dataset with synthetic attacks injected, suitable for
training an intrusion detection model (centralized or federated).

---

## 1. What It Does, End to End

```
raw hourly CSV(s)
    -> load_and_clean()        parse, downcast dtypes, drop unneeded columns
    -> add_tier_a_features()   kinematic features (speed, turn rate, timing)
    -> add_tier_b_features()   sensor/context features (region, squawk)
    -> add_tier_c_features()   identity/metadata features (icao24, message count)
    -> apply_attack_to_subset()  inject 5 attack types into disjoint row slices
    -> _finalize_and_write()   outlier handling, write to disk
```

Processing happens in **chunks** of `--chunk-files` raw hourly files at a
time (default 2), so memory use stays bounded regardless of how many hours
of data are being processed in one run.

---

## 2. Engineered Features (15 total)

| Tier | Feature | What it measures |
|---|---|---|
| A — kinematic | `implied_speed_mps` | Speed derived from position change (haversine distance / time delta) |
| A | `speed_consistency_delta` | \|implied speed − broadcast velocity\| |
| A | `vrate_consistency_delta` | \|altitude-derived climb rate − broadcast vertical rate\| |
| A | `inter_arrival_s` | Time since this aircraft's last sighting |
| A | `turn_rate_deg_s` | Heading change rate |
| B — sensor/context | `region_cell` | Lat/lon grid cell (used for FL client partitioning downstream, not itself a numeric model feature) |
| B | `region_avg_inter_arrival` | Average inter-arrival time for this region (coverage-density baseline) |
| B | `is_low_coverage_region` | Whether this region is naturally sparse (top quartile by avg gap) |
| B | `squawk_changed` | Whether the squawk code changed since last sighting |
| C — identity/metadata | `special_squawk` | Hijack (7500) / radio failure (7600) / emergency (7700) / none |
| C | `icao24_valid_format` | Whether icao24 matches the valid 6-hex-digit format |
| C | `track_message_count` | Total rows for this icao24 in the processed data |
| Derived | `is_stale` | OpenSky's own staleness flag (time − lastcontact > 15s) |
| Derived | `is_physically_implausible` | implied_speed_mps > 400 m/s (sensor-noise filter) |
| Derived | `is_climbing_or_descending` | \|vertrate\| > 2.5 m/s |

13 of these are numeric and usable directly as model input; `region_cell`
is a string identifier, `special_squawk` is categorical text.

---

## 3. Attack Injection Design

### 3.1 Disjoint-slice mechanism

Each attack type is assigned a random, non-overlapping 2% slice of a
chunk's rows (`rate` parameter, default 0.02). 5 attack types × 2% = 10%
of rows become attacks; the remaining 90% stay benign. All five types
mutate one **shared** dataframe in place — this is a deliberate design
choice (see Section 4.1).

### 3.2 The five attack types

| Attack | Mechanism | Recomputation triggered |
|---|---|---|
| `gps_spoof` | Position (lat/lon) shifted | Tier A kinematic features + region context, for the affected rows |
| `velocity_spoof` | Broadcast velocity/vertical rate multiplied | Tier A kinematic features |
| `replay` | Timestamp shifted backward | Tier A kinematic features (via changed time deltas) |
| `ghost_aircraft` | icao24 replaced with a fabricated hex ID | `track_message_count` for both the original and fake ID |
| `flooding` | Selected rows duplicated multiple times with small time offsets | `track_message_count` and Tier A features for the affected aircraft |

`signal_loss` (removing rows rather than labeling them) is defined as a
concept but intentionally excluded from the active attack list — it
doesn't fit the disjoint-labeling model and would need a separate
evaluation path.

### 3.3 Severity randomization

Originally every attack instance used one fixed magnitude (e.g.
`gps_spoof` always ±2°), meaning every example of a given attack type was
equally easy to detect. This was identified as a real limitation — a
model trained on it could learn "detect large obvious jumps" without ever
seeing a stealthy attack. Fixed by randomizing severity per row:

| Attack | Before | After |
|---|---|---|
| `gps_spoof` | Always ±2° | 3-tier: subtle (±0.05°, ~34%) / moderate (±0.5°, ~33%) / severe (±2°, ~33%) |
| `velocity_spoof` | Always 2-5x multiplier | 1.1-8x (now includes near-invisible spoofs) |
| `replay` | Always 60-600s shift | 2-900s (now includes near-instant shifts) |
| `flooding` | Always exactly 5 duplicates, fixed 0.1s offset | 2-10 duplicates (randomized per row), 0.01-0.5s offset (randomized) |

All four randomized signatures were verified empirically (not just by
reading the code) — confirmed genuine spread across severity tiers, and
confirmed the aggregate feature signal is still clearly present despite
the added subtlety.

---

## 4. Key Engineering Decisions and Fixes

### 4.1 The 6x output size bug (fixed)

An earlier version copied the **entire chunk** per attack type and wrote
all 6 pieces (benign + 5 variants) to disk — meaning output was ~6x input
size regardless of how small the attack rate was, since 98% of every
"variant" piece was just re-written benign rows. Root cause: attack
injection needs each row to end up with exactly one label, but the naive
implementation didn't enforce that.

**Fix:** mutate one shared dataframe in place, at disjoint row slices per
attack type, so every row gets exactly one label and the chunk stays ~1x
its input size (plus flooding's small number of genuinely new duplicate
rows). Verified on real data: 4,615,352 output rows from 4,615,352-ish
raw input rows for one chunk — not the ~27M the old design would have
produced.

### 4.2 Malformed source CSV handling (fixed)

Some raw hourly files contain genuinely ragged rows (inconsistent field
counts) that crash pandas' fast C parser with an `IndexError`. Fixed with
a two-pass load: fast C-engine parse first; if that fails on every
encoding attempted, fall back to the slower `python` engine with
`on_bad_lines='skip'`. Critically, `index_col=False` is required in the
fallback — without it, pandas' `python` engine can misinterpret a single
ragged row as evidence the whole file has an implicit leading index
column, silently shifting every other row's data by one column. This was
caught by deliberately forcing the bug with a synthetic malformed file
and confirming the fix resolves it.

### 4.3 Dtype and memory efficiency

- Raw CSVs are read with explicit `float32` dtypes for lat/lon/velocity/
  heading/vertrate/altitude columns from the start — Unix timestamp
  columns (`time`, `lastposupdate`, `lastcontact`) are deliberately kept
  at `float64`, since `float32` cannot represent Unix timestamps precisely
  enough without corrupting every diff-based feature.
- `callsign` is dropped entirely at load time (not needed by any
  engineered feature, and expensive to keep as a string column across
  tens of millions of rows).
- Tier A feature recomputation after an attack is scoped to only the
  affected aircraft (via `icao24` membership), not recomputed across the
  whole chunk — this avoids an unnecessary full-dataframe sort/recompute
  per attack type.

### 4.4 Chunking and early-stop controls

- `--chunk-files`: how many raw hourly files are processed together before
  writing output and freeing memory. Lower this if a machine still runs
  out of memory; the new hourly files (2019 dataset) run larger
  (up to ~450MB each) than the files the original default was tuned
  around.
- `--min-rows-per-class`: stops processing further chunks once every
  attack class has reached this many total rows, avoiding the need to
  process an entire multi-day dataset when a fraction is sufficient.
  Tracked and printed per chunk as running totals.

---

## 5. Validated Output (Current Datasets)

Two versions of one full day (`2019-01-07`, 24 hourly files) have been
processed and validated (via `validate_labeled_dataset.py`):

| | Fixed-severity | Randomized-severity (v2, current) |
|---|---|---|
| Total rows | 42,483,016 | 43,439,234 |
| benign | 34,407,866 (80.99%) | 34,555,449 (79.55%) |
| flooding | 4,845,090 (6.00x ratio) | 5,653,725 (7.00x ratio — expected, wider duplicate-count range) |
| gps_spoof / velocity_spoof / replay / ghost_aircraft | 807,515 each | 807,515 each (row counts unchanged; only magnitude randomized) |

Both datasets pass all validation checks: exact label counts match the
pipeline's own printed totals, and each attack type's rows genuinely
exhibit that attack's expected feature signature (not just a correct
label sitting on unmutated data).

---

## 6. CLI Reference

```bash
python3 opensky_fl_ids_pipeline.py <input_glob_or_files> \
  --chunk-files 1 \
  --rate 0.02 \
  --min-rows-per-class 40000 \
  --output labeled_output.csv \
  --seed 42
```

| Flag | Default | Purpose |
|---|---|---|
| `--chunk-files` | 2 | Raw hourly files processed together per chunk |
| `--rate` | 0.02 | Fraction of rows assigned to EACH attack type (2% × 5 types = 10% attack rows per chunk) |
| `--min-rows-per-class` | None (process everything) | Early-stop once every attack class reaches this row count |
| `--output` | auto-named | Output CSV path |
| `--seed` | 42 | Random seed |

---

## 7. Known Limitations

- `region_avg_inter_arrival` / `is_low_coverage_region` are computed from
  each **chunk's** clean data only, not the full dataset — a deliberate
  memory tradeoff, meaning "low coverage" is judged relative to that
  chunk's time window rather than the whole dataset. Worth a line in any
  methodology section that cites this feature.
- Attack magnitudes, even randomized, are still generated by fixed
  parametric formulas — real-world attacker behavior may not match this
  distribution.
- `signal_loss` (dropped rows) remains unimplemented as an active attack
  type.
- The validator (`validate_labeled_dataset.py`) checks exact label counts
  and per-attack feature signatures, but has not been extended to check
  for blank/corrupted label values specifically — worth adding if a
  similar data-quality issue is ever suspected in the raw OpenSky source
  files (an analogous issue was found and handled in an unrelated
  dataset earlier in this project, so it's a real failure mode worth
  guarding against, just not yet checked for here).
