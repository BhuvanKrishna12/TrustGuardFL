# Data pipeline overview

`opensky_fl_ids_pipeline.py` turns raw OpenSky state-vector CSVs into a labelled,
feature-engineered dataset with synthetic attacks injected. This page gives the structure;
`docs/PIPELINE_V3_CHANGES.md` describes the leakage audit and every v3 correction in detail.

## Stages

Each chunk of `--chunk-files` consecutive hourly files (default 2; the reproduction uses 1)
goes through:

```
load_and_clean()          read, keep timestamps float64, other numerics float32, drop callsign
add_tier_a_features()     kinematic features per aircraft, sorted by (icao24, time)
add_tier_b_features()     region cell, regional inter-arrival baseline, squawk change
add_tier_c_features()     identity features (icao24 format, message count)
apply_attack_to_subset()  the five attacks, each on its own disjoint slice of rows
recompute                 Tier A (and in v3 the causal message count and squawk change)
                          for every aircraft touched by an attack
_finalize_and_write()     derived flags, append to the output CSV
```

Files within a chunk are concatenated before feature engineering, so per-aircraft features
stay continuous across hour boundaries. Gaps longer than one hour
(`SESSION_BREAK_SECONDS`) start a new session, so separate collection days are never treated
as one flight. The RNG is re-seeded for every chunk, which makes a resumed day
(`experiments/reprocess_mv.py`, one hour per chunk) byte-identical to an uninterrupted run.

## Attack injection

Each chunk is permuted once and each attack type takes a disjoint `--rate` share of the rows
(default 2% per type, so about 10% attack rows plus the flooding duplicates). All five types
mutate one shared DataFrame in place, so every row carries exactly one label and the output
stays about the size of the input. Magnitudes are drawn per row from ranges, so each class
contains subtle as well as obvious cases (constants at the top of the pipeline file).

## Features

Thirteen numeric features are used by the models (`FEATURE_COLS` in
`federated_simulation.py`); `region_cell` is the client key, not a feature. The leakage audit
(`leakage_audit.py`) defines the **core** set of seven features that survive the audit:
`implied_speed_mps`, `speed_consistency_delta`, `vrate_consistency_delta`,
`inter_arrival_s`, `turn_rate_deg_s`, `region_avg_inter_arrival` and `is_stale`. The full set
adds `track_message_count`, `squawk_changed`, `icao24_valid_format`,
`is_low_coverage_region`, `is_physically_implausible` and `is_climbing_or_descending`.

## Implementation notes that still hold for v3

* **Timestamp precision.** `time`, `lastposupdate` and `lastcontact` are kept as float64.
  float32 cannot represent Unix timestamps exactly, which would corrupt every time-difference
  feature.
* **Malformed values.** A file whose numeric columns hold stray text is re-read with numeric
  coercion (bad values become missing) instead of failing.
* **Near-zero time gaps.** Rate features use a 0.1 s floor on the time difference
  (`MIN_DT_FLOOR_S`). Without it, flooding duplicates left the four kinematic rate features
  blank, and the blank pattern alone identified the class (the diagnostics in
  `tools/diagnostics/` detect this signature).
* **Regional baseline per chunk.** `region_avg_inter_arrival` and `is_low_coverage_region`
  are computed from the clean rows of the current chunk, not the whole dataset, to bound
  memory. A gps_spoof row is looked up against the clean baseline of the cell it moves to.
* **Memory.** Peak memory scales with the chunk size; lower `--chunk-files` if a run is
  killed for lack of memory.

## Versions

`--pipeline-version v3` (default) is the corrected pipeline used in the paper. `v2` reproduces
the earlier pipeline byte for byte. `mv` writes the v3 file plus the extra columns from which
`experiments/variants.py` rebuilds v2 and every one-correction-at-a-time mix for the ablation.
