# Pipeline v3: leakage fixes and feature audit

`opensky_fl_ids_pipeline.py` now takes `--pipeline-version {v2,v3}` (default `v3`).

* `v2` reproduces the previous pipeline byte-for-byte. Verified with `cmp` against
  `git show HEAD:opensky_fl_ids_pipeline.py` on a synthetic two-hour input, for
  `--chunk-files 2` (default) and `--chunk-files 1`, and for seeds 42 and 7.
* `v3` applies the changes below. The 0.1 s denominator floor (the flooding
  sentinel fix) is kept in both versions.

Reprocess a day with v3 (same style as before):

```
python3 opensky_fl_ids_pipeline.py states_2019-02-04-*.csv --chunk-files 1 --output out.csv
python3 opensky_fl_ids_pipeline.py states_2019-02-04-*.csv --chunk-files 1 --output out.csv --pipeline-version v3   # explicit
python3 opensky_fl_ids_pipeline.py states_2019-02-04-*.csv --chunk-files 1 --output out_v2.csv --pipeline-version v2  # old behaviour
python3 leakage_audit.py out.csv --out audit_out
```

The output schema is unchanged (same columns, same order), so downstream scripts read v3 files without changes.

---

## 1. Flooding labels

**Old (v2).** For each row in the flooding slice, the ORIGINAL row was relabelled
`label=1, attack_type='flooding'` and k duplicates (k uniform in 2..10) were appended,
also labelled flooding. The original is an untouched real message, so about 1/7 of
flooding rows (1/(1+E[k]) with E[k] = 6) carried
no attack signal at all: a recall ceiling of roughly 6/7 by construction, and benign-looking
rows inside the flooding class.

**New (v3).** The selected original stays `label=0, attack_type='benign'`. Only the
appended duplicates are labelled flooding.

* The slice still consumes its 2% of the permutation, and the RNG draws for the
  duplicates (`dup_counts`, then `offsets`) happen at the same point and in the same order
  as v2. No RNG draw depended on the label assignment, so every other attack slice is
  identical. Verified by instrumenting `apply_attack_to_subset`: the selected indices and
  the PCG64 state before and after each of the five attack types are identical between v2
  and v3.
* "Selected" state of the original: in v2 the only effect of being relabelled was the
  label itself, plus surviving the v2 benign speed filter (see change 4). In v3 the
  original carries no attack flag. Its aircraft is still added to the recompute set
  because duplicates were inserted into its track, which is required for correct
  features on the whole track; the original's own Tier A values do not change because
  every duplicate is strictly later in time (offset >= 0.01 s), so its predecessor is
  the same as before injection.

**Why.** Labels should mark rows that differ from real traffic. A real message that a
flooder later copied is still a real message.

**Expected effect.** The flooding class shrinks by exactly the slice size (synthetic test:
8,273 to 7,108 rows, difference 1,165 = 2% of 58,274 clean rows) and the benign class
grows by the same amount. Flooding recall and AUC should rise (single-feature AUC of
`inter_arrival_s` for flooding vs benign: 0.930 in v2, 0.998 in v3 on the synthetic data);
flooding precision is no longer capped by benign-looking originals. Natural class shares
in Table "Class Mix" change (flooding share drops by about 1.9 points per 2-file chunk).

## 2. track_message_count

**Old (v2).** `value_counts()` of the icao24 over the whole chunk (two hourly files by
default), mapped onto every row: a row near the start of a flight already knew how many
messages the aircraft would send later. Recomputed after injection only for aircraft hit
by ghost_aircraft or flooding.

**New (v3).** Causal running count: the 1-based number of messages seen so far for that
icao24 within the chunk, including the current row (first message = 1). Ordering is
exactly the one the Tier A features use for "previous message": the chunk sorted by
`["icao24", "time"]` (`causal_track_message_count`).

* Stage: kept where v2 has it. It is computed on the clean chunk in `add_tier_c_features`
  (before injection, like v2) and then recomputed after injection for affected aircraft.
  Because a running count depends on the ORDER of messages, not just their total, v3
  recomputes it for every aircraft touched by any attack (replay moves rows in time,
  ghost removes a row from one track and creates a new one, flooding inserts rows), on
  the same sorted subset that the Tier A recompute uses, so the count and the Tier A
  "previous row" always refer to the same sequence.
* The count restarts at every chunk boundary (every hour with `--chunk-files 1`). That is
  a chunking artifact, not a label artifact; documented, not changed.

**Why.** The v2 count used future observations and could not be computed online.

**Expected effect.** Ghost rows still get count 1 (a fabricated address is a one-row
track), so the feature stays construction-revealing for ghost_aircraft (synthetic AUC
0.999 vs benign) and is excluded from the core feature set. Its gain in the full model
should fall because it no longer encodes total track length.

## 3. squawk_changed

**Old (v2).** `prev_squawk != squawk` with `prev_squawk = NaN` on each aircraft's first
row. Because `NaN != x` is True, every first row and every row with a missing squawk was
counted as a change (synthetic test: 100% of missing-squawk rows and the first rows of
aircraft with a code were True; overall True rate 0.207).

**New (v3).** True only when both the previous squawk (same icao24, `["icao24","time"]`
order) and the current squawk are present and differ; False otherwise
(`squawk_changed_v3`). In v3 it is also refreshed after injection for affected aircraft
on the same sorted subset as Tier A, so it describes the stream the detector actually
sees (for example a flooding duplicate is compared with its original, not given a copy
of the original's value; a fabricated ghost address has no predecessor and is False).
Synthetic True rate: 0.0015; an independent recompute from the output gives 0 mismatches.

**Why.** The v2 value mixed three things: real code changes, missing codes and track
starts.

**Expected effect.** Near-zero importance. No attack modifies squawk, so it should carry
no attack signal; it is excluded from the core set.

## 4. is_physically_implausible and the 400 m/s handling

**Where the leak was.** `_finalize_and_write` (the last step before writing, after all
attacks and recomputes):

```python
piece["is_physically_implausible"] = piece["implied_speed_mps"] > MAX_PLAUSIBLE_SPEED_MPS
noisy_benign_mask = (piece["label"] == 0) & piece["is_physically_implausible"]
piece = piece.loc[~noisy_benign_mask]
```

Benign rows above 400 m/s were dropped; attack rows at the same speed were kept. So in
the written data the flag is True for benign rows exactly 0% of the time, and any value of
implied speed above 400 m/s is an attack. It also removed benign rows whose kinematics were
disturbed by an injected neighbour: the benign row right after a gps_spoof or replay row
sees a huge implied speed and was silently dropped. In the synthetic test v2 dropped
1,774 benign rows, of which only 12 were genuine sensor glitches; the rest were benign
neighbours of injected rows.

**New (v3).** Label-agnostic, with no row filtering at all: nothing is dropped by implied
speed, before or after injection, and the flag is computed by the same rule on every row
after the final recompute. Real glitches stay in the data as benign rows.

**Why this option and not "drop glitches on clean data before injection".**

1. Dropping clean rows before injection changes the number of rows the permutation runs
   over, which changes every attack selection. v3 would then no longer be comparable with
   v2 row by row, and the requirement that all non-flooding attack rows are identical
   between versions would fail.
2. A pre-injection drop is not one step: removing a glitch row changes the next row's
   implied speed, which can create or remove other >400 m/s rows, so it needs an
   iterative recompute.
3. Glitches are part of what a deployed detector sees. Keeping them labelled benign is the
   honest setting; it also keeps the benign neighbours of injected rows, which is label
   noise created by row-level injection (an attacker's spoofed position does make the next
   real message look implausible), not leakage.

**Expected effect.** Benign rows with the flag set: 0 in v2, 1,807 (3.4% of benign) in the
synthetic v3 output. Benign false positives and the benign class size rise; single-feature
AUCs of the speed-based features fall slightly for gps_spoof and velocity_spoof (synthetic
`speed_consistency_delta`: 0.992 to 0.971 for gps_spoof, 0.993 to 0.961 for velocity_spoof).
Measured precision for the kinematic attacks should fall; this is the correction, not a
regression. The flag itself is now a plain threshold of `implied_speed_mps` and is excluded
from the core set.

## 5. is_climbing_or_descending (documented, not changed)

`|vertrate| > 2.5 m/s`, computed after injection. velocity_spoof sets
`vertrate = fillna(vertrate, 0) + U(5, 40)`, so the boost floor of 5 m/s exceeds the
threshold: almost every velocity_spoof row is flagged unless the original vertrate was
strongly negative. The flag is construction-revealing (synthetic AUC 0.89 for
velocity_spoof vs benign; on real data with fewer climbing aircraft it will be higher).
Computation left unchanged in both versions; it is excluded from the core feature set at
training time. A comment documents this in `_finalize_and_write`.

## 6. Audit of the remaining features

Criteria: (a) uses attack labels or generation flags, (b) uses future observations,
(c) uses statistics over the test day or the whole chunk. None of the feature functions
read `label` or `attack_type`; the only label-dependent step in the pipeline was the v2
row filter (change 4).

| Feature | Labels | Future obs. | Chunk / test-day statistic | Finding | Action |
|---|---|---|---|---|---|
| implied_speed_mps | no | no (previous row only) | no | Missing on every first row: a single-row ghost track is always missing (synthetic missing share 1.000 vs 0.002 benign) | documented |
| speed_consistency_delta | no | no | no | Same ghost missingness; intended velocity_spoof signal | documented |
| vrate_consistency_delta | no | no | no | velocity_spoof `fillna(0)` makes its vertrate never missing (missing share 0.051 vs 0.071 benign) and always offset by >= 5 m/s | documented (generator issue) |
| inter_arrival_s | no | no | no | Benign OpenSky state vectors are on a regular 10 s grid; flooding gaps are 0.01 to 0.5 s and replay shifts are integers that break the grid, so flooding is near-trivially separable (synthetic AUC 0.998) | documented (generator issue) |
| turn_rate_deg_s | no | no | no | No attack edits heading; ghost missingness only | none |
| region_avg_inter_arrival | no | yes, within chunk | yes | Mean clean inter-arrival per 10 degree cell over the whole chunk (1 to 2 h), including later rows and, for the holdout day, the holdout day itself. Label-free and computed before injection. gps_spoof rows moved into a cell without a clean baseline get the fallback median | documented, kept |
| is_low_coverage_region | no | yes, within chunk | yes | Threshold at the chunk's Q75 of the same baseline; redundant with the above | documented |
| icao24_valid_format | no | no | no | Ghost addresses are drawn from 0xf00000 to 0xffffff and formatted as 6 lowercase hex digits, so they are always valid; real addresses are too. Constant, carries nothing | documented |
| is_stale | no | no | no | Computed before injection; replay shifts time and lastcontact together (invariant); flooding duplicates copy the original's value although their time is up to 5 s later (negligible against the 15 s threshold) | documented |

Notes on `region_avg_inter_arrival`. Making it causal (expanding mean in time order, or a
baseline from the previous chunk) would change its distribution for every row and break
comparability with published numbers; it uses no labels and no attack rows, varies
slowly, and a deployed system can reproduce it from a trailing window of unlabeled traffic.
It stays in the core set with this caveat; a trailing-window baseline is the recommended
follow-up. The synthetic audit shows no separability (max AUC 0.515).

Other mechanisms worth stating in the paper:

* Row order in the output: flooding duplicates are appended at the end of each chunk. Not a
  feature, but any order-dependent sampling downstream would see it.
* Ties: replay can create exact `(icao24, time)` ties (172 rows in the synthetic test).
  `sort_values` is not stable for ties, so the "previous row" between two tied rows is
  arbitrary but deterministic for a given input; v3 uses the same order for Tier A, the
  running count and squawk_changed.
* Row-level injection perturbs benign neighbours: the benign row after a gps_spoof or
  replay row inherits a large implied speed. v2 hid most of these by the benign filter;
  v3 keeps them as benign.

## 7. gps_spoof region re-lookup (region_avg_inter_arrival), documented, not changed

After injection, `process_chunk` recomputes `region_cell` for every gps_spoof row from the
SPOOFED position and looks the clean baseline up again for that cell; a cell with no clean
baseline (absent from the clean chunk, or present only with NaN inter-arrival) gets the
baseline median (`fallback = region_baseline.median()`). `is_low_coverage_region` is
recomputed from that value. Consequences:

* a spoofed row that crosses a 10 degree cell boundary moves to a different FL client
  (clients are region cells) and carries that cell's regional value;
* a spoofed row landing in an unmapped cell carries the fallback median, a value benign
  rows almost never carry.

Measured with `region_lookup_check.py` (instrumented run, pipeline unchanged), pipeline v3:

| Synthetic input | --chunk-files | gps rows | changed cell (client) | fallback median | benign rows on fallback |
|---|---|---|---|---|---|
| dense (lat 32 to 62, lon -15 to 35) | 2 | 1,165 | 88 (7.55%) | 0 (0.00%) | 0 of 53,614 |
| dense | 1 | 1,165 | 93 (7.98%) | 0 (0.00%) | 0 of 53,619 |
| sparse (lat -55 to 65, lon -170 to 170) | 2 | 1,165 | 82 (7.04%) | 9 (0.77%) | 0 of 53,614 |
| sparse | 1 | 1,165 | 72 (6.18%) | 14 (1.20%) | 2 of 53,619 (0.004%) |

The cell-change share is essentially data-independent: with tier limit D and positions
uniform within a 10 degree cell, P(change) = 1 - (1 - D/20)^2, i.e. 0.5%, 4.9% and 19.0% for
D = 0.05, 0.5, 2.0 degrees, weighted 0.34/0.33/0.33 to 8.1%, matching the measured 6 to 8%.
The fallback share depends on how sparse real coverage is around the spoofed position
(oceans, coverage edges) and must be measured on the real days. On the synthetic data the
feature's single-feature AUC stays near 0.5, so the effect is small there.

`region_avg_inter_arrival` also uses test-day data: the baseline is the mean over all clean
rows of the chunk, so on the holdout day it is computed from holdout-day traffic, including
rows later than the one being scored (label-free, but transductive). This is stated in the
"Test-day data?" column of the audit table.

## 8. Further construction facts (confirmed in code, quantified on synthetic data)

* **RNG re-seeded per chunk.** `process_chunk` calls `rng = np.random.default_rng(seed)`
  (`opensky_fl_ids_pipeline.py`, `process_chunk`, right after the label columns are
  initialised) for EVERY chunk, with the same seed (42 by default). Each chunk therefore
  replays the same random stream: the permutation differs only through the chunk's row
  count, and the severity tiers, multipliers, replay shifts, ghost addresses, duplicate
  counts and offsets are drawn from the same sequence in every chunk. Chunks of equal size
  get identical row positions selected. Results are reproducible but attack parameters are
  correlated across chunks and days. Not changed (changing it would break v2/v3
  comparability).
* **Ghost address range.** `fake_ids = [f"{rng.integers(0xf00000, 0xffffff):06x}" ...]`:
  `Generator.integers` excludes the upper bound, so 0xffffff is never drawn (range
  0xf00000 to 0xfffffe). There is no check against real addresses in the chunk or against
  other ghost draws. On the synthetic data 0 of 1,165 ghost addresses collided with a real
  address (the synthetic real addresses lie outside 0xf00000 to 0xffffff, so this does not
  test the real-data case; real collisions need checking on real days, and ICAO block
  0xf00000 upward is rarely allocated to aircraft).
* **Chunk boundaries.** All per-aircraft features restart at each chunk: the first message
  of every aircraft in a chunk has no predecessor, so its Tier A kinematics are missing and
  its v3 `track_message_count` is 1, exactly like a ghost row. Measured share of benign rows
  that are first-in-chunk (v3, equal to the share with missing implied speed): 0.345%
  (185 of 53,619) with `--chunk-files 1`, 0.229% (123 of 53,614) with `--chunk-files 2`.
  On real data this share is roughly (distinct aircraft per hour) / (rows per hour) and
  should be measured with `region_lookup_check.py` (metric `benign_first_in_chunk_share`).
  It is what keeps ghost missingness from being perfectly separable.
* **Session-break rule within one-hour chunks.** `SESSION_BREAK_SECONDS = 3600`. With
  `--chunk-files 1` a chunk spans at most 3,599 s of clean data (plus at most a few
  seconds of boundary overlap rows), so the rule cannot trigger on clean data: 0 breaks
  measured (maximum clean gap within a chunk 60 s on the synthetic data). After injection
  it can only trigger through a replayed row shifted back by up to 899 s next to a row more
  than 3,600 s away: 0 breaks with `--chunk-files 1`, 1 break with `--chunk-files 2` on the
  synthetic data. In practice the "missing across session breaks" branch of the Tier A
  features is inactive for one-hour chunks; missingness comes from first-in-chunk rows and
  ghosts.

## Per-feature decisions (consistent with the core set)

CORE (keep): implied_speed_mps, speed_consistency_delta, vrate_consistency_delta,
inter_arrival_s, turn_rate_deg_s, region_avg_inter_arrival, is_stale.

EXCLUDED from core: track_message_count (ghost count 1, chunk position), squawk_changed
(no attack touches squawk; v2 form was a first-row or missing indicator),
icao24_valid_format (constant), is_low_coverage_region (redundant threshold of a core
feature), is_physically_implausible (v2 label filter; in v3 a redundant threshold),
is_climbing_or_descending (velocity_spoof boost floor above threshold).

No core feature uses labels or generation flags. region_avg_inter_arrival is the only
core feature that uses future (within-chunk) observations. However, the audit flags five
core features (all of Tier A) because of attack construction, not feature construction:

* ghost_aircraft: a fabricated address is a one-row track, so every Tier A feature is
  missing on it (missing-indicator AUC 0.999 on synthetic data, 0.98 blank share on the
  real data per the paper). Any model on the core set can still detect ghosts from
  missingness alone; ghost recall should be reported as construction-driven unless the
  generator is changed to emit multi-message ghost tracks.
* flooding: duplicates have sub-second gaps and near-zero implied speed that never occur
  on the 10 s state-vector grid (synthetic AUC 0.998 for `inter_arrival_s`, 0.997 for
  `implied_speed_mps`). This is the intended physical signal, but its strength comes from
  the sampling grid of the benign data.

## leakage_audit.py and region_lookup_check.py

```
python3 region_lookup_check.py states_2019-02-04-*.csv --chunk-files 1 --out region_stats.csv
python3 leakage_audit.py out.csv --out audit_out --data-label "2019-02-04 holdout, pipeline v3" \
    --region-stats region_stats.csv [--sample-rows 2000000 --seed 42]
python3 leakage_audit.py --placeholder paper/generated/tab_audit.tex
```

`leakage_audit.py` streams the CSVs, keeps a seeded uniform sample, and writes
`feature_audit.csv`, `feature_audit.md`, `feature_audit_table.tex`, `separability.csv` and
`region_fallback.csv`. Per feature: missing share and median by class; per attack class vs
benign, the single-feature ROC AUC as max(AUC, 1 - AUC) with missing filled with -1 (as
`prep_features` does) and the AUC of the missing indicator alone; binned mutual information
with the 6-class label. Flag: any single-feature AUC >= 0.95 or missing indicator AUC >= 0.9.

`feature_audit_table.tex` is ONE combined `table*` (caption "Features and Leakage Audit",
label `tab:audit`) that replaces the paper's separate feature-definition table. Columns:
No. | Feature | Definition (v3) | Causal? | Test-day data? | Construction artifact |
Max single-feature AUC (attack) | Max missing-indicator AUC | Core (checkmark) | Decision.
Flagged values are bold. A note under the table names the audited data (`--data-label`,
default built from the input file names, plus sample size and seed) and the thresholds.
The region_avg_inter_arrival artifact cell carries the exact cell-change and fallback
shares when `--region-stats` is given, otherwise an output-only lower bound. Rows follow
the paper's original feature numbering. `--placeholder` writes the same table with the
audit numbers as "--" (used for `paper/generated/tab_audit.tex`). Note that
`experiments/aggregate_results.py` replaces the caption with its own AUDIT_CAPTION when it
copies the table in.

`region_lookup_check.py` runs the pipeline chunk by chunk with light instrumentation
(output discarded) and writes the exact numbers that cannot be recovered from the labeled
output (section 7 and 8 metrics).

Synthetic results (300 aircraft, 2 h, 58,274 clean rows; 50,000-row audit sample):

| Flagged feature | v2 | v3 |
|---|---|---|
| implied_speed_mps (core) | ghost AUC 0.999, ghost missing AUC 0.999 | same, plus flooding AUC 0.997 |
| speed_consistency_delta (core) | gps 0.992, velocity 0.993, ghost 0.999 / missing 0.999 | gps 0.971, velocity 0.961, ghost 0.999 / missing 0.999, flooding 0.966 |
| vrate_consistency_delta (core) | ghost 0.965 / missing 0.965 | same |
| inter_arrival_s (core) | ghost 0.999 / missing 0.999 | same, plus flooding 0.998 |
| turn_rate_deg_s (core) | ghost 0.999 / missing 0.999 | ghost 0.993 / missing 0.999 |
| track_message_count | ghost 1.000 | ghost 0.999 |

Not flagged on synthetic data but excluded for the reasons above:
is_physically_implausible (max AUC 0.90 v2, 0.89 v3), is_climbing_or_descending (0.89 for
velocity_spoof). These thresholds depend on the benign mix; on real data with more
cruising aircraft the climb flag will separate velocity_spoof more strongly.
