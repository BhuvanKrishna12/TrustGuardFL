# Devlog — OpenSky FL-IDS

## `opensky_fl_ids_pipeline.py`
Cleans raw OpenSky ADS-B state vectors, engineers three tiers of kinematic/contextual
features, injects 6 synthetic attack types (gps_spoof, velocity_spoof, replay,
ghost_aircraft, flooding, signal_loss), and writes out a labeled dataset for the
FL-IDS models.

---

## 2026-08-11 — Performance pass

**Problem:** pipeline was taking a very long time on the real 5-day combined dataset
(50M+ rows). Bottleneck was in feature engineering and attack injection, not the CSV
load.

**Root cause:** `groupby("icao24")` was being rebuilt from scratch repeatedly, and
attack injection was re-sorting and re-computing features on the *entire* dataframe
for every one of the 5 attack types, even though each attack only touches ~2% of rows.

**Changes made (all verified numerically identical to the original on synthetic data
before being kept):**

1. **`add_tier_a_features`** — replaced `groupby("icao24").shift()` with a plain
   whole-column `.shift(1)` plus a boundary mask. Works because the df is already
   sorted by `["icao24","time"]` whenever this runs; only the first row of each
   aircraft's block needs special handling.
2. **`add_tier_b_features` (`squawk_changed`)** — same boundary-mask fix instead of
   a groupby.
3. **`add_tier_c_features` (`track_message_count`)** — replaced
   `groupby().transform("count")` with `value_counts().map()`.
4. **`inject_attack` (ghost_aircraft)** — capture the aircraft's real icao24 *before*
   it gets overwritten with a fake one, so the group it just lost a row from can be
   found again later.
5. **`inject_attack` (the big one)** — instead of resorting and recomputing Tier A
   features across the whole dataframe per attack type, isolate only the icao24
   groups actually touched by that attack, recompute just that subset, and write it
   back by index. This is what eliminated 5 full `O(n log n)` sorts of a 50M-row df
   per chunk.
6. **`inject_attack` (`track_message_count` for ghost_aircraft/flooding)** — scoped
   the same way as #3, restricted to affected rows only.
7. **`inject_attack` (`gps_spoof` region recompute)** — was rerunning the entire
   Tier B pass on all 50M+ rows for a lat/lon change on 2% of them; now recomputes
   `region_cell` / `region_avg_inter_arrival` / `is_low_coverage_region` for just the
   spoofed rows via direct map lookups.

**Not changed:** chunking strategy, dtypes, CSV output format, feature definitions,
attack logic itself.

**Result:** on a 1M-row / 30k-aircraft synthetic benchmark, Tier B+C ran ~31% faster
and attack injection ran ~2x faster, with the gap expected to widen further at the
real dataset's scale since the eliminated full-dataframe sorts scale worse than the
scoped subset sorts do.

**Next lever, if still not fast enough:** categorical dtype for `icao24` /
`region_cell` — would cut memory and could allow bigger chunks, but needs care around
ghost_aircraft's fake icao24 values (new categories not seen at load time). Deferred
for now to keep today's change verifiably safe.
