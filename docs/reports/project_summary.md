# Project Summary: FL-IDS for Aviation (OpenSky) + Cross-Domain UAV Extension

**Student:** Tony (Bhuvan Krishna), B.Tech CSE (AI&ML), KMIT Hyderabad, III-I semester.
**Major project mentor:** Dr. M. Srinivas. Hardware: ROG Strix G16, i7-13650HX, RTX 4060 (8GB VRAM), 16GB RAM, WSL2 on Windows.

## Core Project
Federated Learning Intrusion Detection System (FL-IDS) for aviation, using OpenSky ADS-B data.

## Data Pipeline (`opensky_fl_ids_pipeline.py`)
Loads raw OpenSky hourly state-vector CSVs → cleans → engineers features → injects synthetic attacks → outputs labeled dataset.

**6 attack types simulated:** gps_spoof, velocity_spoof, replay, ghost_aircraft, flooding, signal_loss (defined but not active — removes rows rather than labeling them, needs separate evaluation).

**Features, by tier:**
- **Tier A (kinematic/physics):** `implied_speed_mps`, `speed_consistency_delta`, `vrate_consistency_delta`, `inter_arrival_s`, `turn_rate_deg_s`
- **Tier B (sensor/context):** `region_cell` (lat/lon grid, used for FL client partitioning), `region_avg_inter_arrival`, `is_low_coverage_region`, `squawk_changed`
- **Tier C (identity/metadata):** `special_squawk`, `icao24_valid_format`, `callsign_valid_format`, `track_message_count`
- **Derived:** `is_stale`, `is_physically_implausible`, `is_climbing_or_descending`

**Key bugs found & fixed along the way:**
1. `-0:` slicing bug mislabeling all rows as attacks on small samples
2. Tier A features computed *before* injection instead of after (attacks were invisible to the model) — the big one
3. Near-zero-`dt` division producing absurd speeds
4. Missing ghost-aircraft signal → added `track_message_count`
5. Benign sensor-noise contaminating training data → added outlier dropping (>400 m/s implied speed)
6. Stale `region_cell` after GPS spoofing moved the aircraft
7. Multi-day session gaps treated as continuous → added 1-hour `SESSION_BREAK_SECONDS` guard
8. **Major scale rewrite:** attack injection used to copy the ENTIRE dataset per attack type (~6x memory). Rewrote to inject into disjoint 2% chunks of ONE shared dataframe, recomputing features once — cut peak memory from ~6x to ~1.1x
9. **In progress when this chat ended:** load_and_clean was being optimized for memory (per-file float32 downcast) — caught and fixed a serious bug where timestamps (`time`, `lastposupdate`, `lastcontact`) would have been downcast to float32, which can't represent Unix timestamps precisely and would have corrupted every diff-based feature. Fixed to keep timestamps at float64, only downcast lat/lon/velocity/heading/vertrate/altitudes.

**Added features to fix real detection weaknesses:** `region_avg_inter_arrival` / `is_low_coverage_region` / `is_climbing_or_descending` — added because `replay` and `velocity_spoof` were being confused with legitimate oceanic coverage gaps and climb/descent phases respectively.

## Models Built
- **`sanity_check_model.py`** — Random Forest, multi-class (predicts `attack_type`), group-aware train/test split (prevents flooding's duplicated rows leaking across split), includes misclassified-row inspection tool. Fixed several pandas 3.0/sklearn compatibility bugs along the way.
- **`federated_simulation.py`** — real FedAvg implementation, PyTorch MLP, clients partitioned by `region_cell` (min 3,000 rows/client), reports communication cost (bytes/round) tying into the bandwidth-overhead discussion. Fixed a bug where unweighted loss caused 0% recall on rare attack classes.

## Data Access Journey
- OpenSky Trino (historical bulk access) requires institutional approval — still pending
- Currently using the free public "Weekly 24 Hours of State Vector Data" sample (one Monday per week, no approval needed)
- Downloaded 5 non-consecutive Mondays so far; ran into WSL memory limits (WSL defaults to ~half system RAM — fix: `.wslconfig` with `memory=12GB` + `wsl --shutdown`)
- License: non-commercial research is fine; no redistribution; must cite OpenSky Network

## Cross-Domain Extension (separate, ambitious thesis scope)
**Gap statement:** federated learning spanning UAV + commercial aircraft threat intelligence, using GAF (Gramian Angular Field) harmonization + open-set zero-day detection.

**Reality check done:** FL + zero-day for UAVs alone is already a crowded research area (found a Jan 2026 paper doing almost exactly that). The more genuinely novel angle is unifying two *structurally different* data modalities (UAV network/telemetry data vs. ADS-B state vectors) in one federated system — since drones mostly don't broadcast ADS-B (FAA Part 108 restrictions), so "UAV+aircraft" can't just mean "more OpenSky data."

**Notebook: `gaf-p_fixed.ipynb`** (Kaggle-based, uses UAV-NIDD dataset [885k rows, 13 classes] + originally Mendeley ADS-B dataset [22k rows, 4 classes]).

**Fixes made to this notebook:**
1. Caught that `ZERODAY_CLASS=2` was a UAV class (label space: UAV=0-12, ADS-B offset starts at 13) — meant the "cross-domain zero-day" test was actually only testing within-UAV generalization. Fixed to `ZERODAY_CLASS = N_UAV_CLASSES` (dynamically picks first ADS-B class).
2. Went through several iterations on class imbalance handling (UAV 885k vs ADS-B 22k, ~40:1):
   - First: full inverse-frequency loss weighting (capped at 50x after discovering a 6-sample UAV class would've gotten weight ~8656)
   - Then: switched to `WeightedRandomSampler` (corrects *frequency* of exposure per epoch, not just loss magnitude) — discovered this only achieves ~76/24 domain split (not literal 50/50) since it balances per-CLASS and UAV has more classes (13 vs 4)
   - **Latest direction (mid-implementation when chat ended):** user decided to bring in real OpenSky-derived ADS-B data instead of relying on tiny Mendeley set, to naturally match UAV's row count — and explicitly wants to PRESERVE natural within-domain class imbalance (mostly benign, rare attacks) since that's realistic for deployment. User chose **"mild reweighting"** (sqrt-inverse-frequency rather than full inverse-frequency) as the compromise — enough to stop the model collapsing to "always predict benign," without erasing realistic rarity signal.

**⚠️ UNFINISHED WORK — pick up here in new chat:**
- Was mid-edit reverting the `WeightedRandomSampler` back to plain `shuffle=True` and needed to re-add loss weighting using a **sqrt-based mild formula** (`weight = 1/sqrt(count)`, normalized to mean~1, capped ~20x) instead of the old full inverse-frequency formula, in BOTH the main training cell and the zero-day training cell
- Still need to build the adapter that takes `opensky_fl_ids_pipeline.py`'s labeled CSV output and reshapes it into this notebook's ADS-B domain input (replacing Mendeley) — ADS-B will go from 4 classes to 6 (benign + 5 attack types), which changes class-count offsets throughout
- Also flagged as an unresolved design tension: GAF is being applied to a feature *vector* (17 engineered features per row) rather than a genuine time series — feature-index order is being used as a pseudo-temporal axis, which is a legitimate but unconventional design choice worth being explicit about in the report

## Presentation Materials
Built two slide talk-throughs already (Data Pipeline slide, Simulated Attack Classes slide) with full narrative context for presenting — available if more slides need similar treatment.

## Available Output Files (in `/mnt/user-data/outputs/` from this conversation)
- `opensky_fl_ids_pipeline.py` (latest: memory-efficient injection, float32-safe loading — mid-fix)
- `sanity_check_model.py`
- `federated_simulation.py`
- `fl_ids_reference.md` (attack classes + features reference doc for teammates)
- `gaf-p_fixed.ipynb` (mid-edit on the mild-reweighting change)
