# FL-IDS: Federated Learning Intrusion Detection for ADS-B Traffic

A federated learning system that detects synthetic attacks (GPS spoofing,
velocity spoofing, replay, ghost aircraft, message flooding) injected into
real OpenSky Network ADS-B state-vector data, simulating a network of
independent ground receiver stations training a shared detection model
without sharing raw traffic.

> **Scope note:** an earlier cross-domain UAV+ADS-B extension (GAF image
> harmonization, comparing a UAV intrusion dataset against ADS-B) was
> explored and then deliberately deprioritized. That work is preserved
> under `docs/cross-domain/` for reference but is not part of the active
> project.

---

## Project structure

```
fl-ids-project/
├── opensky_fl_ids_pipeline.py       # Raw OpenSky CSV -> labeled, feature-engineered dataset
├── federated_simulation.py          # FedAvg simulation: MLP / CNN / GRU over region_cell clients
├── validate_labeled_dataset.py      # Verifies a labeled dataset is legitimate (not just correctly counted)
├── raw_data/                        # Raw hourly OpenSky CSVs (gitignored -- not tracked)
├── docs/
│   ├── reports/                     # Generated analysis reports (see below)
│   └── cross-domain/                # Deprioritized UAV+ADS-B GAF work (reference only)
└── README.md
```

## Quick start

**1. Process a day of raw OpenSky data into a labeled dataset:**

```bash
python3 opensky_fl_ids_pipeline.py raw_data/states_2019-01-07-*.csv \
  --chunk-files 1 \
  --output labeled_output.csv
```

**2. Validate it before trusting it:**

```bash
python3 validate_labeled_dataset.py --path labeled_output.csv --sample-size 200000
```

**3. Run the federated simulation:**

```bash
python3 federated_simulation.py labeled_output.csv \
  --model cnn --n-rounds 50 --max-total-rows 10000000
```

## `federated_simulation.py` CLI reference

| Flag | Default | Purpose |
|---|---|---|
| `--model` | `gru` | `mlp`, `cnn`, or `gru` |
| `--max-total-rows` | 2,000,000 | Global row budget across all regions/classes -- **the single highest-leverage tuning knob found this session** (see `docs/reports/`) |
| `--n-rounds` | (module default) | FedAvg communication rounds |
| `--seq-window-size` | (module default) | GRU only: messages of history per training sequence |
| `--seq-sort-col` | `time` | GRU only: `time` or `lastposupdate` -- affects `replay` detection significantly |
| `--exclude-classes` | auto (`ghost_aircraft` for GRU) | Drop a class entirely -- GRU structurally cannot detect `ghost_aircraft` (identity-scarcity attack, needs 3+ messages under one ID, but a ghost aircraft appears ~once by design) |
| `--mlp-hidden-sizes` | `64,32` | MLP only: comma-separated hidden layer sizes |
| `--cnn-channels` | `16,32` | CNN only: comma-separated conv layer channel sizes |
| `--plot-output` | `training_curves.png` | Where to save the train/test accuracy + communication cost plot |

Best-checkpoint restoration is automatic: the model is always evaluated
and reported at its best test-accuracy round, not necessarily the final
one -- a real overfitting pattern was found and corrected for during this
project (see `docs/reports/Row_Count_Scaling_Report.pdf`).

## Key findings so far

- **Row budget (`--max-total-rows`) is the single biggest lever tested** --
  raising it from the 2M default to 10M improved every architecture by
  8-13 points of accuracy, larger than any architecture change tried.
- **`replay` is the hardest attack class** across every configuration --
  it requires either a large row budget (less aggressive per-bucket
  sampling) or sequence context (GRU) to detect reliably; both help, and
  the `--seq-sort-col lastposupdate` fix specifically targets a real
  mechanism where the attack's own timestamp manipulation could hide it
  from a naively-sorted sequence.
- **GRU cannot detect `ghost_aircraft`, structurally, at any window size**
  -- excluded by default rather than silently reported as 0%.
- **CNN currently outperforms GRU** on this dataset once the row-budget
  fix is applied to both -- worth keeping in mind before assuming a more
  complex architecture is automatically better.

Full details, numbers, and charts: see `docs/reports/`.

## Data source

Real ADS-B state-vector data from the [OpenSky Network](https://opensky-network.org/).
Raw files are NOT included in this repository (see `.gitignore`) --
`raw_data/` should be populated locally before running the pipeline.
