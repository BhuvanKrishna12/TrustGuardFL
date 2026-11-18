# Reproducing the paper

This guide runs the whole study on one machine, from the raw OpenSky archives to the
LaTeX tables and macros the paper reports. Two placeholders are used throughout:

| Placeholder | Meaning |
|---|---|
| `$RAW_DIR` | Folder holding the raw OpenSky state-vector archives (read only, may be an external drive) |
| `$WORK_DIR` | Folder for every output: labelled day files, test sample, caches, runs, tables (default of `run_revision2.sh`: `~/rev2_work`) |

All commands are run from the repository root.

## 1. Requirements

* Python 3.12 or 3.13 with the packages in `requirements.txt` (`pip install -r requirements.txt`).
  Install the PyTorch wheel that matches your GPU driver, or the CPU wheel.
* A CUDA GPU is strongly recommended. Every script runs on the CPU (`--device auto` picks the
  GPU when there is one), but a federated CNN run takes about 40 to 60 minutes on a laptop GPU
  (RTX 3050 6 GB) and several hours on a CPU.
* About 11 GB of RAM was enough for the measured runs (peak about 4 GB during training plus
  about 1.5 GB for the sampler buffer).
* Disk: about 135 GB free in `$WORK_DIR` for the multi-variant labelled files (about 119 GB)
  plus temporary hourly files, samples and checkpoints. `run_revision2.sh` checks this before
  step 2 (`REV2_NEED_GB` overrides the threshold).
* Total compute: about 39 hours on the laptop GPU above, spread over the steps listed below.
  Every step can be stopped and resumed (section 5).

## 2. Raw data

The study uses OpenSky Network state vectors (https://opensky-network.org/data/datasets),
one full Monday per day file, 24 hourly archives per day named
`states_YYYY-MM-DD-HH.csv.tar` (each holds one `.csv.gz`; plain `.csv.gz` and `.csv` hours are
accepted too). The days are fixed in code (`experiments/rev2_plan.py`, `run_revision2.sh`):

| Role | Days (2019) |
|---|---|
| Training | 2019-01-07, 2019-04-01, 2019-06-24, 2019-09-30 (one Monday per quarter) |
| Test | 2019-12-09, 2019-12-16, 2019-12-23, 2019-12-30 (four December Mondays) |

Download the 8 x 24 hourly archives into `$RAW_DIR`, then write the list of archive paths,
one per line, to a text file:

```bash
find "$RAW_DIR" -name 'states_2019-*.csv.tar' | sort > "$RAW_DIR/tars.txt"
```

OpenSky data is not redistributed in this repository. Please cite OpenSky as its terms ask
(see the README).

## 3. The two driver scripts

* **`run_revision2.sh` (current path, used for the paper's final results).** One resumable
  script with 11 steps: setup and self-test, multi-variant reprocessing of the 8 days, the
  December test sample, the shared sampling passes, the full run grid (main comparison,
  correction ablation, earlier vs corrected pipeline, MLP baselines, FedProx, federated
  ablation) and the tables. The multi-variant labelled files contain the corrected (v3)
  pipeline output plus the columns needed to rebuild the earlier (v2) output and every
  one-correction-at-a-time mix, so all ablations come from one reprocessing pass.
* **`run_everything.sh` (first-revision driver).** Runs pipeline v3 on the 8 days, the data
  checks, the **leakage audit**, a 30-run main grid and a comparison on old v2 files. Its
  experiment grid is superseded by `run_revision2.sh`, but it remains the scripted path for
  the snapshot-interval check and the leakage audit (steps 4a and 4b). Run it with
  `WORK_DIR=... bash run_everything.sh`; its header lists the variables.

The recommended order is: `run_revision2.sh` for everything, then the leakage audit by hand
(section 6), then one more aggregation that includes the audit table (section 7).

## 4. Running `run_revision2.sh`

```bash
export REV2_W="$WORK_DIR"            # outputs (default ~/rev2_work)
export REV2_TARS="$RAW_DIR/tars.txt" # list of raw archives (or put it at $WORK_DIR/tars.txt)
bash run_revision2.sh dry-run        # print every command without running anything
bash run_revision2.sh start          # start or continue in the background
bash run_revision2.sh status         # current step, runs done, time left, free disk
bash run_revision2.sh stop           # stop cleanly after the current small unit
```

Optional variables: `REV2_VENV` (a venv to activate), `REV2_P` (a folder that also receives a
copy of the results zip), `REV2_CODE` (another code folder), `REV2_DEVICE` (`auto`, `cpu`,
`cuda`). Progress is appended to `$WORK_DIR/STATUS.txt` and logs go to `$WORK_DIR/logs/`.

| Step | What it does | Time (laptop GPU) |
|---|---|---|
| 1 | Checks packages and runs `experiments/smoke_test.sh` | about 20 min |
| 2 | Reprocesses the 8 days with `opensky_fl_ids_pipeline.py --pipeline-version mv` (`experiments/reprocess_mv.py`); the only step that reads `$RAW_DIR` | about 6 h |
| 3 | 1% hash sample of the 4 December days (`experiments/make_holdout_sample.py --method hash`) | about 45 min |
| 4 | One-off sampling passes over the training files (stats, uniform sample, legacy samples) | about 3 h |
| 5 | Main grid: LightGBM, central CNN and federated CNN, full and core feature sets, 5 seeds (30 runs) | about 7.5 h |
| 6 | Correction ablation with LightGBM and central CNN (96 runs) | about 3 h |
| 7 | Earlier (v2) vs corrected (v3) pipeline under one protocol (7 runs) | about 3.5 h |
| 8 | MLP baselines, central and federated (10 runs) | about 3 h |
| 9 | FedProx, mu = 0.01 and 0.1 (6 runs) | about 4 h |
| 10 | Federated ablation (12 runs) | about 8 h |
| 11 | Tables and macros into `$WORK_DIR/generated`, plus `$WORK_DIR/results_revision2.zip` | a few minutes |

The run grid is defined in `experiments/rev2_plan.py` (`python3 experiments/rev2_plan.py table`
prints runs and hours per step). Models are selected on training-day validation data only;
the December days are used only for the reported test metrics.

## 5. Pause, resume and cache-only mode

* **Pause and resume.** `stop` asks the worker to finish its current unit (one training round,
  one epoch or one hour of data) and exit; `start` continues from there. A closed terminal,
  sleep, a kill or a power cut lose at most the current unit: every output is written to a
  temporary name and renamed, and finished steps are marked in `$WORK_DIR/done/`. A second
  `start` while a worker is running is refused.
* **Raw archives disappearing.** If the archives become unreadable during step 2 (for example
  an external drive is unplugged), the script stops with exit code 3 and a recovery note in
  `STATUS.txt`; finished hours and days are kept. After step 2 the raw archives are no longer
  needed.
* **Cache-only ("trust cache") mode.** Steps 5 to 10 read only the caches built in step 4
  (`$WORK_DIR/cache/`, about 2 GB) and the test sample, never the 120 GB labelled files. When
  the labelled training files are absent but `cache/input_manifest.json` and
  `holdout_sample_mv.csv` are present, `run_revision2.sh` switches to this mode by itself
  (`status` shows "cache-only"), skips steps 1 to 4 after checking the cache with
  `experiments/verify_cache.py`, and passes `--trust-cache` to `run_experiments.py`, which then
  takes the input file sizes from the manifest. `experiments/test_trust_cache.py` shows the
  results are identical to a run with the labelled files present. This is what the optional
  multi-machine mode uses (`docs/MULTI_MACHINE.md`).
* **Re-scoring without retraining.** `run_experiments.py --holdout-only` recomputes all metrics
  from the saved probabilities, for example for new correction strengths or prevalences
  (see `experiments/README.md`).

## 6. Leakage audit and data checks

The audit table of the paper is produced by `leakage_audit.py`, with exact gps_spoof region
re-lookup numbers from `region_lookup_check.py`. It is not part of `run_revision2.sh`. Run it on
one training day and on the test sample (the multi-variant files hold the v3 columns
unchanged, so they can be audited directly):

```bash
# exact region re-lookup numbers from the raw hours of one training day
mkdir -p "$WORK_DIR/audit_v3" "$WORK_DIR/raw_tmp/2019-01-07"
grep "states_2019-01-07-" "$RAW_DIR/tars.txt" | while read -r t; do
  tar -xOf "$t" --wildcards '*.csv.gz' | gunzip > "$WORK_DIR/raw_tmp/2019-01-07/$(basename "$t" .tar)"
done
python3 region_lookup_check.py $(ls "$WORK_DIR"/raw_tmp/2019-01-07/states_2019-01-07-*.csv | sort) \
    --chunk-files 1 --out "$WORK_DIR/audit_v3/region_stats.csv"
rm -rf "$WORK_DIR/raw_tmp/2019-01-07"

# the audit itself
python3 leakage_audit.py "$WORK_DIR/labeled_mv/opensky_2019-01-07_labeled.csv" \
    "$WORK_DIR/holdout_sample_mv.csv" --out "$WORK_DIR/audit_v3" \
    --region-stats "$WORK_DIR/audit_v3/region_stats.csv"
```

The audit writes `feature_audit.csv`, `feature_audit.md`, `feature_audit_table.tex`,
`separability.csv` and `region_fallback.csv`. The audit method and every pipeline correction
are described in `docs/PIPELINE_V3_CHANGES.md`. `run_everything.sh` (step 4a) also prints the
benign inter-arrival quantiles that confirm the snapshot interval of the data.

## 7. Tables

Step 11 already builds the tables. To include the audit table, aggregate once more:

```bash
python3 experiments/aggregate_results.py --runs-dirs "$WORK_DIR/runs/main" \
    --rev2-root "$WORK_DIR/runs" --audit-table "$WORK_DIR/audit_v3/feature_audit_table.tex" \
    --tex-dir "$WORK_DIR/generated"
```

`--tex-dir` defaults to `paper/generated/` (created if missing; `paper/` is not part of this
repository). The output is `results_macros.tex` (one `\R{key}` macro per number), the
`tab_*.tex` tables and the `fig_*.tex` figures with their `.dat` data.
`aggregate_results.py --placeholder` writes the same files with `--` in place of numbers.

## 8. Tests

| Test | What it checks | Time (CPU) |
|---|---|---|
| `bash experiments/smoke_test.sh [WORK_DIR]` | Everything below plus quick runs of every model, variant and protocol path and the aggregator, on synthetic data; must end with `SMOKE TEST OK`. `SMOKE_SKIP_SIM=1` skips the two simulations | about 20 min |
| `python3 experiments/test_sampler.py DIR` | The uniform sampler is unbiased across files; the legacy one is not | seconds |
| `python3 experiments/test_variants.py DIR` | The multi-variant output contains the v3 and the v2 pipeline output exactly | about 1 min |
| `python3 experiments/test_sampling_mv.py DIR` | Revision-2 sampling passes match the round-1 code on v3 and v2 files | about 1 min |
| `python3 experiments/test_protocol_v1_regression.py ...` | `--protocol v1` reproduces the frozen round-1 harness (`experiments/reference_r1/`); arguments are set up by the smoke test | minutes |
| `python3 experiments/test_fedprox_resume.py ...` | FedProx with mu = 0 equals FedAvg; resume after stop and kill equals an uninterrupted run; arguments are set up by the smoke test | minutes |
| `python3 experiments/test_trust_cache.py DIR` | Cache-only mode gives identical results with the labelled files deleted | minutes |
| `python3 experiments/test_shards.py` | The committed shard files are a disjoint exact cover of the remaining runs | seconds |
| `bash experiments/test_run_revision2.sh DIR` | Simulated full pass of `run_revision2.sh` with missing archives, a vanishing drive, `stop` and `kill -9` | about 10 min |
| `bash experiments/test_multi_machine.sh DIR` | Simulated main machine plus cache-only helper, merge and aggregation | a few minutes |
| `python3 experiments/check_equivalence.py A.csv B.csv` | The harness's FedAvg loop matches `federated_simulation.py` round by round | minutes |

The smoke test needs LightGBM and PyTorch. Its LaTeX compile check is optional and is skipped
when `pdflatex` or the IEEEtran class is missing.
