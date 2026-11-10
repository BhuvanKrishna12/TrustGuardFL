# Revision experiments

Code for the journal revision: FedAvg CNN vs centralized CNN vs LightGBM, on the full
and core feature sets, over several seeds, with class-prior correction evaluated on
held-out December days. The step-by-step laptop guide is `../RUN_ALL.md`.

| File | Purpose |
|---|---|
| `run_experiments.py` | Trains and evaluates every (model, feature set, seed) run; one JSON per run |
| `sampling.py` | Training-file stats pass and the uniform per-(region, class) training sampler |
| `aggregate_results.py` | Turns run JSONs into `paper/generated/*.tex` (macros, tables, figures) |
| `make_holdout_sample.py` | 1% test sample of the December files (seed 42, per 2M-row chunk) |
| `check_equivalence.py` | Shows the FedAvg driver reproduces `federated_simulation.py` round by round |
| `test_sampler.py` | Shows the uniform sampler is unbiased across files and the legacy one is not |
| `make_synthetic_data.py`, `smoke_test.sh` | Synthetic data and a 5-minute end-to-end test |

`federated_simulation.py` is not modified. The harness imports its `prep_features`,
`build_model`, `local_train`, `federated_average`, `model_size_bytes` and (with
`--legacy-sampler`) `load_data`.

## What one run does

1. **Training-file stats** (once, cached in `rev_runs/cache/`). One chunked pass over
   `region_cell`, `attack_type` and `icao24` of every training file. It gives the raw
   row count (`trainrawrows`), the natural class prior pi, the region list and the set of
   training aircraft.
2. **Training sample** (once, cached). By default `sampling.uniform_sample` keeps, for
   every (region_cell, attack_type) bucket, the `cap` rows with the smallest seeded random
   key. Every row of a bucket is equally likely to be kept, whichever file or chunk it
   comes from. The cap formula (`max(500, budget / (regions x 6))`), the columns and the
   dtypes are the same as `federated_simulation.load_data`.
   - `--legacy-sampler` uses `load_data` itself. It re-draws a capped bucket after every
     chunk, so early chunks and files decay geometrically and the last file dominates.
     It also stops reading once the budget is reached. `test_sampler.py` demonstrates
     both behaviours.
   - The sample is the same for every seed. Its composition per source file and per
     training day (overall and per class) is stored in the cache metadata and in every
     run JSON.
3. **Holdout** (once, cached). Each row's day is the UTC date of `lastposupdate`, or of
   `time` when that is missing, because replay shifts `time`. A row is flagged
   "unseen aircraft" when its icao24 never occurs in the training files. Two rule
   baselines are counted on the raw holdout:
   - ghost: all four rate features missing
   - flooding: `inter_arrival_s < 1 s`
4. **Clients** (per seed). Each region_cell with at least 3,000 rows becomes a client.
   Its rows get an 80/20 train/validation split, seeded by the run seed; with seed 42
   this is exactly `build_clients`. All three models use the same training rows and
   select on the same validation rows.
5. **Models.**
   - `fed_cnn`: the FedAvg loop of `federated_simulation.main`. Channels 32 and 64,
     50 rounds, 2 local epochs, batch 512, Adam with lr 1e-3, full participation.
     Selection: best round by unweighted validation accuracy (s=0).
   - `central_cnn`: same architecture, scaler, class weights, optimizer and batch.
     At most 20 epochs, early stopping (patience 3) on validation accuracy.
   - `lgbm`: the `lgbm_baseline.py` parameters with balanced weights. Early stopping on
     class-weighted multi_logloss of the validation rows.
6. **Evaluation.** The holdout is scored once and the probabilities are saved
   (`probs/*.npz`, float16). For each strength s, the prediction is
   `argmax(p * pi**s)`. Metrics are computed on:
   - the pooled holdout
   - each test day
   - the unseen-aircraft subset
   - deployment prevalence rho in {0.1%, 1%}: test rows are reweighted (not resampled)
     so attacks make up rho of traffic, the five attack classes keeping their test
     proportions. Two corrections are evaluated: with the training pi (mismatched, what a
     deployment would do) and with the true deployment prior (oracle). Recall and FAR do
     not change under reweighting; precision, F1, accuracy and the real-alert share do.
     The oracle maximises expected accuracy, not macro F1, so its macro F1 can be lower.

The test days are never used to choose anything. Versions of Python and the main
packages are stored in every run JSON.

## Commands (real run on the laptop)

See `../RUN_ALL.md` for the full sequence. The core commands:

```bash
cd /mnt/d/fl-ids-sweep/rev && source ~/flids-venv/bin/activate
D=/mnt/d/fl-ids-sweep

# test sample from the v3 December files (once)
python3 experiments/make_holdout_sample.py $D/labeled_v3/opensky_2019-12-{09,16,23,30}_labeled.csv \
    --out $D/holdout_sample_v3.csv

# all 30 runs (3 models x 2 feature sets x 5 seeds), in the background
mkdir -p $D/rev_runs
nohup python3 experiments/run_experiments.py \
  --train-files $D/labeled_v3/opensky_2019-01-07_labeled.csv \
                $D/labeled_v3/opensky_2019-04-01_labeled.csv \
                $D/labeled_v3/opensky_2019-06-24_labeled.csv \
                $D/labeled_v3/opensky_2019-09-30_labeled.csv \
  --holdout $D/holdout_sample_v3.csv \
  --out-dir $D/rev_runs --max-total-rows 10000000 --device auto \
  > $D/rev_runs/nohup.out 2>&1 &
tail -f $D/rev_runs/run_experiments.log

# comparison on the old v2 files: LightGBM 5 seeds, then fed CNN seeds 1-3 (full set)
mkdir -p $D/rev_runs_v2
V2="--train-files $D/labeled/opensky_2019-01-07_labeled.csv $D/labeled/opensky_2019-04-01_labeled.csv
    $D/labeled/opensky_2019-06-24_labeled.csv $D/labeled/opensky_2019-09-30_labeled.csv
    --holdout $D/holdout_sample.csv --out-dir $D/rev_runs_v2 --max-total-rows 10000000
    --device auto --feature-sets full"
nohup bash -c "python3 experiments/run_experiments.py $V2 --models lgbm && \
               python3 experiments/run_experiments.py $V2 --models fed_cnn --seeds 1,2,3" \
  > $D/rev_runs_v2/nohup.out 2>&1 &

# paper inputs
python3 experiments/aggregate_results.py --runs-dirs $D/rev_runs --v2-runs-dir $D/rev_runs_v2 \
    --audit-table $D/audit_v3/feature_audit_table.tex
```

Defaults: seeds 1,2,3,4,5; models fed_cnn,central_cnn,lgbm; feature sets full,core;
strengths 0,0.1,...,1.0; rhos 0.001,0.01; min client rows 3000; 50 rounds; central CNN
at most 20 epochs.

**Resume.** Rerunning a command skips every run whose `runs/<model>__<fs>__seed<k>.json`
exists; `--force` retrains.

**Before any results exist.** `aggregate_results.py --placeholder` writes every file
with `--` so the paper compiles. An existing `tab_audit.tex` is never overwritten unless
`--audit-table` is given, in which case that table is copied with its own caption and
label.

**New strengths or prevalences without retraining.** This recomputes all metrics from
the saved probabilities:

```bash
python3 experiments/run_experiments.py --out-dir $D/rev_runs --holdout-only \
    --strengths 0,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0 --rhos 0.0001,0.001,0.01
```

Add `--repredict --holdout <csv>` to re-score a changed test file with the saved models.
Macro keys exist only for multiples of 0.1 and for rho 0.1% and 1%.

## Time estimate (RTX 3050 6 GB, 11 GB RAM, 10M-row sample)

| Step | Time |
|---|---|
| Stats pass (3 columns of 4 x 12 GB) | ~30 to 40 min, once per file set |
| Uniform sample (all columns of 4 x 12 GB, no early stop) | ~40 to 50 min, once |
| Holdout read (~2.1M rows) | ~2 min, once |
| fed_cnn, per run | ~35 min (50 rounds) |
| central_cnn, per run | ~5 to 10 min (early stopping) |
| lgbm, per run | ~5 to 10 min (4 threads, early stopping) |

- **v3, all 30 runs:** about 8.5 h of training plus about 1.5 h of one-off passes, so
  roughly 10 h.
- **v2 comparison:** about 1.5 h of one-off passes, plus about 40 min for LightGBM
  (5 seeds) and about 1 h 45 min for fed CNN (3 seeds), so roughly 4 h.

These are estimates. The first finished run in the log gives the real figure
(`wall_seconds` in its JSON). Peak memory is about 4 GB of RAM during training; the
sampler's buffer adds up to about 1.5 GB during the one-off pass.

## Splitting across two machines

Every run is independent; filter with `--models`, `--seeds` and `--feature-sets`
(for example `--seeds 1,2,3` on one machine and `--seeds 4,5` on the other, or
`--models fed_cnn` on the GPU machine and `--models central_cnn,lgbm` on the other).
Both machines must use the same files, budget and test sample.

Copy `rev_runs/cache/` to the second machine's `--out-dir` to skip its one-off passes.
The cache key is file names, sizes, budget and sampler, not absolute paths.

To merge, either pass both folders to the aggregator
(`--runs-dirs rev_runs rev_runs_B`), or copy `runs/`, `probs/` and `models/` from B into
A (`cp -n`); file names never collide. Use the single merged folder for `--holdout-only`.

## Outputs

`rev_runs/runs/<model>__<fs>__seed<k>.json` contains:

- config and package versions
- data facts:
  - raw rows (total and per file) and raw aircraft
  - sampler, cap, sample rows and sample aircraft
  - sample composition per file and per day
  - class counts, class weights and the natural prior
  - clients and train/validation rows
- training trace: per-round or per-epoch validation accuracy, best round, epoch or
  iteration, gradient steps, model and communication bytes, wall time, selection rule,
  and LightGBM gain shares
- holdout facts: rows, days, support, blank-feature shares, rule-baseline counts and
  aircraft overlap
- `metrics["<s>"]`:
  - `pooled`, `days[date]` and `unseen`, each with accuracy, macro F1, FAR, missed attack
    rate, average attack recall, real-alert share, per-class P/R/F1/support and the
    confusion matrix
  - `prevalence["<rho>"]["mismatched"|"oracle"]`, with the same metrics after reweighting

`aggregate_results.py` writes to `paper/generated/`:

- `results_macros.tex`
- tables: `tab_headline`, `tab_fed_vs_central`, `tab_ablation`, `tab_perclass`,
  `tab_perday`, `tab_clients`, `tab_config`, `tab_prevalence`, `tab_unseen`, `tab_mix`,
  `tab_rules`, `tab_confusion`, `tab_importance`, `tab_v2v3` and `tab_audit`
- figures: `fig_sweep` and `fig_clients`, with their data in `sweep_<model>_<fs>.dat`
  and `clientsizes.dat`
