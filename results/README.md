# Results

The aggregated outputs behind every table and figure of the paper, committed so that a
reader can check any reported number without re-running the grid. The full grid is about
39 hours of compute on a laptop GPU, so the numbers are kept here rather than left to be
rebuilt.

| Path | Contents |
|---|---|
| `generated/` | The LaTeX tables, figure sources, `.dat` data files and `results_macros.tex` written by `experiments/aggregate_results.py`. Build outputs, committed unchanged; never edit them by hand |
| `runs_index.csv` | One row per training run (161 rows): the identity of the run and its headline test metrics, read out of the per-run JSON files |
| `make_runs_index.py` | The script that writes `runs_index.csv`, so the index is reproducible rather than hand-made |

The per-run JSON files themselves are not in this repository; see
[Raw per-run JSON files](#raw-per-run-json-files) below.

## Which file feeds which table and figure

Table and figure numbers are those of the submitted manuscript.

| File | Feeds |
|---|---|
| `generated/tab_mix.tex` | Table 2, class mix, split sizes and aircraft counts |
| `generated/tab_config.tex` | Table 3(a), training configuration |
| `generated/tab_clients.tex` | Table 3(b), client population |
| `generated/tab_audit.tex` | Table 6, leakage audit. In this build the table is a placeholder of `--` rows: the submitted table comes from `feature_audit_table.tex` written by `leakage_audit.py`, which is passed to the aggregator with `--audit-table` |
| `generated/tab_selected_s.tex` | Table 7, correction strength selected on validation data |
| `generated/tab_unified_pipeline.tex` | Table 8, earlier versus corrected pipeline under one protocol |
| `generated/tab_baselines.tex` | Table 9, baselines on the corrected pipeline at the selected strength |
| `generated/tab_fed_vs_central.tex` | Table 10, federated versus centralised training on identical rows |
| `generated/tab_perclass.tex` | Table 11, per-class precision, recall and F1 at `s=1` |
| `generated/tab_confusion.tex` | Table 12, confusion matrices at `s=1` |
| `generated/tab_rules.tex` | Table 13, untrained single-rule baselines |
| `generated/tab_prevalence.tex` | Table 14, macro F1 and genuine-alert share under assumed prevalence |
| `generated/tab_perday.tex` | Table 15(a), results per held-out day |
| `generated/tab_unseen.tex` | Table 15(b), test rows from aircraft never seen in training |
| `generated/tab_ablation_corrections.tex` | Table 16, pipeline corrections one at a time at `s=1` |
| `generated/fig_sweep.tex` plus `generated/sweep_{fedcnn,centralcnn,lgbm}_{full,core}.dat` | Figure 2, the correction-strength sweep (the `.dat` files hold the plotted points and their standard deviations) |
| `generated/results_macros.tex` | Every `\R{key}` number quoted in the running text, including the numbers no table prints |

Four tables and one figure in the manuscript are written by hand and have no file here:
Table 1 (related work), Table 4 (attack constants), Table 5 (feature definitions) and
Figure 1 (the workflow diagram).

The aggregator also writes files that the submitted manuscript does not use:
`tab_headline.tex` (accuracy, FAR, attack recall and macro F1 at `s = 0, 0.5, 1`),
`tab_ablation.tex` (full versus core feature set), `tab_prauc.tex` (PR-AUC per class),
`tab_importance.tex` (LightGBM gain share), `tab_v2v3.tex` (an older old-versus-new table
whose v2 rows are all `--`) and `fig_clients.tex` with `clientsizes.dat` (rows per client).
They are kept because they come from the same aggregation pass.

## How to regenerate them

The tables and macros in `generated/` come from one command, run after the grid has
finished (step 11 of `run_revision2.sh`):

```bash
python3 experiments/aggregate_results.py --runs-dirs "$WORK_DIR/runs/main" \
    --rev2-root "$WORK_DIR/runs" --audit-table "$WORK_DIR/audit_v3/feature_audit_table.tex" \
    --tex-dir "$WORK_DIR/generated"
```

Leave `--audit-table` out and `tab_audit.tex` stays the placeholder it is here. The index
is rebuilt from the merged run JSONs:

```bash
python3 results/make_runs_index.py "$WORK_DIR/merged_runs" -o results/runs_index.csv
```

`make_runs_index.py` needs nothing beyond the standard library. The whole path from the raw
OpenSky archives to these files, with the expected time and disk, is in
[docs/REPRODUCING.md](../docs/REPRODUCING.md); the multi-machine variant is in
[docs/MULTI_MACHINE.md](../docs/MULTI_MACHINE.md).

## What is in `runs_index.csv`

One row per run, 161 rows, sorted by step and then by source file. Columns:

| Column | Meaning |
|---|---|
| `run_key` | `s{step}:{model}:{variant tag}:{feature set}:seed{seed}`, the stable run key of `experiments/rev2_plan.py`. The variant tag carries `-ls` for a legacy-sampler run and `-mu<x>` for FedProx |
| `step` | Step of `run_revision2.sh` that asks for the run: 5 main grid, 6 correction ablation and legacy-sampler pair, 7 earlier versus corrected pipeline, 8 MLP baselines, 9 FedProx, 10 federated ablation |
| `model` | `lgbm`, `central_cnn`, `fed_cnn`, `central_mlp`, `fed_mlp` |
| `feature_set` | `full` (13 features) or `core` (the audit-clean subset) |
| `variant` | Pipeline variant: `corrected`, `legacy`, `loo:C*`, `add:C*`, `prefloor` |
| `protocol` | Evaluation protocol, `v2` for every run here (aircraft-disjoint validation) |
| `sampler` | `uniform` (capped per region cell and class) or `legacy` (the earlier loader) |
| `fedprox_mu` | FedProx proximal weight; empty for FedAvg and for the non-federated models |
| `seed` | 1 to 5 |
| `s_star` | Correction strength chosen on the validation rows |
| `macro_f1_at_s_star`, `far_at_s_star`, `attack_recall_at_s_star` | Test metrics at `s_star`, as fractions, not percentages |
| `pr_auc` | Macro one-vs-rest average precision on the test sample; it does not depend on `s` |
| `macro_f1_at_s1`, `far_at_s1`, `attack_recall_at_s1` | The same three metrics at full correction, `s=1` |
| `n_clients`, `rounds_completed`, `rounds_cap` | Federated runs only: 161 clients, 50 of 50 FedAvg rounds in every run here |
| `device` | Device recorded by the run, `cuda` or `cpu` |
| `wall_clock_minutes` | `wall_seconds` of the run divided by 60, to one decimal place |
| `source_file` | Path of the run JSON inside the merged runs tree |

Every value is read straight out of a run JSON. Nothing is recomputed and nothing is
rounded, with the single exception of `wall_clock_minutes`. The metrics are fractions
exactly as the JSON stores them, so the paper's percentages are these numbers times 100.

A cell is left empty when the field does not apply to the run rather than filled with a
substitute value. In practice that is: `fedprox_mu` for every FedAvg and centralised run,
and `n_clients`, `rounds_completed` and `rounds_cap` for the 123 runs of the three
non-federated models, which have epochs rather than rounds. No other field is missing from
any JSON: all 161 runs carry `s_star`, `device`, `wall_seconds`, `pr_auc.macro` and
`test_summary` at both `s_star` and `s=1`.

The three legacy central-CNN seeds 1 to 3 are asked for by both step 6 and step 7 and exist
as one file each; the index lists them under step 6.

The 71 client-inventory files (`clients__seed*.json`, one per seed and run folder) describe
the client split, not a training run: they hold the region cells, their row counts and the
label divergences, and they feed Table 3(b). They are excluded from the index and live in
the raw archive beside the runs, in the same `*/runs/` folders.

## Provenance

The runs were produced on three machines: an RTX 3050 laptop, an RTX 4060 laptop and a
CPU-only machine. The main-grid central CNN runs used a GPU, while the central CNN
ablation variants (leave-one-out, add-one, legacy, legacy loader and pre-floor) ran on
CPU; the federated CNN runs all used a GPU. The `device` column of the index says which
device each run recorded. The library versions were the same on all three machines (torch
2.14.1, LightGBM 4.7.0). Results on different consumer GPUs can differ slightly per seed,
so an exact re-run on other hardware should be expected to reproduce the reported means and
intervals, not every digit of a single seed.

## Caveats a reader needs

* **Seed counts differ by row.** LightGBM variants have 5 seeds; the CNN ablation variants
  (central and federated) have 3. Inside one column of the ablation table the corrected and
  legacy rows are 5-seed means while the leave-one-out and add-one rows are 3-seed means,
  and the effects are paired over the seeds the two sides share. Quote the paired effect,
  not a difference of two printed means.
* **The federated runs may not have converged.** 8 of the 10 federated main runs have their
  best FedAvg round at or next to the 50-round cap, so the federated-minus-centralised gap
  is an upper bound on the cost of federation at this round budget, not a converged result.
* **Standard deviations are over model seeds with the data held fixed.** The seed changes
  initialisation, the validation split, shuffling and bagging; the sample itself is drawn
  once with seed 42. The paired interval over 5 seeds has 4 degrees of freedom, so it is
  wide and sensitive to one seed.
* **Only the validation split is aircraft-disjoint.** Validation holds about 20 per cent of
  the aircraft of the training sample. The test days are later in the year but are not
  aircraft-disjoint from training; Table 15(b) reports the unseen-aircraft subset
  separately.
* **The pre-floor row of the correction ablation is reconstructed.** It applies the
  documented rule (rate features missing when the gap is below 1 s) rather than a preserved
  earlier pipeline, and it is a separate extra row, not one of the four corrections. Its
  central CNN has 3 seeds and no federated CNN was run for it.
* **One run exists from two machines.** `fed_cnn__full__seed4` was run twice and the two
  copies differ in the fourth decimal of macro F1 at `s_star`. The merged tree, and so this
  index, holds one of them; the difference changes no printed digit.
* **Each variant is scored on its own test sample by design.** The corrected pipeline and
  some variants share 2,194,175 test rows, while others differ, because a correction changes
  which rows exist. The variant runs also carry the same metrics on the fully corrected test
  sample, which is the comparison to use when the question is model quality rather than the
  effect of the test sample.

## Raw per-run JSON files

The 232 raw files (161 run JSONs plus 71 client inventories) total about 75 MB and are not
in this repository. They are archived instead:

```
Zenodo DOI: [PLACEHOLDER: to be filled in once the archive is published]
```

That line is a placeholder and not yet a resolvable DOI.

Each run JSON holds, for one run:

* identity and environment: run id, model, feature set and the feature names, seed, class
  names, library and CUDA versions, device, pipeline variant with the list of corrections
  left in their legacy form, protocol, sampler and FedProx mu;
* selection: the chosen `s_star`, the selection rule, the full `s` grid with validation
  macro F1, accuracy, false alarm rate and attack recall at every point, the validation row
  and class counts and the reweighting to the natural prior;
* configuration and data: the invoking configuration, the sampler and its cap, training
  sample rows and aircraft, the per-file and per-class composition and the class weights;
* training: hyperparameters, per-epoch or per-round curves, the best epoch or round, number
  of gradient steps, communication bytes for the federated runs, and training seconds;
* the test sample as used: path, row count, per-day counts, per-class support and the row
  counts affected by the legacy filter;
* results: metrics at all 11 correction strengths, the summary at `s_star`, `s=0`, `s=0.5`
  and `s=1`, per-class and macro PR-AUC, false-alarm and missed-attack counts, and per-client
  test metrics;
* for the variant runs, the same metrics again on the fully corrected test sample, and for
  the federated runs the client inventory of that run.

The client-inventory files hold the seed, the minimum client size, the dropped region cells,
the retained clients with their row and class counts, and the split report.
