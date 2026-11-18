# Prior-correction exploration (superseded)

Earlier exploratory scripts from the stage when the class-prior correction was
first developed. They work on a single trained checkpoint (`.pt` plus its
`_preprocess.pkl`, as written by `federated_simulation.py --save-model-path`) and
a LightGBM report from `lgbm_baseline.py` that holds the training-day class prior.

They are superseded by `experiments/run_experiments.py`, which applies the
correction to every run, selects on training-day validation data only, and writes
the results the paper reports. None of the paper's scripts import these files.

| Script | Purpose |
|---|---|
| `batch_inference_demo.py` | Scores every row of a labelled or unlabelled CSV with a checkpoint (shared helpers for the other three) |
| `prior_correct_cnn.py` | Applies no, half and full prior correction to a CNN/MLP checkpoint |
| `prior_sweep.py` | Sweeps the correction strength and reports false alarms against missed attacks |
| `per_class_correction.py` | Per-class correction strengths, tuned on half of the holdout days and reported on the other half |

The four scripts import each other, so keep them in the same folder. Run them from
the repository root, for example:

```bash
python3 tools/prior_correction/prior_sweep.py --help
```
