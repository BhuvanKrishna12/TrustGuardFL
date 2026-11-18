# Flooding sentinel diagnostics

One-off checks from the investigation that led to the 0.1 s time-gap floor in
`opensky_fl_ids_pipeline.py` (`MIN_DT_FLOOR_S`). Before the fix, the pipeline
left the four kinematic rate features (`implied_speed_mps`,
`speed_consistency_delta`, `vrate_consistency_delta`, `turn_rate_deg_s`) blank on
flooding duplicates, whose time gap to the previous message is near zero. After
the usual fill with -1, that blank pattern identified flooding rows by itself.
These scripts detect that signature.

| Script | Purpose |
|---|---|
| `check_flooding_fix.py` | Share of rows per class whose four kinematic features are all blank in a labelled CSV |
| `ablation_test.py` | Replaces the -1 pattern on real flooding rows with typical values and measures how much a trained model's flooding confidence drops |
| `stratified_flooding_check.py` | Scores flooding rows with and without the -1 pattern separately, to see whether the model detects flooding without the shortcut |

`ablation_test.py` and `stratified_flooding_check.py` need a trained checkpoint
from `federated_simulation.py --save-model-path`. Run from the repository root,
for example:

```bash
python3 tools/diagnostics/check_flooding_fix.py labeled.csv
```
