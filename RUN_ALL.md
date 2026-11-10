# Revision: everything to run on the laptop, in order

All commands are for WSL on the laptop. Total time is roughly 21 to 24 hours of
unattended running, spread over steps 2, 6 and 7. Everything runs in the background
with `nohup`, so the terminal can be closed. Keep Windows from sleeping.

Paths used below:

| What | Where |
|---|---|
| Revision code (this folder, unzipped) | `/mnt/d/fl-ids-sweep/rev` |
| Raw OpenSky archives | `/mnt/d/opensky-states` |
| Old labeled days (pipeline v2, kept for comparison) | `/mnt/d/fl-ids-sweep/labeled` |
| New labeled days (pipeline v3) | `/mnt/d/fl-ids-sweep/labeled_v3` |
| Experiment outputs (v3) | `/mnt/d/fl-ids-sweep/rev_runs` |
| Comparison runs on the old v2 files | `/mnt/d/fl-ids-sweep/rev_runs_v2` |

## 1. Set up (5 minutes)

```bash
cd /mnt/d/fl-ids-sweep
unzip -o /mnt/c/Users/Dell/Downloads/fl-ids-revision.zip     # creates /mnt/d/fl-ids-sweep/rev
source ~/flids-venv/bin/activate
cd rev
pip install -r requirements.txt     # pinned versions; for torch see the note in the file
bash experiments/smoke_test.sh /tmp/fl_smoke                 # 5-minute self-test, must end with "SMOKE TEST OK"
```

## 2. Reprocess the 8 Mondays with the corrected pipeline (about 5 to 6 hours)

Same loop as before, but writing to `labeled_v3`. The corrected pipeline (v3) is the default.

```bash
cd /mnt/d/fl-ids-sweep/rev
P=/mnt/d/fl-ids-sweep
mkdir -p $P/labeled_v3 ~/raw_tmp
cat > $P/reprocess_v3.sh <<'EOF'
P=/mnt/d/fl-ids-sweep
cd $P/rev
source ~/flids-venv/bin/activate
for d in 2019-12-09 2019-12-16 2019-12-23 2019-12-30 2019-01-07 2019-04-01 2019-06-24 2019-09-30; do
  out=$P/labeled_v3/opensky_${d}_labeled.csv
  [ -f "$out" ] && { echo "skip $d"; continue; }
  echo ">>> $d $(date +%H:%M)"
  tmp=~/raw_tmp/$d; mkdir -p "$tmp"
  grep "states_${d}-" $P/tars.txt | while read t; do
    tar -xOf "$t" --wildcards '*.csv.gz' | gunzip > "$tmp/$(basename "$t" .tar)"
  done
  python3 opensky_fl_ids_pipeline.py $(ls "$tmp"/states_${d}-*.csv | sort) \
    --chunk-files 1 --output "$out.part" && mv "$out.part" "$out" || echo "FAILED $d"
  rm -rf "$tmp"
  echo "<<< $d $(date +%H:%M)"
done
EOF
nohup bash $P/reprocess_v3.sh > $P/reprocess_v3.log 2>&1 &
```

Check progress: `grep -E '^(>>>|<<<|FAILED)' /mnt/d/fl-ids-sweep/reprocess_v3.log`

## 3. Rebuild the December test sample from the new files (about 30 to 60 minutes)

Same 1% sample and seed as before (pandas `sample(frac=0.01, random_state=42)` on every
2,000,000-row chunk), from the v3 files.

```bash
cd /mnt/d/fl-ids-sweep/rev && source ~/flids-venv/bin/activate
P=/mnt/d/fl-ids-sweep
python3 experiments/make_holdout_sample.py \
  $P/labeled_v3/opensky_2019-12-09_labeled.csv $P/labeled_v3/opensky_2019-12-16_labeled.csv \
  $P/labeled_v3/opensky_2019-12-23_labeled.csv $P/labeled_v3/opensky_2019-12-30_labeled.csv \
  --out $P/holdout_sample_v3.csv
```

## 4. Two checks the paper needs (about 1 hour)

**a) Snapshot interval of the data** (the paper says benign messages from one aircraft
never arrive less than a few seconds apart; this confirms the interval):

```bash
python3 - <<'EOF'
import pandas as pd
x = pd.read_csv("/mnt/d/fl-ids-sweep/labeled_v3/opensky_2019-01-07_labeled.csv",
                usecols=["attack_type", "inter_arrival_s"], nrows=3_000_000)
b = x[x.attack_type == "benign"].inter_arrival_s.dropna()
print("benign inter-arrival quantiles (s):")
print(b.quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75]).round(2))
print("share of benign gaps under 1 s:", round((b < 1).mean(), 5))
f = x[x.attack_type == "flooding"].inter_arrival_s.dropna()
print("flooding median gap (s):", round(f.median(), 3))
EOF
```

**b) Leakage audit** on one training day and on the test sample. First the exact region
re-lookup numbers from one training day's raw files (instrumented pipeline run), then the
audit with `--region-stats` so the table gets exact region numbers:

```bash
P=/mnt/d/fl-ids-sweep
mkdir -p $P/audit_v3 ~/raw_tmp/2019-01-07
grep "states_2019-01-07-" $P/tars.txt | while read t; do
  tar -xOf "$t" --wildcards '*.csv.gz' | gunzip > ~/raw_tmp/2019-01-07/$(basename "$t" .tar)
done
python3 region_lookup_check.py $(ls ~/raw_tmp/2019-01-07/states_2019-01-07-*.csv | sort) \
    --chunk-files 1 --out $P/audit_v3/region_stats.csv
rm -rf ~/raw_tmp/2019-01-07
python3 leakage_audit.py $P/labeled_v3/opensky_2019-01-07_labeled.csv \
    $P/holdout_sample_v3.csv --out $P/audit_v3 --region-stats $P/audit_v3/region_stats.csv
```

## 5. (Optional) smoke-check the experiment runner on real data

```bash
python3 experiments/run_experiments.py --quick --seeds 1 --models lgbm --feature-sets full \
  --train-files /mnt/d/fl-ids-sweep/labeled_v3/opensky_2019-01-07_labeled.csv \
  --holdout /mnt/d/fl-ids-sweep/holdout_sample_v3.csv --out-dir /mnt/d/fl-ids-sweep/rev_quick
```

## 6. All experiments (about 10 hours)

3 models x 2 feature sets x 5 seeds = 30 runs. Resumable: rerun the same command after
any crash and finished runs are skipped. The first 1 to 1.5 hours are one-off passes over
the training files (raw counts, prior and aircraft list; then the uniform training sample),
cached in `rev_runs/cache`.

```bash
cd /mnt/d/fl-ids-sweep/rev && source ~/flids-venv/bin/activate
D=/mnt/d/fl-ids-sweep
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
```

To split with your friend's machine, see `experiments/README.md` ("Splitting across two
machines"): for example `--seeds 1,2,3` here and `--seeds 4,5` there.

## 7. Comparison runs on the old v2 files (about 4 hours)

Same harness on the old labelled files and the old test sample, full feature set only:
LightGBM with 5 seeds first, then the federated CNN with seeds 1, 2, 3 if time allows
(each fed CNN run is about 35 minutes; it can be stopped at any point, finished runs are kept).

```bash
cd /mnt/d/fl-ids-sweep/rev && source ~/flids-venv/bin/activate
D=/mnt/d/fl-ids-sweep
mkdir -p $D/rev_runs_v2
V2=(--train-files $D/labeled/opensky_2019-01-07_labeled.csv $D/labeled/opensky_2019-04-01_labeled.csv
    $D/labeled/opensky_2019-06-24_labeled.csv $D/labeled/opensky_2019-09-30_labeled.csv
    --holdout $D/holdout_sample.csv --out-dir $D/rev_runs_v2
    --max-total-rows 10000000 --device auto --feature-sets full)
nohup bash -c "python3 experiments/run_experiments.py ${V2[*]} --models lgbm --seeds 1,2,3,4,5 && \
               python3 experiments/run_experiments.py ${V2[*]} --models fed_cnn --seeds 1,2,3" \
  > $D/rev_runs_v2/nohup.out 2>&1 &
tail -f $D/rev_runs_v2/run_experiments.log
```

## 8. Build the paper's tables and send the results back

```bash
cd /mnt/d/fl-ids-sweep/rev
python3 experiments/aggregate_results.py --runs-dirs /mnt/d/fl-ids-sweep/rev_runs \
    --v2-runs-dir /mnt/d/fl-ids-sweep/rev_runs_v2 \
    --audit-table /mnt/d/fl-ids-sweep/audit_v3/feature_audit_table.tex
cd /mnt/d/fl-ids-sweep
zip -r results_for_paper.zip rev/paper/generated rev_runs/runs rev_runs_v2/runs \
    rev_runs/cache/*.json audit_v3 reprocess_v3.log
```

Send `results_for_paper.zip` plus the printed output of step 4a. The tables and numbers
then fill in automatically; the red "pending" sentences get written from them.
