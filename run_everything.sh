#!/usr/bin/env bash
# Runs every step of RUN_ALL.md in order, unattended, and collects the results.
# Safe to re-run: finished steps are skipped (marker files in $OUT/done/).
#
#   nohup bash /mnt/d/fl-ids-sweep/run_everything.sh > /mnt/d/fl-ids-sweep/all_results/master.log 2>&1 &
#
# Progress:  cat /mnt/d/fl-ids-sweep/all_results/STATUS.txt
# Final:     /mnt/d/fl-ids-sweep/results_for_paper.zip

P=/mnt/d/fl-ids-sweep
ZIP_IN=/mnt/c/Users/Dell/Downloads/fl-ids-revision.zip
OUT=$P/all_results
mkdir -p "$OUT/done" "$OUT/logs"
STATUS=$OUT/STATUS.txt

say()  { echo "$(date '+%a %H:%M')  $*" | tee -a "$STATUS"; }
done_() { touch "$OUT/done/$1"; }
is_done() { [ -f "$OUT/done/$1" ]; }

source ~/flids-venv/bin/activate || { say "ERROR: venv ~/flids-venv not found"; exit 1; }
[ -f "$P/tars.txt" ] || { say "ERROR: $P/tars.txt missing (list of raw .tar files)"; exit 1; }

# ---------------------------------------------------------------- 1. setup
if ! is_done 1_setup; then
  say "STEP 1 setup + smoke test"
  cd "$P"
  [ -f "$ZIP_IN" ] && unzip -oq "$ZIP_IN"
  cd "$P/rev" || { say "ERROR: $P/rev missing"; exit 1; }
  pip install -q -r requirements.txt > "$OUT/logs/1_pip.log" 2>&1 || say "warning: pip install had errors, see logs/1_pip.log"
  if bash experiments/smoke_test.sh /tmp/fl_smoke > "$OUT/logs/1_smoke.log" 2>&1 \
       && grep -q "SMOKE TEST OK" "$OUT/logs/1_smoke.log"; then
    done_ 1_setup; say "STEP 1 ok"
  else
    say "ERROR: smoke test failed, see logs/1_smoke.log. Stopping."; exit 1
  fi
fi
cd "$P/rev"

# ---------------------------------------------------------------- 2. reprocess 8 days (v3)
if ! is_done 2_reprocess; then
  say "STEP 2 reprocess 8 Mondays with pipeline v3 (5-6 h)"
  mkdir -p "$P/labeled_v3" ~/raw_tmp
  fails=0
  for d in 2019-12-09 2019-12-16 2019-12-23 2019-12-30 2019-01-07 2019-04-01 2019-06-24 2019-09-30; do
    out=$P/labeled_v3/opensky_${d}_labeled.csv
    [ -f "$out" ] && { say "  skip $d (exists)"; continue; }
    say "  >>> $d"
    tmp=~/raw_tmp/$d; rm -rf "$tmp"; mkdir -p "$tmp"
    grep "states_${d}-" "$P/tars.txt" | while read -r t; do
      tar -xOf "$t" --wildcards '*.csv.gz' | gunzip > "$tmp/$(basename "$t" .tar)"
    done
    if python3 -u opensky_fl_ids_pipeline.py $(ls "$tmp"/states_${d}-*.csv | sort) \
         --chunk-files 1 --output "$out.part" >> "$OUT/logs/2_reprocess.log" 2>&1; then
      mv "$out.part" "$out"; say "  <<< $d ok"
    else
      say "  FAILED $d (see logs/2_reprocess.log)"; fails=$((fails+1))
    fi
    rm -rf "$tmp"
  done
  [ $fails -eq 0 ] || { say "ERROR: $fails day(s) failed in step 2. Stopping."; exit 1; }
  done_ 2_reprocess
fi

# ---------------------------------------------------------------- 3. test sample
if ! is_done 3_holdout; then
  say "STEP 3 December test sample (30-60 min)"
  L=$P/labeled_v3
  python3 -u experiments/make_holdout_sample.py \
    $L/opensky_2019-12-09_labeled.csv $L/opensky_2019-12-16_labeled.csv \
    $L/opensky_2019-12-23_labeled.csv $L/opensky_2019-12-30_labeled.csv \
    --out $P/holdout_sample_v3.csv > "$OUT/logs/3_holdout.log" 2>&1 \
    && { done_ 3_holdout; say "STEP 3 ok"; } \
    || { say "ERROR: step 3 failed, see logs/3_holdout.log. Stopping."; exit 1; }
fi

# ---------------------------------------------------------------- 4a. data checks
if ! is_done 4a_checks; then
  say "STEP 4a data checks"
  python3 - > "$OUT/step4a_checks.txt" 2>&1 <<'EOF'
import pandas as pd
x = pd.read_csv("/mnt/d/fl-ids-sweep/labeled_v3/opensky_2019-01-07_labeled.csv",
                usecols=["attack_type", "inter_arrival_s"], nrows=3_000_000)
ben = x[x.attack_type == "benign"]
b = ben.inter_arrival_s.dropna()
print("benign inter-arrival quantiles (s):")
print(b.quantile([0.001, 0.01, 0.05, 0.25, 0.5, 0.75]).round(2))
print("share of benign gaps under 1 s:", round((b < 1).mean(), 5))
f = x[x.attack_type == "flooding"].inter_arrival_s.dropna()
print("flooding median gap (s):", round(f.median(), 3))
print("share of benign rows with no previous message in chunk (first-in-chunk):",
      round(ben.inter_arrival_s.isna().mean(), 5))
EOF
  done_ 4a_checks; say "STEP 4a ok (step4a_checks.txt)"
fi

# ---------------------------------------------------------------- 4b. leakage audit
if ! is_done 4b_audit; then
  say "STEP 4b leakage audit (~1 h)"
  mkdir -p "$P/audit_v3" ~/raw_tmp/2019-01-07
  grep "states_2019-01-07-" "$P/tars.txt" | while read -r t; do
    tar -xOf "$t" --wildcards '*.csv.gz' | gunzip > ~/raw_tmp/2019-01-07/$(basename "$t" .tar)
  done
  python3 -u region_lookup_check.py $(ls ~/raw_tmp/2019-01-07/states_2019-01-07-*.csv | sort) \
      --chunk-files 1 --out $P/audit_v3/region_stats.csv > "$OUT/logs/4b_region.log" 2>&1
  rm -rf ~/raw_tmp/2019-01-07
  RS=""; [ -f $P/audit_v3/region_stats.csv ] && RS="--region-stats $P/audit_v3/region_stats.csv"
  python3 -u leakage_audit.py $P/labeled_v3/opensky_2019-01-07_labeled.csv \
      $P/holdout_sample_v3.csv --out $P/audit_v3 $RS > "$OUT/logs/4b_audit.log" 2>&1 \
    && { done_ 4b_audit; say "STEP 4b ok"; } \
    || say "warning: audit failed (logs/4b_audit.log); continuing with experiments"
fi

# ---------------------------------------------------------------- 6. main experiments
D=$P
if ! is_done 6_runs; then
  say "STEP 6 main experiments, 30 runs (~10 h). Live log: rev_runs/run_experiments.log"
  mkdir -p $D/rev_runs
  python3 -u experiments/run_experiments.py \
    --train-files $D/labeled_v3/opensky_2019-01-07_labeled.csv \
                  $D/labeled_v3/opensky_2019-04-01_labeled.csv \
                  $D/labeled_v3/opensky_2019-06-24_labeled.csv \
                  $D/labeled_v3/opensky_2019-09-30_labeled.csv \
    --holdout $D/holdout_sample_v3.csv \
    --out-dir $D/rev_runs --max-total-rows 10000000 --device auto \
    > $D/rev_runs/nohup.out 2>&1 \
    && { done_ 6_runs; say "STEP 6 ok ($(ls $D/rev_runs/runs/*__seed*.json 2>/dev/null | wc -l) run files)"; } \
    || say "warning: step 6 ended with an error (rev_runs/nohup.out); rerun this script to resume"
fi

# ---------------------------------------------------------------- 7. old-pipeline comparison
if ! is_done 7_v2 && [ -f $D/holdout_sample.csv ]; then
  say "STEP 7 comparison on old v2 files (~4 h)"
  mkdir -p $D/rev_runs_v2
  V2=(--train-files $D/labeled/opensky_2019-01-07_labeled.csv $D/labeled/opensky_2019-04-01_labeled.csv
      $D/labeled/opensky_2019-06-24_labeled.csv $D/labeled/opensky_2019-09-30_labeled.csv
      --holdout $D/holdout_sample.csv --out-dir $D/rev_runs_v2
      --max-total-rows 10000000 --device auto --feature-sets full)
  python3 -u experiments/run_experiments.py "${V2[@]}" --models lgbm --seeds 1,2,3,4,5 \
      > $D/rev_runs_v2/nohup_lgbm.out 2>&1 \
    && python3 -u experiments/run_experiments.py "${V2[@]}" --models fed_cnn --seeds 1,2,3 \
      > $D/rev_runs_v2/nohup_fedcnn.out 2>&1 \
    && { done_ 7_v2; say "STEP 7 ok"; } \
    || say "warning: step 7 ended with an error (rev_runs_v2/nohup_*.out)"
elif [ ! -f $D/holdout_sample.csv ]; then
  say "STEP 7 skipped: old $D/holdout_sample.csv not found"
fi

# ---------------------------------------------------------------- 8. tables + zip
say "STEP 8 building tables and results zip"
ARGS=(--runs-dirs $D/rev_runs)
[ -d $D/rev_runs_v2/runs ] && ARGS+=(--v2-runs-dir $D/rev_runs_v2)
[ -f $D/audit_v3/feature_audit_table.tex ] && ARGS+=(--audit-table $D/audit_v3/feature_audit_table.tex)
python3 experiments/aggregate_results.py "${ARGS[@]}" > "$OUT/logs/8_aggregate.log" 2>&1 \
  || say "warning: aggregator error (logs/8_aggregate.log)"
cp $P/reprocess_v3.log "$OUT/logs/" 2>/dev/null
cd $P && rm -f results_for_paper.zip
zip -qr results_for_paper.zip rev/paper/generated rev_runs/runs rev_runs/cache/*.json \
    rev_runs/run_experiments.log rev_runs_v2/runs audit_v3 all_results 2>/dev/null
say "ALL DONE. Send me: $P/results_for_paper.zip ($(du -h $P/results_for_paper.zip | cut -f1))"
