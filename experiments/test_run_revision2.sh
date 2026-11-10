#!/usr/bin/env bash
# Simulated full pass of run_revision2.sh on synthetic data (CPU, about 10 minutes):
# raw synthetic archives for the 8 days -> unzip the code -> every step with tiny settings,
# with, on the way: the external drive missing at the start, the drive vanishing in the
# middle of step 2, a second `start` while running, a graceful `stop` and a kill -9 during
# training, each followed by `start`. Ends with the tables and results_revision2.zip.
# Usage: bash experiments/test_run_revision2.sh WORK_DIR
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
T="${1:?usage: test_run_revision2.sh WORK_DIR}"
case "$T" in /*) ;; *) T="$PWD/$T" ;; esac
[ "$T" != "/" ] || exit 2
mkdir -p "$T"
RUN="$REPO/run_revision2.sh"
bash -n "$RUN"

DAYS_TEST="2019-12-09 2019-12-16 2019-12-23 2019-12-30"
DAYS_TRAIN="2019-01-07 2019-04-01 2019-06-24 2019-09-30"
python3 "$REPO/experiments/make_synthetic_raw.py" --out-dir "$T/D/opensky-states" \
    --date $DAYS_TEST $DAYS_TRAIN --hours 2 --aircraft 140 --seed 9 --tar \
    --tars-list "$T/D/tars.txt" > /dev/null
bash "$REPO/experiments/make_release_zip.sh" "$T/fl-ids-revision2.zip" > /dev/null

export REV2_W="$T/W" REV2_ZIP="$T/fl-ids-revision2.zip" REV2_P="$T/D" REV2_TARS="$T/D/tars.txt"
export REV2_VENV="$T/no-venv" REV2_SIM=1 REV2_SKIP_SMOKE=1 REV2_NEED_GB=0 REV2_MIN_HOURS=2
export REV2_TEST_FRAC=0.3 REV2_DEVICE=cpu REV2_STOP_TIMEOUT=120
unset REV2_CODE || true
S="$REV2_W/STATUS.txt"

is_running() { [[ "$(bash "$RUN" status 2>/dev/null)" == RUNNING* ]]; }
wait_idle() {   # wait until the background worker has exited
  local n=0
  while is_running; do
    sleep 2; n=$((n + 2)); [ $n -lt 1500 ] || { echo "timeout waiting for the worker"; tail -5 "$S"; exit 1; }
  done
}
wait_for() {    # wait until a line matching $1 is in STATUS.txt
  local n=0
  until grep -q "$1" "$S" 2>/dev/null; do
    sleep 1; n=$((n + 1)); [ $n -lt 900 ] || { echo "timeout waiting for '$1'"; tail -5 "$S"; exit 1; }
    is_running || grep -q "$1" "$S" || { echo "worker ended before '$1'"; tail -8 "$S"; exit 1; }
  done
}

echo "== dry-run"
bash "$RUN" dry-run > "$T/dry_before.txt"
grep -q "reprocess_mv.py" "$T/dry_before.txt"

echo "== 1. external drive missing at the start"
mv "$T/D" "$T/D_unplugged"
bash "$RUN" start > /dev/null
wait_idle
grep -q "external drive is not connected" "$S" && grep -q "wsl --shutdown" "$S"
[ -f "$REV2_W/done/1_setup" ] && [ ! -f "$REV2_W/done/2_reprocess" ]
[ -f "$REV2_W/rev2/experiments/run_experiments.py" ]     # step 1 unpacked the zip
bash "$RUN" dry-run > "$T/dry.txt"
[ "$(grep -c 'run_experiments.py' "$T/dry.txt")" -ge 30 ]

echo "== 2. drive back, then it vanishes in the middle of step 2"
mv "$T/D_unplugged" "$T/D"
mv "$T/D/opensky-states/.2019-09-30" "$T/hidden_day"
bash "$RUN" start > /dev/null
sleep 1
[[ "$(bash "$RUN" start)" == *"Already running"* ]]           # no second copy
wait_idle
grep -q "external drive D: is not readable" "$S"
[ ! -f "$REV2_W/done/2_reprocess" ]
[ "$(ls "$REV2_W"/labeled_mv/*.done.json | wc -l)" -eq 7 ]
mv "$T/hidden_day" "$T/D/opensky-states/.2019-09-30"

echo "== 3. start again; graceful stop during the main grid"
bash "$RUN" start > /dev/null
wait_for "STEP 5/11"
grep -q "can be unplugged" "$S"
sleep 6
bash "$RUN" stop > "$T/stop.txt"
! is_running
grep -q "STOPPED on request" "$S" || grep -q "STOPPED (forced" "$S"
[ ! -f "$REV2_W/done/11_results" ]

echo "== 4. start again; kill -9 of the whole job during the ablation"
bash "$RUN" start > /dev/null
wait_for "STEP 6/11"
sleep 8
PID="$(cat "$REV2_W/run.pid")"
kill -KILL -- "-$PID" 2>/dev/null || kill -KILL "$PID"
sleep 1
! is_running

echo "== 5. start again and let it finish"
bash "$RUN" start > /dev/null
wait_idle
bash "$RUN" status > "$T/status_final.txt"
grep -q "everything is finished" "$T/status_final.txt" || { cat "$T/status_final.txt"; tail -20 "$S"; exit 1; }
[ -f "$REV2_W/results_revision2.zip" ] && [ -f "$T/D/results_revision2.zip" ]
[[ "$(unzip -l "$REV2_W/results_revision2.zip")" == *"generated/tab_ablation_corrections.tex"* ]]
for f in tab_ablation_corrections tab_unified_pipeline tab_selected_s tab_baselines tab_prauc; do
  [ -s "$REV2_W/generated/$f.tex" ]
done
! grep -q -- "& -- & -- & -- & -- & -- & -- \\\\\\\\" "$REV2_W/generated/tab_baselines.tex"
[ -z "$(find "$REV2_W" -name '*.tmp' -not -path '*/smoke/*' | head -1)" ] || { echo "leftover .tmp files:"; find "$REV2_W" -name '*.tmp' | head; }
python3 "$REV2_W/rev2/experiments/rev2_plan.py" status
bash "$RUN" start > /dev/null; wait_idle           # a further start has nothing to do
tail -3 "$S"
echo "RUN_REVISION2 SIMULATION OK ($(grep -c 'STOPPED' "$S") stops, results in $REV2_W)"
