#!/usr/bin/env bash
# Revision 2: everything in one script. Can be stopped and started again at ANY point.
#
#   bash run_revision2.sh start     start (or continue) in the background
#   bash run_revision2.sh status    where it is, how long is left
#   bash run_revision2.sh stop      stop cleanly (finishes the current small unit first)
#   bash run_revision2.sh dry-run   print every command without running anything
#
# All new data lives on the WSL disk in ~/rev2_work. Only step 2 reads the external
# drive D:. Closing the lid / sleep just pauses the run; it continues on wake-up.
# Read RUN_REVISION2.md for the short guide.
#
# Paths can be overridden with environment variables (used by the self-test):
#   REV2_W work dir | REV2_ZIP code zip | REV2_CODE use this code folder instead of the zip
#   REV2_P project folder on D: | REV2_TARS list of raw archives | REV2_VENV python venv
#   REV2_SIM=1 tiny settings for synthetic data | REV2_SKIP_SMOKE=1 | REV2_NEED_GB | REV2_STOP_TIMEOUT
set -u

W="${REV2_W:-$HOME/rev2_work}"
ZIP_IN="${REV2_ZIP:-/mnt/c/Users/Dell/Downloads/fl-ids-revision2.zip}"
P="${REV2_P:-/mnt/d/fl-ids-sweep}"
TARS_SRC="${REV2_TARS:-$P/tars.txt}"
VENV="${REV2_VENV:-$HOME/flids-venv}"
CODE="${REV2_CODE:-$W/rev2}"
SIM="${REV2_SIM:-0}"
STOP_TIMEOUT="${REV2_STOP_TIMEOUT:-600}"
TEST_DAYS="${REV2_TEST_DAYS:-2019-12-09 2019-12-16 2019-12-23 2019-12-30}"
TRAIN_DAYS="${REV2_TRAIN_DAYS:-2019-01-07 2019-04-01 2019-06-24 2019-09-30}"
MIN_HOURS="${REV2_MIN_HOURS:-24}"
TEST_FRAC="${REV2_TEST_FRAC:-0.01}"
# Free space needed for step 2: the v3 labelled set is 110 GB; the multi-variant files are
# 8.3% larger (measured: about 25 extra bytes per row, experiments/test_variants.py),
# i.e. about 119 GB, plus 15 GB for temporary hourly files, samples and checkpoints.
NEED_GB="${REV2_NEED_GB:-135}"

[ -n "$W" ] && [ "$W" != "/" ] || { echo "REV2_W is empty or /; refusing"; exit 2; }
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
STATUS="$W/STATUS.txt"; LOCK="$W/run.pid"; STOPF="$W/STOP"; DONE="$W/done"; LOGS="$W/logs"
export REV2_W="$W" REV2_SIM="$SIM" REV2_STOP_FLAG="$STOPF" REV2_TRAIN_DAYS="$TRAIN_DAYS" REV2_TEST_DAYS="$TEST_DAYS"
export PYTHONUNBUFFERED=1

say()      { mkdir -p "$W"; echo "$(date '+%a %d %b %H:%M')  $*" | tee -a "$STATUS"; }
is_done()  { [ -f "$DONE/$1" ]; }
mark_done(){ mkdir -p "$DONE"; date > "$DONE/$1.tmp" && mv "$DONE/$1.tmp" "$DONE/$1"; }
stop_asked(){ [ -f "$STOPF" ]; }
running_pid() {
  [ -f "$LOCK" ] || return 1
  local pid; pid="$(cat "$LOCK" 2>/dev/null)"
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && grep -qa "run_revision2" "/proc/$pid/cmdline" 2>/dev/null \
    && { echo "$pid"; return 0; }
  return 1
}
free_gb()  { df -Pk "$1" 2>/dev/null | awk 'NR==2 {printf "%d", $4/1048576}'; }
drive_help() {
  say "  HOW TO RECOVER: reconnect the external drive. If 'ls /mnt/d' says 'No such device',"
  say "  open PowerShell, run:  wsl --shutdown   then reopen WSL and run:  bash ~/run_revision2.sh start"
  say "  Nothing is lost; finished hours and days are kept."
}
plan()     { python3 "$CODE/experiments/rev2_plan.py" "$@"; }
activate() {
  if [ -f "$VENV/bin/activate" ]; then # shellcheck disable=SC1091
    source "$VENV/bin/activate"
  elif [ "$SIM" != "1" ]; then say "ERROR: python environment $VENV not found"; return 1; fi
}

# ------------------------------------------------------------------ steps
step_1() {  # setup + self-test
  say "STEP 1/11 setup and self-test (about 20 minutes)"
  if [ -z "${REV2_CODE:-}" ]; then
    if [ -f "$ZIP_IN" ]; then
      rm -rf "$W/rev2.new" && mkdir -p "$W/rev2.new" && unzip -oq "$ZIP_IN" -d "$W/rev2.new" \
        && [ -f "$W/rev2.new/rev2/experiments/run_experiments.py" ] \
        || { say "ERROR: could not unpack $ZIP_IN"; return 1; }
      rm -rf "$W/rev2.old"; [ -d "$W/rev2" ] && mv "$W/rev2" "$W/rev2.old"
      mv "$W/rev2.new/rev2" "$W/rev2" && rm -rf "$W/rev2.new" "$W/rev2.old"
    elif [ ! -f "$CODE/experiments/run_experiments.py" ]; then
      say "ERROR: $ZIP_IN not found. Put fl-ids-revision2.zip into the Windows Downloads folder."; return 1
    fi
  fi
  [ -f "$CODE/experiments/run_experiments.py" ] || { say "ERROR: code folder $CODE is incomplete"; return 1; }
  if ! python3 -c "import torch, lightgbm, pandas, sklearn, joblib, scipy" 2>"$LOGS/1_import.log"; then
    say "  some python packages are missing; installing from requirements.txt"
    pip install -q -r "$CODE/requirements.txt" > "$LOGS/1_pip.log" 2>&1 \
      || { say "ERROR: pip install failed, see $LOGS/1_pip.log"; return 1; }
  fi
  if [ "${REV2_SKIP_SMOKE:-0}" != "1" ]; then
    rm -rf "$W/smoke"
    if bash "$CODE/experiments/smoke_test.sh" "$W/smoke" > "$LOGS/1_smoke.log" 2>&1 \
         && grep -q "SMOKE TEST OK" "$LOGS/1_smoke.log"; then
      rm -rf "$W/smoke"
    else
      say "ERROR: self-test failed, see $LOGS/1_smoke.log. Send that file back."; return 1
    fi
  fi
}

step_2() {  # multi-variant reprocess of the 8 Mondays (the only step that reads D:)
  say "STEP 2/11 reprocess the 8 Mondays from the raw archives on D: (about 6 hours; stop/start any time)"
  if [ -r "$TARS_SRC" ]; then cp "$TARS_SRC" "$W/tars.txt.tmp" && mv "$W/tars.txt.tmp" "$W/tars.txt"
  elif [ ! -f "$W/tars.txt" ]; then
    say "STOPPED: the external drive is not connected ($TARS_SRC not readable)."; drive_help; return 3
  fi
  local have; have=$(du -sk "$W/labeled_mv" 2>/dev/null | awk '{printf "%d", $1/1048576}'); have=${have:-0}
  local free; free=$(free_gb "$W"); free=${free:-0}
  if [ $((free + have)) -lt "$NEED_GB" ]; then
    say "ERROR: not enough free space on the WSL disk: $free GB free (+$have GB already written), need $NEED_GB GB."
    say "  Free some space (or ask for help) and run: bash ~/run_revision2.sh start"; return 1
  fi
  say "  disk check ok: $free GB free on the WSL disk, about $((NEED_GB - have)) GB still needed"
  mkdir -p "$W/labeled_mv" "$W/raw_tmp"
  local d rc
  for d in $TEST_DAYS $TRAIN_DAYS; do
    stop_asked && return 75
    if [ -f "$W/labeled_mv/opensky_${d}_labeled.csv.done.json" ]; then continue; fi
    say "  day $d ..."
    python3 -u "$CODE/experiments/reprocess_mv.py" --tars-list "$W/tars.txt" --days "$d" \
        --out-dir "$W/labeled_mv" --tmp-dir "$W/raw_tmp" --stop-flag "$STOPF" --min-hours "$MIN_HOURS" \
        >> "$LOGS/2_reprocess.log" 2>&1
    rc=$?
    if [ $rc -eq 3 ]; then
      say "STOPPED in step 2 (day $d): the external drive D: is not readable any more."; drive_help; return 3
    elif [ $rc -eq 75 ]; then return 75
    elif [ $rc -ne 0 ]; then say "ERROR: day $d failed (see $LOGS/2_reprocess.log)"; return 1; fi
    say "  day $d done"
  done
  for d in $TEST_DAYS $TRAIN_DAYS; do
    python3 - "$W/labeled_mv/opensky_${d}_labeled.csv" <<'PY' || { say "ERROR: $d did not verify"; return 1; }
import json, os, sys
p = sys.argv[1]
d = json.load(open(p + ".done.json"))
assert os.path.getsize(p) == d["bytes"] and d["rows"] > 0
PY
  done
  rm -rf "$W/raw_tmp"
  say "  STEP 2 finished. The external drive D: is NOT needed any more and can be unplugged."
}

step_3() {  # test sample
  say "STEP 3/11 December test sample (about 45 minutes)"
  local files=() d
  for d in $TEST_DAYS; do files+=("$W/labeled_mv/opensky_${d}_labeled.csv"); done
  python3 -u "$CODE/experiments/make_holdout_sample.py" --method hash --frac "$TEST_FRAC" "${files[@]}" \
      --out "$W/holdout_sample_mv.csv" --stop-flag "$STOPF" >> "$LOGS/3_holdout.log" 2>&1
  local rc=$?; [ $rc -eq 0 ] || return $rc
  python3 - "$W/holdout_sample_mv.csv" <<'PY' || return 1
import json, os, sys
p = sys.argv[1]
d = json.load(open(p + ".done.json"))
assert os.path.getsize(p) == d["bytes"] and d["rows"] > 0
PY
}

run_step() {  # steps 4..10: the run grid (experiments/rev2_plan.py)
  local n=$1 title=$2 est=$3
  say "STEP $n/11 $title ($est)"
  plan run --step "$n" >> "$LOGS/${n}_runs.log" 2>&1
}

step_11() { # tables + zip
  say "STEP 11/11 tables and results zip (a few minutes)"
  mkdir -p "$W/generated"
  python3 "$CODE/experiments/aggregate_results.py" --runs-dirs "$W/runs/main" --rev2-root "$W/runs" \
      --tex-dir "$W/generated" > "$LOGS/11_aggregate.log" 2>&1 \
      || { say "ERROR: building the tables failed (see $LOGS/11_aggregate.log)"; return 1; }
  ( cd "$W" && rm -f results_revision2.zip.tmp && \
    zip -qr results_revision2.zip.tmp generated STATUS.txt logs done cache/*.json \
        labeled_mv/*.done.json holdout_sample_mv.csv.done.json \
        runs/*/runs runs/*/run_experiments.log -x '*.tmp' && \
    python3 -c "import zipfile,sys; z=zipfile.ZipFile('results_revision2.zip.tmp'); assert z.testzip() is None and len(z.namelist())>10" && \
    mv results_revision2.zip.tmp results_revision2.zip ) || { say "ERROR: could not write the zip"; return 1; }
  local size; size=$(du -h "$W/results_revision2.zip" | cut -f1)
  if [ -d "$P" ] && cp "$W/results_revision2.zip" "$P/results_revision2.zip.tmp" 2>/dev/null \
       && mv "$P/results_revision2.zip.tmp" "$P/results_revision2.zip"; then
    say "ALL DONE. Send back: $P/results_revision2.zip ($size)  (in Windows: D:\\fl-ids-sweep\\results_revision2.zip)"
  else
    say "ALL DONE. Send back: $W/results_revision2.zip ($size)"
    say "  D: is not connected, so the zip stayed on the WSL disk. In Windows Explorer open:"
    say "  \\\\wsl\$\\Ubuntu\\home\\$(basename "$HOME")\\$(basename "$W")"
  fi
}

STEP_IDS="1_setup 2_reprocess 3_testsample 4_sampling 5_main 6_ablation 7_unified 8_mlp 9_fedprox 10_fedablation 11_results"
do_step() {
  case "$1" in
    1_setup) step_1 ;;
    2_reprocess) step_2 ;;
    3_testsample) step_3 ;;
    4_sampling)     run_step 4  "one-off sampling passes over the training files" "about 3 hours" ;;
    5_main)         run_step 5  "main grid: LightGBM and central CNN first, federated CNN last, 30 runs" "about 7.5 hours" ;;
    6_ablation)     run_step 6  "correction ablation with LightGBM and central CNN, 96 runs" "about 3 hours" ;;
    7_unified)      run_step 7  "earlier vs corrected pipeline, one protocol, 7 runs" "about 3.5 hours" ;;
    8_mlp)          run_step 8  "MLP baselines, 10 runs" "about 3 hours" ;;
    9_fedprox)      run_step 9  "FedProx, 6 runs" "about 4 hours" ;;
    10_fedablation) run_step 10 "federated ablation, 12 runs" "about 8 hours" ;;
    11_results) step_11 ;;
  esac
}

worker() {
  mkdir -p "$W" "$DONE" "$LOGS"
  echo $$ > "$LOCK.tmp" && mv "$LOCK.tmp" "$LOCK"
  trap 'rm -f "$LOCK"' EXIT
  trap 'say "interrupted (signal). Run: bash ~/run_revision2.sh start  to continue."; exit 130' INT TERM
  say "---- run_revision2 started (pid $$) ----"
  activate || exit 1
  local id rc
  for id in $STEP_IDS; do
    if stop_asked; then say "STOPPED on request before step ${id%%_*}. To continue: bash ~/run_revision2.sh start"; exit 0; fi
    is_done "$id" && continue
    if [ "$id" != "1_setup" ] && [ ! -f "$CODE/experiments/rev2_plan.py" ]; then
      say "ERROR: code folder $CODE missing"; exit 1; fi
    do_step "$id"; rc=$?
    if [ $rc -eq 0 ]; then
      mark_done "$id"; say "step ${id%%_*} finished"
    elif [ $rc -eq 75 ]; then
      say "STOPPED on request during step ${id%%_*}; everything finished so far is saved. To continue: bash ~/run_revision2.sh start"; exit 0
    elif [ $rc -eq 3 ]; then
      exit 3
    else
      say "ERROR in step ${id%%_*} (code $rc). Nothing was marked done. Look at $LOGS, then run: bash ~/run_revision2.sh start"; exit 1
    fi
  done
  say "---- everything is finished ----"
}

# ------------------------------------------------------------------ commands
cmd_start() {
  mkdir -p "$W" "$LOGS"
  local pid
  if pid=$(running_pid); then echo "Already running (pid $pid). Use: bash $0 status"; exit 0; fi
  rm -f "$STOPF" "$LOCK"
  if command -v setsid >/dev/null; then
    setsid nohup bash "$SELF" _worker >> "$LOGS/master.log" 2>&1 < /dev/null &
  else
    nohup bash "$SELF" _worker >> "$LOGS/master.log" 2>&1 < /dev/null &
  fi
  local newpid=$!
  # Wait briefly for the worker to claim the lock, so `status` and a second `start`
  # see it as running the instant this returns (avoids a start/status race).
  local i=0
  while [ $i -lt 100 ]; do
    running_pid >/dev/null 2>&1 && break
    kill -0 "$newpid" 2>/dev/null || break   # worker died immediately
    sleep 0.1; i=$((i + 1))
  done
  echo "Started in the background (pid $newpid). It continues from where it stopped."
  echo "You can close this window.   Progress: bash $0 status   Stop: bash $0 stop"
}

cmd_stop() {
  local pid
  if ! pid=$(running_pid); then echo "Not running."; rm -f "$LOCK"; exit 0; fi
  touch "$STOPF"
  echo "Asked it to stop. It finishes the current small unit first (one training round, one"
  echo "epoch or one hour of data; a sampling pass redoes only its current file)."
  local t=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 2; t=$((t + 2))
    if [ $t -ge "$STOP_TIMEOUT" ]; then
      echo "Still busy after $STOP_TIMEOUT s; ending it now (safe: all files are written atomically)."
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
      sleep 5
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null
      say "STOPPED (forced after $STOP_TIMEOUT s). To continue: bash ~/run_revision2.sh start"
      break
    fi
  done
  rm -f "$LOCK"
  echo "Stopped. To continue later: bash $0 start"
}

cmd_status() {
  local pid id cur=""
  if pid=$(running_pid); then echo "RUNNING (pid $pid)"; else echo "NOT running  (start / continue with: bash $0 start)"; fi
  [ -f "$STOPF" ] && echo "A stop was requested."
  for id in $STEP_IDS; do
    if is_done "$id"; then echo "  done     step ${id%%_*}  ${id#*_}"
    elif [ -z "$cur" ]; then cur=$id; echo "  CURRENT  step ${id%%_*}  ${id#*_}"
    else echo "  waiting  step ${id%%_*}  ${id#*_}"; fi
  done
  [ -z "$cur" ] && echo "  everything is finished"
  if [ -f "$W/labeled_mv/reprocess_progress.json" ] && ! is_done 2_reprocess; then
    local nd; nd=$(ls "$W"/labeled_mv/*.done.json 2>/dev/null | wc -l)
    python3 - "$W/labeled_mv/reprocess_progress.json" "$nd" <<'PY' 2>/dev/null
import json, sys
d = json.load(open(sys.argv[1])); nd = int(sys.argv[2])
left = (8 - nd) * 24 - d["hours_done"] % 24 if d["hours_done"] < d["hours_total"] else (8 - nd) * 24
print(f"  step 2: {nd}/8 days finished; about {max(left, 0) * d['avg_seconds_per_hour'] / 3600:.1f} h left "
      f"({d['avg_seconds_per_hour']:.0f} s per hour of data)")
PY
  fi
  if [ -f "$CODE/experiments/rev2_plan.py" ]; then
    ( activate >/dev/null 2>&1; plan status 2>/dev/null )
  fi
  echo "Last lines of $STATUS:"
  tail -n 5 "$STATUS" 2>/dev/null | sed 's/^/  /'
  echo "Free disk: WSL disk $(free_gb "$W" 2>/dev/null || echo '?') GB; D: $( [ -d "$P" ] && echo "$(free_gb "$P") GB" || echo 'not connected')"
}

cmd_dry() {
  echo "# work dir $W, code $CODE, raw archive list $TARS_SRC"
  echo "# step 1: unzip $ZIP_IN -> $W/rev2 ; bash $CODE/experiments/smoke_test.sh $W/smoke"
  local d
  for d in $TEST_DAYS $TRAIN_DAYS; do
    echo "python3 -u $CODE/experiments/reprocess_mv.py --tars-list $W/tars.txt --days $d --out-dir $W/labeled_mv --tmp-dir $W/raw_tmp --stop-flag $STOPF --min-hours $MIN_HOURS    # step 2"
  done
  echo "python3 -u $CODE/experiments/make_holdout_sample.py --method hash --frac $TEST_FRAC $(for d in $TEST_DAYS; do printf '%s ' "$W/labeled_mv/opensky_${d}_labeled.csv"; done)--out $W/holdout_sample_mv.csv    # step 3"
  if [ -f "$CODE/experiments/rev2_plan.py" ]; then
    local n; for n in 4 5 6 7 8 9 10; do echo "# step $n"; plan print --step $n; done
  else
    echo "# steps 4-10: listed after step 1 has unpacked the code (or set REV2_CODE=/path/to/rev2)"
  fi
  echo "python3 $CODE/experiments/aggregate_results.py --runs-dirs $W/runs/main --rev2-root $W/runs --tex-dir $W/generated    # step 11"
  echo "# step 11: zip -> $W/results_revision2.zip, copied to $P if D: is connected"
}

case "${1:-help}" in
  start)   cmd_start ;;
  stop)    cmd_stop ;;
  status)  cmd_status ;;
  dry-run) cmd_dry ;;
  _worker) worker ;;
  *) echo "usage: bash $0 start | status | stop | dry-run" ;;
esac
