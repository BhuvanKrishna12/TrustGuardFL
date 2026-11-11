#!/usr/bin/env bash
# Cache-only multi-machine simulation on synthetic data (CPU, a few minutes):
#   * a "user" machine builds the caches (step 4) + input manifest and runs its
#     own shard ('me') of steps 5-10;
#   * make_share.sh assembles the ~share;
#   * a "helper" machine copies the share, has NO labelled files, sets MY_SHARD,
#     and runs run_revision2.sh (which auto-detects cache-only mode, skips steps
#     1-4, and runs the 'user1' shard straight from the cache);
#   * merge_runs.py unions both machines' runs and aggregate_results.py builds
#     the tables.
# Usage: bash experiments/test_multi_machine.sh WORK_DIR
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
T="${1:?usage: test_multi_machine.sh WORK_DIR}"
case "$T" in /*) ;; *) T="$PWD/$T" ;; esac
[ "$T" != "/" ] || exit 2
rm -rf "$T"; mkdir -p "$T"

TRAIN_DAYS="2019-01-07 2019-04-01"
TEST_DAY="2019-12-09"
U="$T/user/rev2_work"         # the user's work dir
mkdir -p "$U/labeled_mv" "$U/cache"

echo "== build synthetic multi-variant inputs"
python3 - "$U" $TRAIN_DAYS "$TEST_DAY" <<'PY'
import contextlib, io, os, sys
sys.path.insert(0, os.path.join(os.getcwd(), "experiments"))
sys.path.insert(0, os.getcwd())
import make_synthetic_raw as msr
import opensky_fl_ids_pipeline as pl
U = sys.argv[1]; train = sys.argv[2:4]; test = sys.argv[4]
for d in train + [test]:
    raw = msr.write_day(os.path.join(U, "raw"), d, hours=3, n_aircraft=150, seed=5)
    out = (os.path.join(U, "labeled_mv", f"opensky_{d}_labeled.csv") if d in train
           else os.path.join(U, "holdout_sample_mv.csv"))
    with contextlib.redirect_stdout(io.StringIO()):
        pl.main(raw, chunk_files=1, out_path=out, pipeline_version="mv")
print("  wrote", [f for f in os.listdir(os.path.join(U, "labeled_mv"))], "and holdout_sample_mv.csv")
PY
# a .done.json so integrity checks that look for it are satisfied
python3 - "$U/holdout_sample_mv.csv" <<'PY'
import json, os, sys
p = sys.argv[1]; json.dump({"bytes": os.path.getsize(p), "rows": 1}, open(p + ".done.json", "w"))
PY

# sim env shared by both machines
export REV2_SIM=1 REV2_DEVICE=cpu REV2_SKIP_SMOKE=1 REV2_NEED_GB=0
export REV2_TRAIN_DAYS="$TRAIN_DAYS" REV2_TEST_DAYS="$TEST_DAY"
export REV2_MAX_ROWS=60000 REV2_MIN_CLIENT_ROWS=800
export REV2_SHARD_DIR="$T/shards"
PLAN() { REV2_W="$1" python3 "$REPO/experiments/rev2_plan.py" "${@:2}"; }

echo "== user: step 4 (build caches) + manifest"
REV2_W="$U" python3 "$REPO/experiments/rev2_plan.py" run --step 4 >/dev/null
files=(); for d in $TRAIN_DAYS; do files+=("$U/labeled_mv/opensky_${d}_labeled.csv"); done
python3 "$REPO/experiments/make_input_manifest.py" --train-files "${files[@]}" \
    --holdout "$U/holdout_sample_mv.csv" --cache-dir "$U/cache" >/dev/null

echo "== generate 2-machine sim shards (me, user1)"
mkdir -p "$REV2_SHARD_DIR"
REV2_W="$U" python3 "$REPO/experiments/make_shards.py" --machines me,user1 \
    --done "" --out-dir "$REV2_SHARD_DIR" --split-md "$T/SPLIT_sim.md" | sed 's/^/  /'
# make_shards.py self-checks that the shards are a disjoint exact cover; the committed
# real shards are validated separately by experiments/test_shards.py.

echo "== user: run the 'me' shard of steps 5-10 (sim, quick)"
for n in 5 6 7 8 9 10; do
  REV2_W="$U" REV2_SHARD=me python3 "$REPO/experiments/rev2_plan.py" run --step "$n" --shard me \
      >"$T/user_step$n.log" 2>&1 || { echo "user step $n failed"; tail -15 "$T/user_step$n.log"; exit 1; }
done

echo "== make_share.sh -> $T/share  (then delete the labelled files from the helper's copy)"
REV2_W="$U" REV2_CODE="$REPO" bash "$REPO/make_share.sh" "$T/share" >"$T/make_share.log" 2>&1 \
  || { echo "make_share failed"; cat "$T/make_share.log"; exit 1; }
grep -q "DONE. Share is" "$T/make_share.log" || { cat "$T/make_share.log"; exit 1; }

# helper machine: copy the share into ~helper/rev2_work; NO labelled files
H="$T/helper/rev2_work"
mkdir -p "$H"
cp -a "$T/share/rev2" "$H/rev2"
cp -a "$T/share/cache" "$H/cache"
cp -a "$T/share/holdout_sample_mv.csv" "$H/holdout_sample_mv.csv"
[ -f "$T/share/holdout_sample_mv.csv.done.json" ] && cp -a "$T/share/holdout_sample_mv.csv.done.json" "$H/" || true
echo user1 > "$H/MY_SHARD"
[ ! -e "$H/labeled_mv" ] || { echo "helper must not have labelled files"; exit 1; }

echo "== helper: run run_revision2.sh (cache-only, shard user1)"
export REV2_VENV="$T/no-venv"
REV2_W="$H" REV2_CODE="$H/rev2" REV2_SHARD_DIR="$REV2_SHARD_DIR" \
  bash "$H/rev2/run_revision2.sh" start >/dev/null 2>&1

# wait for the helper worker to finish
is_running() { [[ "$(REV2_W="$H" REV2_CODE="$H/rev2" bash "$H/rev2/run_revision2.sh" status 2>/dev/null)" == RUNNING* ]]; }
n=0; while is_running; do sleep 2; n=$((n+2)); [ $n -lt 600 ] || { echo "helper timeout"; tail -20 "$H/STATUS.txt"; exit 1; }; done

grep -q "cache-only" "$H/STATUS.txt" || { echo "helper did not enter cache-only mode"; tail -20 "$H/STATUS.txt"; exit 1; }
# the helper must NOT have run steps 1-4 as real work (no labelled files were read)
[ ! -e "$H/labeled_mv" ] || { echo "helper created labeled_mv (should not)"; exit 1; }
# the helper produced at least one run from the cache
helper_runs=$(find "$H/runs" -path '*/runs/*__*__seed*.json' 2>/dev/null | grep -v clients | wc -l)
[ "$helper_runs" -ge 1 ] || { echo "helper produced no runs"; tail -20 "$H/STATUS.txt"; exit 1; }
echo "  helper produced $helper_runs run(s) from the cache (no labelled files present)"

echo "== merge user + helper, then aggregate"
python3 "$REPO/experiments/merge_runs.py" --inputs "$U/runs" "$H/runs" --out "$T/merged" \
  | sed 's/^/  /'
user_runs=$(find "$U/runs" -path '*/runs/*__*__seed*.json' | grep -v clients | wc -l)
merged_runs=$(find "$T/merged" -path '*/runs/*__*__seed*.json' | grep -v clients | wc -l)
echo "  user runs: $user_runs, helper runs: $helper_runs, merged run files: $merged_runs"
mkdir -p "$T/generated"
python3 "$REPO/experiments/aggregate_results.py" --runs-dirs "$T/merged/main" \
    --rev2-root "$T/merged" --tex-dir "$T/generated" >"$T/aggregate.log" 2>&1 \
  || { echo "aggregate failed"; tail -20 "$T/aggregate.log"; exit 1; }
[ -s "$T/generated/results_macros.tex" ] || { echo "no macros produced"; exit 1; }

echo "MULTI-MACHINE SIM OK: helper ran $helper_runs cache-only run(s), merged $merged_runs total, tables built in $T/generated"
