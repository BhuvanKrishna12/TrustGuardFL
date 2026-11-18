#!/usr/bin/env bash
# make_share.sh - assemble the ~2 GB helper share on the main machine (optional
# multi-machine mode, see docs/MULTI_MACHINE.md).
#
#   bash make_share.sh OUT_DIR
#
# The share has everything a helper machine needs to run steps 5-10 WITHOUT the
# 120 GB labelled files: the code, the cache (stats + training samples + input
# manifest), the holdout CSV and the shard files. Copy the whole OUT_DIR to the
# helper machine, which copies its contents into its own work dir.
# Idempotent: re-running refreshes OUT_DIR in place.
#   REV2_W   work dir of the main machine (default ~/rev2_work, as in run_revision2.sh)
#   REV2_CODE code folder to copy (default: this repository)
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
W="${REV2_W:-$HOME/rev2_work}"
CODE="${REV2_CODE:-$REPO}"
[ -d "$CODE/experiments" ] || CODE="$REPO"         # fall back to this repo
OUT="${1:?usage: bash make_share.sh OUT_DIR}"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
[ -n "$OUT" ] && [ "$OUT" != "/" ] || { echo "refusing OUT='$OUT'"; exit 2; }
TRAIN_DAYS="${REV2_TRAIN_DAYS:-2019-01-07 2019-04-01 2019-06-24 2019-09-30}"

echo "building the helper share in $OUT"
echo "  code:    $CODE"
echo "  workdir: $W"

# 0. sanity: the cache must be complete, or a helper will fail at run time.
if [ ! -f "$W/cache/input_manifest.json" ]; then
  echo "NOTE: $W/cache/input_manifest.json not found; writing it now from the labelled files"
  files=(); for d in $TRAIN_DAYS; do
    files+=("$W/labeled_mv/opensky_${d}_labeled.csv"); done
  python3 "$CODE/experiments/make_input_manifest.py" --train-files "${files[@]}" \
      --holdout "$W/holdout_sample_mv.csv" --cache-dir "$W/cache" --allow-missing || true
fi
if ! python3 "$CODE/experiments/verify_cache.py" --work "$W"; then
  echo "ERROR: the cache is not complete (see above). Finish step 4 of run_revision2.sh on"
  echo "       this machine (the one with the labelled files) before making the share."
  exit 1
fi

mkdir -p "$OUT" 2>/dev/null
# the destination must really be writable (an unmounted drive fails here)
if ! ( touch "$OUT/.w_test" 2>/dev/null && rm -f "$OUT/.w_test" ); then
  echo "ERROR: cannot write to $OUT (is the drive that holds it connected and mounted?)."
  echo "       Nothing was copied. Tip: build it on a local disk instead, e.g.  bash make_share.sh ~/rev2_share"
  exit 3
fi

# 1. code -> OUT/rev2 (same exclusions as the release zip; keeps experiments/shards/*.txt)
echo "  copying code -> $OUT/rev2"
rm -rf "$OUT/rev2.tmp"; mkdir -p "$OUT/rev2.tmp"
( cd "$CODE" && tar -cf - \
    --exclude='./.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pkl' \
    --exclude='*.pt' --exclude='*.pth' --exclude='*.pdf' --exclude='*.zip' --exclude='*.csv' \
    --exclude='./paper' --exclude='./demo/data' --exclude='./demo/runs' --exclude='./demo/models' \
    --exclude='*:Zone.Identifier' --exclude='*.Zone.Identifier' . ) \
  | ( cd "$OUT/rev2.tmp" && tar -xf - )
rm -rf "$OUT/rev2.old"; [ -d "$OUT/rev2" ] && mv "$OUT/rev2" "$OUT/rev2.old"
mv "$OUT/rev2.tmp" "$OUT/rev2" && rm -rf "$OUT/rev2.old"

# 2. cache -> OUT/cache  (stats, training samples, input manifest; holdout npz optional)
echo "  copying cache -> $OUT/cache"
mkdir -p "$OUT/cache"
if command -v rsync >/dev/null; then
  rsync -a --delete "$W/cache/" "$OUT/cache/"
else
  rm -rf "$OUT/cache"/*; cp -a "$W/cache/." "$OUT/cache/"
fi

# 3. holdout CSV (small; carries the variant columns the test arrays need)
cp -f "$W/holdout_sample_mv.csv" "$OUT/holdout_sample_mv.csv"
[ -f "$W/holdout_sample_mv.csv.done.json" ] && cp -f "$W/holdout_sample_mv.csv.done.json" "$OUT/" || true

# 4. a default MY_SHARD: shard id 'user1' (machine B); machine C overwrites it with 'user2'.
#    The shard ids name the files in experiments/shards/, not people.
[ -f "$OUT/MY_SHARD" ] || echo user1 > "$OUT/MY_SHARD"

# verify what was written
for f in rev2/run_revision2.sh cache/input_manifest.json holdout_sample_mv.csv MY_SHARD; do
  [ -s "$OUT/$f" ] || { echo "ERROR: $OUT/$f missing or empty; share is NOT complete"; exit 4; }
done
[ "$(du -sk "$W/holdout_sample_mv.csv" | cut -f1)" = "$(du -sk "$OUT/holdout_sample_mv.csv" | cut -f1)" ] || \
  { echo "ERROR: holdout copy size differs; share is NOT complete"; exit 4; }

SIZE="$(du -sh "$OUT" 2>/dev/null | cut -f1)"
echo "DONE. Share is $SIZE at $OUT"
echo "  it contains: rev2/ (code), cache/, holdout_sample_mv.csv, MY_SHARD"
echo "  copy the whole folder to each helper machine (see rev2/docs/MULTI_MACHINE.md)."
