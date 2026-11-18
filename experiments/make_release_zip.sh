#!/usr/bin/env bash
# Builds a code zip (top-level folder rev2/) from this repository: code and docs only
# (no .git, no __pycache__, no model / pickle files, no PDFs, no data). Optional: used to
# copy the code to another machine (run_revision2.sh REV2_ZIP) and by the self-test
# experiments/test_run_revision2.sh.
# Usage: bash experiments/make_release_zip.sh OUT_ZIP
set -euo pipefail
OUT="${1:?usage: make_release_zip.sh OUT_ZIP}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAGE="$(mktemp -d)"
[ -n "$STAGE" ] && [ -d "$STAGE" ] || { echo "mktemp failed"; exit 1; }
mkdir -p "$STAGE/rev2"
( cd "$REPO" && tar -cf - \
    --exclude='./.git' --exclude='__pycache__' --exclude='*.pyc' --exclude='*.pkl' \
    --exclude='*.pt' --exclude='*.pth' --exclude='*.pdf' --exclude='*.zip' --exclude='*.csv' \
    --exclude='./paper' --exclude='./demo/data' --exclude='./demo/runs' --exclude='./demo/models' \
    --exclude='*:Zone.Identifier' --exclude='*.Zone.Identifier' . ) \
  | ( cd "$STAGE/rev2" && tar -xf - )
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
rm -f "$OUT.tmp"
( cd "$STAGE" && zip -qr "$OUT.tmp" rev2 )
mv "$OUT.tmp" "$OUT"
rm -rf "$STAGE"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1), $(unzip -l "$OUT" | tail -1 | awk '{print $2}') files)"
