#!/usr/bin/env bash
# Builds fl-ids-revision2.zip (top-level folder rev2/) from this repository: code and docs
# only (no .git, no __pycache__, no model / pickle files, no PDFs).
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
    --exclude='./paper/main.fls' --exclude='./paper/main.fdb_latexmk' --exclude='*:Zone.Identifier' . ) \
  | ( cd "$STAGE/rev2" && tar -xf - )
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac
rm -f "$OUT.tmp"
( cd "$STAGE" && zip -qr "$OUT.tmp" rev2 )
mv "$OUT.tmp" "$OUT"
rm -rf "$STAGE"
echo "wrote $OUT ($(du -h "$OUT" | cut -f1), $(unzip -l "$OUT" | tail -1 | awk '{print $2}') files)"
