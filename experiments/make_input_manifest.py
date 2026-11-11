#!/usr/bin/env python3
"""
make_input_manifest.py
-------------------------------------------------------------------
Write cache/input_manifest.json, recording {basename: size} for every labelled
training file and for the holdout CSV. Run this on the machine that HAS the big
labelled files; a machine that only has the ~2 GB cache then reads these sizes
(instead of os.path.getsize on the absent files) so the cache keys match and the
cached stats / training sample / holdout artifacts are all found.

  python3 experiments/make_input_manifest.py \
      --train-files .../opensky_2019-01-07_labeled.csv ... \
      --holdout .../holdout_sample_mv.csv \
      --cache-dir .../cache

run_revision2.sh step 4 calls this automatically, so a normal run always leaves a
manifest behind for the helpers.
"""
import argparse
import json
import os
import sys


def build_manifest(train_files, holdout):
    files = {}
    missing = []
    for p in list(train_files or []) + ([holdout] if holdout else []):
        if not os.path.exists(p):
            missing.append(p)
            continue
        files[os.path.basename(p)] = {"size": os.path.getsize(p)}
    return files, missing


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-files", nargs="+", required=True)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--cache-dir", required=True)
    ap.add_argument("--allow-missing", action="store_true",
                    help="do not fail if some inputs are absent (skip them)")
    a = ap.parse_args(argv)

    files, missing = build_manifest(a.train_files, a.holdout)
    if missing and not a.allow_missing:
        sys.exit("ERROR: these inputs are missing (run this where the labelled files are, "
                 "or pass --allow-missing):\n  " + "\n  ".join(missing))
    os.makedirs(a.cache_dir, exist_ok=True)
    out = os.path.join(a.cache_dir, "input_manifest.json")
    doc = {"version": 1,
           "train_files": [os.path.basename(p) for p in a.train_files],
           "holdout": os.path.basename(a.holdout) if a.holdout else None,
           "files": files}
    tmp = out + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, out)
    print(f"wrote {out} ({len(files)} file(s))")
    for bn, meta in sorted(files.items()):
        print(f"  {bn}: {meta['size']:,} bytes")
    if missing:
        print(f"  (skipped {len(missing)} missing input(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
