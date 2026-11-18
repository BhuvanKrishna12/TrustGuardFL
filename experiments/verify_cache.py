#!/usr/bin/env python3
"""
verify_cache.py
-------------------------------------------------------------------
Check that a work dir has everything a cache-only (helper) machine needs to run
revision-2 steps 5..10 WITHOUT the 120 GB labelled training files: the input
manifest, the per-file stats caches, the uniform union training sample, the two
legacy-sampler samples and the holdout CSV. Prints what is missing and which
machine should have produced it; exit 0 when the cache is complete.

  python3 experiments/verify_cache.py --work $WORK_DIR
  python3 experiments/verify_cache.py --cache-dir C --holdout H --train-files F...
"""
import argparse
import glob
import hashlib
import json
import os
import sys

TRAIN_DAYS = os.environ.get("REV2_TRAIN_DAYS",
                            "2019-01-07 2019-04-01 2019-06-24 2019-09-30").split()
SAMPLE_SEED = 42


def file_key(parts, *extra):
    blob = json.dumps([parts, list(extra)], sort_keys=True)
    return hashlib.sha1(blob.encode()).hexdigest()[:12]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", default=None, help="work dir (sets cache-dir, holdout, train-files)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--holdout", default=None)
    ap.add_argument("--train-files", nargs="+", default=None)
    ap.add_argument("--max-total-rows", type=int,
                    default=int(os.environ.get("REV2_MAX_ROWS", "10000000")))
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.work:
        a.cache_dir = a.cache_dir or os.path.join(a.work, "cache")
        a.holdout = a.holdout or os.path.join(a.work, "holdout_sample_mv.csv")
        if not a.train_files:
            a.train_files = [os.path.join(a.work, "labeled_mv", f"opensky_{d}_labeled.csv")
                             for d in TRAIN_DAYS]
    if not a.cache_dir or not a.train_files:
        ap.error("need --work, or --cache-dir and --train-files")

    problems = []
    cache = a.cache_dir
    mpath = os.path.join(cache, "input_manifest.json")
    manifest = {}
    if not os.path.exists(mpath):
        problems.append(f"input manifest missing: {mpath}  (build it on the machine with the "
                        f"labelled files: python3 experiments/make_input_manifest.py ...)")
    else:
        doc = json.load(open(mpath))
        files = doc.get("files", doc)
        manifest = {os.path.basename(k): int(v["size"] if isinstance(v, dict) else v)
                    for k, v in files.items()}

    def size_of(p):
        bn = os.path.basename(p)
        if os.path.exists(p):
            return os.path.getsize(p)
        return manifest.get(bn)

    # per-file stats caches
    sizes = []
    for p in a.train_files:
        bn = os.path.basename(p)
        sz = size_of(p)
        if sz is None:
            problems.append(f"no size for {bn} (absent and not in manifest)")
            continue
        sizes.append((bn, sz))
        for suffix in (".json", ".icao24.txt.gz"):
            fp = os.path.join(cache, f"filestats_{bn}.{sz}{suffix}")
            if not os.path.exists(fp):
                problems.append(f"stats cache missing: {os.path.basename(fp)}  (step 4 on the "
                                f"machine with the labelled files)")

    # training-sample caches: uniform union + the two legacy-sampler designs
    if len(sizes) == len(a.train_files):
        for tag in ("uniform", "legacy_CN", "legacy_LF"):
            key = file_key(sizes, a.max_total_rows, SAMPLE_SEED, tag, "r2")
            for ext in (".json", ".pkl"):
                fp = os.path.join(cache, f"train_sample_{tag}_{key}{ext}")
                if not os.path.exists(fp):
                    # fall back to a tag glob so a different max-total-rows is reported helpfully
                    alt = glob.glob(os.path.join(cache, f"train_sample_{tag}_*{ext}"))
                    if not alt:
                        problems.append(f"training-sample cache missing: train_sample_{tag}_*{ext} "
                                        f"(step 4 'sampling passes' on the machine with the files)")
                    elif not a.quiet:
                        print(f"note: {os.path.basename(fp)} not found but {len(alt)} other "
                              f"train_sample_{tag}_* present (different --max-total-rows?)")

    if not os.path.exists(a.holdout or ""):
        problems.append(f"holdout CSV missing: {a.holdout}  (copy it into the work dir; it is part "
                        f"of the share)")

    if problems:
        print("CACHE INCOMPLETE for cache-only runs:")
        for p in problems:
            print("  - " + p)
        return 1
    if not a.quiet:
        print(f"CACHE OK: manifest, {len(sizes)} stats caches, uniform + legacy_CN + legacy_LF "
              f"training samples and the holdout CSV are all present in {cache}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
