#!/usr/bin/env python3
"""
merge_runs.py
-------------------------------------------------------------------
Merge the per-run results produced by several machines into one runs tree that
aggregate_results.py can turn into the final tables. Each input is a machine's
`runs/` folder (as in ~/rev2_work/runs): subfolders main/, var_*/, ls_*/,
fedprox_*/ each with a runs/ folder of <model>__<fset>__seed<n>.json files.

  python3 experiments/merge_runs.py --inputs runsA runsB runsC --out merged_runs
  python3 experiments/aggregate_results.py --runs-dirs merged_runs/main \
      --rev2-root merged_runs --tex-dir generated

The union is taken per output file (folder + filename). A file present in two
inputs is a DUPLICATE: identical copies are fine (one is kept); if the metrics
differ it is a CONFLICT (both are reported and the first is kept unless
--prefer is given). Disjoint shards never collide, so any conflict means two
machines ran the same run - worth a look, usually a stale copy.
"""
import argparse
import glob
import json
import os
import shutil
import sys


def key_metrics(path):
    try:
        r = json.load(open(path))
        return (r.get("s_star"), r.get("model"), r.get("variant"),
                round(float(r["metrics"]["1.00"]["pooled"]["macro_f1"]), 6))
    except Exception:
        return None


def iter_run_files(root):
    """Yield (reldir, filename, fullpath) for every <dir>/runs/* file."""
    for p in glob.glob(os.path.join(root, "*", "runs", "*")):
        if not os.path.isfile(p):
            continue
        d = os.path.basename(os.path.dirname(os.path.dirname(p)))
        yield d, os.path.basename(p), p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--inputs", nargs="+", required=True, help="per-machine runs/ folders")
    ap.add_argument("--out", required=True, help="merged runs/ folder to create")
    ap.add_argument("--prefer", default=None,
                    help="on a conflict, keep the file from this input folder")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)

    for r in a.inputs:
        if not os.path.isdir(r):
            sys.exit(f"input not found: {r}")
    os.makedirs(a.out, exist_ok=True) if not a.dry_run else None

    chosen = {}          # (dir, file) -> (src_root, path)
    copied = conflicts = dups_identical = 0
    conflict_list = []
    for root in a.inputs:
        for d, fn, path in iter_run_files(root):
            k = (d, fn)
            if k not in chosen:
                chosen[k] = (root, path)
                continue
            # duplicate: compare
            if fn.endswith(".json"):
                m_old = key_metrics(chosen[k][1])
                m_new = key_metrics(path)
                if m_old == m_new:
                    dups_identical += 1
                    continue
                conflicts += 1
                conflict_list.append((d, fn, chosen[k][0], root))
                if a.prefer and os.path.abspath(root) == os.path.abspath(a.prefer):
                    chosen[k] = (root, path)
            else:
                dups_identical += 1   # non-json siblings (clients, labels): keep the first

    for (d, fn), (root, path) in sorted(chosen.items()):
        dest_dir = os.path.join(a.out, d, "runs")
        if not a.dry_run:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(path, os.path.join(dest_dir, fn))
        copied += 1

    n_runs = sum(1 for (d, fn) in chosen if fn.endswith(".json") and "__seed" in fn and "clients" not in fn)
    print(f"merged {len(a.inputs)} input(s) -> {a.out}")
    print(f"  {copied} files written ({n_runs} run JSONs)")
    print(f"  {dups_identical} identical duplicate(s) skipped")
    if conflicts:
        print(f"  WARNING: {conflicts} CONFLICT(s) (same run, different metrics):")
        for d, fn, r1, r2 in conflict_list[:20]:
            print(f"    {d}/runs/{fn}: {r1}  vs  {r2}")
        print("  kept the first input's copy unless --prefer was given; investigate before "
              "trusting the tables.")
    else:
        print("  no conflicts: the inputs are disjoint (as the shards intend)")
    return 1 if conflicts and not a.prefer else 0


if __name__ == "__main__":
    sys.exit(main())
