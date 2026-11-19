#!/usr/bin/env python3
"""Build results/runs_index.csv from the per-run JSON files of the revision-2 grid.

One row per run. Every value is read straight out of a run JSON; nothing is
recomputed and nothing is rounded, with the single exception of
wall_clock_minutes, which is wall_seconds / 60 written to one decimal place.
A cell is left empty when the field does not apply to that run (for example
rounds_cap for a centralized model) or is absent from the JSON.

The run_key and step columns follow experiments/rev2_plan.py: a run is
identified by  s{step}:{model}:{vtag}:{feature_set}:seed{seed}, where vtag is
the variant slug plus '-ls' for a legacy-sampler run and '-mu<x>' for FedProx.
The step is the step of run_revision2.sh that asks for the run:

  5  main grid (LightGBM, central CNN, federated CNN; corrected pipeline)
  6  correction ablation and the legacy-sampler pair (LightGBM, central CNN)
  7  earlier versus corrected pipeline (central CNN, federated CNN)
  8  MLP baselines (central MLP, federated MLP)
  9  FedProx (federated CNN, mu = 0.01 and 0.1)
  10 federated ablation (federated CNN, leave-one-out variants)

The three legacy central-CNN seeds 1 to 3 are asked for by both step 6 and
step 7 and exist as one file each; they are listed here under step 6.

The client-inventory files (clients__seed*.json) describe the client split,
not a training run, and are skipped.

Usage:
    python3 results/make_runs_index.py [MERGED_RUNS_DIR] [-o OUT_CSV]

Standard library only.
"""

import argparse
import csv
import json
import os

DEFAULT_RUNS_DIR = os.path.join(os.path.expanduser("~"), "rev2_work", "merged_runs")

COLUMNS = [
    "run_key", "step", "model", "feature_set", "variant", "protocol", "sampler",
    "fedprox_mu", "seed", "s_star", "macro_f1_at_s_star", "far_at_s_star",
    "attack_recall_at_s_star", "pr_auc", "macro_f1_at_s1", "far_at_s1",
    "attack_recall_at_s1", "n_clients", "rounds_completed", "rounds_cap",
    "device", "wall_clock_minutes", "source_file",
]

MLP_MODELS = ("central_mlp", "fed_mlp")


def num(value):
    """Full-precision text for a number read from the JSON, '' for a missing one."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return str(value)


def variant_slug(variant):
    """'loo:C1' -> 'loo_C1', as experiments/variants.py writes folder names."""
    return (variant or "corrected").replace(":", "_").replace("+", "")


def vtag(record):
    """Variant tag that makes a run-key unique within a step."""
    tag = variant_slug(record.get("variant"))
    if record.get("sampler") == "legacy":
        tag += "-ls"
    mu = record.get("fedprox_mu")
    if mu is not None:
        tag += "-mu%g" % mu
    return tag


def step_of(folder, record):
    """Step of run_revision2.sh that asks for this run."""
    model = record.get("model")
    seed = record.get("seed")
    if folder.startswith("fedprox_"):
        return 9
    if folder == "main":
        return 8 if model in MLP_MODELS else 5
    if folder.startswith("ls_"):
        return 6
    if folder.startswith("var_"):
        if model == "fed_cnn":
            # federated ablation is step 10; the legacy federated CNN is step 7
            return 7 if folder == "var_legacy" else 10
        if folder == "var_legacy" and model == "central_cnn" and seed not in (1, 2, 3):
            # seeds 4 and 5 exist only because step 7 asks for five seeds
            return 7
        return 6
    raise SystemExit("unknown run folder: %s" % folder)


def row_of(record, folder, source_file):
    test = record.get("test_summary") or {}
    at_star = test.get("s_star") or {}
    at_one = test.get("s1") or {}
    training = record.get("training") or {}
    pr_auc = record.get("pr_auc") or {}
    rounds = training.get("per_round_val_acc")
    wall = record.get("wall_seconds")
    step = step_of(folder, record)
    run_key = "s%d:%s:%s:%s:seed%s" % (
        step, record.get("model"), vtag(record), record.get("feature_set"), record.get("seed"))
    return {
        "run_key": run_key,
        "step": step,
        "model": record.get("model") or "",
        "feature_set": record.get("feature_set") or "",
        "variant": record.get("variant") or "",
        "protocol": record.get("protocol") or "",
        "sampler": record.get("sampler") or "",
        "fedprox_mu": num(record.get("fedprox_mu")),
        "seed": num(record.get("seed")),
        "s_star": num(record.get("s_star")),
        "macro_f1_at_s_star": num(at_star.get("macro_f1")),
        "far_at_s_star": num(at_star.get("false_alarm_rate")),
        "attack_recall_at_s_star": num(at_star.get("avg_attack_recall")),
        "pr_auc": num(pr_auc.get("macro")),
        "macro_f1_at_s1": num(at_one.get("macro_f1")),
        "far_at_s1": num(at_one.get("false_alarm_rate")),
        "attack_recall_at_s1": num(at_one.get("avg_attack_recall")),
        "n_clients": num(training.get("n_clients")),
        "rounds_completed": num(len(rounds) if rounds is not None else None),
        "rounds_cap": num(training.get("rounds")),
        "device": record.get("device") or "",
        "wall_clock_minutes": "" if wall is None else "%.1f" % (wall / 60.0),
        "source_file": source_file,
    }


def collect(runs_dir):
    rows = []
    skipped = []
    for folder in sorted(os.listdir(runs_dir)):
        run_folder = os.path.join(runs_dir, folder, "runs")
        if not os.path.isdir(run_folder):
            continue
        for name in sorted(os.listdir(run_folder)):
            if not name.endswith(".json"):
                continue
            source_file = "%s/runs/%s" % (folder, name)
            if name.startswith("clients__"):
                skipped.append(source_file)
                continue
            with open(os.path.join(run_folder, name), encoding="utf-8") as handle:
                record = json.load(handle)
            rows.append(row_of(record, folder, source_file))
    rows.sort(key=lambda r: (r["step"], r["source_file"]))
    return rows, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs_dir", nargs="?", default=DEFAULT_RUNS_DIR,
                        help="merged runs directory (default: %s)" % DEFAULT_RUNS_DIR)
    parser.add_argument("-o", "--out", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "runs_index.csv"), help="output CSV (default: next to this script)")
    args = parser.parse_args()

    if not os.path.isdir(args.runs_dir):
        raise SystemExit("not a directory: %s" % args.runs_dir)
    rows, skipped = collect(args.runs_dir)
    if not rows:
        raise SystemExit("no run JSONs found under %s" % args.runs_dir)
    with open(args.out, "w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print("%s: %d runs (%d client-inventory files skipped)" % (args.out, len(rows), len(skipped)))


if __name__ == "__main__":
    main()
