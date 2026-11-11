#!/usr/bin/env python3
"""
make_shards.py
-------------------------------------------------------------------
Split the REMAINING revision-2 training runs (steps 5..10, minus the runs that
are already done) into balanced per-machine shard files
(experiments/shards/<name>.txt), weighting each run by its measured run time and
packing the machines into near-equal total time (greedy longest-first).

  python3 experiments/make_shards.py \
      --machines me,user1,user2 \
      --done experiments/shards/DONE.txt \
      [--weights fed_cnn=3480,fed_mlp=3480,central_cnn=180,central_mlp=180,lgbm=90] \
      [--out-dir experiments/shards] [--split-md SPLIT.md]

Rules baked in (overridable with --no-rules):
  * the fast step-6 correction ablation (LightGBM + central CNN) stays on 'me'
    (the user's machine, the slowest GPU; these runs are quick);
  * the heaviest federated chunk, the step-10 federated ablation, prefers the
    helper machines;
  * a run series (same folder+model+feature-set, different seeds) is kept on one
    machine so seeds stay together, except where that would unbalance the split.

Run-keys and the plan come from rev2_plan.py, so this always matches the grid.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev2_plan as P  # noqa: E402

DEFAULT_WEIGHTS = {"fed_cnn": 3480, "fed_mlp": 3480, "fedprox": 3480,
                   "central_cnn": 180, "central_mlp": 180, "lgbm": 90}

# The runs the user's machine has already finished (steps 1-4 and part of step 5):
# central CNN + LightGBM (core & full) all seeds, and federated CNN full seeds 1-3.
DEFAULT_DONE = (
    [f"s5:central_cnn:corrected:{fs}:seed{s}" for fs in ("core", "full") for s in range(1, 6)] +
    [f"s5:lgbm:corrected:{fs}:seed{s}" for fs in ("core", "full") for s in range(1, 6)] +
    [f"s5:fed_cnn:corrected:full:seed{s}" for s in (1, 2, 3)]
)


def model_of(run_id):
    return run_id.split("__", 1)[0]


def weight_of(model, dir_name, weights):
    if dir_name.startswith("fedprox"):
        return weights.get("fedprox", weights.get("fed_cnn", 3480))
    return weights.get(model, 180)


def parse_done(spec_list):
    done = set()
    for spec in spec_list or []:
        if os.path.exists(spec):
            with open(spec) as fh:
                for line in fh:
                    line = line.split("#", 1)[0].strip()
                    if line:
                        done.add(line)
        else:
            done |= {s.strip() for s in spec.split(",") if s.strip()}
    return done


def parse_weights(spec):
    w = dict(DEFAULT_WEIGHTS)
    if spec:
        for item in spec.split(","):
            k, v = item.split("=")
            w[k.strip()] = float(v)
    return w


def build_units(done, weights):
    """Work-units keyed by output file. Each unit carries the run-keys that map
    to it (an output shared by two steps gets both), its step set, model, dir,
    feature set and weight. Done run-keys are removed; a unit all of whose keys
    are done is dropped."""
    rk = P.all_remaining_run_keys(P.env(""))
    units = {}
    for key, (step, d, rid) in rk.items():
        if key in done:
            continue
        u = units.setdefault((d, rid), {"keys": [], "steps": set(), "dir": d,
                                        "run_id": rid, "model": model_of(rid)})
        u["keys"].append(key)
        u["steps"].add(step)
    for u in units.values():
        u["weight"] = weight_of(u["model"], u["dir"], weights)
        u["fset"] = u["run_id"].split("__")[1]
    return units


def unit_flags(u, use_rules):
    u["force_me"] = use_rules and (6 in u["steps"])
    u["prefer_helper"] = use_rules and (u["steps"] == {10})
    return u


def assign(units, machines, use_rules):
    """Greedy longest-first packing at the per-run level (so a heavy federated
    series can split across helpers for a tighter time balance), with the
    me-forced / helper-preferred rules. Light runs (LightGBM, central) are
    grouped so their seeds tend to stay together; heavy federated runs are
    balanced first. Returns {machine: [unit,...]}, load."""
    load = {m: 0.0 for m in machines}
    picked = {m: [] for m in machines}
    helpers = [m for m in machines if m != machines[0]] or machines
    us = [unit_flags(u, use_rules) for u in units.values()]

    def place(u, eligible):
        m = min(eligible, key=lambda x: (load[x], machines.index(x)))
        picked[m].append(u)
        load[m] += u["weight"]

    for u in us:                              # step-6 ablation (and its step-7 twins) -> 'me'
        if u["force_me"]:
            place(u, [machines[0]])
    # Place the helper-restricted (step-10) runs before the flexible ones, so the
    # flexible federated runs are then free to fill the user's machine up to the
    # helpers' level (otherwise they land on the still-empty helpers first and
    # leave 'me' underloaded). Longest-first within each group.
    rest = [u for u in us if not u["force_me"]]
    rest.sort(key=lambda u: (0 if u["prefer_helper"] else 1, -u["weight"]))
    for u in rest:
        eligible = helpers if (u["prefer_helper"] and len(helpers) >= 1) else machines
        place(u, eligible)
    return picked, load


def group_series(picked_units):
    """Group a machine's assigned units into series (dir, model, feature-set)
    for display/writing."""
    series = {}
    for u in picked_units:
        skey = (u["dir"], u["model"], u["fset"])
        s = series.setdefault(skey, {"units": [], "weight": 0.0, "steps": set(),
                                     "dir": u["dir"], "model": u["model"], "fset": u["fset"]})
        s["units"].append(u)
        s["weight"] += u["weight"]
        s["steps"] |= u["steps"]
    return list(series.values())


def hours(load):
    return {m: v / 3600 for m, v in load.items()}


def write_shards(shards, out_dir, machines):
    os.makedirs(out_dir, exist_ok=True)
    for m in machines:
        lines = [f"# shard '{m}' - revision-2 runs assigned to this machine.",
                 "# One run-key per line (step:model:variant:feature-set:seed). Edit freely to",
                 "# rebalance: move lines between me.txt / user1.txt / user2.txt. Keep all the",
                 "# lines of one output together (a '[shared]' key and its twin, same seed).", ""]
        series_sorted = sorted(shards[m], key=lambda s: (min(s["steps"]), s["dir"], s["model"], s["fset"]))
        for s in series_sorted:
            lines.append(f"# step {sorted(s['steps'])} {s['dir']} {s['model']} {s['fset']} "
                         f"({len(s['units'])} run(s), ~{s['weight'] / 60:.0f} min)")
            for u in sorted(s["units"], key=lambda u: u["run_id"]):
                for k in sorted(u["keys"]):
                    shared = "   # shared output (also verified by another step)" if len(u["keys"]) > 1 else ""
                    lines.append(k + shared)
            lines.append("")
        with open(os.path.join(out_dir, m + ".txt"), "w") as fh:
            fh.write("\n".join(lines).rstrip() + "\n")


def write_split_md(path, shards, load, machines, weights):
    h = hours(load)
    lines = ["# Revision-2 run split across machines", "",
             "Generated by `experiments/make_shards.py`. Each machine runs the run-keys in its",
             "shard file (`experiments/shards/<name>.txt`). Move lines between the files to",
             "rebalance, then re-check coverage with the smoke test's sharding check.", "",
             "Measured run times used as weights (seconds): " +
             ", ".join(f"{k}={int(v)}" for k, v in sorted(weights.items())) + ".", "",
             "| machine | runs | est. hours |", "| --- | ---: | ---: |"]
    for m in machines:
        nruns = sum(len(s["units"]) for s in shards[m])
        lines.append(f"| {m} | {nruns} | {h[m]:.1f} |")
    total_runs = sum(len(s["units"]) for m in machines for s in shards[m])
    lines.append(f"| **total** | **{total_runs}** | **{sum(h.values()):.1f}** |")
    lines += ["", "## What each machine runs", ""]
    for m in machines:
        lines.append(f"### {m}  (~{h[m]:.1f} h, {sum(len(s['units']) for s in shards[m])} runs)")
        for s in sorted(shards[m], key=lambda s: (min(s["steps"]), s["dir"], s["model"], s["fset"])):
            lines.append(f"- step {sorted(s['steps'])} `{s['dir']}` {s['model']} {s['fset']}: "
                         f"{len(s['units'])} run(s), ~{s['weight'] / 60:.0f} min")
        lines.append("")
    with open(path, "w") as fh:
        fh.write("\n".join(lines).rstrip() + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machines", default="me,user1,user2",
                    help="comma-separated; the FIRST is the user's machine ('me')")
    ap.add_argument("--done", action="append", default=None,
                    help="file of done run-keys and/or inline comma list; repeatable. "
                         "If omitted, the built-in done set (steps 1-4 and part of step 5) is used.")
    ap.add_argument("--weights", default=None,
                    help="model=seconds,... overrides (e.g. fed_cnn=2600 for a faster helper GPU)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "shards"))
    ap.add_argument("--split-md", default=os.path.join(os.path.dirname(HERE), "SPLIT.md"))
    ap.add_argument("--no-rules", action="store_true",
                    help="pure balance, ignore the me/helper placement rules")
    a = ap.parse_args(argv)

    machines = [m.strip() for m in a.machines.split(",") if m.strip()]
    if len(machines) < 1:
        ap.error("need at least one machine")
    weights = parse_weights(a.weights)
    done = parse_done(a.done) if a.done else set(DEFAULT_DONE)

    units = build_units(done, weights)
    picked, load = assign(units, machines, use_rules=not a.no_rules)
    shards = {m: group_series(picked[m]) for m in machines}
    write_shards(shards, a.out_dir, machines)
    write_split_md(a.split_md, shards, load, machines, weights)

    # coverage self-check: shards partition the remaining run-keys exactly
    assigned = set()
    for m in machines:
        for s in shards[m]:
            for u in s["units"]:
                assigned |= set(u["keys"])
    allrk = set(P.all_remaining_run_keys(P.env(""))) - done
    assert assigned == allrk, ("coverage mismatch: "
                               f"missing {sorted(allrk - assigned)[:3]}, extra {sorted(assigned - allrk)[:3]}")
    h = hours(load)
    print(f"wrote {len(machines)} shard files to {a.out_dir} and {a.split_md}")
    print(f"remaining run-keys: {len(allrk)} (done: {len(done)})")
    for m in machines:
        print(f"  {m}: {sum(len(s['units']) for s in shards[m])} runs, ~{h[m]:.1f} h")
    print(f"  coverage OK: shards partition all {len(allrk)} remaining run-keys with no overlap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
