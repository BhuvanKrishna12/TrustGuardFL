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
      [--out-dir experiments/shards] [--split-md split_summary.md]

Rules baked in (overridable with --no-rules):
  * the fast step-6 correction ablation (LightGBM + central CNN) stays on 'me'
    (shard id of the main machine, the one with the labelled files; these runs are quick);
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

# The runs the main machine had already finished (steps 1-4 and part of step 5):
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


def is_fed(model):
    return model.startswith("fed_")


def assign(units, machines, roles):
    """Assign units to machines, honouring hardware capabilities.

    roles = {cpu_only: set(), speed: {m: factor}, cpu_central_weight: s,
             nonfed_to: machine-or-None}. A cpu_only machine (integrated GPU,
             no CUDA) gets ONLY non-federated runs; its central runs are costed
             at cpu_central_weight. Federated runs are split across the
             GPU machines by SPEED-weighted load (a faster GPU, higher speed,
             takes proportionally more runs). Non-federated runs go to nonfed_to
             if given (the CPU-only machine), else to the least-loaded machine.
    Returns {machine: [unit,...]}, effective load (wall-clock-ish seconds)."""
    cpu_only = set(roles.get("cpu_only", set())) & set(machines)
    speed = {m: float(roles.get("speed", {}).get(m, 1.0)) for m in machines}
    cpu_cw = roles.get("cpu_central_weight", 600.0)
    nonfed_to = roles.get("nonfed_to") if roles.get("nonfed_to") in machines else None
    gpu_machines = [m for m in machines if m not in cpu_only]

    load = {m: 0.0 for m in machines}
    picked = {m: [] for m in machines}

    def eff(u, m):
        """Wall-clock-ish cost of unit u on machine m."""
        w = u["weight"]
        if m in cpu_only and u["model"] in ("central_cnn", "central_mlp"):
            w = cpu_cw                 # central net on CPU is much slower than on a GPU
        return w / speed[m]

    def place(u, eligible):
        m = min(eligible, key=lambda x: (load[x] + eff(u, x), machines.index(x)))
        picked[m].append(u)
        u["placed_eff"] = eff(u, m)       # wall-clock-ish cost on its chosen machine
        load[m] += u["placed_eff"]

    us = list(units.values())
    nonfed = sorted([u for u in us if not is_fed(u["model"])], key=lambda u: -u["weight"])
    fed = sorted([u for u in us if is_fed(u["model"])], key=lambda u: -u["weight"])

    # Non-federated (fast, CPU-friendly): to the CPU-only machine when one is
    # named, else spread over everyone. Seeds stay together because they all land
    # on the same machine.
    for u in nonfed:
        eligible = [nonfed_to] if nonfed_to else machines
        place(u, eligible)
    # Federated (hours each): GPU machines only, split by speed-weighted load.
    if not gpu_machines:
        raise SystemExit("no GPU machine available for the federated runs")
    for u in fed:
        place(u, gpu_machines)
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
        s["weight"] += u.get("placed_eff", u["weight"])     # effective (wall-clock-ish) cost
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


def write_split_md(path, shards, load, machines, weights, roles=None):
    roles = roles or {}
    h = hours(load)
    cpu_only = sorted(set(roles.get("cpu_only", set())) & set(machines))
    speed = roles.get("speed", {})
    note = []
    if cpu_only:
        note.append(f"**{', '.join(cpu_only)}** has an integrated GPU (no CUDA): it runs CPU-only "
                    f"and is assigned ZERO federated runs (fed_cnn/fed_mlp/FedProx take hours on "
                    f"CPU). It gets the fast correction-ablation and other non-federated runs "
                    f"(LightGBM ~1.5 min, central CNN ~{int(roles.get('cpu_central_weight', 600)) // 60} "
                    f"min on CPU), which fit inside the GPU machines' window as free parallel work.")
    if speed:
        note.append("Federated runs are split across the GPU machines by speed ("
                    + ", ".join(f"{k}×{v:g}" for k, v in speed.items())
                    + "); the faster GPU takes proportionally more, so the hours below are "
                      "wall-clock-ish per machine rather than equal run counts.")
    lines = ["# Revision-2 run split across machines", "",
             "Generated by `experiments/make_shards.py`. Each machine runs the run-keys in its",
             "shard file (`experiments/shards/<name>.txt`). Move lines between the files to",
             "rebalance, then re-check coverage with the smoke test's sharding check.", ""]
    for n in note:
        lines += [n, ""]
    lines += ["Base run-time weights (seconds, main machine's GPU): " +
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
                    help="comma-separated shard ids; the FIRST is the main machine ('me')")
    ap.add_argument("--done", action="append", default=None,
                    help="file of done run-keys and/or inline comma list; repeatable. "
                         "If omitted, the built-in done set (steps 1-4 and part of step 5) is used.")
    ap.add_argument("--weights", default=None,
                    help="model=seconds,... overrides (e.g. fed_cnn=2600 for a faster helper GPU)")
    ap.add_argument("--out-dir", default=os.path.join(HERE, "shards"))
    ap.add_argument("--split-md", default=None,
                    help="optional Markdown summary of the split (hours and runs per machine)")
    ap.add_argument("--cpu-only", default="user2",
                    help="comma-separated machines with no CUDA GPU: they get ONLY "
                         "non-federated runs. Default 'user2'. Pass '' to disable.")
    ap.add_argument("--nonfed-to", default="user2",
                    help="send all non-federated (fast, CPU-friendly) runs to this machine "
                         "(default 'user2', the integrated-GPU machine). Pass '' to spread them.")
    ap.add_argument("--speed", default="user1=1.3",
                    help="machine=factor,... GPU speed for splitting federated runs; a faster "
                         "GPU (higher factor) takes proportionally more (default user1=1.3 for a faster GPU)")
    ap.add_argument("--cpu-central-weight", type=float, default=600.0,
                    help="seconds for a central CNN/MLP run on a CPU-only machine (~10 min)")
    ap.add_argument("--no-rules", action="store_true",
                    help="pure balance, ignore the hardware placement rules")
    a = ap.parse_args(argv)

    machines = [m.strip() for m in a.machines.split(",") if m.strip()]
    if len(machines) < 1:
        ap.error("need at least one machine")
    weights = parse_weights(a.weights)
    done = parse_done(a.done) if a.done else set(DEFAULT_DONE)

    if a.no_rules:
        roles = {"cpu_only": set(), "speed": {}, "nonfed_to": None,
                 "cpu_central_weight": a.cpu_central_weight}
    else:
        speed = {}
        for item in (a.speed or "").split(","):
            if "=" in item:
                k, v = item.split("="); speed[k.strip()] = float(v)
        roles = {"cpu_only": {m.strip() for m in a.cpu_only.split(",") if m.strip()},
                 "speed": speed, "nonfed_to": (a.nonfed_to or None),
                 "cpu_central_weight": a.cpu_central_weight}

    units = build_units(done, weights)
    picked, load = assign(units, machines, roles)
    shards = {m: group_series(picked[m]) for m in machines}
    write_shards(shards, a.out_dir, machines)
    if a.split_md:
        write_split_md(a.split_md, shards, load, machines, weights, roles)

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
    print(f"wrote {len(machines)} shard files to {a.out_dir}" + (f" and {a.split_md}" if a.split_md else ""))
    print(f"remaining run-keys: {len(allrk)} (done: {len(done)})")
    for m in machines:
        print(f"  {m}: {sum(len(s['units']) for s in shards[m])} runs, ~{h[m]:.1f} h")
    print(f"  coverage OK: shards partition all {len(allrk)} remaining run-keys with no overlap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
