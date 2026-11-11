"""
rev2_plan.py
-------------------------------------------------------------------
The run grid of revision 2 (single source of truth for run_revision2.sh).

  python3 experiments/rev2_plan.py print  --step N   commands of a step (dry run)
  python3 experiments/rev2_plan.py run    --step N   run them one after another
                                                     (resumable; exit 75 = stopped)
  python3 experiments/rev2_plan.py verify --step N   exit 0 if every expected output
                                                     exists and parses
  python3 experiments/rev2_plan.py status            runs done / total and ETA per step
  python3 experiments/rev2_plan.py table             runs per step and model, hours

Paths come from the environment (set by run_revision2.sh): REV2_W (work dir).
REV2_SIM=1 shrinks everything for the simulated pass on synthetic data.

Steps 5 to 10 (steps 1 to 3 and 11 live in run_revision2.sh):
  4  one-off passes: file stats, uniform sample (all variants at once), legacy-loader
     samples for the corrected and the legacy view
  5  main grid, protocol v2, corrected: LightGBM + central CNN first, federated CNN last
  6  correction ablation: LightGBM (5 seeds) + central CNN (3 seeds) on legacy,
     loo:C1/C2/C3/C5, add:C1/C2/C3/C5, the reconstructed 'prefloor' extra, and
     corrected / legacy with the legacy sampler
  7  unified earlier-vs-corrected: legacy, all three models, 5 seeds
  8  MLP baselines (central + FedAvg), 5 seeds
  9  FedProx mu = 0.01 and 0.1, 3 seeds each
  10 federated ablation: fed CNN, 3 seeds, loo:C1/C2/C3/C5
"""

import argparse
import glob
import json
import os
import shlex
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import variants as V  # noqa: E402

SHARD_DIR = os.environ.get("REV2_SHARD_DIR", os.path.join(HERE, "shards"))

TRAIN_DAYS = ["2019-01-07", "2019-04-01", "2019-06-24", "2019-09-30"]
TEST_DAYS = ["2019-12-09", "2019-12-16", "2019-12-23", "2019-12-30"]
ABL = ["C1", "C2", "C3", "C5"]            # the corrections that differ between v2 and v3
ABL_VARIANTS = ["legacy"] + [f"loo:{c}" for c in ABL] + [f"add:{c}" for c in ABL]
EXTRA_VARIANTS = ["prefloor"]              # reconstructed pre-floor rule; flagged in the tables
DEFAULT_SECONDS = {"lgbm": 60, "central_cnn": 156, "central_mlp": 120, "fed_cnn": 2400, "fed_mlp": 2100}
STEP_NAMES = {4: "sampling passes", 5: "main grid", 6: "correction ablation (LightGBM, central CNN)",
              7: "unified earlier vs corrected", 8: "MLP baselines", 9: "FedProx",
              10: "federated ablation"}
EXIT_STOPPED = 75


def env(shard=None):
    W = os.environ.get("REV2_W", os.path.expanduser("~/rev2_work"))
    sim = os.environ.get("REV2_SIM", "") == "1"
    name = resolve_shard_name(shard)
    return {"W": W, "sim": sim,
            "labeled": os.path.join(W, "labeled_mv"), "holdout": os.path.join(W, "holdout_sample_mv.csv"),
            "runs": os.path.join(W, "runs"), "cache": os.path.join(W, "cache"),
            "stop": os.path.join(W, "STOP"),
            "train_days": os.environ.get("REV2_TRAIN_DAYS", " ".join(TRAIN_DAYS)).split(),
            "test_days": os.environ.get("REV2_TEST_DAYS", " ".join(TEST_DAYS)).split(),
            "device": os.environ.get("REV2_DEVICE", "auto"),
            "trust_cache": os.environ.get("REV2_TRUST_CACHE", "") == "1",
            "shard": name, "shard_keys": load_shard_keys(name)}


# ---------------------------------------------------------------------------
# run-keys and sharding
# ---------------------------------------------------------------------------
# A run-key stably identifies one training run: step, model, variant (with the
# sampler and any FedProx mu folded in so it is unique), feature set and seed.
#   s{step}:{model}:{vtag}:{fset}:seed{seed}
# e.g.  s5:fed_cnn:corrected:full:seed4
#       s6:lgbm:legacy-ls:full:seed2         (legacy-sampler run)
#       s9:fed_cnn:corrected-mu0.1:full:seed3 (FedProx mu=0.1)
# Shard files (experiments/shards/<name>.txt) list one run-key per line
# (# comments allowed). A machine runs exactly the run-keys in its shard.

def job_vtag(job):
    """Variant tag that makes a run-key unique within a step: the variant slug,
    plus '-ls' for the legacy sampler and '-mu<x>' for FedProx."""
    tag = V.variant_slug(V.parse_variant(job["variant"]))
    ex = job["extra"]
    if "--legacy-sampler" in ex:
        tag += "-ls"
    if "--fedprox-mu" in ex:
        tag += "-mu" + ex[ex.index("--fedprox-mu") + 1]
    return tag


def run_id(model, fset, seed):
    return f"{model}__{fset}__seed{seed}"


def job_run_keys(step, job):
    """{run_key: run_id} for every (model, feature-set, seed) of the job."""
    vt = job_vtag(job)
    out = {}
    for m in job["models"].split(","):
        for f in job["fsets"].split(","):
            for s in job["seeds"].split(","):
                out[f"s{step}:{m}:{vt}:{f}:seed{s}"] = run_id(m, f, s)
    return out


def resolve_shard_name(cli=None):
    """Shard name from (in order) the CLI, then REV2_SHARD / MY_SHARD env vars.
    'all', '', 'me,all' and None all mean 'run everything' (single-machine)."""
    name = cli or os.environ.get("REV2_SHARD") or os.environ.get("MY_SHARD") or ""
    name = name.strip()
    return "" if name.lower() in ("", "all") else name


def known_shards():
    if not os.path.isdir(SHARD_DIR):
        return []
    return sorted(f[:-4] for f in os.listdir(SHARD_DIR) if f.endswith(".txt"))


def load_shard_keys(name):
    """Set of run-keys for a shard, or None for the single-machine case."""
    if not name:
        return None
    path = os.path.join(SHARD_DIR, name + ".txt")
    if not os.path.exists(path):
        raise SystemExit(f"shard file not found: {path}\n"
                         f"known shards: {', '.join(known_shards()) or '(none)'}")
    keys = set()
    with open(path) as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if line:
                keys.add(line)
    return keys


def apply_shard(step, J, e):
    """Set job['only'] to the set of run_ids assigned to this machine's shard
    (None = all), and drop jobs with nothing assigned. Step 4 (prepare-only,
    needs the labelled files) is never sharded: it stays on the user's machine."""
    keys = e.get("shard_keys")
    if keys is None or step == 4:
        for j in J:
            j["only"] = None
        return J
    kept = []
    for j in J:
        rk = job_run_keys(step, j)
        only = {rid for k, rid in rk.items() if k in keys}
        if only:
            j["only"] = only
            kept.append(j)
    return kept


def planned(step, e=None):
    """Jobs of a step after shard filtering."""
    e = e or env()
    return apply_shard(step, jobs(step, e), e)


def all_remaining_run_keys(e=None):
    """{run_key: (step, dir, run_id)} for every run in steps 5..10 (unsharded).
    Used by make_shards / coverage checks. Cross-step shared output files (the
    step-7 legacy central-CNN seeds that reuse step 6's) appear under each step's
    run-key but map to the same (dir, run_id)."""
    e = e or env()
    out = {}
    for step in range(5, 11):
        for j in jobs(step, e):
            if "--prepare-only" in j["extra"]:
                continue
            for rk, rid in job_run_keys(step, j).items():
                out[rk] = (step, j["dir"], rid)
    return out


def seeds(n, e):
    if e["sim"]:
        return "1,2" if n == 5 else "1"
    return ",".join(str(i) for i in range(1, n + 1))


def slug(variant):
    return V.variant_slug(V.parse_variant(variant))


def jobs(step, e=None):
    """List of jobs: dict(dir, variant, models, seeds, fsets, extra)."""
    e = e or env()
    J = []

    def add(d, variant, models, nseeds, fsets="full", extra=()):
        J.append({"dir": d, "variant": variant, "models": models, "seeds": seeds(nseeds, e),
                  "fsets": fsets, "extra": list(extra)})

    if step == 4:
        add("main", "corrected", "lgbm", 5, extra=["--prepare-only"])
        add("ls_corrected", "corrected", "lgbm", 5, extra=["--legacy-sampler", "--prepare-only"])
        add("ls_legacy", "legacy", "lgbm", 5, extra=["--legacy-sampler", "--prepare-only"])
    elif step == 5:
        add("main", "corrected", "lgbm,central_cnn", 5, "full,core")
        add("main", "corrected", "fed_cnn", 5, "full,core")
    elif step == 6:
        for v in ABL_VARIANTS + EXTRA_VARIANTS:
            add("var_" + slug(v), v, "lgbm", 5)
            add("var_" + slug(v), v, "central_cnn", 3)
        for v in ("corrected", "legacy"):
            add("ls_" + slug(v), v, "lgbm", 5, extra=["--legacy-sampler"])
            add("ls_" + slug(v), v, "central_cnn", 3, extra=["--legacy-sampler"])
    elif step == 7:
        add("var_legacy", "legacy", "central_cnn", 5)
        add("var_legacy", "legacy", "fed_cnn", 5)
    elif step == 8:
        add("main", "corrected", "central_mlp", 5)
        add("main", "corrected", "fed_mlp", 5)
    elif step == 9:
        for mu in ("0.01", "0.1"):
            add(f"fedprox_mu{mu}", "corrected", "fed_cnn", 3, extra=["--fedprox-mu", mu])
    elif step == 10:
        for c in ABL:
            add("var_" + slug(f"loo:{c}"), f"loo:{c}", "fed_cnn", 3)
    else:
        raise SystemExit(f"step {step} is not a run step (4..10)")
    return J


def command(job, e):
    train = [os.path.join(e["labeled"], f"opensky_{d}_labeled.csv") for d in e["train_days"]]
    cmd = [sys.executable, "-u", os.path.join(HERE, "run_experiments.py"),
           "--train-files", *train, "--holdout", e["holdout"],
           "--out-dir", os.path.join(e["runs"], job["dir"]), "--cache-dir", e["cache"],
           "--variant", job["variant"], "--protocol", "v2",
           "--models", job["models"], "--seeds", job["seeds"], "--feature-sets", job["fsets"],
           "--device", e["device"], "--stop-flag", e["stop"]]
    if e["sim"]:
        cmd += ["--quick", "--max-total-rows", os.environ.get("REV2_MAX_ROWS", "60000"),
                "--min-client-rows", os.environ.get("REV2_MIN_CLIENT_ROWS", "1000"),
                "--strengths", "0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"]
    else:
        cmd += ["--max-total-rows", os.environ.get("REV2_MAX_ROWS", "10000000")]
    if e.get("trust_cache"):
        cmd += ["--trust-cache"]
    only = job.get("only")
    if only is not None:
        cmd += ["--only-runs", ",".join(sorted(only))]
    return cmd + job["extra"]


def expected(job, e):
    if "--prepare-only" in job["extra"]:
        return []
    only = job.get("only")
    out = []
    for m in job["models"].split(","):
        for f in job["fsets"].split(","):
            for s in job["seeds"].split(","):
                rid = run_id(m, f, s)
                if only is not None and rid not in only:
                    continue
                out.append((m, os.path.join(e["runs"], job["dir"], "runs", rid + ".json")))
    return out


def run_ok(path):
    try:
        r = json.load(open(path))
        return "metrics" in r and "s_star" in r
    except Exception:
        return False


def measured_seconds(e):
    """Mean wall time per model from the finished runs (for the ETA)."""
    acc = {}
    for p in glob.glob(os.path.join(e["runs"], "*", "runs", "*__*__seed*.json")):
        try:
            r = json.load(open(p))
            acc.setdefault(r["model"], []).append(float(r["wall_seconds"]))
        except Exception:
            pass
    out = dict(DEFAULT_SECONDS)
    for m, v in acc.items():
        out[m] = sum(v) / len(v)
    return out, {m: len(v) for m, v in acc.items()}


def step_counts(step, e):
    exp = [x for j in planned(step, e) for x in expected(j, e)]
    seen, uniq = set(), []
    for m, p in exp:
        if p not in seen:
            seen.add(p)
            uniq.append((m, p))
    done = [(m, p) for m, p in uniq if run_ok(p)]
    return uniq, done


def prep_done(job, e):
    """A prepare-only job is done when its sample cache meta exists."""
    design = V.design_of(V.parse_variant(job["variant"]))
    tag = f"legacy_{design}" if "--legacy-sampler" in job["extra"] else "uniform"
    return bool(glob.glob(os.path.join(e["cache"], f"train_sample_{tag}_*.json")))


def cmd_print(step, shard=None):
    e = env(shard)
    for j in planned(step, e):
        print(" ".join(shlex.quote(c) for c in command(j, e)))


def cmd_run(step, shard=None):
    e = env(shard)
    plan = planned(step, e)
    if step != 4 and e.get("shard_keys") is not None and not plan:
        print(f"step {step}: no runs assigned to this machine (shard '{e['shard']}'); "
              f"done by another machine")
        return 0
    for j in plan:
        if os.path.exists(e["stop"]):
            return EXIT_STOPPED
        exp = expected(j, e)
        if exp and all(run_ok(p) for _, p in exp):
            continue
        if not exp and prep_done(j, e):
            continue
        cmd = command(j, e)
        print(f"[{time.strftime('%H:%M:%S')}] $ " + " ".join(shlex.quote(c) for c in cmd), flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            return rc
    return cmd_verify(step, e["shard"])


def cmd_verify(step, shard=None):
    e = env(shard)
    if step == 4:
        missing = [j["dir"] for j in planned(4, e) if not prep_done(j, e)]
    else:
        uniq, done = step_counts(step, e)
        missing = [p for m, p in uniq if (m, p) not in done]
    if missing:
        print(f"step {step}: {len(missing)} output(s) missing, e.g. {missing[0]}")
        return 1
    return 0


def cmd_runkeys():
    """Print every remaining run-key (steps 5..10) and the output it maps to."""
    e = env("")   # unsharded
    rk = all_remaining_run_keys(e)
    files = {}
    for k, (step, d, rid) in rk.items():
        files.setdefault((d, rid), []).append(k)
    for k in sorted(rk):
        step, d, rid = rk[k]
        shared = " [shared output]" if len(files[(d, rid)]) > 1 else ""
        print(f"{k}\t-> runs/{d}/runs/{rid}.json{shared}")
    print(f"# {len(rk)} run-keys, {len(files)} distinct output files")
    return 0


def cmd_status(shard=None):
    e = env(shard)
    if e.get("shard_keys") is not None:
        print(f"  shard '{e['shard']}': "
              f"{sum(len(expected(j, e)) for step in range(5, 11) for j in planned(step, e))} "
              f"runs assigned to this machine")
    secs, n = measured_seconds(e)
    total_left = 0.0
    for step in range(5, 11):
        uniq, done = step_counts(step, e)
        donep = {p for _, p in done}
        left = sum(secs[m] for m, p in uniq if p not in donep)
        total_left += left
        print(f"  step {step:2d} {STEP_NAMES[step]:<45} {len(done):3d}/{len(uniq):3d} runs done, "
              f"about {left / 3600:4.1f} h left")
    print(f"  training time left in steps 5-10: about {total_left / 3600:.1f} h "
          f"(measured run times: " + (", ".join(f"{m} {secs[m] / 60:.1f} min" for m in sorted(n))
                                      or "none yet, using defaults") + ")")
    return 0


def cmd_table(shard=None):
    e = env(shard)
    tot_runs, tot_h = 0, 0.0
    print(f"{'step':<48}{'lgbm':>6}{'c_cnn':>7}{'c_mlp':>7}{'f_cnn':>7}{'f_mlp':>7}{'hours':>8}")
    for step in range(5, 11):
        uniq, _ = step_counts(step, e)
        c = {}
        for m, _p in uniq:
            c[m] = c.get(m, 0) + 1
        # runs already counted in an earlier step are not counted twice (step_counts is per step;
        # step 7 re-requests the 3 legacy central CNN seeds of step 6)
        if step == 7 and not e["sim"]:
            c["central_cnn"] = c.get("central_cnn", 0) - 3
        h = sum(DEFAULT_SECONDS[m] * k for m, k in c.items()) / 3600
        tot_runs += sum(c.values())
        tot_h += h
        print(f"{str(step) + ' ' + STEP_NAMES[step]:<48}{c.get('lgbm', 0):>6}{c.get('central_cnn', 0):>7}"
              f"{c.get('central_mlp', 0):>7}{c.get('fed_cnn', 0):>7}{c.get('fed_mlp', 0):>7}{h:>8.1f}")
    print(f"total: {tot_runs} runs, {tot_h:.1f} h of training (default run times)")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("print", "run", "verify", "status", "table", "runkeys"))
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--shard", default=None,
                    help="shard name (experiments/shards/<name>.txt); default from "
                         "REV2_SHARD / MY_SHARD; 'all' or unset = run everything")
    a = ap.parse_args()
    if a.action in ("print", "run", "verify") and a.step is None:
        ap.error("--step is required")
    if a.action == "print":
        return cmd_print(a.step, a.shard) or 0
    if a.action == "run":
        return cmd_run(a.step, a.shard)
    if a.action == "verify":
        return cmd_verify(a.step, a.shard)
    if a.action == "status":
        return cmd_status(a.shard)
    if a.action == "runkeys":
        return cmd_runkeys()
    return cmd_table(a.shard)


if __name__ == "__main__":
    sys.exit(main())
