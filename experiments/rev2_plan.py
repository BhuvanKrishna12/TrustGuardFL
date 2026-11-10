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


def env():
    W = os.environ.get("REV2_W", os.path.expanduser("~/rev2_work"))
    sim = os.environ.get("REV2_SIM", "") == "1"
    return {"W": W, "sim": sim,
            "labeled": os.path.join(W, "labeled_mv"), "holdout": os.path.join(W, "holdout_sample_mv.csv"),
            "runs": os.path.join(W, "runs"), "cache": os.path.join(W, "cache"),
            "stop": os.path.join(W, "STOP"),
            "train_days": os.environ.get("REV2_TRAIN_DAYS", " ".join(TRAIN_DAYS)).split(),
            "test_days": os.environ.get("REV2_TEST_DAYS", " ".join(TEST_DAYS)).split(),
            "device": os.environ.get("REV2_DEVICE", "auto")}


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
    return cmd + job["extra"]


def expected(job, e):
    if "--prepare-only" in job["extra"]:
        return []
    out = []
    for m in job["models"].split(","):
        for f in job["fsets"].split(","):
            for s in job["seeds"].split(","):
                out.append((m, os.path.join(e["runs"], job["dir"], "runs", f"{m}__{f}__seed{s}.json")))
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
    exp = [x for j in jobs(step, e) for x in expected(j, e)]
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


def cmd_print(step):
    e = env()
    for j in jobs(step, e):
        print(" ".join(shlex.quote(c) for c in command(j, e)))


def cmd_run(step):
    e = env()
    for j in jobs(step, e):
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
    return cmd_verify(step)


def cmd_verify(step):
    e = env()
    if step == 4:
        missing = [j["dir"] for j in jobs(4, e) if not prep_done(j, e)]
    else:
        uniq, done = step_counts(step, e)
        missing = [p for m, p in uniq if (m, p) not in done]
    if missing:
        print(f"step {step}: {len(missing)} output(s) missing, e.g. {missing[0]}")
        return 1
    return 0


def cmd_status():
    e = env()
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


def cmd_table():
    e = env()
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
    ap.add_argument("action", choices=("print", "run", "verify", "status", "table"))
    ap.add_argument("--step", type=int, default=None)
    a = ap.parse_args()
    if a.action in ("print", "run", "verify") and a.step is None:
        ap.error("--step is required")
    if a.action == "print":
        return cmd_print(a.step) or 0
    if a.action == "run":
        return cmd_run(a.step)
    if a.action == "verify":
        return cmd_verify(a.step)
    if a.action == "status":
        return cmd_status()
    return cmd_table()


if __name__ == "__main__":
    sys.exit(main())
