"""
test_fedprox_resume.py
-------------------------------------------------------------------
 1. FedProx with mu = 0 (the FedProx code path) reproduces FedAvg exactly.
 2. Resume equals uninterrupted, for the federated CNN, federated MLP and both
    centralized networks, under
      (a) a graceful stop (STOP flag seen after a checkpoint, exit code 75), and
      (b) a SIGKILL (kill -9) in the middle of a round / epoch,
    each followed by rerunning the same command. Compared: per-round / per-epoch
    validation accuracy, best round, s*, validation and test probabilities
    (bit-identical arrays) and all metrics.
 3. mu > 0 changes the result (the term is really applied).

Usage: python3 experiments/test_fedprox_resume.py TRAIN_CSV HOLDOUT_CSV WORK_DIR
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = "fed_cnn,central_cnn,fed_mlp,central_mlp"


def run(train, hold, out, extra, expect=(0,)):
    cmd = [sys.executable, os.path.join(HERE, "run_experiments.py"), "--train-files", train,
           "--holdout", hold, "--out-dir", out, "--cache-dir", os.path.join(os.path.dirname(out), "cache"),
           "--max-total-rows", "60000", "--device", "cpu", "--quick", "--rounds", "4",
           "--central-max-epochs", "4", "--min-client-rows", "1000", "--seeds", "1",
           "--feature-sets", "full", "--stop-flag", os.path.join(out, "STOP")] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode not in expect:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"exit code {r.returncode}, expected {expect}: {' '.join(extra)}")
    return r.returncode


def same(a_dir, b_dir, rid, what):
    a = json.load(open(os.path.join(a_dir, "runs", rid + ".json")))
    b = json.load(open(os.path.join(b_dir, "runs", rid + ".json")))
    for k in ("metrics", "s_star", "pr_auc", "test_summary"):
        assert a[k] == b[k], f"{what}: {rid}: {k} differs"
    assert a["selection"]["val_by_s"] == b["selection"]["val_by_s"], f"{what}: {rid}: validation table"
    for k in ("per_round_val_acc", "per_round_train_acc", "best_round", "per_epoch_val_acc",
              "per_epoch_train_loss", "best_epoch", "epochs_run"):
        assert a["training"].get(k) == b["training"].get(k), f"{what}: {rid}: training.{k} differs"
    for suffix in ("", "__val"):
        pa = np.load(os.path.join(a_dir, "probs", rid + suffix + ".npz"))["proba"]
        pb = np.load(os.path.join(b_dir, "probs", rid + suffix + ".npz"))["proba"]
        assert np.array_equal(pa, pb), f"{what}: {rid}: probabilities{suffix} differ"
    return b


def main():
    train, hold, work = sys.argv[1:4]
    d = lambda n: os.path.join(work, n)  # noqa: E731
    run(train, hold, d("full"), ["--models", MODELS])
    rids = [f"{m}__full__seed1" for m in MODELS.split(",")]

    # 1. FedProx mu = 0 == FedAvg
    run(train, hold, d("mu0"), ["--models", "fed_cnn,fed_mlp", "--fedprox-mu", "0"])
    for rid in (rids[0], rids[2]):
        b = same(d("full"), d("mu0"), rid, "FedProx mu=0 vs FedAvg")
        assert b["training"]["algorithm"] == "fedprox" and b["fedprox_mu"] == 0.0
    run(train, hold, d("mu1"), ["--models", "fed_cnn", "--fedprox-mu", "1.0"])
    a = json.load(open(os.path.join(d("full"), "runs", rids[0] + ".json")))
    b = json.load(open(os.path.join(d("mu1"), "runs", rids[0] + ".json")))
    assert a["training"]["per_round_val_acc"] != b["training"]["per_round_val_acc"], "mu=1 had no effect"
    print("[1] FedProx mu=0 == FedAvg (fed_cnn, fed_mlp: bit-identical probabilities); mu=1 differs")

    # 2a. graceful stop after 2 checkpointed units, then resume (several stops in a row)
    n_stop = 0
    rc = run(train, hold, d("stop"), ["--models", MODELS, "--test-stop-after-units", "2"], expect=(75,))
    while rc == 75:
        n_stop += 1
        assert n_stop < 20
        os.remove(os.path.join(d("stop"), "STOP"))
        rc = run(train, hold, d("stop"), ["--models", MODELS, "--test-stop-after-units", "3"],
                 expect=(0, 75))
    nres = 0
    for rid in rids:
        nres += same(d("full"), d("stop"), rid, "graceful stop + resume")["training"]["resumed_times"]
    assert nres >= 3, f"runs were not resumed from checkpoints ({nres})"
    print(f"[2a] graceful stop: {n_stop} stops, {nres} checkpoint resumes -> identical to uninterrupted")

    # 2b. kill -9 in the middle of a round / epoch, then resume
    n_kill = 0
    rc = run(train, hold, d("kill"), ["--models", MODELS, "--test-kill-after-units", "1"], expect=(-9,))
    while rc == -9:
        n_kill += 1
        assert n_kill < 20
        rc = run(train, hold, d("kill"), ["--models", MODELS, "--test-kill-after-units", "2"],
                 expect=(0, -9))
    nres = 0
    for rid in rids:
        nres += same(d("full"), d("kill"), rid, "kill -9 + resume")["training"]["resumed_times"]
    assert nres >= 3
    print(f"[2b] kill -9 mid-unit: {n_kill} kills, {nres} checkpoint resumes -> identical to uninterrupted")
    print("FEDPROX AND RESUME OK")


if __name__ == "__main__":
    main()
