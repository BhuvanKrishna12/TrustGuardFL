"""
test_protocol_v1_regression.py
-------------------------------------------------------------------
Regression test: `run_experiments.py --protocol v1 --variant corrected` on
v3-format files gives the same results as the round-1 harness (frozen copy in
experiments/reference_r1/) for a fixed seed: same natural prior, same training
trace, same saved test probabilities, same metrics for every strength.
Also runs --protocol v1 on the multi-variant files and checks that it matches
too (the corrected view of a multi-variant file is the v3 file).

Usage: python3 experiments/test_protocol_v1_regression.py V3_TRAIN_CSV MV_TRAIN_CSV V3_HOLDOUT MV_HOLDOUT WORK_DIR
"""

import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def run(script, train, hold, out, extra):
    cmd = [sys.executable, os.path.join(HERE, script), "--train-files", train, "--holdout", hold,
           "--out-dir", out, "--max-total-rows", "60000", "--device", "cpu", "--quick",
           "--min-client-rows", "1000", "--seeds", "1", "--feature-sets", "full,core"] + extra
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-3000:])
        raise SystemExit(f"{script} failed")


def main():
    v3_train, mv_train, v3_hold, mv_hold, work = sys.argv[1:6]
    old, new, newmv = (os.path.join(work, d) for d in ("r1", "r2_v1", "r2_v1_mv"))
    run(os.path.join("reference_r1", "run_experiments_r1.py"), v3_train, v3_hold, old, [])
    run("run_experiments.py", v3_train, v3_hold, new, ["--protocol", "v1", "--variant", "corrected"])
    run("run_experiments.py", mv_train, mv_hold, newmv, ["--protocol", "v1", "--variant", "corrected"])
    n = 0
    for model in ("fed_cnn", "central_cnn", "lgbm"):
        for fset in ("full", "core"):
            rid = f"{model}__{fset}__seed1"
            a = json.load(open(os.path.join(old, "runs", rid + ".json")))
            pa = np.load(os.path.join(old, "probs", rid + ".npz"))["proba"]
            for label, d in (("v3 files", new), ("multi-variant files", newmv)):
                b = json.load(open(os.path.join(d, "runs", rid + ".json")))
                assert b["protocol"] == "v1" and b["variant"] == "corrected"
                for s, m in a["metrics"].items():
                    assert b["metrics"][s] == m, f"{rid} ({label}): metrics differ at s={s}"
                for k in ("natural_prior", "class_weights", "train_sample_rows", "train_raw_rows",
                          "n_clients", "train_split_rows", "val_rows", "train_sample_class_counts"):
                    assert a["data"][k] == b["data"][k], f"{rid} ({label}): data.{k} differs"
                for k in ("per_round_val_acc", "per_round_train_acc", "best_round", "per_epoch_val_acc",
                          "per_epoch_train_loss", "best_epoch", "best_iteration", "best_val_acc",
                          "feature_importance_gain_share"):
                    assert a["training"].get(k) == b["training"].get(k), f"{rid} ({label}): training.{k}"
                pb = np.load(os.path.join(d, "probs", rid + ".npz"))["proba"]
                assert pa.shape == pb.shape and np.array_equal(pa, pb), f"{rid} ({label}): probabilities"
                assert a["holdout"]["support"] == b["holdout"]["support"]
            n += 1
    print(f"PROTOCOL V1 REGRESSION OK: {n} runs identical to the round-1 harness "
          f"(metrics, training trace, saved probabilities), on v3 and on multi-variant files")


if __name__ == "__main__":
    main()
