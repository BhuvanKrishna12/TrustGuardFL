"""
check_equivalence.py
-------------------------------------------------------------------
Checks that run_experiments.py drives exactly the FedAvg procedure of
federated_simulation.main:

 1. make_clients(seed=42) reproduces federated_simulation.build_clients
    (same clients, same train/validation rows, same scaled values).
 2. train_fed_cnn(seed=42, full feature set) gives the same per-round
    validation accuracies as `python federated_simulation.py ... --model cnn
    --cnn-channels 32,64 --n-rounds R` prints as "test acc" (run on CPU).

Usage:
  python3 experiments/check_equivalence.py TRAIN_CSV [TRAIN_CSV ...] \
      --max-total-rows 200000 --rounds 3
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import types

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, REPO)

import run_experiments as rx  # noqa: E402
import federated_simulation as fs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--max-total-rows", type=int, default=200_000)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    rx.setup_device("cpu")

    df = fs.load_data(args.paths, max_total_rows=args.max_total_rows)
    X = fs.prep_features(df)
    classes = sorted(df["attack_type"].astype(str).unique())
    y = df["attack_type"].astype(str).map({c: i for i, c in enumerate(classes)}).values
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    ref = fs.build_clients(df, Xs, y, fs.MIN_CLIENT_ROWS)
    mine, _ = rx.make_clients(df["region_cell"].astype(str).values, fs.MIN_CLIENT_ROWS, 42)
    Xf = Xs.values.astype(np.float32)
    assert [str(k) for k in ref] == [c["region"] for c in mine], "client order differs"
    for c in mine:
        r = ref[c["region"]] if c["region"] in ref else ref[type(next(iter(ref)))(c["region"])]
        assert np.array_equal(r["X_train"], Xf[c["train_idx"]]), c["region"]
        assert np.array_equal(r["X_test"], Xf[c["val_idx"]]), c["region"]
        assert np.array_equal(r["y_train"], y[c["train_idx"]]) and np.array_equal(r["y_test"], y[c["val_idx"]])
    print(f"[1] make_clients(seed=42) == build_clients: {len(mine)} clients identical")

    # 2. FedAvg rounds
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    with tempfile.TemporaryDirectory() as td:
        cmd = [sys.executable, os.path.join(REPO, "federated_simulation.py"), *args.paths,
               "--model", "cnn", "--cnn-channels", "32,64", "--n-rounds", str(args.rounds),
               "--max-total-rows", str(args.max_total_rows),
               "--plot-output", os.path.join(td, "c.png")]
        outp = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=td).stdout
    ref_acc = [float(a) for a in re.findall(r"test acc: ([0-9.]+)", outp)]
    cw = len(y) / (len(classes) * np.bincount(y, minlength=len(classes)))
    a = types.SimpleNamespace(rounds=args.rounds, local_epochs=fs.LOCAL_EPOCHS, cnn_channels=(32, 64))
    _, info = rx.train_fed_cnn(Xf, y.astype(np.int64), mine, cw, len(classes), a, 42, "check")
    my_acc = [round(v, 4) for v in info["per_round_val_acc"]]
    print(f"[2] federated_simulation.py test acc per round: {ref_acc}")
    print(f"    run_experiments train_fed_cnn val acc:      {my_acc}")
    assert ref_acc == my_acc, "round accuracies differ"
    print("EQUIVALENT")


if __name__ == "__main__":
    main()
