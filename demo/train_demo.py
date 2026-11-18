"""
train_demo.py: Flower federated training over the six demo clients, with a live
JSONL event log for the demo UI.

One Flower client per demo/data/Cx_train.csv (written by prepare_demo_data.py),
legacy fl.simulation.start_simulation, sample-weighted FedAvg, CNN (32,64) by
default. Domain logic (prep_features, build_model, local_train, evaluate,
LOCAL_EPOCHS, BATCH_SIZE, LR) is imported unchanged from federated_simulation.py.

Writes (see demo/CONTRACT.md):
    demo/runs/<run>.jsonl      run_start, client_fit, round, run_end events,
                               flushed line by line so a live tail works
    demo/runs/latest.jsonl     copy of the finished run (the replay source)
    demo/models/<run>.pt       best-round checkpoint, same payload as fixed4_cnn.pt
    demo/models/<run>_preprocess.pkl   {"scaler", "label_encoder"} via joblib
    demo/models/latest.pt (+ latest_preprocess.pkl)   copy of the model that goes
                               with latest.jsonl, for `server.py --model demo/models/latest.pt`

Usage (run from the project root):
    python3 demo/train_demo.py                         # 15 rounds, CNN 32,64
    python3 demo/train_demo.py --rounds 5 --run-name rehearsal
    python3 demo/train_demo.py --model mlp --mlp-hidden-sizes 64,32
    python3 demo/train_demo.py --dry-run               # load data, build clients, exit
    python3 demo/train_demo.py --keep-latest           # do not replace an existing latest
    python3 demo/train_demo.py --data-dir /tmp/fl-demo-fixture/demo/data \
        --runs-dir /tmp/fl-demo-fixture/demo/runs --models-dir /tmp/fl-demo-fixture/demo/models
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DEMO_DIR = Path(__file__).resolve().parent
ROOT = DEMO_DIR.parent
sys.path.insert(0, str(ROOT))

import federated_simulation as fs  # noqa: E402
sys.path.insert(0, str(DEMO_DIR))
import fs_compat  # noqa: E402  (demo-side wrapper around fs.prep_features)

import joblib  # noqa: E402
import torch  # noqa: E402
from sklearn.preprocessing import LabelEncoder, StandardScaler  # noqa: E402

CLASSES = ["benign", "flooding", "ghost_aircraft", "gps_spoof", "replay", "velocity_spoof"]
CPU = torch.device("cpu")

# Ray settings from the contract: small and predictable on a 13 GB WSL2 laptop.
RAY_INIT_ARGS = {"num_cpus": 4, "object_store_memory": 400_000_000,
                 "include_dashboard": False, "logging_level": 40}
CLIENT_RESOURCES = {"num_cpus": 1}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def get_parameters(model):
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_parameters(model, parameters):
    keys = model.state_dict().keys()
    model.load_state_dict(OrderedDict((k, torch.tensor(v)) for k, v in zip(keys, parameters)),
                          strict=True)


class EventLog:
    """Append-only JSONL writer. Every line is flushed so the server can tail it."""

    def __init__(self, path):
        self.path = path
        self.f = open(path, "w", encoding="utf-8")

    def write(self, obj):
        obj.setdefault("t", round(time.time(), 3))
        self.f.write(json.dumps(obj) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()


def rel_to_root(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


def load_clients(data_dir):
    """Read clients.json and every Cx_train.csv. Returns (meta list, list of
    (X DataFrame float32, y string array))."""
    meta_path = data_dir / "clients.json"
    if not meta_path.is_file():
        sys.exit(f"ERROR: {meta_path} not found. Run demo/prepare_demo_data.py first.")
    meta = json.loads(meta_path.read_text())["clients"]
    frames = []
    for c in meta:
        path = data_dir / f"{c['id']}_train.csv"
        if not path.is_file():
            sys.exit(f"ERROR: {path} not found.")
        df = pd.read_csv(path, low_memory=False)
        df["attack_type"] = df["attack_type"].astype(str)
        bad = ~df["attack_type"].isin(CLASSES)
        if bad.any():
            print(f"  WARNING: {path.name}: dropping {int(bad.sum())} rows with unknown labels")
            df = df[~bad]
        X = fs_compat.prep_features(fs, df)  # bool strings to 1/0, coerce, NaN to -1, float32
        frames.append((X.reset_index(drop=True), df["attack_type"].to_numpy()))
    return meta, frames


# ---------------------------------------------------------------------------
# Flower pieces (built lazily so --dry-run works without flwr installed)
# ---------------------------------------------------------------------------

def make_flower_classes(fl):

    class DemoClient(fl.client.NumPyClient):
        """One demo region. Loads its own pre-scaled arrays from a small .npz so the
        object Ray ships to each actor stays tiny, then calls fs.local_train."""

        def __init__(self, cid, npz_path, model_cfg, class_weights):
            self.cid = cid
            self.npz_path = npz_path
            self.model_cfg = model_cfg
            self.class_weights = class_weights

        def _build(self):
            return fs.build_model(device=CPU, **self.model_cfg)

        def get_parameters(self, config):
            return get_parameters(self._build())

        def fit(self, parameters, config):
            torch.set_num_threads(1)
            fs.DEVICE = CPU  # local_train moves batches to fs.DEVICE; the demo is CPU only
            with np.load(self.npz_path) as d:
                X_train, y_train = d["X_train"], d["y_train"]
            model = self._build()
            set_parameters(model, parameters)
            cw = torch.tensor(self.class_weights, dtype=torch.float32)
            _, train_acc = fs.local_train(model, X_train, y_train, fs.LOCAL_EPOCHS, cw)
            return get_parameters(model), len(X_train), {
                "client_id": self.cid, "train_acc": float(train_acc),
                "finished_at": time.time(),
            }

        def evaluate(self, parameters, config):
            # unused: fraction_evaluate=0, evaluation happens centrally in the strategy
            return 0.0, 0, {}

    class DemoFedAvg(fl.server.strategy.FedAvg):
        """Sample-weighted FedAvg (Flower default) plus: client_fit and round events,
        byte accounting, and best-round tracking."""

        def __init__(self, *args, log, payload_bytes, eval_global, **kwargs):
            super().__init__(*args, **kwargs)
            self.log = log
            self.payload_bytes = payload_bytes
            self.eval_global = eval_global
            self.bytes_total = 0
            self.bytes_this_round = 0
            self.best_acc = -1.0
            self.best_round = 0
            self.best_params = None
            self.last_acc = None

        def aggregate_fit(self, server_round, results, failures):
            if failures:
                print(f"  WARNING: round {server_round}: {len(failures)} client(s) failed: "
                      f"{[str(f) for f in failures][:3]}")
            ordered = sorted(results, key=lambda r: r[1].metrics.get("finished_at", 0.0))
            for _, res in ordered:
                m = res.metrics
                self.log.write({"type": "client_fit", "round": server_round,
                                "client": m.get("client_id", "?"),
                                "train_acc": round(float(m.get("train_acc", 0.0)), 4),
                                "n": int(res.num_examples),
                                "t": round(float(m.get("finished_at", time.time())), 3)})
            # up and down per participating client, same convention as federated_simulation
            self.bytes_this_round = 2 * self.payload_bytes * len(results)
            self.bytes_total += self.bytes_this_round
            return super().aggregate_fit(server_round, results, failures)

        def evaluate(self, server_round, parameters):
            if server_round == 0:
                return None  # initial random weights, nothing to report
            nds = fl.common.parameters_to_ndarrays(parameters)
            global_acc, per_client = self.eval_global(nds)
            self.last_acc = global_acc
            self.log.write({"type": "round", "round": server_round,
                            "global_acc": round(global_acc, 4), "per_client": per_client,
                            "bytes_this_round": self.bytes_this_round,
                            "bytes_total": self.bytes_total})
            flag = ""
            if global_acc > self.best_acc:
                self.best_acc, self.best_round = global_acc, server_round
                self.best_params = [a.copy() for a in nds]
                flag = "  (best so far)"
            accs = " ".join(f"{k}={v['acc']:.3f}" for k, v in per_client.items())
            print(f"  Round {server_round:2d}: global acc {global_acc:.4f}  [{accs}]  "
                  f"comm {self.bytes_total / 1e6:.2f} MB{flag}", flush=True)
            return 1.0 - global_acc, {"accuracy": global_acc}

    return DemoClient, DemoFedAvg


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Flower FedAvg over the six demo clients.")
    p.add_argument("--data-dir", default=str(DEMO_DIR / "data"))
    p.add_argument("--runs-dir", default=str(DEMO_DIR / "runs"))
    p.add_argument("--models-dir", default=str(DEMO_DIR / "models"))
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--model", default="cnn", choices=["cnn", "mlp"],
                   help="row-level models only (gru needs sequences, not supported here)")
    p.add_argument("--cnn-channels", default="32,64")
    p.add_argument("--mlp-hidden-sizes", default="64,32")
    p.add_argument("--run-name", default=None,
                   help="default: run_YYYYmmdd_HHMMSS (local time)")
    p.add_argument("--dry-run", action="store_true",
                   help="load data and build clients, then exit without training")
    p.add_argument("--keep-latest", action="store_true",
                   help="leave an existing runs/latest.jsonl and models/latest.pt alone "
                        "(the server's Train live button uses this so the rehearsed replay "
                        "survives); if there is no latest yet, this run still becomes latest")
    args = p.parse_args()

    if args.rounds < 1:
        sys.exit("ERROR: --rounds must be at least 1")
    run = args.run_name or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    if run == "latest" or not all(ch.isalnum() or ch in "-_." for ch in run):
        sys.exit("ERROR: --run-name must be letters, digits, '-', '_' or '.', and not 'latest'")
    cnn_channels = tuple(int(x) for x in args.cnn_channels.split(","))
    mlp_hidden = tuple(int(x) for x in args.mlp_hidden_sizes.split(","))
    data_dir = Path(args.data_dir).expanduser().resolve()
    runs_dir = Path(args.runs_dir).expanduser().resolve()
    models_dir = Path(args.models_dir).expanduser().resolve()

    t0 = time.time()
    torch.manual_seed(fs.RNG_SEED)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    fs.DEVICE = CPU

    # ---- data -------------------------------------------------------------
    print(f"Loading client data from {data_dir} ...")
    meta, frames = load_clients(data_dir)

    le = LabelEncoder().fit(CLASSES)  # fixed 6-class order, same as fixed4_cnn
    y_all = np.concatenate([le.transform(y) for _, y in frames])
    X_all = pd.concat([X for X, _ in frames], ignore_index=True)

    # SIMULATION SIMPLIFICATION (same as federated_simulation.py): the scaler is
    # fitted on the pooled training data of all clients. A real deployment would
    # fit it on a public reference sample or use a federated-safe scheme.
    scaler = StandardScaler().fit(X_all)

    # Balanced class weights from the global label counts, same formula as
    # federated_simulation.py (n_samples / (n_classes * count_c)). A class with no
    # rows at all gets weight 0 instead of a division by zero.
    counts = np.bincount(y_all, minlength=len(CLASSES))
    class_weights = np.where(counts > 0, len(y_all) / (len(CLASSES) * np.maximum(counts, 1)), 0.0)
    del X_all, y_all

    clients = []
    tmpdir = Path(tempfile.mkdtemp(prefix="fl_demo_"))
    for c, (X, y_str) in zip(meta, frames):
        Xs = scaler.transform(X).astype(np.float32)
        y = le.transform(y_str)
        rng = np.random.default_rng(fs.RNG_SEED)  # 80/20 split, seed 42, per client
        perm = rng.permutation(len(y))
        cut = int(len(perm) * 0.8)
        tr, te = perm[:cut], perm[cut:]
        npz = tmpdir / f"{c['id']}.npz"
        np.savez(npz, X_train=Xs[tr], y_train=y[tr])
        clients.append({"id": c["id"], "cell": c["cell"], "name": c["name"],
                        "train_rows": int(len(y)), "n_fit": int(len(tr)), "npz": str(npz),
                        "X_test": Xs[te], "y_test": y[te]})
    del frames

    print(f"  {len(clients)} clients, class weights "
          + ", ".join(f"{n} {w:.2f}" for n, w in zip(CLASSES, class_weights)))
    for c in clients:
        print(f"  {c['id']} {c['cell']:<7} {c['name']:<18} rows {c['train_rows']:>7,}  "
              f"fit {c['n_fit']:>7,}  test {len(c['y_test']):>6,}")

    model_cfg = {"model_type": args.model, "n_features": len(fs.FEATURE_COLS),
                 "n_classes": len(CLASSES), "mlp_hidden_sizes": mlp_hidden,
                 "cnn_channels": cnn_channels}
    template = fs.build_model(device=CPU, **model_cfg)
    payload_bytes = fs.model_size_bytes(template)
    print(f"  model {args.model} {cnn_channels if args.model == 'cnn' else mlp_hidden}, "
          f"{payload_bytes:,} bytes per transfer")

    if args.dry_run:
        shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"Dry run OK ({time.time() - t0:.1f}s). Exiting before training.")
        return

    X_glob = np.concatenate([c["X_test"] for c in clients])
    y_glob = np.concatenate([c["y_test"] for c in clients])

    def eval_global(ndarrays):
        model = fs.build_model(device=CPU, **model_cfg)
        set_parameters(model, ndarrays)
        acc, _ = fs.evaluate(model, X_glob, y_glob, CLASSES)
        per = {}
        for c in clients:
            a, _ = fs.evaluate(model, c["X_test"], c["y_test"], CLASSES)
            per[c["id"]] = {"acc": round(float(a), 4), "n": int(len(c["y_test"]))}
        return float(acc), per

    # ---- Flower -------------------------------------------------------------
    import flwr as fl
    from flwr.common import Context

    DemoClient, DemoFedAvg = make_flower_classes(fl)
    client_specs = [(c["id"], c["npz"]) for c in clients]
    cw_list = [float(w) for w in class_weights]

    def client_fn(context: Context):
        cid, npz = client_specs[int(context.node_config["partition-id"])]
        return DemoClient(cid, npz, model_cfg, cw_list).to_client()

    runs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    run_path = runs_dir / f"{run}.jsonl"
    model_path = models_dir / f"{run}.pt"
    log = EventLog(run_path)
    log.write({"type": "run_start", "run": run,
               "clients": [{"id": c["id"], "cell": c["cell"], "name": c["name"],
                            "train_rows": c["train_rows"]} for c in clients],
               "rounds": args.rounds, "model": args.model,
               "cnn_channels": list(cnn_channels), "payload_bytes": payload_bytes})
    print(f"\nRun {run}: {args.rounds} rounds, {fs.LOCAL_EPOCHS} local epochs, "
          f"events -> {run_path}", flush=True)

    n = len(clients)
    strategy = DemoFedAvg(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=n, min_available_clients=n,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(template)),
        fit_metrics_aggregation_fn=lambda metrics: {},
        log=log, payload_bytes=payload_bytes, eval_global=eval_global,
    )

    # Ray workers are spawned by the raylet, which inherits this environment:
    # this is how they find federated_simulation.py. One torch thread per actor.
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(DEMO_DIR)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else []))
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    try:
        fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=n,
            config=fl.server.ServerConfig(num_rounds=args.rounds),
            strategy=strategy,
            client_resources=dict(CLIENT_RESOURCES),
            ray_init_args=dict(RAY_INIT_ARGS),
        )
        if strategy.best_params is None:
            raise RuntimeError("no round completed")
    except BaseException as ex:  # also KeyboardInterrupt: leave a clear marker for the UI
        log.write({"type": "run_error", "error": f"{type(ex).__name__}: {ex}"})
        log.close()
        shutil.rmtree(tmpdir, ignore_errors=True)
        if isinstance(ex, (KeyboardInterrupt, SystemExit)):
            # Ctrl+C, or the server stopping: one line is enough, no traceback
            print(f"Training stopped ({type(ex).__name__}). {run_path.name} ends with run_error.")
        else:
            traceback.print_exc()
        sys.exit(1)

    # ---- save best checkpoint (same payload as federated_simulation.py) -----
    best_model = fs.build_model(device=CPU, **model_cfg)
    set_parameters(best_model, strategy.best_params)
    torch.save({
        "model_state_dict": best_model.state_dict(),
        "model_type": args.model,
        "n_features": len(fs.FEATURE_COLS),
        "feature_cols": fs.FEATURE_COLS,
        "class_names": [str(c) for c in le.classes_],
        "mlp_hidden_sizes": mlp_hidden,
        "cnn_channels": cnn_channels,
        "seq_window_size": fs.SEQ_WINDOW_SIZE,
        "best_round": strategy.best_round,
        "n_rounds": args.rounds,
        "best_test_acc": strategy.best_acc,
        "final_test_acc": strategy.last_acc,
    }, model_path)
    joblib.dump({"scaler": scaler, "label_encoder": le},
                str(model_path).rsplit(".", 1)[0] + "_preprocess.pkl")

    log.write({"type": "run_end", "best_round": strategy.best_round,
               "best_acc": round(strategy.best_acc, 4), "model_path": rel_to_root(model_path)})
    log.close()
    update_latest = not (args.keep_latest and (runs_dir / "latest.jsonl").is_file())
    if update_latest:
        tmp_latest = runs_dir / ".latest.jsonl.tmp"
        shutil.copyfile(run_path, tmp_latest)
        os.replace(tmp_latest, runs_dir / "latest.jsonl")  # atomic for readers
        pre_path = Path(str(model_path).rsplit(".", 1)[0] + "_preprocess.pkl")
        shutil.copyfile(model_path, models_dir / "latest.pt")
        shutil.copyfile(pre_path, models_dir / "latest_preprocess.pkl")
    shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\nBest round {strategy.best_round} (global acc {strategy.best_acc:.4f}), "
          f"final {strategy.last_acc:.4f}")
    print(f"Saved {model_path} and its _preprocess.pkl"
          + (" (also copied to latest.pt)" if update_latest else ""))
    print(f"Events: {run_path}"
          + (" (copied to latest.jsonl)" if update_latest else " (latest.jsonl kept as it was)")
          + f". Total time {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
