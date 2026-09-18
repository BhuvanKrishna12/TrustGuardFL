"""
federated_simulation_flower.py
-------------------------------------------------------------------
Same FL-IDS system as federated_simulation.py, orchestrated via Flower
instead of the hand-rolled FedAvg loop. EVERY piece of domain logic --
data loading, feature prep, client partitioning, sequence building,
model architectures, the local training step, evaluation -- is imported
UNCHANGED from federated_simulation.py. This file only adds the Flower
orchestration layer around them: a NumPyClient wrapper, a client_fn that
maps a Flower partition-id to one of our region_cell clients, and a
custom Strategy subclass that reproduces the two things Flower doesn't
do out of the box but our hand-rolled loop already had: best-checkpoint
tracking (restoring the true best-test-accuracy round, not just the
final one) and the train/test/gap-per-round logging.

Flower's *legacy* simulation API (flwr.simulation.start_simulation) is
used here -- it is officially deprecated in favor of `flwr run`, but
still fully functional as of flwr 1.35.0, and is the simplest path to a
direct, one-machine comparison against the existing hand-rolled loop.
A production port to the newer ClientApp/ServerApp + `flwr run` workflow
is a reasonable next step once this comparison is validated, not
attempted here.

Usage (mirrors federated_simulation.py's CLI):
    python federated_simulation_flower.py labeled.csv --model cnn \
        --n-rounds 30 --max-total-rows 10000000
-------------------------------------------------------------------
"""
import sys
import copy
from collections import OrderedDict

import numpy as np
import torch
import flwr as fl
from flwr.common import Context

import federated_simulation as fs  # reuse ALL data/model/training logic unchanged


# ---------------------------------------------------------------------------
# Parameter <-> state_dict conversion (standard Flower <-> PyTorch bridge)
# ---------------------------------------------------------------------------

def get_parameters(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


# ---------------------------------------------------------------------------
# CLIENT
# ---------------------------------------------------------------------------

class FLIDSClient(fl.client.NumPyClient):
    """One region_cell client. Thin wrapper -- all it does is call
    fs.local_train / fs.evaluate, which are completely unaware they're
    being driven by Flower rather than the hand-rolled loop. This is the
    entire "port": everything else is unchanged."""

    def __init__(self, X_train, y_train, X_test, y_test, model_type,
                 n_features, n_classes, mlp_hidden_sizes, cnn_channels, class_weights):
        self.X_train, self.y_train = X_train, y_train
        self.X_test, self.y_test = X_test, y_test
        self.model_type = model_type
        self.n_features = n_features
        self.n_classes = n_classes
        self.mlp_hidden_sizes = mlp_hidden_sizes
        self.cnn_channels = cnn_channels
        self.class_weights = class_weights

    def _build(self):
        return fs.build_model(self.model_type, self.n_features, self.n_classes,
                               mlp_hidden_sizes=self.mlp_hidden_sizes, cnn_channels=self.cnn_channels)

    def get_parameters(self, config):
        return get_parameters(self._build())

    def fit(self, parameters, config):
        model = self._build()
        set_parameters(model, parameters)
        # Move class_weights onto whatever device THIS model actually
        # landed on (resolved fresh, per-process, by fs.build_model's
        # internal fs.DEVICE) -- never assume it matches the driver's
        # device, since this worker may have no GPU visibility at all.
        weights_on_device = self.class_weights.to(next(model.parameters()).device)
        _, train_acc = fs.local_train(model, self.X_train, self.y_train,
                                       fs.LOCAL_EPOCHS, weights_on_device)
        return get_parameters(model), len(self.X_train), {"train_acc": float(train_acc)}

    def evaluate(self, parameters, config):
        # Centralized evaluation (evaluate_fn below) is used for the
        # actual reported accuracy, matching the hand-rolled loop's
        # X_global_test convention -- client-side evaluate() here is a
        # no-op path, never called since fraction_evaluate=0.0 in main().
        model = self._build()
        set_parameters(model, parameters)
        acc, _ = fs.evaluate(model, self.X_test, self.y_test, [])
        return 1.0 - acc, len(self.X_test), {"accuracy": acc}


# ---------------------------------------------------------------------------
# STRATEGY -- adds best-checkpoint tracking + per-round logging to FedAvg
# ---------------------------------------------------------------------------

class TrackingFedAvg(fl.server.strategy.FedAvg):
    """Everything FedAvg already does, plus the two things our
    hand-rolled loop had that plain Flower doesn't: (1) tracks and keeps
    the best-test-accuracy round's parameters rather than only ever
    reporting the final round (this is what caught real overfitting in
    CNN earlier this project -- test accuracy peaking mid-training then
    declining), and (2) prints the same train/test/gap line per round
    for direct comparability against the hand-rolled loop's output."""

    def __init__(self, *args, payload_bytes, n_rounds, **kwargs):
        super().__init__(*args, **kwargs)
        self.payload_bytes = payload_bytes
        self.n_rounds = n_rounds
        self.history = {"round": [], "train_acc": [], "test_acc": [], "gap": [], "comm_mb": []}
        self.best_test_acc = -1.0
        self.best_round = 0
        self.best_parameters = None
        self.total_bytes_transferred = 0
        self._pending_train_acc = None

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if results:
            total_examples = sum(fit_res.num_examples for _, fit_res in results)
            self._pending_train_acc = sum(
                fit_res.metrics.get("train_acc", 0.0) * fit_res.num_examples for _, fit_res in results
            ) / total_examples
            # Each participating client sends weights up, receives them back
            # down -- 2x payload per client, same convention as the
            # hand-rolled loop's total_bytes_transferred.
            self.total_bytes_transferred += 2 * self.payload_bytes * len(results)
        return aggregated_parameters, aggregated_metrics

    def evaluate(self, server_round, parameters):
        result = super().evaluate(server_round, parameters)
        if result is None or server_round == 0 or self._pending_train_acc is None:
            return result  # round 0 = initial untrained params, nothing to log yet

        loss, metrics = result
        test_acc = metrics["accuracy"]
        train_acc = self._pending_train_acc
        gap = train_acc - test_acc
        gap_flag = "  <-- growing gap = overfitting" if gap > 0.15 else ""
        print(f"  Round {server_round:2d}/{self.n_rounds} -- train acc: {train_acc:.4f}  "
              f"test acc: {test_acc:.4f}  gap: {gap:+.4f}{gap_flag} -- "
              f"cumulative comm: {self.total_bytes_transferred / 1e6:.2f} MB")

        self.history["round"].append(server_round)
        self.history["train_acc"].append(train_acc)
        self.history["test_acc"].append(test_acc)
        self.history["gap"].append(gap)
        self.history["comm_mb"].append(self.total_bytes_transferred / 1e6)

        if test_acc > self.best_test_acc:
            self.best_test_acc = test_acc
            self.best_round = server_round
            self.best_parameters = copy.deepcopy(parameters)

        return result


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main(paths, max_total_rows=2_000_000, max_rows_per_region_class=None, chunksize=200_000,
         n_rounds=None, seq_window_size=None, seq_sort_col="time", preserve_aircraft_groups=False,
         exclude_classes=None, plot_output="training_curves_flower.png", model_type="gru",
         mlp_hidden_sizes=(64, 32), cnn_channels=(16, 32)):

    n_rounds = n_rounds if n_rounds is not None else fs.N_ROUNDS
    window_size = seq_window_size if seq_window_size is not None else fs.SEQ_WINDOW_SIZE

    print(f"Loading {len(paths)} file(s) ...")
    df = fs.load_data(paths, max_total_rows=max_total_rows,
                       max_rows_per_region_class=max_rows_per_region_class,
                       chunksize=chunksize, preserve_aircraft_groups=preserve_aircraft_groups)
    print(f"  {len(df):,} total rows")

    if seq_sort_col not in df.columns:
        print(f"  WARNING: --seq-sort-col '{seq_sort_col}' not found -- falling back to 'time'.")
        seq_sort_col = "time"

    if exclude_classes is None and model_type == "gru":
        exclude_classes = ["ghost_aircraft"]
        print("  GRU auto-excludes ghost_aircraft by default (structurally undetectable). "
              "Pass --exclude-classes '' explicitly to disable this.")
    if exclude_classes:
        before = len(df)
        df = df[~df["attack_type"].isin(exclude_classes)].reset_index(drop=True)
        print(f"  Excluding classes {exclude_classes}: dropped {before - len(df):,} rows.")

    X = fs.prep_features(df)
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    le = LabelEncoder()
    y = le.fit_transform(df["attack_type"].astype(str))
    class_names = le.classes_
    print(f"  Classes: {list(class_names)}")

    class_counts = np.bincount(y, minlength=len(class_names))
    class_weights_np = len(y) / (len(class_names) * class_counts)
    # Deliberately NOT moved to fs.DEVICE here -- this tensor is captured
    # into every FLIDSClient and shipped across Ray's actor process
    # boundary. If it were a CUDA tensor when constructed here (in the
    # driver process, which has full GPU access), Ray would try to
    # deserialize a CUDA tensor inside worker actors that don't have GPU
    # visibility (see client_resources below) -- exactly the crash this
    # avoids. Stays on CPU until FLIDSClient.fit() moves it onto whatever
    # device that WORKER's own model actually resolved to.
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32)

    scaler = StandardScaler()
    import pandas as pd
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    print("\nBuilding clients by region_cell ...")
    use_sequences = (model_type == "gru")
    clients = fs.build_clients(df, X_scaled, y, fs.MIN_CLIENT_ROWS, use_sequences=use_sequences,
                                window_size=window_size, sort_col=seq_sort_col)
    if len(clients) < 2:
        print("ERROR: fewer than 2 usable clients -- lower MIN_CLIENT_ROWS or provide more data.")
        sys.exit(1)

    client_names = list(clients.keys())
    n_features = len(fs.FEATURE_COLS)
    n_classes = len(class_names)

    X_global_test = np.concatenate([c["X_test"] for c in clients.values()])
    y_global_test = np.concatenate([c["y_test"] for c in clients.values()])

    template_model = fs.build_model(model_type, n_features, n_classes,
                                     mlp_hidden_sizes=mlp_hidden_sizes, cnn_channels=cnn_channels)
    payload_bytes = fs.model_size_bytes(template_model)
    print(f"\nModel: {model_type.upper()}")
    print(f"Model size: {payload_bytes / 1024:.1f} KB ({payload_bytes:,} bytes) -- this is what's "
          f"transmitted per client, per round")

    def client_fn(context: Context):
        partition_id = int(context.node_config["partition-id"])
        region = client_names[partition_id]
        c = clients[region]
        return FLIDSClient(
            c["X_train"], c["y_train"], c["X_test"], c["y_test"],
            model_type, n_features, n_classes, mlp_hidden_sizes, cnn_channels, class_weights,
        ).to_client()

    def evaluate_fn(server_round, parameters, config):
        model = fs.build_model(model_type, n_features, n_classes,
                                mlp_hidden_sizes=mlp_hidden_sizes, cnn_channels=cnn_channels)
        set_parameters(model, parameters)
        acc, _ = fs.evaluate(model, X_global_test, y_global_test, class_names)
        return 1.0 - acc, {"accuracy": acc}

    strategy = TrackingFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,  # client-side evaluate() unused -- centralized evaluate_fn instead
        min_fit_clients=len(clients),
        min_available_clients=len(clients),
        evaluate_fn=evaluate_fn,
        initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(template_model)),
        payload_bytes=payload_bytes,
        n_rounds=n_rounds,
    )

    print(f"\nRunning FedAvg (via Flower): {n_rounds} rounds, {fs.LOCAL_EPOCHS} local "
          f"epochs/round, {len(clients)} clients (100% participation)\n")

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(clients),
        config=fl.server.ServerConfig(num_rounds=n_rounds),
        strategy=strategy,
        # Explicitly mask GPU visibility from every worker actor. These
        # models are small enough (12-18 KB) that GPU offers negligible
        # benefit -- established earlier when profiling for Raspberry Pi
        # deployment -- and forcing CPU-only here sidesteps the entire
        # class of CUDA-tensor-crosses-Ray-actor-boundary bugs, not just
        # the one instance already fixed above.
        client_resources={"num_cpus": 1, "num_gpus": 0},
        ray_init_args={"include_dashboard": False, "logging_level": 40},
    )

    final_acc = strategy.history["test_acc"][-1] if strategy.history["test_acc"] else float("nan")
    print(f"\nBest checkpoint: round {strategy.best_round}/{n_rounds} "
          f"(test acc {strategy.best_test_acc:.4f}) vs. final round {n_rounds} "
          f"(test acc {final_acc:.4f})")

    final_model = fs.build_model(model_type, n_features, n_classes,
                                  mlp_hidden_sizes=mlp_hidden_sizes, cnn_channels=cnn_channels)
    if strategy.best_parameters is not None:
        set_parameters(final_model, fl.common.parameters_to_ndarrays(strategy.best_parameters))

    fs.plot_training_curves(strategy.history, plot_output, best_round=strategy.best_round)

    print("\n=== Final per-class recall (global model, pooled test set) ===")
    _, per_class = fs.evaluate(final_model, X_global_test, y_global_test, class_names)
    for cname, recall in per_class.items():
        print(f"  {cname:<16s} {recall:.4f}")

    print(f"\n=== Communication summary ===")
    print(f"  Model size per transfer: {payload_bytes / 1024:.1f} KB")
    print(f"  Total transferred over {n_rounds} rounds: "
          f"{strategy.total_bytes_transferred / 1e6:.2f} MB")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="FL-IDS FedAvg simulation, orchestrated via Flower.")
    p.add_argument("paths", nargs="+")
    p.add_argument("--max-total-rows", type=int, default=2_000_000)
    p.add_argument("--max-rows-per-region-class", type=int, default=None)
    p.add_argument("--load-chunksize", type=int, default=200_000)
    p.add_argument("--n-rounds", type=int, default=None)
    p.add_argument("--seq-window-size", type=int, default=None)
    p.add_argument("--seq-sort-col", type=str, default="time", choices=["time", "lastposupdate"])
    p.add_argument("--exclude-classes", type=str, default=None)
    p.add_argument("--plot-output", type=str, default="training_curves_flower.png")
    p.add_argument("--model", type=str, default="gru", choices=["mlp", "cnn", "gru"])
    p.add_argument("--mlp-hidden-sizes", type=str, default="64,32")
    p.add_argument("--cnn-channels", type=str, default="16,32")
    args = p.parse_args()

    if args.exclude_classes is not None:
        exclude_classes = [c.strip() for c in args.exclude_classes.split(",") if c.strip()]
    else:
        exclude_classes = None
    mlp_hidden_sizes = tuple(int(h.strip()) for h in args.mlp_hidden_sizes.split(","))
    cnn_channels = tuple(int(c.strip()) for c in args.cnn_channels.split(","))

    main(args.paths, max_total_rows=args.max_total_rows,
         max_rows_per_region_class=args.max_rows_per_region_class,
         chunksize=args.load_chunksize, n_rounds=args.n_rounds,
         seq_window_size=args.seq_window_size, seq_sort_col=args.seq_sort_col,
         exclude_classes=exclude_classes, plot_output=args.plot_output,
         model_type=args.model, mlp_hidden_sizes=mlp_hidden_sizes, cnn_channels=cnn_channels)
