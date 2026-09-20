"""
Federated Learning Prototype for the OpenSky FL-IDS project
=============================================================
Simulates a real federated setup: each geographic region (region_cell)
acts as an independent "client" (standing in for a ground receiver /
group of receivers), holding only its own local data. A shared model is
trained via FedAvg -- clients never share raw data, only model weight
updates, which are averaged on a central "server" each round.

This is a SIMULATION on a single machine (all "clients" are just data
partitions in memory) -- it demonstrates the FL mechanics and lets you
measure detection performance + communication cost, without needing
actual distributed infrastructure.

Usage:
    python federated_simulation.py combined_labeled.csv
    python federated_simulation.py day1_labeled.csv day2_labeled.csv ...
"""

import sys
import copy
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib
matplotlib.use("Agg")  # headless -- WSL/servers have no display to render to
import matplotlib.pyplot as plt

RNG_SEED = 42
torch.manual_seed(RNG_SEED)
np.random.seed(RNG_SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- FL simulation knobs -----------------------------------------------
MIN_CLIENT_ROWS = 3000       # clients with fewer rows than this are dropped
                              # (too little local data to train anything
                              # meaningful -- see our earlier region-density
                              # analysis: below ~3k rows, a client is mostly
                              # noise)
N_ROUNDS = 15                 # federated communication rounds
LOCAL_EPOCHS = 2              # local training epochs per client per round
CLIENT_FRACTION = 1.0         # fraction of clients sampled each round
                              # (1.0 = all clients participate every round;
                              # lower this to simulate clients dropping in/
                              # out, closer to real bandwidth-constrained FL)
BATCH_SIZE = 512
LR = 1e-3

# --- Sequence model knobs (GRU only -- MLP/CNN ignore these) ------------
SEQ_WINDOW_SIZE = 10   # messages of chronological history per training
                        # example. Aircraft with fewer than this many rows
                        # in a given region get dropped entirely from
                        # sequence-building -- see build_sequences.
GRU_HIDDEN_SIZE = 32

FEATURE_COLS = [
    "implied_speed_mps",
    "speed_consistency_delta",
    "vrate_consistency_delta",
    "inter_arrival_s",
    "turn_rate_deg_s",
    "track_message_count",
    "is_physically_implausible",
    "squawk_changed",
    "icao24_valid_format",
    "is_stale",
    "region_avg_inter_arrival",
    "is_low_coverage_region",
    "is_climbing_or_descending",
]
# NOTE: callsign_valid_format was removed from this list -- the pipeline
# now drops raw callsign entirely at load time (see DROP_COLS in
# opensky_fl_ids_pipeline.py) for memory, and no longer derives a
# validity flag from it. Using the old FEATURE_COLS here would crash
# with a KeyError on any file produced by the current pipeline.


# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------

class SimpleIDSNet(nn.Module):
    """MLP with configurable hidden layer sizes. Default (64, 32) is
    deliberately lightweight for FedAvg's per-round communication payload.
    Widen this (e.g. hidden_sizes=(128, 64)) if a smaller model's accuracy
    plateaus below what other architectures reach on the SAME data with
    the SAME training budget -- that's a signal of a genuine capacity
    limit, not a training-time problem, and is exactly the case that
    justifies a bigger network. Check the communication-cost tradeoff
    (model_size_bytes) when you do -- a wider MLP still needs to stay
    cheap to be worth using in FL over a fixed architecture like CNN."""

    def __init__(self, n_features: int, n_classes: int, hidden_sizes=(64, 32)):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

    
class SimpleCNNnet(nn.Module):
    """Conv1d over the flat feature vector, with configurable channel
    sizes. Default (16, 32) -- 2 conv layers -- is deliberately small.
    Passing more values (e.g. '32,64') widens each layer; passing more
    comma-separated entries (e.g. '16,32,64') adds another conv layer
    (deeper, not just wider) -- same distinction as SimpleIDSNet's
    hidden_sizes. Widen/deepen this if CNN's accuracy plateaus below
    what the row-count fix alone achieves, the same capacity-limit logic
    used to test a bigger MLP."""

    def __init__(self, n_features, n_classes, channels=(16, 32)):
        super().__init__()
        layers = []
        in_ch = 1
        for out_ch in channels:
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1), nn.ReLU()]
            in_ch = out_ch
        self.conv = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_ch * n_features, n_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.flatten(1)
        return self.classifier(x)


class SimpleGRUNet(nn.Module):
    """GRU over a chronological window of an aircraft's recent messages --
    unlike SimpleIDSNet/SimpleCNNnet, this is the only architecture that
    can actually see an aircraft's HISTORY, which is what replay attacks
    (timing inconsistent with recent messages) fundamentally need to be
    detected reliably. Row-independent models structurally cannot fix
    this no matter how they're tuned -- confirmed empirically, not just in
    theory: the CNN swap left replay recall essentially unchanged (~43%)
    while every other class improved, exactly the signature of a model
    that still can't see sequence context."""

    def __init__(self, n_features, n_classes, hidden_size=GRU_HIDDEN_SIZE):
        super().__init__()
        self.gru = nn.GRU(input_size=n_features, hidden_size=hidden_size, batch_first=True)
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # x: (batch, seq_len, n_features)
        _, h_n = self.gru(x)        # h_n: (1, batch, hidden_size) -- final hidden state
        return self.classifier(h_n.squeeze(0))

def model_size_bytes(model: nn.Module) -> int:
    return sum(p.numel() * p.element_size() for p in model.parameters())


# ---------------------------------------------------------------------------
# DATA PREP
# ---------------------------------------------------------------------------

def _cap_bucket_preserving_aircraft(combined_key, cap, seed, per_aircraft_cap=20):
    """
    Downsamples a (region, attack_type) bucket to roughly `cap` rows,
    keeping each SURVIVING aircraft's rows CONTIGUOUS (not a random
    scattered subset) -- but bounding how many rows any single aircraft
    can contribute (per_aircraft_cap), so one aircraft with an unusually
    large chunk (e.g. lots of benign rows) can't consume the whole
    budget and starve out every other aircraft.

    This two-sided design matters: naive whole-group keeping (no per-
    aircraft cap) was tested and found to make sequence-window dropout
    WORSE, not better -- because attack labels are assigned per ROW, not
    per aircraft (see build_sequences), so one aircraft's history is
    already split across up to 6 buckets, and greedily keeping a few
    aircrafts' full chunks starved out breadth (how many DISTINCT
    aircraft get any representation at all) for the sake of depth on a
    handful of aircraft. Bounding per-aircraft contribution keeps both:
    breadth (more distinct aircraft represented) AND intactness (no
    aircraft's kept rows are a randomly fragmented, out-of-order subset).
    """
    if "icao24" not in combined_key.columns:
        return combined_key.sample(n=cap, random_state=seed)

    rng = np.random.default_rng(seed)
    icao_ids = np.array(combined_key["icao24"].unique(), dtype=object)
    rng.shuffle(icao_ids)

    kept_groups = []
    running_total = 0
    for icao in icao_ids:
        if running_total >= cap:
            break
        group = combined_key[combined_key["icao24"] == icao]
        if len(group) > per_aircraft_cap:
            group = group.iloc[:per_aircraft_cap]  # keep a CONTIGUOUS slice, not scattered rows
        kept_groups.append(group)
        running_total += len(group)

    return pd.concat(kept_groups, ignore_index=True) if kept_groups else combined_key.iloc[0:0]


def load_data(paths, max_total_rows=2_000_000, max_rows_per_region_class=None,
              chunksize=200_000, seed=RNG_SEED, preserve_aircraft_groups=False):
    """
    Streams the labeled CSV(s) in chunks instead of one big pd.read_csv --
    at full-day+ scale (tens of millions of rows) a naive full read can
    use several times the file's on-disk size in RAM.

    IMPORTANT: capping rows per (region_cell, attack_type) bucket alone
    does NOT bound total memory -- if the file spans hundreds or
    thousands of distinct regions (a full day of GLOBAL air traffic
    easily can), per-bucket-cap x huge-region-count can still add up to
    more memory than a per-bucket cap alone would suggest, even though
    each individual bucket looks bounded. This is what caused an OOM
    kill in practice on real full-day data.

    Fixed with two passes:
      1. Cheap first pass over just the region_cell column to count how
         many distinct regions actually exist.
      2. That count is used to auto-derive a per-(region,class) cap that
         keeps the GLOBAL total near max_total_rows regardless of region
         count (unless max_rows_per_region_class is explicitly set,
         which overrides auto-sizing). A hard global stop is ALSO
         enforced as a safety net during the real streaming pass, in
         case region/class skew is worse than the average-case formula
         assumes.

    preserve_aircraft_groups=False (default): EXPERIMENTAL flag, tested
    and found to make sequence dropout WORSE on realistic class-imbalanced
    data -- see _cap_bucket_preserving_aircraft's docstring for why. Left
    available (off by default) rather than deleted, since the tradeoff
    could reverse under different class balance; don't enable it without
    checking the printed dropout line actually improves for your data.
    """
    # icao24 + time are needed to build chronological per-aircraft
    # sequences for the GRU (see build_sequences) -- they weren't loaded
    # at all before, since row-independent models (MLP/CNN) never
    # needed them.
    # icao24 + time are needed to build chronological per-aircraft
    # sequences for the GRU (see build_sequences). lastposupdate is ALSO
    # loaded, defensively, to support sorting sequences by a timestamp
    # field the replay attack does NOT manipulate (see build_sequences'
    # sort_col parameter) -- replay only shifts 'time' and 'lastcontact',
    # so sorting by 'time' lets a replayed row settle into a position
    # consistent with its OWN fake timestamp, hiding the very
    # inconsistency that should make it detectable.
    usecols = FEATURE_COLS + ["region_cell", "attack_type", "icao24", "time", "lastposupdate"]

    # ---- Pass 1: count distinct regions (cheap -- one column, chunked) ----
    distinct_regions = set()
    for path in paths:
        for chunk in pd.read_csv(path, usecols=["region_cell"], chunksize=chunksize):
            distinct_regions.update(chunk["region_cell"].unique())
    n_regions = max(1, len(distinct_regions))
    n_classes_guess = 6  # benign + 5 attack types -- used only for cap sizing

    if max_rows_per_region_class is None:
        computed_cap = max(500, max_total_rows // (n_regions * n_classes_guess))
        print(f"  Found {n_regions:,} distinct region_cell(s) -- auto-sizing per-region/class "
              f"cap to {computed_cap:,} rows (targeting a global total near {max_total_rows:,})")
    else:
        computed_cap = max_rows_per_region_class
        print(f"  Found {n_regions:,} distinct region_cell(s); using explicit cap "
              f"of {computed_cap:,} rows per region/class")

    # ---- Pass 2: real streaming load with reservoir sampling ----
    reservoirs = {}  # (region, attack_type) -> [DataFrame]
    total_rows_seen = 0
    hit_global_cap = False

    for path in paths:
        if hit_global_cap:
            break
        header = pd.read_csv(path, nrows=0)
        cols_here = [c for c in usecols if c in header.columns]
        missing = set(usecols) - set(cols_here)
        if missing:
            print(f"  WARNING: {path} is missing expected columns: {missing}")

        reader = pd.read_csv(path, usecols=cols_here, chunksize=chunksize, low_memory=False)
        for chunk in reader:
            total_rows_seen += len(chunk)
            num_cols = [c for c in FEATURE_COLS if c in chunk.columns
                        and pd.api.types.is_numeric_dtype(chunk[c])]
            if num_cols:
                chunk[num_cols] = chunk[num_cols].astype("float32")

            for (region, atk), group in chunk.groupby(["region_cell", "attack_type"]):
                key = (region, atk)
                reservoirs.setdefault(key, [])
                reservoirs[key].append(group)
                combined_key = pd.concat(reservoirs[key], ignore_index=True)
                if len(combined_key) > computed_cap:
                    if preserve_aircraft_groups:
                        combined_key = _cap_bucket_preserving_aircraft(combined_key, computed_cap, seed)
                    else:
                        combined_key = combined_key.sample(n=computed_cap, random_state=seed)
                reservoirs[key] = [combined_key]

            # Hard global safety net -- recomputed after every chunk. Cheap:
            # bounded by n_keys (region x class combos), not by row count.
            total_kept = sum(len(v[0]) for v in reservoirs.values())
            if total_kept >= max_total_rows:
                print(f"  Reached global cap of {max_total_rows:,} rows after "
                      f"{total_rows_seen:,} raw rows seen -- stopping early "
                      f"(remaining files/chunks skipped).")
                hit_global_cap = True
                break

    df = pd.concat(
        [pd.concat(parts, ignore_index=True) for parts in reservoirs.values()],
        ignore_index=True,
    )
    print(f"  Streamed {total_rows_seen:,} raw rows -> sampled down to {len(df):,} rows "
          f"across {n_regions:,} regions")
    return df


def prep_features(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].copy()
    bool_cols = ["is_physically_implausible", "squawk_changed",
                 "icao24_valid_format", "is_stale",
                 "is_low_coverage_region", "is_climbing_or_descending"]
    for col in bool_cols:
        X[col] = X[col].astype(float)
    X = X.fillna(-1)
    return X


def build_sequences(df_region: pd.DataFrame, feature_cols, window_size=SEQ_WINDOW_SIZE,
                     label_col="_label_encoded", sort_col="time"):
    """
    Builds chronological per-aircraft sequences from one region's rows.
    Each output example is a window of `window_size` consecutive messages
    (sorted by `sort_col`) from ONE aircraft, labeled with the LAST
    message's attack_type -- "given this aircraft's recent history, is the
    newest message an attack?". This is what actually gives a model
    visibility into an aircraft's history, unlike SimpleIDSNet/
    SimpleCNNnet which only ever see one row at a time.

    sort_col matters more than it looks. Default is "time", which the
    replay attack DIRECTLY manipulates (it shifts 'time' and
    'lastcontact' backward). Sorting by the very field an attack
    manipulates lets a replayed row settle into whatever position its
    FAKE timestamp implies -- which can make the row look internally
    consistent with its (wrong) neighbors, hiding the inconsistency that
    should make it detectable. Passing sort_col="lastposupdate" instead
    (a field replay does NOT touch) keeps each row in its TRUE
    chronological position, so a replayed row's manipulated features
    show up as a genuine anomaly relative to its real neighbors, rather
    than being absorbed into a self-consistent fake ordering.

    Aircraft with fewer than window_size rows in this region are dropped
    entirely (not enough history to build even one window) -- a real,
    named tradeoff: a short-lived pass through a region contributes
    nothing to a sequence model, unlike row-independent models which use
    every row regardless of how much history exists.

    NOTE: sequences are built PER REGION (this function only ever sees
    one region_cell's rows, called once per client in build_clients).
    An aircraft's flight crossing a region boundary gets its sequence
    artificially cut off at the boundary -- a known simplification, not
    a bug; building sequences globally before region-partitioning would
    be more correct but is a bigger structural change.

    Returns (sequences, labels, n_aircraft_seen, n_aircraft_dropped):
      sequences -- array of shape (n_windows, window_size, n_features)
      labels    -- array of shape (n_windows,), the last-message label
    """
    sequences, labels = [], []
    n_aircraft_seen = 0
    n_aircraft_dropped = 0
    df_region = df_region.sort_values(["icao24", sort_col])

    for icao, group in df_region.groupby("icao24"):
        n_aircraft_seen += 1
        if len(group) < window_size:
            n_aircraft_dropped += 1
            continue
        feats = group[feature_cols].to_numpy(dtype=np.float32)
        labs = group[label_col].to_numpy()
        for i in range(window_size - 1, len(group)):
            sequences.append(feats[i - window_size + 1: i + 1])
            labels.append(labs[i])

    if len(sequences) == 0:
        return (np.empty((0, window_size, len(feature_cols)), dtype=np.float32),
                np.array([]), n_aircraft_seen, n_aircraft_dropped)
    return np.stack(sequences), np.array(labels), n_aircraft_seen, n_aircraft_dropped


def build_clients(df: pd.DataFrame, X: pd.DataFrame, y: np.ndarray,
                   min_rows: int, use_sequences: bool = False,
                   window_size: int = SEQ_WINDOW_SIZE, sort_col: str = "time"):
    """
    Partitions (X, y) by region_cell into per-client train/test splits.
    Drops clients below min_rows -- see MIN_CLIENT_ROWS comment.

    If use_sequences=True (GRU), builds chronological per-aircraft
    windows via build_sequences() instead of treating rows independently.
    CRITICAL: the non-sequence path shuffles ROWS before splitting
    train/test, which is fine when rows are independent -- but doing that
    BEFORE windowing would glue together unrelated timestamps into fake
    sequences and destroy the entire point of a GRU. The sequence path
    instead shuffles complete SEQUENCES after windowing -- each window's
    internal chronological order stays intact; only the order of windows
    relative to each other is randomized for the train/test split.

    sort_col: which timestamp field build_sequences uses to order each
    aircraft's messages -- see build_sequences' docstring for why this
    is a real choice, not a cosmetic one (replay manipulates 'time' but
    not 'lastposupdate').

    Returns a dict: region -> dict(X_train, y_train, X_test, y_test)
    """
    clients = {}
    dropped = 0
    total_aircraft_seen = 0
    total_aircraft_dropped = 0

    for region, idx in df.groupby("region_cell").groups.items():
        idx = np.array(idx)
        if len(idx) < min_rows:
            dropped += 1
            continue

        if use_sequences:
            region_df = X.iloc[idx].copy()  # SCALED features -- this is the actual model input
            region_df["icao24"] = df["icao24"].iloc[idx].values
            region_df[sort_col] = df[sort_col].iloc[idx].values
            region_df["_label_encoded"] = y[idx]

            seqs, seq_labels, n_seen, n_dropped_aircraft = build_sequences(
                region_df, list(X.columns), window_size=window_size, sort_col=sort_col,
            )
            total_aircraft_seen += n_seen
            total_aircraft_dropped += n_dropped_aircraft

            if len(seqs) < min_rows:
                # Raw row count cleared min_rows, but too many aircraft in
                # this region had fewer than window_size rows each, so not
                # enough actual SEQUENCES survived -- drop this client too.
                dropped += 1
                continue

            rng = np.random.default_rng(RNG_SEED)
            shuffled = rng.permutation(len(seqs))  # shuffles SEQUENCES, not rows within them
            split = int(len(shuffled) * 0.8)
            train_idx, test_idx = shuffled[:split], shuffled[split:]
            clients[region] = dict(
                X_train=seqs[train_idx].astype(np.float32),
                y_train=seq_labels[train_idx],
                X_test=seqs[test_idx].astype(np.float32),
                y_test=seq_labels[test_idx],
            )
        else:
            rng = np.random.default_rng(RNG_SEED)
            shuffled = rng.permutation(idx)
            split = int(len(shuffled) * 0.8)
            train_idx, test_idx = shuffled[:split], shuffled[split:]
            clients[region] = dict(
                X_train=X.iloc[train_idx].values.astype(np.float32),
                y_train=y[train_idx],
                X_test=X.iloc[test_idx].values.astype(np.float32),
                y_test=y[test_idx],
            )

    print(f"  {len(clients)} clients built (min {min_rows:,} rows each), "
          f"{dropped} region(s) dropped for insufficient data")
    if use_sequences and total_aircraft_seen > 0:
        print(f"  Sequence building: {total_aircraft_dropped:,}/{total_aircraft_seen:,} "
              f"aircraft ({total_aircraft_dropped/total_aircraft_seen:.1%}) had fewer than "
              f"{window_size} rows in their region and were dropped entirely (no sequence "
              f"could be built from them)")
    return clients


# ---------------------------------------------------------------------------
# FEDAVG
# ---------------------------------------------------------------------------

def local_train(model: nn.Module, X_train, y_train, epochs: int, class_weights):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long),
    )
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    # Training accuracy AFTER local training, on this client's own training
    # data -- needed to see overfitting at all. Without this, only test
    # accuracy is ever visible, and a growing train/test gap (the actual
    # overfitting signal) is invisible by construction, not just hard to see.
    model.eval()
    with torch.no_grad():
        xb = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        yb = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
        preds = model(xb).argmax(dim=1)
        train_acc = (preds == yb).float().mean().item()

    return model.state_dict(), train_acc


def federated_average(state_dicts, weights):
    """Weighted average of client state_dicts, weighted by each client's
    local dataset size (standard FedAvg -- clients with more data have
    proportionally more influence on the global update)."""
    total = sum(weights)
    avg = copy.deepcopy(state_dicts[0])
    for key in avg:
        avg[key] = sum(
            sd[key].float() * (w / total) for sd, w in zip(state_dicts, weights)
        )
    return avg


@torch.no_grad()
def evaluate(model: nn.Module, X_test, y_test, class_names):
    model.eval()
    xb = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
    preds = model(xb).argmax(dim=1).cpu().numpy()
    acc = (preds == y_test).mean()

    per_class = {}
    for c_idx, c_name in enumerate(class_names):
        mask = y_test == c_idx
        if mask.sum() == 0:
            continue
        per_class[c_name] = (preds[mask] == c_idx).mean()
    return acc, per_class


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def plot_training_curves(history, output_path, best_round=None):
    """
    Saves a two-panel PNG: left = train vs. test accuracy per round (the
    actual overfitting-diagnosis plot -- a widening gap between the two
    lines late in training is what "overfitting" looks like visually,
    not just a single accuracy number), right = cumulative communication
    cost per round (the tradeoff every accuracy gain above is actually
    costing you in bandwidth). If best_round is given, marks it on the
    accuracy panel -- this is the checkpoint the model actually got
    restored to, which can be well before the final round.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(history["round"], history["train_acc"], label="Train accuracy",
              color="#2a78d6", marker="o", markersize=3)
    ax1.plot(history["round"], history["test_acc"], label="Test accuracy",
              color="#e05f2a", marker="o", markersize=3)
    if best_round is not None and best_round in history["round"]:
        ax1.axvline(best_round, color="#1B7A43", linestyle="--", linewidth=1,
                     label=f"Best checkpoint (round {best_round})")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Train vs. Test Accuracy per Round")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(history["round"], history["comm_mb"], color="#1B7A43", marker="o", markersize=3)
    ax2.set_xlabel("Round")
    ax2.set_ylabel("Cumulative Communication (MB)")
    ax2.set_title("Communication Cost per Round")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved training curves: {output_path}")


def build_model(model_type, n_features, n_classes, mlp_hidden_sizes=(64, 32), cnn_channels=(16, 32),
                 device=None):
    """
    Single source of truth for constructing a model given an architecture
    choice -- used by main()'s hand-rolled FedAvg loop, by
    federated_simulation_flower.py's Flower client/server code, AND by
    inference_server.py's demo backend, so all three paths build
    IDENTICAL architectures from the same config.

    device=None (default) uses this module's own DEVICE constant
    (cuda if available), matching every existing caller's behavior
    exactly. Pass an explicit device (e.g. torch.device("cpu")) to
    override -- inference_server.py does this deliberately, since a
    live demo shouldn't depend on whatever GPU happens to be on the
    demo machine.
    """
    if device is None:
        device = DEVICE
    if model_type == "gru":
        model = SimpleGRUNet(n_features, n_classes)
    elif model_type == "cnn":
        model = SimpleCNNnet(n_features, n_classes, channels=cnn_channels)
    elif model_type == "mlp":
        model = SimpleIDSNet(n_features, n_classes, hidden_sizes=mlp_hidden_sizes)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r} (expected 'mlp', 'cnn', or 'gru')")
    return model.to(device)


def main(paths, max_total_rows=2_000_000, max_rows_per_region_class=None, chunksize=200_000,
         n_rounds=None, seq_window_size=None, seq_sort_col="time", preserve_aircraft_groups=False,
         exclude_classes=None, plot_output="training_curves.png", model_type="gru",
         mlp_hidden_sizes=(64, 32), cnn_channels=(16, 32), save_model_path=None):
    # Allow CLI overrides of module-level defaults without editing the file --
    # both were hardcoded constants before, which meant testing a different
    # window size or round count required a code edit each time.
    global N_ROUNDS
    if n_rounds is not None:
        N_ROUNDS = n_rounds
    window_size = seq_window_size if seq_window_size is not None else SEQ_WINDOW_SIZE
    print(f"Loading {len(paths)} file(s) ...")
    df = load_data(paths, max_total_rows=max_total_rows,
                    max_rows_per_region_class=max_rows_per_region_class,
                    chunksize=chunksize, preserve_aircraft_groups=preserve_aircraft_groups)
    print(f"  {len(df):,} total rows")

    # Fall back to 'time' if the requested sort column isn't actually
    # present in this file (e.g. an older labeled export without
    # lastposupdate) -- fail soft with a clear warning, not a KeyError
    # deep inside build_clients.
    if seq_sort_col not in df.columns:
        print(f"  WARNING: --seq-sort-col '{seq_sort_col}' not found in this file's columns "
              f"-- falling back to 'time'.")
        seq_sort_col = "time"

    # Default exclude_classes to ['ghost_aircraft'] ONLY for GRU, and only
    # when the caller hasn't explicitly specified something -- ghost_aircraft
    # is structurally undetectable by GRU (see SimpleGRUNet's docstring),
    # but MLP/CNN catch it at 99.6-99.8% recall. Applying the same
    # exclusion to all models regardless of model_type would silently
    # throw away a real capability MLP/CNN actually have.
    if exclude_classes is None and model_type == "gru":
        exclude_classes = ["ghost_aircraft"]
        print(f"  GRU auto-excludes ghost_aircraft by default (structurally undetectable -- "
              f"see SimpleGRUNet docstring). Pass --exclude-classes '' explicitly to disable this.")

    if exclude_classes:
        before = len(df)
        df = df[~df["attack_type"].isin(exclude_classes)].reset_index(drop=True)
        print(f"  Excluding classes {exclude_classes}: dropped {before - len(df):,} rows "
              f"({(before - len(df)) / before:.1%} of loaded data). Remaining classes: "
              f"{sorted(df['attack_type'].unique())}")

    X = prep_features(df)
    le = LabelEncoder()
    y = le.fit_transform(df["attack_type"].astype(str))
    class_names = le.classes_
    print(f"  Classes: {list(class_names)}")

    # Scale features globally (fit on all data before partitioning -- in a
    # real deployment you'd instead fit on a public reference sample, or
    # use a federated-safe scaling scheme, since fitting on pooled data is
    # itself a mild simplification for this prototype).
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Class weighting to counter severe imbalance (benign heavily
    # dominates). Computed once from the GLOBAL label distribution --
    # in a real deployment the server would aggregate per-class counts
    # from each client (small integers, not raw data) rather than seeing
    # labels directly, so this is a reasonable simplification for the
    # prototype. Same "balanced" formula sklearn uses:
    # weight_c = n_samples / (n_classes * count_c)
    class_counts = np.bincount(y, minlength=len(class_names))
    class_weights_np = len(y) / (len(class_names) * class_counts)
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32).to(DEVICE)
    print(f"  Class weights (balanced): "
          f"{dict(zip(class_names, class_weights_np.round(2)))}")

    print("\nBuilding clients by region_cell ...")
    # USE_SEQUENCES and the model class must always move together --
    # SimpleGRUNet needs real sequences (shape: batch, seq_len, features);
    # SimpleIDSNet/SimpleCNNnet need flat rows (shape: batch, features).
    # Feeding either the wrong shape crashes on the first forward pass.
    # Deriving USE_SEQUENCES directly from model_type (rather than two
    # separately-set variables that could drift apart) makes this
    # impossible to get out of sync.
    USE_SEQUENCES = (model_type == "gru")
    clients = build_clients(df, X_scaled, y, MIN_CLIENT_ROWS, use_sequences=USE_SEQUENCES,
                             window_size=window_size, sort_col=seq_sort_col)
    if len(clients) < 2:
        print("ERROR: fewer than 2 usable clients -- lower MIN_CLIENT_ROWS "
              "or provide more data (more days).")
        sys.exit(1)

    # Global held-out test set: pool every client's test split, so we
    # measure whether the FEDERATED model generalizes across all regions,
    # not just the ones it happened to train on well.
    X_global_test = np.concatenate([c["X_test"] for c in clients.values()])
    y_global_test = np.concatenate([c["y_test"] for c in clients.values()])

    global_model = build_model(model_type, len(FEATURE_COLS), len(class_names),
                                mlp_hidden_sizes=mlp_hidden_sizes, cnn_channels=cnn_channels)
    print(f"\nModel: {model_type.upper()}")
    payload_bytes = model_size_bytes(global_model)
    print(f"Model size: {payload_bytes / 1024:.1f} KB "
          f"({payload_bytes:,} bytes) -- this is what's transmitted "
          f"per client, per round")

    total_bytes_transferred = 0
    client_names = list(clients.keys())
    rng = np.random.default_rng(RNG_SEED)

    history = {"round": [], "train_acc": [], "test_acc": [], "gap": [], "comm_mb": []}

    # Track the best-test-accuracy round and restore it at the end, rather
    # than reporting whatever the LAST round happens to look like. This
    # matters concretely: a real run showed test accuracy peak at round 33
    # (86.99%) then DECLINE to 86.01% by round 50 while train accuracy kept
    # climbing the whole time -- textbook overfitting, and without this,
    # the script would have reported the worse, more-overfit round 50
    # model as the result, at a third more communication cost for the
    # privilege of being worse.
    best_test_acc = -1.0
    best_round = 0
    best_state = None

    print(f"\nRunning FedAvg: {N_ROUNDS} rounds, {LOCAL_EPOCHS} local "
          f"epochs/round, {len(clients)} clients "
          f"({CLIENT_FRACTION:.0%} participation)\n")

    for rnd in range(1, N_ROUNDS + 1):
        n_participants = max(1, int(len(client_names) * CLIENT_FRACTION))
        participants = rng.choice(client_names, size=n_participants, replace=False)

        local_states, weights = [], []
        local_train_accs = []
        for cname in participants:
            local_model = copy.deepcopy(global_model).to(DEVICE)
            c = clients[cname]
            state, train_acc = local_train(local_model, c["X_train"], c["y_train"], LOCAL_EPOCHS, class_weights)
            local_states.append(state)
            weights.append(len(c["X_train"]))
            local_train_accs.append(train_acc)
            # Each participating client sends its weights up, and receives
            # the new global weights back down -- 2x the payload per client.
            total_bytes_transferred += 2 * payload_bytes

        new_state = federated_average(local_states, weights)
        global_model.load_state_dict(new_state)

        # Weighted average training accuracy across participating clients --
        # weighted by each client's local training set size, same convention
        # federated_average already uses for the model weights themselves.
        # This is each client's OWN post-local-training accuracy on its OWN
        # training data, BEFORE averaging into the new global model -- the
        # standard notion of "training accuracy" in an FL setting.
        total_train_rows = sum(weights)
        avg_train_acc = sum(a * w for a, w in zip(local_train_accs, weights)) / total_train_rows

        acc, _ = evaluate(global_model, X_global_test, y_global_test, class_names)
        gap = avg_train_acc - acc
        gap_flag = "  <-- growing gap = overfitting" if gap > 0.15 else ""
        print(f"  Round {rnd:2d}/{N_ROUNDS} -- train acc: {avg_train_acc:.4f}  test acc: {acc:.4f}  "
              f"gap: {gap:+.4f}{gap_flag} -- cumulative comm: {total_bytes_transferred / 1e6:.2f} MB")

        if acc > best_test_acc:
            best_test_acc = acc
            best_round = rnd
            best_state = {k: v.detach().clone() for k, v in global_model.state_dict().items()}

        history["round"].append(rnd)
        history["train_acc"].append(avg_train_acc)
        history["test_acc"].append(acc)
        history["gap"].append(gap)
        history["comm_mb"].append(total_bytes_transferred / 1e6)

    if best_state is not None:
        global_model.load_state_dict(best_state)
    final_acc = history["test_acc"][-1] if history["test_acc"] else float("nan")
    print(f"\nBest checkpoint: round {best_round}/{N_ROUNDS} (test acc {best_test_acc:.4f}) "
          f"vs. final round {N_ROUNDS} (test acc {final_acc:.4f}) -- model restored to best "
          f"checkpoint for all evaluation below.")

    # Persist the best checkpoint to disk -- everything downstream (a
    # standalone inference script, deploying to a Pi, etc.) needs the model
    # weights AND the exact preprocessing that produced the features it was
    # trained on. Saving the model alone without the scaler/encoder is not
    # enough: at inference time raw features must go through the SAME
    # StandardScaler.transform() and the label indices must map back through
    # the SAME LabelEncoder.classes_, or predictions will be silently wrong.
    if save_model_path:
        model_payload = {
            "model_state_dict": best_state if best_state is not None
                                 else global_model.state_dict(),
            "model_type": model_type,
            "n_features": len(FEATURE_COLS),
            "feature_cols": FEATURE_COLS,
            "class_names": list(class_names),
            "mlp_hidden_sizes": mlp_hidden_sizes,
            "cnn_channels": cnn_channels,
            "seq_window_size": window_size,
            "best_round": best_round,
            "n_rounds": N_ROUNDS,
            "best_test_acc": best_test_acc,
            "final_test_acc": final_acc,
        }
        torch.save(model_payload, save_model_path)
        print(f"Saved best-checkpoint model -> {save_model_path}")

        # Scaler/encoder saved alongside the model (same path, .pkl suffix)
        # via joblib rather than torch.save -- these are sklearn objects,
        # not tensors, and joblib is the standard/efficient way to persist
        # them (handles the internal numpy arrays more compactly than
        # pickling generically).
        preprocess_path = save_model_path.rsplit(".", 1)[0] + "_preprocess.pkl"
        joblib.dump({"scaler": scaler, "label_encoder": le}, preprocess_path)
        print(f"Saved scaler + label encoder -> {preprocess_path}")

    plot_training_curves(history, plot_output, best_round=best_round)

    print("\n=== Final per-class recall (global model, pooled test set) ===")
    _, per_class = evaluate(global_model, X_global_test, y_global_test, class_names)
    for cname, recall in per_class.items():
        print(f"  {cname:<16s} {recall:.4f}")

    print(f"\n=== Communication summary ===")
    print(f"  Model size per transfer: {payload_bytes / 1024:.1f} KB")
    print(f"  Total transferred over {N_ROUNDS} rounds: "
          f"{total_bytes_transferred / 1e6:.2f} MB")
    print(f"  Avg per round: {total_bytes_transferred / N_ROUNDS / 1e6:.3f} MB")

    print("\n=== Per-client local test accuracy (final global model) ===")
    for cname, c in clients.items():
        acc, _ = evaluate(global_model, c["X_test"], c["y_test"], class_names)
        print(f"  region {cname:<12s} n_train={len(c['X_train']):>7,}  "
              f"local_test_acc={acc:.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="FedAvg simulation over region_cell clients.")
    p.add_argument("paths", nargs="+", help="Labeled CSV(s) from opensky_fl_ids_pipeline.py")
    p.add_argument("--max-total-rows", type=int, default=2_000_000,
                    help="Hard global cap on total rows loaded, regardless of how many "
                         "distinct regions exist (default 2,000,000). This is the primary "
                         "memory-safety knob -- lower it if you still hit OOM.")
    p.add_argument("--max-rows-per-region-class", type=int, default=None,
                    help="Explicit cap on rows per (region_cell, attack_type) combination. "
                         "Default: auto-derived from --max-total-rows and the actual number "
                         "of distinct regions found in the file, so total stays bounded "
                         "regardless of region count. Set this manually to override "
                         "auto-sizing.")
    p.add_argument("--load-chunksize", type=int, default=200_000,
                    help="Rows read per chunk while streaming each input file.")
    p.add_argument("--n-rounds", type=int, default=None,
                    help=f"Federated communication rounds (default {N_ROUNDS}). Raise this if "
                         f"global accuracy is still climbing at the last round rather than "
                         f"plateaued -- GRU in particular may need more rounds than MLP/CNN "
                         f"to converge.")
    p.add_argument("--seq-window-size", type=int, default=None,
                    help=f"GRU only: messages of chronological history per training example "
                         f"(default {SEQ_WINDOW_SIZE}). Lower this if the printed "
                         f"'Sequence building' line shows a large fraction of aircraft being "
                         f"dropped for having too little history -- a smaller window lets "
                         f"more short-lived aircraft (e.g. ghost_aircraft, which is a "
                         f"near-singleton by design) survive into training.")
    p.add_argument("--seq-sort-col", type=str, default="time", choices=["time", "lastposupdate"],
                    help="GRU only: which timestamp field to sort each aircraft's sequence by "
                         "(default 'time'). The replay attack manipulates 'time' directly, so "
                         "sorting by it lets a replayed row settle into a position consistent "
                         "with its OWN fake timestamp -- hiding the inconsistency that should "
                         "make it detectable. Try 'lastposupdate' (a field replay does not "
                         "touch) to test whether that exposes replay as a genuine anomaly "
                         "relative to a row's TRUE neighbors instead.")
    p.add_argument("--preserve-aircraft-groups", action="store_true",
                    dest="preserve_aircraft_groups", default=False,
                    help="EXPERIMENTAL, default OFF -- tested and found to INCREASE sequence "
                         "dropout on realistic class-imbalanced data, not decrease it. When the "
                         "benign class dominates and needs heavy thinning, keeping whole aircraft "
                         "groups intact concentrates the scarce benign budget onto few aircraft "
                         "rather than spreading it thin across many -- and since surviving the "
                         "window threshold depends on an aircraft's TOTAL row count across all "
                         "buckets, thin-spread (the default row-level sampling) clears more "
                         "aircraft over that threshold than intact-but-concentrated does. Kept "
                         "available for cases where class balance is more even and this tradeoff "
                         "may reverse -- verify with the printed 'Sequence building' line before "
                         "trusting it, don't assume it helps.")
    p.add_argument("--exclude-classes", type=str, default=None,
                    help="Comma-separated attack_type values to drop ENTIRELY before training "
                         "(e.g. 'ghost_aircraft'). Use this when a class is structurally "
                         "undetectable by the chosen architecture -- e.g. GRU requires several "
                         "consecutive messages under one icao24 to form a sequence, but "
                         "ghost_aircraft is defined by having ~1 message under its fake "
                         "identity, so it can never appear in GRU's training or test data "
                         "regardless of tuning. Dropping it is more honest than reporting a "
                         "6-class model that silently never sees one of its classes.")
    p.add_argument("--plot-output", type=str, default="training_curves.png",
                    help="Where to save the train/test accuracy + communication cost plot "
                         "(default training_curves.png in the current directory).")
    p.add_argument("--save-model-path", type=str, default=None,
                    help="If set, save the best-checkpoint model weights to this path (e.g. "
                         "'best_model.pt') via torch.save, plus the fitted StandardScaler and "
                         "LabelEncoder alongside it as '<name>_preprocess.pkl' via joblib. "
                         "Both are required for correct inference later -- the model alone "
                         "cannot reproduce the exact feature scaling/label mapping it was "
                         "trained with. Default: don't save anything (unchanged behavior).")
    p.add_argument("--model", type=str, default="gru", choices=["mlp", "cnn", "gru"],
                    help="Which architecture to train (default gru). mlp/cnn are row-independent "
                         "(--seq-window-size, --seq-sort-col, --preserve-aircraft-groups are "
                         "ignored); gru requires real chronological sequences and uses all of "
                         "those flags. Switching this also switches USE_SEQUENCES automatically "
                         "-- they can never be set inconsistently with each other.")
    p.add_argument("--mlp-hidden-sizes", type=str, default="64,32",
                    help="MLP only: comma-separated hidden layer sizes (default '64,32', "
                         "matching the original architecture, ~3,174 params / 12.4 KB). "
                         "Try '128,64' or '96,48' if MLP's accuracy plateaus below CNN's on "
                         "the same data/rounds with a stable (non-growing) train/test gap -- "
                         "that combination signals a genuine capacity limit worth testing a "
                         "bigger network against, not a training-budget problem. Check the "
                         "printed model size afterward -- a wider MLP needs to stay "
                         "communication-cheap to be worth using over CNN in FL.")
    p.add_argument("--cnn-channels", type=str, default="16,32",
                    help="CNN only: comma-separated conv layer channel sizes (default '16,32', "
                         "matching the original architecture, 16.1 KB). More VALUES = more conv "
                         "layers (deeper, e.g. '16,32,64'); bigger VALUES = wider layers "
                         "(e.g. '32,64', same 2-layer depth). Try this if CNN's accuracy has "
                         "genuinely plateaued (not just needs more rounds) at the current row "
                         "budget -- check model size afterward, same communication-cost caveat "
                         "as --mlp-hidden-sizes.")
    args = p.parse_args()
    # Preserve the distinction between "flag not passed" (None -- lets
    # main() apply the GRU auto-exclude default) and "flag explicitly
    # passed empty" (user wants to disable auto-exclude and use every
    # class) -- a plain `if args.exclude_classes` would treat both cases
    # identically, since an empty string is falsy in Python.
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
         preserve_aircraft_groups=args.preserve_aircraft_groups,
         exclude_classes=exclude_classes, plot_output=args.plot_output,
         model_type=args.model, mlp_hidden_sizes=mlp_hidden_sizes, cnn_channels=cnn_channels,
         save_model_path=args.save_model_path)
