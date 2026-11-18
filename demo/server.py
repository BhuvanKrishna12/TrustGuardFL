#!/usr/bin/env python3
"""
demo/server.py: live backend for the FL-IDS demo (FastAPI + uvicorn, one asyncio loop).

Streams each of the six regional clients' benign holdout traffic at a fixed rate,
scores every message with the trained detection model (prep_features, scaler, model,
softmax, then class prior correction), and pushes the results to any number of
browser tabs over a WebSocket. Attacks are injected on demand from each client's
pool of real labeled attack rows. Training runs can be replayed from their JSONL log
or launched live through demo/train_demo.py. The binding spec is demo/CONTRACT.md.

Usage, from the project root (the folder holding federated_simulation.py):

    python3 demo/server.py
    python3 demo/server.py --model demo/models/latest.pt --rate 2 --strength 0.5
    python3 demo/server.py --data-dir demo/data --runs-dir demo/runs --port 8000

Then open http://localhost:8000 in the Windows browser (WSL2 forwards localhost).

Flags (relative paths resolve against the project root, see --root):
    --model        detection checkpoint, default models/fixed4_cnn.pt (the paper model).
                   demo/models/latest.pt is the model from the latest train_demo.py run.
                   (its scaler and label encoder are read from <model>_preprocess.pkl)
    --prior-json   JSON holding the real class prior, default results/lgbm_feb04.json
    --data-dir     output of prepare_demo_data.py, default demo/data
    --runs-dir     training JSONL logs, default demo/runs
    --port         default 8000
    --host         default 0.0.0.0 so the Windows side can reach WSL
    --rate         messages per second per client, default 4
    --strength     class prior correction strength in [0, 1], default 0.5
    --noisy-rate   'noisy' when this fraction of a tower's benign msgs are false alarms,
                   default 0.25 (with --noisy-window 20 seconds, --noisy-min 3)
    --root         project root, default the parent of this demo/ folder
    --static-dir   frontend folder, default demo/static next to this file
    --train-script script run by POST /api/training/start, default demo/train_demo.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
DEFAULT_ROOT = DEMO_DIR.parent

CLIENT_QUEUE_MAX = 2000          # per browser connection, oldest events dropped beyond this
RECENT_PER_CLIENT = 12
INCIDENTS_MAX = 50
COUNTERS_MIN_INTERVAL = 0.5      # seconds, so at most 2 counters events per second per client
TICK_SECONDS = 0.25              # streamer cadence; each tick emits rate * elapsed rows per client
CATCH_WINDOW = 10.0              # status rule: >= 3 correct catches in the last 10 s
CATCH_THRESHOLD = 3
# "noisy" rule defaults (CLI: --noisy-rate, --noisy-window, --noisy-min). A tower is noisy
# when at least NOISY_RATE of its benign messages in the last NOISY_WINDOW seconds were
# false alarms, and there were at least NOISY_MIN of them. A fraction, not a count: the
# paper model flags about 8.7% of real benign rows at strength 0.5, which at 2 msg/s is
# about 5 false alarms per 30 s, so the old ">= 2 in 30 s" count kept every tower noisy.
NOISY_RATE = 0.25
NOISY_WINDOW = 20.0
NOISY_MIN = 3
MAX_INJECT = 500
RATE_MIN, RATE_MAX = 0.25, 50.0
RUN_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


def fail(msg: str, code: int = 2):
    """Print one clear line and exit, never a traceback."""
    print("ERROR: " + msg, file=sys.stderr)
    sys.exit(code)


# --------------------------------------------------------------------------------------
# CLI and imports (heavy imports happen after argument parsing so --help is instant)
# --------------------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="FL-IDS live demo backend. See demo/CONTRACT.md.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="models/fixed4_cnn.pt",
                   help="detection checkpoint; demo/models/latest.pt uses the 6-client demo model")
    p.add_argument("--prior-json", default="results/lgbm_feb04.json")
    p.add_argument("--data-dir", default="demo/data")
    p.add_argument("--runs-dir", default="demo/runs")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--rate", type=float, default=4.0)
    p.add_argument("--strength", type=float, default=0.5)
    p.add_argument("--noisy-rate", type=float, default=NOISY_RATE,
                   help="a tower turns 'noisy' when this fraction of its benign messages in "
                        "the window were false alarms (the startup table shows each tower's "
                        "expected rate)")
    p.add_argument("--noisy-window", type=float, default=NOISY_WINDOW,
                   help="seconds of traffic the noisy rule looks at")
    p.add_argument("--noisy-min", type=int, default=NOISY_MIN,
                   help="minimum number of false alarms in the window before 'noisy'")
    p.add_argument("--root", default=str(DEFAULT_ROOT),
                   help="project root: relative paths resolve here and train_demo.py runs here")
    p.add_argument("--static-dir", default=str(DEMO_DIR / "static"))
    p.add_argument("--train-script", default=str(DEMO_DIR / "train_demo.py"),
                   help="script launched by /api/training/start")
    return p.parse_args(argv)


ARGS = parse_args() if __name__ == "__main__" else None

# Project root on sys.path so federated_simulation and evaluate_holdout import. The
# real code root is always the folder above this file, even when --root points at a
# test fixture.
for _p in (str(DEFAULT_ROOT), str(DEMO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import numpy as np
    import pandas as pd
    import torch
    import joblib
except ImportError as _exc:  # pragma: no cover
    fail(f"missing package ({_exc}). Install the project requirements first.")

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn
except ImportError as _exc:  # pragma: no cover
    fail(f"missing package ({_exc}). Run: pip3 install --user --break-system-packages "
         f"-r demo/requirements-demo.txt")

try:
    import federated_simulation as fs
    import fs_compat  # demo-side wrapper around fs.prep_features for CSV-read rows
except Exception as _exc:  # noqa: BLE001
    fail(f"could not import federated_simulation.py from {DEFAULT_ROOT} ({_exc}). "
         f"Put the demo/ folder inside the project root.")

try:
    import evaluate_holdout as eh  # reused: build_model_from_payload, model_forward, prior JSON reader
except BaseException as _exc:  # noqa: BLE001  (its own die() raises SystemExit)
    eh = None
    print(f"WARNING: evaluate_holdout.py not importable ({_exc}); using the built in loader.")

FEATURE_COLS = list(fs.FEATURE_COLS)
ATTACK_ORDER = ["flooding", "ghost_aircraft", "gps_spoof", "replay", "velocity_spoof"]


# --------------------------------------------------------------------------------------
# Model, prior and data loading
# --------------------------------------------------------------------------------------

def resolve(root: Path, p: str) -> Path:
    q = Path(os.path.expanduser(p))
    return q if q.is_absolute() else (root / q)


def load_detector(model_path: Path):
    """Load the checkpoint and its _preprocess.pkl. Mirrors evaluate_holdout.main()."""
    if not model_path.is_file():
        fail(f"model checkpoint not found: {model_path} (pass --model)")
    pre_path = Path(str(model_path).rsplit(".", 1)[0] + "_preprocess.pkl")
    if not pre_path.is_file():
        fail(f"preprocess file not found next to the model: {pre_path}")

    pre = joblib.load(pre_path)
    scaler = label_encoder = None
    if isinstance(pre, dict):
        for k in ("scaler", "standard_scaler", "StandardScaler", "sc"):
            if k in pre:
                scaler = pre[k]
                break
        for k in ("label_encoder", "le", "LabelEncoder", "encoder"):
            if k in pre:
                label_encoder = pre[k]
                break
    elif isinstance(pre, (tuple, list)) and len(pre) >= 2:
        scaler, label_encoder = pre[0], pre[1]
    if scaler is None or not hasattr(scaler, "transform"):
        fail(f"no fitted StandardScaler in {pre_path}")

    # weights_only=False: the payload is the project's own dict with numpy scalars.
    payload = torch.load(model_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        fail(f"{model_path} is not a dict payload as written by federated_simulation.py")

    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        class_names = [str(c) for c in label_encoder.classes_]
    elif isinstance(payload.get("class_names"), (list, tuple)):
        class_names = [str(c) for c in payload["class_names"]]
    else:
        fail(f"no LabelEncoder in {pre_path} and no class_names in the checkpoint")

    mtype = payload.get("model_type")
    if isinstance(mtype, str) and "gru" in mtype.lower():
        fail("this checkpoint is a GRU (sequence) model; the live demo scores single rows, "
             "use an mlp or cnn checkpoint")

    if eh is not None:
        try:
            model, desc, n_feat, n_cls = eh.build_model_from_payload(
                payload, fs, len(FEATURE_COLS), len(class_names))
        except (RuntimeError, SystemExit) as exc:
            fail("could not rebuild the model: " + str(exc).splitlines()[0])
    else:
        try:
            model = fs.build_model(
                payload["model_type"], payload.get("n_features", len(FEATURE_COLS)),
                len(class_names),
                mlp_hidden_sizes=tuple(payload.get("mlp_hidden_sizes") or (64, 32)),
                cnn_channels=tuple(payload.get("cnn_channels") or (16, 32)),
                device=torch.device("cpu"))
            model.load_state_dict(payload["model_state_dict"])
        except Exception as exc:  # noqa: BLE001
            fail(f"could not rebuild the model: {exc}")
        n_feat, n_cls = len(FEATURE_COLS), len(class_names)
    if n_cls != len(class_names):
        fail(f"model has {n_cls} outputs but the label encoder has {len(class_names)} classes")
    if n_feat != len(FEATURE_COLS):
        fail(f"model expects {n_feat} inputs but FEATURE_COLS has {len(FEATURE_COLS)}")
    model.eval()
    info = {
        "path": str(model_path),
        "model_type": str(mtype or "unknown"),
        "best_round": payload.get("best_round"),
        "best_test_acc": payload.get("best_test_acc"),
        "cnn_channels": payload.get("cnn_channels"),
    }
    if info["best_round"] is not None:
        info["best_round"] = int(info["best_round"])
    return model, scaler, class_names, info


def load_prior(path: Path, class_names):
    """Read the real class prior. Falls back to uniform (correction is then a no-op)."""
    if not path.is_file():
        print(f"WARNING: prior JSON not found ({path}); using a UNIFORM prior, so the "
              f"strength slider will have no effect.")
        return np.full(len(class_names), 1.0 / len(class_names)), "uniform (prior JSON missing)"
    try:
        obj = json.loads(path.read_text())
    except ValueError as exc:
        fail(f"prior JSON {path} could not be parsed: {exc}")
    mapping = None
    if eh is not None:
        try:
            mapping, _where = eh._mapping_from_json(obj, str(path))
        except SystemExit:
            mapping = None
    if mapping is None:
        if isinstance(obj, dict) and isinstance(obj.get("real_prior"), dict):
            mapping = obj["real_prior"]
        elif isinstance(obj, dict) and all(isinstance(v, (int, float)) for v in obj.values()):
            mapping = obj
        else:
            fail(f"no class prior mapping in {path} (expected a 'real_prior' key)")
    missing = [c for c in class_names if c not in mapping]
    if missing:
        fail(f"prior in {path} has no share for: {', '.join(missing)}")
    raw = np.array([float(mapping[c]) for c in class_names], dtype=np.float64)
    if (raw <= 0).any():
        # a zero share would make that class impossible at any strength > 0
        raw = np.maximum(raw, 1e-6)
    return raw / raw.sum(), str(path)


def fmt_times(series) -> list:
    """Row 'time' (unix seconds) to HH:MM:SS UTC strings."""
    t = pd.to_numeric(series, errors="coerce")
    dt = pd.to_datetime(t, unit="s", utc=True, errors="coerce")
    bad = dt.isna()
    if bad.any():
        dt2 = pd.to_datetime(series[bad], utc=True, errors="coerce")
        dt = dt.copy()
        dt[bad] = dt2
    out = dt.dt.strftime("%H:%M:%S")
    return out.fillna("--:--:--").astype(str).tolist()


def scaler_input(feats, scaler):
    """Hand the scaler its columns in the order it was fitted with (it was fitted on a
    named DataFrame), so a different FEATURE_COLS order can never silently swap inputs."""
    names = getattr(scaler, "feature_names_in_", None)
    if names is None:
        return feats
    names = [str(n) for n in names]
    if sorted(names) != sorted(feats.columns):
        fail(f"the scaler was fitted on columns {names}, but FEATURE_COLS is "
             f"{list(feats.columns)}; this model does not match federated_simulation.py")
    return feats[names]


def raw_probs(model, X: np.ndarray, batch: int = 8192) -> np.ndarray:
    """Uncorrected softmax probabilities for every row, computed once at startup."""
    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xt = torch.from_numpy(X[i:i + batch])
            logits = eh.model_forward(model, xt) if eh is not None else model(xt)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            out.append(torch.softmax(logits.float(), dim=1).numpy().astype(np.float32))
    return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


class Table:
    """A client's stream or pool, converted, scaled and scored (raw probabilities)."""

    def __init__(self, path: Path, scaler, model, n_classes: int):
        df = pd.read_csv(path, dtype=str, keep_default_na=True)
        missing = [c for c in FEATURE_COLS + ["attack_type"] if c not in df.columns]
        if missing:
            fail(f"{path} is missing column(s): {', '.join(missing)}")
        self.n = len(df)
        if self.n:
            feats = fs_compat.prep_features(fs, df)      # bools, text numerics, NaN -> -1, float32
            self.X = np.asarray(scaler.transform(scaler_input(feats, scaler)), dtype=np.float32)
            self.probs = raw_probs(model, self.X)
        else:
            self.X = np.zeros((0, len(FEATURE_COLS)), dtype=np.float32)
            self.probs = np.zeros((0, n_classes), dtype=np.float32)
        self.truth = df["attack_type"].fillna("benign").astype(str).str.strip().tolist()
        self.icao = (df["icao24"].fillna("").astype(str).tolist() if "icao24" in df.columns
                     else [""] * self.n)
        self.time = fmt_times(df["time"]) if "time" in df.columns else ["--:--:--"] * self.n


def load_clients(data_dir: Path, scaler, model, n_classes: int):
    cj = data_dir / "clients.json"
    if not cj.is_file():
        fail(f"{cj} not found. Run `python3 demo/prepare_demo_data.py` first "
             f"(or pass --data-dir).")
    try:
        meta = json.loads(cj.read_text())
    except ValueError as exc:
        fail(f"{cj} could not be parsed: {exc}")
    clients = []
    for c in meta.get("clients", []):
        cid = c["id"]
        sp, pp = data_dir / f"{cid}_stream.csv", data_dir / f"{cid}_pool.csv"
        if not sp.is_file():
            fail(f"{sp} not found. Re-run prepare_demo_data.py.")
        stream = Table(sp, scaler, model, n_classes)
        if stream.n == 0:
            fail(f"{sp} has no rows")
        pool = Table(pp, scaler, model, n_classes) if pp.is_file() else None
        if pool is None:
            print(f"WARNING: {pp} not found; injection disabled for {cid}")
        clients.append(ClientState(c, stream, pool))
    if not clients:
        fail(f"{cj} lists no clients")
    return meta, clients


# --------------------------------------------------------------------------------------
# Live state
# --------------------------------------------------------------------------------------

def zero_counters():
    return {"seen": 0, "flagged": 0, "correct": 0, "false_alarm": 0, "missed": 0}


class ClientState:
    def __init__(self, meta: dict, stream: Table, pool: Table | None):
        self.id = meta["id"]
        self.cell = meta.get("cell", "")
        self.name = meta.get("name", self.id)
        self.lat = meta.get("lat")
        self.lon = meta.get("lon")
        self.stream = stream
        self.pool = pool
        self.cursor = 0
        self.credit = 0.0                       # fractional rows owed by the rate
        self.injections = []                    # [stream_clock_due, pool_idx, burst_id], sorted
        self.pool_by_type: dict[str, np.ndarray] = {}
        self.pool_order: dict[str, list] = {}   # per type: [permutation, pointer]
        if pool is not None and pool.n:
            truth = np.array(pool.truth)
            for t in np.unique(truth):
                self.pool_by_type[str(t)] = np.flatnonzero(truth == t)
        self.reset_counters()

    def reset_counters(self):
        self.counters = zero_counters()
        self.recent = deque(maxlen=RECENT_PER_CLIENT)
        self.catch_times = deque()
        self.fa_times = deque()
        self.benign_times = deque()             # every scored truth-benign message
        self.status = "normal"
        self.counters_dirty = False
        self.counters_sent_at = 0.0

    def draw_pool(self, attack: str, count: int, rng) -> list:
        """Sample without replacement within a burst; cycle a fresh shuffle when exhausted."""
        idx = self.pool_by_type.get(attack)
        if idx is None or len(idx) == 0:
            return []
        perm, ptr = self.pool_order.get(attack, (None, 0))
        out = []
        while len(out) < count:
            if perm is None or ptr >= len(perm):
                perm, ptr = rng.permutation(idx), 0
            take = min(count - len(out), len(perm) - ptr)
            out.extend(int(i) for i in perm[ptr:ptr + take])
            ptr += take
        self.pool_order[attack] = (perm, ptr)
        return out

    def compute_status(self, now: float, noisy_rate: float = NOISY_RATE,
                       noisy_window: float = NOISY_WINDOW, noisy_min: int = NOISY_MIN) -> str:
        while self.catch_times and now - self.catch_times[0] > CATCH_WINDOW:
            self.catch_times.popleft()
        while self.fa_times and now - self.fa_times[0] > noisy_window:
            self.fa_times.popleft()
        while self.benign_times and now - self.benign_times[0] > noisy_window:
            self.benign_times.popleft()
        if len(self.catch_times) >= CATCH_THRESHOLD:
            return "under_attack"
        n_fa, n_ben = len(self.fa_times), len(self.benign_times)
        if n_fa >= noisy_min and n_ben and n_fa / n_ben >= noisy_rate:
            return "noisy"
        return "normal"

    def public(self) -> dict:
        return {"id": self.id, "cell": self.cell, "name": self.name, "lat": self.lat,
                "lon": self.lon, "status": self.status, "counters": dict(self.counters),
                "recent": list(self.recent)}


class Hub:
    """WebSocket fan-out. Each connection has its own bounded queue; never blocks the loop."""

    def __init__(self):
        self.conns: set[asyncio.Queue] = set()
        self.dropped = 0

    def add(self) -> asyncio.Queue:
        q = asyncio.Queue(maxsize=CLIENT_QUEUE_MAX)
        self.conns.add(q)
        return q

    def remove(self, q):
        self.conns.discard(q)

    @staticmethod
    def put(q: asyncio.Queue, text: str) -> int:
        dropped = 0
        while True:
            try:
                q.put_nowait(text)
                return dropped
            except asyncio.QueueFull:
                try:
                    q.get_nowait()
                    dropped += 1
                except asyncio.QueueEmpty:
                    pass

    def broadcast(self, event: dict):
        if not self.conns:
            return
        text = json.dumps(event, separators=(",", ":"))
        for q in list(self.conns):
            self.dropped += self.put(q, text)


class Engine:
    def __init__(self, args, model, scaler, class_names, model_info, prior, prior_src,
                 clients, data_meta, root: Path, runs_dir: Path):
        self.args = args
        self.model = model
        self.class_names = class_names
        self.benign_idx = class_names.index("benign") if "benign" in class_names else None
        if self.benign_idx is None:
            fail("the model has no 'benign' class")
        self.attack_types = [c for c in ATTACK_ORDER if c in class_names] + \
                            [c for c in class_names if c != "benign" and c not in ATTACK_ORDER]
        self.model_info = model_info
        self.prior = prior
        self.prior_src = prior_src
        self.clients = clients
        self.by_id = {c.id: c for c in clients}
        self.data_meta = data_meta
        self.root = root
        self.runs_dir = runs_dir
        self.data_dir = Path(args.data_dir) if Path(args.data_dir).is_absolute() else root / args.data_dir
        self.strength = float(min(1.0, max(0.0, args.strength)))
        self.rate = float(min(RATE_MAX, max(RATE_MIN, args.rate)))
        self.noisy_rate = float(getattr(args, "noisy_rate", NOISY_RATE))
        self.noisy_window = float(getattr(args, "noisy_window", NOISY_WINDOW))
        self.noisy_min = int(getattr(args, "noisy_min", NOISY_MIN))
        self.bg_tasks: set = set()            # strong refs so background tasks are not GC'd
        self.running = True
        self.seq = 0
        self.burst_n = 0
        self.stream_clock = 0.0               # advances only while running
        self.incidents = deque(maxlen=INCIDENTS_MAX)
        self.hub = Hub()
        self.rng = np.random.default_rng(1234)
        self.train_proc = None
        self.train_run = None
        self.train_tail = deque(maxlen=40)
        self.tick_ms = deque(maxlen=200)
        self.task = None

    # ---- snapshot / config ----
    def snapshot(self) -> dict:
        return {
            "type": "snapshot",
            "clients": [c.public() for c in self.clients],
            "strength": self.strength,
            "running": self.running,
            "rate": self.rate,
            "model": {"path": self.model_info["path"], "model_type": self.model_info["model_type"],
                      "best_round": self.model_info["best_round"]},
            "incidents": list(self.incidents),
            "attack_types": list(self.attack_types),
        }

    def config_event(self) -> dict:
        return {"type": "config", "strength": self.strength, "running": self.running,
                "rate": self.rate}

    def toast(self, text: str):
        print(f"[toast] {text}")
        self.hub.broadcast({"type": "toast", "text": text})

    def expected_fa(self, strength: float) -> dict:
        """Share of each client's benign stream that the model flags at this strength."""
        w = np.power(self.prior, strength)
        out = {}
        for c in self.clients:
            pred = (c.stream.probs.astype(np.float64) * w[None, :]).argmax(axis=1)
            out[c.id] = float((pred != self.benign_idx).mean()) if c.stream.n else 0.0
        return out

    # ---- injection ----
    def inject(self, cid: str, attack: str, count: int) -> dict:
        c = self.by_id.get(cid)
        if c is None:
            raise HTTPException(404, f"unknown client {cid!r}")
        if attack not in self.attack_types:
            raise HTTPException(400, f"unknown attack type {attack!r}; one of {self.attack_types}")
        if count < 1 or count > MAX_INJECT:
            raise HTTPException(400, f"count must be 1..{MAX_INJECT}")
        rows = c.draw_pool(attack, count, self.rng)
        if not rows:
            raise HTTPException(409, f"{cid} has no {attack} rows in its pool")
        self.burst_n += 1
        bid = f"b{self.burst_n}"
        # Spread the burst over 3 to 6 seconds of stream time (longer only for very big
        # bursts, about 2x the normal rate), with jitter so it reads as traffic, not a dump.
        dur = min(6.0, max(3.0, count / self.rate))
        dur = max(dur, count / (2.0 * self.rate))
        start = self.stream_clock + 0.1
        offs = np.sort(self.rng.uniform(0.0, dur, size=count))
        for off, idx in zip(offs, rows):
            c.injections.append([start + float(off), idx, bid])
        c.injections.sort(key=lambda e: e[0])
        print(f"[inject] {bid}: {count} x {attack} -> {cid} over {dur:.1f}s")
        return {"ok": True, "queued": count, "burst_id": bid}

    # ---- reset ----
    def reset(self):
        now = time.monotonic()
        for c in self.clients:
            c.reset_counters()
        self.incidents.clear()
        for c in self.clients:
            self.hub.broadcast({"type": "counters", "client": c.id, "counters": dict(c.counters)})
            self.hub.broadcast({"type": "status", "client": c.id, "status": c.status})
            c.counters_sent_at = now

    # ---- the streamer ----
    async def run(self):
        loop = asyncio.get_running_loop()
        last = loop.time()
        next_t = last + TICK_SECONDS
        while True:
            delay = next_t - loop.time()
            if delay > 0:
                await asyncio.sleep(delay)
            now_l = loop.time()
            dt = min(1.0, now_l - last)      # cap catch up after a stall
            last = now_l
            next_t += TICK_SECONDS
            if next_t < now_l:               # we fell behind; do not burst to catch up
                next_t = now_l + TICK_SECONDS
            try:
                t0 = time.perf_counter()
                self.tick(dt)
                self.tick_ms.append((time.perf_counter() - t0) * 1000.0)
            except Exception as exc:  # noqa: BLE001  never let the streamer die
                print(f"[streamer] tick failed: {exc!r}")

    def tick(self, dt: float):
        now = time.monotonic()
        batch = []   # (client, table, row_idx, injected, burst_id, display_time or None)
        if self.running:
            self.stream_clock += dt
            for c in self.clients:
                c.credit += self.rate * dt
                n_normal = int(c.credit)
                c.credit -= n_normal
                due = 0
                while due < len(c.injections) and c.injections[due][0] <= self.stream_clock:
                    due += 1
                inj = c.injections[:due]
                del c.injections[:due]
                # interleave: injected rows land at their scheduled spots among normal rows
                slots = [None] * n_normal
                for e in inj:
                    slots.insert(int(self.rng.integers(0, len(slots) + 1)), e)
                for s in slots:
                    if s is None:
                        batch.append((c, c.stream, c.cursor, False, None, None))
                        c.cursor = (c.cursor + 1) % c.stream.n
                    else:
                        # Pool rows come from any time of day. Show them at the client's
                        # current stream time so the feed and the log stay in order; the
                        # row's real timestamp travels along as orig_time.
                        now_t = c.stream.time[(c.cursor - 1) % c.stream.n]
                        batch.append((c, c.pool, s[1], True, s[2], now_t))
        if batch:
            self.score_and_emit(batch, now)
        for c in self.clients:
            st = c.compute_status(now, self.noisy_rate, self.noisy_window, self.noisy_min)
            if st != c.status:
                c.status = st
                self.hub.broadcast({"type": "status", "client": c.id, "status": st})
            if c.counters_dirty and now - c.counters_sent_at >= COUNTERS_MIN_INTERVAL:
                c.counters_dirty = False
                c.counters_sent_at = now
                self.hub.broadcast({"type": "counters", "client": c.id,
                                    "counters": dict(c.counters)})

    def score_and_emit(self, batch, now):
        # raw probabilities were computed once at startup (Table.probs), so the tick does
        # no model work and never stalls the event loop
        probs = np.stack([tbl.probs[i] for (_c, tbl, i, _inj, _b, _t) in batch]).astype(np.float64)
        # raw probabilities are kept per message; correction uses the CURRENT strength
        w = np.power(self.prior, self.strength)
        corr = probs * w[None, :]
        corr /= corr.sum(axis=1, keepdims=True)
        pred_idx = corr.argmax(axis=1)
        conf = corr[np.arange(len(pred_idx)), pred_idx]
        b = self.benign_idx
        for k, (c, tbl, i, injected, bid, shown_t) in enumerate(batch):
            self.seq += 1
            pred = self.class_names[int(pred_idx[k])]
            truth = tbl.truth[i]
            truth_att = truth != "benign"
            pred_att = int(pred_idx[k]) != b
            wrong_type = False
            cnt = c.counters
            cnt["seen"] += 1
            if pred_att:
                cnt["flagged"] += 1
            if truth_att and pred_att:
                verdict = "correct_catch"
                wrong_type = pred != truth
                cnt["correct"] += 1
                c.catch_times.append(now)
            elif pred_att:
                verdict = "false_alarm"
                cnt["false_alarm"] += 1
                c.fa_times.append(now)
            elif truth_att:
                verdict = "missed"
                cnt["missed"] += 1
            else:
                verdict = "ok"
            if not truth_att:
                c.benign_times.append(now)
            c.counters_dirty = True
            msg = {"type": "msg", "client": c.id, "seq": self.seq,
                   "time": shown_t or tbl.time[i],
                   "icao24": tbl.icao[i], "pred": pred, "conf": round(float(conf[k]), 4),
                   "truth": truth, "injected": bool(injected), "burst_id": bid,
                   "verdict": verdict, "wrong_type": wrong_type}
            if injected:
                msg["orig_time"] = tbl.time[i]
            c.recent.append(msg)
            self.hub.broadcast(msg)
            if pred_att:
                inc = dict(msg)
                inc["type"] = "incident"
                inc["name"] = c.name
                self.incidents.append(inc)
                self.hub.broadcast(inc)

    # ---- training ----
    def list_runs(self) -> list:
        if not self.runs_dir.is_dir():
            return []
        files = sorted(self.runs_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime,
                       reverse=True)
        names = [p.stem for p in files]
        if "latest" in names:
            names.remove("latest")
            names.insert(0, "latest")
        return names

    def read_run(self, run: str) -> list:
        if not RUN_NAME_RE.match(run):
            raise HTTPException(400, "bad run name")
        p = self.runs_dir / f"{run}.jsonl"
        if not p.is_file():
            raise HTTPException(404, f"no run named {run!r} in {self.runs_dir}")
        out = []
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
        return out

    async def start_training(self, rounds: int) -> dict:
        if self.train_proc is not None and self.train_proc.returncode is None:
            raise HTTPException(409, f"training run {self.train_run} is already running")
        script = Path(os.path.expanduser(self.args.train_script)).resolve()
        if not script.is_file():
            self.toast(f"Cannot start training: {script} not found.")
            raise HTTPException(503, f"{script} not found")
        run = "live_" + time.strftime("%Y%m%d_%H%M%S")
        # --keep-latest: a quick live run from the page must not replace the rehearsed
        # runs/latest.jsonl that the Replay button plays (it still shows up by name)
        cmd = [sys.executable, "-u", str(script), "--rounds", str(int(rounds)), "--run-name", run,
               "--data-dir", str(self.data_dir), "--runs-dir", str(self.runs_dir),
               "--keep-latest"]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=str(self.root), env=env,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
        except Exception as exc:  # noqa: BLE001
            self.toast(f"Could not start training: {exc}")
            raise HTTPException(500, f"could not start training: {exc}")
        self.train_proc, self.train_run = proc, run
        self.train_tail.clear()
        print(f"[train] started {run}: {' '.join(cmd)} (pid {proc.pid})")
        for coro in (self._pump_output(proc), self._tail_run(proc, run)):
            t = asyncio.create_task(coro)
            self.bg_tasks.add(t)
            t.add_done_callback(self.bg_tasks.discard)
        return {"ok": True, "run": run, "pid": proc.pid}

    async def _pump_output(self, proc):
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").rstrip()
            self.train_tail.append(text)
            print(f"[train] {text}")

    async def _tail_run(self, proc, run):
        path = self.runs_dir / f"{run}.jsonl"
        pos, buf, n_lines = 0, b"", 0

        def drain():
            nonlocal pos, buf, n_lines
            if not path.is_file():
                return
            with open(path, "rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
                pos = fh.tell()
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                except ValueError:
                    continue
                n_lines += 1
                self.hub.broadcast({"type": "training", "data": obj})

        while proc.returncode is None:
            drain()
            try:
                await asyncio.wait_for(proc.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass
        await asyncio.sleep(0.3)
        drain()
        if buf.strip():
            buf += b"\n"
            drain()
        rc = proc.returncode
        if rc != 0:
            tail = " | ".join(list(self.train_tail)[-4:]) or "(no output)"
            if len(tail) > 600:
                tail = "..." + tail[-600:]
            self.toast(f"Training run {run} failed (exit {rc}): {tail}")
        elif n_lines == 0:
            self.toast(f"Training run {run} exited but wrote no log to {path}.")
        else:
            self.toast(f"Training run {run} finished.")
        print(f"[train] {run} exited with {rc}, {n_lines} log lines forwarded")

    async def shutdown(self):
        if self.task:
            self.task.cancel()
        p = self.train_proc
        if p is not None and p.returncode is None:
            p.terminate()
            try:
                await asyncio.wait_for(p.wait(), 5)
            except asyncio.TimeoutError:
                p.kill()


# --------------------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------------------

class InjectBody(BaseModel):
    client: str
    attack: str
    count: int = 25


class StrengthBody(BaseModel):
    strength: float


class StreamBody(BaseModel):
    running: bool | None = None
    rate: float | None = None


class TrainBody(BaseModel):
    rounds: int = 15


PLACEHOLDER = """<!doctype html><html><head><meta charset="utf-8"><title>AeroGuard demo backend</title>
<style>body{font:13px/1.4 monospace;margin:16px;background:#111;color:#ddd}
#log{white-space:pre-wrap}b{color:#fc6}</style></head><body>
<p><b>Frontend missing:</b> demo/static/index.html was not found. The backend is running;
live WebSocket events are shown below as plain text (newest first, last 300).</p>
<div id="log"></div><script>
const log=document.getElementById('log');const lines=[];
function add(t){lines.unshift(t);if(lines.length>300)lines.pop();log.textContent=lines.join('\\n');}
function go(){const ws=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
ws.onmessage=e=>add(e.data);ws.onclose=()=>{add('-- disconnected, retrying in 2 s');setTimeout(go,2000);};}
go();</script></body></html>"""


def build_app(engine: Engine, static_dir: Path) -> FastAPI:
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def lifespan(_app):
        engine.task = asyncio.create_task(engine.run())
        yield
        await engine.shutdown()

    app = FastAPI(title="AeroGuard live demo", lifespan=lifespan)

    @app.get("/api/state")
    async def api_state():
        return engine.snapshot()

    @app.post("/api/inject")
    async def api_inject(body: InjectBody):
        return engine.inject(body.client, body.attack, int(body.count))

    @app.post("/api/strength")
    async def api_strength(body: StrengthBody):
        engine.strength = float(min(1.0, max(0.0, body.strength)))
        ev = engine.config_event()
        engine.hub.broadcast(ev)
        return {"ok": True, **{k: v for k, v in ev.items() if k != "type"}}

    @app.post("/api/stream")
    async def api_stream(body: StreamBody):
        if body.running is not None:
            engine.running = bool(body.running)
        if body.rate is not None:
            if not (body.rate > 0):
                raise HTTPException(400, "rate must be > 0")
            engine.rate = float(min(RATE_MAX, max(RATE_MIN, body.rate)))
        ev = engine.config_event()
        engine.hub.broadcast(ev)
        return {"ok": True, **{k: v for k, v in ev.items() if k != "type"}}

    @app.post("/api/reset")
    async def api_reset():
        engine.reset()
        return {"ok": True}

    @app.get("/api/training/runs")
    async def api_runs():
        return {"runs": engine.list_runs()}

    @app.get("/api/training/replay")
    async def api_replay(run: str = "latest"):
        return engine.read_run(run)

    @app.post("/api/training/start")
    async def api_train_start(body: TrainBody | None = None):
        rounds = body.rounds if body is not None else 15
        if rounds < 1 or rounds > 200:
            raise HTTPException(400, "rounds must be 1..200")
        return await engine.start_training(rounds)

    @app.get("/api/health")
    async def api_health():
        ticks = list(engine.tick_ms)
        exp = engine.expected_fa(engine.strength)
        fa_now = {c.id: {"observed": round(len(c.fa_times) / len(c.benign_times), 3)
                         if c.benign_times else None,
                         "expected": round(exp[c.id], 3), "status": c.status}
                  for c in engine.clients}
        return {"ok": True, "false_alarm_rate": fa_now,
                "noisy_rule": {"rate": engine.noisy_rate, "window_s": engine.noisy_window,
                               "min": engine.noisy_min}, "ws_connections": len(engine.hub.conns),
                "dropped_events": engine.hub.dropped,
                "tick_ms_avg": round(sum(ticks) / len(ticks), 3) if ticks else None,
                "tick_ms_max": round(max(ticks), 3) if ticks else None,
                "training": engine.train_run if (engine.train_proc is not None
                                                 and engine.train_proc.returncode is None) else None}

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        q = engine.hub.add()
        # snapshot goes first, built after registering so no event is missed between the two
        Hub.put(q, json.dumps(engine.snapshot(), separators=(",", ":")))

        async def reader():
            try:
                while True:
                    await ws.receive_text()     # we ignore client messages, but notice closes
            except Exception:  # noqa: BLE001
                pass

        rtask = asyncio.create_task(reader())
        try:
            while not rtask.done():
                get = asyncio.ensure_future(q.get())
                done, _ = await asyncio.wait({get, rtask}, return_when=asyncio.FIRST_COMPLETED)
                if get not in done:
                    get.cancel()
                    break
                await asyncio.wait_for(ws.send_text(get.result()), timeout=10)
        except (WebSocketDisconnect, asyncio.TimeoutError, RuntimeError, OSError):
            pass
        except Exception as exc:  # noqa: BLE001
            print(f"[ws] connection error: {exc!r}")
        finally:
            engine.hub.remove(q)
            rtask.cancel()
            try:
                await ws.close()
            except Exception:  # noqa: BLE001
                pass

    index = static_dir / "index.html"
    if index.is_file():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
        # index.html references app.js, style.css and fonts/ relatively, so serve at / too
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="root")
    else:
        @app.get("/")
        async def placeholder():
            return HTMLResponse(PLACEHOLDER)

        if static_dir.is_dir():
            app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    return app


def main(args):
    root = Path(os.path.expanduser(args.root)).resolve()
    data_dir = resolve(root, args.data_dir)
    runs_dir = resolve(root, args.runs_dir)
    model_path = resolve(root, args.model)
    prior_path = resolve(root, args.prior_json)
    static_dir = Path(os.path.expanduser(args.static_dir)).resolve()

    # check the cheapest, most likely missing input first
    if not (data_dir / "clients.json").is_file():
        fail(f"{data_dir / 'clients.json'} not found. Run `python3 demo/prepare_demo_data.py` "
             f"first (or pass --data-dir).")

    torch.set_num_threads(2)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    model, scaler, class_names, minfo = load_detector(model_path)
    prior, prior_src = load_prior(prior_path, class_names)
    data_meta, clients = load_clients(data_dir, scaler, model, len(class_names))
    engine = Engine(args, model, scaler, class_names, minfo, prior, prior_src, clients,
                    data_meta, root, runs_dir)

    print("=" * 72)
    print("FL-IDS live demo backend")
    acc = minfo.get("best_test_acc")
    acc_s = f", best_test_acc {float(acc):.4f}" if acc is not None else ""
    print(f"  model      {minfo['path']}")
    print(f"             type {minfo['model_type']}, best round {minfo['best_round']}{acc_s}")
    print(f"  prior      {prior_src}")
    print("             " + ", ".join(f"{c}={p:.3f}" for c, p in zip(class_names, prior)))
    print(f"  strength   {engine.strength}   rate {engine.rate} msg/s per client")
    print(f"  clients    (data from {data_dir})")
    for c in clients:
        pool = ", ".join(f"{t}={len(v)}" for t, v in sorted(c.pool_by_type.items()))
        print(f"    {c.id} {c.name:<18} cell {c.cell:<6} stream {c.stream.n:>7,} rows   "
              f"pool {c.pool.n if c.pool else 0:>6,} ({pool})")
    strengths = sorted({0.0, 0.5, 1.0, engine.strength})
    fa_tab = {s_: engine.expected_fa(s_) for s_ in strengths}
    print(f"  false alarm rate on each tower's benign stream (noisy rule: >= "
          f"{engine.noisy_rate:.0%} of benign msgs in {engine.noisy_window:g} s, "
          f"min {engine.noisy_min})")
    print("    strength " + "".join(f"{c.id:>7}" for c in clients))
    for s_ in strengths:
        cells = "".join(f"{fa_tab[s_][c.id]:>6.1%}" + ("!" if fa_tab[s_][c.id] >= engine.noisy_rate
                                                          else " ") for c in clients)
        print(f"    {s_:<8.2f} {cells}")
    hot = [c.id for c in clients if fa_tab[engine.strength][c.id] >= 0.8 * engine.noisy_rate]
    if hot:
        print(f"    WARNING: at strength {engine.strength} tower(s) {', '.join(hot)} are close to or "
              f"above the noisy threshold and will flicker 'noisy'. Raise --noisy-rate or "
              f"swap those cells.")
    print(f"  runs       {runs_dir} ({len(engine.list_runs())} found)")
    fe = "frontend " + (str(static_dir) if (static_dir / "index.html").is_file()
                        else "MISSING, serving the plain text placeholder")
    print(f"  {fe}")
    print(f"  open       http://localhost:{args.port}  (listening on {args.host}:{args.port})")
    print("=" * 72, flush=True)

    app = build_app(engine, static_dir)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning", access_log=False,
                ws_ping_interval=20, ws_ping_timeout=20)


if __name__ == "__main__":
    try:
        main(ARGS)
    except KeyboardInterrupt:
        pass
