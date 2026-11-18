#!/usr/bin/env python3
"""
demo/tools/test_server.py: automated end to end test of demo/server.py.

By default it builds a small synthetic fixture (a fake project root with
demo/data/clients.json, six C*_stream.csv and C*_pool.csv files, a tiny CNN saved in the
real payload format with a matching _preprocess.pkl, a prior JSON and a short
runs/latest.jsonl), starts the server against it on a free port, runs every check,
measures the server's CPU and memory while streaming, and shuts it down.

Usage, from the project root:

    python3 demo/tools/test_server.py                 # fixture + server + all checks
    python3 demo/tools/test_server.py --keep          # keep the fixture folder afterwards
    python3 demo/tools/test_server.py --url http://localhost:8000
                                                      # test an already running server
                                                      # (real data: detection checks are soft)

Needs: httpx, websockets (plus torch, scikit-learn, joblib for the fixture).
Exit status 0 when every hard check passes.
"""

import argparse
import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import httpx
import websockets

DEMO_DIR = Path(__file__).resolve().parent.parent
ROOT = DEMO_DIR.parent
CLASSES = ["benign", "flooding", "ghost_aircraft", "gps_spoof", "replay", "velocity_spoof"]
CLIENTS = [("C1", "4_-10", "US Midwest", 45, -95), ("C2", "5_1", "Central Europe", 55, 15),
           ("C3", "3_-9", "US Southeast", 35, -85), ("C4", "0_10", "Malay Peninsula", 5, 105),
           ("C5", "-4_14", "SE Australia", -35, 145), ("C6", "5_-14", "Gulf of Alaska", 55, -135)]

RESULTS = []

FAKE_TRAIN = """
import argparse, json, time, pathlib
p = argparse.ArgumentParser()
p.add_argument('--rounds', type=int); p.add_argument('--run-name')
p.add_argument('--data-dir'); p.add_argument('--runs-dir')
a, _extra = p.parse_known_args()
out = pathlib.Path(a.runs_dir) / (a.run_name + '.jsonl')
print('fake trainer starting', flush=True)
with open(out, 'w') as fh:
    def w(o):
        fh.write(json.dumps(o)); fh.flush()      # write the line in two parts on purpose
        time.sleep(0.05); fh.write('\\n'); fh.flush()
    w({'type': 'run_start', 'run': a.run_name, 'rounds': a.rounds, 't': time.time()})
    for r in range(1, a.rounds + 1):
        time.sleep(0.4)
        w({'type': 'round', 'round': r, 'global_acc': 0.8, 't': time.time()})
    w({'type': 'run_end', 'best_round': a.rounds, 'best_acc': 0.8, 't': time.time()})
print('fake trainer done', flush=True)
"""


def check(name, ok, detail="", hard=True):
    tag = "PASS" if ok else ("FAIL" if hard else "WARN")
    RESULTS.append((tag, name, detail))
    print(f"  [{tag}] {name}" + (f": {detail}" if detail else ""), flush=True)
    return ok


# --------------------------------------------------------------------------------------
# Fixture
# --------------------------------------------------------------------------------------

def build_fixture(base: Path):
    """Separable synthetic data: each attack class pushes one continuous feature far out."""
    import numpy as np
    import pandas as pd
    import torch
    import joblib
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(DEMO_DIR))
    import federated_simulation as fs
    import fs_compat

    rng = np.random.default_rng(7)
    cols = fs.FEATURE_COLS
    bool_cols = ["is_physically_implausible", "squawk_changed", "icao24_valid_format",
                 "is_stale", "is_low_coverage_region", "is_climbing_or_descending"]
    signal_col = {"flooding": "inter_arrival_s", "ghost_aircraft": "track_message_count",
                  "gps_spoof": "implied_speed_mps", "replay": "speed_consistency_delta",
                  "velocity_spoof": "vrate_consistency_delta"}

    def rows(cls, n, t0):
        d = {}
        for c in cols:
            if c in bool_cols:
                d[c] = rng.integers(0, 2, n).astype(float)
            else:
                d[c] = rng.normal(5.0, 1.0, n)
        if cls != "benign":
            d[signal_col[cls]] = rng.normal(40.0, 2.0, n)
        df = pd.DataFrame(d)
        df.insert(0, "icao24", [f"{x:06x}" for x in rng.integers(0, 0xFFFFFF, n)])
        df.insert(0, "time", (t0 + np.sort(rng.integers(0, 86_000, n))).astype("int64"))
        df["attack_type"] = cls
        return df

    t0 = 1549238400  # 2019-02-04 00:00:00 UTC
    data = base / "demo" / "data"
    data.mkdir(parents=True)
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "source": {"fixture": True},
            "clients": []}
    for cid, cell, name, lat, lon in CLIENTS:
        stream = rows("benign", 400, t0)
        # exercise prep_features: literal True/False strings and empty values
        for bc in bool_cols[:3]:
            stream[bc] = stream[bc].map({1.0: "True", 0.0: "False"})
        stream.loc[stream.index[::37], "turn_rate_deg_s"] = np.nan
        stream.to_csv(data / f"{cid}_stream.csv", index=False)
        pool = pd.concat([rows(a, 60, t0) for a in CLASSES[1:]]).sort_values("time")
        pool.to_csv(data / f"{cid}_pool.csv", index=False)
        meta["clients"].append({"id": cid, "cell": cell, "name": name, "lat": lat, "lon": lon,
                                "train_rows": 0, "stream_rows": 400,
                                "pool_rows": {a: 60 for a in CLASSES[1:]}})
    (data / "clients.json").write_text(json.dumps(meta, indent=2))

    # train a tiny CNN (channels 32,64) on fresh draws of the same distribution
    train = pd.concat([rows(c, 600, t0) for c in CLASSES]).reset_index(drop=True)
    X = fs_compat.prep_features(fs, train)
    le = LabelEncoder()
    y = le.fit_transform(train["attack_type"])
    assert list(le.classes_) == CLASSES
    scaler = StandardScaler().fit(X)
    Xs = torch.tensor(scaler.transform(X), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    torch.manual_seed(0)
    model = fs.build_model("cnn", len(cols), len(CLASSES), cnn_channels=(32, 64),
                           device=torch.device("cpu"))
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(150):
        idx = torch.randint(0, len(Xs), (256,))
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(Xs[idx]), yt[idx])
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (model(Xs).argmax(1) == yt).float().mean().item()
    models = base / "models"
    models.mkdir()
    payload = {"model_state_dict": model.state_dict(), "model_type": "cnn",
               "n_features": len(cols), "feature_cols": cols, "class_names": CLASSES,
               "mlp_hidden_sizes": (64, 32), "cnn_channels": (32, 64), "seq_window_size": 10,
               "best_round": 7, "n_rounds": 15, "best_test_acc": np.float64(acc),
               "final_test_acc": np.float64(acc)}
    torch.save(payload, models / "fx_cnn.pt")
    joblib.dump({"scaler": scaler, "label_encoder": le}, models / "fx_cnn_preprocess.pkl")

    results = base / "results"
    results.mkdir()
    prior = {"benign": 0.794, "flooding": 0.131, "ghost_aircraft": 0.019,
             "gps_spoof": 0.019, "replay": 0.019, "velocity_spoof": 0.018}
    (results / "prior.json").write_text(json.dumps({"real_prior": prior, "other": 1}))

    runs = base / "demo" / "runs"
    runs.mkdir()
    lines = [{"type": "run_start", "run": "fx", "clients": [
        {"id": c[0], "cell": c[1], "name": c[2], "train_rows": 1000} for c in CLIENTS],
        "rounds": 3, "model": "cnn", "cnn_channels": [32, 64], "payload_bytes": 40000, "t": 1}]
    for r in (1, 2, 3):
        lines.append({"type": "round", "round": r, "global_acc": 0.7 + r / 20,
                      "per_client": {c[0]: {"acc": 0.7, "n": 1000} for c in CLIENTS},
                      "bytes_this_round": 480000, "bytes_total": 480000 * r, "t": 1 + r})
    lines.append({"type": "run_end", "best_round": 3, "best_acc": 0.85,
                  "model_path": "demo/models/fx.pt", "t": 5})
    (runs / "latest.jsonl").write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    # a stand in for train_demo.py: same CLI, writes the JSONL slowly, exits 0
    (base / "fake_train.py").write_text(FAKE_TRAIN)
    print(f"  fixture at {base} (toy model train acc {acc:.3f})")
    return len(lines)


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

class Feed:
    """A WebSocket consumer that records every event with its arrival time."""

    def __init__(self, url):
        self.url = url
        self.events = []
        self.ws = None
        self.task = None

    async def start(self):
        self.ws = await websockets.connect(self.url, max_size=2 ** 24)
        first = json.loads(await self.ws.recv())
        self.events.append((time.monotonic(), first))
        self.task = asyncio.create_task(self._run())
        return first

    async def _run(self):
        try:
            async for raw in self.ws:
                self.events.append((time.monotonic(), json.loads(raw)))
        except Exception:  # noqa: BLE001
            pass

    def since(self, t, typ=None):
        return [e for (ts, e) in self.events if ts >= t and (typ is None or e["type"] == typ)]

    async def close(self):
        if self.task:
            self.task.cancel()
        await self.ws.close()


def proc_sample(pid):
    """(cpu seconds, rss MB) from /proc, including child threads."""
    with open(f"/proc/{pid}/stat") as fh:
        parts = fh.read().rsplit(")", 1)[1].split()
    tck = os.sysconf("SC_CLK_TCK")
    cpu = (int(parts[11]) + int(parts[12])) / tck
    rss = 0.0
    with open(f"/proc/{pid}/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                rss = int(line.split()[1]) / 1024
    return cpu, rss


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

async def run_tests(base_url, pid, rate, expect_detect, n_replay, fake_train=False):
    ws_url = base_url.replace("http", "ws", 1) + "/ws"
    async with httpx.AsyncClient(base_url=base_url, timeout=10) as http:
        print("\n1. snapshot")
        feed = Feed(ws_url)
        snap = await feed.start()
        ok = snap.get("type") == "snapshot"
        keys = {"clients", "strength", "running", "rate", "model", "incidents", "attack_types"}
        ok = ok and keys <= set(snap)
        cl = snap.get("clients", [])
        ckeys = {"id", "cell", "name", "lat", "lon", "status", "counters", "recent"}
        ok = ok and len(cl) == 6 and all(ckeys <= set(c) for c in cl)
        ok = ok and all(set(c["counters"]) == {"seen", "flagged", "correct", "false_alarm",
                                               "missed"} for c in cl)
        ok = ok and {"path", "model_type", "best_round"} <= set(snap["model"])
        ok = ok and len(snap["attack_types"]) == 5
        check("snapshot shape", ok, f"model={snap.get('model')}, attack_types={snap.get('attack_types')}")
        r = await http.get("/api/state")
        check("GET /api/state", r.status_code == 200 and r.json().get("type") == "snapshot")
        r = await http.get("/")
        check("GET / serves a page", r.status_code == 200 and "<html" in r.text.lower())

        print("\n2. streaming rate")
        t0 = time.monotonic()
        await asyncio.sleep(6.0)
        msgs = [m for m in feed.since(t0 + 1.0, "msg") if not m["injected"]]
        per = {c["id"]: 0 for c in cl}
        for m in msgs:
            per[m["client"]] += 1
        exp = rate * 5.0
        ok = all(abs(n - exp) <= max(3, 0.25 * exp) for n in per.values())
        check("msg rate per client", ok, f"expected ~{exp:.0f} in 5 s, got {per}")
        m0 = msgs[0] if msgs else {}
        mkeys = {"type", "client", "seq", "time", "icao24", "pred", "conf", "truth", "injected",
                 "burst_id", "verdict"}
        check("msg shape", mkeys <= set(m0), json.dumps(m0))
        seqs = [m["seq"] for m in feed.since(t0, "msg")]
        check("seq strictly increasing", seqs == sorted(seqs) and len(set(seqs)) == len(seqs))
        cnt = feed.since(t0 + 1.0, "counters")
        perc = {}
        for c in cnt:
            perc[c["client"]] = perc.get(c["client"], 0) + 1
        check("counters throttled (<= 2/s/client)", all(v <= 2 * 5 + 1 for v in perc.values()),
              f"counters events in 5 s: {perc}")

        print("\n3. inject C2 ghost_aircraft x25")
        t_inj = time.monotonic()
        r = await http.post("/api/inject", json={"client": "C2", "attack": "ghost_aircraft",
                                                 "count": 25})
        body = r.json()
        check("POST /api/inject", r.status_code == 200 and body.get("ok") and body.get("queued") == 25
              and str(body.get("burst_id", "")).startswith("b"), json.dumps(body))
        bid = body.get("burst_id")
        deadline = t_inj + 10.0
        while time.monotonic() < deadline:
            got = [m for m in feed.since(t_inj, "msg") if m.get("burst_id") == bid]
            if len(got) >= 25:
                break
            await asyncio.sleep(0.2)
        got = [(ts, m) for ts, m in feed.events if ts >= t_inj and m["type"] == "msg"
               and m.get("burst_id") == bid]
        span = (got[-1][0] - got[0][0]) if got else 0
        check("25 injected msgs within 10 s", len(got) == 25 and all(
            m["client"] == "C2" and m["injected"] and m["truth"] == "ghost_aircraft"
            for _, m in got), f"got {len(got)}, spread over {span:.1f} s")
        check("burst is spread, not a dump", 2.0 <= span <= 7.0, f"{span:.1f} s")
        normal_during = [m for m in feed.since(got[0][0] if got else t_inj, "msg")
                         if m["client"] == "C2" and not m["injected"]
                         and (not got or m["seq"] < got[-1][1]["seq"])]
        check("normal C2 rows interleaved with the burst", len(normal_during) >= 5,
              f"{len(normal_during)} normal C2 rows inside the burst window")
        caught = sum(1 for _, m in got if m["verdict"] == "correct_catch")
        check("injected rows detected", caught >= 20, f"{caught}/25 correct_catch",
              hard=expect_detect)
        await asyncio.sleep(0.5)
        st = [e for e in feed.since(t_inj, "status") if e["client"] == "C2"]
        check("C2 status -> under_attack", any(e["status"] == "under_attack" for e in st),
              f"C2 status events: {[e['status'] for e in st]}", hard=expect_detect)
        inc = [e for e in feed.since(t_inj, "incident") if e.get("burst_id") == bid]
        check("incident events carry name", bool(inc) and all(e.get("name") for e in inc),
              f"{len(inc)} incidents, name={inc[0].get('name') if inc else None}")
        r = await http.post("/api/inject", json={"client": "C9", "attack": "ghost_aircraft",
                                                 "count": 5})
        check("inject unknown client -> 404", r.status_code == 404)
        r = await http.post("/api/inject", json={"client": "C2", "attack": "nope", "count": 5})
        check("inject unknown attack -> 400", r.status_code == 400)

        print("\n4. strength")
        t1 = time.monotonic()
        r = await http.post("/api/strength", json={"strength": 0.3})
        await asyncio.sleep(0.5)
        cfg = feed.since(t1, "config")
        check("POST /api/strength -> config event", r.status_code == 200 and bool(cfg)
              and abs(cfg[-1]["strength"] - 0.3) < 1e-9, json.dumps(cfg[-1] if cfg else None))
        # put the dial back where it was, so a page opened afterwards is not surprised
        await http.post("/api/strength", json={"strength": snap.get("strength", 0.5)})

        print("\n5. pause and resume")
        t_p = time.monotonic()
        r = await http.post("/api/stream", json={"running": False})
        await asyncio.sleep(0.6)
        t2 = time.monotonic()
        await asyncio.sleep(2.0)
        during = feed.since(t2, "msg")
        cfg = feed.since(t_p, "config")
        check("pause stops msgs", r.status_code == 200 and not during and bool(cfg)
              and cfg[-1]["running"] is False, f"{len(during)} msgs while paused")

        print("\n6. reset (while paused)")
        r = await http.post("/api/reset")
        s = (await http.get("/api/state")).json()
        zero = all(all(v == 0 for v in c["counters"].values()) and not c["recent"]
                   for c in s["clients"]) and not s["incidents"]
        check("POST /api/reset zeroes counters, recent, incidents", r.status_code == 200 and zero)

        t3 = time.monotonic()
        r = await http.post("/api/stream", json={"running": True, "rate": rate})
        await asyncio.sleep(2.0)
        after = feed.since(t3, "msg")
        check("resume restarts msgs", r.status_code == 200 and len(after) >= 6 * rate,
              f"{len(after)} msgs in 2 s after resume")

        print("\n7. training endpoints")
        r = await http.get("/api/training/runs")
        check("GET /api/training/runs", r.status_code == 200 and "latest" in r.json()["runs"], r.text[:120])
        r = await http.get("/api/training/replay", params={"run": "latest"})
        ok = r.status_code == 200 and isinstance(r.json(), list)
        check("GET /api/training/replay?run=latest", ok and (n_replay is None or len(r.json()) == n_replay),
              f"{len(r.json()) if ok else r.status_code} lines")
        r = await http.get("/api/training/replay", params={"run": "../etc/passwd"})
        check("replay rejects path traversal", r.status_code in (400, 404))
        r = await http.get("/api/training/replay", params={"run": "nope"})
        check("replay unknown run -> 404", r.status_code == 404)

        print("\n8. fan-out: 3 browser-like connections plus one stalled one")
        feeds = [Feed(ws_url) for _ in range(3)]
        for f in feeds:
            await f.start()
        stalled = await websockets.connect(ws_url, max_queue=1, max_size=2 ** 24)  # never reads
        t4 = time.monotonic()
        await asyncio.sleep(4.0)
        sets = [set(m["seq"] for m in f.since(t4 + 0.5, "msg")) for f in feeds]
        common = set.intersection(*sets) if sets else set()
        ok = all(len(sx) > 0 for sx in sets) and all(len(sx - common) <= 2 * 6 for sx in sets)
        check("3 tabs receive the same msgs", ok, f"counts {[len(sx) for sx in sets]}, common {len(common)}")
        n_rate = len([m for m in feed.since(t4 + 1, "msg")])
        check("stalled tab does not slow others", n_rate >= 0.8 * 6 * rate * 3,
              f"{n_rate} msgs in 3 s on the first tab")
        h = (await http.get("/api/health")).json()
        check("health endpoint", h.get("ok") is True, json.dumps(h))
        for f in feeds:
            await f.close()
        await stalled.close()

        print("\n9. resource use while streaming (rate %g, 6 clients, 1 tab), 20 s" % rate)
        if pid:
            c0, _ = proc_sample(pid)
            w0 = time.monotonic()
            rss_max = 0
            for _ in range(20):
                await asyncio.sleep(1.0)
                _, rss = proc_sample(pid)
                rss_max = max(rss_max, rss)
            c1, rss = proc_sample(pid)
            cpu_pct = 100 * (c1 - c0) / (time.monotonic() - w0)
            check("resource use", True, f"CPU {cpu_pct:.1f}% of one core, RSS {rss:.0f} MB "
                  f"(max {rss_max:.0f} MB)")
            h = (await http.get("/api/health")).json()
            print(f"     tick time avg {h.get('tick_ms_avg')} ms, max {h.get('tick_ms_max')} ms")
        else:
            check("resource use", True, "skipped (no local pid)", hard=False)

        print("\n10. live training launch")
        t5 = time.monotonic()
        r1 = await http.post("/api/training/start", json={"rounds": 2})
        r2 = await http.post("/api/training/start", json={"rounds": 2})
        if r1.status_code == 200:
            check("second start while running -> 409", r2.status_code == 409,
                  f"first {r1.status_code} {r1.text[:80]}, second {r2.status_code}", hard=fake_train)
            deadline = time.monotonic() + 25
            while time.monotonic() < deadline:
                if feed.since(t5, "toast"):
                    break
                await asyncio.sleep(0.5)
            tr = feed.since(t5, "training")
            toasts = feed.since(t5, "toast")
            kinds = [e.get("data", {}).get("type") for e in tr]
            if fake_train:
                check("live training forwards every JSONL line as training events",
                      kinds == ["run_start", "round", "round", "run_end"]
                      and any("finished" in t["text"] for t in toasts),
                      f"{kinds}, toasts: {[t['text'][:100] for t in toasts]}")
                r = await http.get("/api/training/runs")
                check("live run listed afterwards", any(x.startswith("live_") for x in r.json()["runs"]),
                      r.text[:120])
            else:
                check("live training produced events or a toast", bool(tr) or bool(toasts),
                      f"{len(tr)} training events, toasts: {[t['text'][:160] for t in toasts]}",
                      hard=False)
        else:
            check("training start fails gracefully", r1.status_code in (409, 503),
                  f"{r1.status_code} {r1.text[:120]}", hard=False)
        await feed.close()


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", help="test an already running server instead of a fixture")
    ap.add_argument("--rate", type=float, default=4.0)
    ap.add_argument("--keep", action="store_true", help="keep the fixture folder")
    ap.add_argument("--fixture-dir", help="build the fixture here (default: a temp dir)")
    args = ap.parse_args()

    if args.url:
        asyncio.run(run_tests(args.url.rstrip("/"), None, args.rate, False, None))
    else:
        base = Path(args.fixture_dir or tempfile.mkdtemp(prefix="fl_demo_fixture_"))
        if base.exists() and any(base.iterdir()):
            shutil.rmtree(base)
        base.mkdir(parents=True, exist_ok=True)
        print("Building fixture")
        n_replay = build_fixture(base)
        port = free_port()
        cmd = [sys.executable, str(DEMO_DIR / "server.py"), "--root", str(base),
               "--model", "models/fx_cnn.pt", "--prior-json", "results/prior.json",
               "--port", str(port), "--host", "127.0.0.1", "--rate", str(args.rate),
               "--train-script", str(base / "fake_train.py")]
        log = open(base / "server.log", "w")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT)
        url = f"http://127.0.0.1:{port}"
        try:
            for _ in range(120):
                if proc.poll() is not None:
                    break
                try:
                    if httpx.get(url + "/api/health", timeout=1).status_code == 200:
                        break
                except Exception:  # noqa: BLE001
                    time.sleep(0.25)
            if proc.poll() is not None:
                print((base / "server.log").read_text())
                sys.exit("server exited during startup")
            print("Server startup log:")
            print("    " + (base / "server.log").read_text().replace("\n", "\n    "))
            asyncio.run(run_tests(url, proc.pid, args.rate, True, n_replay, fake_train=True))
        finally:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.close()
            tail = (base / "server.log").read_text().splitlines()[-15:]
            print("\nServer log tail:\n    " + "\n    ".join(tail))
            if not args.keep:
                shutil.rmtree(base, ignore_errors=True)
            else:
                print(f"fixture kept at {base}")

    fails = [r for r in RESULTS if r[0] == "FAIL"]
    warns = [r for r in RESULTS if r[0] == "WARN"]
    print(f"\n{len(RESULTS) - len(fails) - len(warns)} passed, {len(warns)} warnings, "
          f"{len(fails)} failed")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
