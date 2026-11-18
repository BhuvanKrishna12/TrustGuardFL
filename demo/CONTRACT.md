# FL-IDS live demo: shared contract

Every component in `demo/` follows this file. If something here is wrong or missing,
fix it HERE and say so, do not silently diverge.

## Where it runs

- User's laptop: WSL2 Ubuntu, 13 GB RAM cap, CPU only, Python 3.12, pandas 3.x, torch 2.6+.
  Browser is on Windows, reaching WSL at `http://localhost:8000`.
- The `demo/` folder is dropped into `~/fl-ids-project/`, so the project root (one level up)
  holds `federated_simulation.py`, `evaluate_holdout.py`, `federated_simulation_flower.py`,
  `models/`, `results/`. Demo scripts import from the project root by adding
  `Path(__file__).resolve().parent.parent` to `sys.path`. Never copy or modify root files.
- Packages the user installs with `pip3 install --user --break-system-packages`.
  Allowed new deps: `flwr[simulation]` (pin one 1.x version that still has the legacy
  `fl.simulation.start_simulation`, which `federated_simulation_flower.py` uses),
  `fastapi`, `uvicorn[standard]`. Nothing else new. No Node, no build step.
- The demo must work with NO internet at demo time (fonts vendored locally, no CDN).

## User inputs (real paths on the laptop, all overridable by CLI flags)

| Input | Default path | What it is |
|---|---|---|
| Cached training sample | `models/sample_3m.pkl` | pickle of `{"meta": {...}, "df": DataFrame}`, 1,855,947 rows, columns include `region_cell`, `attack_type`, `icao24`, `time`, `lastposupdate` and the 13 FEATURE_COLS |
| Holdout | `~/opensky_data/labeled_fixed/holdout_feb04_1pct.csv` | 688,616 labeled rows from 2019-02-04, same columns. Booleans may be literal `True`/`False` strings; some numeric columns may read as text. Always read with `dtype=str` then convert through `federated_simulation.prep_features`. |
| Detection model | `models/fixed4_cnn.pt` + `models/fixed4_cnn_preprocess.pkl` | torch payload dict (`model_state_dict`, `model_type`, `n_features`, `class_names`, `cnn_channels`, `mlp_hidden_sizes`, `best_round`, `best_test_acc`), load with `torch.load(..., weights_only=False)`. The pkl (joblib) holds the fitted StandardScaler and LabelEncoder. `evaluate_holdout.py` already has robust reconstruction code; reuse it by import if practical. |
| Real class prior | `results/lgbm_feb04.json` key `real_prior` | `{"benign":0.794..,"flooding":0.131..,...}` |

Classes, in LabelEncoder order: `benign, flooding, ghost_aircraft, gps_spoof, replay, velocity_spoof`.

## The six clients

Default cells and display names (prep script verifies each has enough rows, and accepts `--cells`):

| id | cell | name |
|---|---|---|
| C1 | 4_-10 | US Midwest |
| C2 | 5_1 | Central Europe |
| C3 | 3_-9 | US Southeast |
| C4 | 0_10 | Malay Peninsula |
| C5 | -4_14 | SE Australia |
| C6 | 5_-14 | Gulf of Alaska |

Cell `a_b` means lat in [10a, 10a+10), lon in [10b, 10b+10); centre lat = 10a+5, lon = 10b+5.

## Files written by `demo/prepare_demo_data.py` (into `demo/data/`)

- `clients.json`: `{"created": "...", "source": {...paths...}, "clients": [{"id":"C1","cell":"4_-10","name":"US Midwest","lat":45,"lon":-95,"train_rows":N,"stream_rows":N,"pool_rows":{"gps_spoof":N,...}}, ...]}`
- `C1_train.csv` ... `C6_train.csv`: that region's rows from the cached sample (all classes), for Flower training. Columns: the 13 FEATURE_COLS + `attack_type`.
- `C1_stream.csv` ...: that region's BENIGN holdout rows sorted by `time`, columns: `time`, `icao24`, the 13 FEATURE_COLS, `attack_type` (always benign). This is the normal traffic each client streams.
- `C1_pool.csv` ...: that region's ATTACK holdout rows (all five attack types), same columns. Injection draws from here. If a region has fewer than 50 rows of an attack type, top up from the other five regions' rows of that type and record `pool_borrowed: {"type": N}` in clients.json.
All feature values written already numeric (run through prep_features semantics: bool strings to 1/0, coerce numerics, but DO NOT fill -1 in the CSV; leave NaN as empty so the server applies prep_features itself).

Clarifications (added by the data/training agent, no format change):
- `pool_borrowed` is ALWAYS present on every client (`{}` when nothing was borrowed). `pool_rows` counts
  every row in that client's pool file, borrowed rows included.
- `time` is unix seconds (integer when the source values are whole numbers). Pool files are sorted by `time` too.
- Boolean feature columns are written as `1`/`0` (empty when missing).

## Files written by `demo/train_demo.py`

- `demo/runs/<run_name>.jsonl`, one JSON object per line:
  - `{"type":"run_start","run":"...","clients":[{"id":"C1","cell":"4_-10","name":"...","train_rows":N}],"rounds":R,"model":"cnn","cnn_channels":[32,64],"payload_bytes":B,"t":<unix seconds>}`
  - `{"type":"round","round":r,"global_acc":0.81,"per_client":{"C1":{"acc":0.83,"n":1234},...},"bytes_this_round":B,"bytes_total":B,"t":...}`
  - `{"type":"client_fit","round":r,"client":"C3","train_acc":0.84,"n":1234,"t":...}` (optional, as clients finish)
  - `{"type":"run_end","best_round":r,"best_acc":0.83,"model_path":"demo/models/<run>.pt","t":...}`
- `demo/runs/latest.jsonl`: copy of the most recent completed run (the replay source).
- `demo/models/<run>.pt` + `_preprocess.pkl`: same payload format as fixed4_cnn.pt.
Flower: 6 clients, legacy `fl.simulation.start_simulation`, FedAvg sample-weighted, CNN channels 32,64,
default 15 rounds, `LOCAL_EPOCHS` as in federated_simulation. Ray kept small: `ray_init_args={"num_cpus":4,"object_store_memory":400_000_000,"include_dashboard":False}`, `client_resources={"num_cpus":1}`.

Additions (data/training agent):
- Extra optional event on failure or Ctrl+C: `{"type":"run_error","error":"RuntimeError: ...","t":...}`. It is
  the last line of a failed run; no `run_end` follows and `latest.jsonl` is NOT updated. Consumers should
  treat it as the end of a live run.
- `client_fit` lines for a round are written together just before that round's `round` line (Flower only
  hands results to the strategy once every client of the round is done). They are ordered by finish time
  and their `t` is the client's actual finish time, so `t` can be slightly earlier than the line before.
- `model_path` in `run_end` is relative to the project root when the file is inside it, else absolute.
- `ray_init_args` also carries `"logging_level": 40` (quieter console). Ray workers find
  `federated_simulation.py` through `PYTHONPATH`, which train_demo.py sets before Ray starts.
- `train_demo.py` CLI: `--rounds` (15), `--run-name` (default `run_YYYYmmdd_HHMMSS`, local time; `latest` is
  refused), `--model cnn|mlp`, `--cnn-channels 32,64`, `--mlp-hidden-sizes 64,32`, `--data-dir demo/data`,
  `--runs-dir demo/runs`, `--models-dir demo/models`, `--dry-run`. Exit code 0 on success, 1 on failure.
- Pins live in `demo/requirements-demo.txt`: flwr[simulation]==1.35.0 (ray 2.55.1). flwr 1.35.0 itself
  pins fastapi 0.138.x, uvicorn 0.49.x and starlette 1.3.x, so the server must work with those versions.
- Test fixture: `python3 demo/tools/make_fixture.py [--root /tmp/fl-demo-fixture]` builds a synthetic
  project root with the four real inputs (see the docstring at the top of that file).

## Server: `demo/server.py` (FastAPI + uvicorn, default port 8000)

CLI: `--model`, `--prior-json`, `--data-dir demo/data`, `--runs-dir demo/runs`, `--port`, `--rate` (messages per second per client, default 4), `--strength` (default 0.5), `--noisy-rate` (0.25), `--noisy-window` (20), `--noisy-min` (3).

Serves `demo/static/` at `/` (index.html) and `/static/...`.

### REST

| Method | Path | Body | Effect |
|---|---|---|---|
| GET | `/api/state` | | full snapshot (see `snapshot` event) |
| POST | `/api/inject` | `{"client":"C2","attack":"ghost_aircraft","count":25}` | queues `count` rows of that type from the client's pool into its stream, interleaved over the next few seconds. Returns `{"ok":true,"queued":25,"burst_id":"b7"}` |
| POST | `/api/strength` | `{"strength":0.5}` | sets correction strength, any float in [0,1] |
| POST | `/api/stream` | `{"running":true,"rate":4}` | pause/resume streaming, change rate |
| POST | `/api/reset` | | zero all counters and logs, keep streaming state |
| GET | `/api/training/runs` | | list of run names in runs dir |
| GET | `/api/training/replay?run=latest` | | the run's JSONL as a JSON array |
| POST | `/api/training/start` | `{"rounds":10}` | launches `train_demo.py` as a subprocess (live mode); progress arrives as `training` WS events. 409 if one is running. |

### WebSocket `/ws`

On connect, server sends one `snapshot`. Then events, each a JSON object with `type`:

- `snapshot`: `{"type":"snapshot","clients":[{"id","cell","name","lat","lon","status","counters":{"seen","flagged","correct","false_alarm","missed"},"recent":[<msg>...up to 12]}],"strength":0.5,"running":true,"rate":4,"model":{"path","model_type","best_round"},"incidents":[<incident>... up to 50],"attack_types":[...5]}`
- `msg`: `{"type":"msg","client":"C2","seq":1234,"time":"14:32:07","icao24":"f3a91c","pred":"ghost_aircraft","conf":0.99,"truth":"ghost_aircraft","injected":true,"burst_id":"b7","verdict":"correct_catch"}`
  `verdict` is one of `correct_catch` (truth attack, pred attack), `false_alarm` (truth benign, pred attack), `missed` (truth attack, pred benign), `ok` (benign, benign). Attack-to-wrong-attack counts as `correct_catch` for the alert but also carries `"wrong_type":true`.
- `incident`: emitted for every msg whose pred is not benign, same fields plus `"name"` of the client.
- `counters`: `{"type":"counters","client":"C2","counters":{...}}` at most 2 per second per client.
- `status`: `{"type":"status","client":"C2","status":"under_attack"|"noisy"|"normal"}`.
  Rule: `under_attack` if >= 3 correct catches in the last 10 seconds; else `noisy` if, over the last
  `--noisy-window` seconds (default 20), false alarms are at least `--noisy-rate` (default 0.25) of that
  client's truth-benign messages AND number at least `--noisy-min` (default 3); else `normal`.
  (Changed by the review agent, 2026-11-11. The old rule, `noisy` if >= 2 false alarms in 30 s, made every
  tower permanently noisy on real data: the paper model flags about 8.7% of real benign rows at strength
  0.5, about 5 false alarms per 30 s at 2 msg/s. A share is also independent of the stream rate. With the
  new defaults, simulated: 8.7% background gives 0% noisy time, 38.7% (strength 0) about 98%.)
- `config`: `{"type":"config","strength":0.5,"running":true,"rate":4}` after any change.
- `training`: any line of the training JSONL, forwarded live during live training.
- `toast`: `{"type":"toast","text":"..."}` for errors or info.

Timestamps shown are the row's original `time` rendered as HH:MM:SS UTC on 2019-02-04.

Clarifications (added by the frontend agent; the frontend tolerates the listed alternatives):

- `seq` is one server-wide, monotonically increasing integer (not per client), so the page can
  order incidents from different clients. In the snapshot, `recent` and `incidents` are ordered
  oldest first; the page sorts by `seq` anyway.
- `training` events: a JSONL line has its own `type` key (`run_start`, `round`, ...), which
  collides with the envelope's `type`. Send it wrapped:
  `{"type":"training","data":{"type":"round","round":3,...}}`.
  (Also accepted: the line under `"line"`, or flattened with the line type under `"event"`.)
- `GET /api/training/runs` returns `{"runs":["latest","run_20261110_1432",...]}` (a bare JSON array
  is also accepted). `GET /api/training/replay` returns a JSON array, or 404 with
  `{"detail":"..."}` when no run exists yet.
- `POST /api/stream` always receives both `running` and `rate`. Rates the UI offers: 1, 2, 4, 8.
- Error responses: any non-2xx with a JSON `detail` (FastAPI default) is shown as a toast.

Clarifications (added by the server agent; `demo/server.py` implements exactly these):

- Counter meanings: `seen` every scored msg; `flagged` pred is any attack; `correct` verdict
  `correct_catch` (includes wrong-type catches); `false_alarm` and `missed` as the verdicts.
- Every `msg` (and `incident`) always carries `burst_id` (`null` when not injected) and
  `wrong_type` (`false` unless an attack was flagged as a different attack type).
- Injection: a burst of N rows is spread over max(3, min(6, N/rate)) seconds of stream time
  (longer only when that would exceed twice the normal rate), with jitter. Injected rows are
  ADDED among the client's normal rows, so that client's feed runs faster during a burst.
  Pool rows are drawn without replacement within a burst, cycling a fresh shuffle when
  exhausted. Paused stream time does not advance, so a paused burst resumes where it was.
  `count` must be 1..500. Errors: 404 unknown client, 400 unknown attack or bad count,
  409 when the client's pool has no rows of that type.
- `POST /api/strength` clamps to [0,1]; `POST /api/stream` clamps `rate` to [0.25, 50]. Both
  reply with the resulting `{"ok":true,"strength","running","rate"}` and broadcast `config`.
- `POST /api/reset` also broadcasts a `counters` (all zeros) and `status` (`normal`) event per client.
- `POST /api/training/start` replies `{"ok":true,"run":"live_YYYYMMDD_HHMMSS","pid":N}`; 409 when a
  run is active; 503 plus a `toast` when train_demo.py is missing. It runs
  `<same python> -u demo/train_demo.py --rounds R --run-name <run> --data-dir <server data dir>
  --runs-dir <server runs dir>` from the project root. When the process ends a `toast` says
  `Training run <run> finished.` or `... failed (exit N): <last output lines>`.
- Extra endpoint `GET /api/health`: `{"ok","ws_connections","dropped_events","tick_ms_avg","tick_ms_max","training"}`.
- Extra server CLI flags: `--host` (default 0.0.0.0), `--root` (project root, default the parent of
  demo/), `--static-dir`, `--train-script`. The streamer ticks every 0.25 s; with the default
  rate 4 that is one msg per client per tick. A slow browser tab never blocks the server: each
  connection has a 2000 event queue that drops its oldest events when full.

### Scoring

Rows are scored with the detection model: `prep_features` then scaler (columns reordered to the scaler's
`feature_names_in_`) then model then softmax, then corrected score `p(k) * prior(k) ** strength`, argmax.
The raw softmax for every stream and pool row is computed once at startup, so the streamer tick does no
model work; the correction and argmax still run per message with the current strength. At startup the
server prints each client's expected false alarm rate at strengths 0, 0.5 and 1, and `GET /api/health`
adds `false_alarm_rate` (per client: observed share in the noisy window, expected share at the current
strength, status) and `noisy_rule`. `conf` is the corrected,
renormalised probability of the predicted class. Changing strength applies to subsequent messages only.

## Frontend: `demo/static/index.html` (+ `app.js`, `style.css`, `fonts/`)

Port of `demo/docs/MOCKUP_A_reference.dc.html` (the approved Flight Strip Board design) to plain HTML/CSS/JS,
fluid at 1366x768 up to 1920x1080 (laptop plus projector). Two views via a tab strip: "Live Detection"
(default) and "Training Run". Consumes only the REST + WS contract above. Has a `?mock=1` mode that fakes
the WebSocket so the page can be developed and shown without a backend.

## Integration changes (added by the integration agent, 2026-11-11)

Found while running all three parts together on a fixture; each item says why.

- `msg` / `incident` for an INJECTED row: `time` is now the client's current stream time (the
  time of the last benign row it streamed), and the row's real timestamp moves to an extra field
  `orig_time`. Why: pool rows come from any hour of Feb 4, so strips and the "newest on top" log
  jumped between e.g. 03:37 and 20:27. The page shows `orig_time` as a tooltip. Benign rows are
  unchanged and carry no `orig_time`.
- `train_demo.py --keep-latest`: leave an existing `runs/latest.jsonl` (and `models/latest.pt`)
  alone; if there is none yet the run still becomes latest. The server's `POST
  /api/training/start` now always passes it. Why: a quick 3 round "Train live" from the page (or
  test_server's 2 round launch) replaced the rehearsed 15 round replay. Live runs still appear by
  name in `GET /api/training/runs`.
- `train_demo.py` also copies the model that goes with `latest.jsonl` to
  `demo/models/latest.pt` + `latest_preprocess.pkl`, so `server.py --model demo/models/latest.pt`
  (already shown in the server docstring) works. The default stays `models/fixed4_cnn.pt`.
- The frontend treats `run_error` as the end of a live run (it used to stay in live mode), shows
  a "Starting Flower and Ray, about 15 seconds" state with an elapsed counter from the Train live
  click until the first `round`, and the training cards show `train_rows` from `run_start`
  (`per_client.n` is the 20% evaluation split, so it is only a fallback).
- `demo/start_demo.sh` is the one command launcher; it starts the server with `--rate 2`
  and `--noisy-rate 0.25` (env `NOISY_RATE` or `--noisy-rate`). The server's own default rate stays 4.
- `prepare_demo_data.py` exits 1 and writes no `clients.json` when a chosen cell has zero train or
  stream rows (so `start_demo.sh` does not skip the prep step on the next run). A `time` column that
  holds date text instead of unix seconds is converted to unix seconds.

