# AeroGuard live demo

A browser demo of the federated intrusion detection project on ADS-B data. It has two views:

- **Live Detection**: six regional "towers" (clients) each stream their own benign ADS-B
  messages from the 2019-02-04 holdout. Every message is scored by the detection model and
  shown as a flight strip. You can inject real labelled attacks into one tower and watch the
  model catch them, and you can turn the correction strength dial to show the trade-off
  between false alarms and missed attacks.
- **Training Run**: a federated training run with Flower over the same six regions, either
  replayed from a saved run (one round per second) or trained live in front of the audience.
  It shows the global accuracy per round, each client's local accuracy, and how many bytes of
  model weights moved (in a real deployment raw messages would never leave a region; here the
  regions are simulated on one laptop).

Everything runs on your laptop. No internet is needed at demo time.

## One-time setup

The demo/ folder sits inside the repository root, next to `federated_simulation.py`
(`evaluate_holdout.py` is optional: without it the server uses its built-in model loader):

```
AeroGuard/
    federated_simulation.py   ...
    models/sample_3m.pkl
    models/fixed4_cnn.pt      models/fixed4_cnn_preprocess.pkl
    results/lgbm_feb04.json
    demo/                     <- this folder
```

The holdout is expected at `~/opensky_data/labeled_fixed/holdout_feb04_1pct.csv`.

You already have torch, pandas, scikit-learn and joblib. Install the three demo packages
(Flower, FastAPI, uvicorn; the versions are pinned to work together) once:

```
cd AeroGuard
pip3 install --user --break-system-packages -r demo/requirements-demo.txt
```

## Running it

```
cd AeroGuard
bash demo/start_demo.sh
```

Then open **http://localhost:8000** in Chrome or Edge on Windows. Keep the WSL terminal open.
Press **Ctrl+C** in that terminal to stop everything.

The first run does two extra steps, and later runs skip them:

1. **Prepare data** (`demo/prepare_demo_data.py`): cuts each region's rows out of the cached
   sample and the holdout into `demo/data/`. It prints a table of rows per region at the end.
2. **Train** (`demo/train_demo.py`, 15 rounds): runs Flower once and saves the run to
   `demo/runs/latest.jsonl`. This is what the Replay button plays.

Useful options:

| Command | What it does |
|---|---|
| `bash demo/start_demo.sh --retrain` | train again (for example after changing code) |
| `bash demo/start_demo.sh --rounds 20 --retrain` | train again with 20 rounds |
| `bash demo/start_demo.sh --reprep` | rebuild `demo/data` (it then retrains too) |
| `bash demo/start_demo.sh --demo-model` | detect with the 6-client demo model instead of `fixed4_cnn.pt` |
| `bash demo/start_demo.sh --port 8001` | use another port |
| `bash demo/start_demo.sh --rate 4` | 4 messages per second per tower (default 2) |
| `bash demo/start_demo.sh --noisy-rate 0.3` | share of a tower's benign messages (over 20 s) that must be false alarms before it shows "noisy" (default 0.25) |
| `HOLDOUT=/path/to/holdout.csv bash demo/start_demo.sh` | holdout somewhere else (also `SAMPLE=`, `MODEL=`, `PRIOR=`) |

You can also run the three parts by hand from the project root:

```
python3 demo/prepare_demo_data.py --holdout ~/opensky_data/labeled_fixed/holdout_feb04_1pct.csv
python3 demo/train_demo.py --rounds 15
python3 demo/server.py --rate 2 --strength 0.5
python3 demo/server.py --model demo/models/latest.pt     # detect with the demo-trained model
```

## Changing the six regions

The default regions are 10 by 10 degree cells. Cell `a_b` covers latitude 10a to 10a+10 and
longitude 10b to 10b+10, so `4_-10` is lat 40 to 50, lon -100 to -90 (US Midwest).

| Client | Cell | Name |
|---|---|---|
| C1 | 4_-10 | US Midwest |
| C2 | 5_1 | Central Europe |
| C3 | 3_-9 | US Southeast |
| C4 | 0_10 | Malay Peninsula |
| C5 | -4_14 | SE Australia |
| C6 | 5_-14 | Gulf of Alaska |

If the prepare step prints a **WARNING** that a cell is too thin (under 2,000 stream rows or
3,000 training rows, or no rows at all of some attack type), pick another cell from the
"densest cells in the holdout" table it prints, and run:

```
CELLS="4_-10,5_1,3_-9,0_10,-4_14,4_0" \
NAMES="US Midwest,Central Europe,US Southeast,Malay Peninsula,SE Australia,Western Europe" \
bash demo/start_demo.sh --reprep
```

Keep six cells. `NAMES` is optional (an unknown cell is shown as "Cell 4_0"). If a region has
fewer than 50 rows of an attack type, the prepare step borrows rows of that type from the
other regions so the inject button still works; the table shows these as "borrowed".

## Demo script for your mentor (about 5 minutes)

Rehearse it once the day before, on the real data, so you know the numbers you will see.

1. **Training Run tab, press Replay run.** "Six regions each train on their own ADS-B
   traffic. Only model weights travel, never raw messages. After each round the server
   averages the weights, weighted by how many rows each region has (FedAvg)." Point at the
   global accuracy line and at "Bytes moved".
2. **Point at the weakest client card.** On the real data this should be Gulf of Alaska (C6):
   it has the least traffic and different traffic, so its local accuracy sits lowest. This is
   the non-IID problem federated learning has to cope with. Check in your rehearsal that C6 is
   really the lowest card; if another one is, point at that one.
3. **Live Detection tab.** "Each tower is now streaming its real benign messages from the
   Feb 4 holdout, and every message is scored by the paper's model." The counters run, the
   strips say benign.
4. **Inject ghost_aircraft into C2 Central Europe** (Mischief Desk: target C2, ghost_aircraft,
   burst 25, Pull to inject). Within a few seconds C2 turns red (UNDER ATTACK), GHOST stamps
   appear, and the incident log fills with correct catches.
5. **Turn the dial to 0, then to 1.0, then back to 0.5.** At 0 the model is jumpy: OOPS
   (false alarm) strips appear at once, and towers turn "noisy" after about 10 seconds (the
   rule looks at the last 20 seconds). Wait for it before you move on. At 0.5 you still see an
   occasional OOPS strip (about 1 in 11 benign messages on the real data) but towers stay
   normal. At 1.0 it is calm, but it misses more attacks. "The dial is the class prior correction from the paper: it scales each class by
   how common it really is."
6. **Inject replay into another tower.** Some rows come back MISSED. "Replay is the hard
   class: the messages are real, just old, so a single message looks normal."
7. **Optional: Train live with 3 rounds.** The page says "Starting Flower and Ray, about 15
   seconds" first, then the rounds land one by one. A live run does not replace the saved
   replay; it appears by name in "Run on file".

Press **Reset counters** before you start so all counters begin at zero.

### Rehearsal checks on the real data

- **Read the false alarm table the server prints at startup.** It shows, per tower, the share
  of its benign stream the model flags at strength 0, 0.5 and 1.0. A `!` marks a tower at or
  above the noisy threshold (25%). You want no `!` in the 0.5 row and `!` in the 0.0 row. If a
  tower is close to 25% at 0.5 it will flicker "noisy" all demo: swap that cell (`CELLS=...
  --reprep`) or raise the threshold (`--noisy-rate 0.35`). `http://localhost:8000/api/health`
  shows the same numbers live (`false_alarm_rate`).
- **Inject replay (burst 25) once.** About 7 in 10 rows should be caught; the tower should go
  UNDER ATTACK within a second or two and back to normal about 15 seconds after the burst.
- **Time a 3 round "Train live"** with the server running, so you know how long the audience
  waits (Ray start plus a few seconds per round on 2 cores in testing).

## Troubleshooting

**"port 8000 is already in use".** The demo is probably still running in another terminal.
Stop it there with Ctrl+C, or find it with `ss -ltnp | grep :8000` and stop that process.
Or use `--port 8001` and open http://localhost:8001.

**Windows browser cannot reach http://localhost:8000.**
- Try http://127.0.0.1:8000.
- In `C:\Users\<you>\.wslconfig`, under `[wsl2]`, make sure `localhostForwarding=true` is set
  (or not set at all, it is on by default). Then run `wsl --shutdown` in PowerShell and open
  Ubuntu again.
- As a fallback, run `hostname -I` in WSL and open `http://<that address>:8000` in Windows.
  The server listens on all interfaces, so this works too.

**Memory.** Measured on a test setup: the server uses about 0.7 GB; training uses up to about
4 GB in total while Ray runs (the main process plus 4 worker processes, each loading torch).
Preparing the data loads the 1.86M row cached sample once: a synthetic sample of the same
row count peaked at 1.4 GB, so expect roughly 2 to 3 GB on your real file. The server's
startup takes longer on the real data (about 10 seconds for 490k rows in testing) because it
scores every stream and pool row once. All of this fits
in the 13 GB WSL limit, but close other heavy programs before a live training run. If WSL
gets killed, check `memory=` in `.wslconfig`.

**Ray is slow to start, or training hangs.** Ray takes 10 to 20 seconds to start; the
Training tab shows "Starting Flower and Ray" during that time. If nothing happens for over two
minutes, press Ctrl+C, run `ray stop --force`, and start again. Warnings that
`start_simulation()` is deprecated are expected (the project uses the older Flower API on
purpose). When you press Ctrl+C during a live run, Ray may print a long block of stack trace
lines: that is Ray shutting down, not an error in the demo.

**Fonts look plain or wrong.** The fonts are stored in `demo/static/fonts/`, so no internet is
needed. If the page looks unstyled, press Ctrl+F5 to reload without the browser cache, and
check that the `fonts/` folder was copied along with `demo/`.

**The page says "Link: reconnecting".** The server is not running or was restarted. The page
reconnects on its own as soon as the server is back.

**The Replay button says there is no run.** Run `bash demo/start_demo.sh --retrain`.

**The prepare step stops with "these cells have NO rows".** One of the six cells is missing
from the sample or the holdout. Pick a dense cell from the table it prints and rerun with
`CELLS=... bash demo/start_demo.sh --reprep`.

**Every tower is "noisy" even at 0.5.** Your model's false alarm rate is above the threshold
on these regions. Check the startup table, then raise `--noisy-rate`.

## What is real and what is simulated

Be ready to say this plainly if asked:

- **Federated training is a simulation on one laptop.** Flower runs the six clients as Ray
  worker processes on your machine (`fl.simulation.start_simulation`). The FedAvg logic, the
  per-client data split and the weight exchange are real Flower code, but there is no network
  between real devices. "Bytes moved" counts the model weights each client would send and
  receive.
- **Detection uses the paper model.** Live Detection scores messages with
  `models/fixed4_cnn.pt`, the model from the paper, not the small model that the 6-client
  demo training produces. Start with `--demo-model` (or `server.py --model
  demo/models/latest.pt`) to use the demo-trained model instead. Hover over the model box in
  the page header to see which model file is loaded.
- **Normal traffic is real.** Each tower streams that region's benign rows from the Feb 4
  holdout, in time order. It is replayed at 2 messages per second per tower by default (the
  page offers 1 to 8, all much slower than real ADS-B), and it loops when it reaches the end.
- **Scoring happens in one place.** The server scores every tower's messages with one copy of
  the model (it computes the model output for all rows once at startup and applies the dial
  live). In a real deployment each tower would run the global model itself.
- **"Noisy" and "under attack" are display rules, not model output.** A tower shows UNDER
  ATTACK after 3 correct catches within 10 seconds, and "noisy" when at least 25% of its benign
  messages in the last 20 seconds were false alarms (and at least 3 of them).
- **Injected attacks are real labelled rows.** The inject button draws attack rows of the
  chosen type from that region's Feb 4 holdout, labelled by the project pipeline (a few may
  be borrowed from other regions, see above). They are mixed in among the normal messages
  over a few seconds. The time shown for an injected row is the tower's current stream time
  so the feed stays in order; hover over the INJ tag or the log row to see the row's real
  time.
- **The dial is the paper's class prior correction.** Each class probability is multiplied by
  that class's real share of traffic (from `results/lgbm_feb04.json`) to the power of the
  dial value, then renormalised. 0 means no correction, 1 means full correction.
- **Training Run accuracies are from the demo training.** They are measured on a 20% split of
  each region's rows from the cached sample, not on the Feb 4 holdout, so they are not the
  numbers in the paper.

## Files

| File | Purpose |
|---|---|
| `start_demo.sh` | one command launcher (checks, prep, training, server) |
| `prepare_demo_data.py` | builds `demo/data/` from the cached sample and the holdout |
| `train_demo.py` | Flower training over the six regions, writes `demo/runs/` and `demo/models/` |
| `server.py` | FastAPI server: streaming, scoring, injection, training replay and launch |
| `fs_compat.py` | converts CSV-read feature columns before `federated_simulation.prep_features` |
| `static/` | the web page (plain HTML, CSS and JavaScript, fonts included) |
| `requirements-demo.txt` | the pinned extra packages |
| `CONTRACT.md` | the technical spec all parts follow (events, endpoints, file formats) |
| `tools/make_fixture.py` | builds a small synthetic project root for testing |
| `tools/test_server.py` | automated server test: `python3 demo/tools/test_server.py --url http://localhost:8000` |
| `../docs/img/demo_*.png` | two screenshots from the integration test (synthetic test data, not real numbers) |
| `docs/MOCKUP_A_reference.dc.html` | the approved design mockup the page was built from |

Open `http://localhost:8000/?mock=1` to see the page with simulated data and no backend
events (useful for checking the layout on the projector).
