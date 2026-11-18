#!/usr/bin/env bash
# FL-IDS live demo launcher.
#
# Run from the project root (the folder that holds federated_simulation.py):
#
#     bash demo/start_demo.sh                 # prepare data and train only if needed, then serve
#     bash demo/start_demo.sh --reprep        # rebuild demo/data (then retrain)
#     bash demo/start_demo.sh --retrain       # run train_demo.py again (15 rounds)
#     bash demo/start_demo.sh --rounds 20 --retrain
#     bash demo/start_demo.sh --demo-model    # detect with the 6-client demo model, not fixed4_cnn.pt
#     bash demo/start_demo.sh --noisy-rate 0.3   # false alarm share that marks a tower "noisy"
#
# Inputs can be moved with environment variables, for example:
#     HOLDOUT=/path/to/holdout_feb04_1pct.csv bash demo/start_demo.sh
#
#     SAMPLE    cached training sample      default models/sample_3m.pkl
#     HOLDOUT   labeled Feb 4 holdout CSV   default ~/opensky_data/labeled_fixed/holdout_feb04_1pct.csv
#     MODEL     detection model             default models/fixed4_cnn.pt (+ _preprocess.pkl)
#     PRIOR     class prior JSON            default results/lgbm_feb04.json
#     CELLS     six region cells, comma separated (only used when data is prepared)
#     NAMES     display names for CELLS, same order (optional)
#     PORT 8000, RATE 2 (messages per second per client), STRENGTH 0.5, ROUNDS 15,
#     NOISY_RATE 0.25 (a tower is "noisy" when this share of its benign msgs in 20 s are false alarms)
#     PYTHON    python interpreter          default python3
#
# Ctrl+C stops everything.

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1

PY="${PYTHON:-python3}"
SAMPLE="${SAMPLE:-models/sample_3m.pkl}"
HOLDOUT="${HOLDOUT:-$HOME/opensky_data/labeled_fixed/holdout_feb04_1pct.csv}"
MODEL="${MODEL:-models/fixed4_cnn.pt}"
PRIOR="${PRIOR:-results/lgbm_feb04.json}"
CELLS="${CELLS:-}"
NAMES="${NAMES:-}"
PORT="${PORT:-8000}"
RATE="${RATE:-2}"
STRENGTH="${STRENGTH:-0.5}"
ROUNDS="${ROUNDS:-15}"
NOISY_RATE="${NOISY_RATE:-0.25}"
REPREP=0
RETRAIN=0

usage() {
    sed -n '2,26p' "$0" | sed 's/^# \{0,1\}//'
}

while [ $# -gt 0 ]; do
    case "$1" in
        --reprep) REPREP=1 ;;
        --retrain) RETRAIN=1 ;;
        --rounds) shift; ROUNDS="${1:?--rounds needs a number}" ;;
        --port) shift; PORT="${1:?--port needs a number}" ;;
        --rate) shift; RATE="${1:?--rate needs a number}" ;;
        --strength) shift; STRENGTH="${1:?--strength needs a number}" ;;
        --noisy-rate) shift; NOISY_RATE="${1:?--noisy-rate needs a number}" ;;
        --demo-model) MODEL="demo/models/latest.pt" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1  (try --help)"; exit 2 ;;
    esac
    shift
done

say()  { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }
die()  { printf '\nSTOPPED: %s\n' "$*" >&2; exit 1; }

trap 'printf "\nStopped by Ctrl+C.\n"; exit 130' INT

step "FL-IDS live demo, project root $ROOT"

# ---- 1. Python and packages ------------------------------------------------------------
command -v "$PY" >/dev/null 2>&1 || die "$PY not found. Install Python 3.12 (sudo apt install python3)."
say "Python: $("$PY" --version 2>&1)"

missing=$("$PY" - <<'EOF'
import importlib
missing = []
for mod in ["flwr", "fastapi", "uvicorn", "torch", "pandas", "sklearn", "joblib"]:
    try:
        importlib.import_module(mod)
    except Exception:
        missing.append(mod)
print(" ".join(missing))
EOF
)
if [ -n "$missing" ]; then
    say ""
    say "Missing Python packages: $missing"
    demo_pkgs=0; base_pkgs=""
    for m in $missing; do
        case "$m" in
            flwr|fastapi|uvicorn) demo_pkgs=1 ;;
            torch) base_pkgs="$base_pkgs torch" ;;
            pandas) base_pkgs="$base_pkgs pandas" ;;
            sklearn) base_pkgs="$base_pkgs scikit-learn" ;;
            joblib) base_pkgs="$base_pkgs joblib" ;;
        esac
    done
    say "Install them with:"
    if [ -n "$base_pkgs" ]; then
        say "    pip3 install --user --break-system-packages$base_pkgs"
        case "$base_pkgs" in
            *torch*) say "    (CPU only torch, smaller download: add --index-url https://download.pytorch.org/whl/cpu)" ;;
        esac
    fi
    if [ "$demo_pkgs" = 1 ]; then
        say "    pip3 install --user --break-system-packages -r demo/requirements-demo.txt"
    fi
    die "install the packages above, then run this script again."
fi
say "Packages: flwr, fastapi, uvicorn, torch, pandas, sklearn, joblib all import."

# ---- 2. Required files -------------------------------------------------------------------
abspath() {  # resolve against the project root, expand ~
    case "$1" in
        "~"*) printf '%s' "$HOME${1#\~}" ;;
        /*) printf '%s' "$1" ;;
        *) printf '%s' "$ROOT/$1" ;;
    esac
}
SAMPLE="$(abspath "$SAMPLE")"; HOLDOUT="$(abspath "$HOLDOUT")"
MODEL="$(abspath "$MODEL")"; PRIOR="$(abspath "$PRIOR")"
MODEL_PRE="${MODEL%.*}_preprocess.pkl"

# evaluate_holdout.py is optional: without it demo/server.py uses its built-in model loader.
for f in federated_simulation.py; do
    [ -f "$ROOT/$f" ] || die "$f is not in $ROOT. The demo/ folder must sit inside the project root (next to $f)."
done

NEED_PREP=0
if [ "$REPREP" = 1 ] || [ ! -f demo/data/clients.json ]; then NEED_PREP=1; fi
NEED_TRAIN=0
if [ "$RETRAIN" = 1 ] || [ "$NEED_PREP" = 1 ] || [ ! -f demo/runs/latest.jsonl ]; then NEED_TRAIN=1; fi

if [ "$NEED_PREP" = 1 ]; then
    [ -f "$SAMPLE" ] || die "cached training sample not found: $SAMPLE
  It is a pickled training-sample DataFrame (or {"meta": ..., "df": DataFrame}), see demo/CONTRACT.md.
  Point to it with SAMPLE=/path/to/sample_3m.pkl"
    [ -f "$HOLDOUT" ] || die "holdout CSV not found: $HOLDOUT
  Point to it with HOLDOUT=/path/to/holdout_feb04_1pct.csv"
fi
if [ "$MODEL" = "$ROOT/demo/models/latest.pt" ] && [ ! -f "$MODEL" ] && [ "$NEED_TRAIN" = 1 ]; then
    say "Detection model: the demo model will be created by the training step below."
else
    [ -f "$MODEL" ] || die "detection model not found: $MODEL
  Point to it with MODEL=/path/to/model.pt (its _preprocess.pkl must sit next to it)."
    [ -f "$MODEL_PRE" ] || die "preprocess file not found next to the model: $MODEL_PRE"
fi
if [ ! -f "$PRIOR" ]; then
    say "WARNING: class prior not found: $PRIOR"
    say "         The demo still runs, but the correction dial will do nothing. Fix with PRIOR=/path/to/lgbm_feb04.json"
fi
say "Inputs look fine."

# ---- 3. Prepare per-client data (only when needed) ---------------------------------------
if [ "$NEED_PREP" = 1 ]; then
    step "Preparing demo data (demo/data). Takes a minute or two on the real files."
    prep_args=(--sample "$SAMPLE" --holdout "$HOLDOUT")
    [ -n "$CELLS" ] && prep_args+=(--cells "$CELLS")
    [ -n "$NAMES" ] && prep_args+=(--names "$NAMES")
    "$PY" demo/prepare_demo_data.py "${prep_args[@]}" || die "prepare_demo_data.py failed (see above)."
    if [ "$RETRAIN" = 0 ] && [ -f demo/runs/latest.jsonl ]; then
        say "Data was rebuilt, so the training run is redone to match it."
    fi
else
    say "Demo data: demo/data/clients.json exists, skipping prep (use --reprep to rebuild)."
fi

# ---- 4. Federated training (only when needed) --------------------------------------------
if [ "$NEED_TRAIN" = 1 ]; then
    step "Training with Flower: $ROUNDS rounds, 6 clients. Ray needs about 15 s to start."
    "$PY" demo/train_demo.py --rounds "$ROUNDS" || die "train_demo.py failed (see above)."
else
    say "Training run: demo/runs/latest.jsonl exists, skipping training (use --retrain to redo it)."
fi

# ---- 5. Server -----------------------------------------------------------------------------
port_busy=$("$PY" - "$PORT" <<'EOF'
import socket, sys
s = socket.socket()
# same option uvicorn uses, so sockets left in TIME_WAIT by a just stopped server do not count
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind(("0.0.0.0", int(sys.argv[1])))
    print("")
except OSError:
    print("busy")
finally:
    s.close()
EOF
)
if [ -n "$port_busy" ]; then
    die "port $PORT is already in use. Is the demo already running in another terminal?
  Stop it with Ctrl+C there, or find it with:  ss -ltnp | grep :$PORT
  Or use another port:  bash demo/start_demo.sh --port 8001"
fi

step "Starting the demo server"
say "Open this in your Windows browser:   http://localhost:$PORT"
say "(Chrome or Edge on Windows reach WSL through localhost. Keep this terminal open.)"
say "Press Ctrl+C here to stop."
say ""
trap - INT
exec "$PY" demo/server.py --model "$MODEL" --prior-json "$PRIOR" --port "$PORT" \
    --rate "$RATE" --strength "$STRENGTH" --noisy-rate "$NOISY_RATE"
