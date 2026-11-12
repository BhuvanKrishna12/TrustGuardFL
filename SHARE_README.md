# Helper share - read me first

This folder lets your machine run part of the training for a research paper. It
has everything needed **except** the 120 GB of labelled data, which stays on the
main machine. Your machine works from the ~2 GB cache in here instead. You will
run a start/status/stop script, then send one zip back. Nothing here touches the
rest of your computer.

You need: Windows with WSL (Ubuntu) and about 9 to 15 hours of compute time spread
over as many sittings as you like (you can stop and resume anytime). You were told
whether you are **`user1`** (NVIDIA GPU) or **`user2`** (no NVIDIA GPU, runs on the
CPU). Follow the matching setup below. Every block is copy-paste.

## 1. One-time WSL setup (about 10 minutes)

Open **Ubuntu (WSL)** from the Start menu. Make a Python 3.12 virtual environment:

```bash
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv ~/flids-venv
source ~/flids-venv/bin/activate
```

### If you are `user1` (you have an NVIDIA GPU)

Check the GPU is visible inside WSL (you should see a table with your GPU):

```bash
nvidia-smi
```

Install the CUDA build of PyTorch (the pinned 2.14.1 is fine for recent NVIDIA
cards, a 4060 included), then the rest:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install -r ~/rev2_work/rev2/requirements.txt
```

Confirm PyTorch sees the GPU (should print `True`):

```bash
python3 -c "import torch; print(torch.cuda.is_available())"
```

### If you are `user2` (integrated graphics, no NVIDIA GPU)

**Skip `nvidia-smi`** - it will fail, which is expected. Your share is only fast
work, so the CPU is fine. Install the smaller **CPU** build of PyTorch, then the
rest:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r ~/rev2_work/rev2/requirements.txt
```

(If you already installed a CUDA wheel such as `cu126`, that works on CPU too.) The
script uses the CPU automatically; you set nothing.

### Either way

If pip refuses with an "externally managed environment" error, add
`--break-system-packages` to the `pip install -r ...` line.

## 2. Copy this share into WSL

The share is on a USB drive. Find its drive letter (look for your USB folder,
e.g. `d`, `e`, `f`):

```bash
ls /mnt
```

Copy it to `~/rev2_work` (replace `e` below with your drive letter from the line
above if it is not `e`):

```bash
mkdir -p ~/rev2_work
cp -a /mnt/e/rev2_share/. ~/rev2_work/
```

## 3. Choose your share: user1 or user2

Run **exactly one** of these, whichever you were told:

```bash
echo user1 > ~/rev2_work/MY_SHARD
```

```bash
echo user2 > ~/rev2_work/MY_SHARD
```

## 4. Start it

```bash
source ~/flids-venv/bin/activate
bash ~/rev2_work/rev2/run_revision2.sh start
```

It runs in the background, so you can close the terminal. It notices it has only
the cache (not the big files) and runs just your share. Resume after any pause
with the same `start` command.

## 5. Check progress / pause

```bash
bash ~/rev2_work/rev2/run_revision2.sh status
```

```bash
bash ~/rev2_work/rev2/run_revision2.sh stop
```

## 6. When your status says everything is finished, send the results back

```bash
cd ~/rev2_work && zip -qr ~/rev2_runs_$(cat MY_SHARD).zip runs && echo "created ~/rev2_runs_$(cat MY_SHARD).zip"
```

Copy that zip back to the USB drive (replace `e` with your drive letter):

```bash
cp ~/rev2_runs_$(cat ~/rev2_work/MY_SHARD).zip /mnt/e/
```

Give the USB back. That zip is all that is needed. Results from different GPUs
differ very slightly per run; that is expected and fine.
