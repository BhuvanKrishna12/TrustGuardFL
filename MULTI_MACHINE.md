# Running the revision across three machines

Split the remaining training (revision-2 steps 5 to 10) across your laptop and
two helpers. Each machine runs the **same** `run_revision2.sh start/status/stop`
script on its own share, using only the ~2 GB cache (not the 120 GB labelled
files). At the end you merge everyone's results into the final tables.

Who does what:

- **You (the user):** finish step 4 so the cache exists, build the share, hand a
  copy to each helper, and keep running your own share (`me`).
- **Each helper:** copy the share, pick `user1` or `user2`, run start/status/stop,
  send one zip back.
- **You, at the end:** unzip both helpers' results next to yours and run the
  merge, which builds the tables.

The split is in `SPLIT.md` (about 12 hours per machine with the measured 3050
times). Every block below is copy-paste WSL. Assume Ubuntu on WSL on all three
machines, the same as your main laptop.

---

## On your (the user's) machine

You have already done steps 1 to 4 and most of step 5. First make sure the cache
is complete and the input manifest exists (safe to re-run):

```bash
source ~/flids-venv/bin/activate
python3 ~/rev2_work/rev2/experiments/make_input_manifest.py \
  --train-files ~/rev2_work/labeled_mv/opensky_2019-01-07_labeled.csv \
                ~/rev2_work/labeled_mv/opensky_2019-04-01_labeled.csv \
                ~/rev2_work/labeled_mv/opensky_2019-06-24_labeled.csv \
                ~/rev2_work/labeled_mv/opensky_2019-09-30_labeled.csv \
  --holdout ~/rev2_work/holdout_sample_mv.csv --cache-dir ~/rev2_work/cache
python3 ~/rev2_work/rev2/experiments/verify_cache.py --work ~/rev2_work
```

Build the ~2 GB share onto your D: drive:

```bash
bash ~/rev2_work/rev2/make_share.sh /mnt/d/rev2_share
```

Tell this machine to run the `me` share, and start it:

```bash
echo me > ~/rev2_work/MY_SHARD
bash ~/rev2_work/rev2/run_revision2.sh start
```

Check progress / pause exactly as before:

```bash
bash ~/rev2_work/rev2/run_revision2.sh status
bash ~/rev2_work/rev2/run_revision2.sh stop
```

Copy `/mnt/d/rev2_share` onto a USB drive and give it to each helper. (It already
contains `SHARE_README.md` with the helper's own instructions.)

---

## On helper 1's machine

Full step-by-step is in `SHARE_README.md` inside the share. In short:

One-time setup (about 10 minutes):

```bash
nvidia-smi
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv ~/flids-venv
source ~/flids-venv/bin/activate
```

Copy the share from the USB drive (find the drive letter with `ls /mnt`; replace
`e` if different), then install the packages:

```bash
ls /mnt
mkdir -p ~/rev2_work
cp -a /mnt/e/rev2_share/. ~/rev2_work/
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r ~/rev2_work/rev2/requirements.txt
```

(If pip complains about an externally managed environment, add
`--break-system-packages` to that last line. The pinned torch 2.14.1 is fine for
recent NVIDIA cards, a 4060 included.)

Pick the `user1` share and start:

```bash
echo user1 > ~/rev2_work/MY_SHARD
bash ~/rev2_work/rev2/run_revision2.sh start
```

Progress / pause:

```bash
bash ~/rev2_work/rev2/run_revision2.sh status
bash ~/rev2_work/rev2/run_revision2.sh stop
```

When status says everything is finished, zip and return the runs (replace `e`
with the USB drive letter):

```bash
cd ~/rev2_work && zip -qr ~/rev2_runs_user1.zip runs
cp ~/rev2_runs_user1.zip /mnt/e/
```

---

## On helper 2's machine

Identical to helper 1, except choose `user2`:

```bash
nvidia-smi
sudo apt update && sudo apt install -y python3.12 python3.12-venv
python3.12 -m venv ~/flids-venv
source ~/flids-venv/bin/activate
```

```bash
ls /mnt
mkdir -p ~/rev2_work
cp -a /mnt/e/rev2_share/. ~/rev2_work/
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
pip install -r ~/rev2_work/rev2/requirements.txt
```

```bash
echo user2 > ~/rev2_work/MY_SHARD
bash ~/rev2_work/rev2/run_revision2.sh start
```

```bash
bash ~/rev2_work/rev2/run_revision2.sh status
bash ~/rev2_work/rev2/run_revision2.sh stop
```

```bash
cd ~/rev2_work && zip -qr ~/rev2_runs_user2.zip runs
cp ~/rev2_runs_user2.zip /mnt/e/
```

---

## When everyone is done (merge)

See `MERGE.md` for the full details. Short version, on your machine, with both
helpers' zips copied into `/mnt/c/Users/Dell/Downloads`:

```bash
source ~/flids-venv/bin/activate
mkdir -p ~/rev2_merge/user1 ~/rev2_merge/user2
unzip -oq /mnt/c/Users/Dell/Downloads/rev2_runs_user1.zip -d ~/rev2_merge/user1
unzip -oq /mnt/c/Users/Dell/Downloads/rev2_runs_user2.zip -d ~/rev2_merge/user2
python3 ~/rev2_work/rev2/experiments/merge_runs.py \
  --inputs ~/rev2_work/runs ~/rev2_merge/user1/runs ~/rev2_merge/user2/runs \
  --out ~/rev2_work/merged_runs
python3 ~/rev2_work/rev2/experiments/aggregate_results.py \
  --runs-dirs ~/rev2_work/merged_runs/main --rev2-root ~/rev2_work/merged_runs \
  --tex-dir ~/rev2_work/generated
```

The tables are in `~/rev2_work/generated`.

---

## Notes

- **Rebalancing:** `SPLIT.md` shows each machine's hours. To move work, edit the
  files in `experiments/shards/` (`me.txt`, `user1.txt`, `user2.txt`): move whole
  run-key lines between them, keeping any `# shared output` pair together. Then
  rebuild the share so the helpers get the new files. To regenerate from scratch
  once you know a helper GPU's speed, run for example
  `python3 experiments/make_shards.py --weights fed_cnn=2600,fed_mlp=2600`.
- **Per-GPU differences:** runs on different GPUs can differ slightly per seed.
  This is expected and acceptable; note it in the paper's reproducibility section.
- **A helper has no big files:** the script detects this automatically ("cache-only
  mode" in `status`) and skips steps 1 to 4; it never needs the external drive.
- **Single machine still works:** with no `MY_SHARD` file (or `MY_SHARD=all`),
  `run_revision2.sh` runs everything, exactly as before.
