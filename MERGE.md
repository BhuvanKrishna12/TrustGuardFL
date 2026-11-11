# Merging results from the three machines

Each machine writes its finished runs under `~/rev2_work/runs`. Merging unions
all three into one tree and builds the final tables. Do this on your (the user's)
machine after both helpers send their zips back.

## 1. Each helper zips and returns its runs

A helper does this when its `status` says everything is finished (already in
`SHARE_README.md`):

```bash
cd ~/rev2_work && zip -qr ~/rev2_runs_$(cat MY_SHARD).zip runs
```

They copy that zip to the USB drive (`cp ~/rev2_runs_user1.zip /mnt/e/`) and hand
it back. You get `rev2_runs_user1.zip` and `rev2_runs_user2.zip`.

## 2. On your machine: unzip each helper into its own folder

Put both zips in your Downloads folder, then:

```bash
source ~/flids-venv/bin/activate
mkdir -p ~/rev2_merge/user1 ~/rev2_merge/user2
unzip -oq /mnt/c/Users/Dell/Downloads/rev2_runs_user1.zip -d ~/rev2_merge/user1
unzip -oq /mnt/c/Users/Dell/Downloads/rev2_runs_user2.zip -d ~/rev2_merge/user2
```

Each now has a `runs/` folder inside it (`~/rev2_merge/user1/runs`, etc.). Your
own runs are already at `~/rev2_work/runs`.

## 3. Merge

```bash
python3 ~/rev2_work/rev2/experiments/merge_runs.py \
  --inputs ~/rev2_work/runs ~/rev2_merge/user1/runs ~/rev2_merge/user2/runs \
  --out ~/rev2_work/merged_runs
```

This copies every run file into `~/rev2_work/merged_runs`, taking the union. It
prints how many runs it merged and checks for duplicates:

- **"no conflicts"** - good, the machines ran disjoint shares (as intended).
- **"identical duplicate(s) skipped"** - harmless (e.g. a run that two steps both
  reference, or a helper that re-ran something you already had).
- **"CONFLICT(s)"** - the same run exists on two machines with *different* numbers.
  That should not happen with the shard split; it usually means a stale copy. The
  merge keeps the first input's copy and lists each conflict so you can check. To
  force one machine's copy, re-run with e.g.
  `--prefer ~/rev2_work/runs`.

## 4. Build the final tables

```bash
python3 ~/rev2_work/rev2/experiments/aggregate_results.py \
  --runs-dirs ~/rev2_work/merged_runs/main \
  --rev2-root ~/rev2_work/merged_runs \
  --tex-dir ~/rev2_work/generated
```

The LaTeX tables and macros land in `~/rev2_work/generated` (the same files step
11 of a single-machine run would have produced). Copy that folder into the
paper's `generated/` directory.

## 5. Check nothing is missing (optional)

To confirm every planned run is present in the merge, point a `status` at the
merged tree:

```bash
REV2_W=~/rev2_work python3 ~/rev2_work/rev2/experiments/rev2_plan.py status --shard all
```

Any step that still shows runs "left" means a machine has not finished (or its
zip was not merged). Re-merge once it is in.

## Notes

- You only need the `runs/` folders to merge; the large `probs/` and `models/`
  folders are not required for the tables, so helpers do not send them.
- Results from different GPUs can differ slightly per seed. That is expected; the
  tables report means with confidence intervals across seeds, so a small per-seed
  difference does not change the conclusions. Note the multi-GPU reproducibility
  point in the paper.
