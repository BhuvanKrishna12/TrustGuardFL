# Optional: splitting the runs across several machines

Steps 5 to 10 of `run_revision2.sh` (about 30 hours of training on one laptop GPU) can be
split across machines. Only the main machine needs the raw archives and the 120 GB labelled
files; the other machines work from the about 2 GB cache built in step 4. A single machine
needs none of this: without a shard setting, `run_revision2.sh` runs everything.

The example below uses three machines:

| Machine | Role | Shard id |
|---|---|---|
| A | Main machine: has the labelled files, runs steps 1 to 4, builds the share, merges at the end | `me` |
| B | Helper with a CUDA GPU | `user1` |
| C | Helper without a CUDA GPU (CPU only) | `user2` |

The shard ids name the files `experiments/shards/me.txt`, `user1.txt` and `user2.txt`. They
are identifiers used by the code and the tests (`experiments/make_shards.py`,
`experiments/test_shards.py`), not names of people. Each file lists the run keys
(`step:model:variant:feature-set:seed`) that one machine runs.

## How the committed split was made

`experiments/make_shards.py` splits the runs that remained after the main machine had finished
steps 1 to 4 and part of step 5. It weights each run by its measured time, packs the machines
to near-equal total time, keeps the seeds of one run series together where possible, gives the
CPU-only shard (`user2`) no federated runs (they take hours each on a CPU) and lets the faster
GPU (`user1`, weight 1.3) take proportionally more. Regenerate or rebalance with, for example:

```bash
python3 experiments/make_shards.py --machines me,user1,user2 --speed user1=1.5
python3 experiments/make_shards.py --done ""                 # split every run, nothing done yet
python3 experiments/test_shards.py                           # check: disjoint exact cover
```

Lines can also be moved between the shard files by hand; keep the lines of one shared output
together (see the comment at the top of each file).

## Machine A: build the cache and the share

Run `run_revision2.sh` until step 4 is finished (see `docs/REPRODUCING.md`). Then make sure the
input manifest exists and the cache is complete, and build the share:

```bash
python3 experiments/make_input_manifest.py \
  --train-files "$WORK_DIR"/labeled_mv/opensky_2019-{01-07,04-01,06-24,09-30}_labeled.csv \
  --holdout "$WORK_DIR/holdout_sample_mv.csv" --cache-dir "$WORK_DIR/cache"
python3 experiments/verify_cache.py --work "$WORK_DIR"
REV2_W="$WORK_DIR" bash make_share.sh "$SHARE_DIR"
```

`$SHARE_DIR` then holds `rev2/` (the code, without data, models or PDFs), `cache/`,
`holdout_sample_mv.csv` and a default `MY_SHARD` file. Copy it to each helper.

Tell machine A to run its own shard and continue as usual:

```bash
echo me > "$WORK_DIR/MY_SHARD"        # or: export MY_SHARD=me
REV2_W="$WORK_DIR" bash run_revision2.sh start
```

## Machines B and C: run a shard from the cache

On each helper, install Python 3.12 or 3.13 and the requirements (on machine C the CPU wheel of
PyTorch is enough; runs use the CPU automatically when no GPU is found). Copy the share into a
local work dir and pick the shard:

```bash
mkdir -p "$WORK_DIR" && cp -a "$SHARE_DIR"/. "$WORK_DIR"/
pip install -r "$WORK_DIR/rev2/requirements.txt"
echo user1 > "$WORK_DIR/MY_SHARD"     # machine C: echo user2
REV2_W="$WORK_DIR" bash "$WORK_DIR/rev2/run_revision2.sh" start
REV2_W="$WORK_DIR" bash "$WORK_DIR/rev2/run_revision2.sh" status
```

The script sees that the labelled files are absent and the cache is present, switches to
cache-only mode (`status` shows it), verifies the cache, skips steps 1 to 4 and runs only the
run keys of its shard. `stop` and `start` work as on a single machine. When `status` reports
that everything is finished, pack the runs and copy the zip back to machine A:

```bash
cd "$WORK_DIR" && zip -qr "runs_$(cat MY_SHARD).zip" runs
```

Only the `runs/` folder is needed for the tables; the larger `probs/` and `models/` folders can
stay on the helper.

## Machine A: merge and build the tables

Unzip each helper's runs into its own folder and merge:

```bash
mkdir -p "$WORK_DIR/merge/user1" "$WORK_DIR/merge/user2"
unzip -oq runs_user1.zip -d "$WORK_DIR/merge/user1"
unzip -oq runs_user2.zip -d "$WORK_DIR/merge/user2"
python3 experiments/merge_runs.py \
  --inputs "$WORK_DIR/runs" "$WORK_DIR/merge/user1/runs" "$WORK_DIR/merge/user2/runs" \
  --out "$WORK_DIR/merged_runs"
python3 experiments/aggregate_results.py \
  --runs-dirs "$WORK_DIR/merged_runs/main" --rev2-root "$WORK_DIR/merged_runs" \
  --tex-dir "$WORK_DIR/generated"
```

`merge_runs.py` takes the union per output file. It reports "no conflicts" when the shards were
disjoint, skips identical duplicates, and lists any run that exists on two machines with
different numbers (the first input wins; `--prefer DIR` picks another). To check that no planned
run is missing:

```bash
REV2_W="$WORK_DIR" python3 experiments/rev2_plan.py status --shard all
```

## Notes

* Runs on different GPUs can differ slightly per seed. The tables report means and confidence
  intervals across seeds, so this does not change the conclusions.
* `experiments/test_multi_machine.sh DIR` simulates the whole flow on synthetic data (a main
  machine, a cache-only helper, the merge and the aggregation); the smoke test runs it.
