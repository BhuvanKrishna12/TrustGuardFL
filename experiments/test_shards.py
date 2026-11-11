#!/usr/bin/env python3
"""
test_shards.py
-------------------------------------------------------------------
Check the committed shard files (experiments/shards/{me,user1,user2}.txt):
  * every run-key is a real remaining run-key (steps 5..10 minus the done set);
  * the three shards are disjoint (no run on two machines);
  * together they cover EXACTLY the full remaining set (no run missing);
  * all run-keys that map to the same output file are on the same machine
    (so two machines never write the same run).

  python3 experiments/test_shards.py [machine1 machine2 ...]   (default me user1 user2)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rev2_plan as P  # noqa: E402
import make_shards as M  # noqa: E402


def main():
    machines = sys.argv[1:] or ["me", "user1", "user2"]
    full = dict(P.all_remaining_run_keys(P.env("")))
    done = set(M.DEFAULT_DONE)
    remaining = {k: v for k, v in full.items() if k not in done}

    shards = {m: (P.load_shard_keys(m) or set()) for m in machines}
    union = set().union(*shards.values()) if shards else set()

    # disjoint
    overlaps = []
    ms = list(machines)
    for i in range(len(ms)):
        for j in range(i + 1, len(ms)):
            inter = shards[ms[i]] & shards[ms[j]]
            if inter:
                overlaps.append((ms[i], ms[j], sorted(inter)[:5]))
    assert not overlaps, f"shards overlap: {overlaps}"

    # exact cover
    missing = set(remaining) - union
    extra = union - set(remaining)
    assert not missing, f"{len(missing)} remaining run(s) in NO shard, e.g. {sorted(missing)[:5]}"
    assert not extra, f"{len(extra)} shard run-key(s) not in the remaining set, e.g. {sorted(extra)[:5]}"

    # shared outputs stay together
    by_file = {}
    for k, (step, d, rid) in remaining.items():
        by_file.setdefault((d, rid), []).append(k)
    split = []
    for (d, rid), keys in by_file.items():
        owners = {m for m in machines for k in keys if k in shards[m]}
        if len(owners) > 1:
            split.append((d, rid, sorted(owners)))
    assert not split, f"output files split across machines: {split}"

    per = {m: len(shards[m]) for m in machines}
    print(f"SHARDS OK: {len(remaining)} remaining run-keys = disjoint union of "
          + " + ".join(f"{m}:{per[m]}" for m in machines)
          + f"; {sum(1 for v in by_file.values() if len(v) > 1)} shared output(s) kept together")


if __name__ == "__main__":
    main()
