"""
reprocess_mv.py
-------------------------------------------------------------------
Resumable multi-variant reprocessing of whole days from the raw hourly
archives (opensky_fl_ids_pipeline.py --pipeline-version mv --chunk-files 1).

For every day:  for every hourly archive
    check that the archive is readable (external drive present)
    extract it to a local temporary CSV
    run the pipeline on that one hour -> parts_<day>/part_<HH>.csv
    (written to a temp name, row count verified, then renamed; a small JSON
    next to it marks the hour as done)
then the 24 parts are concatenated into opensky_<day>_labeled.csv (temp name,
row count verified, renamed), a <file>.done.json is written and the parts are
removed. Because the pipeline re-seeds its RNG for every chunk, the result is
byte-identical to one uninterrupted pipeline run over the day.

Stopping and resuming: finished hours and finished days are skipped on the
next run. A kill or power loss at any moment leaves only *.tmp files, which
are ignored and overwritten. --stop-flag FILE: stop cleanly after the current
hour.

Exit codes: 0 all requested days done; 75 stopped on request; 3 the source
drive / an archive is not readable (nothing is marked done; reconnect and
rerun); 1 other error.

Usage:
  python3 experiments/reprocess_mv.py --tars-list /mnt/d/fl-ids-sweep/tars.txt \
      --days 2019-01-07 2019-04-01 --out-dir ~/rev2_work/labeled_mv --tmp-dir ~/rev2_work/raw_tmp
"""

import argparse
import contextlib
import gzip
import io
import json
import os
import re
import shutil
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
for _p in (REPO, HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

EXIT_STOPPED, EXIT_SOURCE_MISSING = 75, 3


class SourceMissing(Exception):
    pass


def say(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def atomic_json(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def count_lines(path, bufsize=1 << 24):
    n = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                return n
            n += b.count(b"\n")


def day_archives(tars_list, day):
    """Hourly archives (or plain .csv / .csv.gz files) of one day, sorted by hour."""
    pat = re.compile(r"states_" + re.escape(day) + r"-(\d\d)\.csv(\.tar|\.gz)?$")
    out = {}
    with open(tars_list) as f:
        for line in f:
            p = line.strip()
            m = pat.search(p)
            if m:
                out[m.group(1)] = p
    return [(h, out[h]) for h in sorted(out)]


def check_readable(path):
    try:
        with open(path, "rb") as f:
            f.read(1024)
    except OSError as e:
        raise SourceMissing(f"{path}: {e}")


def extract_hour(src, dst):
    """Raw hourly CSV from a .csv.tar (holding one .csv.gz), a .csv.gz or a .csv."""
    tmp = dst + ".tmp"
    try:
        if src.endswith(".tar"):
            with tarfile.open(src, "r") as tf:
                member = next((m for m in tf if m.name.endswith(".csv.gz")), None)
                if member is None:
                    raise RuntimeError(f"no .csv.gz inside {src}")
                with gzip.GzipFile(fileobj=tf.extractfile(member)) as gz, open(tmp, "wb") as out:
                    shutil.copyfileobj(gz, out, 1 << 22)
        elif src.endswith(".gz"):
            with gzip.open(src, "rb") as gz, open(tmp, "wb") as out:
                shutil.copyfileobj(gz, out, 1 << 22)
        else:
            shutil.copyfile(src, tmp)
    except OSError as e:
        if os.path.exists(tmp):
            os.remove(tmp)
        # a vanished drive shows up as an I/O error on the source
        try:
            check_readable(src)
        except SourceMissing:
            raise SourceMissing(f"{src}: {e}")
        raise
    os.replace(tmp, dst)


def process_day(day, archives, out_dir, tmp_dir, stop_flag, seed, progress):
    import opensky_fl_ids_pipeline as pl
    final = os.path.join(out_dir, f"opensky_{day}_labeled.csv")
    done_json = final + ".done.json"
    if os.path.exists(done_json) and os.path.exists(final):
        d = json.load(open(done_json))
        if os.path.getsize(final) == d.get("bytes"):
            say(f"{day}: already done ({d['rows']:,} rows)")
            progress["hours_done"] += len(archives)
            return
    parts_dir = os.path.join(out_dir, f"parts_{day}")
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)
    metas = []
    for i, (hh, src) in enumerate(archives, 1):
        part = os.path.join(parts_dir, f"part_{hh}.csv")
        pj = part + ".json"
        if os.path.exists(pj) and os.path.exists(part):
            m = json.load(open(pj))
            if os.path.getsize(part) == m["bytes"]:
                metas.append(m)
                progress["hours_done"] += 1
                continue
        if stop_flag and os.path.exists(stop_flag):
            raise KeyboardInterrupt("stop flag")
        t0 = time.time()
        check_readable(src)
        raw = os.path.join(tmp_dir, f"states_{day}-{hh}.csv")
        extract_hour(src, raw)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            n_rows, n_attack, type_counts = pl.process_chunk(
                [raw], f"{day} hour {hh}", part + ".tmp", True, seed=seed, pipeline_version="mv")
        os.remove(raw)
        lines = count_lines(part + ".tmp")
        if lines != n_rows + 1:
            raise RuntimeError(f"{part}.tmp has {lines} lines, expected {n_rows + 1}")
        os.replace(part + ".tmp", part)
        m = {"hour": hh, "source": src, "rows": int(n_rows), "attack_rows": int(n_attack),
             "type_counts": {k: int(v) for k, v in type_counts.items()},
             "bytes": os.path.getsize(part), "seconds": time.time() - t0}
        atomic_json(m, pj)
        metas.append(m)
        progress["hours_done"] += 1
        progress["seconds"].append(m["seconds"])
        left = progress["hours_total"] - progress["hours_done"]
        avg = sum(progress["seconds"]) / len(progress["seconds"])
        say(f"{day} hour {hh} done ({i}/{len(archives)}): {n_rows:,} rows in {m['seconds']:.0f}s | "
            f"hours left in this step: {left} (~{left * avg / 3600:.1f} h)")
        atomic_json({"hours_done": progress["hours_done"], "hours_total": progress["hours_total"],
                     "avg_seconds_per_hour": avg, "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                    os.path.join(out_dir, "reprocess_progress.json"))
    # concatenate the parts
    total = sum(m["rows"] for m in metas)
    tmp = final + ".tmp"
    with open(tmp, "wb") as out:
        for k, (hh, _) in enumerate(archives):
            with open(os.path.join(parts_dir, f"part_{hh}.csv"), "rb") as f:
                header = f.readline()
                if k == 0:
                    out.write(header)
                shutil.copyfileobj(f, out, 1 << 24)
        out.flush()
        os.fsync(out.fileno())
    lines = count_lines(tmp)
    if lines != total + 1:
        raise RuntimeError(f"{tmp} has {lines} lines, expected {total + 1}")
    os.replace(tmp, final)
    tc = {}
    for m in metas:
        for k, v in m["type_counts"].items():
            tc[k] = tc.get(k, 0) + v
    atomic_json({"day": day, "rows": int(total), "bytes": os.path.getsize(final),
                 "hours": len(metas), "type_counts": tc, "pipeline_version": "mv",
                 "finished": time.strftime("%Y-%m-%d %H:%M:%S")}, done_json)
    shutil.rmtree(parts_dir, ignore_errors=True)
    say(f"{day}: DONE, {total:,} rows, {os.path.getsize(final) / 1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tars-list", required=True,
                    help="text file with one raw hourly archive path per line")
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--tmp-dir", required=True)
    ap.add_argument("--stop-flag", default=os.environ.get("REV2_STOP_FLAG"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-hours", type=int, default=24,
                    help="refuse a day with fewer hourly archives in the list (default 24)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plan = []
    for day in args.days:
        arch = day_archives(args.tars_list, day)
        if len(arch) < args.min_hours:
            say(f"ERROR: {day}: only {len(arch)} hourly archive(s) in {args.tars_list} "
                f"(expected {args.min_hours}); use --min-hours to allow")
            return 1
        plan.append((day, arch))
    progress = {"hours_total": sum(len(a) for _, a in plan), "hours_done": 0, "seconds": []}
    try:
        for day, arch in plan:
            process_day(day, arch, args.out_dir, args.tmp_dir, args.stop_flag, args.seed, progress)
    except SourceMissing as e:
        say(f"SOURCE NOT READABLE: {e}")
        say("The external drive seems to be disconnected. Nothing was marked as done. "
            "Reconnect it and start again; finished hours are kept.")
        return EXIT_SOURCE_MISSING
    except KeyboardInterrupt:
        say("STOPPED on request; finished hours are kept, rerun to continue.")
        return EXIT_STOPPED
    say("all requested days done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
