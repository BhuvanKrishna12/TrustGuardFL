"""
make_synthetic_raw.py
-------------------------------------------------------------------
Small synthetic RAW OpenSky state-vector files (the input format of
opensky_fl_ids_pipeline.py), one CSV per hour, for testing the pipeline
variants and the revision-2 runner without real data.

The data exercise every code path the corrections touch: 10 s snapshot grid
per aircraft, missing squawk codes and real squawk changes (C2), position
glitches above 400 m/s (C3), pairs of messages less than 1 s apart (C4),
missing vertrate / altitude, rows without a position fix, and a few rows
repeated across the hour boundary.

Usage:
  python3 experiments/make_synthetic_raw.py --out-dir DIR --date 2019-01-07 --hours 2
  python3 experiments/make_synthetic_raw.py --out-dir DIR --date 2019-01-07 --hours 2 --tar
      (--tar writes states_<date>-<HH>.csv.tar holding states_<date>-<HH>.csv.gz,
       the layout of the real archives, and prints the tar paths)
"""

import argparse
import gzip
import io
import os
import tarfile

import numpy as np
import pandas as pd

COLS = ["time", "icao24", "lat", "lon", "velocity", "heading", "vertrate", "callsign",
        "onground", "alert", "spi", "squawk", "baroaltitude", "geoaltitude",
        "lastposupdate", "lastcontact"]


def make_hour(date, hour, n_aircraft=120, seed=0, lat_range=(32.0, 62.0), lon_range=(-15.0, 35.0)):
    rng = np.random.default_rng([seed, int(pd.Timestamp(date).strftime("%Y%m%d")), hour])
    t0 = int(pd.Timestamp(date, tz="UTC").timestamp()) + 3600 * hour
    frames = []
    # aircraft identities are stable across hours and days (seeded by index only)
    id_rng = np.random.default_rng(seed + 1000)
    icaos = [f"{v:06x}" for v in id_rng.integers(0x300000, 0x4fffff, n_aircraft)]
    squawks = id_rng.integers(1000, 7777, n_aircraft)
    for a in range(n_aircraft):
        if rng.random() < 0.15:           # not every aircraft flies every hour
            continue
        start = rng.integers(0, 1800)
        dur = rng.integers(300, 3600 - start)
        phase = rng.integers(0, 10)
        t = np.arange(t0 + start + phase, t0 + start + dur, 10, dtype=np.float64)
        n = len(t)
        if n < 2:
            continue
        lat0 = rng.uniform(*lat_range)
        lon0 = rng.uniform(*lon_range)
        heading = rng.uniform(0, 360)
        speed = rng.uniform(120, 260)
        hd = (heading + np.cumsum(rng.normal(0, 0.4, n))) % 360
        v = speed + rng.normal(0, 2, n)
        dlat = v * np.cos(np.radians(hd)) * 10 / 111_000
        dlon = v * np.sin(np.radians(hd)) * 10 / (111_000 * np.cos(np.radians(lat0)))
        lat = np.clip(lat0 + np.cumsum(dlat), -89, 89)
        lon = lon0 + np.cumsum(dlon)
        vr = rng.choice([0.0, 0.0, 0.0, 6.0, -5.0]) + rng.normal(0, 0.3, n)
        alt = 9000 + np.cumsum(vr * 10)
        d = pd.DataFrame({
            "time": t, "icao24": icaos[a], "lat": lat, "lon": lon, "velocity": v,
            "heading": hd, "vertrate": vr, "callsign": f"SYN{a:04d} ",
            "onground": "False", "alert": "False", "spi": "False",
            "squawk": float(squawks[a]), "baroaltitude": alt, "geoaltitude": alt + 50,
            "lastposupdate": t - rng.uniform(0, 2, n), "lastcontact": t - rng.uniform(0, 1, n),
        })
        # C2: aircraft without a code, codes missing on some rows, real code changes
        r = rng.random()
        if r < 0.2:
            d["squawk"] = np.nan
        elif r < 0.4:
            d.loc[rng.random(n) < 0.2, "squawk"] = np.nan
        elif r < 0.5:
            d.loc[n // 2:, "squawk"] = 7000.0
        # C3: position glitches (benign rows with implied speed far above 400 m/s)
        g = rng.random(n) < 0.004
        d.loc[g, "lat"] += rng.uniform(0.5, 1.5, g.sum())
        # C4: second receiver reporting the same aircraft a fraction of a second later
        k = rng.random(n) < 0.01
        if k.any():
            extra = d[k].copy()
            extra["time"] += rng.uniform(0.2, 0.8, k.sum())
            extra["lat"] += rng.normal(0, 2e-3, k.sum())   # some of these exceed 400 m/s
            d = pd.concat([d, extra], ignore_index=True)
        # stale rows, missing values
        d.loc[rng.random(len(d)) < 0.03, "lastcontact"] -= 30
        d.loc[rng.random(len(d)) < 0.05, "vertrate"] = np.nan
        d.loc[rng.random(len(d)) < 0.03, "baroaltitude"] = np.nan
        d.loc[rng.random(len(d)) < 0.01, ["lat", "lon"]] = np.nan
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out = out.sample(frac=1.0, random_state=int(rng.integers(1 << 30))).reset_index(drop=True)
    return out[COLS]


def write_day(out_dir, date, hours, n_aircraft=120, seed=0, tar=False, first_hour=0):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    prev_tail = None
    for h in range(first_hour, first_hour + hours):
        df = make_hour(date, h, n_aircraft=n_aircraft, seed=seed)
        if prev_tail is not None:     # a few rows repeated across the file boundary
            df = pd.concat([prev_tail, df], ignore_index=True)
        prev_tail = df.tail(3)
        name = f"states_{date}-{h:02d}.csv"
        if not tar:
            p = os.path.join(out_dir, name)
            df.to_csv(p, index=False)
        else:
            day_dir = os.path.join(out_dir, f".{date}", f"{h:02d}")
            os.makedirs(day_dir, exist_ok=True)
            p = os.path.join(day_dir, name + ".tar")
            raw = gzip.compress(df.to_csv(index=False).encode(), mtime=0)
            with tarfile.open(p + ".tmp", "w") as tf:
                ti = tarfile.TarInfo(name + ".gz")
                ti.size = len(raw)
                tf.addfile(ti, io.BytesIO(raw))
                readme = b"synthetic"
                ti = tarfile.TarInfo("README.txt")
                ti.size = len(readme)
                tf.addfile(ti, io.BytesIO(readme))
            os.replace(p + ".tmp", p)
        paths.append(p)
    return paths


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--date", nargs="+", default=["2019-01-07"])
    ap.add_argument("--hours", type=int, default=2)
    ap.add_argument("--aircraft", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tar", action="store_true")
    ap.add_argument("--tars-list", default=None, help="with --tar: write the tar paths here")
    args = ap.parse_args()
    allp = []
    for d in args.date:
        allp += write_day(args.out_dir, d, args.hours, args.aircraft, args.seed, args.tar)
    if args.tars_list:
        with open(args.tars_list, "w") as f:
            f.write("\n".join(allp) + "\n")
    for p in allp:
        print(p)


if __name__ == "__main__":
    main()
