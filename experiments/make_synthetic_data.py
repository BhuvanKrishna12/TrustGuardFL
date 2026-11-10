"""
make_synthetic_data.py
-------------------------------------------------------------------
Small synthetic labelled CSVs with the schema of the real labelled day files
(all columns federated_simulation.load_data and run_experiments.py read), for
smoke-testing the experiment harness. Features carry a learnable,
class-dependent signal; the class mix is the natural one (~79% benign).

Usage:
  python3 experiments/make_synthetic_data.py --out-dir /tmp/synth \
      --train-rows 150000 --holdout-rows 40000
Writes train_day1.csv, train_day2.csv, holdout.csv (holdout spans 4 UTC dates
via lastposupdate; replay rows have 'time' shifted back, sometimes across
midnight, to exercise the day derivation).
"""

import argparse
import os

import numpy as np
import pandas as pd

CLASSES = ["benign", "flooding", "ghost_aircraft", "gps_spoof", "replay", "velocity_spoof"]
MIX = [0.7943, 0.1311, 0.0186, 0.0186, 0.0187, 0.0187]
BOOL_COLS = ["is_physically_implausible", "squawk_changed", "icao24_valid_format",
             "is_stale", "is_low_coverage_region", "is_climbing_or_descending"]


def make(n, dates, seed, n_regions=16):
    rng = np.random.default_rng(seed)
    # skewed region sizes so some regions fall under the client minimum
    w = rng.pareto(1.2, n_regions) + 0.15
    w[:2] = 0.02
    w /= w.sum()
    reg = rng.choice(n_regions, size=n, p=w)
    region_cell = np.array([f"{40 + r // 4}_{-5 + r % 4}" for r in reg])
    lab = rng.choice(len(CLASSES), size=n, p=MIX)
    # regional heterogeneity: some regions see extra flooding
    flood_boost = np.isin(reg, [3, 7]) & (rng.random(n) < 0.15)
    lab[flood_boost] = 1

    speed = rng.gamma(4, 50, n)
    sp_delta = rng.exponential(5, n)
    vr_delta = rng.exponential(1, n)
    inter = rng.exponential(5, n)
    turn = rng.normal(0, 1, n)
    count = rng.poisson(200, n).astype(float)
    region_avg = 3 + reg * 0.3 + rng.normal(0, 0.2, n)
    stale = rng.random(n) < 0.03
    implaus = rng.random(n) < 0.001
    squawk = rng.random(n) < 0.01
    climb = rng.random(n) < 0.3

    m = lab == 1  # flooding: tiny gaps
    inter[m] = rng.uniform(0.01, 0.5, m.sum())
    speed[m] *= rng.uniform(1, 4, m.sum())
    m = lab == 2  # ghost: blank kinematics, count 1
    count[m] = 1
    m = lab == 3  # gps: speed jump
    speed[m] += rng.uniform(100, 4000, m.sum())
    sp_delta[m] += rng.uniform(50, 2000, m.sum())
    implaus[m] = rng.random(m.sum()) < 0.5
    m = lab == 4  # replay: weaker signal, negative-ish gaps
    inter[m] = rng.normal(-50, 150, m.sum())
    m = lab == 5  # velocity spoof
    sp_delta[m] += rng.uniform(20, 600, m.sum())
    vr_delta[m] += rng.uniform(3, 40, m.sum())
    climb[m] = True

    df = pd.DataFrame({
        "implied_speed_mps": speed, "speed_consistency_delta": sp_delta,
        "vrate_consistency_delta": vr_delta, "inter_arrival_s": inter,
        "turn_rate_deg_s": turn, "track_message_count": count,
        "is_physically_implausible": implaus, "squawk_changed": squawk,
        "icao24_valid_format": np.ones(n, bool), "is_stale": stale,
        "region_avg_inter_arrival": region_avg,
        "is_low_coverage_region": np.isin(reg, [0, 1, 15]),
        "is_climbing_or_descending": climb,
    })
    # missing values: first row of an aircraft block, ghosts
    first = rng.random(n) < 0.02
    ghost = lab == 2
    for c in ["implied_speed_mps", "speed_consistency_delta", "vrate_consistency_delta",
              "turn_rate_deg_s"]:
        df.loc[first | ghost, c] = np.nan
    df.loc[first | ghost, "inter_arrival_s"] = np.nan
    df["region_cell"] = region_cell
    df["attack_type"] = np.array(CLASSES)[lab]
    df["label"] = (lab != 0).astype(int)
    df["icao24"] = [f"{v:06x}" for v in rng.integers(0x300000, 0x500000, n)]
    df.loc[ghost, "icao24"] = [f"{v:06x}" for v in rng.integers(0xf00000, 0xffffff, ghost.sum())]

    day_idx = rng.integers(0, len(dates), n)
    base = np.array([pd.Timestamp(d, tz="UTC").timestamp() for d in dates])[day_idx]
    lpu = base + rng.uniform(0, 86400, n)
    t = lpu + rng.uniform(0, 5, n)
    rp = lab == 4
    t[rp] -= rng.integers(2, 900, rp.sum())
    # some replay rows right after midnight: time falls on the previous date
    near = rp & (rng.random(n) < 0.3)
    lpu[near] = base[near] + rng.uniform(0, 300, near.sum())
    t[near] = lpu[near] - 600
    df["time"] = t.round(0)
    df["lastposupdate"] = lpu.round(3)
    # a few rows with missing lastposupdate (day falls back to time)
    df.loc[rng.random(n) < 0.002, "lastposupdate"] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--train-rows", type=int, default=150_000, help="rows per training file")
    ap.add_argument("--holdout-rows", type=int, default=40_000)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    make(args.train_rows, ["2019-01-07"], 11).to_csv(
        os.path.join(args.out_dir, "train_day1.csv"), index=False)
    make(args.train_rows, ["2019-04-01"], 12).to_csv(
        os.path.join(args.out_dir, "train_day2.csv"), index=False)
    make(args.holdout_rows, ["2019-12-02", "2019-12-09", "2019-12-16", "2019-12-23"], 13).to_csv(
        os.path.join(args.out_dir, "holdout.csv"), index=False)
    print(f"wrote synthetic CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
