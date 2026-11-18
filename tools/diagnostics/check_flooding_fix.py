"""
check_flooding_fix.py
-------------------------------------------------------------------
Checks a labelled CSV for the flooding sentinel signature: the share of
rows, per class, whose four kinematic rate features (implied_speed_mps,
speed_consistency_delta, vrate_consistency_delta, turn_rate_deg_s) are all
blank. Before the 0.1 s time-gap floor (MIN_DT_FLOOR_S in
opensky_fl_ids_pipeline.py) this happened almost only on flooding
duplicates, which a model could learn as a shortcut.

Usage:
    python3 tools/diagnostics/check_flooding_fix.py labeled.csv
-------------------------------------------------------------------
"""
import argparse

import pandas as pd

KINEMATIC = ["implied_speed_mps", "speed_consistency_delta",
             "vrate_consistency_delta", "turn_rate_deg_s"]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", help="labelled CSV written by opensky_fl_ids_pipeline.py")
    path = p.parse_args().path

    k = KINEMATIC
    df = pd.read_csv(path, usecols=["attack_type", "inter_arrival_s"] + k)
    df["all_nan"] = df[k].isna().all(axis=1)

    print("Share of rows with all 4 kinematic cols blank, by class:")
    print(df.groupby("attack_type")["all_nan"].mean().round(4))

    fl = df[df.attack_type == "flooding"]
    print("\nFlooding kinematic columns:")
    print(fl[k].describe().T[["count", "mean", "50%", "max"]])
    print("\nFlooding rows with gap < 1s:", round((fl.inter_arrival_s < 1).mean(), 4))


if __name__ == "__main__":
    main()
