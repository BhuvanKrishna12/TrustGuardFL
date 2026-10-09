import sys
import pandas as pd

path = sys.argv[1]
k = ["implied_speed_mps", "speed_consistency_delta",
     "vrate_consistency_delta", "turn_rate_deg_s"]
df = pd.read_csv(path, usecols=["attack_type", "inter_arrival_s"] + k)
df["all_nan"] = df[k].isna().all(axis=1)

print("Share of rows with all 4 kinematic cols blank, by class:")
print(df.groupby("attack_type")["all_nan"].mean().round(4))

fl = df[df.attack_type == "flooding"]
print("\nFlooding kinematic columns:")
print(fl[k].describe().T[["count", "mean", "50%", "max"]])
print("\nFlooding rows with gap < 1s:", round((fl.inter_arrival_s < 1).mean(), 4))
