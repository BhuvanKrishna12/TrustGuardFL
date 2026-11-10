"""
Region re-lookup and chunk-boundary check (instrumented pipeline run)
=====================================================================
Runs opensky_fl_ids_pipeline.process_chunk on raw hourly OpenSky files with
light instrumentation (the pipeline itself is not modified) and measures
quantities that cannot be recovered from the labeled output alone:

  * gps_spoof rows whose region_cell (and therefore FL client) changes after
    the spoofed position is re-mapped, and the share that land in a cell with
    no clean baseline and so receive the baseline-median fallback for
    region_avg_inter_arrival; the same fallback share for benign rows
  * benign rows that are the first message of their aircraft in the chunk
    (missing Tier A kinematics, track_message_count = 1 in v3)
  * session breaks (|dt| > SESSION_BREAK_SECONDS) before and after injection
  * ghost addresses that collide with a real address present in the chunk

The labeled output is discarded (written to os.devnull). Pass the CSV this
writes to leakage_audit.py with --region-stats to put the exact numbers in
the audit table.

Usage:
  python3 region_lookup_check.py states_2019-02-04-*.csv --chunk-files 1 --out region_stats.csv
"""

import argparse
import os

import numpy as np
import pandas as pd

import opensky_fl_ids_pipeline as pipe


def run(paths, chunk_files, seed, rate, pipeline_version):
    cap = {}
    _tb, _ap, _fw = pipe.add_tier_b_features, pipe.apply_attack_to_subset, pipe._finalize_and_write

    def tb(df, region_baseline=None, pipeline_version=pipeline_version):
        out, base = _tb(df, region_baseline, pipeline_version=pipeline_version)
        if region_baseline is None:
            cap["baseline"] = base.copy()
            dt = out.groupby("icao24", sort=False)["time"].diff()
            cap["breaks_clean"] = int((dt.abs() > pipe.SESSION_BREAK_SECONDS).sum())
        return out, base

    def ap(combined, idx, atk, rng, pipeline_version=pipeline_version):
        if atk == "gps_spoof":
            cap["gps_idx"] = np.asarray(idx).copy()
            cap["gps_cell_before"] = pipe.compute_region_cell(
                combined.loc[idx, "lat"], combined.loc[idx, "lon"]).to_numpy()
        if atk == "ghost_aircraft":
            cap["real_ids"] = set(combined["icao24"].unique())
        r = _ap(combined, idx, atk, rng, pipeline_version=pipeline_version)
        if atk == "ghost_aircraft":
            cap["ghost_ids"] = combined.loc[idx, "icao24"].to_numpy()
        return r

    def fw(piece, out_path, write_header, pipeline_version=pipeline_version):
        cap["piece"] = piece[["icao24", "time", "label", "region_cell", "track_message_count",
                              "implied_speed_mps"]].copy()
        return _fw(piece, out_path, write_header, pipeline_version=pipeline_version)

    pipe.add_tier_b_features, pipe.apply_attack_to_subset, pipe._finalize_and_write = tb, ap, fw
    try:
        paths = sorted(paths)
        chunks = [paths[i:i + chunk_files] for i in range(0, len(paths), chunk_files)]
        a = dict(gps_rows=0, gps_cell_changed=0, gps_fallback=0, benign_rows=0, benign_fallback=0,
                 benign_first_in_chunk=0, benign_missing_kinematics=0, session_breaks_clean=0,
                 session_breaks_post_injection=0, ghost_rows=0, ghost_address_collisions=0)
        for i, ch in enumerate(chunks, 1):
            cap.clear()
            pipe.process_chunk(ch, f"chunk {i}/{len(chunks)}", os.devnull, True, rate=rate,
                               seed=seed, pipeline_version=pipeline_version)
            base, piece = cap["baseline"], cap["piece"]
            mapped = set(base.dropna().index)
            gi = cap["gps_idx"]
            after = piece.loc[gi, "region_cell"].to_numpy()
            a["gps_rows"] += len(gi)
            a["gps_cell_changed"] += int((after != cap["gps_cell_before"]).sum())
            a["gps_fallback"] += int((~pd.Series(after).isin(mapped)).sum())
            ben = piece[piece["label"] == 0]
            a["benign_rows"] += len(ben)
            a["benign_fallback"] += int((~ben["region_cell"].isin(mapped)).sum())
            if pipeline_version == "v3":
                a["benign_first_in_chunk"] += int((ben["track_message_count"] == 1).sum())
            else:
                s = piece.sort_values(["icao24", "time"])
                first = ~s["icao24"].duplicated()
                a["benign_first_in_chunk"] += int((first & (s["label"] == 0)).sum())
            a["benign_missing_kinematics"] += int(ben["implied_speed_mps"].isna().sum())
            a["session_breaks_clean"] += cap["breaks_clean"]
            s = piece.sort_values(["icao24", "time"])
            dto = s.groupby("icao24", sort=False)["time"].diff()
            a["session_breaks_post_injection"] += int((dto.abs() > pipe.SESSION_BREAK_SECONDS).sum())
            a["ghost_rows"] += len(cap["ghost_ids"])
            a["ghost_address_collisions"] += int(pd.Series(cap["ghost_ids"]).isin(cap["real_ids"]).sum())
    finally:
        pipe.add_tier_b_features, pipe.apply_attack_to_subset, pipe._finalize_and_write = _tb, _ap, _fw
    a["chunks"] = len(chunks)
    a["chunk_files"] = chunk_files
    a["gps_cell_changed_share"] = a["gps_cell_changed"] / max(a["gps_rows"], 1)
    a["gps_fallback_share"] = a["gps_fallback"] / max(a["gps_rows"], 1)
    a["benign_fallback_share"] = a["benign_fallback"] / max(a["benign_rows"], 1)
    a["benign_first_in_chunk_share"] = a["benign_first_in_chunk"] / max(a["benign_rows"], 1)
    a["benign_missing_kinematics_share"] = a["benign_missing_kinematics"] / max(a["benign_rows"], 1)
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("inputs", nargs="+", help="Raw hourly OpenSky CSVs")
    p.add_argument("--chunk-files", type=int, default=pipe.CHUNK_FILES)
    p.add_argument("--rate", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=pipe.RNG_SEED)
    p.add_argument("--pipeline-version", choices=pipe.PIPELINE_VERSIONS,
                   default=pipe.DEFAULT_PIPELINE_VERSION)
    p.add_argument("--out", default="region_stats.csv")
    args = p.parse_args()
    a = run(args.inputs, args.chunk_files, args.seed, args.rate, args.pipeline_version)
    pd.Series(a, name="value").rename_axis("metric").to_csv(args.out)
    print()
    for k, v in a.items():
        print(f"  {k:34s} {v:.4%}" if k.endswith("_share") else f"  {k:34s} {v:,}")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
