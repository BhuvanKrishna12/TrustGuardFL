"""
Leakage audit for the 13 model features
=======================================
Reads one or more LABELED CSVs produced by opensky_fl_ids_pipeline.py
(v2 or v3), streams them in chunks and keeps a uniform random sample of at
most --sample-rows rows (bottom-k sampling on a seeded random key, so the
sample is the same regardless of chunk size). For every feature it reports:

  * share missing, by class
  * median, by class
  * for each attack class vs benign: single-feature ROC AUC, using
    max(AUC, 1 - AUC) and filling missing values with -1 exactly as
    federated_simulation.prep_features does; plus a separate AUC of the
    "is missing" indicator alone
  * mutual information with the 6-class attack_type label (quantile-binned
    estimate, missing values get their own bin), in nats and normalised by
    the label entropy
  * a FLAG when any single-feature AUC >= 0.95 or any missing-indicator
    AUC >= 0.9 for some attack class

and joins the numbers with a STATIC per-feature description written from
reading the pipeline code (how it is computed, whether it uses future
observations or labels, whether it is available at inference time, risk,
decision). The decisions match the CORE feature set used at training time.

Outputs (in --out):
  feature_audit.csv          one row per feature: static description + summary numbers
  feature_audit.md           human-readable report
  feature_audit_table.tex    booktabs table* "Features and Leakage Audit" (label tab:audit):
                             No. | Feature | Definition | Causal? | Test-day data? |
                             Construction artifact | Max AUC | Max missing AUC | Core | Decision,
                             with a note line naming the data (--data-label) and thresholds
  region_fallback.csv        output-only lower bound of gps_spoof rows in cells without clean rows
                             (exact numbers: region_lookup_check.py, passed with --region-stats)
  separability.csv           full numbers: one row per (feature, class)

Usage:
  python3 leakage_audit.py day_labeled.csv [more.csv ...] --out audit_out
  python3 leakage_audit.py big.csv --sample-rows 2000000 --seed 42 --out audit_out \
      --data-label "2019-01-07 training day, pipeline v3" --region-stats region_stats.csv
  python3 leakage_audit.py --placeholder paper/generated/tab_audit.tex
"""

import argparse
import ast
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Copied from federated_simulation.py (importing it would pull in torch).
# _check_feature_cols() below verifies the copy against that file's source.
FEATURE_COLS = [
    "implied_speed_mps",
    "speed_consistency_delta",
    "vrate_consistency_delta",
    "inter_arrival_s",
    "turn_rate_deg_s",
    "track_message_count",
    "is_physically_implausible",
    "squawk_changed",
    "icao24_valid_format",
    "is_stale",
    "region_avg_inter_arrival",
    "is_low_coverage_region",
    "is_climbing_or_descending",
]
BOOL_COLS = ["is_physically_implausible", "squawk_changed", "icao24_valid_format",
             "is_stale", "is_low_coverage_region", "is_climbing_or_descending"]

CORE_FEATURES = [
    "implied_speed_mps", "speed_consistency_delta", "vrate_consistency_delta",
    "inter_arrival_s", "turn_rate_deg_s", "region_avg_inter_arrival", "is_stale",
]
EXCLUDED_FROM_CORE = [
    "track_message_count", "squawk_changed", "icao24_valid_format",
    "is_low_coverage_region", "is_physically_implausible", "is_climbing_or_descending",
]

CLASSES = ["benign", "gps_spoof", "velocity_spoof", "replay", "ghost_aircraft", "flooding"]
ATTACKS = CLASSES[1:]
AUC_FLAG = 0.95
MISSING_AUC_FLAG = 0.90
FILL_VALUE = -1  # same as federated_simulation.prep_features

# ---------------------------------------------------------------------------
# Static description, written from reading opensky_fl_ids_pipeline.py.
# "causal": yes = uses only the current and earlier messages of the same
# aircraft (or the row itself); partly = uses contemporaneous unlabeled
# traffic, including later rows of the same chunk; no = uses later rows of
# the same aircraft. Where v2 and v3 differ, the v3 behaviour is described.
# ---------------------------------------------------------------------------
STATIC = {
    "implied_speed_mps": dict(
        computation="Haversine distance to the previous fix of the same icao24 / max(dt, 0.1 s); NaN on first row and session breaks",
        short="dist(prev fix) / max(dt, 0.1 s)",
        causal="yes", uses_future="no", uses_labels="no", inference_available="yes",
        risk="medium", decision="keep",
        reason="Causal and label-free. Risk: single-row ghost tracks are always a first row, so missingness reveals ghost_aircraft; "
               "benign rows next to gps_spoof/replay rows inherit implausible speeds (label noise, not leakage)."),
    "speed_consistency_delta": dict(
        computation="|implied_speed_mps - broadcast velocity|",
        short="|implied speed - velocity|",
        causal="yes", uses_future="no", uses_labels="no", inference_available="yes",
        risk="medium", decision="keep",
        reason="Causal and label-free; it is the intended physics check for velocity_spoof. Same ghost missingness route as implied_speed_mps."),
    "vrate_consistency_delta": dict(
        computation="|(baroalt - prev baroalt) / max(dt, 0.1 s) - vertrate|",
        short="|d(baroalt)/dt - vertrate|",
        causal="yes", uses_future="no", uses_labels="no", inference_available="yes",
        risk="medium", decision="keep",
        reason="Causal and label-free. Construction note: velocity_spoof fills missing vertrate with 0 and adds >= 5 m/s, "
               "so it is never missing and always offset on those rows; ghost missingness route as above."),
    "inter_arrival_s": dict(
        computation="Unfloored time gap to the previous message of the same icao24; NaN on first row and session breaks",
        short="time gap to previous message",
        causal="yes", uses_future="no", uses_labels="no", inference_available="yes",
        risk="medium", decision="keep",
        reason="Causal and label-free; the intended flooding signal. Construction note: OpenSky state vectors are on a regular "
               "(10 s) grid, so the 0.01 to 0.5 s duplicate spacing never occurs in benign data and replay's integer shift breaks the grid."),
    "turn_rate_deg_s": dict(
        computation="Wrapped heading change to previous message / max(dt, 0.1 s)",
        short="wrapped d(heading) / max(dt, 0.1 s)",
        causal="yes", uses_future="no", uses_labels="no", inference_available="yes",
        risk="low", decision="keep",
        reason="Causal and label-free; no attack edits heading, so signal comes only from changed neighbours or missingness (ghost)."),
    "region_avg_inter_arrival": dict(
        computation="Mean clean (pre-injection) inter_arrival_s of the row's 10 degree cell over the whole chunk; unseen cells get the baseline median",
        short="chunk mean gap of the 10 deg cell",
        causal="partly", uses_future="yes (later clean rows of the same chunk, incl. test day)", uses_labels="no",
        inference_available="yes, with a trailing-window regional mean",
        risk="medium", decision="keep",
        reason="Label-free and computed before injection, but transductive over the chunk. gps_spoof rows moved into a cell "
               "with no clean baseline get the fallback median, a value benign rows rarely take. Recommended revision: trailing-window baseline."),
    "is_stale": dict(
        computation="time - lastcontact > 15 s, computed once before injection and copied to flooding duplicates",
        short="time - lastcontact > 15 s",
        causal="yes", uses_future="no", uses_labels="no", inference_available="yes",
        risk="low", decision="keep",
        reason="Row-local and label-free; replay shifts both fields so it is invariant; no attack changes it."),
    "track_message_count": dict(
        computation="v2: total rows of the icao24 in the chunk. v3: running count of messages so far for the icao24 (first = 1)",
        short="v3: running count per icao24",
        causal="yes in v3 (no in v2)", uses_future="v2 yes, v3 no", uses_labels="no", inference_available="yes (v3)",
        risk="high", decision="exclude-from-core",
        reason="Fabricated single-row ghost addresses always get count 1, and flooding bursts inflate counts; it also encodes the row's "
               "position within the chunk. Construction-revealing even when causal."),
    "squawk_changed": dict(
        computation="v2: prev squawk != squawk (first rows and missing codes count as changes). v3: both present and different",
        short="v3: prev and current squawk differ",
        causal="yes", uses_future="no", uses_labels="no", inference_available="yes",
        risk="medium", decision="exclude-from-core",
        reason="No attack modifies squawk, so any separability is incidental; the v2 form was a first-row / missing-code indicator."),
    "icao24_valid_format": dict(
        computation="icao24 matches ^[0-9a-f]{6}$",
        short="icao24 is 6 lowercase hex digits",
        causal="yes", uses_future="no", uses_labels="no", inference_available="yes",
        risk="low", decision="exclude-from-core",
        reason="Constant in practice: real OpenSky addresses and the generated ghost addresses are all valid, so it carries no information."),
    "is_low_coverage_region": dict(
        computation="region_avg_inter_arrival > 0.75 quantile of the chunk's regional baseline",
        short="regional mean gap above chunk Q75",
        causal="partly", uses_future="yes (chunk-wide quantile)", uses_labels="no", inference_available="yes, with a trailing window",
        risk="medium", decision="exclude-from-core",
        reason="Deterministic threshold of region_avg_inter_arrival (redundant), with the same chunk-wide transductive baseline."),
    "is_physically_implausible": dict(
        computation="implied_speed_mps > 400 m/s. v2 also DROPPED benign rows above 400 m/s but kept attack rows; v3 drops nothing",
        short="implied speed > 400 m/s",
        causal="yes", uses_future="no", uses_labels="v2: yes (label-dependent row filter); v3: no", inference_available="yes",
        risk="high", decision="exclude-from-core",
        reason="In v2 the label-dependent filter made it a near one-sided attack indicator. In v3 it is label-agnostic but is "
               "just a threshold of implied_speed_mps, so it adds nothing to the core set."),
    "is_climbing_or_descending": dict(
        computation="|vertrate| > 2.5 m/s, computed after injection",
        short="|vertrate| > 2.5 m/s",
        causal="yes", uses_future="no", uses_labels="no", inference_available="yes",
        risk="high", decision="exclude-from-core",
        reason="Construction-revealing: the velocity_spoof vertrate boost is at least 5 m/s, above the 2.5 m/s threshold, so the flag is "
               "near-deterministic on those rows."),
}


# Combined paper table (replaces the separate feature-definition table).
# Order and numbering follow the paper's original feature table.
TABLE_ORDER = [
    "implied_speed_mps", "speed_consistency_delta", "vrate_consistency_delta",
    "inter_arrival_s", "turn_rate_deg_s", "squawk_changed", "region_avg_inter_arrival",
    "is_low_coverage_region", "icao24_valid_format", "track_message_count", "is_stale",
    "is_physically_implausible", "is_climbing_or_descending",
]
# test_day: whether the value for a test-day row uses other test-day rows.
#   "no"        row-local
#   "own past"  earlier messages of the same aircraft only (causal)
#   "chunk"     statistic over all clean rows of the chunk, incl. later ones
# artifact: which attack's generator makes the feature separable by construction.
TABLE = {
    "implied_speed_mps": dict(defn="haversine to previous fix / max(dt, 0.1 s)", causal="yes",
                              test_day="own past", artifact="ghost: missing (1-row track); flood: speed near 0"),
    "speed_consistency_delta": dict(defn="|implied speed - velocity|", causal="yes",
                                    test_day="own past", artifact="ghost: missing"),
    "vrate_consistency_delta": dict(defn="|d(baroalt)/dt - vertrate|", causal="yes", test_day="own past",
                                    artifact="ghost: missing; vel: vertrate filled, boost at least 5 m/s"),
    "inter_arrival_s": dict(defn="unfloored gap to previous message", causal="yes", test_day="own past",
                            artifact="ghost: missing; flood: sub-second gaps off the 10 s grid"),
    "turn_rate_deg_s": dict(defn="wrapped heading change / max(dt, 0.1 s)", causal="yes",
                            test_day="own past", artifact="ghost: missing"),
    "squawk_changed": dict(defn="previous and current code present and different", causal="yes",
                           test_day="own past", artifact="none (no attack edits squawk)"),
    "region_avg_inter_arrival": dict(defn="clean mean gap of the 10 deg cell over the chunk; median if unmapped",
                                     causal="no", test_day="chunk",
                                     artifact="gps: re-looked-up at spoofed cell"),
    "is_low_coverage_region": dict(defn="regional mean gap above chunk Q75", causal="no", test_day="chunk",
                                   artifact="gps: via region lookup"),
    "icao24_valid_format": dict(defn="address is 6 lowercase hex digits", causal="yes", test_day="no",
                                artifact="none (constant; ghost ids valid)"),
    "track_message_count": dict(defn="running count of the address in the chunk", causal="yes",
                                test_day="own past", artifact="ghost: always 1"),
    "is_stale": dict(defn="time - lastcontact over 15 s", causal="yes", test_day="no", artifact="none"),
    "is_physically_implausible": dict(defn="implied speed over 400 m/s", causal="yes", test_day="own past",
                                      artifact="v2 only: benign rows over 400 m/s dropped"),
    "is_climbing_or_descending": dict(defn="|vertrate| over 2.5 m/s", causal="yes", test_day="no",
                                      artifact="vel: boost of at least 5 m/s"),
}
CLASS_ABBR = {"gps_spoof": "gps", "velocity_spoof": "vel", "replay": "replay",
              "ghost_aircraft": "ghost", "flooding": "flood", "benign": "benign"}


def _check_feature_cols():
    """Warn if FEATURE_COLS drifted from federated_simulation.py (parsed, not imported)."""
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "federated_simulation.py")
    if not os.path.exists(path):
        return
    try:
        tree = ast.parse(open(path).read())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                    getattr(t, "id", None) == "FEATURE_COLS" for t in node.targets):
                theirs = ast.literal_eval(node.value)
                if list(theirs) != FEATURE_COLS:
                    print(f"WARNING: FEATURE_COLS differs from federated_simulation.py: {theirs}")
                return
    except Exception as e:  # never fail the audit because of this check
        print(f"  (could not cross-check FEATURE_COLS: {e})")


def load_sample(paths, sample_rows, seed, chunksize=500_000):
    """Uniform sample without replacement of at most sample_rows rows across
    all files: bottom-k on a seeded uniform key, one streaming pass."""
    rng = np.random.default_rng(seed)
    usecols = FEATURE_COLS + ["attack_type"]
    header = pd.read_csv(paths[0], nrows=0).columns
    if "region_cell" in header:
        usecols = usecols + ["region_cell"]
    kept = None
    total = 0
    for path in paths:
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, low_memory=False):
            total += len(chunk)
            chunk = chunk.copy()
            chunk["_key"] = rng.random(len(chunk))
            kept = chunk if kept is None else pd.concat([kept, chunk], ignore_index=True)
            if len(kept) > sample_rows:
                kept = kept.nsmallest(sample_rows, "_key")
    kept = kept.sort_values("_key").drop(columns="_key").reset_index(drop=True)
    return kept, total


def to_numeric_features(df):
    """Coerce features to float, mapping boolean strings. Missing stays NaN."""
    X = pd.DataFrame(index=df.index)
    for c in FEATURE_COLS:
        s = df[c]
        if c in BOOL_COLS or s.dtype == object or str(s.dtype) in ("bool", "boolean", "string", "str"):
            s = s.map(lambda v: 1.0 if v in (True, "True", "true", 1, 1.0)
                      else (0.0 if v in (False, "False", "false", 0, 0.0) else np.nan))
        X[c] = pd.to_numeric(s, errors="coerce").astype("float64")
    return X


def _auc_sym(y, score):
    if len(np.unique(y)) < 2:
        return np.nan
    if np.all(score == score[0]):
        return 0.5
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def _entropy(counts):
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def binned_mi(x, y_codes, n_bins=32):
    """MI (nats) between a feature and the class label. Continuous features
    are binned into up to n_bins quantile bins; missing values get their own
    bin. Plug-in estimate, fine at the sample sizes used here."""
    xb = np.full(len(x), -1, dtype=np.int64)
    ok = ~np.isnan(x)
    if ok.any():
        vals = x[ok]
        if len(np.unique(vals)) <= n_bins:
            _, xb[ok] = np.unique(vals, return_inverse=True)
        else:
            edges = np.unique(np.quantile(vals, np.linspace(0, 1, n_bins + 1)))
            xb[ok] = np.clip(np.searchsorted(edges, vals, side="right") - 1, 0, len(edges) - 2)
    xb = xb + 1
    joint = pd.crosstab(xb, y_codes).to_numpy().astype(float)
    return _entropy(joint.sum(1)) + _entropy(joint.sum(0)) - _entropy(joint.ravel())


def audit(df):
    X = to_numeric_features(df)
    y = df["attack_type"].astype(str).to_numpy()
    present = [c for c in CLASSES if (y == c).any()]
    y_codes = pd.Categorical(y, categories=present).codes
    h_label = _entropy(np.bincount(y_codes).astype(float))
    sep_rows, summary = [], []
    benign = y == "benign"
    for f in FEATURE_COLS:
        x = X[f].to_numpy()
        miss = np.isnan(x)
        filled = np.where(miss, FILL_VALUE, x)
        mi = binned_mi(x, y_codes)
        per = {}
        for c in present:
            m = y == c
            row = dict(feature=f, cls=c, n=int(m.sum()),
                       missing_share=float(miss[m].mean()) if m.any() else np.nan,
                       median=float(np.nanmedian(x[m])) if (m & ~miss).any() else np.nan,
                       auc_vs_benign=np.nan, missing_auc_vs_benign=np.nan)
            if c != "benign" and m.any() and benign.any():
                sel = m | benign
                yy = m[sel].astype(int)
                row["auc_vs_benign"] = _auc_sym(yy, filled[sel])
                row["missing_auc_vs_benign"] = _auc_sym(yy, miss[sel].astype(float))
            per[c] = row
            sep_rows.append(row)
        atk_rows = [per[c] for c in present if c != "benign"]
        max_auc = max((r["auc_vs_benign"] for r in atk_rows), default=np.nan)
        max_mauc = max((r["missing_auc_vs_benign"] for r in atk_rows), default=np.nan)
        reasons = []
        for r in atk_rows:
            if r["auc_vs_benign"] >= AUC_FLAG:
                reasons.append(f"AUC {r['auc_vs_benign']:.3f} vs benign for {r['cls']}")
            if r["missing_auc_vs_benign"] >= MISSING_AUC_FLAG:
                reasons.append(f"missing-indicator AUC {r['missing_auc_vs_benign']:.3f} for {r['cls']}")
        worst = max(atk_rows, key=lambda r: r["auc_vs_benign"]) if atk_rows else None
        summary.append(dict(
            feature=f,
            in_core=f in CORE_FEATURES,
            max_auc_vs_benign=max_auc,
            max_auc_class=worst["cls"] if worst else "",
            max_missing_auc_vs_benign=max_mauc,
            mutual_info_nats=mi,
            mutual_info_norm=mi / h_label if h_label > 0 else np.nan,
            flagged=bool(reasons),
            flag_reasons="; ".join(reasons),
            **{f"missing_{c}": per[c]["missing_share"] for c in present},
            **{f"median_{c}": per[c]["median"] for c in present},
        ))
    return pd.DataFrame(summary), pd.DataFrame(sep_rows)


def region_diagnostic(df):
    """gps_spoof region re-lookup, PROXY measured from pipeline output.
    A cell is treated as unmapped (no clean baseline, so the row gets the
    baseline-median fallback) when no non-gps_spoof row of the sample lies in
    it. This is a lower bound: the output has no chunk id, so a cell that is
    populated in another chunk, or populated only by rows with no
    inter-arrival (NaN baseline), counts as mapped. Exact numbers come from
    region_lookup_check.py (pass its CSV with --region-stats)."""
    if "region_cell" not in df.columns:
        return None
    y = df["attack_type"].astype(str)
    clean_cells = set(df.loc[y != "gps_spoof", "region_cell"].astype(str))
    unmapped = ~df["region_cell"].astype(str).isin(clean_cells)
    out = {}
    for c in CLASSES:
        m = y == c
        if m.any():
            out[c] = dict(n=int(m.sum()), unmapped_cell_share_lower_bound=float(unmapped[m].mean()))
    return out


def region_artifact_text(region_stats, region_diag):
    """Short 'Construction artifact' text for region_avg_inter_arrival."""
    base = "gps: re-mapped to spoofed cell (and client)"
    if region_stats is not None:
        return (f"{base}; {100 * region_stats['gps_cell_changed_share']:.1f}% change cell, "
                f"{100 * region_stats['gps_fallback_share']:.1f}% get the fallback median "
                f"(benign {100 * region_stats['benign_fallback_share']:.2f}%)")
    if region_diag and "gps_spoof" in region_diag:
        g = region_diag["gps_spoof"]["unmapped_cell_share_lower_bound"]
        b = region_diag.get("benign", {}).get("unmapped_cell_share_lower_bound", float("nan"))
        return f"{base}; at least {100 * g:.1f}% in cells without clean rows (benign {100 * b:.1f}%)"
    return base


def tex_escape(s):
    s = str(s)
    for a, b in [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"),
                 ("#", r"\#"), ("{", r"\{"), ("}", r"\}"), ("^", r"\^{}"), ("~", r"\~{}"),
                 ("$", r"\$")]:
        s = s.replace(a, b)
    # Characters that print wrongly in OT1 text mode.
    for a, b in [("|", r"$|$"), (">", r"$>$"), ("<", r"$<$")]:
        s = s.replace(a, b)
    return s


def write_tex(path, summary, data_label, region_diag=None, region_stats=None):
    """One combined table: definitions (v3) + audit numbers + decision."""
    summ = summary.set_index("feature") if summary is not None else None
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Features and Leakage Audit}\label{tab:audit}",
        r"\centering\scriptsize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabularx}{\textwidth}{r l >{\raggedright\arraybackslash}X l l "
        r">{\raggedright\arraybackslash}X l l c l}\toprule",
        r"No. & Feature & Definition (v3) & Causal? & Test-day data? & Construction artifact & "
        r"Max AUC (attack) & Max miss.\ AUC & Core & Decision \\\midrule",
    ]
    for i, f in enumerate(TABLE_ORDER, 1):
        t = TABLE[f]
        artifact = t["artifact"]
        if f == "region_avg_inter_arrival":
            artifact = region_artifact_text(region_stats, region_diag)
        if summ is None:  # placeholder: static columns only
            auc, mauc, r = "--", "--", None
        else:
            r = summ.loc[f]
            auc = "" if np.isnan(r.max_auc_vs_benign) else \
                f"{r.max_auc_vs_benign:.3f} ({CLASS_ABBR.get(r.max_auc_class, r.max_auc_class)})"
            mauc = "" if np.isnan(r.max_missing_auc_vs_benign) else f"{r.max_missing_auc_vs_benign:.3f}"
        if r is not None and r.flagged:
            auc = auc and r"\textbf{" + tex_escape(auc) + "}"
            if r.max_missing_auc_vs_benign >= MISSING_AUC_FLAG:
                mauc = r"\textbf{" + mauc + "}"
        else:
            auc = tex_escape(auc)
        core = r"$\checkmark$" if f in CORE_FEATURES else ""
        decision = "keep" if f in CORE_FEATURES else "exclude"
        cells = [str(i), tex_escape(f), tex_escape(t["defn"]), t["causal"], tex_escape(t["test_day"]),
                 tex_escape(artifact), auc, mauc, core, decision]
        lines.append(" & ".join(cells) + r" \\")
    note = (f"Audit data: {tex_escape(data_label)}. Flag (bold): single-feature ROC AUC $\\geq$ {AUC_FLAG} "
            f"or missing-indicator AUC $\\geq$ {MISSING_AUC_FLAG} for some attack class vs benign; "
            f"AUC reported as $\\max(\\mathrm{{AUC}}, 1-\\mathrm{{AUC}})$ with missing values filled with $-1$. "
            f"Causal: uses only the row and earlier messages of the same aircraft. "
            f"Test-day data: own past = earlier messages of the same aircraft; chunk = statistic over all "
            f"clean rows of the chunk, including later ones.")
    lines += [r"\bottomrule\end{tabularx}",
              r"\par\smallskip\parbox{\textwidth}{\scriptsize " + note + "}",
              r"\end{table*}", ""]
    open(path, "w").write("\n".join(lines))


def fmt(v, nd=3):
    if isinstance(v, (float, np.floating)):
        return "" if np.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def write_md(path, summary, sep, inputs, n_total, n_sample, region_diag=None, region_stats=None):
    present = [c for c in CLASSES if f"missing_{c}" in summary.columns]
    out = ["# Feature leakage audit", "",
           f"Inputs: {', '.join(inputs)}  ",
           f"Rows read: {n_total:,}; audited sample: {n_sample:,} (uniform, seeded)  ",
           f"Flag rule: single-feature AUC >= {AUC_FLAG} or missing-indicator AUC >= {MISSING_AUC_FLAG} "
           f"for some attack class vs benign (missing filled with {FILL_VALUE}, AUC taken as max(AUC, 1-AUC)).", "",
           "Class counts in sample: " + ", ".join(
               f"{c}={int(sep[(sep.feature == FEATURE_COLS[0]) & (sep.cls == c)].n.iloc[0]):,}" for c in present), "",
           "## Summary", "",
           "| Feature | Core | Risk | Decision | max AUC (class) | max missing AUC | MI (norm) | Flag |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        s = STATIC[r.feature]
        out.append(f"| {r.feature} | {'yes' if r.in_core else 'no'} | {s['risk']} | {s['decision']} | "
                   f"{fmt(r.max_auc_vs_benign)} ({r.max_auc_class}) | {fmt(r.max_missing_auc_vs_benign)} | "
                   f"{fmt(r.mutual_info_norm)} | {'FLAG' if r.flagged else ''} |")
    flagged = summary[summary.flagged]
    out += ["", "## Flagged features", ""]
    if flagged.empty:
        out.append("None.")
    for _, r in flagged.iterrows():
        out.append(f"* **{r.feature}** ({'core' if r.in_core else 'excluded from core'}): {r.flag_reasons}")
    core_flagged = flagged[flagged.in_core]
    if not core_flagged.empty:
        out += ["", "Core features that are flagged are not label-derived (see the static description); "
                "the flag reflects how the attack generator constructs that class. Read the reason column "
                "before treating per-class results for that class as behavioural detection."]
    out += ["", "## Missing share by class", "",
            "| Feature | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    for _, r in summary.iterrows():
        out.append(f"| {r.feature} | " + " | ".join(fmt(r[f'missing_{c}']) for c in present) + " |")
    out += ["", "## Median by class", "",
            "| Feature | " + " | ".join(present) + " |", "|---|" + "---|" * len(present)]
    for _, r in summary.iterrows():
        out.append(f"| {r.feature} | " + " | ".join(fmt(r[f'median_{c}'], 3) for c in present) + " |")
    atk = [c for c in present if c != "benign"]
    out += ["", "## Single-feature AUC vs benign (missing indicator AUC in brackets)", "",
            "| Feature | " + " | ".join(atk) + " |", "|---|" + "---|" * len(atk)]
    for f in FEATURE_COLS:
        cells = []
        for c in atk:
            r = sep[(sep.feature == f) & (sep.cls == c)].iloc[0]
            cells.append(f"{fmt(r.auc_vs_benign)} [{fmt(r.missing_auc_vs_benign)}]")
        out.append(f"| {f} | " + " | ".join(cells) + " |")
    out += ["", "## gps_spoof region re-lookup (region_avg_inter_arrival)", "",
            "Table text: " + region_artifact_text(region_stats, region_diag)]
    if region_stats is not None:
        out += ["", "Exact numbers from region_lookup_check.py:", "",
                "| Metric | Value |", "|---|---|"]
        for k, v in region_stats.items():
            out.append(f"| {k} | {100 * v:.3f}% |" if k.endswith("_share") else f"| {k} | {v:,.0f} |")
    if region_diag:
        out += ["", "Output-only proxy (lower bound; a cell counts as unmapped when no non-gps_spoof "
                "row of the sample lies in it):", "",
                "| Class | rows | in cell without clean rows |", "|---|---|---|"]
        for c, d in region_diag.items():
            out.append(f"| {c} | {d['n']:,} | {100 * d['unmapped_cell_share_lower_bound']:.2f}% |")
    out += ["", "## Static per-feature description", ""]
    for f in FEATURE_COLS:
        s = STATIC[f]
        out += [f"### {f}", "",
                f"* Computation: {s['computation']}",
                f"* Causal: {s['causal']}; uses future info: {s['uses_future']}; uses labels: {s['uses_labels']}",
                f"* Available at inference time: {s['inference_available']}",
                f"* Risk: {s['risk']}; decision: {s['decision']}",
                f"* Reason: {s['reason']}", ""]
    open(path, "w").write("\n".join(out))


def main():
    p = argparse.ArgumentParser(description="Leakage audit of the 13 model features.")
    p.add_argument("inputs", nargs="*", help="Labeled CSV(s) from opensky_fl_ids_pipeline.py")
    p.add_argument("--placeholder", default=None, metavar="TEX",
                   help="Write only a placeholder version of the combined LaTeX table (static "
                        "columns filled, audit numbers shown as --) to this path and exit.")
    p.add_argument("--out", default="audit_out", help="Output directory (default audit_out)")
    p.add_argument("--sample-rows", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--data-label", default=None,
                   help="Text for the note under the LaTeX table describing the audited data. "
                        "Default: built from the input file names.")
    p.add_argument("--region-stats", default=None,
                   help="CSV written by region_lookup_check.py (exact gps_spoof region re-lookup "
                        "numbers for the table); without it an output-only lower bound is used.")
    args = p.parse_args()

    # Consistency of the static decisions with the CORE set.
    for f in FEATURE_COLS:
        want = "keep" if f in CORE_FEATURES else "exclude-from-core"
        assert STATIC[f]["decision"] == want, f
    _check_feature_cols()
    if args.placeholder:
        write_tex(args.placeholder, None,
                  args.data_label or "placeholder; audit numbers pending a run of leakage_audit.py on real data")
        print(f"Wrote placeholder table {args.placeholder}")
        return
    if not args.inputs:
        p.error("at least one labeled CSV is required (or use --placeholder)")

    os.makedirs(args.out, exist_ok=True)
    df, n_total = load_sample(args.inputs, args.sample_rows, args.seed, args.chunksize)
    print(f"Read {n_total:,} rows, auditing a sample of {len(df):,}")
    summary, sep = audit(df)
    region_diag = region_diagnostic(df)
    region_stats = None
    if args.region_stats:
        region_stats = pd.read_csv(args.region_stats, index_col=0)["value"].astype(float).to_dict()
    label = args.data_label or ("pipeline output " + ", ".join(os.path.basename(x) for x in args.inputs))
    label = f"{label} ({len(df):,} sampled of {n_total:,} rows, seed {args.seed})"

    static_df = pd.DataFrame([dict(feature=f, **{k: v for k, v in STATIC[f].items() if k != 'short'})
                              for f in FEATURE_COLS])
    full = static_df.merge(summary, on="feature")
    full.to_csv(os.path.join(args.out, "feature_audit.csv"), index=False)
    sep.to_csv(os.path.join(args.out, "separability.csv"), index=False)
    write_md(os.path.join(args.out, "feature_audit.md"), summary, sep, args.inputs, n_total, len(df),
             region_diag, region_stats)
    write_tex(os.path.join(args.out, "feature_audit_table.tex"), summary, label, region_diag, region_stats)
    if region_diag:
        pd.DataFrame(region_diag).T.to_csv(os.path.join(args.out, "region_fallback.csv"))

    print("\nFlagged features:")
    fl = summary[summary.flagged]
    if fl.empty:
        print("  none")
    for _, r in fl.iterrows():
        print(f"  {r.feature:28s} [{'core' if r.in_core else 'non-core'}] {r.flag_reasons}")
    print(f"\nWrote {args.out}/feature_audit.csv, feature_audit.md, feature_audit_table.tex, separability.csv")


if __name__ == "__main__":
    sys.exit(main())
