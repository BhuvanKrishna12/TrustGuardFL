"""
variants.py
-------------------------------------------------------------------
Correction variants for the one-correction-at-a-time ablation.

The multi-variant labelled files (opensky_fl_ids_pipeline.py
--pipeline-version mv, written by reprocess_mv.py) hold the corrected (v3)
columns plus *_legacy columns. A VARIANT is the set of corrections that are
switched back to their legacy behaviour; apply_view() builds that variant's
view of a chunk of rows in memory (column selection + row mask).

Corrections
  C1  track_message_count: causal running count  (legacy: whole-chunk count)
  C2  squawk_changed: both codes present          (legacy: NaN counts as change)
  C3  no label-dependent row filter               (legacy: benign rows with
      implied speed > 400 m/s removed)
  C4  rate denominator max(dt, 0.1 s)             (legacy: the four rate
      features are missing when dt < 1 s)
  C5  flooding originals labelled benign          (legacy: labelled flooding)

What the code shows (opensky_fl_ids_pipeline.py, docs/PIPELINE_V3_CHANGES.md):
pipeline v2 and v3 differ in C1, C2, C3 and C5 only. The 0.1 s floor (C4) is
ALREADY part of v2 ("kept in both versions"). So

  corrected  = no legacy correction            = pipeline v3
  legacy     = {C1, C2, C3, C5} legacy         = pipeline v2 (exact)
  prefloor   = {C1, C2, C3, C4, C5} legacy     = v2 without the floor

C4-legacy is rebuilt from the rule documented in add_tier_a_features ("any
|dt| < 1 s was masked to NaN"); the pre-floor code itself is not in the
repository. It needs no stored column: with dt >= 1 s the floored and
unfloored denominators are identical, with dt < 1 s the legacy rate features
are missing, and dt is inter_arrival_s, which the floor never changed.

Variant specs (--variant)
  corrected | legacy | prefloor
  loo:Cx     all corrected except Cx legacy          (x = 1..5)
  add:Cx     all legacy (= v2) except Cx corrected   (x = 1, 2, 3, 5)
  add:C4     the floor added to the pre-floor pipeline = v2 = same as 'legacy'
  set:C1+C3  explicit set of legacy corrections ('set:' alone = corrected)

Interactions handled here
  * C3 is a pure row mask. In v2 the filter is the last step before writing:
    no feature of a remaining row is recomputed after it, and the whole-chunk
    count (C1 legacy) and the region baseline are computed before it. The
    mask is  benign-under-the-variant's-labels AND implausible-under-the-
    variant's-rate-features, so it follows C5 (flooding originals are benign
    only when C5 is corrected) and C4 (a benign row with dt < 1 s has no
    implied speed without the floor, so the legacy filter keeps it). For the
    pure v2 variant the mask equals the stored legacy_filter_drop column.
  * C4 changes implied_speed_mps, speed_consistency_delta,
    vrate_consistency_delta, turn_rate_deg_s and, through implied speed,
    is_physically_implausible.
  * C1, C2 and C5 are independent column swaps.
Every one of the 32 combinations is represented exactly.
"""

import numpy as np
import pandas as pd

CORRECTIONS = ("C1", "C2", "C3", "C4", "C5")
V2_SET = frozenset({"C1", "C2", "C3", "C5"})       # legacy in pipeline v2
ALL_SET = frozenset(CORRECTIONS)
DESCRIPTIONS = {
    "C1": "causal track_message_count",
    "C2": "squawk_changed needs both codes",
    "C3": "no benign-only >400 m/s row filter",
    "C4": "rate denominator floor max(dt, 0.1 s)",
    "C5": "flooding originals labelled benign",
}
RATE_FEATURES = ["implied_speed_mps", "speed_consistency_delta",
                 "vrate_consistency_delta", "turn_rate_deg_s"]
MV_EXTRA_COLS = ["track_message_count_legacy", "squawk_changed_legacy",
                 "attack_type_legacy", "label_legacy", "legacy_filter_drop"]
# columns of a multi-variant file that the training / test readers need on top
# of the usual ones
MV_READ_COLS = ["track_message_count_legacy", "squawk_changed_legacy", "attack_type_legacy"]
PRE_FLOOR_MIN_DT_S = 1.0

# The six sampling "designs": which labels define the classes, and which rows
# the legacy filter removes. A capped sample depends on the variant only
# through its design.
#   label scheme: "C" corrected labels, "L" legacy labels (C5 legacy)
#   drop:         "N" none, "F" filter with the floor, "P" filter without the floor
DESIGNS = ("CN", "CF", "CP", "LN", "LF", "LP")


def parse_variant(spec):
    """--variant spec -> frozenset of corrections that are LEGACY."""
    s = (spec or "corrected").strip()
    low = s.lower()
    if low in ("corrected", "v3"):
        return frozenset()
    if low in ("legacy", "v2"):
        return V2_SET
    if low == "prefloor":
        return ALL_SET
    if ":" in s:
        kind, arg = s.split(":", 1)
        kind = kind.lower()
        items = [a.strip().upper() for a in arg.replace(",", "+").split("+") if a.strip()]
        for a in items:
            if a not in CORRECTIONS:
                raise ValueError(f"unknown correction {a!r} in variant {spec!r}; use C1..C5")
        if kind == "set":
            return frozenset(items)
        if len(items) != 1:
            raise ValueError(f"{spec!r}: loo:/add: take exactly one correction")
        c = items[0]
        if kind == "loo":
            return frozenset({c})
        if kind == "add":
            return V2_SET - {c}      # add:C4 -> V2_SET, i.e. 'legacy' (see module docstring)
    raise ValueError(f"cannot parse variant {spec!r}")


def variant_name(legacy_set):
    """Canonical name of a variant (used in folder names, tables and JSON)."""
    ls = frozenset(legacy_set)
    if not ls:
        return "corrected"
    if ls == V2_SET:
        return "legacy"
    if ls == ALL_SET:
        return "prefloor"
    if len(ls) == 1:
        return "loo:" + next(iter(ls))
    if ls < V2_SET and len(ls) == len(V2_SET) - 1:
        return "add:" + next(iter(V2_SET - ls))
    return "set:" + "+".join(sorted(ls))


def variant_slug(legacy_set):
    return variant_name(legacy_set).replace(":", "_").replace("+", "")


def design_of(legacy_set):
    ls = frozenset(legacy_set)
    lab = "L" if "C5" in ls else "C"
    drop = "N" if "C3" not in ls else ("P" if "C4" in ls else "F")
    return lab + drop


def is_mv_columns(columns):
    return all(c in columns for c in MV_READ_COLS)


def _as_bool(s):
    """Boolean column as read from CSV (bool, or object with 'True'/'False')."""
    if s.dtype == bool:
        return s.to_numpy()
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).to_numpy() != 0
    return s.astype(str).str.strip().str.lower().isin(("true", "1", "1.0")).to_numpy()


def subsecond_mask(df):
    """Rows whose gap to the previous message is below 1 s (where the pre-floor
    rule made the rate features missing)."""
    ia = pd.to_numeric(df["inter_arrival_s"], errors="coerce").to_numpy(dtype="float64")
    with np.errstate(invalid="ignore"):
        return np.abs(ia) < PRE_FLOOR_MIN_DT_S


def design_labels_and_alive(df, classes_index=None):
    """For a chunk of a multi-variant file: {design: (attack_type array, alive mask)}.
    Needs attack_type, attack_type_legacy, is_physically_implausible, inter_arrival_s."""
    at_c = df["attack_type"].astype(str).to_numpy()
    at_l = df["attack_type_legacy"].astype(str).to_numpy()
    imp_f = _as_bool(df["is_physically_implausible"])
    imp_p = imp_f & ~subsecond_mask(df)
    out = {}
    for d in DESIGNS:
        at = at_c if d[0] == "C" else at_l
        if d[1] == "N":
            alive = np.ones(len(df), bool)
        else:
            alive = ~((at == "benign") & (imp_f if d[1] == "F" else imp_p))
        out[d] = (at, alive)
    return out


def apply_view(df, legacy_set, copy=True):
    """Variant view of rows of a multi-variant file.

    Returns (view, keep): `view` has the standard column names
    (track_message_count, squawk_changed, the rate features,
    is_physically_implausible, attack_type and, if present, label) holding the
    variant's values for EVERY input row; `keep` is the boolean mask of rows
    that exist in the variant (False = removed by the legacy filter). The
    *_legacy helper columns are left in place."""
    ls = frozenset(legacy_set)
    if ls and not is_mv_columns(df.columns):
        raise ValueError("this file has no *_legacy columns (not a multi-variant file); "
                         "only --variant corrected is possible")
    v = df.copy() if copy else df
    if "C1" in ls:
        v["track_message_count"] = df["track_message_count_legacy"].to_numpy()
    if "C2" in ls:
        v["squawk_changed"] = df["squawk_changed_legacy"].to_numpy()
    if "C5" in ls:
        v["attack_type"] = df["attack_type_legacy"].to_numpy()
        if "label" in v.columns and "label_legacy" in df.columns:
            v["label"] = df["label_legacy"].to_numpy()
    imp = _as_bool(df["is_physically_implausible"]) if "is_physically_implausible" in df else None
    if "C4" in ls:
        sub = subsecond_mask(df)
        for c in RATE_FEATURES:
            if c in v.columns:
                col = pd.to_numeric(v[c], errors="coerce").astype("float64").to_numpy(copy=True)
                col[sub] = np.nan
                v[c] = col.astype(df[c].dtype) if df[c].dtype == np.float32 else col
        imp = imp & ~sub
        v["is_physically_implausible"] = imp
    if "C3" in ls:
        keep = ~((v["attack_type"].astype(str).to_numpy() == "benign") & imp)
    else:
        keep = np.ones(len(df), bool)
    return v, keep
