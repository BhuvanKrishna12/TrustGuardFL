"""
fs_compat.py: feature preparation for the demo, on top of
federated_simulation.prep_features.

The demo reads its rows back from CSV files, where boolean features come back
as the strings "True"/"False" and a stray non-numeric value can turn a whole
column into text. federated_simulation.prep_features (left unchanged, because
the paper's experiments use it) expects numeric or real boolean columns. This
wrapper converts those columns first, then calls it, and returns float32 as
the demo's models expect.
"""
import pandas as pd

_BOOL_MAP = {"True": 1.0, "true": 1.0, "False": 0.0, "false": 0.0}


def prep_features(fs, df: pd.DataFrame) -> pd.DataFrame:
    """fs.prep_features(df) for CSV-read frames: text booleans and unparseable
    values are converted (bad values become NaN and are filled like any other
    missing value). Returns a float32 DataFrame with fs.FEATURE_COLS."""
    X = df[list(fs.FEATURE_COLS)].copy()
    for col in fs.FEATURE_COLS:
        if not pd.api.types.is_numeric_dtype(X[col]):
            X[col] = pd.to_numeric(X[col].astype(object).replace(_BOOL_MAP), errors="coerce")
    return fs.prep_features(X).astype("float32")
