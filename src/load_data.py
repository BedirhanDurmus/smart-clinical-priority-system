"""Data loading utilities for the Emergency Triage dataset."""

import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "data.csv"


def load_raw(path: Path | str | None = None) -> pd.DataFrame:
    """Load the raw CSV with proper separator and decimal handling."""
    path = Path(path) if path else DATA_PATH
    df = pd.read_csv(path, sep=";", encoding="latin-1")

    # Decimal comma → dot for numeric-like columns stored as strings
    for col in ["KTAS duration_min"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(",", ".", regex=False)
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Strip whitespace from all string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].str.strip()

    return df


def quick_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict with shape, dtypes, missing counts, etc."""
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing_per_col": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
    }
