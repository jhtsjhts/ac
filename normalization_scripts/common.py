from pathlib import Path

import numpy as np
import pandas as pd


RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "raw_data"
NORMALIZED_DATA_DIR = Path(__file__).resolve().parents[1] / "normalized_data"


def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        RAW_DATA_DIR / filename,
        na_values=["", " ", "NA", "N/A"],
    )


def convert_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Convert selected columns to numeric values where they exist."""
    df = df.copy()

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")    # invalid numeric text becomes missing

    return df


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    """Divide two series while treating zero denominators as missing values."""
    return numerator.div(denominator.replace(0, np.nan))


def save_normalized(
    df: pd.DataFrame,
    filename: str,
) -> None:
    """Write a normalized dataframe to the normalized-data directory."""
    # Keep raw data untouched and write each normalized table to its own file.
    NORMALIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(NORMALIZED_DATA_DIR / filename, index=False)