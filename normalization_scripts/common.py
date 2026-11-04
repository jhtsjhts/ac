from pathlib import Path

import numpy as np
import pandas as pd


RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "raw_data"
NORMALIZED_DATA_DIR = Path(__file__).resolve().parents[1] / "normalized_data"
START_YEAR = 1995
END_YEAR = 2009
EXCLUDED_YEARS = {2004}
LEAGUE = "NHL"


def load_csv(filename: str) -> pd.DataFrame:
    return pd.read_csv(
        RAW_DATA_DIR / filename,
        na_values=["", " ", "NA", "N/A"],
    )


def keep_model_window(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the NHL seasons selected for the rudimentary model."""
    df = df.copy()
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    return df[
        (df["lgID"] == LEAGUE)
        & df["year"].between(START_YEAR, END_YEAR)
        & ~df["year"].isin(EXCLUDED_YEARS)
    ].copy()


def convert_numeric_columns(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    df = df.copy()

    for column in columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def safe_divide(
    numerator: pd.Series,
    denominator: pd.Series,
) -> pd.Series:
    return numerator.div(denominator.replace(0, np.nan))


def add_season_z_scores(
    df: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    df = df.copy()

    for column in columns:
        if column not in df.columns:
            continue

        grouped = df.groupby("year")[column]
        mean = grouped.transform("mean")
        std = grouped.transform("std")

        df[f"{column}_z"] = (
            (df[column] - mean) / std.replace(0, np.nan)
        )

    return df


def save_normalized(
    df: pd.DataFrame,
    filename: str,
) -> None:
    NORMALIZED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(NORMALIZED_DATA_DIR / filename, index=False)