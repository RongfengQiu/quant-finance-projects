from __future__ import annotations

import pandas as pd

from src.data.load_data import detect_columns


def clean_prices(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean raw stock price data.

    Outputs a tidy time-series dataset:
      - date: pandas datetime (timezone-naive)
      - close: float
      - optional OHLC columns if present

    This module is intentionally conservative: it drops invalid rows rather
    than aggressively imputing.
    """

    df = df_raw.copy()
    df.columns = df.columns.str.strip()

    date_col, close_col = detect_columns(df)

    # Parse date; enforce utc=True to avoid mixed-timezone warnings
    dt = pd.to_datetime(df[date_col], errors="coerce", utc=True)
    df["date"] = dt.dt.tz_localize(None)

    # Numeric close
    df["close"] = pd.to_numeric(df[close_col], errors="coerce")

    # Keep any OHLC columns if they exist
    for col in ["Open", "High", "Low", "Volume"]:
        if col in df.columns:
            df[col.lower()] = pd.to_numeric(df[col], errors="coerce")

    # Drop invalid
    df = df.dropna(subset=["date", "close"]).copy()

    # Sort and drop duplicates
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    return df
