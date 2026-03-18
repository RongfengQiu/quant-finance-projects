from __future__ import annotations

import pandas as pd

from src.config import CONFIG


def load_and_clean_prices() -> pd.DataFrame:
    """Load the daily stock-price dataset and apply simple cleaning.

    The main raw file is already fairly clean, so the logic stays deliberately
    small and easy to explain.
    """

    df = pd.read_csv(CONFIG.raw_data_path)

    # Standardize column names
    df.columns = df.columns.str.strip().str.lower()

    # The raw daily file includes timezone text such as -05:00.
    # For a daily dataset we only need the calendar date, so we remove the
    # trailing timezone part and keep a simple naive datetime.
    date_text = df["date"].astype(str).str.replace(r"([+-]\d{2}:\d{2})$", "", regex=True)
    df["date"] = pd.to_datetime(date_text, errors="coerce")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")

    # Convert other numeric columns if present
    for col in ["open", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep rows with valid key information only
    df = df.dropna(subset=["date", "close"]).copy()

    # Daily order and duplicate-date handling
    df = df.sort_values("date")
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.reset_index(drop=True)

    return df
