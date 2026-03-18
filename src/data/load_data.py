from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.config import CONFIG


def _auto_detect_csv(raw_dir: Path) -> Path:
    csvs = sorted(raw_dir.glob("*.csv"))
    if len(csvs) == 1:
        return csvs[0]
    if len(csvs) > 1:
        raise FileNotFoundError(
            f"Multiple CSV files found in {raw_dir}. Keep only one, or set CONFIG.dataset_filename."
        )
    raise FileNotFoundError(
        "No CSV dataset found in data/raw. Put exactly one CSV into data/raw/."
    )


def detect_columns(df: pd.DataFrame) -> tuple[str, str]:
    """Detect date and close columns."""
    date_col = None
    for c in CONFIG.date_candidates:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise KeyError(f"Could not detect a date column. Available columns: {list(df.columns)}")

    close_col = None
    for c in CONFIG.close_candidates:
        if c in df.columns:
            close_col = c
            break
    if close_col is None:
        raise KeyError(f"Could not detect a close/price column. Available columns: {list(df.columns)}")

    return date_col, close_col


def load_raw_prices() -> pd.DataFrame:
    """Load raw CSV into a DataFrame (no cleaning yet)."""
    raw_dir = CONFIG.raw_dir
    raw_dir.mkdir(parents=True, exist_ok=True)

    if CONFIG.dataset_filename:
        path = raw_dir / CONFIG.dataset_filename
        if not path.exists():
            raise FileNotFoundError(f"Dataset file not found: {path}")
    else:
        path = _auto_detect_csv(raw_dir)

    df = pd.read_csv(path)
    # standardize column whitespace
    df.columns = df.columns.str.strip()
    return df
