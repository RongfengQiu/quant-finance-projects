from __future__ import annotations

import pandas as pd

from src.config import CONFIG


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/val/test split to avoid look-ahead bias."""

    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    train_end = pd.to_datetime(CONFIG.train_end)
    val_end = pd.to_datetime(CONFIG.val_end)

    train = out[out["date"] <= train_end].copy()
    val = out[(out["date"] > train_end) & (out["date"] <= val_end)].copy()
    test = out[out["date"] > val_end].copy()

    # Integrity checks
    if train.empty or val.empty or test.empty:
        raise ValueError(
            "One of the splits is empty. Adjust CONFIG.train_end / CONFIG.val_end. "
            f"train={len(train)}, val={len(val)}, test={len(test)}"
        )

    if not train["date"].is_monotonic_increasing:
        raise ValueError("Train split is not sorted by date.")
    if not val["date"].is_monotonic_increasing:
        raise ValueError("Val split is not sorted by date.")
    if not test["date"].is_monotonic_increasing:
        raise ValueError("Test split is not sorted by date.")

    if train["date"].max() >= val["date"].min() or val["date"].max() >= test["date"].min():
        raise ValueError("Splits overlap (data leakage risk).")

    return train, val, test
