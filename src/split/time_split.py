from __future__ import annotations

import pandas as pd

from src.config import CONFIG


def time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train/validation/test split."""

    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    n = len(out)
    if n < 100:
        raise ValueError(f"Dataset is too small for a train/validation/test split: n={n}")

    train_end = int(n * CONFIG.train_ratio)
    val_end = int(n * (CONFIG.train_ratio + CONFIG.val_ratio))

    train = out.iloc[:train_end].copy()
    val = out.iloc[train_end:val_end].copy()
    test = out.iloc[val_end:].copy()

    if train.empty or val.empty or test.empty:
        raise ValueError(
            f"One split is empty. train={len(train)}, val={len(val)}, test={len(test)}"
        )

    return train, val, test
