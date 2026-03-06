from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG


def estimate_sigma_train(train: pd.DataFrame) -> float:
    """Estimate a single annualized volatility using TRAIN only."""
    rets = train["log_return"].dropna()
    if rets.empty:
        raise ValueError("Not enough returns in train to estimate volatility.")
    return float(rets.std(ddof=1) * np.sqrt(CONFIG.trading_days))
