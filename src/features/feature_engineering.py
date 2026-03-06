from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add finance features used in evaluation.

    Minimal but industry-standard:
    - log returns
    - rolling realized volatility (annualized)
    - EWMA volatility (annualized)
    - simple moving average ratios
    """

    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    out["log_return"] = np.log(out["close"]).diff()

    # Rolling realized vol
    for w in CONFIG.rolling_windows:
        out[f"rv_{w}d_ann"] = out["log_return"].rolling(w).std() * np.sqrt(CONFIG.trading_days)

    # EWMA volatility (daily std -> annualized)
    lam = 0.94
    out["ewma_vol_ann"] = out["log_return"].ewm(alpha=(1 - lam), adjust=False).std() * np.sqrt(CONFIG.trading_days)

    # Trend features (simple, interpretable)
    for w in (20, 60):
        out[f"ma_{w}"] = out["close"].rolling(w).mean()
        out[f"ma_{w}_ratio"] = out["close"] / out[f"ma_{w}"]

    out["year"] = out["date"].dt.year
    out["month"] = out["date"].dt.month

    return out
