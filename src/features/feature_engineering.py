from __future__ import annotations#type annotations are stored as strings first, instead of being evaluated immediately


import numpy as np
import pandas as pd

from src.config import CONFIG


CORE_FEATURES = [
    "log_return",      # daily log return of the stock price
    "ret_5d",          # 5-day return, capturing short-term momentum
    "ret_20d",         # 20-day return, capturing medium-term momentum
    "rv_5d_ann",       # annualized realized volatility based on 5-day returns
    "rv_20d_ann",      # annualized realized volatility based on 20-day returns
    "rv_60d_ann",      # annualized realized volatility based on 60-day returns
    "ewma_vol_ann",    # annualized volatility estimated using EWMA (exponentially weighted moving average)
    "ma_20_ratio",     # ratio of current price to 20-day moving average
    "ma_60_ratio",     # ratio of current price to 60-day moving average
    "month",           # calendar month extracted from the date (captures possible seasonal patterns)
]

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create a small, interpretable set of financial ML features."""

    out = df.copy()#irst, I copy the dataset and sort it by date to ensure the time series is in chronological order.
    out = out.sort_values("date").reset_index(drop=True)

    # Basic return feature
    out["log_return"] = np.log(out["close"]).diff()##Then I compute the daily log return .Log returns are commonly used in finance because they are time-additive and more stable.

    # Cumulative historical returns based on log returns
    out["ret_5d"] = out["log_return"].rolling(5).sum()
    out["ret_20d"] = out["log_return"].rolling(20).sum()
    #We compute 5-day and 20-day returns to capture short- and medium-term trends. Normally, multi-period returns cannot be added directly because simple returns must be compounded. But here we use log returns, which are additive over time. So we can sum the daily log returns in a rolling window to obtain multi-day returns.
 # rolling(5).sum() adds the 5 values inside each rolling window

    # Historical realized volatility (annualized)
    for w in CONFIG.rolling_windows:
        out[f"rv_{w}d_ann"] = out["log_return"].rolling(w).std() * np.sqrt(CONFIG.trading_days)

    # EWMA volatility (annualized)
    lam = 0.94
    out["ewma_vol_ann"] = (
        out["log_return"].ewm(alpha=(1 - lam), adjust=False).std() * np.sqrt(CONFIG.trading_days)
    )

    # Trend features
    out["ma_20"] = out["close"].rolling(20).mean()
    out["ma_60"] = out["close"].rolling(60).mean()
    out["ma_20_ratio"] = out["close"] / out["ma_20"]
    out["ma_60_ratio"] = out["close"] / out["ma_60"]

    # Calendar feature
    out["month"] = out["date"].dt.month

    return out


def get_core_feature_columns() -> list[str]:
    return CORE_FEATURES.copy()
