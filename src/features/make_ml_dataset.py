from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG
from src.features.feature_engineering import get_core_feature_columns


TARGET_COL = "future_20d_rv_ann"


def _forward_rolling_std(series: pd.Series, window: int) -> pd.Series:
    """Compute a forward-looking rolling standard deviation.

    For row t, this returns the std of values from t to t+window-1.
    """

    reversed_std = series.iloc[::-1].rolling(window).std()
    return reversed_std.iloc[::-1]


def make_ml_dataset(features_df: pd.DataFrame) -> pd.DataFrame:
    """Create the supervised learning dataset.

    X_t uses information available at time t.
    Y_t is the annualized realized volatility of log returns from t+1 to t+20.
    """

    out = features_df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    future_std = _forward_rolling_std(out["log_return"].shift(-1), CONFIG.future_horizon)
    out[TARGET_COL] = future_std * np.sqrt(CONFIG.trading_days)

    keep_cols = ["date", "close"]
    keep_cols += get_core_feature_columns()
    keep_cols += [TARGET_COL]

    ml_df = out[keep_cols].copy()
    ml_df = ml_df.dropna().reset_index(drop=True)

    return ml_df


def get_target_column() -> str:
    return TARGET_COL
