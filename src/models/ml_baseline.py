from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


@dataclass
class MLResult:
    model_name: str
    mae: float
    rmse: float


def train_and_predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.Series, MLResult]:
    """Optional ML baseline: predict BS price from simple features.

    Keeps things intentionally simple for internship portfolios.

    Requires scikit-learn. If not installed, raise a helpful error.
    """

    try:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_absolute_error, mean_squared_error
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "scikit-learn is required for the ML baseline. Install with: pip install scikit-learn"
        ) from e

    feature_cols = ["log_moneyness", "T", "r", "sigma_used"]
    X_train = train_df[feature_cols].astype(float)
    y_train = train_df["bs_price"].astype(float)

    X_test = test_df[feature_cols].astype(float)
    y_test = test_df["bs_price"].astype(float)

    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)

    pred = pd.Series(model.predict(X_test), index=test_df.index, name="ml_pred")

    mae = float(mean_absolute_error(y_test, pred))
    rmse = float(mean_squared_error(y_test, pred, squared=False))

    return pred, MLResult(model_name="Ridge", mae=mae, rmse=rmse)
