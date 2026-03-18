from __future__ import annotations

import pandas as pd


def regression_metrics(df: pd.DataFrame, y_true: str, y_pred: str) -> dict:
    err = df[y_pred] - df[y_true]
    mae = float(err.abs().mean())
    rmse = float((err.pow(2).mean()) ** 0.5)
    return {"mae": mae, "rmse": rmse}


def group_error(df: pd.DataFrame, group_cols: list[str], y_true: str, y_pred: str) -> pd.DataFrame:
    out = df.copy()
    out["abs_error"] = (out[y_pred] - out[y_true]).abs()
    g = out.groupby(group_cols, as_index=False).agg(mean_abs_error=("abs_error", "mean"), n=("abs_error", "size"))
    return g.sort_values(group_cols).reset_index(drop=True)
