from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.config import CONFIG


@dataclass
class ModelResult:
    model_name: str
    mae: float
    rmse: float
    r2: float


NAIVE_MODEL_NAME = "NaiveEWMA"


def _build_model_dict() -> dict:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression, Ridge

    return {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=CONFIG.ridge_alpha),
        "RandomForest": RandomForestRegressor(
            n_estimators=CONFIG.rf_n_estimators,
            max_depth=CONFIG.rf_max_depth,
            min_samples_leaf=CONFIG.rf_min_samples_leaf,
            random_state=CONFIG.random_state,
            n_jobs=-1,
        ),
    }


def _score_predictions(y_true, y_pred, model_name: str) -> ModelResult:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return ModelResult(
        model_name=model_name,
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(mean_squared_error(y_true, y_pred) ** 0.5),
        r2=float(r2_score(y_true, y_pred)),
    )


def _naive_baseline_predict(df: pd.DataFrame) -> pd.Series:
    feature_name = CONFIG.naive_baseline_feature
    if feature_name not in df.columns:
        raise KeyError(f"Naive baseline feature not found: {feature_name}")
    return df[feature_name].astype(float).copy()


def _predictions_frame(
    df: pd.DataFrame,
    target_col: str,
    predictions: pd.Series,
    model_name: str,
    is_selected_model: bool,
) -> pd.DataFrame:
    out = pd.DataFrame({
        "date": pd.to_datetime(df["date"]),
        target_col: df[target_col].astype(float).values,
        "prediction": pd.Series(predictions, index=df.index).astype(float).values,
    })
    out["residual"] = out[target_col] - out["prediction"]
    out["model_name"] = model_name
    out["is_selected_model"] = int(is_selected_model)
    return out


def train_and_validate_models(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
) -> tuple[pd.DataFrame, str]:
    """Train ML models on the train set and compare them with a naive baseline on validation."""

    X_train = train_df[feature_cols].astype(float)
    y_train = train_df[target_col].astype(float)
    X_val = val_df[feature_cols].astype(float)
    y_val = val_df[target_col].astype(float)

    models = _build_model_dict()

    metric_rows = []
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        val_pred = model.predict(X_val)
        result = _score_predictions(y_val, val_pred, model_name)
        metric_rows.append(result.__dict__)

    naive_pred = _naive_baseline_predict(val_df)
    naive_result = _score_predictions(y_val, naive_pred, NAIVE_MODEL_NAME)
    metric_rows.append(naive_result.__dict__)

    metrics_df = pd.DataFrame(metric_rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    best_model_name = str(metrics_df.loc[0, "model_name"])
    metrics_df["selected_on_validation"] = (metrics_df["model_name"] == best_model_name).astype(int)

    return metrics_df, best_model_name


def evaluate_models_on_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    best_model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, object | None]:
    """Refit ML models on train+validation, compare all models on test, and return selected-model predictions."""

    full_train = pd.concat([train_df, val_df], ignore_index=True)
    X_train = full_train[feature_cols].astype(float)
    y_train = full_train[target_col].astype(float)
    X_test = test_df[feature_cols].astype(float)
    y_test = test_df[target_col].astype(float)

    models = _build_model_dict()
    metric_rows = []
    prediction_frames = []
    selected_model_object = None

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        test_pred = pd.Series(model.predict(X_test), index=test_df.index)
        result = _score_predictions(y_test, test_pred, model_name)
        metric_rows.append(result.__dict__)
        prediction_frames.append(
            _predictions_frame(
                test_df,
                target_col,
                test_pred,
                model_name=model_name,
                is_selected_model=(model_name == best_model_name),
            )
        )
        if model_name == best_model_name:
            selected_model_object = model

    naive_pred = _naive_baseline_predict(test_df)
    naive_result = _score_predictions(y_test, naive_pred, NAIVE_MODEL_NAME)
    metric_rows.append(naive_result.__dict__)
    prediction_frames.append(
        _predictions_frame(
            test_df,
            target_col,
            naive_pred,
            model_name=NAIVE_MODEL_NAME,
            is_selected_model=(NAIVE_MODEL_NAME == best_model_name),
        )
    )
    if best_model_name == NAIVE_MODEL_NAME:
        selected_model_object = None

    test_metrics_df = pd.DataFrame(metric_rows).sort_values(["rmse", "mae"]).reset_index(drop=True)
    test_metrics_df["selected_on_validation"] = (test_metrics_df["model_name"] == best_model_name).astype(int)

    test_predictions_df = pd.concat(prediction_frames, ignore_index=True)
    test_predictions_df = test_predictions_df.sort_values(["model_name", "date"]).reset_index(drop=True)

    return test_predictions_df, test_metrics_df, selected_model_object
