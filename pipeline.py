from __future__ import annotations

import argparse
import shutil

import pandas as pd

from src.config import CONFIG
from src.data.load_and_clean_data import load_and_clean_prices
from src.database.db_utils import connect, write_table
from src.evaluation.evaluate_models import save_ml_plots
from src.features.feature_engineering import compute_features, get_core_feature_columns
from src.features.feature_selection import light_feature_screening
from src.features.make_ml_dataset import get_target_column, make_ml_dataset
from src.models.train_models import evaluate_models_on_test, train_and_validate_models
from src.split.time_split import time_split


def _clear_outputs() -> None:
    for folder in [CONFIG.processed_dir, CONFIG.splits_dir, CONFIG.results_dir, CONFIG.reports_dir]:
        if folder.exists():
            for path in folder.glob("*"):
                if path.name == ".gitkeep":
                    continue
                if path.is_file():
                    path.unlink()
                else:
                    shutil.rmtree(path, ignore_errors=True)

    if CONFIG.db_path.exists():
        CONFIG.db_path.unlink()


def run_pipeline(recompute: bool = False) -> None:
    """Run the main machine-learning workflow for future realized-volatility forecasting."""

    if recompute:
        _clear_outputs()

    CONFIG.processed_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.splits_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.results_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.reports_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.db_path.parent.mkdir(parents=True, exist_ok=True)

    print("Step 1/8 - Load and clean raw data")
    cleaned = load_and_clean_prices()
    cleaned.to_csv(CONFIG.processed_dir / "cleaned_prices.csv", index=False)

    print("Step 2/8 - Feature engineering")
    features = compute_features(cleaned)
    features.to_csv(CONFIG.processed_dir / "features_prices.csv", index=False)

    print("Step 3/8 - Build supervised ML dataset")
    ml_dataset = make_ml_dataset(features)
    ml_dataset.to_csv(CONFIG.processed_dir / "ml_dataset.csv", index=False)

    print("Step 4/8 - Chronological train/validation/test split")
    train_df, val_df, test_df = time_split(ml_dataset)
    train_df.to_csv(CONFIG.splits_dir / "train.csv", index=False)
    val_df.to_csv(CONFIG.splits_dir / "val.csv", index=False)
    test_df.to_csv(CONFIG.splits_dir / "test.csv", index=False)
    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")

    feature_cols = get_core_feature_columns()
    target_col = get_target_column()

    print("Step 5/8 - Light feature screening (train only)")
    screening = light_feature_screening(train_df, feature_cols, target_col)
    selected_features = screening.selected_features
    print("Selected features:", selected_features)

    screening.variance_table.to_csv(CONFIG.results_dir / "feature_variance_report.csv", index=False)
    screening.target_correlation_table.to_csv(CONFIG.results_dir / "feature_target_correlation.csv", index=False)
    screening.high_corr_pairs_table.to_csv(CONFIG.results_dir / "feature_high_correlation_pairs.csv", index=False)
    pd.DataFrame({"selected_feature": selected_features}).to_csv(
        CONFIG.results_dir / "selected_features.csv", index=False
    )

    print("Step 6/8 - Train models and compare them with a naive volatility baseline on validation")
    val_metrics_df, best_model_name = train_and_validate_models(
        train_df=train_df,
        val_df=val_df,
        feature_cols=selected_features,
        target_col=target_col,
    )
    val_metrics_df.to_csv(CONFIG.results_dir / "validation_metrics.csv", index=False)
    print("Validation metrics:")
    print(val_metrics_df)
    print(f"Best model selected on validation: {best_model_name}")

    print("Step 7/8 - Evaluate all models on test after validation-based model selection")
    test_predictions_df, test_metrics_df, selected_model = evaluate_models_on_test(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        feature_cols=selected_features,
        target_col=target_col,
        best_model_name=best_model_name,
    )
    test_predictions_df.to_csv(CONFIG.results_dir / "test_predictions.csv", index=False)
    test_metrics_df.to_csv(CONFIG.results_dir / "test_metrics.csv", index=False)
    print("Test metrics:")
    print(test_metrics_df)

    save_ml_plots(test_predictions_df, selected_model, selected_features, CONFIG.reports_dir)

    print("Step 8/8 - Save tables to SQLite")
    conn = connect(CONFIG.db_path)
    write_table(conn, "prices_cleaned", cleaned)
    write_table(conn, "features_prices", features)
    write_table(conn, "ml_dataset", ml_dataset)
    write_table(conn, "train_split", train_df)
    write_table(conn, "validation_split", val_df)
    write_table(conn, "test_split", test_df)
    write_table(conn, "validation_metrics", val_metrics_df)
    write_table(conn, "test_metrics", test_metrics_df)
    write_table(conn, "test_predictions", test_predictions_df)
    write_table(conn, "feature_variance_report", screening.variance_table)
    write_table(conn, "feature_target_correlation", screening.target_correlation_table)
    write_table(conn, "feature_high_correlation_pairs", screening.high_corr_pairs_table)
    conn.close()

    print("Pipeline finished successfully.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Financial ML workflow: predict future 20-day annualized realized volatility"
    )
    parser.add_argument("--recompute", action="store_true", help="Delete old generated outputs first")
    args = parser.parse_args()

    run_pipeline(recompute=args.recompute)


if __name__ == "__main__":
    main()
