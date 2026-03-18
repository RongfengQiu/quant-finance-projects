from __future__ import annotations

from pathlib import Path


class Settings:
    """Simple project settings for the main ML volatility-forecasting workflow."""

    project_root = Path(__file__).resolve().parents[1]

    # Paths
    raw_data_path = project_root / "data" / "raw" / "AAPL_historical_data.csv"
    dirty_practice_path = project_root / "data" / "raw" / "AAPL_dirty_practice_comma.csv"
    processed_dir = project_root / "data" / "processed"
    splits_dir = project_root / "data" / "splits"
    results_dir = project_root / "data" / "results"
    reports_dir = project_root / "reports" / "figures"
    db_path = project_root / "database" / "vol_forecasting.db"

    # Finance / ML settings
    trading_days = 252
    rolling_windows = (5, 20, 60)
    future_horizon = 20
    naive_baseline_feature = "ewma_vol_ann"

    # Chronological split ratios
    train_ratio = 0.70
    val_ratio = 0.15

    # Light feature screening
    variance_threshold = 1e-8
    correlation_threshold = 0.90

    # Model settings
    ridge_alpha = 1.0
    random_state = 42
    rf_n_estimators = 300
    rf_max_depth = 6
    rf_min_samples_leaf = 5


CONFIG = Settings()
