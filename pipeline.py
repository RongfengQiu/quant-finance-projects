from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import CONFIG
from src.data.load_data import load_raw_prices
from src.data.clean_data import clean_prices
from src.features.feature_engineering import compute_features
from src.split.time_split import time_split
from src.volatility.estimate_volatility import estimate_sigma_train
from src.features.option_dataset import build_option_dataset
from src.evaluation.metrics import regression_metrics, group_error
from src.evaluation.visualization import save_error_plots
from src.database.db_utils import connect, init_db, write_table, append_table



def _clear_outputs() -> None:
    """Delete generated artifacts for a clean recompute run."""
    # intermediate outputs
    for d in [CONFIG.processed_dir, CONFIG.splits_dir]:
        if d.exists():
            for p in d.glob("*"):
                if p.name == ".gitkeep":
                    continue
                if p.is_file():
                    p.unlink()
                else:
                    import shutil
                    shutil.rmtree(p, ignore_errors=True)

    # results
    if CONFIG.results_dir.exists():
        for p in CONFIG.results_dir.glob("*"):
            if p.name == ".gitkeep":
                continue
            if p.is_file():
                p.unlink()
            else:
                import shutil
                shutil.rmtree(p, ignore_errors=True)

    # figures
    fig_dir = Path("reports/figures")
    if fig_dir.exists():
        for p in fig_dir.glob("*.png"):
            p.unlink()

    # db
    if CONFIG.db_path.exists():
        CONFIG.db_path.unlink()


def run_pipeline(
    recompute: bool = False,
    n_sim: int | None = None,
    sample_every_n_days: int | None = None,
    run_ml: bool | None = None,
) -> None:
    """Run the end-to-end data science workflow.

    Steps:
      1) Load raw CSV
      2) Clean data
      3) Feature engineering
      4) Time split (train/val/test)
      5) Estimate volatility from TRAIN only
      6) Build option dataset and run pricing (BS + MC)
      7) Evaluate (MAE/RMSE) + save plots
      8) Write results to SQLite + export CSV summaries
    """

    # Use CLI overrides if provided
    n_sim = int(n_sim) if n_sim is not None else CONFIG.n_sim
    sample_every_n_days = int(sample_every_n_days) if sample_every_n_days is not None else CONFIG.sample_every_n_days
    run_ml = bool(run_ml) if run_ml is not None else CONFIG.run_ml_baseline


    if recompute:
        _clear_outputs()
    # Ensure dirs
    CONFIG.processed_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.splits_dir.mkdir(parents=True, exist_ok=True)
    CONFIG.results_dir.mkdir(parents=True, exist_ok=True)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    CONFIG.db_path.parent.mkdir(parents=True, exist_ok=True)

    print("Step 1/9 - Load raw data")
    raw = load_raw_prices()

    print("Step 2/9 - Data cleaning")
    cleaned = clean_prices(raw)

    cleaned_out = CONFIG.processed_dir / "cleaned_prices.csv"
    cleaned.to_csv(cleaned_out, index=False)
    print(f"Saved: {cleaned_out}")

    print("Step 3/9 - Feature engineering")
    features = compute_features(cleaned)

    features_out = CONFIG.processed_dir / "features_prices.csv"
    features.to_csv(features_out, index=False)
    print(f"Saved: {features_out}")

    print("Step 4/9 - Train/Val/Test split (chronological)")
    train_df, val_df, test_df = time_split(features)

    train_df.to_csv(CONFIG.splits_dir / "train.csv", index=False)
    val_df.to_csv(CONFIG.splits_dir / "val.csv", index=False)
    test_df.to_csv(CONFIG.splits_dir / "test.csv", index=False)

    print(f"Train rows: {len(train_df)}, Val rows: {len(val_df)}, Test rows: {len(test_df)}")

    print("Step 5/9 - Estimate volatility (TRAIN only)")
    sigma_train = estimate_sigma_train(train_df)
    print(f"sigma_train_ann = {sigma_train:.6f}")

    print("Step 6/9 - Build option dataset + run pricing models")
    train_opts = build_option_dataset(
        train_df,
        split_name="train",
        sigma_train_ann=sigma_train,
        r=CONFIG.r,
        maturities=CONFIG.maturities_years,
        moneyness=CONFIG.moneyness,
        n_sim=n_sim,
        seed=CONFIG.seed,
        sample_every_n_days=sample_every_n_days,
    )
    val_opts = build_option_dataset(
        val_df,
        split_name="val",
        sigma_train_ann=sigma_train,
        r=CONFIG.r,
        maturities=CONFIG.maturities_years,
        moneyness=CONFIG.moneyness,
        n_sim=n_sim,
        seed=CONFIG.seed,
        sample_every_n_days=sample_every_n_days,
    )
    test_opts = build_option_dataset(
        test_df,
        split_name="test",
        sigma_train_ann=sigma_train,
        r=CONFIG.r,
        maturities=CONFIG.maturities_years,
        moneyness=CONFIG.moneyness,
        n_sim=n_sim,
        seed=CONFIG.seed,
        sample_every_n_days=sample_every_n_days,
    )

    all_opts = pd.concat([train_opts, val_opts, test_opts], ignore_index=True)

    # Step 7: Evaluation
    print("Step 7/9 - Model evaluation")
    # Here, BS is benchmark label; MC is the model output.
    mc_metrics = regression_metrics(test_opts, y_true="bs_price", y_pred="mc_price")
    print("Test metrics (MC vs BS):", mc_metrics)

    err_by_T = group_error(test_opts, group_cols=["T"], y_true="bs_price", y_pred="mc_price")

    # Save results
    results_csv = CONFIG.results_dir / "pricing_results.csv"
    test_opts.to_csv(results_csv, index=False)
    print(f"Saved: {results_csv}")

    err_csv = CONFIG.results_dir / "error_by_T.csv"
    err_by_T.to_csv(err_csv, index=False)
    print(f"Saved: {err_csv}")

    # Plots
    save_error_plots(test_opts, Path("reports/figures"))

    # Optional ML baseline
    ml_summary = None
    if run_ml:
        print("Step 7b/9 - Train ML baseline (optional)")
        from src.models.ml_baseline import train_and_predict

        # Use option dataset for supervised learning
        pred, ml_res = train_and_predict(train_opts.dropna(), test_opts.dropna())
        test_ml = test_opts.copy()
        test_ml["ml_pred"] = pred

        ml_summary = pd.DataFrame([
            {"model": ml_res.model_name, "mae": ml_res.mae, "rmse": ml_res.rmse}
        ])

        ml_summary.to_csv(CONFIG.results_dir / "ml_metrics.csv", index=False)
        test_ml.to_csv(CONFIG.results_dir / "ml_predictions.csv", index=False)
        print("Saved ML outputs: data/results/ml_metrics.csv, ml_predictions.csv")

    # Step 8: Write to DB
    print("Step 8/9 - Write results to SQLite")
    conn = connect(CONFIG.db_path)
    init_db(conn)

    # Store core tables (replace for reproducibility)
    write_table(conn, "prices_cleaned", cleaned[[c for c in cleaned.columns if c in ["date", "close", "open", "high", "low", "volume"]]])
    write_table(conn, "features", features[[c for c in features.columns if c in [
        "date","close","log_return","rv_20d_ann","rv_30d_ann","rv_60d_ann","ewma_vol_ann","ma_20","ma_20_ratio","ma_60","ma_60_ratio","year","month"
    ]]])

    # Replace options_dataset table
    write_table(conn, "options_dataset", all_opts)

    if ml_summary is not None:
        append_table(conn, "ml_results", ml_summary.rename(columns={"model": "model_name"}))

    conn.close()
    print(f"Saved SQLite DB: {CONFIG.db_path}")

    # Step 9: SQL evaluation (queries provided in sql/)
    print("Step 9/9 - SQL evaluation")
    print("Open sql/evaluation_queries.sql for example JOIN/FILTER/AGG queries.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Option pricing DS workflow pipeline")
    parser.add_argument("--recompute", action="store_true", help="Recompute outputs from scratch")
    parser.add_argument("--n-sim", type=int, default=CONFIG.n_sim)
    parser.add_argument("--sample-every", type=int, default=CONFIG.sample_every_n_days, help="Use every N-th day to build option rows")
    parser.add_argument("--run-ml", action="store_true", help="Run optional ML baseline (requires scikit-learn)")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML baseline")

    args = parser.parse_args()

    run_ml = None
    if args.run_ml:
        run_ml = True
    if args.no_ml:
        run_ml = False

    run_pipeline(
        recompute=args.recompute,
        n_sim=args.n_sim,
        sample_every_n_days=args.sample_every,
        run_ml=run_ml,
    )


if __name__ == "__main__":
    main()
