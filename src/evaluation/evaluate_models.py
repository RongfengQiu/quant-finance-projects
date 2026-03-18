from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


TARGET_COL = "future_20d_rv_ann"


def save_ml_plots(
    predictions_df: pd.DataFrame,
    trained_model: object | None,
    feature_cols: list[str],
    out_dir: Path,
) -> None:
    """Save a few simple plots for the selected final model."""

    out_dir.mkdir(parents=True, exist_ok=True)

    selected = predictions_df.loc[predictions_df["is_selected_model"] == 1].copy()
    if selected.empty:
        raise ValueError("No selected-model predictions found for plotting.")

    # Actual vs predicted
    plt.figure(figsize=(7, 4))
    plt.scatter(selected[TARGET_COL], selected["prediction"], alpha=0.6)
    mn = min(selected[TARGET_COL].min(), selected["prediction"].min())
    mx = max(selected[TARGET_COL].max(), selected["prediction"].max())
    plt.plot([mn, mx], [mn, mx])
    plt.xlabel("Actual future 20d realized volatility")
    plt.ylabel("Predicted future 20d realized volatility")
    plt.title(f"Actual vs Predicted ({selected['model_name'].iloc[0]})")
    plt.tight_layout()
    plt.savefig(out_dir / "actual_vs_predicted.png")
    plt.close()

    # Residual distribution
    plt.figure(figsize=(7, 4))
    plt.hist(selected["residual"].dropna(), bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title(f"Residual Distribution ({selected['model_name'].iloc[0]})")
    plt.tight_layout()
    plt.savefig(out_dir / "residual_distribution.png")
    plt.close()

    # Feature importance only when available
    if trained_model is not None and hasattr(trained_model, "feature_importances_"):
        fi = pd.DataFrame(
            {
                "feature": feature_cols,
                "importance": trained_model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        plt.figure(figsize=(8, 4.5))
        plt.bar(fi["feature"], fi["importance"])
        plt.xticks(rotation=45, ha="right")
        plt.ylabel("Importance")
        plt.title("Feature Importance (selected model)")
        plt.tight_layout()
        plt.savefig(out_dir / "feature_importance.png")
        plt.close()
