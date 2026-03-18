from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_error_plots(results: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Error by maturity
    if "T" in results.columns and "abs_error" in results.columns:
        agg = results.groupby("T", as_index=False)["abs_error"].mean()
        plt.figure(figsize=(7, 4))
        plt.plot(agg["T"], agg["abs_error"], marker="o")
        plt.xlabel("Maturity (years)")
        plt.ylabel("Mean Absolute Error")
        plt.title("Monte Carlo vs Black–Scholes: MAE by Maturity")
        plt.tight_layout()
        plt.savefig(out_dir / "mae_by_maturity.png")
        plt.close()

    # Error distribution
    if "abs_error" in results.columns:
        plt.figure(figsize=(7, 4))
        plt.hist(results["abs_error"].dropna(), bins=40)
        plt.xlabel("Absolute Error")
        plt.ylabel("Count")
        plt.title("Monte Carlo vs Black–Scholes: Error Distribution")
        plt.tight_layout()
        plt.savefig(out_dir / "error_distribution.png")
        plt.close()
