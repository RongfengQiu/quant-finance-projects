from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectConfig:
    """Central configuration for the project.

    Keep this file small and boring. Recruiters like boring config 🙂
    """

    # Dataset
    raw_dir: Path = Path("data/raw")
    dataset_filename: str | None = None  # if None, auto-detect a single CSV in data/raw

    # Column inference (we auto-detect, but these provide defaults)
    date_candidates: tuple[str, ...] = ("Date", "date", "timestamp", "time")
    close_candidates: tuple[str, ...] = ("Close", "close", "Adj Close", "AdjClose", "adj_close", "price", "Price")

    # Output folders
    processed_dir: Path = Path("data/processed")
    splits_dir: Path = Path("data/splits")
    results_dir: Path = Path("data/results")
    db_path: Path = Path("database/option_pricing.db")

    # Time split (chronological) – default matches a realistic DS workflow
    # Train: <= 2014-12-31, Val: 2015-01-01..2019-12-31, Test: >= 2020-01-01
    train_end: str = "2014-12-31"
    val_end: str = "2019-12-31"

    # Option dataset construction
    r: float = 0.03
    maturities_years: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0)
    moneyness: tuple[float, ...] = (0.8, 0.9, 1.0, 1.1, 1.2)

    # Sampling (to keep runtime reasonable)
    sample_every_n_days: int = 20  # use every N-th row to build option rows

    # Monte Carlo
    n_sim: int = 20_000
    seed: int = 42

    # Feature engineering
    trading_days: int = 252
    rolling_windows: tuple[int, ...] = (20, 30, 60)

    # Optional ML baseline
    run_ml_baseline: bool = False


CONFIG = ProjectConfig()
