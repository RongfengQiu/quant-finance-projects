from __future__ import annotations

import math
from typing import Iterable
import pandas as pd

from src.models.black_scholes import bs_call_price
from src.models.monte_carlo import mc_call_price


def build_option_dataset(
    df: pd.DataFrame,
    split_name: str,
    sigma_train_ann: float,
    r: float,
    maturities: Iterable[float],
    moneyness: Iterable[float],
    n_sim: int,
    seed: int,
    sample_every_n_days: int,
    vol_col: str = "rv_30d_ann",
) -> pd.DataFrame:
    """Create a supervised-learning-friendly option dataset.

    Each sampled date generates a grid of options (K, T). For each option, we compute:
      - Black–Scholes price (benchmark)
      - Monte Carlo price
      - absolute error

    This yields a realistic DS table you can split by time and evaluate.
    """

    if df.empty:
        return pd.DataFrame()

    df_sorted = df.sort_values("date").reset_index(drop=True)
    df_sample = df_sorted.iloc[:: max(1, int(sample_every_n_days))].copy()

    rows: list[dict] = []
    idx = 0

    for _, rrow in df_sample.iterrows():
        date = rrow["date"]
        S = float(rrow["close"])

        sigma_used = rrow.get(vol_col)
        if sigma_used is None or (isinstance(sigma_used, float) and math.isnan(sigma_used)):
            sigma_used = sigma_train_ann
        sigma_used = float(sigma_used)

        for T in maturities:
            for m in moneyness:
                K = float(m * S)

                bs = bs_call_price(S0=S, K=K, T=float(T), r=float(r), sigma=sigma_used)
                mc = mc_call_price(S0=S, K=K, T=float(T), r=float(r), sigma=sigma_used, n_sim=int(n_sim), seed=int(seed + idx))
                abs_error = abs(mc - bs)

                rows.append(
                    {
                        "date": date,
                        "split": split_name,
                        "S": S,
                        "K": K,
                        "T": float(T),
                        "r": float(r),
                        "sigma_used": sigma_used,
                        "log_moneyness": math.log(S / K),
                        "bs_price": bs,
                        "mc_price": mc,
                        "abs_error": abs_error,
                    }
                )
                idx += 1

    return pd.DataFrame(rows)
