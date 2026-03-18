from __future__ import annotations

import math
import numpy as np


def mc_call_price(S0: float, K: float, T: float, r: float, sigma: float, n_sim: int, seed: int) -> float:
    """Monte Carlo European call under risk-neutral GBM."""
    if T <= 0 or sigma <= 0 or S0 <= 0 or K <= 0 or n_sim <= 0:
        return float("nan")

    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_sim)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * math.sqrt(T) * z)
    payoff = np.maximum(ST - K, 0.0)
    return float(math.exp(-r * T) * payoff.mean())
