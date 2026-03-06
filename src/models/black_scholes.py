from __future__ import annotations

import math
import numpy as np


def _norm_cdf(x: np.ndarray | float) -> np.ndarray | float:
    """Standard normal CDF without SciPy."""
    if isinstance(x, float):
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def bs_call_price(S0: float, K: float, T: float, r: float, sigma: float) -> float:
    """Black–Scholes European call."""
    if T <= 0 or sigma <= 0 or S0 <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return float(S0 * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2))
