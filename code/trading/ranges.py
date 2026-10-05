"""Prediction ranges: volatility-scaled and adaptively conformal-calibrated.

Validated walk-forward on 141 NSE stocks (code/research/ranges.py): the 68% band caught 68.0% of
next-day closes and 67.8% of 5-day closes, the 90% band 89.6% / 89.2% -- in calm and volatile
markets alike (the previous bootstrap band fell to 55-58% in volatile spells). Says nothing about
direction: the band is centred on the last close.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

EWMA_LAMBDA = 0.94
LONG_WINDOW = 250
BLEND = 0.7
CAL_WINDOW = 120
ACI_GAMMA = 0.01


def _sigma(r: pd.Series) -> np.ndarray:
    ewma_var = (r ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean()
    long_var = (r ** 2).rolling(LONG_WINDOW, min_periods=60).mean()
    return np.sqrt(BLEND * ewma_var + (1 - BLEND) * long_var).to_numpy()


def latest_band(close: pd.Series, horizon: int, cov: float = 0.68) -> Dict[str, float]:
    """Band for the close `horizon` sessions after the last one: {"lo", "hi"} in rupees."""
    close = close.dropna().astype(float)
    logc = np.log(close.to_numpy())
    r = pd.Series(logc).diff()
    sig = _sigma(r)
    n = len(logc)
    fwd = np.full(n, np.nan)
    fwd[:-horizon] = logc[horizon:] - logc[:-horizon]
    z = np.abs(fwd / (sig * np.sqrt(horizon)))
    known = np.concatenate([np.full(horizon, np.nan), z[:-horizon]])

    level, q_last = cov, np.nan
    hits = [None] * n
    for t in range(n):
        if t >= horizon and hits[t - horizon] is not None:
            level = min(0.999, max(0.5, level + ACI_GAMMA * ((0 if hits[t - horizon] else 1) - (1 - cov))))
        window = known[max(0, t - CAL_WINDOW + 1):t + 1]
        window = window[~np.isnan(window)]
        if len(window) >= CAL_WINDOW:
            q_last = float(np.quantile(window, level))
            if not np.isnan(z[t]):
                hits[t] = z[t] <= q_last
    if np.isnan(q_last) or np.isnan(sig[-1]):
        raise ValueError("not enough history for a calibrated band")
    width = q_last * sig[-1] * np.sqrt(horizon)
    last = float(close.iloc[-1])
    return {"lo": last * float(np.exp(-width)), "hi": last * float(np.exp(width))}
