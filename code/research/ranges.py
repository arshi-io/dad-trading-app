"""Prediction ranges v2: volatility-scaled, conformally calibrated.

band(t, h) = close_t × exp(± q · σ_t · √h), where
  σ_t  = blend of a fast EWMA volatility (reacts to the current regime) and the 1-year volatility
         (keeps one noisy week from dominating), and
  q    = the empirical quantile of |h-day return| / (σ · √h) over the stock's own recent past
         (split-conformal on volatility-normalised residuals): if the band has been too narrow
         lately, q grows, so the stated coverage holds across calm and rough markets.
Nothing here trades or is shown to the user until the backtest says it beats v1.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

EWMA_LAMBDA = 0.94         # RiskMetrics daily decay
LONG_WINDOW = 250
BLEND = 0.7                # weight on the fast estimate
CAL_WINDOW = 120           # days of normalised residuals used to calibrate q
ACI_GAMMA = 0.01           # adaptive-conformal step: how fast the target level reacts to misses


def sigma_series(r: pd.Series) -> pd.Series:
    """One-day volatility forecast for day t+1 made at the close of day t (no lookahead)."""
    ewma_var = (r ** 2).ewm(alpha=1 - EWMA_LAMBDA, adjust=False).mean()
    long_var = (r ** 2).rolling(LONG_WINDOW, min_periods=60).mean()
    return np.sqrt(BLEND * ewma_var + (1 - BLEND) * long_var)


def bands(close: pd.Series, horizon: int = 1, coverages=(0.68, 0.90)) -> pd.DataFrame:
    """Forecast bands made at each close for the close `horizon` sessions later."""
    r = np.log(close).diff()
    sig = sigma_series(r)
    fwd = np.log(close).shift(-horizon) - np.log(close)            # realised h-day return (target)
    z = (fwd / (sig * np.sqrt(horizon))).abs()                      # normalised residual
    # Calibrate only on residuals whose outcome was already known at time t: shift by horizon.
    known = z.shift(horizon)
    out = pd.DataFrame({"close": close, "sigma": sig, "fwd": fwd})
    for cov in coverages:
        q = _adaptive_quantiles(known.to_numpy(), z.to_numpy(), cov, horizon)
        width = q * sig.to_numpy() * np.sqrt(horizon)
        out[f"lo{int(cov*100)}"] = close * np.exp(-width)
        out[f"hi{int(cov*100)}"] = close * np.exp(width)
    return out


def _adaptive_quantiles(known: np.ndarray, z: np.ndarray, cov: float, horizon: int) -> np.ndarray:
    """Adaptive conformal inference (Gibbs & Candès 2021): the quantile level drifts up after a
    miss and down after a hit, so the long-run hit rate converges to `cov` even when the residual
    distribution shifts. Only outcomes known at time t (h sessions old) feed the update."""
    n = len(z)
    q = np.full(n, np.nan)
    level = cov
    hits: list = [None] * n
    for t in range(n):
        if t >= horizon and hits[t - horizon] is not None:          # outcome of the band made h days ago
            level = min(0.999, max(0.5, level + ACI_GAMMA * ((0 if hits[t - horizon] else 1) - (1 - cov))))
        window = known[max(0, t - CAL_WINDOW + 1):t + 1]
        window = window[~np.isnan(window)]
        if len(window) >= min(120, CAL_WINDOW):
            q[t] = np.quantile(window, level)
            if not np.isnan(z[t]):
                hits[t] = z[t] <= q[t]
    return q


def backtest_v1_vs_v2(closes: Dict[str, pd.Series], horizon: int = 1) -> pd.DataFrame:
    """Coverage and width of the shipped bootstrap band (v1) vs v2, split by volatility regime."""
    rows = []
    for sym, c in closes.items():
        c = c.dropna().astype(float)
        if len(c) < 450:
            continue
        b = bands(c, horizon)
        r = np.log(c).diff().values
        fwd = b["fwd"].values
        for t in range(300, len(c) - horizon):
            if np.isnan(b["lo68"].iat[t]) or np.isnan(fwd[t]):
                continue
            hist = r[t - 249:t + 1]
            hist = hist[~np.isnan(hist)]
            adj = hist - hist.mean() + 0.25 * hist.mean()
            # v1: bootstrap of single days scaled by sqrt(h) -- same spirit as the shipped cone
            lo1, hi1 = np.percentile(adj, [16, 84]) * np.sqrt(horizon)
            lo2, hi2 = np.log(b["lo68"].iat[t] / c.iat[t]), np.log(b["hi68"].iat[t] / c.iat[t])
            recent = np.nanstd(r[t - 19:t + 1])
            regime = "volatile" if recent > np.std(hist) * 1.2 else "calm"
            rows.append((sym, regime, lo1 <= fwd[t] <= hi1, hi1 - lo1,
                         lo2 <= fwd[t] <= hi2, hi2 - lo2,
                         np.log(b["lo90"].iat[t] / c.iat[t]) <= fwd[t] <= np.log(b["hi90"].iat[t] / c.iat[t])))
    return pd.DataFrame(rows, columns=["symbol", "regime", "v1_in", "v1_width", "v2_in", "v2_width", "v2_in90"])
