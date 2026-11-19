"""Kalman-filter hedge ratio for a pair -- a time-varying alternative to the static OLS slope.

State theta_t = [beta_t, alpha_t] follows a random walk: theta_t = theta_{t-1} + w_t, w ~ N(0, Vw)
with Vw = delta / (1 - delta) * diag(1, ALPHA_RATIO). Observation: y_t = beta_t * x_t + alpha_t + v_t,
v ~ N(0, Ve).

Two choices that matter (tested on synthetic data and ICICIBANK/SBIN):
- alpha drifts 100x slower than beta. With equal drift, beta and alpha are nearly interchangeable
  over the filter's short memory (x barely moves), so beta wandered (0.56 for a true 1.0) while
  alpha absorbed the difference.
- Ve is learned from the filter's own past forecast errors (an EWMA of e^2 - x P x'), starting at
  VE_INIT. A fixed Ve made the z-score's spread depend on how noisy each pair happens to be.

Each day, before seeing y_t, the filter forecasts y_hat = x_t . theta_{t-1} with variance
Q_t = x P x' + Ve. The forecast error e_t = y_t - y_hat is the spread, and z_t = e_t / sqrt(Q_t)
is how unusual today's spread is given everything known yesterday. Then theta is updated with
y_t. Every output at t uses data up to t only.

Prices are divided by their first value before filtering so delta and Ve mean the same thing for
a Rs 100 and a Rs 10,000 stock; beta, alpha and the spread are converted back to rupee units.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

KALMAN_DELTA = 1e-4        # how fast beta may drift; bigger = faster tracking, noisier beta
KALMAN_ALPHA_RATIO = 1e-2  # alpha's drift relative to beta's
KALMAN_VE_INIT = 1e-3      # starting observation-noise variance on the scaled (first value = 1) prices
KALMAN_VE_LAMBDA = 0.02    # EWMA weight for learning Ve from past forecast errors (~50-day memory)


def kalman_filter(y: np.ndarray, x: np.ndarray, delta: float = KALMAN_DELTA, ve: float = KALMAN_VE_INIT,
                  alpha_ratio: float = KALMAN_ALPHA_RATIO, ve_lambda: float = KALMAN_VE_LAMBDA) -> dict:
    """Run the filter over aligned 1-D arrays. Returns arrays beta, alpha, error, q (forecast variance)."""
    n = len(y)
    vw = delta / (1 - delta) * np.diag([1.0, alpha_ratio])
    theta = np.zeros(2)
    r = np.zeros((2, 2))          # state covariance after the last update
    out = {k: np.full(n, np.nan) for k in ("beta", "alpha", "error", "q")}
    for t in range(n):
        h = np.array([x[t], 1.0])
        p = r + vw                 # predict: random walk keeps theta, widens covariance
        hph = h @ p @ h
        q = hph + ve
        e = y[t] - h @ theta       # forecast error, uses theta from t-1 only
        k = p @ h / q
        theta = theta + k * e
        r = p - np.outer(k, h) @ p
        out["beta"][t], out["alpha"][t], out["error"][t], out["q"][t] = theta[0], theta[1], e, q
        ve = max((1 - ve_lambda) * ve + ve_lambda * (e * e - hph), 1e-10)  # learned after z_t is computed
    return out


def kalman_pair(price_a: pd.Series, price_b: pd.Series, delta: float = KALMAN_DELTA, ve: float = KALMAN_VE_INIT) -> pd.DataFrame:
    """Daily beta (shares of B per share of A), alpha, spread (forecast error, rupees) and z-score
    for A regressed on B. Same orientation as the OLS hedge ratio in pairs_trading.py."""
    a, b = price_a.astype(float).align(price_b.astype(float), join="inner")
    keep = a.notna() & b.notna()
    a, b = a[keep], b[keep]
    if len(a) < 2:
        return pd.DataFrame(columns=["beta", "alpha", "spread", "zscore"])
    a0, b0 = float(a.iloc[0]), float(b.iloc[0])
    f = kalman_filter(a.to_numpy() / a0, b.to_numpy() / b0, delta, ve)
    return pd.DataFrame({
        "beta": f["beta"] * a0 / b0,
        "alpha": f["alpha"] * a0,
        "spread": f["error"] * a0,
        "zscore": f["error"] / np.sqrt(f["q"]),
    }, index=a.index)
