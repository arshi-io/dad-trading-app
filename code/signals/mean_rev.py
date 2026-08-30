"""
Mean-reversion z-score signal generator.

Implements the strategy documented in
``wiki/strategies/mean-reversion-zscore.md``. See that page for the math,
intuition, and risks. This module is the wiki page's executable form:
params, formulas, and signal contract must match.

A signal is emitted only when the z-score **crosses** a threshold, not on
every bar where the condition is merely true — this mirrors real trading
where a position is opened once and held, not rebought daily.

Usage
-----
>>> from code.signals.mean_rev import generate_signals
>>> sigs = generate_signals(df, window=60, entry_z=2.0, exit_z=0.5)
"""
from __future__ import annotations

import logging
import os
import sys
from typing import List

import numpy as np
import pandas as pd

# Allow running this file directly: add the repo root to sys.path so the
# "code.*" absolute imports below resolve whether we invoke via
# `python code/signals/mean_rev.py` or `python -m code.signals.mean_rev`.
if __name__ == "__main__" and __package__ is None:
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from code.signals.base import Signal

logger = logging.getLogger(__name__)

STRATEGY_NAME = "mean_reversion_zscore"
CONFIDENCE_CAP = 3.0  # z_cap from the wiki page


def compute_zscore(prices: pd.Series, window: int) -> pd.Series:
    """Return the rolling z-score of a price series.

    :math:`z_t = (P_t - \\mu_t) / \\sigma_t` with ``\\mu_t`` and ``\\sigma_t``
    computed over the trailing ``window`` observations.

    Parameters
    ----------
    prices : pd.Series
        Float-like price series (typically ``df['close']``).
    window : int
        Lookback length ``n``.

    Returns
    -------
    pd.Series
        Series named ``zscore``. Leading ``window-1`` values are NaN.
    """
    p = prices.astype(float)
    rolling = p.rolling(window=window, min_periods=window)
    mu = rolling.mean()
    sigma = rolling.std(ddof=1)
    z = (p - mu) / sigma
    z.name = "zscore"
    return z


def adf_is_stationary(
    prices: pd.Series,
    pvalue_threshold: float = 0.05,
) -> dict:
    """Run the Augmented Dickey-Fuller test on a price series.

    The ADF null hypothesis is: "the series has a unit root"
    (i.e. it is trending / non-stationary).

    If ``p-value < threshold``: reject the null → the series IS stationary
    → mean reversion is valid → return ``tradeable = True``.

    If ``p-value >= threshold``: fail to reject the null → the series is
    trending → mean reversion is dangerous → return ``tradeable = False``.

    Parameters
    ----------
    prices : pd.Series
        Price series. Log is taken internally for numerical stability.
    pvalue_threshold : float, default 0.05
        Significance level :math:`\\alpha`. Lower = stricter gate.

    Returns
    -------
    dict
        ``{"tradeable": bool, "pvalue": float, "adf_stat": float,
           "critical_values": {"1%": ..., "5%": ..., "10%": ...},
           "interpretation": str}``.
        On test failure (insufficient data, non-positive prices) returns
        ``tradeable=False`` and an error string in ``interpretation``.
    """
    from statsmodels.tsa.stattools import adfuller

    clean = prices.dropna()
    clean = clean[clean > 0]
    if len(clean) < 20:
        return {
            "tradeable": False,
            "pvalue": float("nan"),
            "adf_stat": float("nan"),
            "critical_values": {},
            "interpretation": f"insufficient data (n={len(clean)}) — SKIP",
        }

    try:
        result = adfuller(np.log(clean), autolag="AIC")
    except Exception as exc:
        return {
            "tradeable": False,
            "pvalue": float("nan"),
            "adf_stat": float("nan"),
            "critical_values": {},
            "interpretation": f"ADF test failed: {exc} — SKIP",
        }

    pvalue = float(result[1])
    tradeable = pvalue < pvalue_threshold

    if tradeable:
        interp = (f"p={pvalue:.4f} < {pvalue_threshold} -> "
                  f"stationary -> mean reversion valid")
    else:
        interp = (f"p={pvalue:.4f} >= {pvalue_threshold} -> "
                  f"non-stationary -> trending -> SKIP")

    return {
        "tradeable": tradeable,
        "pvalue": round(pvalue, 6),
        "adf_stat": round(float(result[0]), 4),
        "critical_values": {k: round(float(v), 4) for k, v in result[4].items()},
        "interpretation": interp,
    }


def _classify(z: float, entry_z: float, exit_z: float) -> str:
    """Map a z-score to a desired target state in {BUY, SELL, HOLD}."""
    if z <= -entry_z:
        return "BUY"
    if z >= entry_z:
        return "SELL"
    if abs(z) < exit_z:
        return "HOLD"
    return "NEUTRAL"  # in-between zone — keep whatever state we had


def generate_signals(
    df: pd.DataFrame,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    run_adf_gate: bool = True,
    adf_pvalue_threshold: float = 0.05,
) -> List[Signal]:
    """Generate mean-reversion signals from OHLCV bars.

    Only emits a ``Signal`` when the desired state *changes* (i.e. the
    z-score crosses a threshold). Returns an empty list when the input is
    empty, shorter than ``window``, or fails the ADF stationarity gate.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``close`` column and a ``DatetimeIndex``.
    window : int, default 60
        Rolling lookback length.
    entry_z : float, default 2.0
        Absolute z-score beyond which a position is opened.
    exit_z : float, default 0.5
        Absolute z-score below which the position is closed (``HOLD``).
    run_adf_gate : bool, default True
        If True, run ``adf_is_stationary`` on the close price and return
        ``[]`` when the series fails the gate. Set False to bypass (e.g.
        for forced backtests or diagnostic runs).
    adf_pvalue_threshold : float, default 0.05
        p-value below which the series is considered stationary.

    Returns
    -------
    list[Signal]
        Signals in chronological order. Empty list if the ADF gate blocks.
    """
    if df.empty or "close" not in df.columns or len(df) < window:
        logger.info("generate_signals: not enough data (len=%d, window=%d)", len(df), window)
        return []

    asset_guess = getattr(df, "name", None) or df.attrs.get("ticker", "UNKNOWN")

    if run_adf_gate:
        adf = adf_is_stationary(df["close"], pvalue_threshold=adf_pvalue_threshold)
        if not adf["tradeable"]:
            logger.info("ADF gate blocked %s: %s", asset_guess, adf["interpretation"])
            return []

    close = df["close"].astype(float)
    z_series = compute_zscore(close, window)
    mu_series = close.rolling(window=window, min_periods=window).mean()
    sigma_series = close.rolling(window=window, min_periods=window).std(ddof=1)

    signals: List[Signal] = []
    prev_state = "HOLD"  # start flat

    for ts, z in z_series.items():
        if pd.isna(z):
            continue

        target = _classify(z, entry_z, exit_z)

        # In the "NEUTRAL" band we do nothing — preserve whatever state we're in.
        if target == "NEUTRAL":
            continue

        if target == prev_state:
            continue  # no crossing → no new signal

        confidence = min(abs(float(z)) / CONFIDENCE_CAP, 1.0)
        metadata = {
            "zscore": float(z),
            "window": int(window),
            "rolling_mean": float(mu_series.loc[ts]),
            "rolling_std": float(sigma_series.loc[ts]),
        }
        signals.append(
            Signal(
                asset=asset_guess,
                direction=target,
                confidence=confidence,
                strategy=STRATEGY_NAME,
                timeframe="1d",
                timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                metadata=metadata,
            )
        )
        prev_state = target

    logger.info(
        "generate_signals(%s): %d signals (window=%d, entry_z=%.2f, exit_z=%.2f)",
        asset_guess, len(signals), window, entry_z, exit_z,
    )
    return signals


if __name__ == "__main__":
    from datetime import date, timedelta

    from code.data.cleaners.ohlcv import clean_ohlcv
    from code.data.fetchers.equity import fetch_ohlcv

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    ticker = "RELIANCE.NS"
    end = date.today()
    start = end - timedelta(days=365 * 2)

    raw = fetch_ohlcv(ticker, start.isoformat(), end.isoformat())
    df = clean_ohlcv(raw)
    df.attrs["ticker"] = ticker

    sigs = generate_signals(df, window=60, entry_z=2.0, exit_z=0.5)

    print(f"\n{ticker}: {len(sigs)} signals over {len(df)} bars "
          f"[{df.index[0].date()}..{df.index[-1].date()}]\n")
    header = f"{'date':<12} {'direction':<10} {'confidence':<10} {'z-score':<8}"
    print(header)
    print("-" * len(header))
    for s in sigs:
        print(f"{s.timestamp.date().isoformat():<12} "
              f"{s.direction:<10} "
              f"{s.confidence:<10.3f} "
              f"{s.metadata['zscore']:<8.3f}")

    buys = sum(1 for s in sigs if s.direction == "BUY")
    sells = sum(1 for s in sigs if s.direction == "SELL")
    holds = sum(1 for s in sigs if s.direction == "HOLD")
    print(f"\nTotals — BUY: {buys}  SELL: {sells}  HOLD(exit): {holds}")
