"""
Pairs-trading signal generator.

Applies the same z-score logic used in ``mean_rev.py``, but to a **spread**
between two cointegrated assets rather than a single price. Cointegration
is tested with the Engle-Granger two-step procedure
(``statsmodels.tsa.stattools.coint``); when two price series are
cointegrated, a linear combination of them is stationary by construction,
so the z-score of that spread is a legitimate mean-reversion signal — the
very setting the v2 ADF gate on single equities rejected.

Signal semantics
----------------
For a pair ``(A, B)`` with hedge ratio :math:`h`, we define

    spread_t = P_a(t) - h * P_b(t)

and the rolling z-score of ``spread_t`` drives the signal:

  * ``z < -entry_z`` → spread is unusually **low** → long A, short B
    (emit ``BUY`` at the pair level; ``leg_a=BUY, leg_b=SELL`` in metadata).
  * ``z > +entry_z`` → spread is unusually **high** → short A, long B
    (emit ``SELL``; ``leg_a=SELL, leg_b=BUY``).
  * ``|z|`` returns below ``exit_z`` → close both legs (``HOLD``).
"""
from __future__ import annotations

import logging
import math
import os
import sys
from itertools import combinations
from typing import Iterable, List

import numpy as np
import pandas as pd

if __name__ == "__main__" and __package__ is None:
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from code.data.cleaners.ohlcv import clean_ohlcv
from code.data.fetchers.equity import fetch_ohlcv
from code.signals.base import Signal
from code.signals.mean_rev import compute_zscore

logger = logging.getLogger(__name__)

STRATEGY_NAME = "pairs_trading_zscore"
CONFIDENCE_CAP = 3.0


def _ols_slope(y: pd.Series, x: pd.Series) -> float:
    """Return the OLS slope of ``y`` on ``x`` with an intercept.

    Used for the hedge ratio (regress price_a on price_b → slope = hedge).
    """
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 3:
        return float("nan")
    X = np.column_stack([np.ones(len(df)), df.iloc[:, 1].values])
    yv = df.iloc[:, 0].values
    # lstsq is robust to collinearity and NaNs-already-dropped data.
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    return float(coef[1])


def compute_spread(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float,
) -> pd.Series:
    """Return ``price_a - hedge_ratio * price_b`` as a Series named ``spread``."""
    a, b = price_a.align(price_b, join="inner")
    s = a.astype(float) - float(hedge_ratio) * b.astype(float)
    s.name = "spread"
    return s


def compute_half_life(spread: pd.Series) -> float:
    """Half-life of mean reversion for a spread series.

    Fits an AR(1) with intercept: ``spread_t = c + phi * spread_{t-1} + e_t``
    and returns ``-log(2) / log(phi)`` — the number of days for the spread
    to cover half the distance back to its long-run mean.

    Returns ``+inf`` when the fit indicates no reversion (``phi >= 1`` or
    ``phi <= 0``) and ``NaN`` for degenerate input.
    """
    s = spread.dropna()
    if len(s) < 5:
        return float("nan")

    lag = s.shift(1).dropna()
    cur = s.loc[lag.index]
    X = np.column_stack([np.ones(len(lag)), lag.values])
    try:
        coef, *_ = np.linalg.lstsq(X, cur.values, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan")
    phi = float(coef[1])

    if phi <= 0 or phi >= 1:
        return float("inf")
    return float(-math.log(2) / math.log(phi))


def find_cointegrated_pairs(
    tickers: Iterable[str],
    start: str,
    end: str,
    pvalue_threshold: float = 0.05,
) -> List[dict]:
    """Find Engle-Granger-cointegrated pairs inside ``tickers``.

    For every unordered pair ``(A, B)`` with ``p < pvalue_threshold``,
    return a dict with the hedge ratio (OLS slope of A on B), the spread's
    half-life, and the test p-value.

    Parameters
    ----------
    tickers : iterable of str
        Ticker universe (yfinance symbols).
    start, end : str
        Date range for the fetch.
    pvalue_threshold : float, default 0.05
        Significance level for the cointegration test.

    Returns
    -------
    list[dict]
        Each dict has keys
        ``{ticker_a, ticker_b, pvalue, hedge_ratio, half_life}``,
        sorted by ``pvalue`` ascending (most significant first).
    """
    from statsmodels.tsa.stattools import coint

    tickers = list(tickers)

    # Fetch once, align on the common date index.
    frames = {}
    for t in tickers:
        raw = fetch_ohlcv(t, start, end)
        df = clean_ohlcv(raw)
        if df.empty:
            logger.warning("find_cointegrated_pairs: no data for %s — skipping", t)
            continue
        frames[t] = df["close"].astype(float).rename(t)

    if len(frames) < 2:
        logger.warning("find_cointegrated_pairs: fewer than 2 usable tickers")
        return []

    prices = pd.concat(frames.values(), axis=1, join="inner").dropna()
    logger.info("find_cointegrated_pairs: aligned panel %s", prices.shape)

    results: List[dict] = []
    for a, b in combinations(prices.columns, 2):
        pa, pb = prices[a], prices[b]
        try:
            t_stat, pvalue, _crit = coint(pa, pb)
        except Exception as exc:
            logger.warning("coint(%s,%s) failed: %s", a, b, exc)
            continue

        if pvalue >= pvalue_threshold:
            continue

        hedge = _ols_slope(pa, pb)
        if not math.isfinite(hedge):
            continue
        spread = compute_spread(pa, pb, hedge)
        hl = compute_half_life(spread)

        results.append({
            "ticker_a": a,
            "ticker_b": b,
            "pvalue": float(pvalue),
            "hedge_ratio": float(hedge),
            "half_life": float(hl),
        })

    results.sort(key=lambda d: d["pvalue"])
    logger.info("find_cointegrated_pairs: %d cointegrated pairs at p<%.2f",
                len(results), pvalue_threshold)
    return results


def _classify(z: float, entry_z: float, exit_z: float) -> str:
    if z <= -entry_z:
        return "BUY"     # spread too low → long A, short B
    if z >= entry_z:
        return "SELL"    # spread too high → short A, long B
    if abs(z) < exit_z:
        return "HOLD"    # exit both legs
    return "NEUTRAL"


def generate_signals(
    price_a: pd.Series,
    price_b: pd.Series,
    hedge_ratio: float,
    window: int = 60,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
    ticker_a: str = "A",
    ticker_b: str = "B",
) -> List[Signal]:
    """Generate pair-level mean-reversion signals from two aligned price series.

    A signal is emitted only when the spread z-score **crosses** a
    threshold — not on every bar inside the zone — so a pair is entered
    once and held until a new state is reached.

    Returns ``[]`` if the aligned series are shorter than ``window``.
    """
    a, b = price_a.align(price_b, join="inner")
    if len(a) < window:
        logger.info("generate_signals(%s/%s): not enough aligned data (%d<%d)",
                    ticker_a, ticker_b, len(a), window)
        return []

    spread = compute_spread(a, b, hedge_ratio)
    z_series = compute_zscore(spread, window)
    half_life = compute_half_life(spread)
    asset_label = f"{ticker_a}/{ticker_b}"

    signals: List[Signal] = []
    prev_state = "HOLD"

    for ts, z in z_series.items():
        if pd.isna(z):
            continue
        target = _classify(z, entry_z, exit_z)
        if target == "NEUTRAL":
            continue
        if target == prev_state:
            continue

        if target == "BUY":
            leg_a, leg_b = "BUY", "SELL"
        elif target == "SELL":
            leg_a, leg_b = "SELL", "BUY"
        else:  # HOLD → exit both
            leg_a, leg_b = "HOLD", "HOLD"

        confidence = min(abs(float(z)) / CONFIDENCE_CAP, 1.0)
        metadata = {
            "zscore": float(z),
            "hedge_ratio": float(hedge_ratio),
            "half_life": float(half_life),
            "spread_value": float(spread.loc[ts]),
            "leg_a": leg_a,
            "leg_b": leg_b,
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "window": int(window),
        }
        signals.append(
            Signal(
                asset=asset_label,
                direction=target,
                confidence=confidence,
                strategy=STRATEGY_NAME,
                timeframe="1d",
                timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
                metadata=metadata,
            )
        )
        prev_state = target

    logger.info("generate_signals(%s): %d signals", asset_label, len(signals))
    return signals
