"""
OHLCV cleaning utilities.

Small, composable functions that take a DataFrame of daily bars (as produced
by ``code.data.fetchers.equity.fetch_ohlcv``) and return a sanitised
equivalent. Kept intentionally minimal — no forward-looking operations, no
interpolation beyond short forward-fills.

Usage
-----
>>> from code.data.cleaners.ohlcv import clean_ohlcv, compute_log_returns
>>> clean = clean_ohlcv(raw)
>>> r = compute_log_returns(clean)
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_log_returns(df: pd.DataFrame) -> pd.Series:
    """Compute daily log returns from a DataFrame with a ``close`` column.

    Formula: :math:`r_t = \\ln(P_t / P_{t-1})`.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``close`` column.

    Returns
    -------
    pd.Series
        Series named ``log_return`` indexed identically to ``df`` but with
        the first row (NaN) dropped.
    """
    if "close" not in df.columns:
        raise KeyError("compute_log_returns requires a 'close' column")

    close = df["close"].astype(float)
    log_r = np.log(close / close.shift(1))
    log_r.name = "log_return"
    return log_r.dropna()


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of an OHLCV DataFrame.

    Steps:
      1. Drop rows where ``close`` is missing or zero (corrupt bars).
      2. Forward-fill short gaps (up to 2 days) to bridge Indian market
         holidays without fabricating long stretches of synthetic data.
      3. Keep the original ``DatetimeIndex`` and column order.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV frame as returned by ``fetch_ohlcv``.

    Returns
    -------
    pd.DataFrame
        Cleaned copy. Empty input yields an empty output.
    """
    if df.empty:
        return df.copy()

    out = df.copy()

    before = len(out)
    out = out[(out["close"].notna()) & (out["close"] > 0)]
    dropped = before - len(out)
    if dropped:
        logger.info("clean_ohlcv dropped %d rows with NaN or zero close", dropped)

    # Short-gap forward fill: limit=2 means at most two consecutive NaNs are filled.
    out = out.ffill(limit=2)

    # A final pass: anything still NaN (e.g. leading NaNs or gaps > 2 days) is removed.
    still_na = out.isna().any(axis=1).sum()
    if still_na:
        logger.info("clean_ohlcv dropping %d rows still NaN after ffill(limit=2)", still_na)
        out = out.dropna()

    return out
