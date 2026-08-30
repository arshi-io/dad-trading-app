"""RS Rank — weighted relative-strength percentile vs the NIFTY 500 universe.

No PRD exists in this repo specifying the exact weights; P2.md only says
"weighted 3/6/9/12-mo return percentile vs NIFTY500, 0-99". This uses the
standard IBD-style weighting (0.4/0.2/0.2/0.2), approved by the user
2026-07-22.

    RS_raw   = 0.4*r3m + 0.2*r6m + 0.2*r9m + 0.2*r12m
    RS_rank  = percentile_rank(RS_raw across universe) * 99, rounded to int

Percentile ranking requires the whole universe's closes as of the SAME
as-of date -- callers must slice every series to the same cutoff before
calling ``compute_rs_ranks``.
"""
from __future__ import annotations

from typing import Dict, Optional

import pandas as pd

TRADING_DAYS_PER_MONTH = 21
WEIGHTS = {3: 0.4, 6: 0.2, 9: 0.2, 12: 0.2}


def _period_return(close: pd.Series, months: int) -> Optional[float]:
    """Simple return over ``months`` calendar months (~21 trading days/mo)."""
    n = months * TRADING_DAYS_PER_MONTH
    if len(close) <= n:
        return None
    end = close.iloc[-1]
    start = close.iloc[-1 - n]
    if pd.isna(start) or pd.isna(end) or start == 0:
        return None
    return float(end / start - 1.0)


def compute_rs_raw(close: pd.Series) -> Optional[float]:
    """Weighted 3/6/9/12-month return. None if any period lacks history."""
    close = close.astype(float).dropna()
    returns = {months: _period_return(close, months) for months in WEIGHTS}
    if any(value is None for value in returns.values()):
        return None
    return sum(WEIGHTS[months] * returns[months] for months in WEIGHTS)


def compute_rs_ranks(closes_by_ticker: Dict[str, pd.Series]) -> Dict[str, Optional[int]]:
    """Percentile-rank RS_raw across the given universe, scaled 0-99.

    Parameters
    ----------
    closes_by_ticker : dict[str, pd.Series]
        Ticker -> close series, each already sliced to the same as-of date.

    Returns
    -------
    dict[str, int | None]: 0-99 rank per ticker; None where history was
    insufficient to compute RS_raw.
    """
    raw_scores: Dict[str, float] = {}
    for ticker, close in closes_by_ticker.items():
        raw = compute_rs_raw(close)
        if raw is not None:
            raw_scores[ticker] = raw

    result: Dict[str, Optional[int]] = {ticker: None for ticker in closes_by_ticker}
    if not raw_scores:
        return result

    series = pd.Series(raw_scores)
    percentiles = series.rank(pct=True, method="average")
    scaled = (percentiles * 99).round().astype(int)

    for ticker, value in scaled.items():
        result[ticker] = int(value)
    return result
