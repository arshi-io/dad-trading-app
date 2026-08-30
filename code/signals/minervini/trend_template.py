"""Minervini Trend Template — 8-point Stage 2 uptrend checklist.

No PRD exists in this repo specifying exact thresholds; these are the
standard published criteria (Minervini, "Trade Like a Stock Market Wizard").
Confirmed against papa-ux's one in-repo data point (>=30% above 52wk low)
and approved by the user 2026-07-22.

Conditions (evaluated on the last row of the supplied ``df``):
  1. close > SMA150 and close > SMA200
  2. SMA150 > SMA200
  3. SMA200 trending up (today > ~1 month / 21 trading days ago)
  4. SMA50 > SMA150 and SMA50 > SMA200
  5. close > SMA50
  6. close >= 1.30 x 52-week low
  7. close >= 0.75 x 52-week high (within 25% of the high)
  8. RS rank >= 70 (passed in by the caller; None if unavailable)
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

SMA_TRENDING_LOOKBACK_DAYS = 21
LOW_BAND_MULTIPLIER = 1.30
HIGH_BAND_MULTIPLIER = 0.75
RS_RANK_THRESHOLD = 70
TRADING_DAYS_52WK = 252


def evaluate_trend_template(df: pd.DataFrame, rs_rank: Optional[int] = None) -> Dict[str, Any]:
    """Evaluate the 8-point trend template as of the last row of ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Must have a ``close`` column, ascending by date, sliced up to the
        as-of date (the last row is treated as "today"). Needs >=252 bars
        of history for a fully meaningful result.
    rs_rank : int, optional
        Precomputed 0-99 RS rank as of the same as-of date (see rs_rank.py).
        None if not available -- condition 8 fails in that case.

    Returns
    -------
    dict with ``passed`` (bool, all 8 conditions true), ``conditions_passed``
    (int 0-8), ``checklist`` (list of per-condition dicts: condition, label,
    passed, detail), and ``insufficient_history`` (bool).
    """
    close = df["close"].astype(float).dropna()

    if len(close) < 200:
        checklist = [
            {"condition": name, "label": label, "passed": False, "detail": "insufficient history"}
            for name, label in _CONDITION_LABELS
        ]
        return {
            "passed": False,
            "conditions_passed": 0,
            "checklist": checklist,
            "insufficient_history": True,
        }

    sma50 = close.rolling(50, min_periods=50).mean()
    sma150 = close.rolling(150, min_periods=150).mean()
    sma200 = close.rolling(200, min_periods=200).mean()

    c = float(close.iloc[-1])
    s50 = _last_or_none(sma50)
    s150 = _last_or_none(sma150)
    s200 = _last_or_none(sma200)
    s200_prev = _at_or_none(sma200, -1 - SMA_TRENDING_LOOKBACK_DAYS)

    window_52wk = close.iloc[-TRADING_DAYS_52WK:] if len(close) >= TRADING_DAYS_52WK else close
    low_52wk = float(window_52wk.min())
    high_52wk = float(window_52wk.max())

    checklist: List[Dict[str, Any]] = []

    cond1 = s150 is not None and s200 is not None and c > s150 and c > s200
    checklist.append({
        "condition": "price_above_sma150_sma200",
        "label": "Price above 150-day and 200-day SMA",
        "passed": bool(cond1),
        "detail": f"close {c:.2f} vs SMA150 {_fmt(s150)} / SMA200 {_fmt(s200)}",
    })

    cond2 = s150 is not None and s200 is not None and s150 > s200
    checklist.append({
        "condition": "sma150_above_sma200",
        "label": "150-day SMA above 200-day SMA",
        "passed": bool(cond2),
        "detail": f"SMA150 {_fmt(s150)} vs SMA200 {_fmt(s200)}",
    })

    cond3 = s200 is not None and s200_prev is not None and s200 > s200_prev
    checklist.append({
        "condition": "sma200_trending_up",
        "label": "200-day SMA trending up (vs ~1 month ago)",
        "passed": bool(cond3),
        "detail": f"SMA200 now {_fmt(s200)} vs {_fmt(s200_prev)} 21d ago",
    })

    cond4 = s50 is not None and s150 is not None and s200 is not None and s50 > s150 and s50 > s200
    checklist.append({
        "condition": "sma50_above_sma150_sma200",
        "label": "50-day SMA above 150-day and 200-day SMA",
        "passed": bool(cond4),
        "detail": f"SMA50 {_fmt(s50)} vs SMA150 {_fmt(s150)} / SMA200 {_fmt(s200)}",
    })

    cond5 = s50 is not None and c > s50
    checklist.append({
        "condition": "price_above_sma50",
        "label": "Price above 50-day SMA",
        "passed": bool(cond5),
        "detail": f"close {c:.2f} vs SMA50 {_fmt(s50)}",
    })

    pct_above_low = (c / low_52wk - 1.0) * 100.0 if low_52wk else None
    cond6 = pct_above_low is not None and c >= LOW_BAND_MULTIPLIER * low_52wk
    checklist.append({
        "condition": "price_30pct_above_52wk_low",
        "label": "Price at least 30% above 52-week low",
        "passed": bool(cond6),
        "detail": f"{_fmt(pct_above_low)}% above 52wk low ({low_52wk:.2f}) — need >=30%",
    })

    pct_of_high = (c / high_52wk) * 100.0 if high_52wk else None
    cond7 = pct_of_high is not None and c >= HIGH_BAND_MULTIPLIER * high_52wk
    checklist.append({
        "condition": "price_within_25pct_of_52wk_high",
        "label": "Price within 25% of 52-week high",
        "passed": bool(cond7),
        "detail": f"{_fmt(pct_of_high)}% of 52wk high ({high_52wk:.2f}) — need >=75%",
    })

    cond8 = rs_rank is not None and rs_rank >= RS_RANK_THRESHOLD
    checklist.append({
        "condition": "rs_rank_70_plus",
        "label": "RS rank 70 or higher",
        "passed": bool(cond8),
        "detail": f"RS rank {rs_rank if rs_rank is not None else 'unavailable'} — need >=70",
    })

    conditions_passed = sum(1 for item in checklist if item["passed"])

    return {
        "passed": conditions_passed == len(checklist),
        "conditions_passed": conditions_passed,
        "checklist": checklist,
        "insufficient_history": False,
    }


_CONDITION_LABELS = [
    ("price_above_sma150_sma200", "Price above 150-day and 200-day SMA"),
    ("sma150_above_sma200", "150-day SMA above 200-day SMA"),
    ("sma200_trending_up", "200-day SMA trending up (vs ~1 month ago)"),
    ("sma50_above_sma150_sma200", "50-day SMA above 150-day and 200-day SMA"),
    ("price_above_sma50", "Price above 50-day SMA"),
    ("price_30pct_above_52wk_low", "Price at least 30% above 52-week low"),
    ("price_within_25pct_of_52wk_high", "Price within 25% of 52-week high"),
    ("rs_rank_70_plus", "RS rank 70 or higher"),
]


def _last_or_none(series: pd.Series) -> Optional[float]:
    if len(series) == 0 or pd.isna(series.iloc[-1]):
        return None
    return float(series.iloc[-1])


def _at_or_none(series: pd.Series, position: int) -> Optional[float]:
    if len(series) < abs(position) or pd.isna(series.iloc[position]):
        return None
    return float(series.iloc[position])


def _fmt(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "NA"
