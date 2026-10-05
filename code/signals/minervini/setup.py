"""Where a trend-template stock sits in its base: the entry a Minervini trader would use.

Rules (daily bars, algorithm-detected -- verify on the chart):
- Base: bars since the highest high of the last ~13 weeks, excluding today. Needs >= 10
  sessions and a depth <= 35% to count.
- Pivot: that highest high. Buy point = a move through it.
- IN BUY RANGE: closed above the pivot but no more than 5% past it.
- EXTENDED: more than 5% past the pivot, or no base and > 10% above the 50-day average.
- VCP: the base split into three parts shows shrinking ranges and drying-up volume.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

BASE_LOOKBACK = 65
MIN_BASE_BARS = 10
MAX_BASE_DEPTH = 0.35
BUY_RANGE = 0.05
EXTENDED_FROM_50 = 0.10


def stock_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """Day change, distance from 52-week high/low and the 50-day average, volume vs normal."""
    close = df["close"].astype(float)
    vol = df["volume"].astype(float)
    last = float(close.iloc[-1])
    prev = float(close.iloc[-2]) if len(close) > 1 else last
    year = df.iloc[-250:]
    high52, low52 = float(year["high"].max()), float(year["low"].min())
    sma50 = float(close.rolling(50).mean().iloc[-1]) if len(close) >= 50 else None
    sma200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else None
    avg_vol50 = float(vol.iloc[-51:-1].mean()) if len(vol) > 51 else None
    return {
        "close": round(last, 2),
        "chg_pct": round((last / prev - 1) * 100, 2) if prev else None,
        "high52": round(high52, 2),
        "pct_from_high": round((last / high52 - 1) * 100, 1) if high52 else None,
        "at_high": last >= high52 * 0.995,
        "at_low": last <= low52 * 1.005,
        "pct_above_50": round((last / sma50 - 1) * 100, 1) if sma50 else None,
        "above_200": (last > sma200) if sma200 else None,
        "vol_ratio": round(float(vol.iloc[-1]) / avg_vol50, 2) if avg_vol50 else None,
    }


def detect_setup(df: pd.DataFrame) -> Dict[str, Any]:
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < 60:
        return {"status": "NO DATA", "pivot": None, "vcp": False}
    close = float(df["close"].iloc[-1])
    sma50 = float(df["close"].rolling(50).mean().iloc[-1])

    # Most recent valid base, ending 1..10 sessions ago -- a breakout from a few days back
    # makes the breakout bars the new high, so the base sits just before them.
    for skip in range(1, 11):
        prior = df.iloc[-BASE_LOOKBACK - skip:-skip]
        peak_pos = int(np.argmax(prior["high"].to_numpy()))
        pivot = float(prior["high"].iloc[peak_pos])
        base = prior.iloc[peak_pos:]
        base_bars = len(base)
        depth = (pivot - float(base["low"].min())) / pivot if pivot else 0.0
        has_base = base_bars >= MIN_BASE_BARS and depth <= MAX_BASE_DEPTH
        if has_base:
            break

    vcp = False
    contractions = []
    if base_bars >= 12:
        cuts = np.linspace(0, base_bars, 4).astype(int)
        parts = [base.iloc[cuts[k]:cuts[k + 1]] for k in range(3)]
        contractions = [round((float(p["high"].max()) - float(p["low"].min())) / float(p["high"].max()) * 100, 1) for p in parts]
        vol_dry = float(parts[2]["volume"].mean()) < float(df["volume"].iloc[-51:-1].mean())
        vcp = contractions[0] > contractions[1] > contractions[2] and vol_dry

    past = close / pivot - 1
    if close > pivot and past <= BUY_RANGE and has_base:
        status = "IN BUY RANGE"
    elif close > pivot and has_base:
        status = "EXTENDED"
    elif has_base:
        status = "BASING"
    elif (close / sma50 - 1) > EXTENDED_FROM_50:
        status = "EXTENDED"
    else:
        status = "NO BASE YET"

    if not has_base:
        return {"status": status, "pivot": None, "vcp": False, "base_bars": 0, "contractions": []}
    return {
        "status": status,
        "pivot": round(pivot, 2),
        "pct_to_pivot": round((pivot / close - 1) * 100, 1),
        "pct_past_pivot": round(past * 100, 1),
        "base_bars": base_bars,
        "base_depth_pct": round(depth * 100, 1),
        "contractions": contractions,
        "vcp": bool(vcp),
    }
