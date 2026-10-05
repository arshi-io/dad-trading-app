"""Market timing the O'Neil / Minervini way: distribution days, follow-through, breadth, sector leaders."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

DIST_WINDOW = 25
DIST_DROP = -0.002
FTD_GAIN = 0.0125   # Indian indices move less than the US; 1.25% is the common local threshold
FTD_MIN_DAY = 4
RALLY_LOOKBACK = 60


def index_health(nifty: pd.DataFrame) -> Dict[str, Any]:
    df = nifty.dropna(subset=["close"]).copy()
    close = df["close"].astype(float)
    vol = df["volume"].astype(float) if "volume" in df else pd.Series(0.0, index=df.index)
    ret = close.pct_change()
    last = float(close.iloc[-1])
    high52 = float(close.iloc[-250:].max())

    recent = df.index[-DIST_WINDOW:]
    has_volume = bool((vol.loc[recent] > 0).all())
    dist_days: List[str] = []
    if has_volume:
        for ts in recent:
            pos = df.index.get_loc(ts)
            if pos and ret.iloc[pos] <= DIST_DROP and vol.iloc[pos] > vol.iloc[pos - 1]:
                dist_days.append(pd.Timestamp(ts).date().isoformat())

    return {
        "close": round(last, 2),
        "chg_pct": round(float(ret.iloc[-1]) * 100, 2),
        "pct_from_high": round((last / high52 - 1) * 100, 1),
        "above_50": bool(last > close.rolling(50).mean().iloc[-1]),
        "above_200": bool(last > close.rolling(200).mean().iloc[-1]),
        "distribution_days": len(dist_days) if has_volume else None,
        "distribution_dates": dist_days,
        "rally": _follow_through(close, vol if has_volume else None),
    }


def _follow_through(close: pd.Series, vol: Optional[pd.Series]) -> Dict[str, Any]:
    """Rally attempt from the lowest close of the last ~3 months; a follow-through day
    (day 4+, up >= 1.25% on rising volume) confirms it. Undercutting the low resets it."""
    window = close.iloc[-RALLY_LOOKBACK:]
    low_pos = int(window.values.argmin())
    low_date = pd.Timestamp(window.index[low_pos]).date().isoformat()
    since = window.iloc[low_pos:]
    day = len(since) - 1
    if day == 0:
        return {"state": "AT LOW", "low_date": low_date, "day": 0, "ftd_date": None}
    ret = close.pct_change()
    for k in range(FTD_MIN_DAY, len(since)):
        ts = since.index[k]
        pos = close.index.get_loc(ts)
        vol_up = vol is None or vol.iloc[pos] > vol.iloc[pos - 1]
        if ret.iloc[pos] >= FTD_GAIN and vol_up:
            return {"state": "CONFIRMED", "low_date": low_date, "day": day, "ftd_date": pd.Timestamp(ts).date().isoformat()}
    return {"state": "ATTEMPT", "low_date": low_date, "day": day, "ftd_date": None}


def breadth(metrics: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    chg = [m["chg_pct"] for m in metrics.values() if m.get("chg_pct") is not None]
    return {
        "advancers": sum(1 for c in chg if c > 0),
        "decliners": sum(1 for c in chg if c < 0),
        "new_highs": sum(1 for m in metrics.values() if m.get("at_high")),
        "new_lows": sum(1 for m in metrics.values() if m.get("at_low")),
    }


def sector_leaders(items: List[Dict[str, Any]], top: int = 5) -> List[Dict[str, Any]]:
    """Groups lead together: rank sectors by how many members pass the template, then median RS."""
    by_sector: Dict[str, List[Dict[str, Any]]] = {}
    for item in items:
        if item.get("sector"):
            by_sector.setdefault(item["sector"], []).append(item)
    rows = []
    for sector, members in by_sector.items():
        if len(members) < 8:  # a 1-of-4 "group" is noise, not leadership
            continue
        rs = sorted(m["rs_rank"] for m in members if m.get("rs_rank") is not None)
        passing = sum(1 for m in members if m.get("passed"))
        rows.append({
            "sector": sector,
            "members": len(members),
            "passing": passing,
            "pct_passing": round(100 * passing / len(members)),
            "median_rs": rs[len(rs) // 2] if rs else None,
            "leaders": [m["symbol"] for m in sorted(members, key=lambda m: m.get("rs_rank") or 0, reverse=True) if m.get("passed")][:3],
        })
    rows.sort(key=lambda r: (r["pct_passing"], r["median_rs"] or 0), reverse=True)
    return rows[:top]


BREADTH_OK = 0.60   # same threshold as code/research/base_rates.py


def breadth_200(items: List[Dict[str, Any]]) -> Optional[float]:
    """Share of screened stocks above their own 200-day average."""
    flags = [i["above_200"] for i in items if i.get("above_200") is not None]
    return round(sum(flags) / len(flags), 3) if flags else None


def market_grade(breadth: Optional[float], nifty_above_50: bool) -> str:
    """A: breadth ≥60% and NIFTY above its 50-day. B: one of them. C: neither.
    Backtest 2012-2026 (trailing exit): A +0.05R, B −0.06R, C −0.10R per breakout."""
    good = int(breadth is not None and breadth >= BREADTH_OK) + int(bool(nifty_above_50))
    return {2: "A", 1: "B", 0: "C"}[good]


def risk_posture(regime: str, health: Dict[str, Any], grade: str = "B") -> Dict[str, Any]:
    """How hard to press: % of account risked per trade and how many positions to hold.
    Grade C (no breadth, no trend) is defensive whatever the regime label says; full size only
    when the grade is A and the index is in a confirmed uptrend."""
    dd = health.get("distribution_days") or 0
    confirmed = health.get("rally", {}).get("state") == "CONFIRMED"
    if grade == "C" or (regime == "TRENDING_DOWN" and not confirmed):
        return {"level": "DEFENSIVE", "risk_pct": 0.25, "max_positions": 2,
                "text": "Mostly cash. Quarter size, at most 2 pilot positions, only the tightest setups."}
    if grade == "A" and regime == "TRENDING_UP" and dd < 5:
        return {"level": "PRESS", "risk_pct": 1.0, "max_positions": 8,
                "text": "Normal size, up to 8 positions."}
    return {"level": "CAUTIOUS", "risk_pct": 0.5, "max_positions": 4,
            "text": "Half size, at most 4 positions."}
