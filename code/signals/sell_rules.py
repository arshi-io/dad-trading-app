"""The two kinds of selling: getting out of what you own, and short-selling what's breaking down.

Pure functions over daily OHLCV; the nightly pipeline calls them, the dashboard shows the result.
Informational only -- nothing here places an order.

Exits are Minervini's sell rules. Shorts are the mirror image of the buy setup (a Stage-4
downtrend breaking below support) and are limited to F&O stocks: in the cash market a short must
be closed the same day, so an overnight short has to go through futures.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import pandas as pd

ATR_WINDOW = 14
ATR_STOP_MULTIPLIER = 1.5
TARGET_R = 2.0
MAX_STOP_PCT = 8.0

HEAVY_VOLUME = 1.5          # × 50-day average
CLIMAX_RUN_PCT = 25.0       # up this much in CLIMAX_BARS sessions = take profits into strength
CLIMAX_BARS = 15

SHORT_RS_MAX = 30           # weakest 30% of the market
SUPPORT_BARS = 20           # support = lowest low of the prior 20 sessions
NEAR_SUPPORT_PCT = 3.0      # within this much above support: set a sell-stop under it
SHORT_RANGE_PCT = 3.0       # broke support by more than this = extended down, don't chase


def _atr(df: pd.DataFrame) -> Optional[float]:
    high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean().iloc[-1]
    return None if pd.isna(atr) else float(atr)


def sell_check(df: pd.DataFrame, stop: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """Exit verdict for a stock you hold, as of the last bar. Always names a price level: the close
    below which you sell (`sell_below`)."""
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < 60:
        return None
    close, vol = df["close"].astype(float), df["volume"].astype(float)
    last = float(close.iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    ema21 = float(close.ewm(span=21, adjust=False).mean().iloc[-1])
    avg_vol = float(vol.iloc[-51:-1].mean())
    vol_x = float(vol.iloc[-1]) / avg_vol if avg_vol else 0.0
    run = (last / float(close.iloc[-CLIMAX_BARS - 1]) - 1) * 100
    sell_below = round(max(stop or 0.0, sma50), 2)
    base = {"last": round(last, 2), "sma50": round(sma50, 2), "ema21": round(ema21, 2), "sell_below": sell_below}

    if stop is not None and last <= stop:
        return {**base, "action": "SELL", "cls": "bad", "reason": f"Closed at ₹{last:,.2f}, below the stop ₹{stop:,.2f}. Sell at the open."}
    if last < sma50 and vol_x >= HEAVY_VOLUME:
        return {**base, "action": "SELL", "cls": "bad",
                "reason": f"Broke the 50-day average ₹{sma50:,.2f} on {vol_x:.1f}× volume: big funds are selling. Sell at the open."}
    if run >= CLIMAX_RUN_PCT:
        return {**base, "action": "SELL HALF", "cls": "good", "sell_below": round(ema21, 2),
                "reason": f"Up {run:.0f}% in {CLIMAX_BARS} sessions: a climax run. Sell half into strength; sell the rest on a close below ₹{ema21:,.2f} (21-day)."}
    if last < sma50:
        return {**base, "action": "WEAK", "cls": "wait",
                "reason": f"Below the 50-day average ₹{sma50:,.2f}, on light volume. Sell if it doesn't get back above it within 2–3 sessions."}
    if last < ema21:
        return {**base, "action": "TIGHTEN", "cls": "wait",
                "reason": f"Slipped under the 21-day ₹{ema21:,.2f}. Hold, but sell on a close below ₹{sell_below:,.2f}."}
    return {**base, "action": "HOLD", "cls": "neutral", "reason": f"Trend intact. Sell on a close below ₹{sell_below:,.2f}."}


def stage4(df: pd.DataFrame, rs_rank: Optional[int]) -> bool:
    """Mirror of the trend template: price under falling averages, near its lows, weak RS."""
    close = df["close"].dropna().astype(float)
    if len(close) < 221 or rs_rank is None or rs_rank > SHORT_RS_MAX:
        return False
    last = float(close.iloc[-1])
    sma50, sma150, sma200 = (float(close.rolling(n).mean().iloc[-1]) for n in (50, 150, 200))
    sma200_month_ago = float(close.rolling(200).mean().iloc[-22])
    year = df.iloc[-250:]
    low52, high52 = float(year["low"].min()), float(year["high"].max())
    return (last < sma50 < sma150 < sma200 and sma200 < sma200_month_ago
            and last <= low52 * 1.25 and last <= high52 * 0.75)


def breakdown_setup(df: pd.DataFrame) -> Dict[str, Any]:
    """Where price is against support (lowest low of the prior SUPPORT_BARS sessions)."""
    last = float(df["close"].iloc[-1])
    support = float(df["low"].iloc[-SUPPORT_BARS - 1:-1].min())
    pct = (last / support - 1) * 100
    if pct > NEAR_SUPPORT_PCT:
        status = "NO SETUP"
    elif pct >= 0:
        status = "NEAR BREAKDOWN"
    elif pct >= -SHORT_RANGE_PCT:
        status = "IN SHORT RANGE"
    else:
        status = "EXTENDED DOWN"
    return {"status": status, "support": round(support, 2), "pct_vs_support": round(pct, 1)}


def short_plan(df: pd.DataFrame, lot: int, risk_rupees: float) -> Optional[Dict[str, Any]]:
    """Entry, stop, target and futures lots for a short. Entry is a sell-stop just under support
    when price is still above it, else the last close."""
    setup = breakdown_setup(df)
    atr = _atr(df)
    if setup["status"] not in ("NEAR BREAKDOWN", "IN SHORT RANGE") or atr is None:
        return None
    last = float(df["close"].iloc[-1])
    order = "STOP" if setup["status"] == "NEAR BREAKDOWN" else "NOW"
    entry = setup["support"] * 0.999 if order == "STOP" else last
    stop = entry + ATR_STOP_MULTIPLIER * atr
    risk_per_share = stop - entry
    target = entry - TARGET_R * risk_per_share
    risk_per_lot = risk_per_share * lot
    return {
        **setup, "order": order, "last": round(last, 2), "atr": round(atr, 2),
        "entry": round(entry, 2), "stop": round(stop, 2), "target": round(target, 2),
        "stop_pct": round(risk_per_share / entry * 100, 1), "wide_stop": risk_per_share / entry * 100 > MAX_STOP_PCT,
        "avg_vol50": round(float(df["volume"].iloc[-51:-1].mean())),
        "lot": lot, "lots": int(risk_rupees // risk_per_lot) if risk_per_lot > 0 else 0,
        "risk_per_lot": round(risk_per_lot), "notional_per_lot": round(entry * lot),
    }
