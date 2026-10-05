"""Trade plans for the setups list: entry, stop, target, size, and calibrated price ranges.

Ranges come from code/trading/ranges.py (validated walk-forward). No direction is forecast: an
honest backtest found no reliable edge in predicting which way a stock moves next. Callers pass in
OHLCV they already have -- no network here.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from code.signals.minervini.setup import detect_setup, stock_metrics
from code.trading.ranges import latest_band

logger = logging.getLogger(__name__)

HORIZON_DAYS = 5          # projected cone on the chart
ATR_WINDOW = 14
ATR_STOP_MULTIPLIER = 1.5  # locked risk rule
TARGET_R = 2.0
CHART_BARS = 90

PAPER_CAPITAL = 1_000_000
RISK_PER_TRADE = 0.01
MAX_POSITION_FRACTION = 0.20

PAIR_HARD_STOP_Z = 3.5


def _next_trading_days(start: date, n: int) -> List[date]:
    # Last bar can be days old (weekend, holiday). Before today's 15:30 close (the 08:30 run, or a
    # catch-up run after midnight) today's session is still ahead; after it, the next is tomorrow.
    now = datetime.now()
    today = now.date() - timedelta(days=1) if (now.hour, now.minute) < (15, 30) else now.date()
    days, d = [], max(start, today)
    while len(days) < n:
        d += timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    return days


def _unix(d: Any) -> int:
    return int(pd.Timestamp(d).normalize().timestamp())


def _atr(df: pd.DataFrame) -> Optional[float]:
    high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev = close.shift(1)
    tr = pd.concat([high - low, (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean().iloc[-1]
    return None if pd.isna(atr) else float(atr)


def suggested_qty(entry: float, stop: float, risk_pct: float = RISK_PER_TRADE * 100) -> int:
    risk_per_share = abs(entry - stop)
    if risk_per_share <= 0 or entry <= 0:
        return 0
    by_risk = (PAPER_CAPITAL * risk_pct / 100) / risk_per_share
    by_size = (PAPER_CAPITAL * MAX_POSITION_FRACTION) / entry
    return max(1, int(min(by_risk, by_size)))


MAX_STOP_PCT = 8.0          # Minervini's ceiling on an initial stop
RESULTS_HOLD_DAYS = 28      # a typical swing hold; results inside it = gap risk
NEAR_PIVOT_PCT = 5.0        # a breakout order further away than this is a hope, not a setup


def _results_fields(results_date: Optional[str]) -> Dict[str, Any]:
    if not results_date:
        return {"results_date": None, "results_days": None, "results_in_hold": False}
    days = (date.fromisoformat(results_date) - date.today()).days
    return {"results_date": results_date, "results_days": days, "results_in_hold": 0 <= days <= RESULTS_HOLD_DAYS}


def predict_stock(
    symbol: str, df: pd.DataFrame, reason: str = "",
    risk_pct: float = RISK_PER_TRADE * 100, results_date: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < 60:
        return None
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    atr = _atr(df)
    if atr is None:
        return None
    last_day = pd.Timestamp(df.index[-1]).date()
    setup = detect_setup(df)
    metrics = stock_metrics(df)

    # BASING: a buy-stop just through the pivot -- the trade only exists if the breakout happens.
    order = "STOP" if setup["status"] == "BASING" else "NOW"
    entry = round(setup["pivot"] * 1.001, 2) if order == "STOP" else last

    stop = entry - ATR_STOP_MULTIPLIER * atr
    target = entry + TARGET_R * (entry - stop)
    stop_pct = (entry - stop) / entry * 100

    # Calibrated ranges (code/trading/ranges.py) for each of the next five sessions. They say how
    # far price is likely to move, not which way, so the band is centred on tonight's close.
    future = _next_trading_days(last_day, HORIZON_DAYS)
    try:
        bands = [latest_band(close, h) for h in range(1, HORIZON_DAYS + 1)]
    except ValueError:
        bands = []

    tail = df.iloc[-CHART_BARS:]
    candles = [[_unix(ts), round(float(r.open), 2), round(float(r.high), 2), round(float(r.low), 2), round(float(r.close), 2)]
               for ts, r in tail.iterrows()]
    start_pt = [_unix(last_day), round(last, 2)]

    def _line(key: str) -> List[List[float]]:
        return [start_pt] + [[_unix(d), round(b[key], 2)] for d, b in zip(future, bands)] if bands else []

    return {
        "kind": "stock",
        "symbol": symbol,
        "side": "BUY",
        "reason": reason,
        "asof": last_day.isoformat(),
        "next_day": future[0].isoformat(),
        "last": round(last, 2),
        "order": order,
        "tradeable": setup["status"] == "IN BUY RANGE"
                     or (setup["status"] == "BASING" and setup["pct_to_pivot"] <= NEAR_PIVOT_PCT),
        "setup": setup,
        "metrics": metrics,
        "stop_pct": round(stop_pct, 1),
        "wide_stop": stop_pct > MAX_STOP_PCT,
        **_results_fields(results_date),
        "entry": round(entry, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "atr": round(atr, 2),
        "qty": suggested_qty(entry, stop, risk_pct),
        "next_low": round(bands[0]["lo"], 2) if bands else None,
        "next_high": round(bands[0]["hi"], 2) if bands else None,
        "week_low": round(bands[-1]["lo"], 2) if bands else None,
        "week_high": round(bands[-1]["hi"], 2) if bands else None,
        "candles": candles,
        "path_low": _line("lo"),
        "path_high": _line("hi"),
    }


def predict_pair(item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    spread_pts = item.get("spread_series") or []
    upper = item.get("upper_band_series") or []
    lower = item.get("lower_band_series") or []
    if len(spread_pts) < 20 or not upper or not lower:
        return None
    entry_z, exit_z = float(item["entry_z"]), float(item["exit_z"])
    s0 = float(spread_pts[-1][1])
    mean = (float(upper[-1][1]) + float(lower[-1][1])) / 2
    sigma = (float(upper[-1][1]) - float(lower[-1][1])) / (2 * entry_z)
    if sigma <= 0:
        return None
    side = item["status"]  # BUY spread when it is stretched low, SELL when stretched high
    sign = 1 if side == "BUY" else -1
    target = mean - sign * exit_z * sigma
    stop = mean - sign * PAIR_HARD_STOP_Z * sigma
    half_life = max(1.0, float(item.get("half_life_days") or 10))

    last_day = pd.Timestamp(spread_pts[-1][0], unit="ms").date()
    horizon = int(min(30, max(HORIZON_DAYS, round(2 * half_life))))
    future = _next_trading_days(last_day, horizon)
    path = [[_unix(last_day), round(s0, 2)]]
    for t, d in enumerate(future, start=1):
        path.append([_unix(d), round(mean + (s0 - mean) * 0.5 ** (t / half_life), 2)])
    next_mid = path[1][1]
    qty_a = int(item.get("qty_a") or 100)
    near_stop = abs(s0 - stop) < 0.5 * sigma

    return {
        "kind": "pair",
        "symbol": item["pair"],
        "ticker_a": item["ticker_a"],
        "ticker_b": item["ticker_b"],
        "hedge_ratio": float(item["hedge_ratio"]),
        "qty": qty_a,
        "qty_b": int(item.get("qty_b") or round(qty_a * float(item["hedge_ratio"]))),
        "fno": bool(item.get("fno")),
        "lots_b": item.get("lots_b"),
        "notional_a": item.get("notional_a"),
        "notional_b": item.get("notional_b"),
        "tradeable": bool(item.get("fno")),
        "side": side,
        "reason": f"Spread stretched to z {float(item['zscore']):.2f}; usually snaps back halfway in ~{half_life:.0f} days",
        "asof": last_day.isoformat(),
        "next_day": future[0].isoformat(),
        "entry": round(s0, 2),
        "stop": round(stop, 2),
        "target": round(target, 2),
        "mean": round(mean, 2),
        "zscore": round(float(item["zscore"]), 2),
        "half_life": round(half_life, 1),
        "next_mid": next_mid,
        "next_move": round(next_mid - s0, 2),
        "expected_profit": round(abs(target - s0) * qty_a),
        "near_stop": bool(near_stop),
        "expected_days": round(half_life * np.log2(max(abs(s0 - mean), 1e-9) / (exit_z * sigma))) if abs(s0 - mean) > exit_z * sigma else 0,
        "spread": [[int(p[0] // 1000), round(float(p[1]), 2)] for p in spread_pts[-120:]],
        "upper": [[int(p[0] // 1000), round(float(p[1]), 2)] for p in upper[-120:]],
        "lower": [[int(p[0] // 1000), round(float(p[1]), 2)] for p in lower[-120:]],
        "path_mid": path,
    }


MAX_SCREEN_PREDICTIONS = 60


def prediction_candidates(screen: Dict[str, Any], pairs: Dict[str, Any], watchlist: List[str]) -> List[Dict[str, Any]]:
    """Every stock passing all 8 conditions, every pair in its entry zone, and the watchlist."""
    by_symbol = {i["symbol"]: i for i in screen.get("items", [])}
    out: List[Dict[str, Any]] = []
    passed = sorted((i for i in screen.get("items", []) if i.get("passed")), key=lambda i: i.get("rs_rank") or 0, reverse=True)
    for i in passed[:MAX_SCREEN_PREDICTIONS]:
        out.append({"symbol": i["symbol"], "source": "screen", "rs_rank": i.get("rs_rank"),
                    "reason": f"Passed all 8 Minervini conditions, RS rank {i.get('rs_rank')}", "link": f"/stock/{i['symbol']}"})
    for p in pairs.get("items", []):
        if p.get("status") in ("BUY", "SELL"):
            out.append({"symbol": p["pair"], "source": "pair", "link": "/pairs"})
    seen = {c["symbol"] for c in out}
    for sym in watchlist:
        if sym in seen:
            next(c for c in out if c["symbol"] == sym)["on_watchlist"] = True
            continue
        item = by_symbol.get(sym, {})
        cond = item.get("conditions_passed")
        reason = "On your watchlist" + (f" — {cond}/8 Minervini conditions, RS rank {item.get('rs_rank')}" if cond is not None else "")
        out.append({"symbol": sym, "source": "watchlist", "rs_rank": item.get("rs_rank"), "reason": reason, "link": f"/stock/{sym}"})
    return out


def build_predictions(
    candidates: List[Dict[str, Any]], pairs: Dict[str, Any], fetch,
    risk_pct: float = RISK_PER_TRADE * 100, results: Optional[Dict[str, Optional[str]]] = None,
) -> List[Dict[str, Any]]:
    """``fetch(ticker) -> OHLCV DataFrame`` is injected so tests and the pipeline share one code path."""
    pair_items = {p["pair"]: p for p in pairs.get("items", [])}
    results = results or {}
    out: List[Dict[str, Any]] = []
    for entry in candidates:
        symbol = entry["symbol"]
        try:
            if symbol in pair_items:
                pred = predict_pair(pair_items[symbol])
            else:
                pred = predict_stock(symbol, fetch(symbol), entry.get("reason", ""),
                                     risk_pct=risk_pct, results_date=results.get(symbol))
        except Exception:
            logger.exception("prediction failed for %s", symbol)
            pred = None
        if pred:
            pred["link"] = entry.get("link")
            pred["source"] = entry.get("source", "screen")
            pred["rs_rank"] = entry.get("rs_rank")
            pred["on_watchlist"] = entry.get("on_watchlist", entry.get("source") == "watchlist")
            out.append(pred)
    return out
