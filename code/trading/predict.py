"""Trade plans for the setups list: entry, stop, target, size, and calibrated price ranges.

Ranges come from code/trading/ranges.py (validated walk-forward). No direction is forecast: an
honest backtest found no reliable edge in predicting which way a stock moves next. Callers pass in
OHLCV they already have -- no network here.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
import math
from typing import Any, Dict, List, Optional, Tuple

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

PAPER_CAPITAL = 1_000_000  # default; the Settings page can change it (paper_capital())


def paper_capital() -> int:
    """Account size from Settings, or the default when the app database isn't available (tests, research)."""
    try:
        from code.live.state import pref
        return int(pref("paper_capital"))
    except Exception:
        return PAPER_CAPITAL
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
    capital = paper_capital()
    by_risk = (capital * risk_pct / 100) / risk_per_share
    by_size = (capital * MAX_POSITION_FRACTION) / entry
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
    setup: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """`setup`: the screen's own base/pivot for this stock. Pass it so the plan and the screen agree;
    detecting again on a different-length history can find a different base."""
    df = df.dropna(subset=["open", "high", "low", "close"])
    if len(df) < 60:
        return None
    close = df["close"].astype(float)
    last = float(close.iloc[-1])
    atr = _atr(df)
    if atr is None:
        return None
    last_day = pd.Timestamp(df.index[-1]).date()
    setup = setup or detect_setup(df)
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


def prediction_candidates(screen: Dict[str, Any], pairs: Dict[str, Any], watchlist: List[str],
                          setups: List[str] = ()) -> List[Dict[str, Any]]:
    """Every stock passing all 8 conditions, every pair in its entry zone, and the watchlist.
    `setups` (Today's list) always get a plan, even below the RS cut -- live guidance needs one."""
    by_symbol = {i["symbol"]: i for i in screen.get("items", [])}
    out: List[Dict[str, Any]] = []
    passed = sorted((i for i in screen.get("items", []) if i.get("passed")), key=lambda i: i.get("rs_rank") or 0, reverse=True)
    top = passed[:MAX_SCREEN_PREDICTIONS]
    top += [by_symbol[s] for s in setups if s in by_symbol and by_symbol[s] not in top]
    for i in top:
        out.append({"symbol": i["symbol"], "source": "screen", "rs_rank": i.get("rs_rank"), "setup": i.get("setup"),
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
        out.append({"symbol": sym, "source": "watchlist", "rs_rank": item.get("rs_rank"), "setup": item.get("setup"),
                    "reason": reason, "link": f"/stock/{sym}"})
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
                                     risk_pct=risk_pct, results_date=results.get(symbol), setup=entry.get("setup"))
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


def daily_sigma(df: pd.DataFrame) -> float:
    """The app's calibrated-range volatility (EWMA blended with one year), as a daily σ of log returns."""
    from code.trading.ranges import _sigma
    s = _sigma(np.log(df["close"].astype(float)).diff())
    s = s[~np.isnan(s)]
    return float(s[-1]) if len(s) else 0.02


def touch_probability(price: float, level: float, sigma: float) -> float:
    """Chance the price trades at `level` at some point next session, assuming no drift
    (reflection principle: P(max ≥ level) = 2·(1 − Φ(ln(level/price)/σ)))."""
    if price <= 0 or sigma <= 0:
        return 0.0
    d = abs(np.log(level / price)) / sigma
    return float(min(1.0, 2 * (1 - 0.5 * (1 + math.erf(d / math.sqrt(2))))))


def portfolio_outlook(open_items: List[Tuple[Dict[str, Any], pd.DataFrame]], capital: float) -> Dict[str, Any]:
    """Next-session and next-week P&L ranges for the open book by historical simulation: replay the
    last year's daily (and 5-day) moves of the stocks you hold, at today's quantities, and read the
    quantiles. Keeps the real correlation between holdings instead of assuming one."""
    out: Dict[str, Any] = {"n": len(open_items), "invested": 0.0, "pnl": 0.0, "heat": 0.0}
    if not open_items:
        return out
    rets, value = {}, {}
    for t, df in open_items:
        qty, last = (t.get("qty") or 1), float(df["close"].iloc[-1])
        out["invested"] += qty * last
        out["pnl"] += qty * (last - t["entry_price"])
        if t.get("stop") is not None:
            out["heat"] += qty * max(0.0, last - t["stop"])
        rets[t["id"]] = np.log(df["close"].astype(float)).diff()
        value[t["id"]] = qty * last
    joint = pd.DataFrame(rets).dropna().iloc[-250:]
    out["heat_pct"] = out["heat"] / capital * 100 if capital else None
    if len(joint) < 60:
        return out
    w = pd.Series(value)
    day = ((np.exp(joint) - 1) * w).sum(axis=1)
    week = ((np.exp(joint.rolling(5).sum().dropna()) - 1) * w).sum(axis=1)
    q = lambda s, p: float(np.quantile(s, p))
    out.update(day_lo=q(day, 0.16), day_hi=q(day, 0.84), day_bad=q(day, 0.05),
               week_lo=q(week, 0.16), week_hi=q(week, 0.84), week_bad=q(week, 0.05), days_used=len(joint))
    return out


def position_view(trade: Dict[str, Any], df: pd.DataFrame, setup: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Chart data, calibrated ranges, the step-by-step plan and tomorrow's action for one long
    position, open or still waiting to fill."""
    from code.signals.sell_rules import sell_check
    plan = predict_stock(trade["ticker"], df, setup=setup)
    if plan is None:
        return None
    entry, stop, target = trade["entry_price"], trade.get("stop"), trade.get("target")
    last = plan["last"]
    pending = trade.get("status") == "pending"
    check = None if pending else sell_check(df, stop)
    out = {k: plan[k] for k in ("candles", "path_low", "path_high", "next_low", "next_high", "week_low", "week_high", "last")}
    out.update(entry=entry, stop=stop, target=target, steps=[], outlook=[], pending=pending, sigma=round(daily_sigma(df), 5))
    if pending:
        since = pd.Timestamp(str(trade.get("created_at", ""))[:10] or df.index[-1])
        left = max(0, 10 - int((pd.DatetimeIndex(df.index) > since).sum()))
        p = touch_probability(last, entry, out["sigma"]) if last < entry else 1.0
        out["trigger_prob"] = round(p, 2)
        out["tomorrow"] = {"action": "WAIT FOR BREAKOUT", "cls": "wait",
                           "text": f"Order waits at ₹{entry:,.2f}, {(entry / last - 1) * 100:.1f}% above the last close. About {p:.0%} chance it trades there next session. {left} session{'s' if left != 1 else ''} left before it expires."}
        out["steps"].append({"state": "now", "title": "Waiting",
                             "text": f"Buys only if it trades at ₹{entry:,.2f} or higher; cancelled if it falls to ₹{stop:,.2f} first, expires after {left} more session{'s' if left != 1 else ''}. Chance of a fill next session ≈ {p:.0%} (from its calibrated volatility, no direction assumed)."})
    if stop is None:
        out["steps"].append({"state": "now", "title": "Set a stop", "text": "This position has no stop. Without one there is no limit on the loss."})
        return out
    risk = entry - stop
    one_r = entry + risk
    r_now = 0.0 if pending else (last - entry) / risk if risk else 0.0
    out["one_r"] = round(one_r, 2)
    out["r_now"] = round(r_now, 2)
    out["steps"] += [
        {"state": "next" if pending else "done" if r_now >= 1 else "now", "title": "Stop",
         "text": f"Sell if it trades at ₹{stop:,.2f} (−1R, ₹{risk * (trade.get('qty') or 1):,.0f}). Never move a stop down."},
        {"state": "done" if r_now >= 1 else "next", "title": "Protect",
         "text": f"At ₹{one_r:,.2f} (+1R) raise the stop to break-even ₹{entry:,.2f}, so the trade can't turn into a loss."},
        {"state": "next", "title": "Take profit",
         "text": (f"Hold while it closes above the 21-day average (₹{check['ema21']:,.2f} now); sell on the first close below it."
                  if trade.get("exit_rule") == "trail21" and check else
                  f"Sell at the target ₹{target:,.2f} (+2R)." if target else "No target set: trail the stop up under each higher low.")},
    ]
    if check:
        out["steps"].append({"state": "now", "title": f"Tonight: {check['action']}", "text": check["reason"], "cls": check["cls"]})
        if check["action"] in ("SELL", "SELL HALF"):
            out["tomorrow"] = {"action": check["action"], "cls": "bad" if check["action"] == "SELL" else "good", "text": check["reason"]}
        elif r_now >= 1:
            out["tomorrow"] = {"action": "RAISE STOP", "cls": "good", "text": f"Up {r_now:.1f}R: move the stop to break-even ₹{entry:,.2f}. Then {check['reason'][0].lower()}{check['reason'][1:]}"}
        else:
            p = touch_probability(last, stop, out["sigma"])
            out["tomorrow"] = {"action": check["action"], "cls": check["cls"],
                               "text": f"{check['reason']} Stop ₹{stop:,.2f} is {(1 - stop / last) * 100:.1f}% below the last close (≈{p:.0%} chance it's hit next session)."}
    if plan.get("week_low") is not None:
        wl, wh = plan["week_low"], plan["week_high"]
        out["outlook"] = [
            f"Next session: likely ₹{plan['next_low']:,.2f}–₹{plan['next_high']:,.2f}. One week: ₹{wl:,.2f}–₹{wh:,.2f} (68% calibrated ranges; they say how far, not which way).",
            (f"The stop is inside next week's likely range: a real chance it's hit. That is the risk the position was sized for."
             if stop >= wl else f"The stop is below next week's likely range: it would take an unusually bad week to hit it."),
            (f"The target is inside next week's likely range." if target and target <= wh else
             f"The target is above next week's likely range: expect it to take more than a week." if target else ""),
        ]
        out["outlook"] = [o for o in out["outlook"] if o]
    return out
