"""Market-hours guidance: what to do right now with each setup and each open position.

Pure functions over tonight's trade plan (code/trading/predict.py) and the latest live quote --
no network, no orders. Every output is a suggestion for a human; nothing here can trade.

The process is Minervini's: buy only as a stock clears its pivot on strong volume, never more
than 5% past it, cut every loss at the stop, and let winners pay for the losers.
"""
from __future__ import annotations

from datetime import datetime, time
from typing import Any, Dict, Optional

BUY_RANGE_PCT = 5.0         # past the pivot by more than this = extended, don't chase
BUY_RANGE_PCT_SHORT = 3.0   # under support by more than this = extended down
VOLUME_PACE_MIN = 1.4       # breakout volume should run well above normal
SESSION_OPEN, SESSION_CLOSE = time(9, 15), time(15, 30)
SETTLE_UNTIL = time(9, 30)  # first 15 minutes: opening prices are noisy
CLOSING_FROM = time(15, 0)
SESSION_MINUTES = 375

# The trading day, in order. `key` is matched by day_phase().
ROUTINE = [
    {"key": "before", "when": "Before 9:15", "title": "Plan the day",
     "text": "Check the market grade and today's size. For each setup, write down the pivot, the stop and the quantity. Skip anything with results inside the holding period."},
    {"key": "settle", "when": "9:15 – 9:30", "title": "Hands off",
     "text": "Let the opening settle. Don't buy in the first 15 minutes, and never buy a gap that opens more than 5% past the pivot."},
    {"key": "session", "when": "9:30 – 3:00", "title": "Act only on signals",
     "text": "Buy only on BUY, short only on SHORT: through the level, on strong volume, not yet extended. Put the stop in the moment you enter. Sell at once anything that shows EXIT."},
    {"key": "closing", "when": "3:00 – 3:30", "title": "Check the close",
     "text": "Breakouts that close near the day's high are the strong ones. Anything you hold that shows AT SELL LEVEL: sell if it's still under that level at the close. Confirm every open position has a stop."},
    {"key": "after", "when": "After 3:30", "title": "Review",
     "text": "Write one line per trade: why you bought, what happened. Tonight's run builds tomorrow's list."},
]


def day_phase(now: datetime, trading_day: bool = True) -> str:
    """Which ROUTINE step `now` (IST) falls in. Non-trading days are review days."""
    if not trading_day:
        return "after"
    t = now.time()
    if t < SESSION_OPEN:
        return "before"
    if t < SETTLE_UNTIL:
        return "settle"
    if t < CLOSING_FROM:
        return "session"
    if t < SESSION_CLOSE:
        return "closing"
    return "after"


def volume_pace(day_volume: float, avg_volume: Optional[float], now: datetime) -> Optional[float]:
    """Today's volume projected to a full session, as a multiple of the 50-day average.

    ponytail: linear projection; intraday volume is U-shaped so early readings run high. That's why
    no buy signal fires before SETTLE_UNTIL. Upgrade path: a per-minute volume curve from 1m bars.
    """
    if not avg_volume or day_volume <= 0:
        return None
    minutes = (now.hour * 60 + now.minute) - (SESSION_OPEN.hour * 60 + SESSION_OPEN.minute)
    if minutes <= 0:
        return None
    elapsed = min(1.0, minutes / SESSION_MINUTES)
    return day_volume / elapsed / avg_volume


def _rs(x: float) -> str:
    return f"₹{x:,.2f}"


def setup_signal(plan: Dict[str, Any], quote: Dict[str, Any], now: datetime) -> Dict[str, str]:
    """Live state of one setup. `plan` is a stock prediction from the snapshot; `quote` is a
    /api/quotes entry (ltp, day_high, volume). Returns {state, cls, text}."""
    pivot = (plan.get("setup") or {}).get("pivot") or plan["entry"]
    stop, qty = plan["stop"], plan.get("qty")
    ltp, high = float(quote["ltp"]), float(quote.get("day_high") or quote["ltp"])
    past = (ltp / pivot - 1) * 100
    top = pivot * (1 + BUY_RANGE_PCT / 100)
    pace = volume_pace(float(quote.get("volume") or 0), (plan.get("metrics") or {}).get("avg_vol50"), now)
    pace_txt = f"volume pace {pace:.1f}× normal" if pace is not None else "volume unknown"

    if ltp <= stop:
        return {"state": "AVOID", "cls": "bad", "text": f"Below the stop {_rs(stop)}. The setup is broken; leave it."}
    if past > BUY_RANGE_PCT:
        return {"state": "EXTENDED", "cls": "bad", "text": f"{past:.1f}% past the pivot. Don't chase; wait for a new base or a pullback."}
    if ltp < pivot:
        if high >= pivot:
            return {"state": "FAILED", "cls": "bad", "text": f"Crossed {_rs(pivot)} and fell back below it. Failed breakout; skip it today."}
        return {"state": "WAIT", "cls": "neutral", "text": f"{-past:.1f}% below the pivot {_rs(pivot)}. Nothing to do until it trades through."}
    # Inside the buy range.
    if now.time() < SETTLE_UNTIL:
        return {"state": "WAIT", "cls": "wait", "text": f"Through the pivot, but it's the first 15 minutes. Let the open settle."}
    size = f" Buy {qty} shares," if qty else " Buy,"
    levels = f"{size} no higher than {_rs(top)}. Stop {_rs(stop)}, target {_rs(plan['target'])}."
    prev = float(quote.get("prev_close") or 0)
    if prev >= pivot:  # broke out on an earlier day: the volume test was that day's, not today's
        return {"state": "BUY RANGE", "cls": "good", "text": f"Broke out earlier and still within {BUY_RANGE_PCT:.0f}% of the pivot {_rs(pivot)}.{levels}"}
    if pace is None:
        return {"state": "CHECK", "cls": "wait", "text": f"Through the pivot; volume unknown. Buy only if the broker shows volume well above normal."}
    if pace < VOLUME_PACE_MIN:
        return {"state": "WEAK", "cls": "wait", "text": f"Through the pivot but {pace_txt}. Light-volume breakouts fail more often; wait for volume or skip."}
    return {"state": "BUY", "cls": "good", "text": f"Breakout today, {pace_txt}.{levels}"}


def short_signal(plan: Dict[str, Any], quote: Dict[str, Any], now: datetime) -> Dict[str, str]:
    """Live state of one short (a `shorts` row from the snapshot): the mirror of setup_signal."""
    support, stop = plan["support"], plan["stop"]
    ltp, low = float(quote["ltp"]), float(quote.get("day_low") or quote["ltp"])
    below = (1 - ltp / support) * 100
    pace = volume_pace(float(quote.get("volume") or 0), plan.get("avg_vol50"), now)

    if ltp >= stop:
        return {"state": "AVOID", "cls": "bad", "text": f"Back above the stop {_rs(stop)}. The short idea is off."}
    if below > BUY_RANGE_PCT_SHORT:
        return {"state": "EXTENDED", "cls": "bad", "text": f"{below:.1f}% under support already. Don't chase it down; wait for a bounce."}
    if ltp > support:
        if low <= support:
            return {"state": "FAILED", "cls": "bad", "text": f"Broke {_rs(support)} and bounced back above it. Failed breakdown; skip it."}
        return {"state": "WAIT", "cls": "neutral", "text": f"{-below:.1f}% above support {_rs(support)}. Sell-stop at {_rs(plan['entry'])}; nothing until it breaks."}
    if now.time() < SETTLE_UNTIL:
        return {"state": "WAIT", "cls": "wait", "text": "Under support, but it's the first 15 minutes. Let the open settle."}
    if pace is None:
        return {"state": "CHECK", "cls": "wait", "text": "Under support; volume unknown. Short only if volume is well above normal."}
    if pace < VOLUME_PACE_MIN:
        return {"state": "WEAK", "cls": "wait", "text": f"Under support but volume pace only {pace:.1f}× normal. Wait for volume or skip."}
    levels = f"no lower than {_rs(support * (1 - BUY_RANGE_PCT_SHORT / 100))}. Stop {_rs(stop)}, target {_rs(plan['target'])}."
    if not plan.get("lots"):
        return {"state": "SHORT", "cls": "wait",
                "text": f"Breakdown, volume pace {pace:.1f}× normal, but one futures lot risks more than today's budget. Skip, or take 1 lot only if your account is big enough: {levels}"}
    return {"state": "SHORT", "cls": "bad",
            "text": f"Breakdown, volume pace {pace:.1f}× normal. Sell {plan['lots']} futures lot{'s' if plan['lots'] != 1 else ''}, {levels}"}


def sell_level_signal(item: Dict[str, Any], ltp: float) -> Optional[Dict[str, str]]:
    """A stock you hold trading under its sell level right now. The rule is on the close, so this
    is a warning to watch the last half hour, not a sell by itself."""
    level = item.get("sell_below")
    if not level or ltp >= level:
        return None
    return {"state": "AT SELL LEVEL", "cls": "bad", "text": f"Trading at {_rs(ltp)}, under the sell level {_rs(level)}. If it closes below it, sell."}


def position_signal(trade: Dict[str, Any], ltp: float) -> Dict[str, str]:
    """Live guidance for one open paper position (portal.get_trades row)."""
    entry, stop, target = trade["entry_price"], trade.get("stop"), trade.get("target")
    if stop is None:
        return {"state": "NO STOP", "cls": "bad", "text": "No stop on this position. Set one now: no stop means no limit on the loss."}
    sign = -1 if trade["side"] == "SELL" else 1
    risk = abs(entry - stop)
    r = sign * (ltp - entry) / risk if risk else 0.0
    if sign * (ltp - stop) <= 0:
        return {"state": "EXIT", "cls": "bad", "text": f"At the stop {_rs(stop)}. Sell. A small loss taken now is the plan working."}
    if target is not None and trade.get("exit_rule") != "trail21" and sign * (ltp - target) >= 0:
        return {"state": "TAKE PROFIT", "cls": "good", "text": f"Target {_rs(target)} reached ({r:+.1f}R). Sell at least half; move the stop to {_rs(entry)} on the rest."}
    if r >= 1:
        return {"state": "PROTECT", "cls": "good", "text": f"Up {r:.1f}R. Raise the stop to break-even {_rs(entry)} so this can't become a loss."}
    if r < 0:
        return {"state": "HOLD", "cls": "wait", "text": f"{r:+.1f}R. Above the stop, so hold. Don't widen the stop."}
    return {"state": "HOLD", "cls": "neutral", "text": f"{r:+.1f}R. Hold; let the stop and target do the work."}

