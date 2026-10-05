"""Mock trading portal — record paper trades, settle them against real prices, score predictions."""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "papa.db"  # same file as watchlist/memos -- the persisted volume

_TRADE_EXTRA_COLUMNS = {
    "qty": "INTEGER DEFAULT 1",
    "target": "REAL",
    "stop": "REAL",
    "kind": "TEXT DEFAULT 'stock'",
    "ticker_a": "TEXT",
    "ticker_b": "TEXT",
    "hedge_ratio": "REAL",
    "last_price": "REAL",
    "last_price_date": "TEXT",
    "exit_reason": "TEXT",
    "exit_rule": "TEXT DEFAULT 'target'",   # 'target' (fixed 2R) | 'trail21' (close below 21-day EMA, ≤60 sessions)
}


def _init_tables() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS paper_trades (
                id INTEGER PRIMARY KEY,
                ticker TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                entry_date TEXT NOT NULL,
                exit_date TEXT,
                signal_type TEXT,
                status TEXT DEFAULT 'open',
                pnl REAL,
                pnl_pct REAL,
                notes TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        existing = {row[1] for row in conn.execute("PRAGMA table_info(paper_trades)")}
        for col, decl in _TRADE_EXTRA_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE paper_trades ADD COLUMN {col} {decl}")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS prediction_log (
                asof TEXT NOT NULL,
                symbol TEXT NOT NULL,
                entry REAL NOT NULL,
                next_low REAL NOT NULL,
                next_mid REAL NOT NULL,
                next_high REAL NOT NULL,
                actual REAL,
                inside INTEGER,
                direction_ok INTEGER,
                PRIMARY KEY (asof, symbol)
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


_init_tables()


def _pnl(side: str, entry: float, exit_price: float, qty: int) -> float:
    return (exit_price - entry) * (1 if side == "BUY" else -1) * (qty or 1)


def place_trade(
    ticker: str,
    side: str,
    entry_price: float,
    signal_type: str,
    exit_price: Optional[float] = None,
    notes: str = "",
    qty: int = 1,
    target: Optional[float] = None,
    stop: Optional[float] = None,
    kind: str = "stock",
    ticker_a: Optional[str] = None,
    ticker_b: Optional[str] = None,
    hedge_ratio: Optional[float] = None,
    price_date: Optional[str] = None,
    pending: bool = False,
    max_positions: Optional[int] = None,
    exit_rule: str = "target",
) -> Dict[str, Any]:
    """``pending`` = a buy-stop at ``entry_price`` that fills only if price trades through it."""
    now = datetime.now().isoformat()
    conn = _conn()
    try:
        if max_positions is not None:
            held = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE status IN ('open', 'pending')").fetchone()[0]
            if held >= max_positions:
                return {"status": "error", "msg": f"You already hold {held} positions/orders — the limit in this market is {max_positions}. Close or cancel one first."}
        cursor = conn.execute(
            """INSERT INTO paper_trades
               (ticker, side, entry_price, exit_price, entry_date, signal_type, status, notes, created_at,
                qty, target, stop, kind, ticker_a, ticker_b, hedge_ratio, last_price, last_price_date, exit_rule)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ticker, side, entry_price, exit_price, now, signal_type, "pending" if pending else "open", notes, now,
             qty, target, stop, kind, ticker_a, ticker_b, hedge_ratio, entry_price, price_date or now[:10],
             exit_rule if exit_rule in ("target", "trail21") else "target"),
        )
        conn.commit()
        return {"id": cursor.lastrowid, "status": "success"}
    finally:
        conn.close()


def set_note(trade_id: int, note: str) -> Dict[str, Any]:
    conn = _conn()
    try:
        conn.execute("UPDATE paper_trades SET notes=? WHERE id=?", (note.strip()[:2000], trade_id))
        conn.commit()
        return {"status": "saved"}
    finally:
        conn.close()


def close_trade(
    trade_id: int, exit_price: Optional[float] = None, exit_date: Optional[str] = None, reason: str = "Closed by you"
) -> Dict[str, Any]:
    """Close at ``exit_price``, or at the last known market price when none is given."""
    conn = _conn()
    try:
        trade = conn.execute("SELECT * FROM paper_trades WHERE id=?", (trade_id,)).fetchone()
        if not trade or trade["status"] != "open":
            return {"status": "error", "msg": "Trade not found or already closed"}
        if exit_price is None:
            exit_price = trade["last_price"] if trade["last_price"] is not None else trade["entry_price"]
        entry = float(trade["entry_price"])
        pnl = _pnl(trade["side"], entry, float(exit_price), trade["qty"])
        pnl_pct = None if trade["kind"] == "pair" else (pnl / (entry * (trade["qty"] or 1))) * 100
        conn.execute(
            """UPDATE paper_trades
               SET exit_price=?, exit_date=?, status='closed', pnl=?, pnl_pct=?, exit_reason=?
               WHERE id=?""",
            (exit_price, exit_date or datetime.now().isoformat(), pnl, pnl_pct, reason, trade_id),
        )
        conn.commit()
        return {"id": trade_id, "pnl": pnl, "pnl_pct": pnl_pct, "status": "closed"}
    finally:
        conn.close()


def delete_trade(trade_id: int) -> Dict[str, Any]:
    """Closed/expired trades and unfilled orders only -- an open one is closed first, so it can't vanish with P&L still running."""
    conn = _conn()
    try:
        cur = conn.execute("DELETE FROM paper_trades WHERE id=? AND status IN ('closed', 'expired', 'pending')", (trade_id,))
        conn.commit()
        return {"status": "deleted"} if cur.rowcount else {"status": "error", "msg": "Only closed trades can be deleted"}
    finally:
        conn.close()


def delete_all_closed() -> Dict[str, Any]:
    conn = _conn()
    try:
        cur = conn.execute("DELETE FROM paper_trades WHERE status IN ('closed', 'expired')")
        conn.commit()
        return {"status": "deleted", "count": cur.rowcount}
    finally:
        conn.close()


def get_watchlist() -> List[str]:
    conn = _conn()
    try:
        return [r[0] for r in conn.execute("SELECT symbol FROM watchlist ORDER BY added_at")]
    except sqlite3.OperationalError:
        return []
    finally:
        conn.close()


def _decorate(row: sqlite3.Row) -> Dict[str, Any]:
    t = dict(row)
    qty = t.get("qty") or 1
    if t["status"] == "open":
        mark = t.get("last_price") if t.get("last_price") is not None else t["entry_price"]
        t["mtm_pnl"] = _pnl(t["side"], t["entry_price"], mark, qty)
    if t.get("stop") is not None:
        risk = abs(t["entry_price"] - t["stop"]) * qty
        realised = t["pnl"] if t["status"] == "closed" else t.get("mtm_pnl")
        t["r_multiple"] = (realised / risk) if risk and realised is not None else None
    return t


def get_trades(limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
    conn = _conn()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM paper_trades WHERE status=? ORDER BY created_at DESC LIMIT ?", (status, limit)
            ).fetchall()
        else:
            rows = conn.execute("SELECT * FROM paper_trades ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        return [_decorate(r) for r in rows]
    finally:
        conn.close()


def get_trade_stats() -> Dict[str, Any]:
    conn = _conn()
    try:
        closed = conn.execute(
            """SELECT COUNT(*) AS cnt, SUM(pnl) AS total_pnl,
                      SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) AS wins
               FROM paper_trades WHERE status='closed'"""
        ).fetchone()
        open_rows = conn.execute("SELECT * FROM paper_trades WHERE status='open'").fetchall()
        pending = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE status='pending'").fetchone()[0]
        open_pnl = sum(_decorate(r).get("mtm_pnl") or 0 for r in open_rows)
        cnt = closed["cnt"] or 0
        return {
            "closed_trades": cnt,
            "total_pnl": closed["total_pnl"] or 0.0,
            "win_rate": round(100 * (closed["wins"] or 0) / cnt) if cnt else None,
            "open_trades": len(open_rows),
            "pending_orders": pending,
            "open_pnl": open_pnl,
        }
    finally:
        conn.close()


def _bars_after(df: pd.DataFrame, since_iso: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.dropna(subset=["open", "high", "low", "close"])
    cutoff = pd.Timestamp(since_iso[:10])
    idx = pd.DatetimeIndex(df.index).tz_localize(None) if pd.DatetimeIndex(df.index).tz else pd.DatetimeIndex(df.index)
    return df[idx > cutoff]


def settle_open_trades(fetch: Callable[[str], pd.DataFrame]) -> Dict[str, int]:
    """Walk each open trade's bars since entry: auto-close on stop/target, otherwise mark to last close.

    If one bar touches both, assume the stop hit first -- the conservative read of a daily bar.
    """
    closed = marked = filled = expired = 0
    for t in get_trades(limit=500, status="open") + get_trades(limit=500, status="pending"):
        side, target, stop = t["side"], t.get("target"), t.get("stop")
        long = side == "BUY"
        if t.get("kind") == "pair":
            a, b = fetch(t["ticker_a"]), fetch(t["ticker_b"])
            if a.empty or b.empty:
                continue
            ca, cb = a["close"].astype(float).align(b["close"].astype(float), join="inner")
            spread = (ca - float(t["hedge_ratio"]) * cb).to_frame("close")
            spread["open"] = spread["high"] = spread["low"] = spread["close"]
            bars = _bars_after(spread, t["entry_date"])
        else:
            full = fetch(t["ticker"])
            bars = _bars_after(full, t["entry_date"])
        if bars.empty:
            continue
        trailing = t.get("exit_rule") == "trail21" and t.get("kind") != "pair"
        ema21 = full["close"].astype(float).ewm(span=21, adjust=False).mean() if trailing else None

        if t["status"] == "pending":
            bars, outcome = _try_fill(t, bars)
            if outcome != "filled":
                if outcome:
                    _set_status(t["id"], "expired", outcome)
                    expired += 1
                continue
            filled += 1
            t = _decorate_by_id(t["id"])
            if bars.empty:
                continue

        exit_hit = None
        for k, (ts, bar) in enumerate(bars.iterrows()):
            o, h, l, c = float(bar["open"]), float(bar["high"]), float(bar["low"]), float(bar["close"])
            if stop is not None and ((long and l <= stop) or (not long and h >= stop)):
                fill = min(o, stop) if long else max(o, stop)
                exit_hit = (fill, ts, "Stop hit")
                break
            if trailing:
                # Let winners run: sell on a close below the 21-day average (not on the entry day),
                # or at the 60-session limit. Backtest: ~0R vs −0.06R for the fixed 2R target.
                if k > 0 and ts in ema21.index and c < float(ema21.loc[ts]):
                    exit_hit = (c, ts, "Closed below 21-day average")
                    break
                if k + 1 >= TRAIL_MAX_SESSIONS:
                    exit_hit = (c, ts, f"{TRAIL_MAX_SESSIONS}-session limit")
                    break
                continue
            if target is not None and ((long and h >= target) or (not long and l <= target)):
                fill = max(o, target) if long else min(o, target)
                exit_hit = (fill, ts, "Target hit")
                break

        if exit_hit:
            fill, ts, reason = exit_hit
            close_trade(t["id"], round(fill, 2), pd.Timestamp(ts).date().isoformat(), reason)
            closed += 1
        else:
            conn = _conn()
            try:
                conn.execute(
                    "UPDATE paper_trades SET last_price=?, last_price_date=? WHERE id=?",
                    (round(float(bars["close"].iloc[-1]), 2), pd.Timestamp(bars.index[-1]).date().isoformat(), t["id"]),
                )
                conn.commit()
            finally:
                conn.close()
            marked += 1
    return {"closed": closed, "marked": marked, "filled": filled, "expired": expired}


PENDING_EXPIRY_BARS = 10
TRAIL_MAX_SESSIONS = 60


def _try_fill(t: Dict[str, Any], bars: pd.DataFrame):
    """Buy-stop: fills at the trigger (or the open, if it gaps through). Cancelled if the stock
    breaks its stop first, expired after 10 sessions without a breakout.
    Returns (bars from the fill onward, "filled") or (bars, reason-or-None)."""
    trigger, stop = float(t["entry_price"]), t.get("stop")
    for k, (ts, bar) in enumerate(bars.iterrows()):
        if stop is not None and float(bar["low"]) <= stop:
            return bars, "Cancelled — fell through the stop before breaking out"
        if float(bar["high"]) >= trigger:
            fill = max(float(bar["open"]), trigger)
            conn = _conn()
            try:
                conn.execute(
                    "UPDATE paper_trades SET status='open', entry_price=?, entry_date=?, last_price=?, last_price_date=? WHERE id=?",
                    (round(fill, 2), pd.Timestamp(ts).date().isoformat(), round(float(bar["close"]), 2),
                     pd.Timestamp(ts).date().isoformat(), t["id"]),
                )
                conn.commit()
            finally:
                conn.close()
            # The fill bar itself can't be trusted for exits (order of high/low unknown); check from the next bar.
            return bars.iloc[k + 1:], "filled"
        if k + 1 >= PENDING_EXPIRY_BARS:
            return bars, "Expired — no breakout within 10 sessions"
    return bars, None


def _set_status(trade_id: int, status: str, reason: str) -> None:
    conn = _conn()
    try:
        conn.execute("UPDATE paper_trades SET status=?, exit_reason=?, exit_date=? WHERE id=?",
                     (status, reason, datetime.now().isoformat(), trade_id))
        conn.commit()
    finally:
        conn.close()


def _decorate_by_id(trade_id: int) -> Dict[str, Any]:
    conn = _conn()
    try:
        return _decorate(conn.execute("SELECT * FROM paper_trades WHERE id=?", (trade_id,)).fetchone())
    finally:
        conn.close()


def log_predictions(preds: List[Dict[str, Any]]) -> None:
    conn = _conn()
    try:
        for p in preds:
            if p.get("kind") != "stock":
                continue
            conn.execute(
                """INSERT OR IGNORE INTO prediction_log (asof, symbol, entry, next_low, next_mid, next_high)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (p["asof"], p["symbol"], p.get("last", p["entry"]), p["next_low"], p.get("next_mid", p.get("last", p["entry"])), p["next_high"]),
            )
        conn.commit()
    finally:
        conn.close()


def score_predictions(fetch: Callable[[str], pd.DataFrame]) -> int:
    conn = _conn()
    try:
        pending = conn.execute("SELECT * FROM prediction_log WHERE actual IS NULL").fetchall()
    finally:
        conn.close()
    scored = 0
    for p in pending:
        bars = _bars_after(fetch(p["symbol"]), p["asof"])
        if bars.empty:
            continue
        actual = float(bars["close"].iloc[0])
        inside = int(p["next_low"] <= actual <= p["next_high"])
        predicted_dir = p["next_mid"] - p["entry"]
        direction_ok = None if abs(predicted_dir) < 1e-9 else int((actual - p["entry"]) * predicted_dir > 0)
        conn = _conn()
        try:
            conn.execute(
                "UPDATE prediction_log SET actual=?, inside=?, direction_ok=? WHERE asof=? AND symbol=?",
                (actual, inside, direction_ok, p["asof"], p["symbol"]),
            )
            conn.commit()
        finally:
            conn.close()
        scored += 1
    return scored


def get_scorecard(last_n: int = 30) -> Dict[str, Any]:
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT * FROM prediction_log WHERE actual IS NOT NULL ORDER BY asof DESC LIMIT ?", (last_n,)
        ).fetchall()
    finally:
        conn.close()
    n = len(rows)
    dir_rows = [r for r in rows if r["direction_ok"] is not None]
    return {
        "scored": n,
        "inside": sum(r["inside"] for r in rows),
        "inside_pct": round(100 * sum(r["inside"] for r in rows) / n) if n else None,
        "direction_pct": round(100 * sum(r["direction_ok"] for r in dir_rows) / len(dir_rows)) if dir_rows else None,
        "recent": [dict(r) for r in rows[:8]],
    }
