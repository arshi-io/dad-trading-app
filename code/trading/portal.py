"""Mock trading portal — record paper trades, track predictions, show P&L."""
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "papa.db"


def _init_trades_table() -> None:
    """Ensure paper_trades table exists."""
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
        conn.commit()
    finally:
        conn.close()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# Initialize table when module is imported
_init_trades_table()


def place_trade(
    ticker: str,
    side: str,  # BUY or SELL
    entry_price: float,
    signal_type: str,  # "mean_reversion", "pairs_trading", "garch_volatility"
    exit_price: Optional[float] = None,
    notes: str = "",
) -> Dict[str, Any]:
    """Record a new paper trade."""
    conn = _conn()
    try:
        cursor = conn.execute(
            """INSERT INTO paper_trades
               (ticker, side, entry_price, exit_price, entry_date, signal_type, status, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker,
                side,
                entry_price,
                exit_price,
                datetime.now().isoformat(),
                signal_type,
                "open",
                notes,
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        trade_id = cursor.lastrowid
        return {"id": trade_id, "status": "success"}
    finally:
        conn.close()


def close_trade(
    trade_id: int, exit_price: float, exit_date: Optional[str] = None
) -> Dict[str, Any]:
    """Close a paper trade and compute P&L."""
    conn = _conn()
    try:
        trade = conn.execute(
            "SELECT * FROM paper_trades WHERE id=?", (trade_id,)
        ).fetchone()
        if not trade:
            return {"status": "error", "msg": "Trade not found"}

        entry = float(trade["entry_price"])
        side = trade["side"]
        pnl = (exit_price - entry) * (1 if side == "BUY" else -1)
        pnl_pct = (pnl / entry) * 100

        conn.execute(
            """UPDATE paper_trades
               SET exit_price=?, exit_date=?, status=?, pnl=?, pnl_pct=?
               WHERE id=?""",
            (exit_price, exit_date or datetime.now().isoformat(), "closed", pnl, pnl_pct, trade_id),
        )
        conn.commit()
        return {"id": trade_id, "pnl": pnl, "pnl_pct": pnl_pct, "status": "closed"}
    finally:
        conn.close()


def get_trades(limit: int = 50, status: Optional[str] = None) -> List[Dict[str, Any]]:
    """Fetch paper trades."""
    conn = _conn()
    try:
        if status:
            rows = conn.execute(
                "SELECT * FROM paper_trades WHERE status=? ORDER BY created_at DESC LIMIT ?",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM paper_trades ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_trade_stats() -> Dict[str, Any]:
    """Summary stats across all closed trades."""
    conn = _conn()
    try:
        closed = conn.execute(
            "SELECT COUNT(*) as cnt, SUM(pnl) as total_pnl, AVG(pnl_pct) as avg_pnl_pct FROM paper_trades WHERE status='closed'"
        ).fetchone()
        open_trades = conn.execute(
            "SELECT COUNT(*) as cnt FROM paper_trades WHERE status='open'"
        ).fetchone()
        return {
            "closed_trades": closed["cnt"] or 0,
            "total_pnl": closed["total_pnl"] or 0.0,
            "avg_pnl_pct": closed["avg_pnl_pct"] or 0.0,
            "open_trades": open_trades["cnt"] or 0,
        }
    finally:
        conn.close()
