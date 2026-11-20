"""Gatekeeper (PRD F1 + F2). Lives outside the research agent: the agent may call get_data() for
Periods A and B and submit() a strategy, nothing else. Only this module reads Period C.

Strategy contract: Python source defining

    def positions(df: pd.DataFrame) -> pd.Series   # index = df.index, values -1 / 0 / 1

decided at the close of each day and held over the next day. df has open/high/low/close/volume,
forward-adjusted for splits and bonuses. The source runs in a separate isolated Python process,
so it cannot patch this module or read the attempt log from memory.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

PERIODS = {  # inclusive end dates; C runs to the latest data
    "A": ("2011-01-01", "2020-12-31"),   # search
    "B": ("2021-01-01", "2023-12-31"),   # selection
    "C": ("2024-01-01", "2099-12-31"),   # vault: gatekeeper only
}
COST = 0.0025          # round trip: half charged on entry, half on exit
MIN_TRADES = 30
LOOKAHEAD_CHECKS = 8   # cut-off dates in Period C where the strategy is re-run on truncated data
TIMEOUT_S = 600
DB_PATH = Path(__file__).with_name("attempts.db")
RUNNER = Path(__file__).with_name("_run_strategy.py")


# ---------- data ----------------------------------------------------------------------------

def _load(symbol: str) -> pd.DataFrame:
    """Full history from the survivorship-free bhavcopy store, forward-adjusted.

    The bhavcopy is unadjusted and its prev_close is too, so splits and bonuses show up as
    overnight gaps. Dividing by the running product of past factors uses no future data.
    """
    from code.research.bhavcopy import load   # ponytail: reads every day file (~7 s); cache per symbol if it bites
    raw = load([symbol])
    raw = raw[raw["series"] == "EQ"].drop_duplicates("date").sort_values("date")
    return _adjust(raw)


def _adjust(raw: pd.DataFrame) -> pd.DataFrame:
    # Most NSE stocks have a 20% circuit, so a gap down of more than 25% is a split or bonus
    # (1:1 bonus = -50%, 1:2 = -33%). The gap is removed; the day's own open→close move is kept.
    # ponytail: misses demergers and small dividends (<25%); a corporate-actions feed fixes both.
    gap = (raw["open"] / raw["close"].shift(1)).fillna(1.0)
    f = gap.where(gap < 0.75, 1.0)
    cum = f.cumprod().values
    df = raw[["open", "high", "low", "close"]].div(cum, axis=0)
    df["volume"] = raw["volume"].values
    df.index = pd.DatetimeIndex(raw["date"], name="date")
    return df.astype(float)


def get_data(symbol: str, period: str) -> pd.DataFrame:
    """History from the start through the end of Period A or B (earlier data kept for warm-up)."""
    if period not in ("A", "B"):
        raise PermissionError("Only the gatekeeper reads Period C.")
    return _upto(_load(symbol), PERIODS[period][1])


def _upto(df: pd.DataFrame, end: str) -> pd.DataFrame:
    return df[df.index <= pd.Timestamp(end)]


# ---------- running a strategy --------------------------------------------------------------

def _run(source: str, df: pd.DataFrame, cuts: List[pd.Timestamp]) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        t = Path(tmp)
        (t / "strategy.py").write_text(source, encoding="utf-8")
        df.to_parquet(t / "data.parquet")
        (t / "cuts.json").write_text(json.dumps([str(c.date()) for c in cuts]))
        p = subprocess.run([sys.executable, "-I", str(RUNNER), str(t)], cwd=tmp,
                           capture_output=True, text=True, timeout=TIMEOUT_S)
        if p.returncode != 0:
            raise RuntimeError(p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "strategy failed")
        out = json.loads((t / "out.json").read_text())
    pos = pd.Series(out["full"], index=df.index, dtype=float).fillna(0).clip(-1, 1).round()
    return {"pos": pos, "cut": {pd.Timestamp(k): v for k, v in out["cut"].items()}}


# ---------- scoring -------------------------------------------------------------------------

def trades(pos: pd.Series, close: pd.Series) -> Dict[str, Any]:
    """Daily P&L of holding pos[t] over t→t+1, costs on every change in position."""
    ret = close.pct_change().shift(-1).fillna(0)          # return earned by the position held at t
    turn = pos.diff().abs().fillna(pos.abs())             # entering or leaving a position
    daily = pos * ret - turn * COST / 2 - _exit_cost(pos)
    run_id = (pos != pos.shift()).cumsum()
    nets = []
    for _, g in pos[pos != 0].groupby(run_id[pos != 0]):
        gross = float(np.prod(1 + g * ret[g.index]) - 1)
        nets.append(gross - COST)
    equity = (1 + daily).cumprod()
    dd = float((equity / equity.cummax() - 1).min()) if len(equity) else 0.0
    nets = np.array(nets)
    wins, losses = nets[nets > 0], nets[nets <= 0]
    return {
        "n_trades": int(len(nets)),
        "expectancy": float(nets.mean()) if len(nets) else 0.0,
        "win_rate": float(len(wins) / len(nets)) if len(nets) else 0.0,
        "avg_win": float(wins.mean()) if len(wins) else 0.0,
        "avg_loss": float(losses.mean()) if len(losses) else 0.0,
        "max_drawdown": dd,
        "total_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0,
        "trade_returns": nets.tolist(),
    }


def _exit_cost(pos: pd.Series) -> pd.Series:
    """A position still open on the last day is closed there and pays its exit half."""
    c = pd.Series(0.0, index=pos.index)
    if len(pos):
        c.iloc[-1] = abs(pos.iloc[-1]) * COST / 2
    return c


def _vault(pos: pd.Series) -> pd.Series:
    """Only trades opened inside Period C count; whatever is held when C starts is ignored."""
    return pos.where(pos.index >= pd.Timestamp(PERIODS["C"][0]), 0.0)


# ---------- attempt log ---------------------------------------------------------------------

def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS attempts (
        strategy_hash TEXT, symbol TEXT, name TEXT, idea_attempts INTEGER,
        scored_at TEXT, passed INTEGER, report TEXT, PRIMARY KEY (strategy_hash, symbol))""")
    return conn


def strategy_hash(source: str) -> str:
    # ponytail: exact-text hash; a comment edit counts as a new strategy. The idea's attempt
    # count (reported below) is what exposes re-submission games.
    return hashlib.sha256(source.strip().encode()).hexdigest()


# ---------- the gate ------------------------------------------------------------------------

def submit(source: str, symbol: str, name: str, idea_attempts: int) -> Dict[str, Any]:
    """Score once on Period C. PASS needs: no lookahead, beats the no-change baseline, expectancy
    above 0 after costs, 30+ trades, and no earlier vault attempt for this exact strategy."""
    h = strategy_hash(source)
    with _db() as conn:
        if conn.execute("SELECT 1 FROM attempts WHERE strategy_hash=? AND symbol=?", (h, symbol)).fetchone():
            return {"passed": False, "failed": ["already scored on Period C"], "symbol": symbol, "name": name}
        # Burn the attempt before scoring, so a crash or timeout cannot buy a second look.
        conn.execute("INSERT INTO attempts VALUES (?,?,?,?,?,?,?)",
                     (h, symbol, name, idea_attempts, datetime.now().isoformat(timespec="seconds"), 0, "{}"))

    report = _score(source, symbol, name, idea_attempts)
    with _db() as conn:
        conn.execute("UPDATE attempts SET passed=?, report=? WHERE strategy_hash=? AND symbol=?",
                     (int(report["passed"]), json.dumps(report), h, symbol))
    return report


def _score(source: str, symbol: str, name: str, idea_attempts: int) -> Dict[str, Any]:
    base = {"symbol": symbol, "name": name, "idea_attempts": idea_attempts}
    df = _load(symbol)
    c_idx = df.index[df.index >= pd.Timestamp(PERIODS["C"][0])]
    if len(c_idx) < 2:
        return {**base, "passed": False, "failed": ["no Period C data"]}
    cuts = list(c_idx[np.linspace(0, len(c_idx) - 2, LOOKAHEAD_CHECKS).astype(int)])
    try:
        run = _run(source, df, cuts)
    except Exception as e:   # strategy errors are a failed attempt, not a gate crash
        return {**base, "passed": False, "failed": [f"strategy error: {e}"]}

    pos = run["pos"]
    leaked = [str(t.date()) for t, v in run["cut"].items()
              if round(float(np.clip(np.nan_to_num(v), -1, 1))) != pos.get(t, 0)]
    stats = trades(_vault(pos), df["close"])
    stats.pop("trade_returns")
    failed = []
    if leaked:
        failed.append(f"lookahead: positions change when later data is removed ({', '.join(leaked[:3])})")
    # The baseline (No change + ATR vol + Conformal calibration) forecasts zero move, so it never
    # trades and nets exactly 0 on Period C; beating it means a positive net return.
    if stats["total_return"] <= 0:
        failed.append("does not beat the no-change baseline")
    if stats["expectancy"] <= 0:
        failed.append("expectancy not above 0 after costs")
    if stats["n_trades"] < MIN_TRADES:
        failed.append(f"only {stats['n_trades']} trades (need {MIN_TRADES})")
    return {**base, **stats, "passed": not failed, "failed": failed,
            "period": f"{c_idx[0].date()} to {c_idx[-1].date()}", "cost": COST}
