"""The fund manager: one live ₹10,000 fund (goal ₹20,000) run by an agent that lives and dies by profit.

- Starts on the first live evening; replayed history never counts toward it.
- Divides the money across three sleeves, each a book the daily loop already trades:
    ALS  intraday long-short (buy top 5, short bottom 5 of the 1-day ranking)
    S    5-session swing book
    B    20-session position book
  Each week the split moves toward the sleeves earning most per unit of risk over the last 20
  sessions (settled days only), never above CAP or below FLOOR.
- Penalties: a losing day adds one, a winning day removes one (never below zero).
- Death: PENALTY_LIMIT net penalties, or STREAK_LIMIT losing days in a row. A post-mortem records
  which sleeve lost most; the successor takes the next best untried model setup from the model
  searches, preferring a different model family for the sleeve that failed. The money carries over.
Decision support only: nothing here places orders.
"""
from __future__ import annotations

import json
import sqlite3
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

START_CASH = 10_000.0
GOAL = 20_000.0
SLEEVES = {"ALS": 1, "S": 5, "B": 20}          # sleeve book -> the model horizon it trades on
FLOOR, CAP = 0.10, 0.50
REWEIGHT_EVERY, LOOKBACK = 5, 20
PENALTY_LIMIT, STREAK_LIMIT = 15, 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS fund_agents (
  id INTEGER PRIMARY KEY, born TEXT, died TEXT, config TEXT, weights TEXT, start_value REAL, end_value REAL,
  days INTEGER DEFAULT 0, penalties INTEGER DEFAULT 0, streak INTEGER DEFAULT 0, cause TEXT, postmortem TEXT);
CREATE TABLE IF NOT EXISTS fund_ledger (
  date TEXT PRIMARY KEY, agent INTEGER, start_value REAL, end_value REAL, ret REAL, weights TEXT,
  sleeve_rets TEXT, penalties INTEGER, streak INTEGER, event TEXT);
"""


def _init(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def family(key: Optional[str]) -> str:
    if not key:
        return "none"
    if key.startswith("blend("):
        return "blend"
    return "linear" if key.startswith("linear") else "trees"


def _leaderboard(horizon: int) -> List[Dict[str, Any]]:
    from code.research import model_search as ms
    path = ms.paths(horizon)[0]
    try:
        return json.loads(path.read_text(encoding="utf-8"))["tried"]
    except (OSError, ValueError, KeyError):
        return []


def _spec_of(key: str) -> Dict[str, Any]:
    from code.research import model_search as ms
    return {"blend": key[len("blend("):-1].split(" + ")} if key.startswith("blend(") else ms._spec(key)


def current(conn: sqlite3.Connection) -> Optional[sqlite3.Row]:
    _init(conn)
    return conn.execute("SELECT * FROM fund_agents WHERE died IS NULL ORDER BY id DESC LIMIT 1").fetchone()


def current_specs(conn: sqlite3.Connection) -> Dict[int, Dict[str, Any]]:
    a = current(conn)
    if not a:
        return {}
    cfg = json.loads(a["config"])
    return {int(h): _spec_of(k) for h, k in cfg.items() if k}


def first_config(boards: Dict[int, List[Dict[str, Any]]]) -> Dict[str, str]:
    return {str(h): (b[0]["key"] if b else None) for h, b in boards.items()}


def successor_config(boards: Dict[int, List[Dict[str, Any]]], used: List[Dict[str, str]], dead: Dict[str, str],
                     worst_horizon: Optional[int]) -> Dict[str, str]:
    """Next best untried setup per horizon; for the sleeve that lost most, a different model family."""
    out = {}
    for h, board in boards.items():
        tried = {u.get(str(h)) for u in used}
        avoid = family(dead.get(str(h))) if h == worst_horizon else None
        pick = next((r["key"] for r in board if r["key"] not in tried and family(r["key"]) != avoid), None)
        pick = pick or next((r["key"] for r in board if r["key"] not in tried), None) or dead.get(str(h))
        out[str(h)] = pick
    return out


def ensure_agent(conn: sqlite3.Connection, born: str, boards: Optional[Dict[int, List[Dict[str, Any]]]] = None) -> sqlite3.Row:
    """The live fund's first agent is born on the first live evening."""
    a = current(conn)
    if a:
        return a
    boards = boards or {h: _leaderboard(h) for h in set(SLEEVES.values())}
    w = {s: round(1 / len(SLEEVES), 4) for s in SLEEVES}
    conn.execute("INSERT INTO fund_agents (born, config, weights, start_value) VALUES (?,?,?,?)",
                 (born, json.dumps(first_config(boards)), json.dumps(w), START_CASH))
    return current(conn)


def fund_value(conn: sqlite3.Connection) -> float:
    r = conn.execute("SELECT end_value FROM fund_ledger ORDER BY date DESC LIMIT 1").fetchone()
    return r["end_value"] if r else START_CASH


def reweight(conn: sqlite3.Connection, upto: str) -> Dict[str, float]:
    """Weights ∝ each sleeve's return per unit of risk over the last LOOKBACK settled sessions
    (only positive ones count), then held between FLOOR and CAP."""
    led = pd.read_sql("SELECT date, book, start_value, pnl FROM daily_ledger WHERE date <= ?", conn, params=(upto,))
    raw = {}
    for s in SLEEVES:
        l = led[led["book"] == s].sort_values("date").tail(LOOKBACK)
        r = (l["pnl"] / l["start_value"]).to_numpy() if len(l) else np.array([])
        raw[s] = max(float(r.mean() / r.std()), 0.0) if len(r) >= 5 and r.std() > 0 else 0.0
    return {s: round(v, 4) for s, v in bounded(raw).items()}


def bounded(raw: Dict[str, float]) -> Dict[str, float]:
    """Split 100% in proportion to `raw`, no sleeve above CAP or below FLOOR: sleeves that hit a
    bound are fixed there and the rest is shared by the others in proportion."""
    fixed: Dict[str, float] = {}
    for _ in range(len(raw)):
        free = [s for s in raw if s not in fixed]
        rem = 1.0 - sum(fixed.values())
        tot = sum(raw[s] for s in free)
        cand = {s: rem * (raw[s] / tot if tot > 0 else 1 / len(free)) for s in free}
        viol = {s: (CAP if v > CAP else FLOOR) for s, v in cand.items() if v > CAP or v < FLOOR}
        if not viol:
            fixed.update(cand)
            break
        fixed.update(viol)
    t = sum(fixed.values())
    return {s: fixed[s] / t for s in raw}


def settle(conn: sqlite3.Connection, day: str, replayed: int,
           boards: Optional[Dict[int, List[Dict[str, Any]]]] = None) -> Optional[Dict[str, Any]]:
    """Book the fund's day from its sleeves' settled returns; apply penalties; maybe die."""
    if replayed:
        return None                                              # the fund is live-only
    a = current(conn)
    if not a or a["born"] >= day or conn.execute("SELECT 1 FROM fund_ledger WHERE date = ?", (day,)).fetchone():
        return None
    w = json.loads(a["weights"])
    rows = {r["book"]: r for r in conn.execute("SELECT book, start_value, pnl FROM daily_ledger WHERE date = ?", (day,))}
    sr = {s: (rows[s]["pnl"] / rows[s]["start_value"]) if s in rows and rows[s]["start_value"] else 0.0 for s in SLEEVES}
    ret = sum(w[s] * sr[s] for s in SLEEVES)
    start = fund_value(conn)
    end = start * (1 + ret)
    pen = max(0, a["penalties"] + (1 if ret < 0 else -1 if ret > 0 else 0))
    streak = a["streak"] + 1 if ret < 0 else 0
    days = a["days"] + 1
    event = ""
    conn.execute("UPDATE fund_agents SET days=?, penalties=?, streak=?, end_value=? WHERE id=?", (days, pen, streak, end, a["id"]))
    if days % REWEIGHT_EVERY == 0:
        nw = reweight(conn, day)
        conn.execute("UPDATE fund_agents SET weights=? WHERE id=?", (json.dumps(nw), a["id"]))
        event = "reweighted " + ", ".join(f"{k} {v:.0%}" for k, v in nw.items())
    conn.execute("INSERT INTO fund_ledger VALUES (?,?,?,?,?,?,?,?,?,?)",
                 (day, a["id"], start, end, ret, json.dumps(w), json.dumps(sr), pen, streak, event))
    if pen >= PENALTY_LIMIT or streak >= STREAK_LIMIT:
        cause = f"{pen} net penalties" if pen >= PENALTY_LIMIT else f"{streak} losing days in a row"
        die(conn, a["id"], day, cause, boards)
        event = (event + "; " if event else "") + f"agent #{a['id']} died: {cause}"
        conn.execute("UPDATE fund_ledger SET event=? WHERE date=?", (event, day))
    return {"ret": ret, "value": end, "penalties": pen, "streak": streak, "event": event}


def die(conn: sqlite3.Connection, agent_id: int, day: str, cause: str,
        boards: Optional[Dict[int, List[Dict[str, Any]]]] = None) -> int:
    """Post-mortem, then a successor with the next best untried setups (money carries over)."""
    a = conn.execute("SELECT * FROM fund_agents WHERE id = ?", (agent_id,)).fetchone()
    life = pd.read_sql("SELECT ret, weights, sleeve_rets FROM fund_ledger WHERE agent = ?", conn, params=(agent_id,))
    contrib = {s: 0.0 for s in SLEEVES}
    for _, r in life.iterrows():
        wts, sr = json.loads(r["weights"]), json.loads(r["sleeve_rets"])
        for s in SLEEVES:
            contrib[s] += wts.get(s, 0) * sr.get(s, 0)
    worst = min(contrib, key=contrib.get)
    pm = {"days": a["days"], "return": round((a["end_value"] or a["start_value"]) / a["start_value"] - 1, 4),
          "sleeve_contribution": {s: round(v, 4) for s, v in contrib.items()}, "worst_sleeve": worst,
          "lesson": f"{worst} sleeve lost most; its model family is swapped out for the next agent"}
    conn.execute("UPDATE fund_agents SET died=?, cause=?, postmortem=? WHERE id=?", (day, cause, json.dumps(pm), agent_id))
    boards = boards or {h: _leaderboard(h) for h in set(SLEEVES.values())}
    used = [json.loads(r["config"]) for r in conn.execute("SELECT config FROM fund_agents")]
    cfg = successor_config(boards, used, json.loads(a["config"]), SLEEVES[worst])
    w = reweight(conn, day)
    conn.execute("INSERT INTO fund_agents (born, config, weights, start_value) VALUES (?,?,?,?)",
                 (day, json.dumps(cfg), json.dumps(w), a["end_value"] or a["start_value"]))
    return conn.execute("SELECT MAX(id) FROM fund_agents").fetchone()[0]


def report(conn: sqlite3.Connection) -> Dict[str, Any]:
    _init(conn)
    agents = [dict(r) for r in conn.execute("SELECT * FROM fund_agents ORDER BY id")]
    ledger = [dict(r) for r in conn.execute("SELECT * FROM fund_ledger ORDER BY date")]
    return {"start": START_CASH, "goal": GOAL, "value": fund_value(conn), "agents": agents, "ledger": ledger,
            "rules": {"sleeves": SLEEVES, "floor": FLOOR, "cap": CAP, "reweight_every": REWEIGHT_EVERY,
                      "lookback": LOOKBACK, "penalty_limit": PENALTY_LIMIT, "streak_limit": STREAK_LIMIT}}
