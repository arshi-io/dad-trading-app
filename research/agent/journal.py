"""Research journal (PRD F3): one SQLite row per idea per stock. Rows are never deleted."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

JOURNAL_PATH = Path(__file__).with_name("journal.db")
JSON_COLS = ("parents", "backtest", "checks", "critic_reasons", "cloud_reasons", "gate", "models")

SCHEMA = """
CREATE TABLE IF NOT EXISTS ideas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    session TEXT NOT NULL,
    symbol TEXT NOT NULL,
    family TEXT NOT NULL,
    kind TEXT NOT NULL,              -- single | ensemble
    parents TEXT,                    -- ensemble member ids
    idea TEXT, source TEXT, hypothesis TEXT, rule TEXT, code TEXT,
    attempt_no INTEGER,              -- how many times this family has been tried, this one included
    backtest TEXT,                   -- Period A / B stats, yearly returns, lookahead result
    checks TEXT,                     -- automatic critic checks that failed
    critic_verdict TEXT, critic_reasons TEXT,
    cloud_verdict TEXT, cloud_reasons TEXT,
    status TEXT NOT NULL,            -- built | error | critic_failed | awaiting_cloud | cloud_failed | gated
    gate TEXT, gate_passed INTEGER,
    models TEXT NOT NULL             -- {"librarian": {"model":..,"provider":..}, ...}
);
CREATE TRIGGER IF NOT EXISTS ideas_keep BEFORE DELETE ON ideas
BEGIN SELECT RAISE(ABORT, 'journal rows are never deleted'); END;
"""


def connect(path: Path = JOURNAL_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def add(conn: sqlite3.Connection, row: Dict[str, Any]) -> int:
    row = {"created_at": datetime.now().isoformat(timespec="seconds"), **row}
    row = {k: json.dumps(v) if k in JSON_COLS and v is not None else v for k, v in row.items()}
    cur = conn.execute(f"INSERT INTO ideas ({','.join(row)}) VALUES ({','.join('?' * len(row))})", list(row.values()))
    conn.commit()
    return cur.lastrowid


def update(conn: sqlite3.Connection, idea_id: int, **fields: Any) -> None:
    fields = {k: json.dumps(v) if k in JSON_COLS and v is not None else v for k, v in fields.items()}
    conn.execute(f"UPDATE ideas SET {','.join(f'{k}=?' for k in fields)} WHERE id=?", [*fields.values(), idea_id])
    conn.commit()


def rows(conn: sqlite3.Connection, where: str = "1", args: tuple = ()) -> List[Dict[str, Any]]:
    out = []
    for r in conn.execute(f"SELECT * FROM ideas WHERE {where} ORDER BY id", args):
        d = dict(r)
        for k in JSON_COLS:
            if d.get(k) is not None:
                d[k] = json.loads(d[k])
        out.append(d)
    return out


def attempts(conn: sqlite3.Connection, family: str) -> int:
    """Distinct strategies tried in a family (one strategy tested on 3 stocks is one attempt)."""
    return conn.execute("SELECT COUNT(DISTINCT code) FROM ideas WHERE family=?", (family,)).fetchone()[0]


def pass_rate_by_model(conn: sqlite3.Connection, role: str) -> List[Dict[str, Any]]:
    """Which model in a role produced ideas that reached and passed the gate."""
    q = f"""SELECT json_extract(models, '$.{role}.model') AS model, COUNT(*) AS ideas,
                   SUM(critic_verdict = 'pass') AS critic_passed,
                   SUM(status = 'gated') AS gated, SUM(COALESCE(gate_passed, 0)) AS gate_passed
            FROM ideas GROUP BY model ORDER BY gate_passed DESC, ideas DESC"""
    return [dict(r) for r in conn.execute(q)]


def summary(conn: sqlite3.Connection, limit: int = 40) -> str:
    """Short plain-text view of recent attempts for the librarian."""
    lines = []
    for r in conn.execute("SELECT id, symbol, family, hypothesis, status, critic_verdict, gate_passed "
                          "FROM ideas ORDER BY id DESC LIMIT ?", (limit,)):
        gate = {1: "gate PASS", 0: "gate fail"}.get(r["gate_passed"], "")
        lines.append(f"#{r['id']} {r['symbol']} [{r['family']}] {r['hypothesis']} -> {r['status']} {gate}".strip())
    return "\n".join(lines) or "(journal is empty)"
