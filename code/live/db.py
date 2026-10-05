"""Versioned schema for the market-hours features, in the same SQLite file as the rest of the app.

Migrations are append-only: add a new (version, sql) pair, never edit an applied one.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "papa.db"

MIGRATIONS: List[Tuple[int, str]] = [
    (1, """
    CREATE TABLE IF NOT EXISTS audit_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        actor TEXT NOT NULL,
        action TEXT NOT NULL,
        entity TEXT,
        entity_id TEXT,
        detail TEXT,
        prev_hash TEXT NOT NULL,
        hash TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS app_state (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        updated_by TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        symbol TEXT NOT NULL,
        kind TEXT NOT NULL,
        severity TEXT NOT NULL DEFAULT 'info',
        title TEXT NOT NULL,
        body TEXT,
        price REAL,
        dedupe_key TEXT UNIQUE,
        status TEXT NOT NULL DEFAULT 'new',
        telegram_status TEXT,
        push_status TEXT
    );
    CREATE TABLE IF NOT EXISTS proposed_orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT NOT NULL,
        symbol TEXT NOT NULL,
        side TEXT NOT NULL,
        qty INTEGER NOT NULL,
        order_type TEXT NOT NULL,
        limit_price REAL,
        trigger_price REAL,
        stop_loss REAL NOT NULL,
        target REAL,
        source_alert_id INTEGER REFERENCES alerts(id),
        status TEXT NOT NULL DEFAULT 'proposed',
        decided_at TEXT,
        decided_by TEXT,
        reject_reason TEXT,
        broker_order_id INTEGER
    );
    CREATE TABLE IF NOT EXISTS broker_orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        proposed_order_id INTEGER NOT NULL REFERENCES proposed_orders(id),
        broker TEXT NOT NULL,
        broker_ref TEXT,
        status TEXT NOT NULL,
        filled_qty INTEGER NOT NULL DEFAULT 0,
        avg_price REAL,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        raw TEXT
    );
    CREATE TABLE IF NOT EXISTS risk_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        kind TEXT NOT NULL,
        detail TEXT,
        proposed_order_id INTEGER
    );
    CREATE TABLE IF NOT EXISTS push_subscriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        endpoint TEXT NOT NULL UNIQUE,
        keys TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS ix_alerts_created ON alerts(created_at);
    CREATE INDEX IF NOT EXISTS ix_proposed_status ON proposed_orders(status);
    """),
]


def connect(path: Path | None = None) -> sqlite3.Connection:
    path = path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def migrate(path: Path | None = None) -> List[int]:
    """Apply pending migrations; returns the versions applied on this call."""
    conn = connect(path)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
        done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
        applied = []
        for version, sql in MIGRATIONS:
            if version in done:
                continue
            conn.executescript(sql)
            conn.execute("INSERT INTO schema_migrations VALUES (?, ?)", (version, datetime.now().isoformat()))
            conn.commit()
            applied.append(version)
        return applied
    finally:
        conn.close()
