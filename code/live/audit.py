"""Append-only audit log. Each row carries a hash of the previous row, so an edited or deleted
entry breaks the chain and `verify()` says where."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from code.live import db

IST = ZoneInfo("Asia/Kolkata")
GENESIS = "0" * 64


def _digest(prev_hash: str, ts: str, actor: str, action: str, entity: str, entity_id: str, detail: str) -> str:
    return hashlib.sha256("|".join([prev_hash, ts, actor, action, entity, entity_id, detail]).encode()).hexdigest()


def record(action: str, actor: str = "system", entity: str = "", entity_id: Any = "",
           detail: Optional[Dict[str, Any]] = None) -> int:
    ts = datetime.now(IST).isoformat()
    detail_s = json.dumps(detail or {}, sort_keys=True, default=str)
    conn = db.connect()
    try:
        conn.execute("BEGIN IMMEDIATE")  # serialises writers so the chain can't fork
        row = conn.execute("SELECT hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
        prev = row["hash"] if row else GENESIS
        h = _digest(prev, ts, actor, action, entity, str(entity_id), detail_s)
        cur = conn.execute(
            "INSERT INTO audit_log (ts, actor, action, entity, entity_id, detail, prev_hash, hash) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, actor, action, entity, str(entity_id), detail_s, prev, h),
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def recent(limit: int = 100) -> List[Dict[str, Any]]:
    conn = db.connect()
    try:
        rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) | {"detail": json.loads(r["detail"] or "{}")} for r in rows]
    finally:
        conn.close()


def verify() -> Dict[str, Any]:
    """Recompute the whole chain. Returns {"ok": True, "rows": n} or the first broken row id."""
    conn = db.connect()
    try:
        prev = GENESIS
        n = 0
        for r in conn.execute("SELECT * FROM audit_log ORDER BY id"):
            expected = _digest(prev, r["ts"], r["actor"], r["action"], r["entity"] or "", r["entity_id"] or "", r["detail"] or "")
            if r["prev_hash"] != prev or r["hash"] != expected:
                return {"ok": False, "rows": n, "broken_at": r["id"]}
            prev = r["hash"]
            n += 1
        return {"ok": True, "rows": n}
    finally:
        conn.close()
