"""App mode and the emergency halt switch. Every change is audited; reads are cheap."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict
from zoneinfo import ZoneInfo

from code.app import settings
from code.live import audit, db

IST = ZoneInfo("Asia/Kolkata")


def _get(key: str, default: str) -> str:
    conn = db.connect()
    try:
        row = conn.execute("SELECT value FROM app_state WHERE key=?", (key,)).fetchone()
        return row["value"] if row else default
    finally:
        conn.close()


def _set(key: str, value: str, actor: str) -> None:
    conn = db.connect()
    try:
        conn.execute(
            "INSERT INTO app_state (key, value, updated_at, updated_by) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at, updated_by=excluded.updated_by",
            (key, value, datetime.now(IST).isoformat(), actor),
        )
        conn.commit()
    finally:
        conn.close()


def mode() -> str:
    value = _get("app_mode", settings.APP_MODE)
    return value if value in settings.APP_MODES else "PAPER_TRADING"


def set_mode(new_mode: str, actor: str) -> str:
    new_mode = new_mode.upper()
    if new_mode not in settings.APP_MODES:
        raise ValueError(f"mode must be one of {settings.APP_MODES}")
    old = mode()
    _set("app_mode", new_mode, actor)
    audit.record("mode_changed", actor, "app_state", "app_mode", {"from": old, "to": new_mode})
    return new_mode


def is_halted() -> bool:
    return _get("halted", "0") == "1"


def set_halt(on: bool, actor: str, reason: str = "") -> bool:
    _set("halted", "1" if on else "0", actor)
    audit.record("halt_on" if on else "halt_off", actor, "app_state", "halted", {"reason": reason})
    return on


def snapshot() -> Dict[str, Any]:
    return {"mode": mode(), "halted": is_halted()}
