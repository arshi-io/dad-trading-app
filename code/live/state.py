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


# User preferences: (default, parser, check, what the Settings page says when the check fails).
PREFS = {
    "paper_capital": ("1000000", int, lambda v: 100_000 <= v <= 100_000_000, "Account size must be between ₹1 lakh and ₹10 crore."),
    "default_exit_rule": ("trail21", str, lambda v: v in ("trail21", "target"), "Unknown exit rule."),
    "beginner_mode": ("1", str, lambda v: v in ("0", "1"), "Beginner hints must be on or off."),
}


def pref(key: str) -> Any:
    default, parse, check, _ = PREFS[key]
    try:
        value = parse(_get(f"pref:{key}", default))
        return value if check(value) else parse(default)
    except Exception:
        return parse(default)


def set_pref(key: str, raw: str, actor: str) -> Any:
    _, parse, check, problem = PREFS[key]
    try:
        value = parse(str(raw).strip().replace(",", ""))
    except ValueError:
        raise ValueError(problem)
    if not check(value):
        raise ValueError(problem)
    old = pref(key)
    if value != old:
        _set(f"pref:{key}", str(value), actor)
        audit.record("setting_changed", actor, "app_state", key, {"from": old, "to": value})
    return value
