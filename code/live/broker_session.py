"""The one Motilal session for this process: built from env, connected daily, never persisted.

Two ways to connect (tokens expire daily):
  * auto -- MOTILAL_PASSWORD + MOTILAL_TOTP_SECRET in the server's environment; secrets never
    leave the server. Runs at startup and before the session opens.
  * manual -- password + 6-digit authenticator code typed on the Broker page.
Either way the password is used once to log in and is not stored.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

from code.live import audit
from code.live.brokers.motilal import MotilalClient, MotilalError, totp_now

logger = logging.getLogger(__name__)

_client: Optional[MotilalClient] = None
_last_error: str = ""


def configured() -> bool:
    return bool(os.getenv("MOTILAL_API_KEY") and os.getenv("MOTILAL_USERID"))


def client() -> Optional[MotilalClient]:
    global _client
    if _client is None and configured():
        _client = MotilalClient(
            api_key=os.getenv("MOTILAL_API_KEY", ""),
            api_secret=os.getenv("MOTILAL_API_SECRET", ""),
            env=os.getenv("MOTILAL_ENV", "live").lower(),
            vendor_info=os.getenv("MOTILAL_USERID", ""),
            client_code=os.getenv("MOTILAL_CLIENT_CODE", ""),   # dealers only
        )
    return _client


def connect(password: str, totp: str, actor: str) -> Dict[str, Any]:
    global _last_error
    c = client()
    if c is None:
        return {"ok": False, "message": "Motilal isn't configured — set MOTILAL_API_KEY and MOTILAL_USERID in .env."}
    try:
        c.login(os.getenv("MOTILAL_USERID", ""), password, os.getenv("MOTILAL_2FA", ""), totp)
        _last_error = ""
        audit.record("broker_connected", actor, "broker", "motilal", {"env": os.getenv("MOTILAL_ENV", "live")})
        return {"ok": True, "message": "Connected to Motilal Oswal."}
    except (MotilalError, Exception) as exc:  # network errors included; message only, never the password
        _last_error = str(exc)
        audit.record("broker_connect_failed", actor, "broker", "motilal", {"error": _last_error[:200]})
        return {"ok": False, "message": f"Login failed: {_last_error}"}


def auto_connect() -> Optional[Dict[str, Any]]:
    """Log in from server-side secrets if they're configured and we aren't connected."""
    c = client()
    password, secret = os.getenv("MOTILAL_PASSWORD", ""), os.getenv("MOTILAL_TOTP_SECRET", "")
    if c is None or c.connected or not (password and secret):
        return None
    return connect(password, totp_now(secret), actor="scheduler")


def disconnect(actor: str) -> None:
    c = client()
    if c is not None:
        c.logout()
        audit.record("broker_disconnected", actor, "broker", "motilal", {})


def status() -> Dict[str, Any]:
    c = client()
    return {
        "configured": configured(),
        "connected": bool(c and c.connected),
        "since": c.logged_in_at.isoformat() if c and c.logged_in_at else None,
        "env": os.getenv("MOTILAL_ENV", "live").lower(),
        "auto_login": bool(os.getenv("MOTILAL_PASSWORD") and os.getenv("MOTILAL_TOTP_SECRET")),
        "last_error": _last_error,
    }
