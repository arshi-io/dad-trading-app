"""Choice FinX: key status only, until Choice's official API reference is in hand.

The API key is a JWT issued by FinX. Its payload (not secret, just signed) says whose key it is,
when it expires and which IP it's locked to -- enough to tell the user whether a login would
even be accepted from this server. No Choice endpoints are called: none are guessed.
"""
from __future__ import annotations

import base64
import ipaddress
import json
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional

import requests

IST = timezone(timedelta(hours=5, minutes=30))
_ip_cache: Dict[str, Any] = {"at": 0.0, "ips": {}}


def key_info(token: str) -> Optional[Dict[str, Any]]:
    """Read the JWT payload without verifying it (verification is Choice's job, server-side)."""
    try:
        payload = token.split(".")[1]
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
    except (IndexError, ValueError):
        return None
    exp = datetime.fromtimestamp(int(data.get("exp", 0)), IST) if data.get("exp") else None
    return {
        "user": data.get("UserId"),
        "issuer": data.get("iss"),
        "issued": datetime.fromtimestamp(int(data["iat"]), IST) if data.get("iat") else None,
        "expires": exp,
        "days_left": (exp - datetime.now(IST)).days if exp else None,
        "expired": bool(exp and exp <= datetime.now(IST)),
        "locked_ip": data.get("CliIPAddress"),
    }


def public_ips(max_age: float = 600) -> Dict[str, Optional[str]]:
    """This server's public IPv4 and IPv6 as the internet sees them (cached 10 min)."""
    if time.time() - _ip_cache["at"] < max_age and _ip_cache["ips"]:
        return _ip_cache["ips"]
    ips: Dict[str, Optional[str]] = {"ipv4": None, "ipv6": None}
    for kind, url in (("ipv4", "https://api.ipify.org"), ("ipv6", "https://api64.ipify.org")):
        try:
            ip = requests.get(url, timeout=4).text.strip()
            if ipaddress.ip_address(ip).version == (4 if kind == "ipv4" else 6):
                ips[kind] = ip
        except Exception:
            pass
    _ip_cache.update(at=time.time(), ips=ips)
    return ips


def ip_matches(locked: Optional[str], ips: Dict[str, Optional[str]]) -> Optional[bool]:
    if not locked:
        return None
    try:
        want = ipaddress.ip_address(locked)
    except ValueError:
        return None
    have = ips.get("ipv6" if want.version == 6 else "ipv4")
    return None if have is None else ipaddress.ip_address(have) == want


_client = None
_last_error = ""


def client():
    """The one read-only Choice session for this process (None until key, vendor id and mobile are set)."""
    global _client
    from code.live.brokers.choice_client import ChoiceClient
    needed = ("CHOICE_API_KEY", "CHOICE_VENDOR_ID", "CHOICE_MOBILE")
    if _client is None and all(os.getenv(k) for k in needed):
        _client = ChoiceClient(
            api_key=os.getenv("CHOICE_API_KEY", ""), vendor_id=os.getenv("CHOICE_VENDOR_ID", ""),
            mobile=os.getenv("CHOICE_MOBILE", ""), env=os.getenv("CHOICE_ENV", "live").lower(),
            vendor_key=os.getenv("CHOICE_VENDOR_KEY", ""),
            aes_key=os.getenv("CHOICE_AES_KEY", ""), aes_iv=os.getenv("CHOICE_AES_IV", ""),
        )
    return _client


def connect(actor: str) -> Dict[str, Any]:
    """Automatic login (no password, no SMS to type). Audited either way."""
    global _last_error
    from code.live import audit
    c = client()
    if c is None:
        return {"ok": False, "message": "Add CHOICE_VENDOR_ID and CHOICE_MOBILE to .env, then restart."}
    try:
        c.login()
        c.load_scrips()
        _last_error = ""
        audit.record("broker_connected", actor, "broker", "choice", {"base": c.active_base})
        return {"ok": True, "message": "Connected to Choice FinX."}
    except Exception as exc:
        _last_error = str(exc)[:300]
        audit.record("broker_connect_failed", actor, "broker", "choice", {"error": _last_error})
        return {"ok": False, "message": f"Choice login failed: {_last_error}"}


def auto_connect() -> Optional[Dict[str, Any]]:
    c = client()
    if c is None or c.connected:
        return None
    return connect("scheduler")


def disconnect(actor: str) -> None:
    from code.live import audit
    c = client()
    if c is not None:
        c.logout()
        audit.record("broker_disconnected", actor, "broker", "choice", {})


def status() -> Dict[str, Any]:
    st = _key_status()
    c = client()
    st.update({
        "ready": c is not None,
        "connected": bool(c and c.connected),
        "since": c.logged_in_at if c and c.logged_in_at else None,
        "scrips": len(c.scrips) if c else 0,
        "last_error": _last_error,
        "raw_sample": json.dumps(c.last_raw)[:800] if c and c.last_raw else "",
        "missing": [k for k in ("CHOICE_VENDOR_ID", "CHOICE_MOBILE") if not os.getenv(k)],
    })
    return st


def _key_status() -> Dict[str, Any]:
    token = os.getenv("CHOICE_API_KEY", "")
    if not token:
        return {"configured": False}
    info = key_info(token)
    if info is None:
        return {"configured": True, "valid_format": False}
    ips = public_ips()
    locked = info["locked_ip"]
    return {
        "configured": True,
        "valid_format": True,
        **info,
        "server_ips": ips,
        "ip_match": ip_matches(locked, ips),
        "locked_is_ipv6": bool(locked and ":" in locked),
    }
