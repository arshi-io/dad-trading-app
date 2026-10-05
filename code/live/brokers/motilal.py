"""Read-only Motilal Oswal OpenAPI client: login, last-traded prices, instrument master.

Built from Motilal's official Python SDK (github.com/motradingapi/PythonSDK, "Python 5.0",
01/09/2026) -- same endpoints, payloads and header names -- without its install footprint
(it pins urllib3 1.26 and Windows-only packages, and geolocates the machine on import).

There is deliberately NO order placement here. Orders go through the approval service and a
separately reviewed adapter (docs/RISK_POLICY.md), never through the market-data client.

Unverified against a live account (no credentials yet): response field names beyond
status/message/errorcode/AuthToken/data are taken from the official docs as cited by the
open-source OpenAlgo integration; tests pin that behaviour so a mismatch fails loudly.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import struct
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import requests

logger = logging.getLogger(__name__)

BASE_URLS = {"live": "https://openapi.motilaloswal.com", "uat": "https://openapi.motilaloswaluat.com"}
PATHS = {
    "login": "/rest/login/v7/authdirectapi",
    "access_token": "/rest/login/v1/getaccesstoken",
    "logout": "/rest/login/v5/logout",
    "ltp": "/rest/report/v3/getltpdata",
    "instruments": "/rest/report/v3/getscripsbyexchangename",
}
INSTRUMENT_CACHE = Path(__file__).resolve().parents[2] / "data" / "motilal_nse_instruments.json"
INSTRUMENT_TTL = timedelta(days=1)
TIMEOUT = 10


class MotilalError(RuntimeError):
    def __init__(self, message: str, errorcode: str = ""):
        super().__init__(f"{message} ({errorcode})" if errorcode else message)
        self.errorcode = errorcode


def totp_now(secret_b32: str, at: Optional[float] = None, step: int = 30, digits: int = 6) -> str:
    """RFC 6238 TOTP -- the same 6-digit code an authenticator app shows for this secret."""
    key = base64.b32decode(secret_b32.replace(" ", "").upper() + "=" * (-len(secret_b32.replace(" ", "")) % 8))
    counter = int((time.time() if at is None else at) // step)
    digest = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    offset = digest[-1] & 0x0F
    code = (struct.unpack(">I", digest[offset:offset + 4])[0] & 0x7FFFFFFF) % (10 ** digits)
    return str(code).zfill(digits)


class MotilalClient:
    def __init__(self, api_key: str, api_secret: str = "", env: str = "live", vendor_info: str = "",
                 client_code: str = "", post: Callable[..., Any] = requests.post):
        if env not in BASE_URLS:
            raise ValueError(f"env must be one of {sorted(BASE_URLS)}")
        self.base = BASE_URLS[env]
        self.api_key = api_key
        self.api_secret = api_secret
        self.vendor_info = vendor_info
        self.client_code = client_code      # dealers only
        self._post = post
        self._lock = threading.Lock()
        self.auth_token = ""
        self.access_token = ""
        self.logged_in_at: Optional[datetime] = None

    # -- transport ---------------------------------------------------------------------------
    def _headers(self, with_auth: bool = True) -> Dict[str, str]:
        # Header names per Motilal's documented common headers (SDK validate()).
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "MOSL/V.1.1.0",
            "ApiKey": self.api_key,
            "ClientLocalIp": "127.0.0.1",
            "ClientPublicIp": "127.0.0.1",
            "MacAddress": "00:00:00:00:00:00",
            "SourceId": "WEB",
            "vendorinfo": self.vendor_info,
            "osname": "Linux",
            "osversion": "1.0",
            "devicemodel": "server",
            "manufacturer": "MarketPilot",
            "productname": "MarketPilot",
            "productversion": "1.0",
            "browsername": "Chrome",
            "browserversion": "120.0",
        }
        if self.api_secret:
            h["apisecretkey"] = self.api_secret
        if with_auth and self.auth_token:
            h["Authorization"] = self.auth_token
            if self.access_token:
                h["accesstoken"] = self.access_token
        return h

    def _call(self, name: str, body: Dict[str, Any], with_auth: bool = True) -> Dict[str, Any]:
        resp = self._post(self.base + PATHS[name], headers=self._headers(with_auth), data=json.dumps(body), timeout=TIMEOUT)
        try:
            data = resp.json()
        except ValueError:
            raise MotilalError(f"{name}: non-JSON response (HTTP {getattr(resp, 'status_code', '?')})")
        return data

    def _report(self, name: str, body: Dict[str, Any]) -> Dict[str, Any]:
        """Report calls take clientcode only for dealer logins; a plain client that sends it gets
        MO2031, a dealer that omits it gets MO1062 -- so retry once with it on MO1062."""
        if self.client_code:
            body = {**body, "clientcode": self.client_code}
        data = self._call(name, body)
        if data.get("status") != "SUCCESS" and str(data.get("errorcode", "")).upper() == "MO1062" and self.vendor_info:
            self.client_code = self.vendor_info
            data = self._call(name, {**body, "clientcode": self.client_code})
        if data.get("status") != "SUCCESS":
            raise MotilalError(data.get("message", f"{name} failed"), str(data.get("errorcode", "")))
        return data

    # -- session -----------------------------------------------------------------------------
    @property
    def connected(self) -> bool:
        # Motilal expires the AuthToken daily; treat anything from before today's 06:00 as gone.
        if not (self.auth_token and self.logged_in_at):
            return False
        now = datetime.now()
        cutoff = now.replace(hour=6, minute=0, second=0, microsecond=0)
        if now < cutoff:
            cutoff -= timedelta(days=1)
        return self.logged_in_at >= cutoff

    def login(self, userid: str, password: str, two_fa: str, totp: str = "") -> Dict[str, Any]:
        """Password is hashed client-side as SHA-256(password + api_key), as the SDK does; it is
        never stored or logged."""
        with self._lock:
            body = {"userid": userid, "password": hashlib.sha256((password + self.api_key).encode()).hexdigest(), "2FA": two_fa}
            if totp:
                body["totp"] = totp
            self.vendor_info = self.vendor_info or userid
            data = self._call("login", body, with_auth=False)
            if data.get("status") != "SUCCESS" or not data.get("AuthToken"):
                raise MotilalError(data.get("message", "login failed"), str(data.get("errorcode", "")))
            self.auth_token = data["AuthToken"]
            self.logged_in_at = datetime.now()
            try:
                tok = self._call("access_token", {"clientcode": ""})
                if tok.get("status") == "SUCCESS" and tok.get("accesstoken"):
                    self.access_token = tok["accesstoken"]
            except Exception as exc:  # optional second token; the AuthToken alone is enough for reports
                logger.info("Motilal access token not issued: %s", exc)
            return {"status": "SUCCESS", "verified": str(data.get("isAuthTokenVerified", "")).upper() == "TRUE"}

    def logout(self) -> None:
        with self._lock:
            if self.auth_token:
                try:
                    self._call("logout", {"clientcode": self.client_code or None})
                except Exception as exc:
                    logger.info("Motilal logout call failed: %s", exc)
            self.auth_token = self.access_token = ""
            self.logged_in_at = None

    # -- market data -------------------------------------------------------------------------
    def ltp(self, scripcode: int, exchange: str = "NSE") -> Dict[str, float]:
        """Last traded price and today's OHLC/volume, converted from paise to rupees (Motilal
        returns open/high/low/close/ltp in paise; 'close' is the previous close)."""
        data = self._report("ltp", {"exchange": exchange, "scripcode": int(scripcode)}).get("data") or {}
        if not data:
            raise MotilalError(f"no LTP data for {exchange}:{scripcode}")
        paise = lambda k: float(data.get(k) or 0) / 100.0  # noqa: E731
        return {"ltp": paise("ltp"), "open": paise("open"), "high": paise("high"), "low": paise("low"),
                "prev_close": paise("close"), "volume": int(data.get("volume") or 0)}

    def nse_equity_codes(self, cache_path: Path = INSTRUMENT_CACHE) -> Dict[str, int]:
        """{"RELIANCE": 2885, ...} for NSE cash, preferring the regular EQ series; cached daily."""
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if datetime.now() - datetime.fromisoformat(cached["fetched_at"]) < INSTRUMENT_TTL:
                return cached["codes"]
        rows = self._report("instruments", {"exchangename": "NSE"}).get("data") or []
        codes = parse_nse_equity_codes(rows)
        if codes:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps({"fetched_at": datetime.now().isoformat(), "codes": codes}), encoding="utf-8")
        return codes


def parse_nse_equity_codes(rows) -> Dict[str, int]:
    """NSE cash rows carry the bare symbol in `scripshortname` and the series in `optiontype`.
    When a scrip is listed in several series, the EQ row wins."""
    out: Dict[str, int] = {}
    series_of: Dict[str, str] = {}
    for r in rows:
        sym = str(r.get("scripshortname") or "").strip().upper()
        code = r.get("scripcode")
        if not sym or code in (None, ""):
            continue
        series = str(r.get("optiontype") or "").strip().upper()
        if sym not in out or (series == "EQ" and series_of.get(sym) != "EQ"):
            out[sym] = int(code)
            series_of[sym] = series
    return out
