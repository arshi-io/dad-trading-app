"""Read-only Choice FinX client: login, live touchline quotes, NSE symbol → token.

Request side follows Choice's official OpenAPI spec (finx.choiceindia.com/api/OpenAPI/Info,
"OpenAPIInterface_API V1") and integration guide (May 2026). Where the spec is silent, the
behaviour of the community client github.com/SomeshD24/Kkunal (updated 22 Sep 2026) is followed
and marked: the API key travels in a header literally named "Bearer"; without an issued AES key
the mobile number is sent base64-encoded; the daily scrip master is public on
scripmaster.choiceindia.com.

No order, payment, payout or DIS endpoints exist in this module.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import requests

logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))
BASE_URLS = {"live": "https://finx.choiceindia.com", "uat": "https://uat.jiffy.in"}
SCRIP_MASTER_URL = "https://scripmaster.choiceindia.com/scripmaster/SCRIP_MASTER_{d}.csv"
SCRIP_CACHE = Path(__file__).resolve().parents[2] / "data" / "choice_nse_scrips.json"
NSE_CASH = 1
TOUCHLINE_BATCH = 50

# Quote fields by the names Choice's own price feed uses (FIX tags 8/75/76/77/78/79). The REST
# touchline's keys aren't in the spec's (empty) schema, so a parse miss keeps the raw response for
# the Broker page rather than guessing a price.
FIELD_CANDIDATES = {
    "ltp": ("LTP", "Ltp", "LastTradedPrice", "LastRate"),
    "open": ("Open", "OpenPrice"),
    "high": ("High", "HighPrice"),
    "low": ("Low", "LowPrice"),
    "prev_close": ("Close", "PrevClose", "ClosePrice", "PreviousClose"),
    "volume": ("Volume", "TotalTradedQty", "VolumeTraded"),
}


class ChoiceError(RuntimeError):
    pass


def _ok(resp: Any) -> bool:
    return isinstance(resp, dict) and str(resp.get("Status", "")).strip().lower() == "success"


def _reason(resp: Any) -> str:
    if isinstance(resp, dict):
        for k in ("Reason", "Message", "Error"):
            if resp.get(k):
                return str(resp[k])
    return str(resp)[:200]


def encrypt_mobile(mobile: str, aes_key: str = "", aes_iv: str = "") -> str:
    """Spec: AES-256-CBC, PKCS7, with a key and IV issued by Choice. Without them, plain base64 --
    what the community client sends for API-key (non-auth) logins."""
    if aes_key and aes_iv:
        from cryptography.hazmat.primitives import padding
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        key, iv = aes_key.encode(), aes_iv.encode()
        if len(key) != 32 or len(iv) != 16:
            raise ChoiceError("CHOICE_AES_KEY must be 32 characters and CHOICE_AES_IV 16 characters")
        padder = padding.PKCS7(128).padder()
        data = padder.update(mobile.encode()) + padder.finalize()
        enc = Cipher(algorithms.AES(key), modes.CBC(iv)).encryptor()
        return base64.b64encode(enc.update(data) + enc.finalize()).decode()
    return base64.b64encode(mobile.encode()).decode()


def parse_scrip_master(text: str) -> Dict[str, Dict[str, Any]]:
    """{"RELIANCE": {"token": 2885, "divisor": 100.0, "series": "EQ"}} for NSE cash, EQ preferred."""
    out: Dict[str, Dict[str, Any]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        if (row.get("Exchange") or "").strip() != "NSE" or (row.get("Segment") or "").strip() != str(NSE_CASH):
            continue
        sym, series = (row.get("Symbol") or "").strip().upper(), (row.get("Series") or "").strip().upper()
        try:
            token = int(row["Token"])
            divisor = float((row.get("PriceDivisor") or "").strip() or 1) or 1.0
        except (KeyError, ValueError):
            continue
        if sym and (sym not in out or (series == "EQ" and out[sym]["series"] != "EQ")):
            out[sym] = {"token": token, "divisor": divisor, "series": series}
    return out


def parse_touchline(resp: Any, divisors: Dict[int, float]) -> Dict[int, Dict[str, float]]:
    """{token: {ltp, open, high, low, prev_close, volume}} in rupees. Rows without a recognisable
    price are left out."""
    body = resp.get("Response") if isinstance(resp, dict) else None
    if isinstance(body, dict):
        body = body.get("lstTouchline") or body.get("Data") or []
    out: Dict[int, Dict[str, float]] = {}
    for row in body if isinstance(body, list) else []:
        if not isinstance(row, dict):
            continue
        try:
            token = int(row.get("Token") or row.get("token"))
        except (TypeError, ValueError):
            continue
        div = divisors.get(token, 100.0)
        vals: Dict[str, float] = {}
        for name, keys in FIELD_CANDIDATES.items():
            raw = next((row[k] for k in keys if row.get(k) not in (None, "")), None)
            if raw is None:
                continue
            try:
                vals[name] = float(raw) if name == "volume" else float(raw) / div
            except (TypeError, ValueError):
                pass
        if vals.get("ltp"):
            out[token] = vals
    return out


class ChoiceClient:
    def __init__(self, api_key: str, vendor_id: str, mobile: str, env: str = "live", vendor_key: str = "",
                 aes_key: str = "", aes_iv: str = "", http: Optional[Callable[..., Any]] = None):
        if env not in BASE_URLS:
            raise ValueError(f"env must be one of {sorted(BASE_URLS)}")
        self.base = self.active_base = BASE_URLS[env]
        self.api_key, self.vendor_id, self.vendor_key = api_key, vendor_id, vendor_key
        self.mobile, self.aes_key, self.aes_iv = mobile, aes_key, aes_iv
        self.http = http or requests.request
        self.session_id = ""
        self.logged_in_at: Optional[datetime] = None
        self.scrips: Dict[str, Dict[str, Any]] = {}
        self.last_raw: Any = None

    def _headers(self, auth: bool = True) -> Dict[str, str]:
        h = {"Content-Type": "application/json", "VendorId": self.vendor_id, "Bearer": self.api_key}
        if self.vendor_key:
            h["VendorKey"] = self.vendor_key
        if auth and self.session_id:
            h["Authorization"] = f"SessionId {self.session_id}"
        return h

    def _call(self, method: str, path: str, body: Optional[dict] = None, auth: bool = True) -> Any:
        resp = self.http(method, f"{self.active_base}/{path.lstrip('/')}", headers=self._headers(auth),
                         data=json.dumps(body) if body is not None else None, timeout=15)
        if getattr(resp, "status_code", 200) == 401:
            self.session_id = ""
            raise ChoiceError("not authorised (HTTP 401) — usually the server's IP doesn't match the key's, or the key expired")
        try:
            return resp.json()
        except ValueError:
            raise ChoiceError(f"{path}: non-JSON response (HTTP {getattr(resp, 'status_code', '?')})")

    @property
    def connected(self) -> bool:
        return bool(self.session_id and self.logged_in_at and self.logged_in_at.date() == datetime.now(IST).date())

    def login(self) -> None:
        """LoginTOTP → GetClientLoginTOTP (returns the OTP) → ValidateTOTP. No manual step."""
        if not (self.api_key and self.vendor_id and self.mobile):
            raise ChoiceError("CHOICE_API_KEY, CHOICE_VENDOR_ID and CHOICE_MOBILE are all needed")
        self.active_base, self.session_id = self.base, ""
        mob = encrypt_mobile(self.mobile, self.aes_key, self.aes_iv)
        r1 = self._call("POST", "api/OpenAPIV1/LoginTOTP", {"MobileNo": mob}, auth=False)
        if not _ok(r1):
            raise ChoiceError(f"LoginTOTP: {_reason(r1)}")
        r2 = self._call("POST", "api/OpenAPIV1/GetClientLoginTOTP", {"MobileNo": mob}, auth=False)
        otp = r2.get("Response") if _ok(r2) else None
        if not otp or isinstance(otp, (dict, list)):
            raise ChoiceError(f"GetClientLoginTOTP: {_reason(r2)}")
        r3 = self._call("POST", "api/OpenAPIV1/ValidateTOTP", {"MobileNo": mob, "OTP": str(otp)}, auth=False)
        if not _ok(r3):
            raise ChoiceError(f"ValidateTOTP: {_reason(r3)}")
        data = r3.get("Response")
        sid = data if isinstance(data, str) else (data.get("SessionId") if isinstance(data, dict) else None)
        if not sid:
            raise ChoiceError("ValidateTOTP returned no SessionId")
        if isinstance(data, dict) and data.get("BaseURL"):   # spec: use the returned BaseURL for the session
            self.active_base = str(data["BaseURL"]).rstrip("/")
        self.session_id, self.logged_in_at = sid, datetime.now(IST)

    def logout(self) -> None:
        if self.session_id:
            try:
                self._call("GET", "api/OpenAPI/Logoff")
            except Exception as exc:
                logger.info("Choice logoff failed: %s", exc)
        self.session_id, self.logged_in_at = "", None

    def load_scrips(self, cache: Path = SCRIP_CACHE, fetch: Callable[..., Any] = requests.get) -> Dict[str, Dict[str, Any]]:
        today = datetime.now(IST).date()
        if cache.exists():
            cached = json.loads(cache.read_text(encoding="utf-8"))
            if cached.get("date") == today.isoformat():
                self.scrips = cached["scrips"]
                return self.scrips
        for back in range(7):   # published per trading day
            d = today - timedelta(days=back)
            resp = fetch(SCRIP_MASTER_URL.format(d=f"{d.day:02d}{d.strftime('%b')}{d.year}"),
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
            if getattr(resp, "status_code", 0) == 200 and resp.text.startswith("Exchange,"):
                self.scrips = parse_scrip_master(resp.text)
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps({"date": today.isoformat(), "scrips": self.scrips}), encoding="utf-8")
                return self.scrips
        raise ChoiceError("couldn't download Choice's scrip master for the last 7 days")

    def touchline(self, symbols: Iterable[str]) -> Dict[str, Dict[str, float]]:
        """{"RELIANCE.NS": {ltp, open, high, low, prev_close, volume}} for NSE cash symbols."""
        if not self.scrips:
            self.load_scrips()
        wanted: Dict[int, tuple] = {}
        for s in symbols:
            info = self.scrips.get(s.replace(".NS", "").upper())
            if info:
                wanted[int(info["token"])] = (s, float(info["divisor"]))
        out: Dict[str, Dict[str, float]] = {}
        tokens = list(wanted)
        for i in range(0, len(tokens), TOUCHLINE_BATCH):
            batch = tokens[i:i + TOUCHLINE_BATCH]
            resp = self._call("POST", "api/OpenAPI/MultipleTouchline",
                              {"MultipleSegToken": ",".join(f"{NSE_CASH}@{t}" for t in batch)})
            if not _ok(resp):
                raise ChoiceError(f"MultipleTouchline: {_reason(resp)}")
            parsed = parse_touchline(resp, {t: wanted[t][1] for t in batch})
            if not parsed:
                self.last_raw = resp
            for token, vals in parsed.items():
                out[wanted[token][0]] = vals
        return out
