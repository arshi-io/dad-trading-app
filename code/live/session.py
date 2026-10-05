"""NSE cash-market session clock. Hours live in data, holidays come from NSE's own holiday list
(cached; the cache is used when NSE is unreachable) -- no trading dates are hardcoded."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable, Dict, Optional
from zoneinfo import ZoneInfo

import requests

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
HOLIDAY_CACHE = Path(__file__).resolve().parents[1] / "data" / "nse_holidays.json"
HOLIDAY_TTL = timedelta(days=7)
NSE_HOLIDAY_API = "https://www.nseindia.com/api/holiday-master?type=trading"
_NSE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
    "Accept": "application/json,text/plain,*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/resources/exchange-communication-holidays",
}

# Regular NSE equity timings (IST). Kept as data so a timing change is a one-line edit.
SESSIONS = {
    "PRE_OPEN": (time(9, 0), time(9, 8)),
    "OPEN": (time(9, 15), time(15, 30)),
    "POST_CLOSE": (time(15, 40), time(16, 0)),
}


@dataclass(frozen=True)
class SessionState:
    phase: str               # PRE_OPEN | OPEN | CLOSED | WEEKEND | HOLIDAY
    is_open: bool
    now: datetime
    holiday_name: Optional[str]
    next_open: datetime

    def as_dict(self) -> dict:
        return {"phase": self.phase, "is_open": self.is_open, "now": self.now.isoformat(),
                "holiday": self.holiday_name, "next_open": self.next_open.isoformat()}


def fetch_nse_holidays() -> Dict[str, str]:
    """{"2026-10-02": "Mahatma Gandhi Jayanti", ...} for the cash-market (CM) segment."""
    s = requests.Session()
    s.get("https://www.nseindia.com/", headers=_NSE_HEADERS, timeout=15)  # sets the cookies the API requires
    resp = s.get(NSE_HOLIDAY_API, headers=_NSE_HEADERS, timeout=15)
    resp.raise_for_status()
    out = {}
    for row in resp.json().get("CM", []):
        d = datetime.strptime(row["tradingDate"], "%d-%b-%Y").date()
        out[d.isoformat()] = row.get("description", "Holiday")
    return out


def load_holidays(fetch: Callable[[], Dict[str, str]] = fetch_nse_holidays) -> Dict[str, str]:
    cached = json.loads(HOLIDAY_CACHE.read_text(encoding="utf-8")) if HOLIDAY_CACHE.exists() else None
    if cached and datetime.now() - datetime.fromisoformat(cached["fetched_at"]) < HOLIDAY_TTL:
        return cached["holidays"]
    try:
        holidays = fetch()
        if holidays:
            HOLIDAY_CACHE.parent.mkdir(parents=True, exist_ok=True)
            HOLIDAY_CACHE.write_text(json.dumps({"fetched_at": datetime.now().isoformat(), "holidays": holidays}), encoding="utf-8")
            return holidays
    except Exception as exc:
        logger.warning("NSE holiday fetch failed, using cache: %s", exc)
    return cached["holidays"] if cached else {}


class MarketSession:
    def __init__(self, holidays: Optional[Dict[str, str]] = None):
        self._holidays = holidays

    @property
    def holidays(self) -> Dict[str, str]:
        if self._holidays is None:
            self._holidays = load_holidays()
        return self._holidays

    def is_trading_day(self, d: date) -> bool:
        return d.weekday() < 5 and d.isoformat() not in self.holidays

    def next_trading_day(self, d: date) -> date:
        d += timedelta(days=1)
        while not self.is_trading_day(d):
            d += timedelta(days=1)
        return d

    def state(self, now: Optional[datetime] = None) -> SessionState:
        now = (now or datetime.now(IST)).astimezone(IST)
        today, t = now.date(), now.time()
        open_t, close_t = SESSIONS["OPEN"]

        if not self.is_trading_day(today):
            phase = "WEEKEND" if today.weekday() >= 5 else "HOLIDAY"
            nxt = self.next_trading_day(today)
            return SessionState(phase, False, now, self.holidays.get(today.isoformat()),
                                datetime.combine(nxt, open_t, IST))

        if SESSIONS["PRE_OPEN"][0] <= t < SESSIONS["PRE_OPEN"][1]:
            phase = "PRE_OPEN"
        elif open_t <= t < close_t:
            phase = "OPEN"
        else:
            phase = "CLOSED"
        next_day = today if t < open_t else self.next_trading_day(today)
        return SessionState(phase, phase == "OPEN", now, None, datetime.combine(next_day, open_t, IST))


_default: Optional[MarketSession] = None


def market_session() -> MarketSession:
    global _default
    if _default is None:
        _default = MarketSession()
    return _default
