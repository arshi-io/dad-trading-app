"""Next results date per stock (yfinance calendar), cached to disk for 3 days."""
from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, Optional

import yfinance as yf

logger = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "earnings_cache.json"
CACHE_TTL = timedelta(days=3)


def _one(symbol: str) -> Optional[str]:
    try:
        cal = yf.Ticker(symbol).calendar
        dates = cal.get("Earnings Date") if isinstance(cal, dict) else None
        upcoming = sorted(d for d in (dates or []) if d >= date.today())
        return upcoming[0].isoformat() if upcoming else None
    except Exception as exc:
        logger.debug("earnings lookup failed for %s: %s", symbol, exc)
        return None


def fetch_next_results(symbols: Iterable[str]) -> Dict[str, Optional[str]]:
    """{"RELIANCE.NS": "2026-10-16", "CUPID.NS": None, ...} -- None means no date published."""
    cache = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
    now = datetime.now()
    out: Dict[str, Optional[str]] = {}
    todo = []
    for sym in dict.fromkeys(symbols):
        hit = cache.get(sym)
        fresh = hit and now - datetime.fromisoformat(hit["at"]) < CACHE_TTL
        if fresh and (hit["date"] is None or hit["date"] >= date.today().isoformat()):
            out[sym] = hit["date"]
        else:
            todo.append(sym)
    if todo:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for sym, d in zip(todo, pool.map(_one, todo)):
                out[sym] = d
                cache[sym] = {"date": d, "at": now.isoformat()}
        CACHE_PATH.write_text(json.dumps(cache))
    return out
