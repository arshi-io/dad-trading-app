"""NSE stock-futures lot sizes, cached to disk. A pairs trade can only be held
overnight through futures (cash shorts must be squared off the same day)."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from io import StringIO
from pathlib import Path
from typing import Dict

import pandas as pd
import requests

logger = logging.getLogger(__name__)

LOTS_URL = "https://nsearchives.nseindia.com/content/fo/fo_mktlots.csv"
CACHE_PATH = Path(__file__).resolve().parents[2] / "data" / "fno_lot_sizes.json"
CACHE_TTL = timedelta(days=3)


def _parse(text: str) -> Dict[str, int]:
    df = pd.read_csv(StringIO(text))
    df.columns = [c.strip() for c in df.columns]
    month_col = df.columns[2]  # nearest expiry month
    lots: Dict[str, int] = {}
    for _, row in df.iterrows():
        sym, lot = str(row["SYMBOL"]).strip(), str(row[month_col]).strip()
        if lot.isdigit():
            lots[f"{sym}.NS"] = int(lot)
    return lots


def fetch_lot_sizes() -> Dict[str, int]:
    """{"RELIANCE.NS": 500, ...}; stocks without futures are absent. Stale cache beats nothing."""
    cached = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else None
    if cached and datetime.now() - datetime.fromisoformat(cached["fetched_at"]) < CACHE_TTL:
        return cached["lots"]
    try:
        resp = requests.get(LOTS_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        resp.raise_for_status()
        lots = _parse(resp.text)
        if lots:
            CACHE_PATH.write_text(json.dumps({"fetched_at": datetime.now().isoformat(), "lots": lots}))
            return lots
    except Exception as exc:
        logger.warning("lot size fetch failed: %s", exc)
    return cached["lots"] if cached else {}
