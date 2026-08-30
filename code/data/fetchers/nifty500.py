"""NIFTY 500 constituent list — the universe for the Minervini screen and RS rank.

Source: the official niftyindices.com constituent CSV, fetched directly with
``requests`` (the ``nsepython`` wrapper's ``nse_index()`` hits a broken
certificate on ``iislliveblob.niftyindices.com`` and cannot be used).
Cached to disk for 24h so the nightly pipeline doesn't refetch on every run.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

logger = logging.getLogger(__name__)

CONSTITUENT_URL = "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv"
_HEADERS = {"User-Agent": "Mozilla/5.0"}

ROOT = Path(__file__).resolve().parents[2]
CACHE_PATH = ROOT / "data" / "nifty500_constituents.csv"
CACHE_TTL = timedelta(hours=24)


def _cache_is_fresh() -> bool:
    if not CACHE_PATH.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(CACHE_PATH.stat().st_mtime)
    return age < CACHE_TTL


def _load_constituent_df(force_refresh: bool = False) -> pd.DataFrame:
    """Return the raw constituent DataFrame (Company Name, Industry, Symbol, ...).

    Falls back to the on-disk cache (however stale) if the live fetch fails,
    so a network hiccup never breaks the nightly pipeline. Raises only if
    there is no cache and the live fetch also fails.
    """
    if not force_refresh and _cache_is_fresh():
        return pd.read_csv(CACHE_PATH)

    try:
        resp = requests.get(CONSTITUENT_URL, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(CACHE_PATH, index=False)
        logger.info("fetched %d NIFTY 500 constituents", len(df))
        return df
    except Exception as exc:
        logger.warning("live NIFTY 500 fetch failed: %s", exc)
        if CACHE_PATH.exists():
            return pd.read_csv(CACHE_PATH)
        raise


def fetch_nifty500_constituents(force_refresh: bool = False) -> List[str]:
    """Return NIFTY 500 constituents as yfinance-style tickers (SYMBOL.NS)."""
    df = _load_constituent_df(force_refresh)
    return [f"{sym}.NS" for sym in df["Symbol"]]


def fetch_nifty500_directory(force_refresh: bool = False) -> List[Dict[str, str]]:
    """Return [{"symbol": "RELIANCE.NS", "name": "Reliance Industries Ltd."}, ...].

    Same source/cache as fetch_nifty500_constituents, just with the company
    name kept for search/autocomplete UI instead of discarded.
    """
    df = _load_constituent_df(force_refresh)
    return [
        {"symbol": f"{row['Symbol']}.NS", "name": str(row["Company Name"])}
        for _, row in df.iterrows()
    ]
