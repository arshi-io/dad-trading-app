"""NIFTY 500 constituent list — the universe for the Minervini screen and RS rank.
Also fetches the NIFTY 100 constituent list with sector (Industry) grouping,
used to build the pairs-trading candidate universe (see nightly_pipeline.py's
monthly rescan).

Source: the official niftyindices.com constituent CSVs, fetched directly with
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

_HEADERS = {"User-Agent": "Mozilla/5.0"}

ROOT = Path(__file__).resolve().parents[2]
CACHE_TTL = timedelta(hours=24)

CONSTITUENT_URL = "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv"
CACHE_PATH = ROOT / "data" / "nifty500_constituents.csv"

NIFTY100_CONSTITUENT_URL = "https://niftyindices.com/IndexConstituent/ind_nifty100list.csv"
NIFTY100_CACHE_PATH = ROOT / "data" / "nifty100_constituents.csv"


def _cache_is_fresh(cache_path: Path) -> bool:
    if not cache_path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
    return age < CACHE_TTL


def _load_constituent_csv(url: str, cache_path: Path, force_refresh: bool = False) -> pd.DataFrame:
    """Return the raw constituent DataFrame (Company Name, Industry, Symbol, ...).

    Falls back to the on-disk cache (however stale) if the live fetch fails,
    so a network hiccup never breaks the pipeline. Raises only if there is no
    cache and the live fetch also fails.
    """
    if not force_refresh and _cache_is_fresh(cache_path):
        return pd.read_csv(cache_path)

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        resp.raise_for_status()
        from io import StringIO
        df = pd.read_csv(StringIO(resp.text))
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path, index=False)
        logger.info("fetched %d constituents from %s", len(df), url)
        return df
    except Exception as exc:
        logger.warning("live constituent fetch failed for %s: %s", url, exc)
        if cache_path.exists():
            return pd.read_csv(cache_path)
        raise


def fetch_nifty500_constituents(force_refresh: bool = False) -> List[str]:
    """Return NIFTY 500 constituents as yfinance-style tickers (SYMBOL.NS)."""
    df = _load_constituent_csv(CONSTITUENT_URL, CACHE_PATH, force_refresh)
    return [f"{sym}.NS" for sym in df["Symbol"]]


def fetch_nifty500_directory(force_refresh: bool = False) -> List[Dict[str, str]]:
    """Return [{"symbol": "RELIANCE.NS", "name": "Reliance Industries Ltd."}, ...].

    Same source/cache as fetch_nifty500_constituents, just with the company
    name kept for search/autocomplete UI instead of discarded.
    """
    df = _load_constituent_csv(CONSTITUENT_URL, CACHE_PATH, force_refresh)
    return [
        {"symbol": f"{row['Symbol']}.NS", "name": str(row["Company Name"])}
        for _, row in df.iterrows()
    ]


def fetch_nifty100_sector_map(force_refresh: bool = False) -> Dict[str, List[str]]:
    """Return {"Industry name": ["SYMBOL.NS", ...]} for the NIFTY 100.

    Used to build the pairs-trading candidate universe: same-sector stocks
    are the only ones worth cointegration-testing (cross-sector pairs are
    rarely cointegrated and just waste compute).
    """
    df = _load_constituent_csv(NIFTY100_CONSTITUENT_URL, NIFTY100_CACHE_PATH, force_refresh)
    sectors: Dict[str, List[str]] = {}
    for _, row in df.iterrows():
        sectors.setdefault(str(row["Industry"]), []).append(f"{row['Symbol']}.NS")
    return sectors
