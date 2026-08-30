"""
Equity OHLCV fetcher (yfinance backend).

Single-asset price downloader for NSE/BSE tickers (e.g. ``RELIANCE.NS``) and
any yfinance-supported symbol. The entire module is designed to be *safe to
call in a loop*: any failure — network error, bad ticker, empty response — is
logged and returns an empty DataFrame rather than raising, so a batch job
never dies on one bad symbol.

Usage
-----
>>> from code.data.fetchers.equity import fetch_ohlcv
>>> df = fetch_ohlcv("RELIANCE.NS", "2024-01-01", "2024-12-31")
>>> df.tail()
"""
from __future__ import annotations

import logging
from datetime import datetime, date
from typing import Union

import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover - optional dependency guard
    yf = None

logger = logging.getLogger(__name__)

DateLike = Union[str, datetime, date, pd.Timestamp]

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


def fetch_ohlcv(
    ticker: str,
    start: DateLike,
    end: DateLike,
    interval: str = "1d",
) -> pd.DataFrame:
    """Fetch OHLCV bars for ``ticker`` between ``start`` and ``end``.

    Parameters
    ----------
    ticker : str
        Any yfinance-compatible symbol (e.g. ``RELIANCE.NS``, ``BTC-USD``).
    start, end : str | datetime | date | pd.Timestamp
        Date range. ``end`` is exclusive in yfinance convention.
    interval : str, default ``"1d"``
        Bar size. Common values: ``1d``, ``1h``, ``15m``, ``5m``.

    Returns
    -------
    pd.DataFrame
        Columns ``[open, high, low, close, volume]`` (lowercase), index is a
        ``DatetimeIndex`` named ``"timestamp"``. Returns an **empty**
        DataFrame with the correct schema on any failure — never raises.
    """
    empty = pd.DataFrame(columns=_OHLCV_COLS)
    empty.index = pd.DatetimeIndex([], name="timestamp")

    if yf is None:
        logger.warning("fetch_ohlcv(%s) skipped because yfinance is unavailable", ticker)
        return empty

    try:
        raw = yf.download(
            tickers=ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        logger.warning("fetch_ohlcv(%s) failed during download: %s", ticker, exc)
        return empty

    if raw is None or raw.empty:
        logger.warning("fetch_ohlcv(%s) returned no data for %s..%s", ticker, start, end)
        return empty

    # yfinance sometimes returns a MultiIndex column frame even for a single ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        try:
            raw = raw.xs(ticker, axis=1, level=-1)
        except KeyError:
            # Fall back to the first ticker level if the expected key isn't there.
            raw.columns = raw.columns.get_level_values(0)

    raw = raw.rename(columns=str.lower)
    missing = [c for c in _OHLCV_COLS if c not in raw.columns]
    if missing:
        logger.warning("fetch_ohlcv(%s) missing columns %s", ticker, missing)
        return empty

    df = raw[_OHLCV_COLS].copy()
    df.index = pd.DatetimeIndex(df.index, name="timestamp")
    df.columns.name = None

    logger.info(
        "fetch_ohlcv(%s) fetched %d bars [%s..%s] interval=%s",
        ticker, len(df), df.index[0].date(), df.index[-1].date(), interval,
    )
    return df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    df = fetch_ohlcv("RELIANCE.NS", "2024-01-01", "2024-12-31")
    print(f"\nRELIANCE.NS — rows fetched: {len(df)}")
    print(df.tail(5))
