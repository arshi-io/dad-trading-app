from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from datetime import datetime, date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from code.data.cleaners.ohlcv import clean_ohlcv
from code.data.fetchers.earnings import fetch_next_results
from code.data.fetchers.equity import fetch_ohlcv
from code.data.fetchers.fno import fetch_lot_sizes
from code.data.fetchers.nifty500 import fetch_nifty100_sector_map, fetch_nifty500_constituents, fetch_nifty500_sectors
from code.regime.market_health import breadth, breadth_200, index_health, market_grade, risk_posture, sector_leaders
from code.signals.minervini.setup import detect_setup, stock_metrics
from code.pipeline.news_rss import fetch_news_items
from code.regime.market_regime import classify_market_regime
from code.signals.garch_volatility import forecast_volatility, classify_regime
from code.signals.mean_rev import compute_zscore
from code.signals.minervini.rs_rank import compute_rs_ranks
from code.signals.minervini.trend_template import evaluate_trend_template
from code.signals.pairs_trading import (
    _classify as _classify_pair_zscore,
    compute_spread,
    find_cointegrated_pairs,
)

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "data" / "snapshots"
PRICES_DIR = ROOT / "data" / "prices"

# NIFTY 50 on yfinance. NOT "^N225" (that's Japan's Nikkei 225).
NIFTY_INDEX_TICKER = "^NSEI"

# Breadth universe: NIFTY 50 constituents (.NS). This list drifts as NSE
# reconstitutes the index periodically and should be refreshed from an
# authoritative NSE source rather than hand-maintained long-term.
NIFTY50_CONSTITUENTS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "BAJFINANCE.NS",
    "KOTAKBANK.NS", "LT.NS", "HCLTECH.NS", "AXISBANK.NS", "MARUTI.NS",
    "SUNPHARMA.NS", "ASIANPAINT.NS", "TITAN.NS", "ULTRACEMCO.NS", "NESTLEIND.NS",
    "WIPRO.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "M&M.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS", "JSWSTEEL.NS", "ADANIENT.NS", "ADANIPORTS.NS",
    "COALINDIA.NS", "BAJAJFINSV.NS", "HDFCLIFE.NS", "SBILIFE.NS", "GRASIM.NS",
    "INDUSINDBK.NS", "TECHM.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS",
    "EICHERMOT.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "BPCL.NS", "UPL.NS",
    "APOLLOHOSP.NS", "BAJAJ-AUTO.NS", "TATACONSUM.NS", "SHRIRAMFIN.NS", "LTIM.NS",
]


def _ensure_dirs() -> None:
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    PRICES_DIR.mkdir(parents=True, exist_ok=True)


def _to_ist(dt: datetime) -> datetime:
    return dt


# EOD bars don't change during the day, but every Stock Room view (and every
# /api/candles call behind its chart) was re-downloading them from yfinance --
# measured at 74-1207ms of dead weight per view, twice per page. Same on-disk
# parquet store the nightly universe fetch already uses, with a shorter TTL so
# the evening's new bar still lands the same day.
KITE_CACHE_TTL = timedelta(hours=6)


def _kite_cache_path(ticker: str) -> Path:
    safe = ticker.replace("&", "_AND_").replace("/", "_")
    return PRICES_DIR / f"kite_{safe}.parquet"


@lru_cache(maxsize=1)
def _recent_bhav(hour_key: str) -> pd.DataFrame:
    """Last few NSE bhavcopy days indexed by (symbol, date); downloads missing days first. Cached per hour."""
    from code.research import bhavcopy
    try:
        bhavcopy.backfill(start=date.today() - timedelta(days=10), workers=1)
    except Exception as exc:
        logger.warning("bhavcopy refresh failed: %s", exc)
    files = sorted(bhavcopy.DAYS.glob("*.parquet"))[-5:]
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    df = df.sort_values("series", key=lambda s: s != "EQ").drop_duplicates(["symbol", "date"])
    return df.set_index(["symbol", "date"]).sort_index()


def _patch_from_bhavcopy(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Yahoo often publishes the latest NSE session late, or as a row with volume but NaN prices
    (seen 5 Oct 2026). Fill those recent bars from NSE's own bhavcopy."""
    if not ticker.endswith(".NS") or df.empty or "close" not in df.columns:
        return df
    try:
        rows = _recent_bhav(datetime.now().strftime("%Y%m%d%H")).loc[ticker[:-3]]
    except KeyError:
        return df
    cols = ["open", "high", "low", "close"]
    for d, r in rows[rows.index > df.index[-1] - pd.Timedelta(days=10)].iterrows():
        if d not in df.index or df.loc[d, cols].isna().any():
            df.loc[d, cols] = [float(r[c]) for c in cols]
            if pd.isna(df.loc[d].get("volume")):
                df.loc[d, "volume"] = float(r["volume"])
    return df.sort_index()


def kite_fetch(ticker: str, start: Optional[str] = None, end: Optional[str] = None, use_cache: bool = True) -> pd.DataFrame:
    """Fetch EOD OHLCV data using yfinance, cached to parquet (6h TTL).

    ``use_cache=False`` forces a live pull -- used by the nightly pipeline,
    which must not build a snapshot out of the cache it wrote itself.
    """
    if start is None:
        start = (date.today().replace(year=date.today().year - 2)).strftime("%Y-%m-%d")
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    cache_path = _kite_cache_path(ticker)
    if use_cache and cache_path.exists():
        age = datetime.now() - datetime.fromtimestamp(cache_path.stat().st_mtime)
        if age < KITE_CACHE_TTL:
            try:
                return _patch_from_bhavcopy(ticker, pd.read_parquet(cache_path))
            except Exception as exc:  # corrupt/partial file -- fall through to a live fetch
                logger.warning("kite_fetch cache unreadable for %s: %s", ticker, exc)

    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    except Exception as exc:
        logger.warning("yfinance failed for %s: %s", ticker, exc)
        df = pd.DataFrame()

    if df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    if "close" in df.columns:
        df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.DatetimeIndex(df.index, name="timestamp")
    df = _patch_from_bhavcopy(ticker, df)

    if use_cache and not df.empty:
        try:
            PRICES_DIR.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path)
        except Exception as exc:  # caching is best-effort; never fail the request over it
            logger.warning("kite_fetch cache write failed for %s: %s", ticker, exc)
    return df


def nse_extras() -> Dict[str, Any]:
    """Fetch NIFTY constituents and a simple breadth proxy using yfinance."""
    nifty = kite_fetch(NIFTY_INDEX_TICKER)
    if nifty.empty:
        nifty = kite_fetch(NIFTY_INDEX_TICKER)
    return {"nifty_symbol": NIFTY_INDEX_TICKER, "rows": len(nifty)}


def compute_breadth_pct(tickers: List[str]) -> float:
    """% of ``tickers`` whose latest close is above their own 200-day SMA.

    Real computation over the breadth universe — never a placeholder
    constant. Returns NaN if fewer than 200 bars of history are available
    for any ticker in the batch (caller decides the fallback).
    """
    start = (date.today().replace(year=date.today().year - 2)).strftime("%Y-%m-%d")
    end = date.today().strftime("%Y-%m-%d")

    try:
        raw = yf.download(
            tickers=tickers, start=start, end=end,
            auto_adjust=True, progress=False, threads=True, group_by="ticker",
        )
    except Exception as exc:
        logger.warning("breadth fetch failed: %s", exc)
        return float("nan")

    if raw is None or raw.empty:
        return float("nan")

    above = 0
    counted = 0
    for ticker in tickers:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                close = raw[ticker]["Close"].dropna()
            else:
                close = raw["Close"].dropna()
        except (KeyError, TypeError):
            continue
        if len(close) < 200:
            continue
        sma200 = close.rolling(200, min_periods=200).mean()
        if pd.isna(sma200.iloc[-1]):
            continue
        counted += 1
        if close.iloc[-1] > sma200.iloc[-1]:
            above += 1

    if counted == 0:
        return float("nan")
    return above / counted * 100.0


# Enough history for a 12-month RS return plus a 52-week trend-template
# window with margin, even for the earliest as-of date a caller might slice to.
UNIVERSE_HISTORY_YEARS = 3
UNIVERSE_CACHE_TTL = timedelta(hours=24)
UNIVERSE_BATCH = 100


def _universe_cache_path(ticker: str) -> Path:
    safe = ticker.replace("&", "_AND_")
    return PRICES_DIR / f"{safe}.parquet"


def fetch_universe_closes(
    tickers: List[str],
    start: Optional[str] = None,
    end: Optional[str] = None,
    use_cache: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Batch-fetch OHLCV for ``tickers``, cached per-ticker to parquet (24h TTL).

    Returns ``{ticker: df}`` with a lowercase ``close`` column, ascending by
    date. Tickers with no data (delisted / not yet listed in the window) are
    simply absent from the result -- callers treat that as insufficient
    history, not an error.
    """
    if start is None:
        start = (date.today().replace(year=date.today().year - UNIVERSE_HISTORY_YEARS)).strftime("%Y-%m-%d")
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    result: Dict[str, pd.DataFrame] = {}
    to_fetch: List[str] = []

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)

    if use_cache:
        PRICES_DIR.mkdir(parents=True, exist_ok=True)
        for ticker in tickers:
            path = _universe_cache_path(ticker)
            if path.exists():
                age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
                if age < UNIVERSE_CACHE_TTL:
                    try:
                        cached = pd.read_parquet(path)
                        # A cache written for a different (narrower/later-
                        # starting or earlier-ending) [start, end] window --
                        # e.g. a historical golden-test fetch, or an older
                        # live run -- must not be reused just because the
                        # file itself is young. Check both bounds.
                        covers_start = not cached.empty and cached.index.min() <= start_ts + pd.Timedelta(days=5)
                        # Must reach the last weekday before `end` (yfinance `end` is exclusive); a 5-day slack
                        # kept serving 1 Oct bars on 6 Oct. Holidays just cost one refetch.
                        last_weekday = end_ts - pd.offsets.BDay(1)
                        covers_end = not cached.empty and cached.index.max() >= last_weekday
                        if covers_start and covers_end:
                            result[ticker] = cached
                            continue
                    except Exception:
                        pass
            to_fetch.append(ticker)
    else:
        to_fetch = list(tickers)

    # 100 tickers per download: one 500-ticker call peaked high enough to get a 512 MB container killed.
    for i in range(0, len(to_fetch), UNIVERSE_BATCH):
        batch = to_fetch[i:i + UNIVERSE_BATCH]
        try:
            raw = yf.download(
                tickers=batch, start=start, end=end,
                auto_adjust=True, progress=False, threads=True, group_by="ticker",
            )
        except Exception as exc:
            logger.warning("universe fetch failed: %s", exc)
            raw = pd.DataFrame()

        for ticker in batch:
            try:
                if isinstance(raw.columns, pd.MultiIndex):
                    sub = raw[ticker].copy()
                else:
                    sub = raw.copy()
            except (KeyError, TypeError):
                continue
            sub = sub.rename(columns=str.lower)
            if "close" not in sub.columns:
                continue
            sub.index = pd.DatetimeIndex(sub.index, name="timestamp")
            sub = _patch_from_bhavcopy(ticker, sub)
            sub = sub.dropna(subset=["close"])
            if sub.empty:
                continue
            sub.index = pd.DatetimeIndex(sub.index, name="timestamp")
            result[ticker] = sub
            if use_cache:
                try:
                    sub.to_parquet(_universe_cache_path(ticker))
                except Exception as exc:
                    logger.debug("cache write failed for %s: %s", ticker, exc)

    return result


def build_screen_section(
    price_data: Optional[Dict[str, pd.DataFrame]] = None,
    as_of: Optional[pd.Timestamp] = None,
) -> Dict[str, Any]:
    """Build the Minervini screen: trend template + RS rank across the universe.

    Parameters
    ----------
    price_data : dict[str, pd.DataFrame], optional
        Ticker -> OHLCV df. Fetched fresh (NIFTY 500 constituents) if omitted.
    as_of : pd.Timestamp, optional
        If given, every series is sliced to ``index <= as_of`` first -- this
        is what makes the golden tests possible (evaluate as of a historical
        date rather than "today").
    """
    if price_data is None:
        tickers = fetch_nifty500_constituents()
        price_data = fetch_universe_closes(tickers)

    if as_of is not None:
        price_data = {ticker: df[df.index <= as_of] for ticker, df in price_data.items()}

    closes = {ticker: df["close"] for ticker, df in price_data.items() if not df.empty}
    rs_ranks = compute_rs_ranks(closes)
    try:
        sectors = fetch_nifty500_sectors()
    except Exception:
        sectors = {}

    items: List[Dict[str, Any]] = []
    for ticker, df in price_data.items():
        if df.empty:
            continue
        rank = rs_ranks.get(ticker)
        result = evaluate_trend_template(df, rs_rank=rank)
        item = {
            "symbol": ticker,
            "rs_rank": rank,
            "passed": result["passed"],
            "conditions_passed": result["conditions_passed"],
            "checklist": result["checklist"],
            "insufficient_history": result["insufficient_history"],
            "sector": sectors.get(ticker),
        }
        bars = df.dropna(subset=["close"])
        if {"open", "high", "low", "volume"} <= set(bars.columns) and len(bars) >= 60:
            item.update(stock_metrics(bars))
            if result["passed"]:
                item["setup"] = detect_setup(bars)
        items.append(item)

    items.sort(key=lambda item: (item["rs_rank"] if item["rs_rank"] is not None else -1), reverse=True)
    return {"items": items}


PAIRS_WINDOW = 60
PAIRS_ENTRY_Z = 2.0
PAIRS_EXIT_Z = 0.5
PAIRS_HISTORY_YEARS = 3
PAIRS_CHART_DAYS = 180
PAIR_BASE_QTY = 100  # shares of ticker_a used to express hedge ratio as a share count
PAIRS_TOP_N = 10  # keep only the N most-significant survivors -- enough to fill the tab, not overwhelm it
PAIRS_CACHE_PATH = ROOT / "data" / "pairs_cointegration_cache.json"


def run_pairs_monthly_rescan() -> Dict[str, Any]:
    """The expensive O(n^2)-per-sector cointegration search -- MONTHLY ONLY,
    never from the nightly job.

    Candidate universe: every within-sector pair from the NIFTY 100 (Industry
    column from fetch_nifty100_sector_map). Cross-sector pairs are skipped --
    they're rarely cointegrated and just waste compute. find_cointegrated_pairs
    itself (pairs_trading.py, unchanged) does the actual Engle-Granger test
    and hedge-ratio OLS for every pair within each sector bucket.

    Caches the top PAIRS_TOP_N survivors (by p-value) to PAIRS_CACHE_PATH;
    build_pairs_section() (the nightly job) reads this cache and only
    recomputes the live z-score, never re-runs the search.
    """
    started = time.monotonic()
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=365 * PAIRS_HISTORY_YEARS)).strftime("%Y-%m-%d")

    sectors = fetch_nifty100_sector_map()
    tested = 0
    survivors: List[Dict[str, Any]] = []
    for sector, tickers in sectors.items():
        n = len(tickers)
        if n < 2:
            continue
        tested += n * (n - 1) // 2
        try:
            scan = find_cointegrated_pairs(tickers, start=start, end=end, pvalue_threshold=0.05)
        except Exception as exc:
            logger.warning("pairs monthly rescan failed for sector %r (%d tickers): %s", sector, n, exc)
            continue
        for info in scan:
            info["sector"] = sector
        survivors.extend(scan)

    # A negative hedge ratio means both legs move the same way -- not a hedged pair.
    # Without futures on both legs the short can't be held overnight.
    lots = fetch_lot_sizes()
    survivors = [s for s in survivors if s["hedge_ratio"] > 0 and (not lots or (s["ticker_a"] in lots and s["ticker_b"] in lots))]
    survivors.sort(key=lambda d: d["pvalue"])
    kept = survivors[:PAIRS_TOP_N]
    elapsed = time.monotonic() - started

    cache_payload = {
        "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "candidates_tested": tested,
        "candidates_passed": len(survivors),
        "elapsed_seconds": round(elapsed, 1),
        "pairs": kept,
    }
    _atomic_write_json(PAIRS_CACHE_PATH, cache_payload)
    logger.info(
        "pairs monthly rescan done: %d sectors, %d candidates tested, %d passed p<0.05, kept top %d, %.1fs",
        len(sectors), tested, len(survivors), len(kept), elapsed,
    )
    return cache_payload


def _load_pairs_cache() -> Optional[Dict[str, Any]]:
    if not PAIRS_CACHE_PATH.exists():
        return None
    try:
        return json.loads(PAIRS_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("pairs cache unreadable, treating as empty: %s", exc)
        return None


def _futures_legs(ticker_a: str, ticker_b: str, hedge: float, price_a: float, price_b: float, lots: Dict[str, int]) -> Dict[str, Any]:
    """One lot of A against the nearest whole number of B lots that matches the hedge ratio.
    Overrides qty_a/qty_b with real futures quantities when both legs trade in F&O."""
    lot_a, lot_b = lots.get(ticker_a), lots.get(ticker_b)
    if not (lot_a and lot_b):
        return {"fno": False, "lot_a": lot_a, "lot_b": lot_b}
    lots_b = max(1, round(hedge * lot_a / lot_b))
    qty_b = lots_b * lot_b
    return {
        "fno": True, "lot_a": lot_a, "lot_b": lot_b, "lots_a": 1, "lots_b": lots_b,
        "qty_a": lot_a, "qty_b": qty_b,
        "notional_a": round(lot_a * price_a), "notional_b": round(qty_b * price_b),
        "hedge_mismatch_pct": round((qty_b / (hedge * lot_a) - 1) * 100, 1),
    }


def build_pairs_section() -> Dict[str, Any]:
    """Pairs (spread trading) section for the NIGHTLY job.

    Reads the cointegrated survivors cached by run_pairs_monthly_rescan()
    (the expensive O(n^2) search never runs here) and only recomputes each
    survivor's live 60d z-score / spread chart via compute_spread/
    compute_zscore (mean_rev.py) -- cheap, safe to run every night.
    """
    cache = _load_pairs_cache()
    if not cache or not cache.get("pairs"):
        return {"items": [], "last_scan": cache.get("generated_at") if cache else None}

    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=365 * PAIRS_HISTORY_YEARS)).strftime("%Y-%m-%d")

    lots = fetch_lot_sizes()
    items: List[Dict[str, Any]] = []
    for info in cache["pairs"]:
        ticker_a, ticker_b = info["ticker_a"], info["ticker_b"]
        if info["hedge_ratio"] <= 0:
            continue
        df_a = clean_ohlcv(_patch_from_bhavcopy(ticker_a, fetch_ohlcv(ticker_a, start, end)))
        df_b = clean_ohlcv(_patch_from_bhavcopy(ticker_b, fetch_ohlcv(ticker_b, start, end)))
        if df_a.empty or df_b.empty:
            continue
        close_a, close_b = df_a["close"], df_b["close"]

        spread = compute_spread(close_a, close_b, info["hedge_ratio"])
        rolling = spread.rolling(window=PAIRS_WINDOW, min_periods=PAIRS_WINDOW)
        band_mean = rolling.mean()
        band_std = rolling.std(ddof=1)
        z_series = compute_zscore(spread, PAIRS_WINDOW).dropna()
        if z_series.empty:
            continue
        latest_z = float(z_series.iloc[-1])
        status = _classify_pair_zscore(latest_z, PAIRS_ENTRY_Z, PAIRS_EXIT_Z)

        chart_index = z_series.iloc[-PAIRS_CHART_DAYS:].index
        spread_chart = spread.reindex(chart_index)
        upper_chart = (band_mean + PAIRS_ENTRY_Z * band_std).reindex(chart_index)
        lower_chart = (band_mean - PAIRS_ENTRY_Z * band_std).reindex(chart_index)

        def _series_points(s: pd.Series) -> List[List[float]]:
            s = s.dropna()
            return [[int(pd.Timestamp(ts).value // 1_000_000), float(v)] for ts, v in s.items()]

        items.append({
            "pair": f"{ticker_a}/{ticker_b}",
            "ticker_a": ticker_a,
            "ticker_b": ticker_b,
            "sector": info.get("sector"),
            "hedge_ratio": float(info["hedge_ratio"]),
            "qty_a": PAIR_BASE_QTY,
            "qty_b": int(round(PAIR_BASE_QTY * info["hedge_ratio"])),
            "pvalue": float(info["pvalue"]),
            "half_life_days": float(info["half_life"]),
            "zscore": latest_z,
            "status": status,
            "window": PAIRS_WINDOW,
            "entry_z": PAIRS_ENTRY_Z,
            "exit_z": PAIRS_EXIT_Z,
            **_futures_legs(ticker_a, ticker_b, float(info["hedge_ratio"]), float(close_a.iloc[-1]), float(close_b.iloc[-1]), lots),
            "spread_series": _series_points(spread_chart),
            "upper_band_series": _series_points(upper_chart),
            "lower_band_series": _series_points(lower_chart),
        })

    return {"items": items, "last_scan": cache["generated_at"]}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path = path.resolve()  # write through a symlink (Railway volume) instead of replacing the link
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


ACTION_QUEUE_STOCKS = 8
NEAR_PIVOT_PCT = 5.0


def _fmt_rs(v: float) -> str:
    return f"₹{v:,.2f}"


def _build_action_queue(screen: Dict[str, Any], pairs: Dict[str, Any]) -> List[Dict[str, Any]]:
    """What could actually be acted on tomorrow: template passes in their buy range or within
    5% of a pivot, plus every pair in its entry zone that can be held through futures.
    Extended stocks are left out on purpose -- chasing them is the classic mistake."""
    stocks = []
    for item in screen.get("items", []):
        setup = item.get("setup") or {}
        if not item.get("passed"):
            continue
        if setup.get("status") == "IN BUY RANGE":
            action, reason = "BUY RANGE", f"Broke out above {_fmt_rs(setup['pivot'])}, now {setup['pct_past_pivot']:.1f}% past it — still inside the 5% buy range"
        elif setup.get("status") == "BASING" and setup.get("pct_to_pivot") is not None and setup["pct_to_pivot"] <= NEAR_PIVOT_PCT:
            action, reason = "NEAR PIVOT", f"{setup['pct_to_pivot']:.1f}% below its {_fmt_rs(setup['pivot'])} pivot after a {setup['base_bars']}-day base"
        else:
            continue
        if setup.get("vcp"):
            reason += " · tightening like a VCP"
        stocks.append({"symbol": item["symbol"], "kind": "stock", "action": action, "reason": reason,
                       "rs_rank": item.get("rs_rank"), "vcp": setup.get("vcp", False), "link": f"/stock/{item['symbol']}"})
    stocks.sort(key=lambda s: (s["action"] == "BUY RANGE", s["vcp"], s["rs_rank"] or 0), reverse=True)

    pair_rows = []
    for pair in pairs.get("items", []):
        if pair.get("status") in ("BUY", "SELL") and pair.get("fno", True):
            a, b = pair["ticker_a"].replace(".NS", ""), pair["ticker_b"].replace(".NS", "")
            legs = f"buy {a} / sell {b}" if pair["status"] == "BUY" else f"sell {a} / buy {b}"
            pair_rows.append({"symbol": pair["pair"], "kind": "pair", "action": f"SPREAD {pair['status']}",
                              "reason": f"Spread stretched {abs(pair['zscore']):.1f}σ from normal — {legs} (futures)",
                              "link": "/pairs"})
    return stocks[:ACTION_QUEUE_STOCKS] + pair_rows


def _grade_record(grade: str) -> Optional[Dict[str, Any]]:
    path = ROOT / "data" / "base_rates.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))["grades"].get(grade)
    except (OSError, ValueError, KeyError):
        return None


def _build_briefing(
    health: Dict[str, Any], brd: Dict[str, Any], posture: Dict[str, Any], leaders: List[Dict[str, Any]],
    changes: Dict[str, List[str]], results_soon: List[tuple], grade: str = "", b200: Optional[float] = None,
) -> List[Dict[str, str]]:
    """Plain-language market read, every line from a computed number -- no filler."""
    def sym(s: str) -> str:
        return s.replace(".NS", "")

    bullets = []
    where = []
    where.append("above" if health.get("above_50") else "below")
    where.append("above" if health.get("above_200") else "below")
    bullets.append(
        f"NIFTY {health['close']:,.2f} ({health['chg_pct']:+.1f}% on the day), {abs(health['pct_from_high']):.1f}% below its 52-week high — "
        f"{where[0]} its 50-day and {where[1]} its 200-day average."
    )
    dd = health.get("distribution_days")
    if dd is not None:
        weight = "heavy institutional selling" if dd >= 5 else "some selling pressure" if dd >= 3 else "light selling"
        bullets.append(f"{dd} distribution days in the last 25 sessions — {weight}.")
    rally = health.get("rally") or {}
    if rally.get("state") == "CONFIRMED":
        bullets.append(f"Follow-through day on {rally['ftd_date']} confirmed the rally off the {rally['low_date']} low.")
    elif rally.get("state") == "ATTEMPT":
        bullets.append(f"Rally attempt is on day {rally['day']} from the {rally['low_date']} low — no follow-through day yet, so no confirmed uptrend.")
    else:
        bullets.append("NIFTY closed at a fresh low for the last three months — no rally attempt under way.")
    if grade and b200 is not None:
        record = _grade_record(grade)
        bullets.append(
            f"Market grade {grade}: {b200 * 100:.0f}% of stocks above their 200-day average, NIFTY "
            f"{'above' if health.get('above_50') else 'below'} its 50-day."
            + (f" Since 2012, breakouts in grade-{grade} markets averaged {record['avg_r_trail']:+.2f}R with a trailing exit." if record else "")
        )
    bullets.append(
        f"NIFTY 500: {brd['advancers']} up, {brd['decliners']} down; "
        f"{brd['new_highs']} at 52-week highs vs {brd['new_lows']} at 52-week lows."
    )
    if leaders:
        bullets.append("Leading groups: " + "; ".join(
            f"{r['sector']} ({r['passing']} of {r['members']} pass{', led by ' + ', '.join(sym(s) for s in r['leaders'][:2]) if r['leaders'] else ''})"
            for r in leaders[:3]) + ".")
    if changes.get("new") or changes.get("dropped"):
        line = f"Since the last run: {len(changes.get('new', []))} new template passes"
        if changes.get("new"):
            line += " (" + ", ".join(sym(s) for s in changes["new"][:4]) + ("…" if len(changes["new"]) > 4 else "") + ")"
        line += f", {len(changes.get('dropped', []))} dropped out."
        bullets.append(line)
    if results_soon:
        bullets.append("Results inside the next 4 weeks: " + ", ".join(
            f"{sym(s)} {pd.Timestamp(d).strftime('%d %b')}" for s, d in results_soon[:6]) + ".")
    bullets.append("Stance: " + posture["text"])
    return [{"text": b} for b in bullets]


# RS rank threshold for "strong" in a BULLISH SETUP stance -- notably above
# the trend template's own >=70 pass gate (condition 8), not just at it.
STANCE_RS_STRONG = 85
# conditions_passed at or below this -> a clean template fail, not a "close call"
STANCE_WEAK_CONDITIONS = 5

_STANCE_REGIME_WORD = {"TRENDING_UP": "uptrend", "TRENDING_DOWN": "correction", "CHOPPY": "choppy"}


def _compute_screen_stance(item: Dict[str, Any], regime: str) -> Dict[str, Any]:
    """Deterministic BULLISH SETUP / WATCH / AVOID stance for one screen row.

    No LLM call -- purely the template pass/fail, RS rank, and market regime
    already computed for the 500-row screen, so it stays cheap. Evidence
    cites the exact signals behind the word; the word is never shown bare.
    """
    passed = item["passed"]
    conditions_passed = item["conditions_passed"]
    rs_rank = item["rs_rank"]

    evidence = [f"Stage 2 {'✓' if passed else f'{conditions_passed}/8'}"]
    if rs_rank is not None:
        evidence.append(f"RS {rs_rank}")
    evidence.append(f"market {_STANCE_REGIME_WORD.get(regime, (regime or 'unknown').lower())}")

    if passed and rs_rank is not None and rs_rank >= STANCE_RS_STRONG and regime == "TRENDING_UP":
        stance, stance_class = "BULLISH SETUP", "good"
    elif rs_rank is None or conditions_passed <= STANCE_WEAK_CONDITIONS:
        stance, stance_class = "AVOID / NO SETUP", "bad"
    else:
        stance, stance_class = "WATCH", "wait"

    return {"stance": stance, "stance_class": stance_class, "stance_evidence": evidence}


REVIEW_PRECOMPUTE_MAX = 15
RESULTS_SOON_DAYS = 28


def _previous_snapshot(today_path: Path) -> Optional[Dict[str, Any]]:
    older = [p for p in sorted(SNAPSHOT_DIR.glob("*.json")) if p.name < today_path.name]
    if not older:
        return None
    try:
        return json.loads(older[-1].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _screen_changes(screen: Dict[str, Any], previous: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    if not previous:
        return {}
    now = {i["symbol"] for i in screen.get("items", []) if i.get("passed")}
    before = {i["symbol"] for i in previous.get("screen", {}).get("items", []) if i.get("passed")}
    return {"new": sorted(now - before), "dropped": sorted(before - now)}


def _build_predictions_and_settle(
    screen_section: Dict[str, Any], pairs_section: Dict[str, Any], posture: Dict[str, Any], results: Dict[str, Optional[str]],
) -> List[Dict[str, Any]]:
    """Projections for screen passes, entry-zone pairs and the watchlist; then settle paper
    trades and score yesterday's calls. A failure here must never cost the rest of the snapshot."""
    from code.trading.portal import get_watchlist, log_predictions, score_predictions, settle_open_trades
    from code.trading.predict import build_predictions, prediction_candidates

    predictions: List[Dict[str, Any]] = []
    try:
        candidates = prediction_candidates(screen_section, pairs_section, get_watchlist())
        predictions = build_predictions(candidates, pairs_section, kite_fetch, risk_pct=posture["risk_pct"], results=results)
        log_predictions(predictions)
    except Exception:
        logger.exception("prediction build failed")
    try:
        settled = settle_open_trades(kite_fetch)
        scored = score_predictions(kite_fetch)
        logger.info("paper trades settled=%s, predictions scored=%d", settled, scored)
    except Exception:
        logger.exception("paper trade settlement failed")
    return predictions


def _precompute_reviews(snapshot: Dict[str, Any], symbols: List[str]) -> None:
    """Write the Setup Review for tomorrow's actionable names now, so Stock Room opens instantly
    instead of waiting ~7s on the model. Off-list stocks still generate on first view."""
    from concurrent.futures import ThreadPoolExecutor
    from code.data.fetchers.nifty500 import fetch_nifty500_directory
    from code.evaluator.stock_review import get_review

    try:
        names = {d["symbol"]: d["name"] for d in fetch_nifty500_directory()}
    except Exception:
        names = {}
    news_pool = fetch_news_items(limit=60)

    def one(sym: str) -> None:
        try:
            get_review(sym, kite_fetch(sym).dropna(subset=["close"]), snapshot, news_pool, names.get(sym))
        except Exception:
            logger.exception("review pre-compute failed for %s", sym)

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(one, symbols[:REVIEW_PRECOMPUTE_MAX]))


def _attach_reviews(snapshot: Dict[str, Any]) -> None:
    """Put the Stock Room verdict on each prediction card that has one, so the two pages agree."""
    from code.evaluator.stock_review import cached_review
    for pred in snapshot.get("predictions", []):
        if pred.get("kind") != "stock":
            continue
        memo = cached_review(pred["symbol"], snapshot)
        if memo and not memo.get("sentinel"):
            pred["review"] = {"score": memo.get("score"), "verdict": memo.get("verdict")}


NSE_INDEX_CLOSE_URL = "https://nsearchives.nseindia.com/content/indices/ind_close_all_{d:%d%m%Y}.csv"


def _nse_nifty_row(d: date, session) -> Optional[Dict[str, float]]:
    """NIFTY 50's OHLC/volume for one day from NSE's daily index file (None on holidays)."""
    r = session.get(NSE_INDEX_CLOSE_URL.format(d=d), timeout=20)
    if r.status_code != 200:
        return None
    for line in r.text.splitlines():
        f = line.split(",")
        if f[0] == "Nifty 50" and len(f) > 8:
            return {"open": float(f[2]), "high": float(f[3]), "low": float(f[4]), "close": float(f[5]), "volume": float(f[8])}
    return None


def _patch_nifty_from_nse(nifty: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    """Yahoo's NIFTY series sometimes lags its stock data by days. Fill the gap up to the latest
    stock bar from NSE's own index file. NSE counts index volume differently, so its volume is
    rescaled by the ratio on the last day both sources have (the distribution-day rule compares
    each day with the one before)."""
    if nifty.empty or reference.empty or "volume" not in nifty:
        return nifty
    last, through = nifty.index[-1].date(), pd.Timestamp(reference.index[-1]).date()
    if through <= last:
        return nifty
    from code.research.bhavcopy import _session
    s = _session()
    anchor = _nse_nifty_row(last, s)
    scale = (float(nifty["volume"].iloc[-1]) / anchor["volume"]) if anchor and anchor["volume"] else float("nan")
    rows = {}
    d = last
    while d < through:
        d += timedelta(days=1)
        if d.weekday() < 5 and (row := _nse_nifty_row(d, s)):
            row["volume"] *= scale
            rows[pd.Timestamp(d)] = row
    if not rows:
        return nifty
    logger.info("NIFTY patched from NSE for %s (Yahoo index feed lagging)", ", ".join(str(k.date()) for k in rows))
    extra = pd.DataFrame.from_dict(rows, orient="index")[nifty.columns.intersection(["open", "high", "low", "close", "volume"])]
    return pd.concat([nifty, extra]).sort_index()


def run_pipeline() -> Dict[str, Any]:
    _ensure_dirs()
    today = date.today().strftime("%Y-%m-%d")
    snapshot_path = SNAPSHOT_DIR / f"{today}.json"
    generated_at = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    nifty = kite_fetch(NIFTY_INDEX_TICKER, use_cache=False)
    if nifty.empty:
        nifty = pd.DataFrame({"close": [None]}, index=[pd.Timestamp.today()])
    nifty.index = pd.DatetimeIndex(nifty.index, name="timestamp")
    nifty = nifty.rename(columns={"close": "close"}) if "close" in nifty.columns else nifty.assign(close=pd.Series([None] * len(nifty), index=nifty.index))
    nifty = _patch_nifty_from_nse(nifty, kite_fetch("RELIANCE.NS", use_cache=False))

    returns = nifty["close"].pct_change().dropna()
    garch_vol = forecast_volatility(returns, window=60, refit_every=5)

    breadth_pct = compute_breadth_pct(NIFTY50_CONSTITUENTS)
    breadth_series = pd.Series([breadth_pct], dtype=float) if pd.notna(breadth_pct) else pd.Series([], dtype=float)
    regime_info = classify_market_regime(garch_vol, nifty["close"], breadth_series)
    regime = regime_info["regime"]

    screen_section = build_screen_section()
    for item in screen_section["items"]:
        item.update(_compute_screen_stance(item, regime))
    pairs_section = build_pairs_section()

    health = index_health(nifty)
    brd = breadth({i["symbol"]: i for i in screen_section["items"]})
    leaders = sector_leaders(screen_section["items"])
    b200 = breadth_200(screen_section["items"])
    grade = market_grade(b200, health.get("above_50"))
    posture = risk_posture(regime, health, grade)

    from code.trading.portal import get_watchlist
    watchlist = get_watchlist()
    stock_syms = [i["symbol"] for i in screen_section["items"] if i.get("passed")] + watchlist
    try:
        results = fetch_next_results(stock_syms)
    except Exception:
        logger.exception("results-date fetch failed")
        results = {}
    today_d = date.today()
    results_soon = sorted(
        ((s, d) for s, d in results.items() if d and 0 <= (date.fromisoformat(d) - today_d).days <= RESULTS_SOON_DAYS),
        key=lambda sd: sd[1],
    )

    action_queue = _build_action_queue(screen_section, pairs_section)
    predictions = _build_predictions_and_settle(screen_section, pairs_section, posture, results)
    changes = _screen_changes(screen_section, _previous_snapshot(snapshot_path))

    payload = {
        "regime": regime,
        "indices": {
            "nifty": {
                "symbol": NIFTY_INDEX_TICKER,
                "close": health["close"],
                "chg_pct": health["chg_pct"],
                "pct_from_high": health["pct_from_high"],
                "vol_state": regime_info["vol_state"],
                "breadth_pct": regime_info["breadth_pct"],
            }
        },
        "market": {
            **health,
            "nifty_pct_from_high": health["pct_from_high"],
            "breadth": brd,
            "breadth_200": b200,
            "grade": grade,
            "sector_leaders": leaders,
            "posture": posture,
        },
        "briefing": {"items": _build_briefing(health, brd, posture, leaders, changes, results_soon, grade, b200)},
        "action_queue": action_queue,
        "predictions": predictions,
        "results": results,
        "screen": screen_section,
        "screen_changes": changes,
        "pairs": pairs_section,
        "meta": {
            "generated_at": generated_at,
            "data_asof": pd.Timestamp(nifty.index[-1]).date().isoformat(),
            "status": "ok",
            "source": "yfinance",
        },
    }

    review_syms = list(dict.fromkeys([a["symbol"] for a in action_queue if a["kind"] == "stock"] + watchlist))
    try:
        _precompute_reviews(payload, review_syms)
        _attach_reviews(payload)
    except Exception:
        logger.exception("review pre-compute failed")

    _atomic_write_json(snapshot_path, payload)
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_pipeline()
    print(json.dumps(result, indent=2))
