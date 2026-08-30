from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from code.data.cleaners.ohlcv import clean_ohlcv
from code.data.fetchers.equity import fetch_ohlcv
from code.data.fetchers.nifty500 import fetch_nifty500_constituents
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


def kite_fetch(ticker: str, start: Optional[str] = None, end: Optional[str] = None) -> pd.DataFrame:
    """Fetch EOD OHLCV data using yfinance with a lightweight fallback path."""
    if start is None:
        start = (date.today().replace(year=date.today().year - 2)).strftime("%Y-%m-%d")
    if end is None:
        end = date.today().strftime("%Y-%m-%d")

    try:
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
    except Exception as exc:
        logger.warning("yfinance failed for %s: %s", ticker, exc)
        df = pd.DataFrame()

    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    if "close" in df.columns:
        df = df[["open", "high", "low", "close", "volume"]].copy()
    df.index = pd.DatetimeIndex(df.index, name="timestamp")
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
                        covers_end = not cached.empty and cached.index.max() >= end_ts - pd.Timedelta(days=5)
                        if covers_start and covers_end:
                            result[ticker] = cached
                            continue
                    except Exception:
                        pass
            to_fetch.append(ticker)
    else:
        to_fetch = list(tickers)

    if to_fetch:
        try:
            raw = yf.download(
                tickers=to_fetch, start=start, end=end,
                auto_adjust=True, progress=False, threads=True, group_by="ticker",
            )
        except Exception as exc:
            logger.warning("universe fetch failed: %s", exc)
            raw = pd.DataFrame()

        for ticker in to_fetch:
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

    items: List[Dict[str, Any]] = []
    for ticker, df in price_data.items():
        if df.empty:
            continue
        rank = rs_ranks.get(ticker)
        result = evaluate_trend_template(df, rs_rank=rank)
        items.append({
            "symbol": ticker,
            "rs_rank": rank,
            "passed": result["passed"],
            "conditions_passed": result["conditions_passed"],
            "checklist": result["checklist"],
            "insufficient_history": result["insufficient_history"],
        })

    items.sort(key=lambda item: (item["rs_rank"] if item["rs_rank"] is not None else -1), reverse=True)
    return {"items": items}


# Candidate universe for the pairs scan -- the pairs validated in the v1
# backtest (raw/data/backtest_pairs_trading_v1.csv). Monthly re-scan re-runs
# find_cointegrated_pairs/hedge-ratio over this fixed candidate set rather
# than combinatorics across the whole NIFTY 500 universe.
PAIR_CANDIDATES = [
    ("ICICIBANK.NS", "SBIN.NS"),
    ("AXISBANK.NS", "MARUTI.NS"),
    ("RELIANCE.NS", "MARUTI.NS"),
]
PAIRS_WINDOW = 60
PAIRS_ENTRY_Z = 2.0
PAIRS_EXIT_Z = 0.5
PAIRS_HISTORY_YEARS = 3
PAIRS_CHART_DAYS = 180
PAIR_BASE_QTY = 100  # shares of ticker_a used to express hedge ratio as a share count


def build_pairs_section() -> Dict[str, Any]:
    """Pairs (spread trading) section: cointegration re-scan + current z-score.

    Wraps the EXISTING pairs_trading.py unchanged -- find_cointegrated_pairs
    for the hedge ratio (rolling OLS) + cointegration p-value + half-life,
    compute_spread/compute_zscore (mean_rev.py) for the live 60d z-score.
    """
    end = date.today().strftime("%Y-%m-%d")
    start = (date.today() - timedelta(days=365 * PAIRS_HISTORY_YEARS)).strftime("%Y-%m-%d")

    items: List[Dict[str, Any]] = []
    for ticker_a, ticker_b in PAIR_CANDIDATES:
        try:
            scan = find_cointegrated_pairs(
                [ticker_a, ticker_b], start=start, end=end, pvalue_threshold=1.0,
            )
        except Exception as exc:
            logger.warning("pairs scan failed for %s/%s: %s", ticker_a, ticker_b, exc)
            continue
        if not scan:
            continue
        info = scan[0]

        df_a = clean_ohlcv(fetch_ohlcv(ticker_a, start, end))
        df_b = clean_ohlcv(fetch_ohlcv(ticker_b, start, end))
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
            "spread_series": _series_points(spread_chart),
            "upper_band_series": _series_points(upper_chart),
            "lower_band_series": _series_points(lower_chart),
        })

    return {"items": items, "last_scan": datetime.now().strftime("%Y-%m-%dT%H:%M:%S")}


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=str(path.parent), delete=False) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        tmp_path = Path(handle.name)
    os.replace(tmp_path, path)


ACTION_QUEUE_MAX = 6


def _build_action_queue(screen: Dict[str, Any], pairs: Dict[str, Any]) -> List[Dict[str, str]]:
    """Real action queue: top Minervini passes + pairs currently in an entry zone.

    Not a trade call -- "REVIEW" for a Minervini pass means "worth a look on
    the chart", matching the VCP section's own "algorithm-detected -- verify
    on chart" language. Empty when nothing qualifies; the Today template
    already renders a proper empty state for that case.
    """
    queue: List[Dict[str, str]] = []

    passed = [item for item in screen.get("items", []) if item.get("passed")]
    passed.sort(key=lambda item: (item.get("rs_rank") or 0), reverse=True)
    for item in passed[:3]:
        queue.append({
            "symbol": item["symbol"],
            "action": "REVIEW",
            "reason": f"Passed all 8 Minervini conditions, RS rank {item.get('rs_rank')}",
            "link": f"/stock/{item['symbol']}",
        })

    # Pair symbols contain "/" (e.g. "ICICIBANK.NS/SBIN.NS") which breaks the
    # single-segment /stock/{asset} route -- link these to the Spread Trades
    # tab instead, not a per-symbol Stock Room page.
    for pair in pairs.get("items", []):
        if pair.get("status") in ("BUY", "SELL"):
            queue.append({
                "symbol": pair["pair"],
                "action": pair["status"],
                "reason": f"Spread z-score {pair['zscore']:.2f} -- entry zone (beyond {pair['entry_z']}sigma)",
                "link": "/pairs",
            })

    return queue[:ACTION_QUEUE_MAX]


def _build_briefing(regime: str, vol_state: str, breadth: float, slope: float) -> List[Dict[str, str]]:
    bullets = [
        f"Regime: {regime} | vol {vol_state} | breadth {breadth:.1f}%",
        f"200DMA slope: {slope:.2%}",
        "Watch for confirmation on price action before adding risk.",
        "Keep sizing disciplined and avoid hero trades.",
    ]
    return [{"text": bullet} for bullet in bullets]


# RS rank threshold for "strong" in a BULLISH SETUP stance -- notably above
# the trend template's own >=70 pass gate (condition 8), not just at it.
STANCE_RS_STRONG = 85
# conditions_passed at or below this -> a clean template fail, not a "close call"
STANCE_WEAK_CONDITIONS = 5

_STANCE_REGIME_WORD = {"TRENDING_UP": "trending up", "TRENDING_DOWN": "trending down", "CHOPPY": "choppy"}


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
    evidence.append(f"regime {_STANCE_REGIME_WORD.get(regime, (regime or 'unknown').lower())}")

    if passed and rs_rank is not None and rs_rank >= STANCE_RS_STRONG and regime == "TRENDING_UP":
        stance, stance_class = "BULLISH SETUP", "good"
    elif rs_rank is None or conditions_passed <= STANCE_WEAK_CONDITIONS:
        stance, stance_class = "AVOID / NO SETUP", "bad"
    else:
        stance, stance_class = "WATCH", "wait"

    return {"stance": stance, "stance_class": stance_class, "stance_evidence": evidence}


def run_pipeline() -> Dict[str, Any]:
    _ensure_dirs()
    today = date.today().strftime("%Y-%m-%d")
    snapshot_path = SNAPSHOT_DIR / f"{today}.json"

    nifty = kite_fetch(NIFTY_INDEX_TICKER)
    if nifty.empty:
        nifty = pd.DataFrame({"close": [None]}, index=[pd.Timestamp.today()])
    nifty.index = pd.DatetimeIndex(nifty.index, name="timestamp")
    nifty = nifty.rename(columns={"close": "close"}) if "close" in nifty.columns else nifty.assign(close=pd.Series([None] * len(nifty), index=nifty.index))

    returns = nifty["close"].pct_change().dropna()
    garch_vol = forecast_volatility(returns, window=60, refit_every=5)

    breadth_pct = compute_breadth_pct(NIFTY50_CONSTITUENTS)
    breadth_series = pd.Series([breadth_pct], dtype=float) if pd.notna(breadth_pct) else pd.Series([], dtype=float)
    regime_info = classify_market_regime(garch_vol, nifty["close"], breadth_series)

    screen_section = build_screen_section()
    for item in screen_section["items"]:
        item.update(_compute_screen_stance(item, regime_info["regime"]))
    pairs_section = build_pairs_section()

    payload = {
        "regime": regime_info["regime"],
        "indices": {
            "nifty": {
                "symbol": NIFTY_INDEX_TICKER,
                "close": float(nifty["close"].dropna().iloc[-1]) if nifty["close"].dropna().size and not isinstance(nifty["close"].dropna().iloc[-1], pd.Series) else None,
                "vol_state": regime_info["vol_state"],
                "breadth_pct": regime_info["breadth_pct"],
            }
        },
        "briefing": {
            "items": _build_briefing(
                regime_info["regime"],
                regime_info["vol_state"],
                regime_info["breadth_pct"],
                regime_info["slope"],
            )
        },
        "action_queue": _build_action_queue(screen_section, pairs_section),
        "screen": screen_section,
        "pairs": pairs_section,
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
            "status": "ok",
            "source": "yfinance",
        },
    }
    _atomic_write_json(snapshot_path, payload)
    return payload


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_pipeline()
    print(json.dumps(result, indent=2))
