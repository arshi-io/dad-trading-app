"""One place that assembles a stock's Setup Review, shared by Stock Room and the nightly
pre-compute so both produce -- and cache -- the same review for the same snapshot."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import pandas as pd

from code.evaluator.memo import _load_cached_memo, build_bundle, generate_memo, match_headlines
from code.pipeline.news_rss import fetch_company_news
from code.signals.mean_rev import generate_signals as generate_mean_rev_signals
from code.signals.minervini.setup import detect_setup, stock_metrics


def latest_signal(asset: str, df: pd.DataFrame) -> Dict[str, Any]:
    if df.empty or len(df) < 60:
        return {"asset": asset, "direction": "WAIT", "confidence": 0.0,
                "strategy": "mean_reversion_zscore", "reason": "insufficient price history"}
    signals = generate_mean_rev_signals(df, window=60, entry_z=2.0, exit_z=0.5)
    if not signals:
        return {"asset": asset, "direction": "HOLD", "confidence": 0.0,
                "strategy": "mean_reversion_zscore", "reason": "no active mean-reversion signal today"}
    latest = signals[-1]
    return {
        "asset": asset,
        "direction": latest.direction,
        "confidence": round(float(latest.confidence), 3),
        "strategy": latest.strategy,
        "timeframe": latest.timeframe,
        "timestamp": latest.timestamp.isoformat() if hasattr(latest.timestamp, "isoformat") else str(latest.timestamp),
        "metadata": latest.metadata,
    }


def review_key(snapshot: Dict[str, Any]) -> str:
    """Reviews are tied to the market data they were written from: the 08:30 refresh re-reads
    the same closing prices as last night's run, so it reuses those reviews; a new trading day doesn't."""
    meta = snapshot.get("meta", {})
    return meta.get("data_asof") or meta.get("generated_at") or "no-snapshot"


def find_screen_item(snapshot: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
    for item in snapshot.get("screen", {}).get("items", []):
        if item.get("symbol") == ticker:
            return item
    return None


def setup_for(ticker: str, df: pd.DataFrame, snapshot: Dict[str, Any]) -> Dict[str, Any]:
    item = find_screen_item(snapshot, ticker) or {}
    setup = item.get("setup") or detect_setup(df)
    metrics = stock_metrics(df)
    return {**setup, "pct_above_50dma": metrics.get("pct_above_50"), "pct_from_52wk_high": metrics.get("pct_from_high")}


def review_bundle(
    ticker: str, df: pd.DataFrame, snapshot: Dict[str, Any],
    news_pool: List[Dict[str, Any]], company_name: Optional[str] = None, results_date: Optional[str] = None,
) -> Dict[str, Any]:
    item = find_screen_item(snapshot, ticker)
    trend_template = None
    if item is not None:
        trend_template = {k: item.get(k) for k in ("passed", "conditions_passed", "rs_rank", "checklist")}
    market = snapshot.get("market") or {}
    return build_bundle(
        ticker,
        signal=latest_signal(ticker, df),
        headlines=fetch_company_news(company_name, 5, ticker) if company_name else match_headlines(ticker, news_pool, limit=5),
        trend_template=trend_template,
        market_regime=snapshot.get("regime"),
        setup=setup_for(ticker, df, snapshot),
        market={k: market.get(k) for k in ("nifty_pct_from_high", "distribution_days", "rally", "posture") if market.get(k) is not None},
        results_date=results_date if results_date is not None else (snapshot.get("results") or {}).get(ticker),
    )


def get_review(ticker: str, df: pd.DataFrame, snapshot: Dict[str, Any], news_pool: List[Dict[str, Any]],
               company_name: Optional[str] = None, results_date: Optional[str] = None) -> Dict[str, Any]:
    bundle = review_bundle(ticker, df, snapshot, news_pool, company_name, results_date)
    return generate_memo(bundle, ticker, snapshot_date=review_key(snapshot))


def cached_review(ticker: str, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    return _load_cached_memo(ticker, review_key(snapshot))
