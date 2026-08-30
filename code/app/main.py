from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import secrets
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
import pandas as pd
import yfinance as yf

from code.data.fetchers.nifty500 import fetch_nifty500_directory
from code.evaluator.memo import build_bundle, generate_memo, match_headlines
from code.pipeline.news_rss import fetch_news_items
from code.pipeline.nightly_pipeline import kite_fetch
from code.signals.mean_rev import generate_signals as generate_mean_rev_signals

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

ATR_WINDOW = 14
ATR_STOP_MULTIPLIER = 1.5  # per the locked risk rule: stop-loss = 1.5x ATR from entry

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"
SNAPSHOT_DIR = DATA_ROOT / "snapshots"
STALE_AFTER = timedelta(hours=24)

_TICKER_DIRECTORY_CACHE: Optional[List[Dict[str, str]]] = None


def _ticker_directory() -> List[Dict[str, str]]:
    """NIFTY 500 symbol+name list for the search/autocomplete box, cached in-process.

    The underlying fetch already has its own 24h on-disk cache; this just
    avoids re-parsing that ~500-row CSV on every single page view. The
    constituent list itself only changes on quarterly index reconstitution,
    so a per-process cache (cleared by a server restart) is plenty fresh.
    """
    global _TICKER_DIRECTORY_CACHE
    if _TICKER_DIRECTORY_CACHE is None:
        try:
            _TICKER_DIRECTORY_CACHE = fetch_nifty500_directory()
        except Exception:
            _TICKER_DIRECTORY_CACHE = []
    return _TICKER_DIRECTORY_CACHE


def _inject_ticker_directory(request: Request) -> Dict[str, Any]:
    return {"ticker_directory": _ticker_directory()}


TEMPLATES = Jinja2Templates(directory=str(ROOT / "templates"), context_processors=[_inject_ticker_directory])


def _normalize_ticker(raw: str) -> str:
    """"reliance", "RELIANCE ", "reliance.ns" -> "RELIANCE.NS".

    Mirrors the client-side normalization in base.html's search box, so a
    direct URL hit (bookmark, typed URL, old link) behaves the same as
    going through the search UI.
    """
    ticker = re.sub(r"\s+", "", raw).upper()
    if ticker and not ticker.endswith(".NS"):
        ticker = f"{ticker}.NS"
    return ticker


def _load_latest_snapshot() -> Dict[str, Any]:
    """Dashboard reads pre-computed JSON snapshots only -- never recomputes live."""
    files = sorted(SNAPSHOT_DIR.glob("*.json"))
    if not files:
        return {"screen": {"items": []}, "meta": {"generated_at": None, "status": "missing"}}
    with open(files[-1], "r", encoding="utf-8") as handle:
        return json.load(handle)


def _snapshot_is_stale(snapshot: Dict[str, Any]) -> bool:
    generated_at = snapshot.get("meta", {}).get("generated_at")
    if not generated_at:
        return True
    try:
        generated = datetime.fromisoformat(generated_at)
    except ValueError:
        return True
    return datetime.now() - generated > STALE_AFTER


def _run_nightly_pipeline_sync() -> None:
    """The actual pipeline call -- synchronous and slow (yfinance/OpenAI/RSS
    over the network), so every caller below runs it via asyncio.to_thread
    rather than blocking the event loop."""
    from code.pipeline.nightly_pipeline import run_pipeline
    try:
        logger.info("nightly pipeline run starting")
        run_pipeline()
        logger.info("nightly pipeline run finished OK")
    except Exception:
        logger.exception("nightly pipeline run failed")


async def _run_nightly_pipeline_job() -> None:
    await asyncio.to_thread(_run_nightly_pipeline_sync)


scheduler = AsyncIOScheduler(timezone=IST)

AUTH_USER = os.getenv("AUTH_USER")
AUTH_PASS = os.getenv("AUTH_PASS")
# Railway sets this automatically on every deployment -- the presence check,
# not its value, is what matters here.
RAILWAY_ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT")


def _enforce_auth_before_serving() -> None:
    """This app must never serve unauthenticated on Railway.

    If a Railway environment indicator is present but AUTH_USER/AUTH_PASS
    aren't both set, refuse to start at all -- a crashed deploy is loud and
    impossible to miss in Railway's dashboard; a running-but-open dashboard
    is not. Local dev (no Railway indicator) keeps the existing
    bypass-when-unset behavior unchanged.
    """
    if RAILWAY_ENVIRONMENT and not (AUTH_USER and AUTH_PASS):
        logger.critical(
            "REFUSING TO START: RAILWAY_ENVIRONMENT=%r is set (this is a Railway "
            "deployment) but AUTH_USER/AUTH_PASS are not both configured. This app "
            "must never serve unauthenticated in production. Set AUTH_USER and "
            "AUTH_PASS in Railway's Variables panel and redeploy.",
            RAILWAY_ENVIRONMENT,
        )
        raise RuntimeError(
            "Refusing to start: RAILWAY_ENVIRONMENT is set but AUTH_USER/AUTH_PASS "
            "are missing. Set both env vars in Railway and redeploy."
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _enforce_auth_before_serving()

    scheduler.add_job(
        _run_nightly_pipeline_job,
        CronTrigger(hour=18, minute=30, timezone=IST),
        id="nightly_1830_ist",
        name="Nightly pipeline (18:30 IST)",
        replace_existing=True,
    )
    scheduler.add_job(
        _run_nightly_pipeline_job,
        CronTrigger(hour=8, minute=30, timezone=IST),
        id="morning_refresh_0830_ist",
        name="Morning refresh (08:30 IST)",
        replace_existing=True,
    )
    scheduler.start()
    for job in scheduler.get_jobs():
        logger.info("scheduled job registered: %s (id=%s) next run: %s", job.name, job.id, job.next_run_time)

    # A deploy or restart can land at any time of day, hours from the next
    # 18:30/08:30 fire -- if the snapshot on disk is already missing or
    # stale, kick one immediate background run so the site isn't stuck on
    # "no data" for up to a full cycle. Fire-and-forget; never blocks startup.
    if _snapshot_is_stale(_load_latest_snapshot()):
        logger.info("snapshot missing/stale at startup -- kicking an immediate background pipeline run")
        asyncio.create_task(_run_nightly_pipeline_job())

    yield

    scheduler.shutdown(wait=False)


class BasicAuthMiddleware(BaseHTTPMiddleware):
    """HTTP Basic Auth gate for the whole app.

    Credentials come from AUTH_USER/AUTH_PASS env vars, read live -- never
    hardcoded. If neither is set (local dev), auth is skipped entirely, the
    same env-only/no-lockout pattern already used for the LLM API keys.
    """

    async def dispatch(self, request: Request, call_next):
        if not AUTH_USER or not AUTH_PASS:
            return await call_next(request)

        header = request.headers.get("authorization", "")
        if header.startswith("Basic "):
            try:
                decoded = base64.b64decode(header[6:]).decode("utf-8")
                username, _, password = decoded.partition(":")
            except Exception:
                username, password = "", ""
            if secrets.compare_digest(username, AUTH_USER) and secrets.compare_digest(password, AUTH_PASS):
                return await call_next(request)

        return Response(
            status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="Papa Terminal"'},
            content="Authentication required.",
        )


app = FastAPI(title="Papa Terminal", lifespan=lifespan)
app.add_middleware(BasicAuthMiddleware)


def _latest_signal_dict(asset: str, df: pd.DataFrame) -> Dict[str, Any]:
    """Live mean-reversion signal for the Stock Room's search-any-ticker box.

    Stock Room already fetches OHLCV live for the chart (a deliberate,
    pre-existing exception to "dashboard renders snapshots only" -- the
    search box must work for tickers outside the nightly universe). This
    reuses that same fetch rather than hitting the network again.
    """
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


def _compute_atr_stop(df: pd.DataFrame, window: int = ATR_WINDOW, multiplier: float = ATR_STOP_MULTIPLIER) -> Dict[str, Optional[float]]:
    """14-day ATR and a suggested stop (1.5x ATR below the last close).

    "Suggested" because Stock Room has no concept of Papa's actual entry
    price (decision support only, no position tracking) -- this is the
    stop distance from the last close, per the locked risk rule, not tied
    to a real open position.
    """
    if df.empty or len(df) < window + 1:
        return {"atr": None, "stop_price": None}

    high, low, close = df["high"].astype(float), df["low"].astype(float), df["close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(window=window, min_periods=window).mean().iloc[-1]
    if pd.isna(atr):
        return {"atr": None, "stop_price": None}

    last_close = float(close.iloc[-1])
    return {"atr": round(float(atr), 2), "stop_price": round(last_close - multiplier * float(atr), 2)}


def _find_screen_item(snapshot: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
    """This ticker's Minervini screen row from the latest snapshot, or None if
    it isn't in the NIFTY 500 screen universe (e.g. searched from outside it)."""
    for item in snapshot.get("screen", {}).get("items", []):
        if item.get("symbol") == ticker:
            return item
    return None


_REGIME_WORD = {"TRENDING_UP": "trending up", "TRENDING_DOWN": "trending down", "CHOPPY": "choppy"}

# Relabels the memo's locked verdict enum (TRADE_VALID/WAIT/AVOID -- never
# changed, the score<60 enforcement and tests key off these exact strings)
# to the trader-facing stance word Papa actually reads.
_VERDICT_TO_STANCE = {
    "TRADE_VALID": ("BULLISH SETUP", "good"),
    "WAIT": ("WATCH", "wait"),
    "AVOID": ("AVOID / NO SETUP", "bad"),
}


def _compute_stockroom_stance(
    memo: Dict[str, Any], screen_item: Optional[Dict[str, Any]], regime: Optional[str], signal: Dict[str, Any],
) -> Dict[str, Any]:
    """Stance word + the 2-3 signals behind it -- never a bare verdict.

    The word comes from the memo's own verdict (already the LLM's synthesis
    of template + regime + signal + headlines); the evidence line is built
    directly from the same underlying facts so it's checkable, not restated
    LLM prose.
    """
    if memo is None or memo.get("reasoning") == "Analysis unavailable":
        return {"stance": None, "stance_class": None, "stance_evidence": []}

    stance, stance_class = _VERDICT_TO_STANCE.get(memo.get("verdict"), ("WATCH", "wait"))

    evidence: List[str] = []
    if screen_item is not None:
        passed = screen_item.get("passed")
        conditions_passed = screen_item.get("conditions_passed")
        evidence.append(f"Stage 2 {'✓' if passed else f'{conditions_passed}/8'}")
        if screen_item.get("rs_rank") is not None:
            evidence.append(f"RS {screen_item['rs_rank']}")
    if regime:
        evidence.append(f"regime {_REGIME_WORD.get(regime, regime.lower())}")
    zscore = (signal or {}).get("metadata", {}).get("zscore")
    if zscore is not None and len(evidence) < 3:
        evidence.append(f"z-score {zscore:.2f}")
    if not evidence:
        evidence.append(f"memo score {memo.get('score')}")

    return {"stance": stance, "stance_class": stance_class, "stance_evidence": evidence[:3]}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    snapshot = _load_latest_snapshot()
    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "regime": snapshot.get("regime", "CHOPPY"),
            "indices": snapshot.get("indices", {}),
            "briefing": snapshot.get("briefing", {}).get("items", []),
            "action_queue": snapshot.get("action_queue", []),
            "generated_at": snapshot.get("meta", {}).get("generated_at"),
            "stale": _snapshot_is_stale(snapshot),
            "active_tab": "today",
        },
    )


@app.get("/stock/{asset}", response_class=HTMLResponse)
async def stock_room(asset: str, request: Request) -> HTMLResponse:
    asset = _normalize_ticker(asset)
    snapshot = _load_latest_snapshot()
    df = kite_fetch(asset)

    base_context = {
        "asset": asset,
        "generated_at": snapshot.get("meta", {}).get("generated_at"),
        "stale": _snapshot_is_stale(snapshot),
        "active_tab": "stock",
    }

    if df.empty:
        # Bad/unknown symbol -- a clean message, never a 500 or a blank page,
        # and skip the memo call entirely so a mistyped ticker doesn't burn
        # a real LLM request on nothing.
        return TEMPLATES.TemplateResponse(
            request,
            "stock_room.html",
            {
                **base_context,
                "not_found": True,
                "latest": None,
                "memo": None,
                "signal": None,
                "headlines": [],
                "delivery_pct": None,
                "atr": None,
                "atr_stop": None,
                "position_size": None,
                "stance": None,
                "stance_class": None,
                "stance_evidence": [],
            },
        )

    latest = df.iloc[-1]
    news_pool = fetch_news_items(limit=20)
    matched_headlines = match_headlines(asset, news_pool, limit=5)
    signal = _latest_signal_dict(asset, df)
    risk = _compute_atr_stop(df)
    regime = snapshot.get("regime")
    screen_item = _find_screen_item(snapshot, asset)
    trend_template = None
    if screen_item is not None:
        trend_template = {
            "passed": screen_item.get("passed"),
            "conditions_passed": screen_item.get("conditions_passed"),
            "rs_rank": screen_item.get("rs_rank"),
            "checklist": screen_item.get("checklist"),
        }
    bundle = build_bundle(
        asset, signal=signal, headlines=matched_headlines, diary=[],
        trend_template=trend_template, market_regime=regime, delivery_pct=None,
    )
    memo = generate_memo(bundle, asset, snapshot_date="today")
    stance = _compute_stockroom_stance(memo, screen_item, regime, signal)
    return TEMPLATES.TemplateResponse(
        request,
        "stock_room.html",
        {
            **base_context,
            "not_found": False,
            "latest": latest,
            "memo": memo,
            "signal": bundle["signal"],
            "headlines": bundle["headlines"],
            "delivery_pct": None,  # needs NSE bhavcopy -- not wired in v1, shown as "--" not a fake 0.0
            "atr": risk["atr"],
            "atr_stop": risk["stop_price"],
            "position_size": None,  # Papa Terminal doesn't track capital/portfolio size -- decision support only
            "stance": stance["stance"],
            "stance_class": stance["stance_class"],
            "stance_evidence": stance["stance_evidence"],
        },
    )


@app.get("/screen", response_class=HTMLResponse)
async def minervini_screen(request: Request) -> HTMLResponse:
    snapshot = _load_latest_snapshot()
    items = snapshot.get("screen", {}).get("items", [])
    generated_at = snapshot.get("meta", {}).get("generated_at")
    return TEMPLATES.TemplateResponse(
        request,
        "screen.html",
        {
            "items": items,
            "generated_at": generated_at,
            "stale": _snapshot_is_stale(snapshot),
            "active_tab": "screen",
        },
    )


@app.get("/pairs", response_class=HTMLResponse)
async def spread_trades(request: Request) -> HTMLResponse:
    snapshot = _load_latest_snapshot()
    pairs = snapshot.get("pairs", {})
    return TEMPLATES.TemplateResponse(
        request,
        "pairs.html",
        {
            "items": pairs.get("items", []),
            "last_scan": pairs.get("last_scan"),
            "generated_at": snapshot.get("meta", {}).get("generated_at"),
            "stale": _snapshot_is_stale(snapshot),
            "active_tab": "pairs",
        },
    )


@app.get("/diary", response_class=HTMLResponse)
async def diary(request: Request) -> HTMLResponse:
    snapshot = _load_latest_snapshot()
    return TEMPLATES.TemplateResponse(
        request,
        "diary.html",
        {
            "generated_at": snapshot.get("meta", {}).get("generated_at"),
            "stale": _snapshot_is_stale(snapshot),
            "active_tab": "diary",
        },
    )


@app.get("/api/candles/{asset}")
async def api_candles(asset: str) -> JSONResponse:
    df = kite_fetch(_normalize_ticker(asset))
    if df.empty:
        return JSONResponse({"candles": [], "sma50": [], "sma150": [], "sma200": [], "volume": []})

    # Lightweight Charts' numeric `time` type is Unix seconds, not milliseconds.
    candles = []
    for ts, row in df.iterrows():
        candles.append([int(pd.Timestamp(ts).value // 1_000_000_000), float(row["close"]), float(row["open"]), float(row["high"]), float(row["low"]), float(row["volume"])])

    sma50 = []
    sma150 = []
    sma200 = []
    for ts, row in df.iterrows():
        # rolling mean is NaN before min_periods bars accumulate -- JSONResponse
        # rejects NaN outright (Starlette calls json.dumps with allow_nan=False),
        # so those leading points are simply omitted rather than crashing the fetch.
        v50 = float(df["close"].rolling(50, min_periods=50).mean().loc[ts])
        v150 = float(df["close"].rolling(150, min_periods=150).mean().loc[ts])
        v200 = float(df["close"].rolling(200, min_periods=200).mean().loc[ts])
        ts_sec = int(pd.Timestamp(ts).value // 1_000_000_000)
        if not pd.isna(v50):
            sma50.append([ts_sec, v50])
        if not pd.isna(v150):
            sma150.append([ts_sec, v150])
        if not pd.isna(v200):
            sma200.append([ts_sec, v200])

    volume = []
    for ts, row in df.iterrows():
        volume.append([int(pd.Timestamp(ts).value // 1_000_000_000), float(row["volume"])])

    return JSONResponse({"candles": candles, "sma50": sma50, "sma150": sma150, "sma200": sma200, "volume": volume})
