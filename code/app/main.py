from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import bcrypt
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
import pandas as pd
import yfinance as yf

from code.data.fetchers.nifty500 import fetch_nifty500_directory
from code.evaluator.memo import build_bundle, generate_memo, match_headlines, _init_db as init_memo_db
from code.pipeline.news_rss import fetch_news_items
from code.pipeline.nightly_pipeline import kite_fetch
from code.signals.mean_rev import generate_signals as generate_mean_rev_signals
from code.trading.portal import place_trade, close_trade, get_trades, get_trade_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")

ATR_WINDOW = 14
ATR_STOP_MULTIPLIER = 1.5  # per the locked risk rule: stop-loss = 1.5x ATR from entry

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT.parent / "data"
SNAPSHOT_DIR = DATA_ROOT / "snapshots"
DB_PATH = DATA_ROOT / "papa.db"  # same file evaluator/memo.py's cache uses -- same Railway volume, persists across restarts
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


def _evaluate_screen_item_live(ticker: str, df: pd.DataFrame, regime: Optional[str]) -> Optional[Dict[str, Any]]:
    """Build a screen-row-shaped dict for a ticker the nightly screen never saw.

    Same evaluator the pipeline runs (evaluate_trend_template), just fed this
    one stock's freshly-fetched bars. Returns None if it can't be evaluated at
    all, so callers fall back to the no-checklist state rather than showing a
    row of false negatives.
    """
    if df.empty or "close" not in df.columns:
        return None
    try:
        from code.signals.minervini.trend_template import evaluate_trend_template
        result = evaluate_trend_template(df, rs_rank=None)
    except Exception as exc:
        logger.warning("live trend-template evaluation failed for %s: %s", ticker, exc)
        return None
    if result.get("insufficient_history"):
        return None

    item: Dict[str, Any] = {
        "symbol": ticker,
        "rs_rank": None,
        "passed": result["passed"],
        "conditions_passed": result["conditions_passed"],
        "checklist": result["checklist"],
        # Flags this row as computed on request rather than read from the
        # nightly snapshot -- the UI says so, so an RS-less checklist isn't
        # mistaken for a screened result.
        "computed_live": True,
    }
    try:
        from code.pipeline.nightly_pipeline import _compute_screen_stance
        item.update(_compute_screen_stance(item, regime or "CHOPPY"))
    except Exception as exc:
        logger.warning("live stance computation failed for %s: %s", ticker, exc)
    return item


def _drop_incomplete_ohlcv_rows(df: pd.DataFrame) -> pd.DataFrame:
    """kite_fetch's raw yfinance pull can carry a trailing row with OHLC=NaN
    (volume already populated) when the market is still open or today's bar
    hasn't settled yet -- a candle needs all four prices, so that row is
    unusable, not just incomplete. Dropping it here, once, keeps every
    downstream consumer (the chart's /api/candles, the O/H/L/C stat line,
    DMA rolling windows) from ever seeing it, instead of each guessing at
    NaN-handling independently."""
    return df.dropna(subset=["open", "high", "low", "close", "volume"])


def _load_latest_snapshot() -> Dict[str, Any]:
    """Dashboard reads pre-computed JSON snapshots only -- never recomputes live."""
    files = sorted(SNAPSHOT_DIR.glob("*.json"))
    if not files:
        return {"screen": {"items": []}, "meta": {"generated_at": None, "status": "missing"}}
    with open(files[-1], "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_previous_snapshot() -> Optional[Dict[str, Any]]:
    """The snapshot immediately before the latest one, for day-over-day
    diffing. None if there isn't a second snapshot yet -- a fresh deploy or
    a single pipeline run so far -- never an error."""
    files = sorted(SNAPSHOT_DIR.glob("*.json"))
    if len(files) < 2:
        return None
    try:
        with open(files[-2], "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def _diff_screen_items(today_items: List[Dict[str, Any]], previous_snapshot: Optional[Dict[str, Any]]) -> "tuple[Dict[str, Dict[str, Any]], List[str]]":
    """symbol -> {"badge", "badge_class", "detail", "rs_delta"} for today's
    rows, plus the list of symbols present yesterday but absent today
    (DROPPED -- delisted, or fell out of the fetchable universe for a day).
    No previous snapshot -> ({}, []), never an error.
    """
    if previous_snapshot is None:
        return {}, []

    prev_items = {i["symbol"]: i for i in previous_snapshot.get("screen", {}).get("items", [])}
    today_symbols = {i["symbol"] for i in today_items}

    diffs: Dict[str, Dict[str, Any]] = {}
    for item in today_items:
        sym = item["symbol"]
        prev = prev_items.get(sym)

        rs_delta = None
        if prev and prev.get("rs_rank") is not None and item.get("rs_rank") is not None:
            rs_delta = item["rs_rank"] - prev["rs_rank"]

        if prev is None:
            diffs[sym] = {"badge": "NEW", "badge_class": "good", "detail": None, "rs_delta": rs_delta}
        elif item["conditions_passed"] > prev["conditions_passed"]:
            diffs[sym] = {
                "badge": "UPGRADED", "badge_class": "good",
                "detail": f"{prev['conditions_passed']}/8→{item['conditions_passed']}/8",
                "rs_delta": rs_delta,
            }
        elif item["conditions_passed"] < prev["conditions_passed"]:
            diffs[sym] = {
                "badge": "DOWNGRADED", "badge_class": "bad",
                "detail": f"{prev['conditions_passed']}/8→{item['conditions_passed']}/8",
                "rs_delta": rs_delta,
            }
        else:
            diffs[sym] = {"badge": None, "badge_class": None, "detail": None, "rs_delta": rs_delta}

    dropped = sorted(sym for sym in prev_items if sym not in today_symbols)
    return diffs, dropped


# Seed list for Dad's Watchlist -- edit this to change who's tracked. Only
# used to populate the SQLite table on its very first run; after that,
# storage is the source of truth so future growth (a picked stock added
# later) survives restarts and isn't wiped by editing this constant.
DEFAULT_WATCHLIST = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS"]

# BULLISH SETUP > WATCH > AVOID / NO SETUP -- ordinal so yesterday->today
# stance changes can be read as "better" or "worse", not just "different".
_STANCE_RANK = {"AVOID / NO SETUP": 0, "WATCH": 1, "BULLISH SETUP": 2}


def _init_watchlist_db() -> None:
    """Creates the watchlist table if needed and seeds it from
    DEFAULT_WATCHLIST exactly once -- only when the table is empty, so
    edits/growth already in storage are never overwritten."""
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS watchlist (symbol TEXT PRIMARY KEY, added_at TEXT NOT NULL)")
        (count,) = conn.execute("SELECT COUNT(*) FROM watchlist").fetchone()
        if count == 0:
            now = datetime.now().isoformat()
            conn.executemany(
                "INSERT OR IGNORE INTO watchlist (symbol, added_at) VALUES (?, ?)",
                [(sym, now) for sym in DEFAULT_WATCHLIST],
            )
            conn.commit()
    finally:
        conn.close()


def _get_watchlist_symbols() -> List[str]:
    conn = sqlite3.connect(DB_PATH)
    try:
        rows = conn.execute("SELECT symbol FROM watchlist ORDER BY added_at").fetchall()
    finally:
        conn.close()
    return [row[0] for row in rows]


def _add_to_watchlist(symbol: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            "INSERT OR IGNORE INTO watchlist (symbol, added_at) VALUES (?, ?)",
            (symbol, datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def _remove_from_watchlist(symbol: str) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("DELETE FROM watchlist WHERE symbol = ?", (symbol,))
        conn.commit()
    finally:
        conn.close()


def _build_watchlist_section(
    today_items: List[Dict[str, Any]], previous_snapshot: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Dad's watchlist: stored tickers (SQLite-backed, seeded from
    DEFAULT_WATCHLIST) with each one's stance yesterday->today -- reuses the
    exact same previous-snapshot lookup as the Minervini day-over-day diff.
    """
    today_by_symbol = {i["symbol"]: i for i in today_items}
    prev_items = previous_snapshot.get("screen", {}).get("items", []) if previous_snapshot else []
    prev_by_symbol = {i["symbol"]: i for i in prev_items}

    rows: List[Dict[str, Any]] = []
    for symbol in _get_watchlist_symbols():
        today_item = today_by_symbol.get(symbol)
        prev_item = prev_by_symbol.get(symbol)
        stance_today = today_item.get("stance") if today_item else None
        stance_yesterday = prev_item.get("stance") if prev_item else None

        arrow, arrow_class = "—", "neutral"
        rank_today = _STANCE_RANK.get(stance_today)
        rank_yesterday = _STANCE_RANK.get(stance_yesterday)
        if rank_today is not None and rank_yesterday is not None:
            if rank_today > rank_yesterday:
                arrow, arrow_class = "↑", "good"
            elif rank_today < rank_yesterday:
                arrow, arrow_class = "↓", "bad"

        rows.append({
            "symbol": symbol,
            "in_screen": today_item is not None,
            "stance_today": stance_today,
            "stance_today_class": today_item.get("stance_class") if today_item else None,
            "stance_yesterday": stance_yesterday,
            "arrow": arrow,
            "arrow_class": arrow_class,
        })
    return rows


def _snapshot_is_stale(snapshot: Dict[str, Any]) -> bool:
    generated_at = snapshot.get("meta", {}).get("generated_at")
    if not generated_at:
        return True
    try:
        generated = datetime.fromisoformat(generated_at)
    except ValueError:
        return True
    return datetime.now() - generated > STALE_AFTER


def _freshness_labels(snapshot: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Two readings of the one generated_at timestamp this pipeline actually
    has -- 'market data' (is tonight's price data current) and 'analysis'
    (how long ago the pipeline computed everything from it) happen
    atomically together here, but showing them separately answers Papa's
    real question -- "is the PRICE stale or is the VERDICT stale" -- instead
    of one ambiguous blob.
    """
    generated_at = snapshot.get("meta", {}).get("generated_at")
    if not generated_at:
        return {"market_data_label": None, "analysis_age_label": None}
    try:
        generated = datetime.fromisoformat(generated_at)
    except ValueError:
        return {"market_data_label": None, "analysis_age_label": None}

    delta = datetime.now() - generated
    hours = delta.total_seconds() / 3600
    if hours < 1:
        age_label = f"{max(1, int(delta.total_seconds() / 60))}m old"
    else:
        age_label = f"{hours:.0f}h old"

    return {
        "market_data_label": "CURRENT" if delta <= STALE_AFTER else None,
        "analysis_age_label": age_label,
    }


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


def _run_pairs_monthly_rescan_sync() -> None:
    """The expensive sector-bucketed cointegration search -- monthly only,
    same asyncio.to_thread pattern as the nightly job since it's also slow
    network+CPU work that must not block the event loop."""
    from code.pipeline.nightly_pipeline import run_pairs_monthly_rescan
    try:
        logger.info("pairs monthly rescan starting")
        result = run_pairs_monthly_rescan()
        logger.info(
            "pairs monthly rescan finished OK: tested=%d passed=%d kept=%d elapsed=%.1fs",
            result["candidates_tested"], result["candidates_passed"], len(result["pairs"]), result["elapsed_seconds"],
        )
    except Exception:
        logger.exception("pairs monthly rescan failed")


async def _run_pairs_monthly_rescan_job() -> None:
    await asyncio.to_thread(_run_pairs_monthly_rescan_sync)


scheduler = AsyncIOScheduler(timezone=IST)

APP_PIN_HASH = os.getenv("APP_PIN")  # a bcrypt hash of the PIN -- never the raw digits
SESSION_SECRET = os.getenv("SESSION_SECRET")
# Railway sets this automatically on every deployment -- the presence check,
# not its value, is what matters here.
RAILWAY_ENVIRONMENT = os.getenv("RAILWAY_ENVIRONMENT")

SESSION_COOKIE = "papa_session"
SESSION_MAX_AGE = 30 * 24 * 3600  # ~30 days -- enter the PIN once, stay in

RATE_LIMIT_MAX_ATTEMPTS = 5
RATE_LIMIT_LOCKOUT_SECONDS = 90
# In-memory, per-process -- fine for a single-worker personal deploy; a
# restart clears it, which is acceptable (worst case: 5 free attempts again).
_pin_attempts: Dict[str, Dict[str, float]] = {}


def _enforce_auth_before_serving() -> None:
    """This app must never serve unauthenticated on Railway.

    If a Railway environment indicator is present but APP_PIN/SESSION_SECRET
    aren't both set, refuse to start at all -- a crashed deploy is loud and
    impossible to miss in Railway's dashboard; a running-but-open dashboard
    is not. Local dev (no Railway indicator) keeps the existing
    bypass-when-unset behavior unchanged.
    """
    if RAILWAY_ENVIRONMENT and not (APP_PIN_HASH and SESSION_SECRET):
        logger.critical(
            "REFUSING TO START: RAILWAY_ENVIRONMENT=%r is set (this is a Railway "
            "deployment) but APP_PIN/SESSION_SECRET are not both configured. This app "
            "must never serve unauthenticated in production. Set APP_PIN (a bcrypt "
            "hash of the PIN -- never the raw digits) and SESSION_SECRET in Railway's "
            "Variables panel and redeploy.",
            RAILWAY_ENVIRONMENT,
        )
        raise RuntimeError(
            "Refusing to start: RAILWAY_ENVIRONMENT is set but APP_PIN/SESSION_SECRET "
            "are missing. Set both env vars in Railway and redeploy."
        )


def _rate_limit_key(request: Request) -> str:
    return request.client.host if request.client else "unknown"


def _is_locked_out(key: str) -> Optional[int]:
    """Seconds remaining if this key is currently locked out, else None.
    Clears expired lockouts so the next attempt starts a fresh count.

    locked_until == 0 means "accumulating failures but hasn't hit the
    threshold yet", not "was locked, now expired" -- must not be treated
    the same as a real expired lock, or every check-in-between-attempts
    would wipe the in-progress fail count before it ever reaches the
    threshold.
    """
    state = _pin_attempts.get(key)
    if not state or not state["locked_until"]:
        return None
    remaining = state["locked_until"] - time.time()
    if remaining > 0:
        return int(remaining) + 1
    _pin_attempts.pop(key, None)
    return None


def _record_failed_attempt(key: str) -> int:
    """Increments the fail count, locking out once RATE_LIMIT_MAX_ATTEMPTS is
    hit. Returns attempts remaining before that lockout (0 once locked)."""
    state = _pin_attempts.setdefault(key, {"fails": 0, "locked_until": 0.0})
    state["fails"] += 1
    remaining = max(0, RATE_LIMIT_MAX_ATTEMPTS - state["fails"])
    if state["fails"] >= RATE_LIMIT_MAX_ATTEMPTS:
        state["locked_until"] = time.time() + RATE_LIMIT_LOCKOUT_SECONDS
        state["fails"] = 0  # the next window (after the lock clears) starts clean
    return remaining


def _clear_attempts(key: str) -> None:
    _pin_attempts.pop(key, None)


def _check_pin(pin: str) -> bool:
    if not APP_PIN_HASH:
        return False
    try:
        return bcrypt.checkpw(pin.encode("utf-8"), APP_PIN_HASH.encode("utf-8"))
    except (ValueError, TypeError) as exc:
        logger.error("APP_PIN is not a valid bcrypt hash: %s", exc)
        return False


def _sign_session(expiry: int) -> str:
    payload = str(expiry)
    sig = hmac.new(SESSION_SECRET.encode("utf-8"), payload.encode("utf-8"), "sha256").hexdigest()
    return f"{payload}.{sig}"


def _verify_session(cookie_value: Optional[str]) -> bool:
    if not cookie_value or not SESSION_SECRET:
        return False
    payload, _, sig = cookie_value.partition(".")
    if not payload or not sig:
        return False
    expected = hmac.new(SESSION_SECRET.encode("utf-8"), payload.encode("utf-8"), "sha256").hexdigest()
    if not hmac.compare_digest(sig, expected):
        return False
    try:
        return int(payload) > time.time()
    except ValueError:
        return False


def _safe_next(next_path: str) -> str:
    """Only ever redirect somewhere on this same site -- next is
    user-controllable (query/form param), so an unvalidated value is an
    open-redirect vector."""
    if next_path and next_path.startswith("/") and not next_path.startswith("//"):
        return next_path
    return "/"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    _enforce_auth_before_serving()
    _init_watchlist_db()
    init_memo_db()  # Creates both memos and paper_trades tables

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
    scheduler.add_job(
        _run_pairs_monthly_rescan_job,
        CronTrigger(day=1, hour=20, minute=0, timezone=IST),
        id="pairs_monthly_rescan",
        name="Pairs cointegration rescan (1st of month, 20:00 IST)",
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

    # The RSS pull is the single slowest thing on a Stock Room view (measured
    # 4.3-29.5s cold) and it's identical for every ticker, so warm it once in
    # the background rather than making whoever opens the first stock pay it.
    async def _warm_news_cache() -> None:
        try:
            await asyncio.to_thread(fetch_news_items, 20)
            logger.info("news cache warmed at startup")
        except Exception:
            logger.exception("news cache warm failed (non-fatal)")

    asyncio.create_task(_warm_news_cache())

    from code.pipeline.nightly_pipeline import PAIRS_CACHE_PATH
    if not PAIRS_CACHE_PATH.exists():
        logger.info("pairs cointegration cache missing at startup -- kicking an immediate background monthly rescan")
        asyncio.create_task(_run_pairs_monthly_rescan_job())

    yield

    scheduler.shutdown(wait=False)


PUBLIC_AUTH_PATHS = {"/login", "/logout"}


class SessionAuthMiddleware(BaseHTTPMiddleware):
    """PIN-session gate for the whole app.

    APP_PIN holds a bcrypt HASH of the PIN (checked in the /login route),
    never the raw digits -- this middleware only ever verifies the signed
    session cookie /login hands out. If APP_PIN isn't set at all (local
    dev), auth is skipped entirely, the same env-only/no-lockout-when-unset
    pattern already used for the LLM API keys.
    """

    async def dispatch(self, request: Request, call_next):
        if not APP_PIN_HASH:
            return await call_next(request)

        if request.url.path in PUBLIC_AUTH_PATHS:
            return await call_next(request)

        if _verify_session(request.cookies.get(SESSION_COOKIE)):
            return await call_next(request)

        if request.url.path.startswith("/api/"):
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)

        next_qs = f"?next={request.url.path}" if request.url.path != "/" else ""
        return RedirectResponse(url=f"/login{next_qs}", status_code=303)


class RequestTimingMiddleware(BaseHTTPMiddleware):
    """Logs wall-clock time per request.

    Cheap (one perf_counter pair) and always on -- "the dashboard feels slow"
    is otherwise unfalsifiable, and the expensive work here (yfinance, RSS,
    OpenAI) is all network-bound and invisible without a number next to it.
    """

    async def dispatch(self, request: Request, call_next):
        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - started) * 1000
        logger.info("TIMING %s %s -> %d in %.0fms", request.method, request.url.path, response.status_code, elapsed_ms)
        response.headers["Server-Timing"] = f"total;dur={elapsed_ms:.0f}"
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Standard hardening headers on every response, auth-gated or not."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        if RAILWAY_ENVIRONMENT:
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        return response


app = FastAPI(title="Papa Terminal", lifespan=lifespan)
app.add_middleware(RequestTimingMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(SessionAuthMiddleware)


@app.get("/login", response_class=HTMLResponse)
async def login_form(request: Request, next: str = "/") -> Response:
    if not APP_PIN_HASH or _verify_session(request.cookies.get(SESSION_COOKIE)):
        return RedirectResponse(url=_safe_next(next))
    locked_seconds = _is_locked_out(_rate_limit_key(request))
    return TEMPLATES.TemplateResponse(
        request, "login.html",
        {"next": _safe_next(next), "error": False, "attempts_left": None, "locked_seconds": locked_seconds},
    )


@app.post("/login")
async def login_submit(request: Request, pin: str = Form(...), next: str = Form("/")) -> Response:
    key = _rate_limit_key(request)
    next_url = _safe_next(next)
    locked_seconds = _is_locked_out(key)
    if locked_seconds:
        return TEMPLATES.TemplateResponse(
            request, "login.html",
            {"next": next_url, "error": False, "attempts_left": None, "locked_seconds": locked_seconds},
            status_code=429,
        )

    if not _check_pin(pin):
        attempts_left = _record_failed_attempt(key)
        locked_seconds = _is_locked_out(key)
        return TEMPLATES.TemplateResponse(
            request, "login.html",
            {
                "next": next_url, "error": True,
                "attempts_left": attempts_left if not locked_seconds else None,
                "locked_seconds": locked_seconds,
            },
            status_code=401,
        )

    _clear_attempts(key)
    expiry = int(time.time()) + SESSION_MAX_AGE
    response = RedirectResponse(url=next_url, status_code=303)
    response.set_cookie(
        SESSION_COOKIE, _sign_session(expiry),
        max_age=SESSION_MAX_AGE, httponly=True, samesite="lax",
        secure=bool(RAILWAY_ENVIRONMENT), path="/",
    )
    return response


@app.get("/logout")
async def logout() -> RedirectResponse:
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return response


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

# Engine-internal strategy ids (Signal.strategy) -> what Papa should read.
# The ids are a locked contract in code/signals/*.py and stay as they are;
# this is purely the label shown on screen.
_STRATEGY_LABELS = {
    "mean_reversion_zscore": "Mean reversion",
    "pairs_trading_zscore": "Pair spread",
    "garch_volatility": "Volatility",
}


def _strategy_label(strategy: Optional[str]) -> str:
    if not strategy:
        return "No signal"
    return _STRATEGY_LABELS.get(strategy, strategy.replace("_", " ").capitalize())

# Relabels the memo's locked verdict enum (TRADE_VALID/WAIT/AVOID -- never
# changed, the score<60 enforcement and tests key off these exact strings)
# to the trader-facing stance word Papa actually reads.
_VERDICT_TO_STANCE = {
    "TRADE_VALID": ("BULLISH SETUP", "good"),
    "WAIT": ("WATCH", "wait"),
    "AVOID": ("NO SETUP", "neutral"),
    "NO_CLEAR_EDGE": ("NO CLEAR EDGE", "neutral"),
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
    if memo is None or memo.get("sentinel"):
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


def _setup_quality(score: Optional[int]) -> Dict[str, Optional[str]]:
    """Reframes the bare 0-100 memo score as a quality word -- STRONG/MODERATE/
    WEAK are about how clean the setup itself is, not a probability of profit
    (that distinction matters enough to print on the page, not just imply).
    Reuses the same 60/80 boundaries already drawn on the score gauge.
    """
    if score is None:
        return {"word": None, "class": None}
    if score >= 80:
        return {"word": "STRONG", "class": "good"}
    if score >= 60:
        return {"word": "MODERATE", "class": "wait"}
    return {"word": "WEAK", "class": "bad"}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request) -> HTMLResponse:
    snapshot = _load_latest_snapshot()
    screen_items = snapshot.get("screen", {}).get("items", [])
    previous_snapshot = _load_previous_snapshot()
    watchlist_rows = _build_watchlist_section(screen_items, previous_snapshot)
    diffs, _dropped = _diff_screen_items(screen_items, previous_snapshot)

    # "Strongest setups to review": same screen data as the shortlist below,
    # just re-ranked by stance (regime fit) first and RS second using the
    # existing stance ordinal table (_STANCE_RANK, already used for the
    # watchlist's yesterday->today arrow) -- no new scoring, just a sort key.
    passed_items = [item for item in screen_items if item.get("passed")]
    top_today = sorted(
        passed_items,
        key=lambda item: (_STANCE_RANK.get(item.get("stance"), 0), item.get("rs_rank") or 0),
        reverse=True,
    )[:5]

    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "regime": snapshot.get("regime", "CHOPPY"),
            "indices": snapshot.get("indices", {}),
            "briefing": snapshot.get("briefing", {}).get("items", []),
            "action_queue": snapshot.get("action_queue", []),
            "screen_items": screen_items,
            "top_today": top_today,
            "diffs": diffs,
            "watchlist_rows": watchlist_rows,
            "generated_at": snapshot.get("meta", {}).get("generated_at"),
            "stale": _snapshot_is_stale(snapshot),
            "active_tab": "today",
            **_freshness_labels(snapshot),
        },
    )


@app.post("/api/watchlist/add")
async def watchlist_add(symbol: str = Form(...), next: str = Form("/")) -> RedirectResponse:
    _add_to_watchlist(_normalize_ticker(symbol))
    return RedirectResponse(url=next, status_code=303)


@app.post("/api/watchlist/remove")
async def watchlist_remove(symbol: str = Form(...), next: str = Form("/")) -> RedirectResponse:
    _remove_from_watchlist(_normalize_ticker(symbol))
    return RedirectResponse(url=next, status_code=303)


@app.get("/stock/{asset}", response_class=HTMLResponse)
async def stock_room(asset: str, request: Request) -> HTMLResponse:
    asset = _normalize_ticker(asset)
    snapshot = _load_latest_snapshot()
    _t = time.perf_counter()
    df = _drop_incomplete_ohlcv_rows(kite_fetch(asset))
    _phase_fetch = (time.perf_counter() - _t) * 1000

    base_context = {
        "asset": asset,
        "generated_at": snapshot.get("meta", {}).get("generated_at"),
        "stale": _snapshot_is_stale(snapshot),
        "active_tab": "stock",
        "in_watchlist": asset in _get_watchlist_symbols(),
        **_freshness_labels(snapshot),
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
                "quality_word": None,
                "quality_class": None,
                "screen_item": None,
                "regime": None,
            },
        )

    latest = df.iloc[-1]
    _t = time.perf_counter()
    news_pool = fetch_news_items(limit=20)
    matched_headlines = match_headlines(asset, news_pool, limit=5)
    _phase_news = (time.perf_counter() - _t) * 1000
    _t = time.perf_counter()
    signal = _latest_signal_dict(asset, df)
    risk = _compute_atr_stop(df)
    _phase_signal = (time.perf_counter() - _t) * 1000
    regime = snapshot.get("regime")
    screen_item = _find_screen_item(snapshot, asset)
    if screen_item is None:
        # Off-universe ticker (KRBL, RISHABH, anything outside the NIFTY 500
        # screen): the 8-point template is a pure function of this stock's own
        # OHLCV, which we already have, so evaluate it live rather than
        # showing an empty checklist. RS rank is the one thing that genuinely
        # can't be computed here -- it's a ranking *against the universe* --
        # so it stays None and the template says so instead of inventing one.
        screen_item = _evaluate_screen_item_live(asset, df, regime)
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
    _t = time.perf_counter()
    memo = generate_memo(bundle, asset, snapshot_date="today")
    _phase_memo = (time.perf_counter() - _t) * 1000
    logger.info(
        "TIMING /stock/%s phases: fetch=%.0fms news=%.0fms signal=%.0fms memo=%.0fms",
        asset, _phase_fetch, _phase_news, _phase_signal, _phase_memo,
    )
    stance = _compute_stockroom_stance(memo, screen_item, regime, signal)
    quality = _setup_quality(memo.get("score") if not memo.get("sentinel") else None)
    return TEMPLATES.TemplateResponse(
        request,
        "stock_room.html",
        {
            **base_context,
            "not_found": False,
            "latest": latest,
            "memo": memo,
            "signal": bundle["signal"],
            "signal_label": _strategy_label((bundle["signal"] or {}).get("strategy")),
            "headlines": bundle["headlines"],
            "delivery_pct": None,  # needs NSE bhavcopy -- not wired in v1, shown as "--" not a fake 0.0
            "atr": risk["atr"],
            "atr_stop": risk["stop_price"],
            "position_size": None,  # Papa Terminal doesn't track capital/portfolio size -- decision support only
            "stance": stance["stance"],
            "stance_class": stance["stance_class"],
            "stance_evidence": stance["stance_evidence"],
            "quality_word": quality["word"],
            "quality_class": quality["class"],
            "screen_item": screen_item,
            "regime": regime,
        },
    )


@app.get("/screen", response_class=HTMLResponse)
async def minervini_screen(request: Request) -> HTMLResponse:
    snapshot = _load_latest_snapshot()
    items = snapshot.get("screen", {}).get("items", [])
    generated_at = snapshot.get("meta", {}).get("generated_at")
    diffs, dropped = _diff_screen_items(items, _load_previous_snapshot())
    return TEMPLATES.TemplateResponse(
        request,
        "screen.html",
        {
            "items": items,
            "diffs": diffs,
            "dropped": dropped,
            "generated_at": generated_at,
            "stale": _snapshot_is_stale(snapshot),
            "active_tab": "screen",
            "regime": snapshot.get("regime", "CHOPPY"),
            "watchlist_symbols": set(_get_watchlist_symbols()),
            **_freshness_labels(snapshot),
        },
    )


@app.get("/pairs", response_class=HTMLResponse)
async def spread_trades(request: Request) -> HTMLResponse:
    snapshot = _load_latest_snapshot()
    pairs = snapshot.get("pairs", {})

    from code.pipeline.nightly_pipeline import PAIRS_CACHE_PATH
    rescan_stats = None
    if PAIRS_CACHE_PATH.exists():
        try:
            cache = json.loads(PAIRS_CACHE_PATH.read_text(encoding="utf-8"))
            rescan_stats = {
                "tested": cache.get("candidates_tested"),
                "passed": cache.get("candidates_passed"),
                "rescanned_at": cache.get("generated_at"),
            }
        except (json.JSONDecodeError, OSError):
            pass

    return TEMPLATES.TemplateResponse(
        request,
        "pairs.html",
        {
            "items": pairs.get("items", []),
            "last_scan": pairs.get("last_scan"),
            "rescan_stats": rescan_stats,
            "generated_at": snapshot.get("meta", {}).get("generated_at"),
            "stale": _snapshot_is_stale(snapshot),
            "active_tab": "pairs",
            **_freshness_labels(snapshot),
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
            **_freshness_labels(snapshot),
        },
    )


@app.get("/api/screen-row/{symbol}", response_class=HTMLResponse)
async def api_screen_row(symbol: str, request: Request) -> HTMLResponse:
    """The expanded checklist for one screen row, fetched when it's opened.

    Same pre-computed snapshot the table itself renders from -- this only
    changes *when* the markup is built, so the 500-row page ships summaries
    instead of 2.7MB of collapsed detail nobody has looked at yet.
    """
    symbol = _normalize_ticker(symbol)
    item = _find_screen_item(_load_latest_snapshot(), symbol)
    if item is None:
        return HTMLResponse('<div class="row-expand-inner">No checklist for this stock.</div>', status_code=404)
    return TEMPLATES.TemplateResponse(
        request,
        "_screen_row_detail.html",
        {"item": item, "on_watchlist": symbol in _get_watchlist_symbols()},
    )


@app.get("/api/candles/{asset}")
async def api_candles(asset: str) -> JSONResponse:
    df = _drop_incomplete_ohlcv_rows(kite_fetch(_normalize_ticker(asset)))
    if df.empty:
        return JSONResponse({"candles": [], "sma50": [], "sma150": [], "sma200": [], "volume": []})

    # Lightweight Charts' numeric `time` type is Unix seconds, not milliseconds.
    # Normalise to ns first: the index comes back from the parquet cache as
    # datetime64[ms], so a raw .view("int64") here yields milliseconds and the
    # //1e9 silently produced timestamps ~1e6x too small (1725 vs 1725494400),
    # which renders as a completely blank chart.
    secs = (df.index.astype("datetime64[ns]").astype("int64") // 1_000_000_000).tolist()
    closes = df["close"].astype(float).tolist()
    opens = df["open"].astype(float).tolist()
    highs = df["high"].astype(float).tolist()
    lows = df["low"].astype(float).tolist()
    volumes = df["volume"].astype(float).tolist()

    candles = [[t, c, o, h, l, v] for t, c, o, h, l, v in zip(secs, closes, opens, highs, lows, volumes)]
    volume = [[t, v] for t, v in zip(secs, volumes)]

    # Each rolling mean is computed once over the whole series, not re-derived
    # per row -- the previous per-row form recomputed all three full rolling
    # windows on every bar (O(n^2): ~1500 full passes over 500 bars), which
    # was the bulk of this endpoint's time.
    # rolling mean is NaN before min_periods bars accumulate -- JSONResponse
    # rejects NaN outright (Starlette calls json.dumps with allow_nan=False),
    # so those leading points are simply omitted rather than crashing the fetch.
    def _sma_points(window: int) -> List[List[float]]:
        values = df["close"].rolling(window, min_periods=window).mean()
        return [[t, float(v)] for t, v in zip(secs, values.tolist()) if not pd.isna(v)]

    return JSONResponse({
        "candles": candles,
        "sma50": _sma_points(50),
        "sma150": _sma_points(150),
        "sma200": _sma_points(200),
        "volume": volume,
    })


@app.get("/api/memo/{ticker}", response_class=HTMLResponse)
async def api_memo_lazy(ticker: str) -> HTMLResponse:
    """Lazy-load memo HTML for a stock. Used by stock_room.html to defer LLM call."""
    ticker = _normalize_ticker(ticker)
    df = kite_fetch(ticker)
    if df.empty:
        return HTMLResponse("<p>Could not load memo — no data for this stock.</p>", status_code=404)

    news_pool = fetch_news_items(limit=20)
    matched_headlines = match_headlines(ticker, news_pool, limit=5)
    signal = _latest_signal_dict(ticker, df)

    snapshot = _load_latest_snapshot()
    screen_item = _find_screen_item(snapshot, ticker)
    regime = snapshot.get("regime", "CHOPPY")

    trend_template = None
    if screen_item:
        trend_template = {
            "passed": screen_item.get("passed"),
            "conditions_passed": screen_item.get("conditions_passed"),
            "rs_rank": screen_item.get("rs_rank"),
            "checklist": screen_item.get("checklist"),
        }

    bundle = build_bundle(
        ticker, signal=signal, headlines=matched_headlines, diary=[],
        trend_template=trend_template, market_regime=regime, delivery_pct=None,
    )
    memo = generate_memo(bundle, ticker, snapshot_date="today")
    stance = _compute_stockroom_stance(memo, screen_item, regime, signal)
    quality = _setup_quality(memo.get("score") if not memo.get("sentinel") else None)

    html = f"""
    <div class="card">
      <h3 class="supporting">Setup Review</h3>
      <div class="verdict-top">
        <div class="verdict-score verdict-{stance['stance_class']}">
          <span class="score-number">{memo.get('score', 0)}</span>
          <span class="score-label">{quality}</span>
        </div>
        <div>
          <div class="verdict-rule">{stance['stance']}</div>
          <div style="font-size: 14px; color: var(--muted); margin-top: 4px;">{', '.join(stance.get('stance_evidence', []))}</div>
        </div>
      </div>
      <div class="memo-body">{memo.get('reasoning', 'No analysis available.')}</div>
      {f'<div style="color: var(--red); font-size: 14px; margin-top: 12px;"><strong>Flags:</strong> {", ".join(memo.get("risk_flags", []))}</div>' if memo.get('risk_flags') else ''}
    </div>
    """
    return HTMLResponse(html)


@app.get("/predictions", response_class=HTMLResponse)
async def predictions(request: Request) -> HTMLResponse:
    """Trading portal with next-day predictions and paper trading."""
    snapshot = _load_latest_snapshot()
    regime = snapshot.get("regime", "CHOPPY")

    # Get signal setups from the action queue (which has trade ideas)
    action_queue = snapshot.get("action_queue", [])

    # Get trade stats
    stats = get_trade_stats()
    open_trades = get_trades(limit=10, status="open")
    closed_trades = get_trades(limit=20, status="closed")

    return TEMPLATES.TemplateResponse(
        request,
        "predictions.html",
        {
            "active_tab": "predictions",
            "regime": regime,
            "action_queue": action_queue,
            "stats": stats,
            "open_trades": open_trades,
            "closed_trades": closed_trades,
            **_freshness_labels(snapshot),
        },
    )


@app.post("/api/place-trade")
async def api_place_trade(
    ticker: str = Form(...),
    side: str = Form(...),
    entry_price: float = Form(...),
    signal_type: str = Form(...),
    notes: str = Form(""),
) -> JSONResponse:
    """Place a paper trade."""
    result = place_trade(ticker, side, entry_price, signal_type, notes=notes)
    return JSONResponse(result)


@app.post("/api/close-trade")
async def api_close_trade(
    trade_id: int = Form(...),
    exit_price: float = Form(...),
) -> JSONResponse:
    """Close a paper trade."""
    result = close_trade(trade_id, exit_price)
    return JSONResponse(result)
