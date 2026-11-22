from __future__ import annotations

import asyncio
import collections
from functools import lru_cache
import subprocess
import sys
import hmac
import json
import logging
import os
import re
import secrets
import sqlite3
import time
from urllib.parse import quote
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional
from zoneinfo import ZoneInfo

import bcrypt
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from fastapi import Body, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
import pandas as pd
import yfinance as yf

from code.app import settings  # first: loads .env before anything reads the environment
from code.data.fetchers.nifty500 import fetch_nifty500_directory
from code.live import audit as live_audit, db as live_db, state as live_state
from code.live.session import market_session
from code.live import broker_session
from code.live.brokers import choice as choice_broker
from code.live.feed import LiveFeed
from code.live.market_data import ChoiceProvider, MotilalProvider, SimulatedProvider, YFinanceDelayedProvider
from code.data.fetchers.earnings import fetch_next_results
from code.evaluator.memo import _init_db as init_memo_db
from code.evaluator.stock_review import cached_review, get_review, latest_signal
from code.pipeline.news_rss import fetch_company_news, fetch_news_items
from code.pipeline.nightly_pipeline import kite_fetch
from code.signals.live_rules import ROUTINE, day_phase, position_signal, sell_level_signal, setup_signal, short_signal
from code.signals.minervini.setup import stock_metrics
from code.trading.portal import (
    close_trade, delete_all_closed, delete_trade, fill_pending_live, get_scorecard, get_trade_stats, get_trades, place_trade, set_note,
)
from code.trading import model_lab
from code.trading.predict import portfolio_outlook, position_view, predict_stock

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


def _market_pill() -> Dict[str, str]:
    """Navbar status: open with time to close, or closed with the next open."""
    s = market_session().state()
    if s.is_open:
        return {"cls": "open", "text": "Market open", "sub": "closes 3:30 pm"}
    if s.phase == "PRE_OPEN":
        return {"cls": "pre", "text": "Pre-open", "sub": "opens 9:15 am"}
    nxt = s.next_open
    when = "today" if nxt.date() == s.now.date() else "tomorrow" if (nxt.date() - s.now.date()).days == 1 else f"{nxt:%a}"
    return {"cls": "closed", "text": s.holiday_name or "Market closed", "sub": f"opens {when} 9:15 am"}


def _template_globals(request: Request) -> Dict[str, Any]:
    nifty = (_load_latest_snapshot().get("indices") or {}).get("nifty") or {}
    try:
        halted, beginner, pill = live_state.is_halted(), live_state.pref("beginner_mode") == "1", _market_pill()
    except Exception:  # a page must render even if the state database is unavailable
        halted, beginner, pill = False, True, None
    return {"csp_nonce": getattr(request.state, "csp_nonce", ""), "nav_nifty": nifty,
            "halted": halted, "beginner": beginner, "market_pill": pill}


TEMPLATES = Jinja2Templates(directory=str(ROOT / "templates"), context_processors=[_template_globals])


def _sym(value: Any) -> str:
    return str(value or "").replace(".NS", "")


def _nice_date(value: Any, with_time: bool = False) -> str:
    """'2026-10-03T17:08:25+05:30' -> '3 Oct, 5:08 pm' (or '3 Oct 2026' for dates)."""
    if not value:
        return ""
    try:
        dt = datetime.fromisoformat(str(value))
    except ValueError:
        return str(value)
    if dt.tzinfo is not None:
        dt = dt.astimezone(IST)
    day = f"{dt.day} {dt.strftime('%b')}"
    if with_time and (dt.hour or dt.minute):
        return f"{day}, {dt.strftime('%I:%M %p').lstrip('0').lower()}"
    return day if dt.year == datetime.now().year else f"{day} {dt.year}"


TEMPLATES.env.filters["sym"] = _sym
TEMPLATES.env.filters["nicedate"] = _nice_date
TEMPLATES.env.filters["nicetime"] = lambda v: _nice_date(v, with_time=True)


def _inr(v: float) -> str:
    """Indian digit grouping: 1000000 -> 10,00,000."""
    s = str(int(round(v)))
    head, tail = s[:-3], s[-3:]
    while len(head) > 2:
        tail, head = head[-2:] + "," + tail, head[:-2]
    return (head + "," + tail) if head else tail


TEMPLATES.env.filters["inr"] = _inr


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


_SNAPSHOT_CACHE: Dict[str, Any] = {"key": None, "data": None}


def _load_latest_snapshot() -> Dict[str, Any]:
    """Dashboard reads pre-computed JSON snapshots only -- never recomputes live.
    Parsed once per file version: the snapshot is ~2.5 MB and every page (plus the header quote) reads it."""
    files = sorted(SNAPSHOT_DIR.glob("*.json"))
    if not files:
        return {"screen": {"items": []}, "meta": {"generated_at": None, "status": "missing"}}
    key = (str(files[-1]), files[-1].stat().st_mtime_ns)
    if _SNAPSHOT_CACHE["key"] != key:
        with open(files[-1], "r", encoding="utf-8") as handle:
            _SNAPSHOT_CACHE.update(key=key, data=json.load(handle))
    return _SNAPSHOT_CACHE["data"]


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

        facts = today_item or {}
        rows.append({
            "symbol": symbol,
            "close": facts.get("close"),
            "chg_pct": facts.get("chg_pct"),
            "pct_from_high": facts.get("pct_from_high"),
            "conditions": facts.get("conditions_passed"),
            "setup_status": (facts.get("setup") or {}).get("status"),
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
    meta = snapshot.get("meta", {})
    generated_at = meta.get("generated_at")
    base = {"market_data_label": None, "analysis_age_label": None, "generated_at": generated_at, "data_asof": meta.get("data_asof"),
            "market_open": market_session().state().is_open}
    if not generated_at:
        return base
    try:
        generated = datetime.fromisoformat(generated_at)
    except ValueError:
        return base

    delta = datetime.now() - generated
    hours = delta.total_seconds() / 3600
    if hours < 1:
        age_label = f"{max(1, int(delta.total_seconds() / 60))}m old"
    else:
        age_label = f"{hours:.0f}h old"

    return {
        **base,
        "market_data_label": "CURRENT" if delta <= STALE_AFTER else None,
        "analysis_age_label": age_label,
    }


def _run_job_in_subprocess(name: str, call: str) -> None:
    """Heavy jobs run in a child process: Python never hands a finished run's ~500 MB back to the OS,
    so in-process the web server sat at ~1 GB after 18:30 and the 08:30 run got the whole container
    OOM-killed on Railway. A child returns everything on exit, and if it is killed only it dies."""
    code = ("import logging; logging.basicConfig(level=logging.INFO, "
            "format='%(asctime)s %(levelname)s %(name)s: %(message)s'); " + call)
    logger.info("%s starting", name)
    try:
        r = subprocess.run([sys.executable, "-c", code], stdout=subprocess.DEVNULL, timeout=3600)
    except subprocess.TimeoutExpired:
        logger.error("%s failed: still running after an hour, stopped", name)
        return
    if r.returncode == 0:
        logger.info("%s finished OK", name)
    else:
        oom = " (killed -- most likely out of memory)" if r.returncode in (-9, 137) else ""
        logger.error("%s failed: exit code %s%s", name, r.returncode, oom)


def _run_nightly_pipeline_sync() -> None:
    _run_job_in_subprocess("nightly pipeline run",
                           "from code.pipeline.nightly_pipeline import run_pipeline; run_pipeline()")


# One heavy job at a time (pipeline ~380 MB, pairs rescan ~270 MB): a scheduled run landing during
# the startup catch-up waits instead of doubling memory.
_HEAVY_JOB_LOCK = asyncio.Lock()


async def _run_nightly_pipeline_job() -> None:
    async with _HEAVY_JOB_LOCK:
        await asyncio.to_thread(_run_nightly_pipeline_sync)


def _run_pairs_monthly_rescan_sync() -> None:
    _run_job_in_subprocess("pairs monthly rescan",
                           "from code.pipeline.nightly_pipeline import run_pairs_monthly_rescan; run_pairs_monthly_rescan()")


async def _run_pairs_monthly_rescan_job() -> None:
    async with _HEAVY_JOB_LOCK:
        await asyncio.to_thread(_run_pairs_monthly_rescan_sync)


# A sleeping laptop wakes up after the 18:30 / 08:30 slots; by default APScheduler then skips the run
# (1s grace). Run a missed job once on wake instead, if it is less than 12h late.
scheduler = AsyncIOScheduler(timezone=IST, job_defaults={"coalesce": True, "misfire_grace_time": 12 * 3600})

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
    live_db.migrate()
    live_audit.record("app_start", "system", "app", "", settings.summary())
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
    scheduler.add_job(
        broker_session.auto_connect,  # plain function: APScheduler runs it in its thread pool
        CronTrigger(day_of_week="mon-fri", hour=8, minute=50, timezone=IST),
        id="broker_auto_connect", name="Motilal auto-connect (08:50 IST, if configured)", replace_existing=True,
    )
    scheduler.add_job(
        choice_broker.auto_connect,
        CronTrigger(day_of_week="mon-fri", hour=8, minute=55, timezone=IST),
        id="choice_auto_connect", name="Choice auto-connect (08:55 IST, if configured)", replace_existing=True,
    )
    scheduler.start()
    for job in scheduler.get_jobs():
        logger.info("scheduled job registered: %s (id=%s) next run: %s", job.name, job.id, job.next_run_time)

    # A deploy or restart can land at any time of day, hours from the next
    # 18:30/08:30 fire -- if the snapshot on disk is already missing or
    # stale, kick one immediate background run so the site isn't stuck on
    # "no data" for up to a full cycle. Fire-and-forget; never blocks startup.
    # Pairs rescan (if its cache is missing) and pipeline run one after the other: together they
    # peak ~650 MB and got OOM-killed on a 512 MB Railway container, restarting forever.
    from code.pipeline.nightly_pipeline import PAIRS_CACHE_PATH
    need_pairs = not PAIRS_CACHE_PATH.exists()
    need_snapshot = _snapshot_is_stale(_load_latest_snapshot())

    async def _startup_catch_up() -> None:
        if need_pairs:
            logger.info("pairs cointegration cache missing at startup -- running the monthly rescan first")
            await _run_pairs_monthly_rescan_job()
        if need_snapshot:
            logger.info("snapshot missing/stale at startup -- running the pipeline now")
            await _run_nightly_pipeline_job()

    if need_pairs or need_snapshot:
        asyncio.create_task(_startup_catch_up())

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
    asyncio.create_task(asyncio.to_thread(broker_session.auto_connect))
    asyncio.create_task(asyncio.to_thread(choice_broker.auto_connect))
    asyncio.create_task(LIVE_FEED.run())

    yield

    scheduler.shutdown(wait=False)


# /api/daily/sync skips the PIN session but has its own token check (laptop -> Railway, no browser)
PUBLIC_AUTH_PATHS = {"/login", "/logout", "/healthz", "/api/daily/sync"}


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


_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Hardening on every response (OWASP ASVS L1, secure headers project):
    a nonce-based CSP so only this app's own scripts run, no framing, no sniffing, no
    powerful browser features, and a same-origin check on every state-changing request
    (defence in depth on top of the SameSite=Lax session cookie)."""

    async def dispatch(self, request: Request, call_next):
        if request.method not in _SAFE_METHODS and not _same_origin(request):
            return JSONResponse({"detail": "Cross-site request blocked"}, status_code=403)
        nonce = secrets.token_urlsafe(16)
        request.state.csp_nonce = nonce
        response = await call_next(request)
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            f"script-src 'self' 'nonce-{nonce}'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; connect-src 'self'; font-src 'self'; "
            "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=(), payment=(), usb=()"
        response.headers["Cross-Origin-Resource-Policy"] = "same-origin"
        if request.url.path.startswith("/api/") or request.url.path == "/login":
            response.headers.setdefault("Cache-Control", "no-store")
        if RAILWAY_ENVIRONMENT:  # HTTPS only: browsers ignore both over plain http on the home Wi-Fi
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
            response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        return response


def _same_origin(request: Request) -> bool:
    """Origin (or Referer) must name this host. Browsers always send one of them on a
    form/fetch POST; if neither is present, fall back to Fetch Metadata."""
    from urllib.parse import urlsplit
    source = request.headers.get("origin") or request.headers.get("referer")
    if source:
        return urlsplit(source).netloc == request.headers.get("host", "")
    return request.headers.get("sec-fetch-site", "same-origin") in ("same-origin", "none")


app = FastAPI(title="Marketly", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
app.add_middleware(GZipMiddleware, minimum_size=1000)
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


_REGIME_WORD = {"TRENDING_UP": "uptrend", "TRENDING_DOWN": "correction", "CHOPPY": "choppy"}

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
        evidence.append(f"market {_REGIME_WORD.get(regime, regime.lower())}")
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
    """One market read, one list of actionable setups, the watchlist -- in that order."""
    snapshot = _load_latest_snapshot()
    screen_items = snapshot.get("screen", {}).get("items", [])
    previous_snapshot = _load_previous_snapshot()
    diffs, _dropped = _diff_screen_items(screen_items, previous_snapshot)
    results = snapshot.get("results") or {}
    reviews = {p["symbol"]: p.get("review") for p in snapshot.get("predictions", []) if p.get("review")}
    pivots = {i["symbol"]: (i.get("setup") or {}).get("pivot") for i in screen_items}
    setups = []
    for item in snapshot.get("action_queue", []):
        row = dict(item)
        row["pivot"] = pivots.get(item["symbol"])
        row["results_date"] = results.get(item["symbol"])
        row["review"] = reviews.get(item["symbol"])
        row["badge"] = (diffs.get(item["symbol"]) or {}).get("badge")
        setups.append(row)

    return TEMPLATES.TemplateResponse(
        request,
        "index.html",
        {
            "regime": snapshot.get("regime", "CHOPPY"),
            "indices": snapshot.get("indices", {}),
            "market": snapshot.get("market") or {},
            "posture": _posture(snapshot),
            "briefing": snapshot.get("briefing", {}).get("items", []),
            "setups": setups,
            "shorts": snapshot.get("shorts", []),
            "sells": snapshot.get("sells", []),
            "watchlist_rows": _build_watchlist_section(screen_items, previous_snapshot),
            "stale": _snapshot_is_stale(snapshot),
            "routine": ROUTINE,
            "routine_now": day_phase(datetime.now(IST), market_session().is_trading_day(datetime.now(IST).date())),
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
    """Everything that's cheap renders now; the Setup Review (an LLM call when not pre-computed
    overnight) and the company headlines load into the page afterwards."""
    asset = _normalize_ticker(asset)
    if asset.endswith(".NS") and asset not in _RECENT_VIEWED:
        _RECENT_VIEWED.append(asset)
    snapshot = _load_latest_snapshot()
    df = _drop_incomplete_ohlcv_rows(await asyncio.to_thread(kite_fetch, asset))

    base_context = {
        "asset": asset,
        "generated_at": snapshot.get("meta", {}).get("generated_at"),
        "stale": _snapshot_is_stale(snapshot),
        "active_tab": "stock",
        "in_watchlist": asset in _get_watchlist_symbols(),
        **_freshness_labels(snapshot),
    }
    if df.empty:
        return TEMPLATES.TemplateResponse(request, "stock_room.html", {**base_context, "not_found": True})

    regime = snapshot.get("regime")
    screen_item = _find_screen_item(snapshot, asset)
    if screen_item is None:
        # Off-universe ticker: the 8-point template is a pure function of this stock's own OHLCV,
        # so evaluate it live. RS rank ranks against the universe, so it stays None.
        screen_item = _evaluate_screen_item_live(asset, df, regime)
    results = snapshot.get("results") or {}
    results_date = results[asset] if asset in results else (await asyncio.to_thread(fetch_next_results, [asset])).get(asset)
    posture = _posture(snapshot)
    plan = predict_stock(asset, df, risk_pct=posture["risk_pct"], results_date=results_date, setup=(screen_item or {}).get("setup"))
    signal = latest_signal(asset, df)
    review = cached_review(asset, snapshot)

    return TEMPLATES.TemplateResponse(
        request,
        "stock_room.html",
        {
            **base_context,
            "not_found": False,
            "company": _company_name(asset),
            "latest": df.iloc[-1],
            "metrics": stock_metrics(df),
            "plan": plan,
            "posture": posture,
            "grade": (snapshot.get("market") or {}).get("grade"),
            "history": _grade_history((snapshot.get("market") or {}).get("grade")),
            "signal": signal if signal.get("direction") in ("BUY", "SELL") else None,
            "screen_item": screen_item,
            "regime": regime,
            **_review_context(review, screen_item, regime),
        },
    )


_BASE_RATES_PATH = ROOT.parent / "data" / "base_rates.json"


def _grade_history(grade: Optional[str]) -> Optional[Dict[str, Any]]:
    """Historical breakout record for this market grade (code/research/base_rates.py)."""
    if not grade:
        return None
    try:
        return json.loads(_BASE_RATES_PATH.read_text(encoding="utf-8"))["grades"].get(grade)
    except (OSError, ValueError, KeyError):
        return None


def _posture(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    posture = (snapshot.get("market") or {}).get("posture")
    if posture:
        return posture
    regime = snapshot.get("regime") or "CHOPPY"
    return {"TRENDING_UP": {"level": "PRESS", "risk_pct": 1.0, "max_positions": 8, "text": "Uptrend: normal size."},
            "TRENDING_DOWN": {"level": "DEFENSIVE", "risk_pct": 0.25, "max_positions": 2, "text": "Market in correction: mostly cash."},
            }.get(regime, {"level": "CAUTIOUS", "risk_pct": 0.5, "max_positions": 4, "text": "Mixed market: half size."})


def _company_name(ticker: str) -> Optional[str]:
    return next((d["name"] for d in _ticker_directory() if d["symbol"] == ticker), None)


def _review_context(review: Optional[Dict[str, Any]], screen_item: Optional[Dict[str, Any]], regime: Optional[str]) -> Dict[str, Any]:
    if review is None:
        return {"memo": None}
    stance = _compute_stockroom_stance(review, screen_item, regime, {})
    quality = _setup_quality(review.get("score") if not review.get("sentinel") else None)
    return {
        "memo": review,
        "stance": stance["stance"], "stance_class": stance["stance_class"], "stance_evidence": stance["stance_evidence"],
        "quality_word": quality["word"], "quality_class": quality["class"],
    }


@app.get("/api/memo/{ticker}", response_class=HTMLResponse)
async def api_memo_lazy(ticker: str, request: Request) -> HTMLResponse:
    ticker = _normalize_ticker(ticker)
    snapshot = _load_latest_snapshot()
    df = _drop_incomplete_ohlcv_rows(await asyncio.to_thread(kite_fetch, ticker))
    if df.empty:
        return HTMLResponse('<div class="empty-state">No price data, so no review.</div>', status_code=404)
    review = await asyncio.to_thread(
        get_review, ticker, df, snapshot, [], _company_name(ticker) or ticker.replace(".NS", ""),
    )
    screen_item = _find_screen_item(snapshot, ticker) or _evaluate_screen_item_live(ticker, df, snapshot.get("regime"))
    return TEMPLATES.TemplateResponse(request, "_review_card.html", _review_context(review, screen_item, snapshot.get("regime")))


@app.get("/api/headlines/{ticker}", response_class=HTMLResponse)
async def api_headlines(ticker: str, request: Request) -> HTMLResponse:
    ticker = _normalize_ticker(ticker)
    name = _company_name(ticker) or ticker.replace(".NS", "")
    items = await asyncio.to_thread(fetch_company_news, name, 5, ticker)
    return TEMPLATES.TemplateResponse(request, "_headlines.html", {"headlines": items})


@app.get("/screen", response_class=HTMLResponse)
async def minervini_screen(request: Request) -> HTMLResponse:
    """Defaults to the stocks that pass -- 359 'avoid' rows were most of a 97,000-pixel page."""
    snapshot = _load_latest_snapshot()
    all_items = snapshot.get("screen", {}).get("items", [])
    view = request.query_params.get("view", "pass")
    diffs, dropped = _diff_screen_items(all_items, _load_previous_snapshot())
    passing = [i for i in all_items if i.get("passed")]
    close = [i for i in all_items if not i.get("passed") and i.get("conditions_passed") == 7]
    items = {"pass": passing, "close": close}.get(view, all_items)
    setups = [i.get("setup") or {} for i in passing]
    vcp = sorted((i for i in passing if (i.get("setup") or {}).get("vcp")),
                 key=lambda i: (i["setup"]["status"] != "IN BUY RANGE", i["setup"].get("pct_to_pivot") or 99))
    return TEMPLATES.TemplateResponse(
        request,
        "screen.html",
        {
            "items": items,
            "view": view if view in ("pass", "close", "all") else "all",
            "vcp": vcp,
            "counts": {
                "all": len(all_items), "passing": len(passing), "close": len(close),
                "buy_range": sum(1 for s in setups if s.get("status") == "IN BUY RANGE"),
                "near_pivot": sum(1 for s in setups if s.get("status") == "BASING" and (s.get("pct_to_pivot") or 99) <= 5),
            },
            "changes": snapshot.get("screen_changes") or {},
            "results": snapshot.get("results") or {},
            "names": {d["symbol"]: d["name"] for d in _ticker_directory()},
            "diffs": diffs,
            "dropped": dropped,
            "stale": _snapshot_is_stale(snapshot),
            "active_tab": "screen",
            "regime": snapshot.get("regime", "CHOPPY"),
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

    # Live trades first, then the most stretched; chart series go to the page as one JSON blob
    # drawn when scrolled into view, not as inline scripts per card.
    items = sorted(pairs.get("items", []), key=lambda i: (i.get("status") not in ("BUY", "SELL"), -abs(i.get("zscore") or 0)))
    items_chart = [{"spread": i.get("spread_series", [])[-120:], "upper": i.get("upper_band_series", [])[-120:],
                    "lower": i.get("lower_band_series", [])[-120:]} for i in items]
    return TEMPLATES.TemplateResponse(
        request,
        "pairs.html",
        {
            "items": items,
            "items_chart": items_chart,
            "last_scan": pairs.get("last_scan"),
            "rescan_stats": rescan_stats,
            "generated_at": snapshot.get("meta", {}).get("generated_at"),
            "stale": _snapshot_is_stale(snapshot),
            "active_tab": "pairs",
            **_freshness_labels(snapshot),
        },
    )


@app.get("/diary")
async def diary() -> RedirectResponse:
    """The diary lives with the trades now: every paper trade carries its own note."""
    return RedirectResponse(url="/predictions#book", status_code=307)


NIFTY_LIVE = "^NSEI"
_RECENT_VIEWED: "collections.deque[str]" = collections.deque(maxlen=8)  # Stock Room pages opened today


def _live_symbols() -> List[str]:
    """What's worth a live price: watchlist, tonight's buy-ready setups, open positions, waiting orders."""
    snapshot = _load_latest_snapshot()
    syms = [NIFTY_LIVE] + list(_RECENT_VIEWED) + list(_get_watchlist_symbols())
    syms += [a["symbol"] for a in snapshot.get("action_queue", []) if a.get("kind") == "stock"]
    syms += [r["symbol"] for r in snapshot.get("shorts", [])]
    syms += [p["symbol"] for p in snapshot.get("predictions", []) if p.get("kind") == "stock" and p.get("tradeable")]
    syms += [t["ticker"] for t in get_trades(limit=100, status="open") + get_trades(limit=100, status="pending") if t.get("kind") != "pair"]
    syms += [s + ".NS" for s in _daily_symbols()]
    return list(dict.fromkeys(syms))


def _daily_symbols() -> List[str]:
    """Tonight's daily-loop corpus and both books' stocks, so they get live prices tomorrow."""
    try:
        conn = sqlite3.connect(DB_PATH)
        try:
            a = conn.execute("SELECT MAX(asof) FROM daily_predictions").fetchone()[0]
            if not a:
                return []
            rows = conn.execute("SELECT symbol FROM daily_predictions WHERE asof = ? UNION "
                                "SELECT symbol FROM daily_plan WHERE asof = ?", (a, a)).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()
    except sqlite3.Error:
        return []


_SIMULATED: Optional[SimulatedProvider] = None


def _live_provider():
    global _SIMULATED
    choice = settings.MARKET_DATA_PROVIDER
    if choice == "broker" and choice_broker.client() is not None:
        return ChoiceProvider(choice_broker.client())
    if choice == "broker" and broker_session.client() is not None:
        return MotilalProvider(broker_session.client())
    if choice == "yfinance_delayed":
        return YFinanceDelayedProvider()
    if _SIMULATED is None:
        snapshot = _load_latest_snapshot()
        closes = {i["symbol"]: i["close"] for i in snapshot.get("screen", {}).get("items", []) if i.get("close")}
        closes.update({p["symbol"]: p["last"] for p in snapshot.get("predictions", []) if p.get("last")})
        _SIMULATED = SimulatedProvider(closes)
    return _SIMULATED


def _fill_on_poll(quotes: Dict[str, dict]) -> None:
    """Waiting paper orders fill during market hours, not only at the 18:30 settlement."""
    st = market_session().state()
    if not st.is_open or live_state.is_halted():
        return
    for trade_id in fill_pending_live(quotes, st.now.date().isoformat()):
        live_audit.record("paper_fill_live", "system", "paper_trades", trade_id, {"source": LIVE_FEED.provider_name})
        logger.info("paper order %s filled live", trade_id)


LIVE_FEED = LiveFeed(_live_provider, _live_symbols, on_poll=_fill_on_poll)


@app.get("/api/quotes")
async def api_quotes() -> JSONResponse:
    """Latest validated quote per watched symbol. `stale` is judged against QUOTE_STALE_SECONDS;
    `simulated` marks demo prices."""
    return JSONResponse({
        "provider": LIVE_FEED.provider_name or settings.MARKET_DATA_PROVIDER,
        "session": market_session().state().as_dict(),
        "last_poll": LIVE_FEED.last_poll.isoformat() if LIVE_FEED.last_poll else None,
        "error": LIVE_FEED.last_error,
        "quotes": LIVE_FEED.view(),
        "rejected": LIVE_FEED.rejected,
        **_live_guidance(LIVE_FEED.view()),
    })


def _live_guidance(quotes: Dict[str, dict], now: Optional[datetime] = None) -> Dict[str, Any]:
    """What to do right now, per setup (by symbol) and per open paper position (by trade id).
    Stale quotes get no guidance: acting on an old price is worse than not acting."""
    now = now or datetime.now(IST)
    fresh = {s: q for s, q in quotes.items() if not q.get("stale")}
    snapshot = _load_latest_snapshot()
    wanted = {a["symbol"] for a in snapshot.get("action_queue", []) if a.get("kind") == "stock"}
    signals = {p["symbol"]: setup_signal(p, fresh[p["symbol"]], now) for p in snapshot.get("predictions", [])
               if p.get("kind") == "stock" and p["symbol"] in fresh and (p["symbol"] in wanted or p.get("tradeable"))}
    signals.update({f"short:{r['symbol']}": short_signal(r, fresh[r["symbol"]], now)
                    for r in snapshot.get("shorts", []) if r["symbol"] in fresh})
    for r in snapshot.get("sells", []):
        g = sell_level_signal(r, fresh[r["symbol"]]["ltp"]) if r["symbol"] in fresh else None
        if g:
            signals[f"sell:{r['symbol']}"] = g
    positions = {str(t["id"]): position_signal(t, fresh[t["ticker"]]["ltp"]) for t in get_trades(limit=100, status="open")
                 if t.get("kind") != "pair" and t["ticker"] in fresh}
    return {"signals": signals, "positions": positions}


@app.get("/broker", response_class=HTMLResponse)
async def broker_page(request: Request) -> HTMLResponse:
    snapshot = _load_latest_snapshot()
    secure = request.url.scheme == "https" or request.client is None or request.client.host in ("127.0.0.1", "::1")
    return TEMPLATES.TemplateResponse(request, "broker.html", {
        "choice": await asyncio.to_thread(choice_broker.status),
        "status": broker_session.status(),
        "provider": settings.MARKET_DATA_PROVIDER,
        "session": market_session().state().as_dict(),
        "feed": {"last_poll": LIVE_FEED.last_poll, "error": LIVE_FEED.last_error, "count": len(LIVE_FEED.quotes)},
        "secure": secure,
        "message": request.query_params.get("m", ""),
        "active_tab": "broker",
        "stale": _snapshot_is_stale(snapshot),
        **_freshness_labels(snapshot),
    })


@app.post("/broker/connect")
async def broker_connect(password: str = Form(...), totp: str = Form("")) -> RedirectResponse:
    from urllib.parse import quote as urlquote
    result = await asyncio.to_thread(broker_session.connect, password, totp.strip(), "papa")
    return RedirectResponse(url="/broker?m=" + urlquote(result["message"]), status_code=303)


@app.post("/broker/choice/connect")
async def broker_choice_connect() -> RedirectResponse:
    from urllib.parse import quote as urlquote
    result = await asyncio.to_thread(choice_broker.connect, "papa")
    return RedirectResponse(url="/broker?m=" + urlquote(result["message"]), status_code=303)


@app.post("/broker/choice/disconnect")
async def broker_choice_disconnect() -> RedirectResponse:
    await asyncio.to_thread(choice_broker.disconnect, "papa")
    return RedirectResponse(url="/broker?m=Disconnected", status_code=303)


@app.post("/broker/disconnect")
async def broker_disconnect() -> RedirectResponse:
    await asyncio.to_thread(broker_session.disconnect, "papa")
    return RedirectResponse(url="/broker?m=Disconnected", status_code=303)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    """Liveness + readiness for monitors: no secrets, no market data, cheap."""
    checks: Dict[str, Any] = {}
    try:
        conn = live_db.connect()
        conn.execute("SELECT 1").fetchone()
        conn.close()
        checks["database"] = "ok"
    except Exception as exc:
        checks["database"] = f"error: {type(exc).__name__}"
    snapshot = _load_latest_snapshot()
    checks["snapshot_stale"] = _snapshot_is_stale(snapshot)
    checks["data_asof"] = snapshot.get("meta", {}).get("data_asof")
    checks["session"] = market_session().state().phase
    checks.update(live_state.snapshot())
    checks["config"] = settings.summary()
    ok = checks["database"] == "ok"
    return JSONResponse({"status": "ok" if ok else "degraded", "checks": checks}, status_code=200 if ok else 503)


@app.get("/api/state")
async def api_state() -> JSONResponse:
    return JSONResponse({**live_state.snapshot(), "session": market_session().state().as_dict()})


@app.post("/api/halt")
async def api_halt(on: int = Form(...), reason: str = Form("")) -> JSONResponse:
    """Emergency halt: blocks every new order until switched off. Authenticated + audited."""
    live_state.set_halt(bool(on), actor="papa", reason=reason[:200])
    return JSONResponse(live_state.snapshot())


@app.get("/api/tickers.json")
async def api_tickers() -> JSONResponse:
    """Search directory for the typeahead -- one cached download instead of 30 KB inside every page."""
    return JSONResponse(_ticker_directory(), headers={"Cache-Control": "private, max-age=86400"})


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


@app.get("/predictions", response_class=HTMLResponse)
async def predictions(request: Request) -> HTMLResponse:
    """Paper trading on the nightly projections. A searched off-list stock is projected live,
    the same way Stock Room evaluates off-universe tickers live."""
    snapshot = _load_latest_snapshot()
    posture = _posture(snapshot)
    open_trades = get_trades(limit=50, status="open")
    pending = get_trades(limit=50, status="pending")
    held = {t["ticker"] for t in open_trades + pending}
    preds = snapshot.get("predictions", [])
    search = request.query_params.get("symbol", "").strip()
    search_error = None
    if search:
        ticker = _normalize_ticker(search)
        existing = next((p for p in preds if p["symbol"] == ticker), None)
        live = existing or await asyncio.to_thread(_live_prediction, ticker, snapshot)
        if live:
            preds = [dict(live, searched=True)] + [p for p in preds if p["symbol"] != ticker]
        else:
            search_error = f"Couldn't find price history for {ticker.replace('.NS', '')} — check the spelling (NSE symbol, e.g. TATAMOTORS)."
    for p in preds:
        p["held"] = p["symbol"] in held

    def _views():
        views, held_dfs = {}, []
        for t in open_trades + pending:
            if t.get("kind") == "pair":
                continue
            try:
                df = _drop_incomplete_ohlcv_rows(kite_fetch(t["ticker"]))
                view = position_view(t, df, (_find_screen_item(snapshot, t["ticker"]) or {}).get("setup")) if not df.empty else None
            except Exception:
                logger.exception("position view failed for %s", t["ticker"])
                view = None
            if view:
                views[str(t["id"])] = view
                if t["status"] == "open":
                    held_dfs.append((t, df))
        try:
            outlook = portfolio_outlook(held_dfs, live_state.pref("paper_capital"))
        except Exception:
            logger.exception("portfolio outlook failed")
            outlook = {"n": len(held_dfs)}
        return views, outlook
    position_views, outlook = await asyncio.to_thread(_views)

    # Tomorrow: every position and order, then new buys that fit the free slots.
    tomorrow = [{"symbol": t["ticker"], "id": t["id"], **position_views[str(t["id"])]["tomorrow"]}
                for t in open_trades + pending if position_views.get(str(t["id"]), {}).get("tomorrow")]
    free_slots = max(0, posture["max_positions"] - len(open_trades) - len(pending))
    ideas = sorted((p for p in preds if p.get("kind") == "stock" and p.get("tradeable") and p["symbol"] not in held and not p.get("searched")),
                   key=lambda p: p.get("rs_rank") or 0, reverse=True)[:free_slots]

    return TEMPLATES.TemplateResponse(
        request,
        "predictions.html",
        {
            "active_tab": "predictions",
            "regime": snapshot.get("regime", "CHOPPY"),
            "posture": posture,
            "market": snapshot.get("market") or {},
            "grade": (snapshot.get("market") or {}).get("grade"),
            "history": _grade_history((snapshot.get("market") or {}).get("grade")),
            "slots_used": len(open_trades) + len(pending),
            "predictions": preds,
            "search": search,
            "search_error": search_error,
            "stats": get_trade_stats(),
            "capital": live_state.pref("paper_capital"),
            "exit_rule_default": live_state.pref("default_exit_rule"),
            "scorecard": get_scorecard(),
            "open_trades": open_trades,
            "position_views": position_views,
            "outlook": outlook,
            "tomorrow": tomorrow,
            "ideas": ideas,
            "free_slots": free_slots,
            "pending_trades": pending,
            "closed_trades": get_trades(limit=30, status="closed") + get_trades(limit=10, status="expired"),
            "generated_at": snapshot.get("meta", {}).get("generated_at"),
            "stale": _snapshot_is_stale(snapshot),
            **_freshness_labels(snapshot),
        },
    )


def _lab_symbols() -> List[str]:
    """What the lab offers first: paper positions (open, then waiting), then the watchlist."""
    trades = get_trades(limit=100, status="open") + get_trades(limit=100, status="pending")
    return list(dict.fromkeys([t["ticker"] for t in trades if t.get("kind") != "pair"] + list(_get_watchlist_symbols())))


def _lab_row(r: Dict[str, Any]) -> Dict[str, Any]:
    return {k: r[k] for k in ("name", "graph", "select", "report", "select_span", "report_span", "forecast")}


@app.get("/lab", response_class=HTMLResponse)
async def lab_page(request: Request) -> HTMLResponse:
    snapshot = _load_latest_snapshot()
    held = {t["ticker"] for t in get_trades(limit=100, status="open") + get_trades(limit=100, status="pending")}
    symbols = _lab_symbols()
    symbol = _normalize_ticker(request.query_params.get("symbol", "")) if request.query_params.get("symbol") else (symbols[0] if symbols else "RELIANCE.NS")
    return TEMPLATES.TemplateResponse(request, "lab.html", {
        "active_tab": "lab", "symbol": symbol, "symbols": symbols, "held": held,
        "node_types": model_lab.NODE_TYPES, "presets": [{"name": n, "graph": g} for n, g in model_lab.PRESETS],
        "horizon": model_lab.HORIZON, "stale": _snapshot_is_stale(snapshot), **_freshness_labels(snapshot),
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings_page(request: Request) -> HTMLResponse:
    from code.app import config as app_config
    snapshot = _load_latest_snapshot()
    provider = app_config.MODEL_PROVIDER
    key_set = bool(os.getenv("OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"))
    jobs = sorted(({"name": j.name, "next": j.next_run_time} for j in scheduler.get_jobs() if j.next_run_time), key=lambda j: j["next"])
    return TEMPLATES.TemplateResponse(request, "settings.html", {
        "active_tab": "settings", "saved": request.query_params.get("saved"), "error": request.query_params.get("error"),
        "capital": live_state.pref("paper_capital"), "exit_rule": live_state.pref("default_exit_rule"),
        "mode": live_state.mode(), "config": settings.summary(), "session": market_session().state(),
        "ai": {"provider": provider, "key_set": key_set}, "jobs": jobs, "meta": snapshot.get("meta", {}),
        "stale": _snapshot_is_stale(snapshot), **_freshness_labels(snapshot),
    })


@app.post("/settings")
async def settings_save(request: Request) -> RedirectResponse:
    form = await request.form()
    try:
        if "halt" in form:
            live_state.set_halt(form["halt"] == "1", actor="papa", reason=str(form.get("reason", ""))[:200])
            return RedirectResponse("/settings?saved=" + quote("Trading halted" if form["halt"] == "1" else "Halt switched off"), status_code=303)
        live_state.set_pref("paper_capital", str(form.get("paper_capital", "")), actor="papa")
        live_state.set_pref("default_exit_rule", str(form.get("default_exit_rule", "")), actor="papa")
        live_state.set_pref("beginner_mode", "1" if form.get("beginner_mode") else "0", actor="papa")
    except ValueError as exc:
        return RedirectResponse(f"/settings?error={quote(str(exc))}", status_code=303)
    return RedirectResponse("/settings?saved=" + quote("Settings saved"), status_code=303)


@app.get("/api/lab/recommend/{symbol}")
async def lab_recommend(symbol: str) -> JSONResponse:
    symbol = _normalize_ticker(symbol)
    rec = await asyncio.to_thread(model_lab.recommend_for, symbol)
    if not rec or not rec["best"]:
        return JSONResponse({"symbol": symbol, "error": "Not enough price history to backtest this stock (needs about 2½ years)."})
    return JSONResponse({"symbol": symbol, "best": _lab_row(rec["best"]), "baseline": _lab_row(rec["baseline"]) if rec["baseline"] else None,
                         "ranking": [_lab_row(r) for r in rec["ranking"]]})


@app.get("/api/lab/updown/{symbol}")
async def lab_updown(symbol: str) -> JSONResponse:
    """Chance up in 5 sessions, precomputed by code/research/updown.py, with its unseen-data record."""
    from code.research.updown import OUT
    symbol = _normalize_ticker(symbol)
    try:
        data = json.loads(OUT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return JSONResponse({"symbol": symbol, "error": "The up/down score hasn't been built yet."})
    stock = data["stocks"].get(symbol.removesuffix(".NS"))
    return JSONResponse({"symbol": symbol, "stock": stock, **{k: data[k] for k in
                         ("asof", "question", "period", "no_edge_auc", "overall", "by_grade", "ranking", "ranked_today")}})


@lru_cache(maxsize=1)
def _replay_data(stamp: float) -> Dict[str, Any]:
    """Replay file + per-week record per model, cached until the file changes (stamp = mtime)."""
    from code.research import model_search as ms
    df = ms.replay_load()
    models = [m for m in ("current", "best") if m in df.columns]
    return {"df": df, "weeks": {m: ms.replay_weeks(df, m) for m in models}}


@app.get("/lab/replay", response_class=HTMLResponse)
async def replay_page(request: Request) -> HTMLResponse:
    snapshot = _load_latest_snapshot()
    return TEMPLATES.TemplateResponse(request, "replay.html", {
        "active_tab": "lab", "stale": _snapshot_is_stale(snapshot), **_freshness_labels(snapshot)})


@app.get("/daily", response_class=HTMLResponse)
async def daily_page(request: Request) -> HTMLResponse:
    snapshot = _load_latest_snapshot()
    return TEMPLATES.TemplateResponse(request, "daily.html", {
        "active_tab": "lab", "stale": _snapshot_is_stale(snapshot), **_freshness_labels(snapshot)})


@app.get("/api/daily")
async def api_daily() -> JSONResponse:
    """Daily loop: latest calls and plans, last scored day, the ₹10,000 ledger, miss patterns."""
    from code.research import daily as dl
    try:
        conn = dl.connect(DB_PATH)
        try:
            data = dl.report(conn)                      # the laptop, where the loop runs
            if data is None:                            # Railway: the laptop's last pushed copy
                row = conn.execute("SELECT body, received_at FROM daily_export WHERE id = 1").fetchone() \
                    if conn.execute("SELECT name FROM sqlite_master WHERE name = 'daily_export'").fetchone() else None
                if row is None:
                    return JSONResponse({"error": "The daily loop hasn't run yet. It runs on the laptop each evening and sends its results here."})
                data = {**json.loads(row["body"]), "synced_at": row["received_at"]}
        finally:
            conn.close()
    except Exception:
        logger.exception("daily report failed")
        return JSONResponse({"error": "Couldn't read the daily loop's results."})
    return JSONResponse({**data, "provider": LIVE_FEED.provider_name or settings.MARKET_DATA_PROVIDER})


DAILY_SYNC_MAX_BYTES = 8_000_000


@app.post("/api/daily/sync")
async def api_daily_sync(request: Request) -> JSONResponse:
    """The laptop pushes each night's daily-loop report here (token in X-Sync-Token)."""
    token = os.getenv("DAILY_SYNC_TOKEN")
    given = request.headers.get("x-sync-token", "")
    if not token or not hmac.compare_digest(given.encode(), token.encode()):
        return JSONResponse({"detail": "Not authorised"}, status_code=403)
    body = await request.body()
    if len(body) > DAILY_SYNC_MAX_BYTES:
        return JSONResponse({"detail": "Too large"}, status_code=413)
    try:
        data = json.loads(body)
        if not isinstance(data, dict) or not {"summary", "predictions", "ledger"} <= set(data):
            raise ValueError("missing keys")
    except ValueError:
        return JSONResponse({"detail": "Not a daily report"}, status_code=400)
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS daily_export (id INTEGER PRIMARY KEY CHECK (id = 1), body TEXT, received_at TEXT)")
        conn.execute("INSERT OR REPLACE INTO daily_export VALUES (1, ?, ?)", (body.decode("utf-8"), datetime.now().isoformat(timespec="seconds")))
        conn.commit()
    finally:
        conn.close()
    return JSONResponse({"ok": True, "predictions": len(data["predictions"]), "ledger_rows": len(data["ledger"])})


@app.get("/api/daily/live")
async def api_daily_live() -> JSONResponse:
    """Live prices for tonight's calls, so the page can cross out calls going wrong during the day."""
    view = LIVE_FEED.view()
    return JSONResponse({"provider": LIVE_FEED.provider_name or settings.MARKET_DATA_PROVIDER,
                         "open": market_session().state().is_open,
                         "quotes": {s.removesuffix(".NS"): q for s, q in view.items()}})


@app.get("/api/lab/point/{symbol}")
async def lab_point(symbol: str) -> JSONResponse:
    """One price for the next 5 sessions plus a BEAT/LAG call, precomputed by code/research/point_forecast.py."""
    from code.research.point_forecast import OUT
    try:
        d = json.loads(OUT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return JSONResponse({"error": "The single-price forecast hasn't been built yet."})
    return JSONResponse({"asof": d["asof"], "horizon": d["horizon"], "overall": d["overall"], "record_from": d["record_from"],
                         "stock": d["stocks"].get(_normalize_ticker(symbol).removesuffix(".NS"))})


@app.get("/api/lab/zoo")
async def lab_zoo(horizon: int = 1) -> JSONResponse:
    """Model Zoo results for a horizon, precomputed by code/research/zoo.py."""
    path = DATA_ROOT / f"zoo_h{int(horizon)}.json"
    try:
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return JSONResponse({"error": f"The {int(horizon)}-session Model Zoo hasn't been run yet."})


@app.get("/api/lab/portfolio")
async def lab_portfolio() -> JSONResponse:
    """Phase 1 portfolio test, precomputed by code/research/portfolio.py."""
    from code.research.portfolio import OUT
    try:
        return JSONResponse(json.loads(OUT.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return JSONResponse({"error": "The portfolio test hasn't been run yet."})


@app.get("/api/lab/replay")
async def lab_replay(date: str = "", model: str = "") -> JSONResponse:
    from code.research import model_search as ms
    try:
        search = json.loads(ms.OUT.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        search = None
    if not ms.REPLAY.exists():
        return JSONResponse({"error": "Replay data hasn't been built on this machine yet.", "search": search})
    try:
        data = _replay_data(ms.REPLAY.stat().st_mtime)
        if model not in data["weeks"]:
            model = "best" if "best" in data["weeks"] else "current"     # open on the search winner
        weeks = data["weeks"][model]
        day = ms.replay_day(data["df"], weeks, model, date or str(weeks.index.max().date()))
    except Exception:
        logger.exception("replay failed")
        return JSONResponse({"error": "Couldn't replay that date.", "search": search})
    return JSONResponse({"model": model, "models": list(data["weeks"]), "day": day, "search": search})


@app.post("/api/lab/run")
async def lab_run(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    symbol = _normalize_ticker(str(payload.get("symbol", "")))
    df = await asyncio.to_thread(model_lab.load_prices, symbol)
    if df.empty:
        return JSONResponse({"error": f"No price history for {symbol}."}, status_code=404)
    try:
        bt = await asyncio.to_thread(model_lab.backtest, payload.get("graph") or {}, df)
    except ValueError as exc:  # an unfinished model is a normal state while building, not a failed request
        return JSONResponse({"error": str(exc)})
    tail = df.iloc[-90:]
    candles = [[int(pd.Timestamp(t).normalize().timestamp()), round(float(r.open), 2), round(float(r.high), 2), round(float(r.low), 2), round(float(r.close), 2)]
               for t, r in tail.iterrows()]
    return JSONResponse({"symbol": symbol, "candles": candles, **bt})


@app.get("/api/lab/models")
async def lab_models() -> JSONResponse:
    return JSONResponse(model_lab.list_models())


@app.post("/api/lab/models")
async def lab_save(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    try:
        model_id = model_lab.save_model(str(payload.get("name", "")), payload.get("graph") or {})
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return JSONResponse({"id": model_id})


@app.post("/api/lab/models/{model_id}/delete")
async def lab_delete(model_id: int) -> JSONResponse:
    model_lab.delete_model(model_id)
    return JSONResponse({"status": "deleted"})


@app.post("/api/place-trade")
async def api_place_trade(symbol: str = Form(...), qty: int = Form(0), exit_rule: str = Form("target")) -> JSONResponse:
    """Paper-trade a projected setup: entry, stop and target come from tonight's snapshot, not the client.
    Extended stocks and non-futures pairs are refused -- the page says why instead of offering a button."""
    snapshot = _load_latest_snapshot()
    pred = next((p for p in snapshot.get("predictions", []) if p["symbol"] == symbol), None)
    if pred is None and "/" not in symbol:
        pred = await asyncio.to_thread(_live_prediction, symbol, snapshot)
    if pred is None:
        return JSONResponse({"status": "error", "msg": "That setup is no longer in today's list."}, status_code=404)
    if not pred.get("tradeable", True):
        return JSONResponse({"status": "error", "msg": "Not a valid entry right now — wait for a proper base and pivot."}, status_code=400)
    if live_state.is_halted():
        return JSONResponse({"status": "error", "msg": "Trading is halted. Switch the halt off in Settings to place trades."}, status_code=409)
    qty = qty if qty > 0 else pred["qty"]
    result = place_trade(
        pred["symbol"], pred["side"], pred["entry"], "pairs" if pred["kind"] == "pair" else "minervini",
        qty=qty, target=pred["target"], stop=pred["stop"], kind=pred["kind"],
        ticker_a=pred.get("ticker_a"), ticker_b=pred.get("ticker_b"), hedge_ratio=pred.get("hedge_ratio"),
        price_date=pred["asof"], pending=pred.get("order") == "STOP",
        max_positions=_posture(snapshot)["max_positions"],
        exit_rule=exit_rule if pred["kind"] == "stock" else "target",
    )
    return JSONResponse(result, status_code=200 if result["status"] == "success" else 400)


@app.post("/api/close-trade")
async def api_close_trade(trade_id: int = Form(...)) -> JSONResponse:
    """Close at the last known market price."""
    return JSONResponse(close_trade(trade_id))


@app.post("/api/delete-trade")
async def api_delete_trade(trade_id: int = Form(...)) -> JSONResponse:
    return JSONResponse(delete_trade(trade_id))


@app.post("/api/delete-closed-trades")
async def api_delete_closed_trades() -> JSONResponse:
    return JSONResponse(delete_all_closed())


@app.post("/api/trade-note")
async def api_trade_note(trade_id: int = Form(...), note: str = Form("")) -> JSONResponse:
    return JSONResponse(set_note(trade_id, note))


def _live_prediction(ticker: str, snapshot: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    df = _drop_incomplete_ohlcv_rows(kite_fetch(ticker))
    if df.empty:
        return None
    results = snapshot.get("results") or {}
    results_date = results[ticker] if ticker in results else fetch_next_results([ticker]).get(ticker)
    pred = predict_stock(ticker, df, "Your search", risk_pct=_posture(snapshot)["risk_pct"], results_date=results_date,
                         setup=(_find_screen_item(snapshot, ticker) or {}).get("setup"))
    if pred:
        pred.update(link=f"/stock/{ticker}", source="search", rs_rank=None, on_watchlist=False)
    return pred
