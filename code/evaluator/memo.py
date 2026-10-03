from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "papa.db"

from code.app import config


def set_model_provider(provider: str) -> None:
    config.MODEL_PROVIDER = provider


def _init_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS memos (
            ticker TEXT NOT NULL,
            snapshot_date TEXT NOT NULL,
            payload TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (ticker, snapshot_date)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_trades (
            id INTEGER PRIMARY KEY,
            ticker TEXT NOT NULL,
            side TEXT NOT NULL,
            entry_price REAL NOT NULL,
            exit_price REAL,
            entry_date TEXT NOT NULL,
            exit_date TEXT,
            signal_type TEXT,
            status TEXT DEFAULT 'open',
            pnl REAL,
            pnl_pct REAL,
            notes TEXT,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def _load_cached_memo(ticker: str, snapshot_date: str) -> Optional[Dict[str, Any]]:
    conn = _init_db()
    try:
        row = conn.execute(
            "SELECT payload, created_at FROM memos WHERE ticker=? AND snapshot_date=?",
            (ticker, snapshot_date),
        ).fetchone()
        if row is None:
            return None
        payload_text, created_at = row
        try:
            created = datetime.fromisoformat(created_at)
        except ValueError:
            return None
        if datetime.utcnow() - created > timedelta(hours=24):
            return None
        payload = json.loads(payload_text)
        return payload
    finally:
        conn.close()


def _store_cached_memo(ticker: str, snapshot_date: str, payload: Dict[str, Any]) -> None:
    conn = _init_db()
    try:
        conn.execute(
            "INSERT OR REPLACE INTO memos (ticker, snapshot_date, payload, created_at) VALUES (?, ?, ?, ?)",
            (ticker, snapshot_date, json.dumps(payload), datetime.utcnow().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def flush_memo_cache() -> int:
    """Delete all cached memos so they regenerate for real. Run once by hand after setting ANTHROPIC_API_KEY -- not called automatically."""
    conn = _init_db()
    try:
        cursor = conn.execute("DELETE FROM memos")
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def _strip_fences(text: str) -> str:
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.S | re.I)
    if match:
        return match.group(1).strip()
    return text


def _parse_json_response(text: str) -> Dict[str, Any]:
    cleaned = _strip_fences(text)
    parsed = json.loads(cleaned)
    if not isinstance(parsed, dict):
        raise ValueError("response is not a JSON object")
    return parsed


def _call_llm(prompt: str) -> str:
    if config.MODEL_PROVIDER == "openai":
        return _call_openai(prompt)
    return _call_claude(prompt)


def _short_error(exc: Exception) -> str:
    """A one-line, secret-free summary of a failed API call -- HTTP status
    when there is one (no headers/body, so never leaks the key), otherwise
    the exception text truncated so a stray huge payload can't blow up the
    card."""
    response = getattr(exc, "response", None)
    if response is not None:
        return f"HTTP {response.status_code} {response.reason}"
    return str(exc)[:160]


def _sentinel(reasoning: str) -> str:
    """A gracefully-degraded memo -- always valid JSON, always score<60/AVOID
    so the '60=no trade' rule holds, but reasoning says specifically WHY
    there's no real analysis rather than a single catch-all message.
    sentinel=true lets callers detect this case without string-matching
    the (now variable) reasoning text."""
    return json.dumps({
        "score": 55, "verdict": "AVOID", "reasoning": reasoning,
        "risk_flags": [], "sentinel": True,
    })


def _call_claude(prompt: str) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return _sentinel("Analysis unavailable — MODEL_PROVIDER=claude but no ANTHROPIC_API_KEY is configured.")
    payload = {
        "model": "claude-sonnet-4-6",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    try:
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data["content"][0]["text"]
    except Exception as exc:
        logger.warning("Claude call failed: %s", exc)
        return _sentinel(f"Analysis unavailable — Claude call failed: {_short_error(exc)}")


OPENAI_MODEL = "gpt-5.6-terra"  # verified against developers.openai.com/api/docs/models 2026-08-20 -- balanced mid tier, not the flagship "sol" or budget "luna"


def _call_openai(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _sentinel("Analysis unavailable — MODEL_PROVIDER=openai but no OPENAI_API_KEY is configured.")
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "content-type": "application/json"}
    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload, timeout=20)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        logger.warning("OpenAI call failed: %s", exc)
        return _sentinel(f"Analysis unavailable — OpenAI call failed: {_short_error(exc)}")


def build_bundle(
    ticker: str,
    signal: Optional[Dict[str, Any]] = None,
    headlines: Optional[List[Dict[str, Any]]] = None,
    diary: Optional[List[str]] = None,
    trend_template: Optional[Dict[str, Any]] = None,
    market_regime: Optional[str] = None,
    delivery_pct: Optional[float] = None,
) -> Dict[str, Any]:
    """Full evidence bundle for the memo prompt.

    trend_template: this ticker's Minervini 8-point checklist result + RS
    rank from the latest screen snapshot, or None if it isn't in the
    NIFTY 500 screen universe -- never fabricated.
    market_regime: TRENDING_UP/CHOPPY/TRENDING_DOWN from the snapshot, or
    None if unavailable.
    delivery_pct: NSE delivery %, or None -- not wired yet (needs bhavcopy),
    passed through as None rather than a fake number so the LLM can say so
    honestly instead of inventing a figure.
    """
    return {
        "ticker": ticker,
        "signal": signal or {},
        "headlines": headlines or [],
        "diary": diary or [],
        "trend_template": trend_template,
        "market_regime": market_regime,
        "delivery_pct": delivery_pct,
    }


# Exchange suffixes ("RELIANCE.NS" -> "ns") and single-char fragments (from
# splitting e.g. "M&M") are too generic to signal a real match -- "ns" alone
# substring-matches almost any headline (e.g. "conso[ns]olidated") and was
# drowning out the actual ticker-name token.
_TOKEN_STOPWORDS = {"ns", "bo"}


def match_headlines(ticker: str, headlines: List[Dict[str, Any]], limit: int = 5) -> List[Dict[str, Any]]:
    """Select the most relevant headlines for a ticker using a lightweight keyword matcher."""
    if not headlines:
        return []
    tokens = {
        token.lower() for token in re.split(r"[^a-z0-9]+", ticker.lower())
        if token and len(token) >= 2 and token.lower() not in _TOKEN_STOPWORDS
    }
    scored: List[tuple[int, Dict[str, Any]]] = []
    for item in headlines:
        title = str(item.get("title", "")).lower()
        text = title + " " + str(item.get("source", "")).lower()
        score = 0
        for token in tokens:
            if token and token in text:
                score += 3
        for keyword in ("results", "rbi", "expiry", "earnings", "policy", "bank", "infra", "oil", "fii"):
            if keyword in text:
                score += 1
        scored.append((score, item))
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("title", "")))
    return [item for _, item in scored[:limit]]


def generate_memo(bundle: Dict[str, Any], ticker: str, snapshot_date: Optional[str] = None, provider: Optional[str] = None) -> Dict[str, Any]:
    snapshot_date = snapshot_date or datetime.utcnow().date().isoformat()
    if provider:
        set_model_provider(provider)

    cached = _load_cached_memo(ticker, snapshot_date)
    if cached:
        return cached

    matched_headlines = match_headlines(ticker, bundle.get("headlines", []))
    prompt_bundle = {
        **bundle,
        "matched_headlines": matched_headlines,
    }
    prompt = (
        "You are scoring a trade setup for a seasoned NSE trader who reads Minervini. "
        "Return strict JSON only. "
        "Schema: {\"score\": 0-100, \"verdict\": \"TRADE_VALID\"|\"WAIT\"|\"AVOID\"|\"NO_CLEAR_EDGE\", "
        "\"reasoning\": \"120-180 words, plain trader English\", \"risk_flags\": [\"...\"], "
        "\"invalidation\": [\"3-4 concrete conditions\"]}. "
        "Bundle field meanings -- trend_template: this ticker's Minervini 8-point Stage-2 checklist result and RS rank (0-99) from tonight's screen, null if not in the NIFTY 500 screen universe. "
        "market_regime: overall NIFTY regime (TRENDING_UP/CHOPPY/TRENDING_DOWN), null if unavailable. "
        "signal: the mean-reversion z-score signal for this ticker. "
        "delivery_pct: NSE delivery percentage, null if not available -- never invent a number for it. "
        "matched_headlines: recent news matched to this ticker; treat unmatched/generic market headlines as weak, not company-specific, evidence. "
        "Weigh trend_template + market_regime + signal as the primary evidence; headlines and diary are supporting context only. "
        "Do not invent facts; cite only the provided inputs, and say plainly when a field is null/empty rather than working around it. "
        "If score < 60, verdict must be AVOID or NO_CLEAR_EDGE -- never TRADE_VALID or WAIT. "
        "Use NO_CLEAR_EDGE instead of forcing AVOID when the evidence is genuinely mixed or too thin to take a directional view either way (e.g. trend_template and signal disagree, or most fields are null) -- it is a valid abstention, not a synonym for AVOID. "
        "invalidation: 3-4 short bullets naming the SPECIFIC numeric levels already present in trend_template/signal/market_regime that would flip this call if breached -- e.g. \"Close below the 50-day SMA (use the exact SMA50 value from trend_template)\", \"RS rank falls below 70\", \"200-day SMA turns down\", \"Regime shifts to TRENDING_DOWN\". Use the real numbers from the bundle, never placeholders. "
        f"Bundle: {json.dumps(prompt_bundle)}"
    )
    raw_text = _call_llm(prompt)
    parsed: Dict[str, Any] | None = None
    try:
        parsed = _parse_json_response(raw_text)
    except Exception:
        retry_prompt = prompt + "\nReturn only JSON again; do not add prose."
        retry_text = _call_llm(retry_prompt)
        try:
            parsed = _parse_json_response(retry_text)
        except Exception:
            parsed = {
                "score": 55,
                "verdict": "AVOID",
                "reasoning": "Analysis unavailable — the model's response couldn't be parsed as JSON, even after a retry.",
                "risk_flags": [],
                "invalidation": [],
                "sentinel": True,
            }

    payload = {
        "ticker": ticker,
        "snapshot_date": snapshot_date,
        "score": int(parsed.get("score", 55)),
        "verdict": parsed.get("verdict", "AVOID"),
        "reasoning": parsed.get("reasoning", "Analysis unavailable"),
        "risk_flags": parsed.get("risk_flags", []),
        "invalidation": parsed.get("invalidation", []),
        "sentinel": bool(parsed.get("sentinel", False)),
        "model_provider": config.MODEL_PROVIDER,
        "generated_at": datetime.utcnow().isoformat(),
    }
    # "below 60 = no trade" -- AVOID and NO_CLEAR_EDGE both already satisfy
    # that (an abstention IS a no-trade outcome), only force-correct a
    # verdict that wrongly stayed TRADE_VALID/WAIT under a low score.
    if payload["score"] < 60 and payload["verdict"] not in ("AVOID", "NO_CLEAR_EDGE"):
        payload["verdict"] = "AVOID"
    _store_cached_memo(ticker, snapshot_date, payload)
    return payload
