"""Runtime settings from the environment (and an optional .env file at the repo root).

Secrets never live in code: tokens and keys are read here and nowhere else. Defaults are the
safe ones -- paper trading, simulated prices, alerts in dry-run.
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv(path: Path) -> None:
    """Minimal KEY=VALUE loader; real environment variables always win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if value[:1] in ("'", '"') and value[1:].find(value[0]) >= 0:
            value = value[1:1 + value[1:].find(value[0])]
        else:
            value = value.split(" #", 1)[0].split("\t#", 1)[0].strip()  # trailing comment
        os.environ.setdefault(key.strip(), value)


_load_dotenv(REPO_ROOT / ".env")


def _bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    return default if raw is None else raw.strip().lower() in ("1", "true", "yes", "on")


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# Modes that can exist. LIVE execution is deliberately absent: enabling real orders is a
# later, explicit decision (see docs/RISK_POLICY.md), not a config flag.
APP_MODES = ("PAPER_TRADING", "ALERT_ONLY")
APP_MODE = os.getenv("APP_MODE", "PAPER_TRADING").upper()
if APP_MODE not in APP_MODES:
    APP_MODE = "PAPER_TRADING"

TIMEZONE = "Asia/Kolkata"

BROKER = os.getenv("BROKER", "paper").lower()                                  # paper | mock
MARKET_DATA_PROVIDER = os.getenv("MARKET_DATA_PROVIDER", "simulated").lower()  # simulated | yfinance_delayed | broker
QUOTE_STALE_SECONDS = int(_float("QUOTE_STALE_SECONDS", 90))
LIVE_POLL_SECONDS = int(_float("LIVE_POLL_SECONDS", 30))

# Risk policy defaults (docs/RISK_POLICY.md); the market-posture limits still apply on top.
DAILY_LOSS_LIMIT_PCT = _float("DAILY_LOSS_LIMIT_PCT", 2.0)
MAX_PORTFOLIO_HEAT_PCT = _float("MAX_PORTFOLIO_HEAT_PCT", 6.0)
MAX_STOP_PCT = _float("MAX_STOP_PCT", 8.0)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
TELEGRAM_DRY_RUN = _bool("TELEGRAM_DRY_RUN", True) or not (TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

VAPID_PUBLIC_KEY = os.getenv("VAPID_PUBLIC_KEY", "")
VAPID_PRIVATE_KEY = os.getenv("VAPID_PRIVATE_KEY", "")
VAPID_SUBJECT = os.getenv("VAPID_SUBJECT", "mailto:admin@example.com")
WEB_PUSH_DRY_RUN = not (VAPID_PUBLIC_KEY and VAPID_PRIVATE_KEY)


def summary() -> dict:
    """Non-secret view of the configuration, for /healthz and the runbook."""
    return {
        "app_mode": APP_MODE,
        "broker": BROKER,
        "market_data_provider": MARKET_DATA_PROVIDER,
        "telegram": "dry-run" if TELEGRAM_DRY_RUN else "live",
        "web_push": "dry-run" if WEB_PUSH_DRY_RUN else "live",
        "timezone": TIMEZONE,
    }
