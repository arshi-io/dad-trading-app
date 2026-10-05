# MarketPilot — implementation plan

MarketPilot is Papa Terminal grown into a market-hours copilot: the existing after-market
engine stays exactly as it is, and live quotes, live alerts and a human-approved order
workflow are added around it. Paper trading is the default and the only mode that can
place anything until real execution is explicitly requested.

## Decisions (2026-10-04)

| Topic | Spec said | Decided | Why |
|---|---|---|---|
| Stack | Next.js + Postgres + Redis + Celery monorepo | **Extend the existing FastAPI + Jinja app** | The after-market product works and must not be rewritten. Repo rule: no React. |
| Database | Postgres + SQLAlchemy + Alembic | **SQLite (existing `code/data/papa.db`) + a small versioned migration runner** (`code/live/db.py`) | One user, one process. Postgres can be swapped in later behind the same functions. |
| Jobs | Redis + Celery/Dramatiq | **APScheduler (already running) + an in-process async loop for live quotes** | No second process to operate on a laptop or one Railway service. |
| Live prices | Mock first | **Broker feed** behind a `MarketDataProvider` interface; mock + yfinance-delayed providers ship first so everything runs without keys | Broker not chosen yet — see open questions. |
| Alerts | Telegram | **Telegram + browser push**, both dry-run until configured | User choice. Browser push needs HTTPS (Railway) except on localhost. |

## What already exists (unchanged by this plan)

Nightly + 08:30 pipeline · Minervini 8-point screen + RS · base/pivot/VCP setup detection ·
market health (distribution days, follow-through, breadth, sectors, risk posture) · pairs with
futures lots · results dates · publisher-dated company news · AI setup review · Monte-Carlo
projections and scorecard · paper trading (buy-now + breakout orders, auto stop/target,
journal notes) · PIN auth, CSP, CSRF origin check, security headers · Kite-style UI · 67 tests.

## Phases

Each phase ends with: files changed, commands run, test results, limitations, manual QA steps.

| # | Phase | Contents | Status |
|---|---|---|---|
| L0 | Foundation | `.env.example` + settings module, migration runner + new tables, audit log, market-session service (NSE hours + editable holiday calendar), app mode (PAPER_TRADING default) + emergency halt state, `/healthz`, Dockerfile + compose, docs set, simulated demo feed | done |
| L1 | Live market data | `MarketDataProvider` protocol: `SimulatedProvider`, `YFinanceDelayedProvider`, broker provider; quote cache with staleness flags; session-aware polling loop | done (Motilal untested until credentials) |
| L2 | Live UI | Live watchlist page, live quotes on Stock Room/Paper Trading (additive), intraday chart | |
| L3 | Live signals | Pivot breakout on volume, pullback to 10/21-day EMA / 50-day, sell/risk exits (stop, 50-day break, climax run); unit tests | |
| L4 | Risk engine | Pre-trade checks: stale quote, market closed, halt, daily stop, portfolio heat, max positions, missing stop; exhaustive tests | |
| L5 | Alerts | Signal → alert lifecycle, alert centre, Telegram + Web Push adapters (dry-run first) | |
| L6 | Approval workflow | Proposed order → human approves exact order in UI → risk re-check → broker adapter (Paper/Mock only) → fills → positions | |
| L7 | Backtesting | VectorBT runs for the stock strategies, results UI | |
| L8 | Reports | Daily report, weekly setup-performance report, journal analytics | |
| L9 | Real broker scaffold | Adapter + credential/session design for the chosen broker; execution disabled | |
| L10 | Hardening | Full test pass, security review, runbook, demo walkthrough | |

## Safety invariants (enforced in code, tested)

- Default mode `PAPER_TRADING`. `LIVE_EXECUTION` cannot be enabled from config alone in this plan.
- No code path lets an LLM, scheduler or alert place an order; only an authenticated UI approval can.
- Every order attempt re-runs all risk/market checks immediately before the adapter call.
- Emergency halt blocks all new orders and is checked on every attempt.
- Every state change (mode, halt, approval, order, fill, alert) is written to the audit log.
- All signals are labelled informational, not advice; no profit claims anywhere.

## Prediction work (2026-10-04)
Shipped: calibrated ranges, market grade + breadth posture, honest history per grade, trailing-exit option.
Research code: `code/research/` (bhavcopy backfill 2011+, events, walk-forward model, base rates).
Retracted: a stock-level breakout model that looked strong only because of same-day volume lookahead.

## Open questions

1. ~~Which broker?~~ **Motilal Oswal** (2026-10-04). Read-only client built from Motilal's official SDK; needs API key + account to verify.
2. Telegram bot token + chat id (created via @BotFather) — until then Telegram runs dry-run.
3. Hosting for browser push — needs HTTPS (Railway deploy) or localhost.
