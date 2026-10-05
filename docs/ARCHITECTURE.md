# Architecture

One Python 3.12 process (FastAPI + Jinja + APScheduler) and one SQLite file. Chosen over the
spec's Next.js/Postgres/Redis/Celery to extend the working app instead of rewriting it
(see IMPLEMENTATION_PLAN.md, Decisions).

```
                 ┌──────────────── nightly 18:30 / 08:30 (APScheduler) ───────────────┐
 yfinance EOD ──►│ pipeline: screen · setups · market health · pairs · predictions    │──► snapshot JSON
 NSE lots/hols ─►│           results dates · settle paper trades · AI reviews         │
                 └─────────────────────────────────────────────────────────────────────┘
                 ┌──────────────── market hours (planned, L1–L5) ─────────────────────┐
 broker feed ───►│ MarketDataProvider → quote cache (staleness) → signal engine       │──► alerts ──► Telegram / Web Push
 (simulated now) │                                                                     │
                 └─────────────────────────────────────────────────────────────────────┘
 browser (PIN) ──► FastAPI pages + JSON API ──► proposed order ──► human approval ──► risk re-check ──► BrokerAdapter (paper/mock)
                                                                                                      │
                                         every step ──► audit_log (hash-chained)  ◄────────────────────┘
```

## Modules

| Path | Role |
|---|---|
| `code/pipeline/` | nightly pipeline, news |
| `code/signals/` | Minervini template, setup/VCP detection, mean reversion, pairs |
| `code/regime/` | market regime and market health (distribution days, follow-through, posture) |
| `code/trading/` | paper portal (positions, fills, settlement) and projections |
| `code/evaluator/` | AI setup review (LLM), shared review builder |
| `code/live/` | **new:** `settings`-driven mode/halt (`state.py`), audit log, schema migrations, market session, market-data providers |
| `code/app/` | FastAPI app, templates, static assets, settings |

## Data
`code/data/papa.db` (SQLite): watchlist, memos, paper_trades, prediction_log, and the live tables
from `code/live/db.py` (audit_log, app_state, alerts, proposed_orders, broker_orders, risk_events,
push_subscriptions). Snapshots are JSON in `code/data/snapshots/`. Caches (prices, lots, holidays,
results dates) are regenerated automatically.

## Time
Everything market-related uses Asia/Kolkata. Trading days come from NSE's published holiday list
(`code/live/session.py`, cached weekly), never from hardcoded dates.
