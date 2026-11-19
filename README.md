# Papa Terminal (MarketPilot)

A private, single-user NSE swing-trading copilot for a Minervini-style trader: after-market
screening and setups, paper trading, and (in progress) market-hours alerts with a
human-approved order workflow. **Decision support only — signals are informational, not
advice, and nothing here guarantees returns.** Default mode is paper trading.

## Run locally (Windows / macOS / Linux, Python 3.12)

```bash
python -m venv .venv
.venv/Scripts/activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # then fill APP_PIN (bcrypt hash) and SESSION_SECRET
python -m code.pipeline.nightly_pipeline   # first data build, ~2 min
uvicorn code.app.main:app --host 0.0.0.0 --port 8731
```

Open http://localhost:8731 (or http://<your-LAN-IP>:8731 from a phone on the same Wi-Fi).

## Run with Docker

```bash
cp .env.example .env              # fill it in
docker compose up --build
docker compose exec app python -m code.pipeline.nightly_pipeline   # first data build
```

## Tests

```bash
python -m pytest tests/ -q
```

## Models

| Model | Where | What it does | Used for |
|---|---|---|---|
| Minervini trend template + RS | `code/signals/minervini/` | 8-point Stage-2 screen of the NIFTY 500 | Screen, setups |
| Market grade A/B/C | `code/regime/market_health.py` | breadth + NIFTY vs 50-day | Position size, posture |
| GARCH(1,1) volatility regime | `code/signals/garch_volatility.py` | LOW/MED/HIGH volatility (expanding cut-offs) | Today headline |
| HMM regime | `code/regime/hmm_regime.py` | 3-state HMM, filtered probabilities, monthly refit | Shown beside the grade only |
| Engle-Granger pairs + OLS hedge | `code/signals/pairs_trading.py` | cointegration scan, 5–40 day half-life filter | Pairs signals |
| Kalman hedge ratio | `code/signals/kalman_pairs.py` | daily-updating hedge and surprise z-score | Shown beside OLS only |
| Calibrated ranges | `code/trading/ranges.py` | 68/90% next-day and 1-week ranges, adaptive conformal | Predictions |

Research scripts (`python -m code.research.<name>`): `base_rates`, `hmm_base_rates`, `kalman_vs_ols`.
Results and caveats: [docs/STRATEGIES.md](docs/STRATEGIES.md).

## Health

`GET /healthz` (no login) → database, snapshot freshness, market session, mode, halt state.

## Docs

| | |
|---|---|
| [docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) | phases, decisions, open questions |
| [docs/PRD.md](docs/PRD.md) | who it's for and what it does |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | components and data flow |
| [docs/RISK_POLICY.md](docs/RISK_POLICY.md) | hard limits and order safety rules |
| [docs/STRATEGIES.md](docs/STRATEGIES.md) | screen, setups, signals, projections |
| [docs/BROKER_INTEGRATION.md](docs/BROKER_INTEGRATION.md) | adapter design and broker status |
| [docs/API.md](docs/API.md) | HTTP endpoints |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | operating, halting, recovering |
