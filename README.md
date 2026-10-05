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
