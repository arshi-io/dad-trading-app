# HTTP API

All routes except `/login`, `/logout` and `/healthz` require the PIN session cookie. All POSTs must be
same-origin (Origin/Referer checked). Responses are HTML pages or JSON.

## Pages
| Route | |
|---|---|
| `GET /` | Today: market, setups, briefing, watchlist |
| `GET /screen?view=pass|close|all` | Minervini screen, VCP watch |
| `GET /pairs` | Pairs |
| `GET /stock/{symbol}` | Stock Room |
| `GET /predictions?symbol=` | Paper trading (optional live projection for any symbol) |

## JSON / fragments
| Route | |
|---|---|
| `GET /healthz` | Public health: database, snapshot freshness, session phase, mode, halt, non-secret config |
| `GET /api/state` | `{mode, halted, session: {phase, is_open, next_open, holiday}}` |
| `POST /api/halt` | form `on=1|0`, `reason` — emergency halt switch (audited) |
| `GET /api/candles/{symbol}` | OHLCV + 50/150/200-day series |
| `GET /api/memo/{symbol}` | Setup review card (HTML fragment) |
| `GET /api/headlines/{symbol}` | Company news (HTML fragment) |
| `GET /api/screen-row/{symbol}` | Checklist detail (HTML fragment) |
| `GET /api/tickers.json` | Search directory |
| `POST /api/place-trade` | form `symbol`, `qty` — paper trade from tonight's setup |
| `POST /api/close-trade` | form `trade_id` — exit at last price |
| `POST /api/delete-trade` | form `trade_id` — closed/expired/pending only |
| `POST /api/delete-closed-trades` | |
| `POST /api/trade-note` | form `trade_id`, `note` |
| `POST /api/watchlist/add` · `/remove` | form `symbol`, `next` |
| `GET /api/quotes` | Latest validated live quote per watched symbol, with `stale`, `simulated`, provider and session |
| `GET /broker` | Broker status page (connect / disconnect) |
| `POST /broker/connect` | form `password`, `totp` — Motilal login; password used once, not stored (audited) |
| `POST /broker/disconnect` | (audited) |

Planned (L2–L6): alerts, proposed orders, approvals, push subscriptions.
