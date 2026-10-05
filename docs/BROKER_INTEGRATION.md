# Broker integration

## Status
| Adapter | Status |
|---|---|
| `PaperBrokerAdapter` | Planned L6 — wraps the existing paper book (`code/trading/portal.py`). |
| `MockBrokerAdapter` | Planned L6 — deterministic fills/rejections for tests. |
| Motilal Oswal — market data | **Built, read-only, unverified** (`code/live/brokers/motilal.py`). Login, LTP, NSE instrument master. No order endpoints. |
| Motilal Oswal — orders | L9 scaffold only, execution disabled. |

## Market-data provider (L1)
`code/live/market_data.py` defines `MarketDataProvider.quotes(symbols) -> {symbol: Quote}`.
Providers: `SimulatedProvider` (labelled `simulated=True`), `YFinanceDelayedProvider` (free, delayed) and
`MotilalProvider` (live LTP via the read-only client). Chosen by `MARKET_DATA_PROVIDER`.

## Adapter contract (L6)
```
place(order: ApprovedOrder) -> BrokerOrder        # only callable from the approval service
status(broker_ref) -> BrokerOrder
cancel(broker_ref) -> BrokerOrder
positions() -> list[Position]
```
The approval service is the only caller of `place`, and it re-runs every check in RISK_POLICY.md
first.

## Credentials (L9 design)
API key/secret from the environment only. Daily access tokens (most Indian broker APIs expire them
every day) are obtained through the broker's own login redirect, held in memory or encrypted at
rest, never logged, never sent to the browser or to any LLM.

## Motilal Oswal details
Source of truth: Motilal's official SDK `github.com/motradingapi/PythonSDK` (Python 5.0, 01/09/2026), not installed
(it pins urllib3 1.26 with known CVEs and Windows-only packages) but mirrored call for call:

| Call | Endpoint | Notes |
|---|---|---|
| Login | `POST /rest/login/v7/authdirectapi` | `{userid, password: sha256(password+apikey), 2FA: DOB or PAN, totp}` → `AuthToken` |
| Access token | `POST /rest/login/v1/getaccesstoken` | optional second token |
| LTP | `POST /rest/report/v3/getltpdata` | `{exchange: "NSE", scripcode}`; prices in **paise**; `close` = previous close |
| Instruments | `POST /rest/report/v3/getscripsbyexchangename` | `{exchangename: "NSE"}`; symbol in `scripshortname`, series in `optiontype` (EQ preferred) |
| Logout | `POST /rest/login/v5/logout` | |

Dealer accounts need `clientcode` on report calls (error MO1062 triggers one retry with it); normal accounts must not send it (MO2031).
Tokens expire daily; the session counts as gone after 06:00. Base URLs: live `openapi.motilaloswal.com`, test `openapi.motilaloswaluat.com`.

## Open questions
1. Live verification: field names beyond status/AuthToken/data come from the docs as cited by the open-source OpenAlgo integration — confirm on first real login (tests pin the expected shape).
2. Request-rate limits for `getltpdata` aren't published in the SDK; polling is 30 s for ≤60 symbols with 4 parallel requests. Adjust if Motilal throttles.
3. The streaming WebSocket (`wss://ws1feed.motilaloswal.com/jwebsocket/jwebsocket`, binary packets) would cut latency; not used until REST polling is verified.
4. Whether order placement will ever be wanted, or live data + alerts only.
