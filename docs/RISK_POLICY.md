# Risk policy

These rules are enforced in code and covered by tests. Changing one is a code change with a
review, not a settings tweak.

## Modes
| Mode | What can happen |
|---|---|
| `PAPER_TRADING` (default) | Alerts, proposed orders, approvals, simulated fills in the paper book. |
| `ALERT_ONLY` | Alerts only. No proposed orders are created. |
| Live execution | **Not available.** Requires an explicit request after paper testing, a reviewed real-broker adapter, and a code change. |

## Order safety (every attempt, immediately before any broker call)
An order is blocked, and a `risk_event` + audit entry written, if any of these is true:
1. Emergency halt is on.
2. Market session is not `OPEN` (weekend, NSE holiday, pre-open, after close).
3. The quote used is older than `QUOTE_STALE_SECONDS` or failed validation (zero/negative price, price outside the day's range).
4. The order has no stop loss, or the stop is wider than `MAX_STOP_PCT` (default 8%).
5. Today's realised + open loss has reached `DAILY_LOSS_LIMIT_PCT` of the account (daily stop).
6. Open risk across positions after this order would exceed `MAX_PORTFOLIO_HEAT_PCT`.
7. Positions + pending orders would exceed the market posture's cap (correction 2, mixed 4, uptrend 8).
8. The order was not approved by the authenticated user in the web UI, or differs from what was approved.

## Sizing
Risk per trade = market posture: 1% (uptrend), 0.5% (mixed), 0.25% (correction) of the account,
with stop = 1.5 × ATR(14) below entry and position value capped at 20% of the account.

## Machines never trade
No scheduler, alert, LLM or tool has an order-placing code path. Proposed orders are created
from alerts; only a human approval request can move one forward.

## Language
Signals are labelled informational. No screen, alert or message claims or implies guaranteed profit.
