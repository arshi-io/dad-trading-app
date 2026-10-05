# PRD — MarketPilot

## User
One experienced discretionary swing trader on NSE cash equities who trades Mark Minervini's
SEPA approach (Stage-2 trend template, VCP bases, pivot breakouts, tight stops). Not a
programmer; uses a phone more than a laptop.

## Problem
The after-market work (screening 500 stocks, finding bases and pivots, reading the market)
is done for him already. What's missing is the market-hours half: knowing *when* a stock on
his list actually breaks out, or when a holding breaks down, without staring at screens — and
acting on it with the same discipline (size, stop, market posture) every time.

## What it does
- **After market (done):** screen, setups, market health, projections, AI setup review, paper trading.
- **Market hours (building):** live quotes for the watchlist and setups; alerts for pivot
  breakouts on volume, pullbacks to support, and sell/risk exits; delivered by Telegram and
  browser push.
- **Orders (building):** an alert can become a *proposed order* with size and stop already
  computed. Nothing is sent anywhere until he approves that exact order in the web app, and
  every risk check runs again at that moment. Paper broker first.

## Non-goals
Autonomous trading. Intraday scalping. Options. Any claim of guaranteed returns.

## Success
- Every breakout on his list during market hours reaches his phone within a minute (once a live feed is configured).
- Zero orders without an explicit approval; zero orders while halted, stale, closed, or over a risk limit.
- He can see, for every alert and order, why it happened and what was checked (audit log).
