# Strategies

All outputs are informational. Code references in brackets.

## Trend template (done) [`code/signals/minervini/trend_template.py`]
Minervini's 8 conditions: price > 150/200-day; 150 > 200; 200-day rising ~1 month; 50 > 150/200;
price > 50-day; ≥30% above 52-week low; within 25% of 52-week high; RS rank ≥ 70 (vs NIFTY 500).

## Setup / base detection (done) [`code/signals/minervini/setup.py`]
Most recent base ending 1–10 sessions ago: ≥10 bars, ≤35% deep, pivot = base high.
- `IN BUY RANGE`: closed above pivot, ≤5% past it.  `BASING`: below pivot.
- `EXTENDED`: >5% past pivot, or no base and >10% above the 50-day.  `NO BASE YET`.
- VCP flag: three base segments with shrinking ranges and last-segment volume below the 50-day average.

## Market posture (done) [`code/regime/market_health.py`]
Distribution days (index −0.2%+ on higher volume, last 25 sessions), follow-through day (day 4+,
+1.25% on higher volume off the 3-month low), breadth, sector leadership → risk % and position cap.

## Price ranges (done) [`code/trading/ranges.py`]
Volatility-scaled (EWMA blended with 1-year vol) and adaptively conformal-calibrated. Walk-forward on
141 stocks: 68% band hit 68.0% next-day / 67.8% at 5 days; 90% band 89.6% / 89.2%; holds in volatile
spells (the old bootstrap band fell to 55–58%). Centred on the last close — no direction is forecast.

## Market grade (done) [`code/regime/market_health.py`, `code/research/base_rates.py`]
A: ≥60% of stocks above their 200-day AND NIFTY above its 50-day; B: one; C: neither. From 20,535
honest NSE breakouts 2012–2026 (decide at close, buy next open, costs in): trailing-exit average
A +0.05R, B −0.06R, C −0.10R. Grade C forces the defensive posture.

## Exits (done) [`code/trading/portal.py`]
Paper trades choose "let it run" (stop, then a close below the 21-day average, ≤60 sessions; backtest ≈0R,
best 5% ≈+7R) or a fixed 2R target (backtest −0.06R).

## What did NOT survive testing
Stock-level breakout scoring from daily data: no out-of-sample edge (AUC 0.52) once same-day volume
lookahead was removed. Retest when intraday history is available.

## Projections (superseded) [`code/trading/predict.py`]
2,000 bootstrap paths from the stock's own last year of returns (drift damped to 25%): next-day
range, 5-session cone, odds of 2R target before 1.5×ATR stop. Scored daily against the next close.

## Pairs (done) [`code/signals/pairs_trading.py`, pipeline]
Same-sector NIFTY 100 cointegration (monthly), 60-day z-score, entry ±2σ, exit ±0.5σ, hard exit ±3.5σ,
futures-lot sizing; negative-hedge pairs excluded.

## Live signals (planned, L3)
- **Pivot breakout:** price trades through a `BASING` pivot with projected day volume ≥ 1.4× the 50-day average.
- **Pullback:** a passing stock touches its 10/21-day EMA or 50-day average on below-average volume and holds.
- **Sell / risk exits:** stop hit, close below the 50-day on heavy volume, close back below the pivot after a breakout, climax run (largest up day of the move after an extended advance).
