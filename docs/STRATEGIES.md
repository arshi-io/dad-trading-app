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
Same-sector NIFTY 100 cointegration (Engle-Granger, monthly), OLS hedge ratio, 60-day z-score, entry ±2σ,
exit ±0.5σ, hard exit ±3.5σ, futures-lot sizing; negative-hedge pairs excluded. Half-life filter: only
pairs whose spread halves its gap in 5–40 trading days (`PAIRS_HALF_LIFE_MIN/MAX` in the pipeline);
half-life comes from an AR(1) fit on the spread, the discrete form of an Ornstein-Uhlenbeck process.

## Kalman hedge ratio (shown beside OLS) [`code/signals/kalman_pairs.py`, `code/research/kalman_vs_ols.py`]
State [β, α] follows a random walk (δ = 1e-4; α drifts 100× slower than β so the two don't trade off
against each other); observation A = β·B + α + noise, with the noise variance learned from the
filter's own past forecast errors. Output per day: β, α, spread = forecast error, z = error / its
predicted std. Pure NumPy, no lookahead (tested). The Pairs page shows the Kalman β and z next to
the OLS ones; signals still come from OLS.

Walk-forward backtest, 7 cached pairs + ICICIBANK/SBIN, 2017–2026, same rules (entry |z|>2, exit
|z|<0.5 or >3.5), hedge fixed per trade, 0.10% cost per side:

| | Sharpe | Max DD | Return | Trades | Win rate |
|---|---|---|---|---|---|
| OLS (trailing-year refit monthly) | 0.19 | −20.2% | +8.0% | 343 | 67% |
| Kalman | −0.02 | −5.6% | −0.3% | 439 | 54% |

Before costs Kalman is ahead (Sharpe 0.53 vs 0.39); they cross at about 0.05% per side. Kalman's
z is a one-day surprise, so trades last ~3 days against ~19 for OLS and costs eat the edge. Kalman
won on 4 of 8 pairs. Neither is a strong strategy after costs, so OLS stays in charge. The pairs were
picked by a 2023–26 scan, so the later years are in-sample for both methods.

## HMM market regime (shown beside the grade) [`code/regime/hmm_regime.py`, `code/research/hmm_base_rates.py`]
3-state Gaussian HMM (hmmlearn) on NIFTY's daily log return and 20-day realised volatility,
standardised on the training window. Expanding window, refitted monthly: each day is labelled by a
model trained only on days before its month. Labels come from filtered probabilities (forward pass
up to that day; no Viterbi, no smoothing). States are named by volatility: Calm trend < Choppy < Stress.
Best of 5 fixed-seed fits. Today page shows e.g. "HMM: Choppy 99%"; it does not change sizing.

Breakouts by HMM state on the decision day (20,475 events 2012–2026, trailing exit):

| State | Share of days | Breakouts | Hit 2R | Avg R |
|---|---|---|---|---|
| Calm trend | 69% | 13,891 | 28.0% | −0.02 |
| Choppy | 24% | 5,065 | 25.5% | −0.07 |
| Stress | 7% | 1,519 | 35.2% | +0.44 |

Within grades: A/Calm +0.05, A/Stress +0.37; C/Calm −0.24 (the worst cell: a quiet, grinding
decline), C/Stress +0.63. So the HMM does split each grade further: breakouts right after a volatility
shock (Apr 2021, Jun 2024, Apr 2026) were the best trades in the sample. But the median Stress day
averaged −0.03R, a third of all Stress R came from 5 days, and the result rests on ~10 episodes.
Treat it as a hypothesis worth watching, not a sizing rule.

## Volatility regime (GARCH) [`code/signals/garch_volatility.py`]
GARCH(1,1) on NIFTY returns, 60-day rolling refit every 5 days. LOW/MEDIUM/HIGH cut-offs are the
33rd/66th percentiles of volatility seen *so far* (expanding window), so historical labels don't use
later data. Feeds the Uptrend / Choppy / In correction headline with the 200-day slope and breadth.

## Live signals (planned, L3)
- **Pivot breakout:** price trades through a `BASING` pivot with projected day volume ≥ 1.4× the 50-day average.
- **Pullback:** a passing stock touches its 10/21-day EMA or 50-day average on below-average volume and holds.
- **Sell / risk exits:** stop hit, close below the 50-day on heavy volume, close back below the pivot after a breakout, climax run (largest up day of the move after an extended advance).
