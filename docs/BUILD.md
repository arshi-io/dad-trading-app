# Marketly prediction build: what was asked, what was built, what it found

This file is the record of the prediction work on branch `feat/gatekeeper` (merged to `master`).
It lists every request in order, what was built for it, the honest results, the mistakes made
along the way, and what is still open. Numbers marked *2024+* come from a period no model was
chosen on (the "locked" test).

---

## 1. Did it get built the way you asked?

| # | You asked for | What was built | Status |
|---|---|---|---|
| 1 | Research agent as a multi-model team (librarian, idea maker, coder, critic), one Ollama model in RAM at a time, critic checks, ensembles, journal, optional cloud review; gatekeeper untouched | `research/agent/` + `gatekeeper/` exactly as the PRD said | Built. First real sessions: every idea failed the critic, none reached the gatekeeper |
| 2 | Model Lab: clearer ranges + a cumulative score with prediction accuracy | Track record chart (how often the range caught the price), honest/too-narrow verdict | Built |
| 3 | "A predictive score, not a range I can figure out myself" | Chance-up-in-5-sessions score per stock (`updown.py`) | Built. **Result: no edge** (AUC 0.516) on its own |
| 4 | Make it predict like a quant | Daily ranking of all ~1,000 liquid stocks; rank IC (the quant measure) | Built. **Result: real edge**, IC 0.060, positive 12 of 12 years |
| 5 | Sandbox: replay old weeks, agents improving by trying combinations until it's close | Replay page + model search (rounds of combinations, best kept, locked test) | Built. *Deviation:* the "agents" are a plain search loop, not LLM agents (faster, repeatable). Search lifted 2024+ IC 0.060 → 0.089 |
| 6 | Guide me: the ideal engine, ask my use first, plan then build | 9-layer engine write-up, 4 questions, 5-phase plan | Done |
| 7 | Phase 1: hedged swing portfolio | Weekly portfolio, real Indian costs, Nifty futures / puts hedges | Built. 2024+: +16.2%/yr, Sharpe 0.66 vs Nifty +6.3%, 0.04 |
| 8 | Slippage stress test | Flat 0.1–0.5% and size-based (₹10 lakh–₹2 crore) slippage | Built. Edge survives to ~0.4%/side |
| 9 | Phase 2: 1–3 month position book | 20- and 60-session models, position books, combined book | Built. 20-session book 2024+: +25.4%/yr, Sharpe 1.16; combined (rules fixed in advance) +18.8%, 0.92 |
| 10 | Predict tomorrow for a corpus, trading plan, cross out misses during the day, learn, day-wise P&L on ₹10,000 | Daily loop (`daily.py`, `/daily`): Nifty 100 calls, 5 stocks × ₹2,000, nightly refit, frozen predictions, ledger, live crossing-out, miss patterns | Built. *Limits:* live crossing-out needs a real price feed (default is simulated); "learning" = nightly refit + miss-pattern table |
| 11 | Proven public models / papers | Qlib Alpha158, WorldQuant 101 (25 OHLCV-only), Gu-Kelly-Xiu NN3, linear, 7 classic factors | Code + tests built. **Real-data results pending** (first run stopped: data hole, see §4) |
| 12 | Model Lab: understand common patterns across models, blend them | Model Zoo panel: what each model leans on, agreement, closest classic factor, blends | Built, waiting on the same run |
| 13 | Commit and deploy to Railway, nightly push of results | Committed, pushed to `master`; token-protected `/api/daily/sync`; Windows tasks 18:45 + 21:00 | Pushed. *Not verified:* Railway build (CLI logged out); you still set `DAILY_SYNC_TOKEN` in Railway and `DAILY_SYNC_URL` in `.env` |
| 14 | One price, not a range | Model Lab leads with one price + BEAT/LAG call + its record; range demoted to a stop band | Built |
| 15 | Teach it to short | Book A-short (short bottom 5) and Book A long-short, intraday | Built. Book B stays long-only (overnight shorts need futures lots far above ₹10,000) |
| 16 | Act like a seasoned trader: news, chart patterns, quant models, know when to use what | Not built yet | **Open** — plan in §6 |
| 18 | Pause the governor. Run it like a quant hedge-fund manager: divide the risk, run for profit; a losing day = penalty; 15 net penalties or 3 losing days in a row = the agent dies and a different setup takes over; live from November, ₹10,000 → ₹20,000 | `code/research/fund.py`: one live fund (born on the first live evening; replayed history never counts), three sleeves (intraday long-short, new 5-session swing book, 20-session position book), weekly split toward the sleeves earning most per unit of risk (10–50% each), penalties and death exactly as chosen, post-mortem, successor = next best untried setup with a different model family for the sleeve that lost most; governor switched off (`GOVERNOR_ON = False`) | Built. *Heads-up:* 3 losing days in a row will happen every few weeks even for a profitable fund, so agents will die often at first; both limits are single settings. A 2%-per-day loss stop isn't modelled (needs intraday prices) |
| 17 | When it loses: re-test, retrain, change patterns; never sit under Nifty; penalties every time it does; trades change daily | Governor in the daily loop: every book scored against Nifty each evening with a 2× penalty for every shortfall; a main ₹10,000 book follows the best score (Nifty itself is a candidate scoring 0, so it falls back to Nifty when everything trails); a retrain (model search on the latest year) fires when the main book trails Nifty by 2% over 20 sessions; contrarian short-top-5 book added as a candidate; position book now checks holdings daily (exit at mid-ranking or −5%) | Built. *Honest limit:* no rule can guarantee never being under Nifty on a given day; the fallback keeps long stretches below Nifty short |

Phases from the plan that are **not** built: Phase 3 (new data: F&O open interest, results
surprises, FII flows) and Phase 5 (paper tables generated from the same numbers).

---

## 2. Build process, in order

1. **Gatekeeper** (`gatekeeper/gate.py`): Periods A 2011–20, B 2021–23, C 2024+ (vault). Strategies
   run in an isolated process, lookahead caught by re-running on cut-off data, one Period C attempt
   per strategy, split/bonus adjustment from overnight gaps (NSE `prev_close` is unadjusted).
2. **Research agent** (`research/agent/`): roles in `roles.yaml`, one Ollama model loaded at a time,
   batches by role, critic checks (lookahead, <30 trades, attempts, one lucky year, costs),
   ensembles (vote / HMM regime), SQLite journal by model. Fixes after real runs: RAM retries,
   qwen3 thinking off, one coder repair attempt.
3. **Model Lab track record**: past ranges drawn on the chart, misses marked, running coverage.
4. **Up/down score** (`code/research/updown.py`): one model across all liquid NSE stocks,
   refitted monthly, survivorship-free bhavcopy 2011+. Overall no edge; the 60%+ calls came from a
   few 2025 sell-off weeks (84% of profit from 20 days).
5. **Ranking view**: same scores ranked within each day. Rank IC 0.060, top tenth beat bottom tenth
   by 0.62% per 5 sessions, positive every year 2015–2026. Mostly an avoid list.
6. **Model search** (`code/research/model_search.py`): model (trees ×2, linear) × target (up, beat the
   median, rank of move) × inputs (base, + quant factors, + Minervini trend) × scaling (raw, ranked
   within the day). Rounds change one thing; keep a change only if IC rises > 0.002; then blends.
   Chosen on 2021–23, checked once on 2024+, every look logged.
7. **Portfolio test** (`code/research/portfolio.py`): 20 stocks, weekly, buffer rule, STT/stamp/fees +
   slippage, hedges; slippage stress; position books; combined book.
8. **Daily loop** (`code/research/daily.py`): every evening settle → refit → predict → plan; SQLite
   triggers stop predictions being edited; ledger per book; push to Railway.
9. **Model Zoo** (`code/research/signals_lib.py`, `zoo.py`): published feature sets and models,
   family importance by within-day shuffling, agreement, IC-weighted blend.
10. **Single price** (`code/research/point_forecast.py`): tenth-of-ranking → average move → price,
    record learned on 2021–23 and scored on 2024+.
11. **Shorting**: intraday books (buy top 5, short bottom 5, both, and contrarian short top 5).
12. **Governor** (`daily.py`: `scores`, `decide`, `settle_nifty_and_main`, `retrain`): the layer that
    judges results and changes course — penalised score vs Nifty, champion switching with a margin
    and a cost, Nifty fallback, retrain trigger. The main book always uses the pick made the evening
    before, never one made after seeing the day. *Paused* at your request.
13. **Fund manager** (`code/research/fund.py`): the live fund and its agents. Each evening the daily
    loop settles the sleeves, then the fund books its day (weights × sleeve returns), applies
    penalties, re-splits the money every 5 sessions, and replaces a dead agent. The agent's model
    setups drive the next day's plans. Tables: `fund_agents` (lineage, cause of death,
    post-mortem) and `fund_ledger` (day by day).

Commands (laptop; the NSE history lives only here):

```
python -m code.research.updown                 # up/down score (~12 min)
python -m code.research.model_search --horizon 5|1|20|60
python -m code.research.portfolio [--stress]
python -m code.research.point_forecast
python -m code.research.zoo --horizon 1
python -m code.research.daily [--replay 60]    # nightly; also pushes to Railway when configured
```

---

## 3. Results that hold up (2024+, never used to choose)

| What | Number | Compared with |
|---|---|---|
| 5-session ranking, search winner | rank IC 0.089 | live model 0.060 |
| 20-session ranking | rank IC 0.091 | 0.070 |
| 60-session ranking | rank IC 0.105 | 0.097 |
| 1-day ranking | rank IC 0.092 | 0.052 |
| Swing book, Nifty futures hedge | +16.2%/yr, Sharpe 0.66, worst fall −10% | Nifty +6.3%, 0.04, −16% |
| 20-session position book, hedged | +25.4%/yr, Sharpe 1.16, −13% | singled out after seeing 2024+, so treat as optimistic |
| Combined book (rules fixed before results) | +18.8%/yr, Sharpe 0.92, −13% | the honest headline |
| Beat/lag call per stock | right 53.5% of weeks | coin flip 50% |
| Exact price | off by 4.3% on average | same as "no change" — trade the call, not the rupee |

For reference, Microsoft Qlib's public daily benchmark (Alpha158 + LightGBM, China CSI300) reports
rank IC ≈ 0.047. The 2024+ period has now been looked at many times; the only fully fresh test from
here is the live daily loop.

---

## 4. Mistakes found and fixed

| Problem | Effect | Fix |
|---|---|---|
| NSE `prev_close` is unadjusted | splits looked like −50% crashes | gap-based adjustment everywhere; old `setups.build_panel` still uses `prev_close` (flagged, not fixed) |
| 6 ETFs slipped past the stock filter | HDFCSML250 etc. ranked as stocks | explicit ETF list |
| Replay judged picks by "rose / fell" | correct picks in a falling week showed ✗ | judged against the middle stock |
| "RIGHT" when top beat bottom by 0.02% | a tie looked like a win | gaps under 0.25% are ties; record shows right / wrong / tie |
| **17 trading days missing (15 Oct–6 Nov 2026)** | the 9 Nov ledger row absorbed four weeks (Nifty "+10.7%"); late-2026 features and outcomes spanned the hole | days re-fetched; nightly run now fills holes over 400 days and refuses to settle if a trading day is missing (`missing_sessions`) |
| Long-only Book A | threw away the half of the edge that worked (bottom 5 fell 0.45%/day) | short and long-short books |

Everything computed before the data-hole fix that touches Oct–Nov 2026 is being rebuilt
(feature caches, searches' 2024+ numbers, portfolio, replay, zoo).

---

## 5. How to run it day to day

- **Every weekday evening** the laptop runs `code\research\run_daily.cmd` (18:45, retry 21:00):
  fetch NSE data → check for holes → settle yesterday → refit → predict → plan → push to Railway.
- **During market hours** `/daily` checks each call against live prices every 30 s and crosses out
  calls going wrong. Set the market data provider to Yahoo (delayed) or your broker in Settings;
  the default is simulated prices.
- **Railway**: add `DAILY_SYNC_TOKEN` (copy it from the laptop's `.env`) in Railway → Variables, and
  set `DAILY_SYNC_URL=https://<your-app>/api/daily/sync` in the laptop's `.env`.
- Decision support only: nothing places orders.

---

## 6. Open: the "seasoned trader" layer (request 16)

What the repo already has to build it from: news (`code/pipeline/news_rss.py`), chart patterns
(Minervini trend template, VCP, breakouts), quant models (search winners, Model Zoo), regime
(HMM states, market breadth grade), and an LLM evaluator for memos. Proposed design:

1. **Regime playbook** ("know when to use what"): track every model's and book's rank IC by
   regime (trend up / choppy / down, high / low volatility). Each evening pick the weights from
   that table, chosen on 2021–23 and checked on 2024+ like everything else.
2. **Setup layer**: chart patterns become inputs and filters (e.g. only buy top-ranked stocks in a
   Stage 2 trend; only short bottom-ranked stocks below their 200-day average).
3. **News layer**: headline sentiment and event flags (results date, block deals) as inputs, plus
   an event veto (no new position into results day).
4. **Trader's note**: an evening memo per pick — why this model today, which pattern, what news,
   where the stop is — written from the numbers, never inventing them.
5. Same gate: each layer only stays if it raises rank IC / Sharpe on 2024+ and then live.
