---
name: repo-conventions
description: Load before writing or editing any Python in this repo. Folder layout, Signal contract, snapshot format, and extension rules for the Papa Terminal codebase.
---

# Repo conventions

## Layout (extend the EXISTING research repo, never restructure)
```
code/
  signals/            base.py, mean_rev.py, pairs_trading.py, garch_volatility.py
                      (EXISTING + working — port, don't rewrite)
                      momentum.py EXISTS but is RETIRED (negative result in paper;
                      never import it in Papa code)
    minervini/        trend_template.py, rs_rank.py, vcp.py (NEW)
  regime/             market_regime.py (NEW — composes EXISTING garch_volatility
                      states + NIFTY 200DMA slope + breadth %>200DMA →
                      TRENDING_UP/CHOPPY/TRENDING_DOWN. No HMM in v1.)
  pipeline/           nightly_pipeline.py, news_rss.py (NEW; reuse
                      data/fetchers/equity.py + cleaners/ohlcv.py, but replace
                      hardcoded 2020–2024 windows with rolling current-date)
  evaluator/          memo.py (NEW in existing empty pkg — see llm-evaluator skill)
  app/                main.py (FastAPI), templates/, static/ (NEW)
  dashboard/          EXISTING research HTML — leave untouched, Papa doesn't use it
  data/               prices/*.parquet, snapshots/*.json, papa.db (SQLite)
  tests/              EXISTING empty pkg — fill: one test file per new module
```

## Contracts
- Every engine emits the EXISTING dataclass in `code/signals/base.py`:
  `Signal(asset, direction: "BUY"|"SELL"|"HOLD", confidence: 0-1, strategy, timeframe, timestamp, metadata: dict)`. Do NOT redefine it. Dashboard renders Signals; it never imports engine internals.
- Nightly output = `data/snapshots/{date}.json` with keys: `regime`, `indices`, `briefing`, `action_queue`, `screen`, `pairs`, `meta.generated_at`. App serves latest snapshot; missing/old → amber banner.
- Hedge ratios: rolling-window OLS (existing `_ols_slope`), re-estimated monthly in the pipeline, stored in snapshot metadata. Never a hardcoded constant. Kalman = v1.1, not now.
- Update the SKILL.md state block at session end (it's currently stale — garch is built).
- All timestamps IST, ISO format.

## Style
- Type hints everywhere. No classes where a function does. No new deps without asking.
- Errors in pipeline: log + continue per-stock; whole-pipeline failure writes `meta.status="failed"` and yesterday's snapshot stays live.
- Tests: golden-file tests for trend_template (5 known Stage 2 dates) and pairs z-score (values from research paper backtest).
