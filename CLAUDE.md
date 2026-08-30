# Papa Terminal — EOD trading dashboard for one non-technical user

## Session protocol (always)
- One phase per session. Spec: docs/phases/P{n}.md — read ONLY the current phase file.
- Load skills on demand: repo-conventions (before writing code), papa-ux (before any UI), llm-evaluator (before any Anthropic API code).
- Never read a whole file to edit part of it: grep first, view ranges.
- Terse output. No summaries of what you just did unless asked. No plans longer than 5 lines.
- Plan → confirm → build in one pass. Don't rebuild working code.
- If blocked twice on the same bug, stop and state 2 hypotheses; wait for pick.

## Hard rules
- Extend the existing repo structure; never restructure or rename existing modules.
- No auto-execution of trades. Decision support only.
- Dashboard reads pre-computed JSON snapshots only; all computation happens in nightly_pipeline.py.
- Stale data (>24h) must show amber "Data update pending" banner. User must NEVER see a stack trace.
- Python 3.11, FastAPI + Jinja + htmx. No React. Charts: TradingView Lightweight Charts + Plotly.
- Data: yfinance EOD (.NS), parquet cache. Diary/signals/memos: SQLite.

## Definition of done (per phase)
Exit test in the phase file passes + `python -m pytest tests/ -q` green + one manual check listed in phase file.
