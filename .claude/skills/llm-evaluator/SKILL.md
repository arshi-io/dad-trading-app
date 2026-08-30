---
name: llm-evaluator
description: Load before writing any code that calls the Anthropic API — evaluator memos, morning briefing, diary digest. Contracts, models, caching, cost rules.
---

# LLM evaluator

## Models
- Memos + morning briefing: claude-sonnet-4-6
- Headline↔stock matching, Hinglish diary normalization, digest draft: claude-haiku-4-5-20251001
- max_tokens 1000, temperature default. API key from env ANTHROPIC_API_KEY (never in code/repo).

## Memo contract (Stock Room verdict)
Input bundle (build in evaluator/memo.py): signal dict (template checklist, z-scores, pair status, GARCH/HMM state), 5 matched headlines (title, source, ts), up to 3 relevant diary observations.
Output: strict JSON only, no prose outside it:
`{"score": 0-100, "verdict": "TRADE_VALID"|"WAIT"|"AVOID", "reasoning": "120-180 words, plain trader English", "risk_flags": ["..."]}`
Rules baked into the system prompt:
- score < 60 ⇒ verdict must be AVOID; render as AVOID regardless.
- Every claim in reasoning must reference a provided input (name the signal or headline). No invented facts, no price targets, no hedging filler.
- Upcoming binary events (results, RBI, expiry) found in headlines ⇒ must appear in risk_flags.
Parse with json.loads after stripping fences; on parse failure retry once, then render "Analysis unavailable" card — never a broken card.

## Cost control
- On-demand memos cached 24h in SQLite keyed (ticker, snapshot_date).
- Nightly batch: watchlist + top 10 new screen entrants only.
- Briefing = 1 call/night; digest = 1 call/week. Target spend ≤ ₹600/mo.
