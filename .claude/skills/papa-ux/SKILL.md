---
name: papa-ux
description: Load before building or editing any template, page, or visual element. Design system for a 55+ non-technical trader. Overrides generic styling instincts.
---

# Papa UX — design system

User: seasoned trader, reads Minervini, hates computers. Dumb down the TECH, never the trading language.

## Non-negotiables
- Base font 18px, headings 24–32px, line-height 1.6. Dark theme: bg #0E1117, surface #171B23, text #E8EAED.
- Status = word + color, never color alone: PASS/FAIL/WAIT, ENTRY ZONE/EXIT. Green #22C55E, red #EF4444, amber #F59E0B, blue #3B82F6 (links/buttons only).
- Buttons are words, not icons. No hamburgers, no settings pages, no modals unless unavoidable.
- Header always shows "Data as of {dt} IST". Snapshot >24h old → amber banner "Data update pending — showing yesterday". Never an error page.
- Every technical term gets `<span class="term" title="...">`: one-line trader-language tooltip (e.g., z-score = "how stretched the rubber band is; ±2 = fully stretched").
- Empty states say something ("No new signals — nothing to do is also a decision"), never blank.

## Screens (5 tabs, fixed)
1. **Today**: regime banner card → 3 index tiles → Morning Briefing card (broker-note style, 6 bullets) → Action Queue (signal cards, big "Open Stock Room →" button).
2. **Minervini Screen**: table sorted by RS rank; row click expands 8-point checklist, each line ✅/❌ + actual value ("31% above 52wk low ✅ need ≥30%"). VCP Watch section below with pivot + trigger price, labeled "algorithm-detected — verify on chart".
3. **Spread Trades**: per pair, horizontal gauge −3σ…+3σ with needle (Plotly indicator), green zones beyond ±2σ; spread chart with bands; hedge ratio translated to share quantities; status word; footer: last cointegration re-test date + "hard exit beyond ±3.5σ".
4. **Stock Room**: TradingView Lightweight Chart (candles + 50/150/200 DMA + volume) top half; Verdict card (score, memo, risk flags, "below 60 = no trade" always printed); signal accordion; 5 headlines with source + time.
5. **Diary**: one big textbox + optional fields (stock, side, price, reason dropdown + free text). Hinglish fine. Observations quick-capture box. Sunday digest card.

## Reference
Stitch mockups in docs/mockups/ are the visual source of truth. Match them; don't freestyle.
