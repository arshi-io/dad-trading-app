"""One research session (PRD F7), batched by role so each local model loads once:
librarian prepares N ideas -> idea_maker writes N hypotheses -> coder builds N strategies (+ ensembles)
-> backtests on A/B with no model loaded -> critic reviews every result -> gate.
Each model is unloaded after its batch. Only critic-passed ideas reach the gatekeeper.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import gatekeeper
from research.agent import critic, journal, lab
from research.agent.team import chat, load_config, parse_code, parse_json, unload

log = logging.getLogger(__name__)
LIBRARY = Path(__file__).resolve().parents[1] / "library"


def _ask_json(llm: Callable, cfg: Dict, role: Dict, system: str, user: str) -> Any:
    reply = llm(cfg, role, system, user)
    try:
        return parse_json(reply)
    except ValueError:
        return parse_json(llm(cfg, role, system, user + "\n\nYour last reply was not valid JSON. Reply with JSON only."))


def _library(folder: Path, per_file: int = 3000, total: int = 20000) -> str:
    # ponytail: first part of each .md/.txt note; add a search index when the library outgrows this.
    parts = [f"--- {p.name} ---\n{p.read_text(encoding='utf-8', errors='ignore')[:per_file]}"
             for p in sorted(folder.glob("*")) if p.suffix in (".md", ".txt")] if folder.exists() else []
    return "\n\n".join(parts)[:total] or "(library is empty)"


def run_session(symbols: List[str], cfg: Optional[Dict] = None, conn=None, llm: Callable = chat,
                free: Callable = unload, submit: Callable = gatekeeper.submit,
                test: Callable = lab.backtest, smoke: Callable = lab.smoke,
                library: Path = LIBRARY) -> Dict[str, Any]:
    cfg = cfg or load_config()
    conn = conn or journal.connect()
    R = cfg["roles"]
    n = cfg["batch_size"]
    session = datetime.now().strftime("%Y%m%d-%H%M%S")
    models = {r: {"model": R[r]["model"], "provider": R[r]["provider"]} for r in ("librarian", "idea_maker", "coder", "critic")}

    # 1. Librarian: N ideas with sources.
    ideas = _ask_json(llm, cfg, R["librarian"], R["librarian"]["system"],
                      f"Library notes:\n{_library(library)}\n\nJournal of past attempts:\n{journal.summary(conn)}\n\n"
                      f"Give exactly {n} ideas.")[:n]
    free(cfg, R["librarian"])

    # 2. Idea maker: one hypothesis per idea.
    hyps = []
    for it in ideas:
        try:
            h = _ask_json(llm, cfg, R["idea_maker"], R["idea_maker"]["system"],
                          f"Idea: {it['idea']}\nSource: {it.get('source', '')}")
            if h["family"] == "not_testable":
                log.info("idea_maker: not testable with daily price data: %s", it["idea"])
                continue
            hyps.append({**it, **{k: str(h[k]) for k in ("family", "hypothesis", "rule")}})
        except (ValueError, KeyError, TypeError) as e:
            log.warning("idea_maker skipped an idea: %s", e)
    free(cfg, R["idea_maker"])

    # 3. Coder: one strategy per hypothesis, then optional ensembles of already-tested strategies.
    candidates = []
    for h in hyps:
        ask = f"Hypothesis: {h['hypothesis']}\nRule: {h['rule']}"
        try:
            code = parse_code(llm(cfg, R["coder"], R["coder"]["system"], ask))
            try:
                smoke(code, symbols[0])
            except Exception as e:   # one repair attempt with the error; a second failure is logged in step 4
                code = parse_code(llm(cfg, R["coder"], R["coder"]["system"],
                                      f"{ask}\n\nYour code:\n```python\n{code}```\nfailed with: {e}\nFix it."))
            candidates.append({**h, "kind": "single", "code": code})
        except ValueError as e:
            log.warning("coder skipped %s: %s", h["family"], e)
    tested = {r["id"]: r for r in journal.rows(conn, "backtest IS NOT NULL AND status != 'error'")}
    if tested and cfg.get("max_ensembles", 0) and "ensemble_system" in R["coder"]:
        best = sorted(tested.values(), key=lambda r: -r["backtest"]["B"]["expectancy"])[:15]
        menu = "\n".join(f"#{r['id']} {r['symbol']} [{r['family']}] {r['hypothesis']} | B: {r['backtest']['B']['n_trades']} trades, "
                         f"expectancy {r['backtest']['B']['expectancy']:.4f}" for r in best)
        try:
            specs = _ask_json(llm, cfg, R["coder"], R["coder"]["ensemble_system"], f"Tested strategies:\n{menu}")
            for spec in (specs if isinstance(specs, list) else [])[: cfg["max_ensembles"]]:
                try:
                    candidates.append(lab.build_ensemble(spec, tested))
                except (ValueError, KeyError, TypeError) as e:
                    log.warning("ensemble rejected: %s", e)
        except ValueError as e:
            log.warning("coder ensemble reply unreadable: %s", e)
    free(cfg, R["coder"])

    # 4. Backtests on Periods A and B (pure Python, no model in memory).
    built = []
    for c in candidates:
        for sym in symbols:
            row = {"session": session, "symbol": sym, "models": models, "status": "built",
                   **{k: c.get(k) for k in ("family", "kind", "parents", "idea", "source", "hypothesis", "rule", "code")}}
            try:
                row["backtest"] = test(c["code"], sym)
            except Exception as e:   # a broken strategy is a logged result, not a crashed session
                row.update(status="error", checks=[f"backtest error: {e}"])
            rid = journal.add(conn, row)
            journal.update(conn, rid, attempt_no=journal.attempts(conn, c["family"]))
            if row["status"] == "built":
                built.append(rid)

    # 5. Critic: automatic checks first; the model only sees ideas that survive them.
    passed = []
    for rid in built:
        row = journal.rows(conn, "id=?", (rid,))[0]
        failed = critic.code_checks(row["backtest"], row["attempt_no"], cfg["max_attempts"])
        verdict, reasons = "fail", failed
        if not failed:
            try:
                v = _ask_json(llm, cfg, R["critic"], R["critic"]["system"], critic.brief(row, failed))
                verdict, reasons = ("pass" if str(v.get("verdict")).lower() == "pass" else "fail"), list(v.get("reasons", []))
            except (ValueError, AttributeError) as e:
                reasons = [f"critic reply unreadable: {e}"]
        status = "critic_failed" if verdict == "fail" else ("awaiting_cloud" if cfg.get("cloud_review") else "built")
        journal.update(conn, rid, checks=failed, critic_verdict=verdict, critic_reasons=reasons, status=status)
        if verdict == "pass" and not cfg.get("cloud_review"):
            passed.append(rid)
    free(cfg, R["critic"])

    # 6. Gate: only critic-passed ideas.
    for rid in passed:
        _gate(conn, rid, submit)
    return {"session": session, "ideas": len(ideas), "candidates": len(candidates), "tested": len(built),
            "critic_passed": len(passed), "rows": journal.rows(conn, "session=?", (session,))}


def _gate(conn, rid: int, submit: Callable) -> None:
    row = journal.rows(conn, "id=?", (rid,))[0]
    res = submit(row["code"], row["symbol"], f"journal #{rid} {row['family']}", row["attempt_no"])
    journal.update(conn, rid, gate=res, gate_passed=int(bool(res.get("passed"))), status="gated")


def cloud_review(cfg: Optional[Dict] = None, conn=None, llm: Callable = chat,
                 submit: Callable = gatekeeper.submit) -> List[Dict[str, Any]]:
    """Weekly second critic: the top ideas waiting for review go to the cloud model; passes go to the gate."""
    cfg = cfg or load_config()
    conn = conn or journal.connect()
    role = cfg["roles"]["cloud_critic"]
    waiting = sorted(journal.rows(conn, "status='awaiting_cloud'"), key=lambda r: -r["backtest"]["B"]["expectancy"])
    for row in waiting[: cfg.get("cloud_review_top", 5)]:
        try:
            v = _ask_json(llm, cfg, role, role["system"], critic.brief(row, row["checks"] or []))
            verdict, reasons = ("pass" if str(v.get("verdict")).lower() == "pass" else "fail"), list(v.get("reasons", []))
        except (ValueError, AttributeError) as e:
            verdict, reasons = "fail", [f"cloud reply unreadable: {e}"]
        models = {**row["models"], "cloud_critic": {"model": role["model"], "provider": role["provider"]}}
        journal.update(conn, row["id"], cloud_verdict=verdict, cloud_reasons=reasons, models=models,
                       status="cloud_failed" if verdict == "fail" else row["status"])
        if verdict == "pass":
            _gate(conn, row["id"], submit)
    return journal.rows(conn, "cloud_verdict IS NOT NULL")
