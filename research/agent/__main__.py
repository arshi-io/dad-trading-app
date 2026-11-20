"""python -m research.agent session RELIANCE TCS   |   cloud-review   |   report"""
import argparse
import logging

from research.agent import journal
from research.agent.session import cloud_review, run_session
from research.agent.team import load_config

p = argparse.ArgumentParser(prog="python -m research.agent")
sub = p.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("session", help="run one batched research session")
s.add_argument("symbols", nargs="+")
s.add_argument("--cloud-review", action="store_true", help="hold critic-passed ideas for the weekly cloud review")
sub.add_parser("cloud-review", help="send the top waiting ideas to the cloud critic, gate the passes")
sub.add_parser("report", help="gate pass rate by model for each role")
a = p.parse_args()
logging.basicConfig(level=logging.INFO, format="%(message)s")

cfg = load_config()
if a.cmd == "session":
    cfg["cloud_review"] = cfg.get("cloud_review") or a.cloud_review
    out = run_session(a.symbols, cfg)
    print(f"session {out['session']}: {out['ideas']} ideas, {out['candidates']} strategies, "
          f"{out['tested']} backtests, {out['critic_passed']} sent to the gate")
    for r in out["rows"]:
        print(f"  #{r['id']} {r['symbol']} {r['family']}: {r['status']}"
              + (f" gate {'PASS' if r['gate_passed'] else 'fail'}" if r["status"] == "gated" else ""))
elif a.cmd == "cloud-review":
    for r in cloud_review(cfg):
        print(f"  #{r['id']} {r['symbol']} {r['family']}: cloud {r['cloud_verdict']}, {r['status']}")
else:
    conn = journal.connect()
    for role in ("librarian", "idea_maker", "coder", "critic", "cloud_critic"):
        print(role)
        for r in journal.pass_rate_by_model(conn, role):
            if r["model"]:
                print(f"  {r['model']}: {r['ideas']} ideas, {r['critic_passed'] or 0} critic-passed, "
                      f"{r['gate_passed'] or 0} gate-passed")
