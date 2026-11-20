"""The critic's automatic checks (PRD F5). The critic model may add a failure, never remove one."""
from __future__ import annotations

from typing import Any, Dict, List

MIN_TRADES = 30
MIN_COST = 0.0025


def code_checks(bt: Dict[str, Any], attempt_no: int, max_attempts: int) -> List[str]:
    failed = []
    if bt.get("lookahead"):
        failed.append("lookahead bias: positions change when later data is removed")
    if bt["B"]["n_trades"] < MIN_TRADES:
        failed.append(f"too few trades: {bt['B']['n_trades']} on Period B (need {MIN_TRADES})")
    if attempt_no > max_attempts:
        failed.append(f"too many attempts on this idea: {attempt_no} (limit {max_attempts})")
    if bt.get("cost", 0) < MIN_COST:
        failed.append("missing costs: backtest charged less than 0.25% per round trip")
    if bt["B"]["expectancy"] <= 0:
        failed.append("no edge: expectancy on Period B is not above 0 after costs")
    # One lucky period: over A and B together, take away the single best year. If what is left
    # is not positive, the whole result is that one year.
    years = list(bt.get("years", {}).values())
    if years and sum(years) > 0 and sum(years) - max(years) <= 0:
        best = max(bt["years"], key=bt["years"].get)
        failed.append(f"depends on one lucky period: without {best} the total is not positive")
    return failed


def brief(row: Dict[str, Any], failed: List[str]) -> str:
    """What the critic model is shown for one idea."""
    bt = row["backtest"]
    fmt = lambda s: (f"trades {s['n_trades']}, expectancy {s['expectancy']:.4f}, win rate {s['win_rate']:.0%}, "
                     f"avg win {s['avg_win']:.4f}, avg loss {s['avg_loss']:.4f}, max drawdown {s['max_drawdown']:.1%}, "
                     f"total {s['total_return']:.1%}")
    years = ", ".join(f"{y}: {r:+.1%}" for y, r in bt["years"].items())
    return (f"Stock: {row['symbol']}\nIdea: {row['idea']} (source: {row['source']})\n"
            f"Hypothesis: {row['hypothesis']}\nRule: {row['rule']}\n"
            f"Attempts on this idea family: {row['attempt_no']}\n\nCode:\n```python\n{row['code']}```\n\n"
            f"Period A (search): {fmt(bt['A'])}\nPeriod B (selection): {fmt(bt['B'])}\n"
            f"Return by year: {years}\nCosts charged: {bt['cost']:.2%} per round trip\n\n"
            f"Automatic checks failed: {failed or 'none'}")
