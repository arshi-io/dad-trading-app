"""Historical breakout outcomes by market grade -> code/data/base_rates.json (read by the app).

Grade A: ≥60% of liquid stocks above their 200-day average AND NIFTY above its 50-day.
Grade B: one of the two.  Grade C: neither.
Outcomes are from code/research/setups.py events (decide at the breakout close, enter next open,
1.5×ATR stop, 0.25% costs): hit rate of 2R-before-stop within 20 sessions, and average R for both
the fixed 2R exit and the trailing exit (close below the 21-day average, up to 60 sessions).
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from code.research import bhavcopy

OUT = Path(__file__).resolve().parents[1] / "data" / "base_rates.json"
BREADTH_OK = 0.60


def grade(breadth: float, nifty_above_50: bool) -> str:
    good = int(breadth >= BREADTH_OK) + int(bool(nifty_above_50))
    return {2: "A", 1: "B", 0: "C"}[good]


def build() -> dict:
    ev = pd.read_parquet(bhavcopy.ROOT / "breakout_events.parquet").dropna(subset=["r", "r_trail", "breadth"])
    ev["grade"] = [grade(b, a == 1) for b, a in zip(ev.breadth, ev.mkt_above_50)]
    rows = {}
    for g, grp in [("ALL", ev)] + list(ev.groupby("grade")):
        rows[g] = {
            "n": int(len(grp)),
            "hit_2r": round(float(grp.win.mean()), 3),
            "avg_r_fixed": round(float(grp.r.mean()), 3),
            "avg_r_trail": round(float(grp.r_trail.mean()), 3),
            "share_of_time": round(len(grp) / len(ev), 3),
        }
    return {
        "generated": date.today().isoformat(),
        "period": f"{ev.date.min():%Y-%m-%d} to {ev.date.max():%Y-%m-%d}",
        "breadth_ok": BREADTH_OK,
        "method": "NSE daily bhavcopy 2011+, liquid stocks (≥₹5cr turnover, ≥₹30), pivot breakouts from "
                  "10-65 session bases ≤35% deep; decide at close, enter next open; 1.5×ATR stop; 0.25% costs.",
        "grades": rows,
    }


if __name__ == "__main__":
    data = build()
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(json.dumps(data["grades"], indent=1))
