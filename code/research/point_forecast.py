"""One price per stock for the next 5 sessions, from the 5-session search winner, with its record.

The winner ranks every liquid stock each evening. Each tenth of that ranking has an average
5-session move (next open to the close 5 sessions later); a stock's point forecast is its last
close times (1 + its tenth's average move), plus a BEAT / LAG call against the middle stock.
Track record (2024+): moves learned on 2021-23 only, applied to 2024+, so nothing in the record
was used to make it. Today's forecast uses every settled week.

Run:  python -m code.research.point_forecast     (writes code/data/point_forecast.json)
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from code.research import model_search as ms

OUT = Path(__file__).resolve().parents[1] / "data" / "point_forecast.json"
HORIZON = 5
TRAIN = ("2021-01-01", "2024-01-01")
TEST_FROM = "2024-01-01"


def tenth(rank: pd.Series) -> pd.Series:
    return np.minimum((rank * 10).astype(int), 9)


def decile_moves(df: pd.DataFrame) -> List[float]:
    m = df.groupby(tenth(df["best"]))["fwd"].mean().reindex(range(10))
    return [round(float(v), 5) for v in m.fillna(0.0)]


def record(g: pd.DataFrame) -> Dict[str, Any]:
    """g: rows with pred (expected move), fwd (actual), med (that day's middle stock), best (rank)."""
    called = g[g["pred"] != 0]
    beat_call = g["best"] >= 0.5
    return {
        "n": int(len(g)),
        "direction_hit": round(float((np.sign(called["pred"]) == np.sign(called["fwd"])).mean()), 4) if len(called) else None,
        "beat_hit": round(float((beat_call == (g["fwd"] > g["med"])).mean()), 4),
        "mae_point": round(float((g["fwd"] - g["pred"]).abs().mean()), 5),
        "mae_flat": round(float(g["fwd"].abs().mean()), 5),           # 'no change' as the forecast
        "top_tenth_avg": round(float(g.loc[tenth(g["best"]) == 9, "fwd"].mean()), 5) if (tenth(g["best"]) == 9).any() else None,
        "top_tenth_n": int((tenth(g["best"]) == 9).sum()),
    }


def build(rp: pd.DataFrame) -> Dict[str, Any]:
    from code.research.setups import is_equity
    rp = rp[rp["best"].notna() & rp["symbol"].map(is_equity)].copy()
    rp["med"] = rp.groupby("date")["fwd"].transform("median")
    done = rp[rp["fwd"].notna()]
    train = done[(done["date"] >= TRAIN[0]) & (done["date"] < TRAIN[1])]
    test = done[done["date"] >= TEST_FROM].copy()
    learned = decile_moves(train)
    test["pred"] = tenth(test["best"]).map(dict(enumerate(learned)))
    now_moves = decile_moves(done)
    latest = rp["date"].max()
    today = rp[rp["date"] == latest].set_index("symbol")
    stocks = {}
    for sym, g in test.groupby("symbol"):
        if len(g) < 60:
            continue
        stocks[sym] = {"record": record(g)}
    for sym, r in today.iterrows():
        t = int(min(int(r["best"] * 10), 9))
        stocks.setdefault(sym, {})
        stocks[sym].update(rank=round(float(r["best"]), 4), tenth=t + 1, exp_move=now_moves[t],
                           call="BEAT" if r["best"] >= 0.5 else "LAG")
    return {"generated": date.today().isoformat(), "asof": latest.strftime("%Y-%m-%d"), "horizon": HORIZON,
            "moves_by_tenth": now_moves, "learned_2021_23": learned, "record_from": TEST_FROM,
            "overall": record(test), "stocks": stocks}


if __name__ == "__main__":
    d = build(pd.read_parquet(ms.REPLAY, columns=["date", "symbol", "fwd", "best"]))
    OUT.write_text(json.dumps(d), encoding="utf-8")
    o = d["overall"]
    print(d["asof"], "moves by tenth:", [f"{m:+.2%}" for m in d["moves_by_tenth"]])
    print(f"2024+: direction right {o['direction_hit']:.1%}, beat/lag right {o['beat_hit']:.1%}, "
          f"avg error {o['mae_point']:.2%} vs no-change {o['mae_flat']:.2%}, top tenth avg {o['top_tenth_avg']:+.2%}")
