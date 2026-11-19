"""Does the HMM regime say anything about breakouts that the A/B/C grade doesn't?

Labels every breakout in code/research/setups.py's event study with the HMM state on its decision
day (filtered probabilities, monthly expanding refits -- see code/regime/hmm_regime.py), then reports
hit rate and average R by state, and by state within each grade, the same way base_rates.py does
for grades. Writes code/data/hmm_base_rates.json.

Run: python -m code.research.hmm_base_rates   (~5 min: one HMM fit per month since 2010)
"""
from __future__ import annotations

import json
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from code.data.fetchers.equity import fetch_ohlcv
from code.regime.hmm_regime import LABELS, regime_probabilities
from code.research import bhavcopy
from code.research.base_rates import grade

OUT = Path(__file__).resolve().parents[1] / "data" / "hmm_base_rates.json"
STATES = bhavcopy.ROOT / "hmm_states.parquet"


def states() -> pd.DataFrame:
    if STATES.exists():
        return pd.read_parquet(STATES)
    nifty = fetch_ohlcv("^NSEI", "2007-09-17", date.today().isoformat())["close"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = regime_probabilities(nifty)
    df.to_parquet(STATES)
    return df


def summarise(grp: pd.DataFrame, total: int) -> dict:
    se = grp.r_trail.std(ddof=1) / np.sqrt(len(grp)) if len(grp) > 1 else float("nan")
    return {"n": int(len(grp)), "share": round(len(grp) / total, 3), "hit_2r": round(float(grp.win.mean()), 3),
            "avg_r_trail": round(float(grp.r_trail.mean()), 3), "se_r_trail": round(float(se), 3),
            "avg_r_fixed": round(float(grp.r.mean()), 3)}


def build() -> dict:
    ev = pd.read_parquet(bhavcopy.ROOT / "breakout_events.parquet").dropna(subset=["r", "r_trail", "breadth"])
    ev["grade"] = [grade(b, a == 1) for b, a in zip(ev.breadth, ev.mkt_above_50)]
    st = states()
    ev = ev.merge(st[["label"]], left_on="date", right_index=True, how="left").dropna(subset=["label"])
    by_state = {s: summarise(g, len(ev)) for s, g in ev.groupby("label")}
    by_grade = {g: summarise(x, len(ev)) for g, x in ev.groupby("grade")}
    cross = {f"{g}/{s}": summarise(x, len(ev)) for (g, s), x in ev.groupby(["grade", "label"]) if len(x) >= 30}
    days = st.dropna(subset=["label"])
    return {
        "generated": date.today().isoformat(),
        "period": f"{ev.date.min():%Y-%m-%d} to {ev.date.max():%Y-%m-%d}",
        "events_labelled": int(len(ev)),
        "share_of_days": {s: round(float((days.label == s).mean()), 3) for s in LABELS},
        "by_state": by_state, "by_grade": by_grade, "grade_x_state": cross,
        "note": "se_r_trail is a naive standard error; breakouts on the same day are correlated, so true "
                "uncertainty is larger. States: filtered probabilities, monthly expanding refits, no lookahead.",
    }


if __name__ == "__main__":
    data = build()
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(json.dumps({k: data[k] for k in ("period", "events_labelled", "share_of_days", "by_state", "by_grade", "grade_x_state")}, indent=1))
