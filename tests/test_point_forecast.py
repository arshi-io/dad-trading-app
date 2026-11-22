import numpy as np
import pandas as pd
import pytest

from code.research import point_forecast as pf


def _replay(edge_from_2024: float):
    rng = np.random.default_rng(0)
    days = pd.bdate_range("2021-01-01", "2024-12-31")
    rows = []
    for d in days:
        r = rng.uniform(0, 1, 50)
        edge = 0.02 if d < pd.Timestamp("2024-01-01") else edge_from_2024
        fwd = (r - 0.5) * edge + rng.normal(0, 0.01, 50)
        rows += [(d, f"S{i:02d}", f, k) for i, (f, k) in enumerate(zip(fwd, r))]
    df = pd.DataFrame(rows, columns=["date", "symbol", "fwd", "best"])
    df.loc[df["date"] == days[-1], "fwd"] = np.nan                  # today: not settled yet
    return df


def test_record_uses_moves_learned_before_2024_only():
    d = pf.build(_replay(edge_from_2024=0.0))                         # the edge disappears in 2024
    assert d["learned_2021_23"][9] > d["learned_2021_23"][0]         # learned on 2021-23
    assert d["overall"]["beat_hit"] == pytest.approx(0.5, abs=0.03)  # so 2024+ shows no skill: honest
    assert d["asof"] == "2024-12-31"


def test_todays_call_and_price_move_follow_the_rank():
    d = pf.build(_replay(edge_from_2024=0.02))
    s = d["stocks"]
    top = max(s, key=lambda k: s[k]["rank"])
    bot = min(s, key=lambda k: s[k]["rank"])
    assert s[top]["call"] == "BEAT" and s[bot]["call"] == "LAG" and s[top]["exp_move"] > s[bot]["exp_move"]
    assert d["overall"]["beat_hit"] > 0.6
