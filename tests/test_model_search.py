import numpy as np
import pandas as pd
import pytest

from code.research import model_search as ms
from code.research import updown as u
from tests.test_updown import _raw

VALID = ("2014-07-01", "2015-01-01")
LOCKED = "2015-01-01"


@pytest.fixture(scope="module")
def edge():
    return ms.prepare(u.features(u.panel(_raw(0.5, n_sym=30))))


@pytest.fixture(scope="module")
def noise():
    return ms.prepare(u.features(u.panel(_raw(0.0, n_sym=30))))


def test_each_round_changes_exactly_one_thing():
    for n in ms.neighbours(ms.BASELINE):
        assert sum(n[k] != ms.BASELINE[k] for k in ms.SPACE) == 1
    assert len(ms.neighbours(ms.BASELINE)) == sum(len(v) - 1 for v in ms.SPACE.values())


def test_walk_only_trains_on_outcomes_settled_before_each_quarter(edge):
    seen = []

    class Spy:
        def __init__(self, spec):
            pass

        def fit(self, X, y):
            seen.append(edge.loc[X.index, "label_date"].max())
            return self

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

    pred = ms.walk(edge, ms.BASELINE, *VALID, make=Spy)
    quarters = sorted(pred["date"].dt.to_period("Q").unique())
    assert len(seen) == len(quarters) == 2
    for latest, q in zip(seen, quarters):
        assert latest < q.start_time


def test_ranked_inputs_use_only_that_days_stocks(edge):
    day = edge[edge["date"] == edge["date"].iloc[-1]]
    assert day["x_r5"].to_numpy() == pytest.approx(day["r5"].rank(pct=True).to_numpy())
    assert "x_breadth" not in edge.columns


def test_search_finds_a_planted_edge_and_keeps_the_best(edge):
    res = ms.search(edge, valid=VALID, max_tries=12, train_rows=50_000, log=lambda s: None)
    best, base = res["best"], res["baseline"]
    assert best["ic"] > 0.05 and best["ic"] >= base["ic"]
    assert [r["ic"] for r in res["tried"]] == sorted((r["ic"] for r in res["tried"]), reverse=True)
    assert any(r["key"].startswith("blend(") for r in res["tried"])
    locked, pb = ms.locked_test(edge, best, locked=LOCKED, train_rows=50_000)
    assert pb["date"].min() >= pd.Timestamp(LOCKED)
    assert locked["best"]["ic"] > 0.05


def test_search_stops_when_a_round_gains_nothing(noise):
    res = ms.search(noise, valid=VALID, max_tries=30, train_rows=50_000, min_gain=0.5, log=lambda s: None)
    singles = [r for r in res["tried"] if not r["key"].startswith("blend(")]
    assert len(singles) == 1 + len(ms.neighbours(ms.BASELINE))      # one round, then stop
    assert res["best"]["key"] == ms.key(ms.BASELINE)


def test_replay_day_reports_what_the_ranking_did_next():
    rng = np.random.default_rng(1)
    days = pd.bdate_range("2020-01-01", periods=20)
    df = pd.DataFrame([(d, f"S{i:02d}", r) for d in days for i, r in enumerate(np.linspace(0.02, 1, 50))],
                      columns=["date", "symbol", "current"])
    df["fwd"] = (df["current"] - 0.5) * 0.1 + rng.normal(0, 0.001, len(df))     # top ranks really rose
    weeks = ms.replay_weeks(df, "current")
    out = ms.replay_day(df, weeks, "current", "2020-01-12")                         # a Sunday: snaps back to Friday
    assert out["date"] == "2020-01-10" and out["right"] and out["top"] > out["all"] > out["bottom"]
    assert out["best_picks"][0]["symbol"] == "S49" and out["worst_picks"][0]["symbol"] == "S00"
    assert out["record"] == {**out["record"], "weeks": 20, "right": 20, "upto_weeks": 8}
    top = out["best_picks"][0]
    assert top["vs_median"] == pytest.approx(top["move"] - out["median"], abs=2e-4)     # judged against the middle stock


def test_etfs_without_markers_are_not_stocks():
    from code.research.setups import is_equity
    assert not is_equity("HDFCSML250") and not is_equity("MON100") and is_equity("SMLMAH")


def test_blend_averages_same_day_ranks():
    d = pd.to_datetime(["2020-01-01"] * 3)
    a = pd.DataFrame({"date": d, "symbol": list("xyz"), "fwd": [0.0, 0, 0], "score": [1.0, 2, 3]})
    b = a.assign(score=[300.0, 200, 100])
    assert ms.blend([a, b])["score"].tolist() == pytest.approx([2 / 3, 2 / 3, 2 / 3])
