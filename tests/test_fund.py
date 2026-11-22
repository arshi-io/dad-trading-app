import json

import pandas as pd
import pytest

from code.research import daily as dl
from code.research import fund

BOARDS = {1: [{"key": "hgb6 | beat | base | raw"}, {"key": "hgb3 | beat | base | raw"}, {"key": "linear | beat | base | raw"}],
          5: [{"key": "hgb3 | rankret | base | raw"}, {"key": "linear | up | base | raw"}],
          20: [{"key": "hgb3 | rankret | base+quant | raw"}, {"key": "hgb6 | up | base | raw"}]}


@pytest.fixture
def conn(tmp_path):
    return dl.connect(tmp_path / "f.db")


def _day(conn, day, rets):
    for book, r in rets.items():
        conn.execute("INSERT OR REPLACE INTO daily_ledger VALUES (?,?,?,?,?,?,?,?)", (day, book, 10000, 10000 * (1 + r), 10000 * r, 0, 0, 0))


def test_fund_starts_live_only_after_the_agent_is_born(conn):
    fund.ensure_agent(conn, "2026-11-23", BOARDS)
    _day(conn, "2026-11-23", {"ALS": 0.01, "S": 0.01, "B": 0.01})
    assert fund.settle(conn, "2026-11-23", 0) is None               # born that evening: trades from the next session
    _day(conn, "2026-11-25", {"ALS": 0.03, "S": 0.0, "B": 0.0})
    assert fund.settle(conn, "2026-11-25", 1) is None               # replayed days never count
    r = fund.settle(conn, "2026-11-25", 0)
    assert r["ret"] == pytest.approx(0.01, rel=1e-3) and r["value"] == pytest.approx(10100, rel=1e-4)   # thirds: 0.03 / 3


def test_penalties_streak_death_and_successor(conn):
    fund.ensure_agent(conn, "2026-11-01", BOARDS)
    days = pd.bdate_range("2026-11-02", periods=5).strftime("%Y-%m-%d")
    _day(conn, days[0], {"ALS": 0.02, "S": 0.0, "B": 0.0})          # win: no penalty
    for d in days[1:4]:
        _day(conn, d, {"ALS": -0.03, "S": 0.0, "B": 0.0})            # 3 losing days in a row, all from ALS
    for d in days[:4]:
        out = fund.settle(conn, d, 0, BOARDS)
    a = conn.execute("SELECT * FROM fund_agents ORDER BY id").fetchall()
    assert a[0]["died"] == days[3] and "3 losing days" in a[0]["cause"]
    pm = json.loads(a[0]["postmortem"])
    assert pm["worst_sleeve"] == "ALS"
    new = json.loads(a[1]["config"])
    assert fund.family(new["1"]) != fund.family(json.loads(a[0]["config"])["1"]) or new["1"] != json.loads(a[0]["config"])["1"]
    assert new["1"] == "linear | beat | base | raw"                 # untried and a different family for the failed sleeve
    assert a[1]["start_value"] == pytest.approx(a[0]["end_value"])  # the money carries over
    assert "died" in out["event"]


def test_a_winning_day_removes_a_penalty_never_below_zero(conn):
    fund.ensure_agent(conn, "2026-11-01", BOARDS)
    seq = [-0.01, 0.01, 0.01, -0.01]
    days = pd.bdate_range("2026-11-02", periods=4).strftime("%Y-%m-%d")
    pens = []
    for d, r in zip(days, seq):
        _day(conn, d, {"ALS": r, "S": r, "B": r})
        pens.append(fund.settle(conn, d, 0, BOARDS)["penalties"])
    assert pens == [1, 0, 0, 1]


def test_weights_move_toward_better_sleeves_within_floor_and_cap(conn):
    for i, d in enumerate(pd.bdate_range("2026-10-01", periods=20).strftime("%Y-%m-%d")):
        _day(conn, d, {"ALS": 0.01 + 0.001 * (i % 2), "S": -0.01 + 0.001 * (i % 2), "B": 0.002 + 0.002 * (i % 3)})
    w = fund.reweight(conn, "2026-12-31")
    assert w["S"] == pytest.approx(fund.FLOOR, abs=1e-3) and w["ALS"] <= fund.CAP + 1e-9
    assert sum(w.values()) == pytest.approx(1.0) and w["ALS"] > w["S"]
