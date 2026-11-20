import numpy as np
import pandas as pd
import pytest

from gatekeeper import gate

CYCLE = """
import numpy as np, pandas as pd
def positions(df):
    i = np.arange(len(df))
    return pd.Series(((i // 5) % 2 == 0).astype(int), index=df.index)   # long 5 days, flat 5
"""
PEEK = """
def positions(df):
    return (df["close"].shift(-1) > df["close"]).astype(int)            # uses tomorrow's close
"""
HOLD = """
import pandas as pd
def positions(df):
    return pd.Series(1, index=df.index)                                  # one trade only
"""


def _df(drift=0.002, seed=0):
    idx = pd.bdate_range("2011-01-03", "2026-06-30")
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.002, len(idx))))
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)


@pytest.fixture
def vault(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "DB_PATH", tmp_path / "attempts.db")
    monkeypatch.setattr(gate, "_load", lambda s: _df())
    return gate


def test_get_data_never_returns_period_c(vault):
    with pytest.raises(PermissionError):
        vault.get_data("X", "C")
    assert vault.get_data("X", "A").index.max() <= pd.Timestamp("2020-12-31")
    assert vault.get_data("X", "B").index.max() <= pd.Timestamp("2023-12-31")


def test_real_edge_passes(vault):
    r = vault.submit(CYCLE, "X", "cycle", idea_attempts=1)
    assert r["passed"], r["failed"]
    assert r["n_trades"] >= 30 and r["expectancy"] > 0 and r["period"].startswith("2024-01-01")
    for k in ("win_rate", "avg_win", "avg_loss", "max_drawdown", "idea_attempts"):
        assert k in r


def test_lookahead_is_rejected(vault):
    r = vault.submit(PEEK, "X", "peek", idea_attempts=1)
    assert not r["passed"] and any("lookahead" in f for f in r["failed"])


def test_one_vault_attempt_per_strategy(vault):
    vault.submit(CYCLE, "X", "cycle", 1)
    again = vault.submit(CYCLE, "X", "cycle again", 2)
    assert not again["passed"] and again["failed"] == ["already scored on Period C"]


def test_too_few_trades(vault):
    r = vault.submit(HOLD, "X", "hold", 1)
    assert not r["passed"] and any("trades" in f for f in r["failed"])


def test_crashing_strategy_burns_its_attempt(vault):
    bad = "def positions(df):\n    raise ValueError('boom')\n"
    assert "strategy error" in vault.submit(bad, "X", "bad", 1)["failed"][0]
    assert vault.submit(bad, "X", "bad", 1)["failed"] == ["already scored on Period C"]


def test_strategy_cannot_patch_the_gate(vault):
    evil = ("import sys\n"
            "def positions(df):\n"
            "    g = sys.modules.get('gatekeeper.gate')\n"
            "    if g: g.MIN_TRADES = 0\n"
            "    import pandas as pd\n"
            "    return pd.Series(1, index=df.index)\n")
    r = vault.submit(evil, "X", "evil", 1)
    assert gate.MIN_TRADES == 30 and any("trades" in f for f in r["failed"])


def test_costs_are_charged_per_round_trip():
    idx = pd.bdate_range("2024-01-01", periods=4)
    close = pd.Series([100.0, 101.0, 102.01, 102.01], index=idx)
    t = gate.trades(pd.Series([1, 1, 0, 0], index=idx, dtype=float), close)
    assert t["n_trades"] == 1
    assert t["expectancy"] == pytest.approx(0.0201 - gate.COST)
    assert t["total_return"] == pytest.approx((1.01 - gate.COST / 2) * 1.01 * (1 - gate.COST / 2) - 1)


def test_split_adjustment_removes_the_jump():
    raw = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=4),
                        "open": [200.0, 202.0, 101.0, 85.0], "high": 1.0, "low": 1.0,
                        "close": [200.0, 202.0, 102.0, 86.0], "volume": 1})   # 1:1 bonus on day 3
    adj = gate._adjust(raw)["close"]
    assert adj.pct_change().iloc[2] == pytest.approx(102 / 101 - 1)   # only the day's own move
    assert adj.pct_change().iloc[3] == pytest.approx(86 / 102 - 1)    # -16% is a real fall, kept
    assert adj.iloc[0] == 200   # earlier prices untouched: no future information used

