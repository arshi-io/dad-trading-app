import numpy as np
import pandas as pd
import pytest

from code.trading import model_lab as m


def _df(n=700, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2023-01-02", periods=n)
    c = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.015, n))), index=idx)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1000.0})


@pytest.mark.parametrize("name,graph", [p for p in m.PRESETS if "GARCH" not in p[0]])
def test_no_lookahead(name, graph):
    """A forecast made on day t must not change when later prices arrive."""
    df = _df()
    short, full = m.evaluate(graph, df.iloc[:500]), m.evaluate(graph, df)
    for k in ("mu", "w68", "w90"):
        np.testing.assert_allclose(short[k][:500 - m.HORIZON], full[k][:500 - m.HORIZON], equal_nan=True)


def test_calibrated_baseline_covers_and_has_zero_skill():
    bt = m.backtest(m.PRESETS[0][1], _df())
    assert 0.6 < bt["report"]["coverage68"] < 0.76 and 0.84 < bt["report"]["coverage90"] < 0.95
    assert bt["report"]["skill_vs_no_change"] == 0 and bt["report"]["direction_hit"] is None
    f = bt["forecast"]
    assert len(f["mid"]) == m.HORIZON + 1 and f["end_lo68"] < f["end_mid"] < f["end_hi68"]


def test_recommender_picks_on_selection_period_and_ranks_everything():
    rec = m.recommend(_df(), [("Yours: momentum only", m._chain([("momentum", {})], ("hist_vol", {})))])
    scores = [r["select"]["interval_score"] for r in rec["ranking"]]
    assert scores == sorted(scores) and rec["best"] is rec["ranking"][0]
    assert any(r["name"].startswith("Yours:") for r in rec["ranking"])


def test_invalid_graphs_explain_themselves():
    with pytest.raises(ValueError, match="exactly one Forecast"):
        m.validate({"nodes": [{"id": "p", "type": "price"}], "edges": []})
    with pytest.raises(ValueError, match="Can't connect"):
        m.validate({"nodes": [{"id": "p", "type": "price"}, {"id": "o", "type": "forecast"}], "edges": [{"from": "p", "to": "o"}]})
    g = {"nodes": [{"id": "p", "type": "price"}, {"id": "d", "type": "drift"}, {"id": "o", "type": "forecast"}],
         "edges": [{"from": "p", "to": "d"}, {"from": "d", "to": "o"}]}
    with pytest.raises(ValueError, match="one volatility input"):
        m.evaluate(g, _df())


FAST_DIR = [k for k, t in m.NODE_TYPES.items() if t["group"] == "Direction"]
FAST_VOL = [k for k, t in m.NODE_TYPES.items() if t["group"] == "Volatility" and "garch" not in k]


def _no_lookahead(graph):
    df = _df()
    short, full = m.evaluate(graph, df.iloc[:500]), m.evaluate(graph, df)
    for k in ("mu", "w68", "w90"):
        np.testing.assert_allclose(short[k][:500 - m.HORIZON], full[k][:500 - m.HORIZON], equal_nan=True, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("block", FAST_DIR)
def test_every_direction_block_is_past_only(block):
    _no_lookahead(m._chain([(block, {})], ("ewma_vol", {})))


@pytest.mark.parametrize("block", FAST_VOL)
def test_every_volatility_block_is_past_only(block):
    _no_lookahead(m._chain([("no_change", {})], (block, {})))


def _graph(nodes, edges):
    return {"nodes": [{"id": i, "type": t, "params": p} for i, t, p in nodes], "edges": [{"from": a, "to": b} for a, b in edges]}


def test_combine_blocks_work_and_are_past_only():
    blend_gate = _graph(
        [("p", "price", {}), ("a", "momentum", {}), ("b", "rsi", {}), ("bl", "blend", {"weight": 0.7}), ("sh", "shrink", {"factor": 0.5}),
         ("v1", "ewma_vol", {}), ("v2", "parkinson", {}), ("va", "vol_avg", {}), ("vs", "vol_scale", {"factor": 1.2}),
         ("g", "vol_gate", {}), ("c", "calibrate", {}), ("o", "forecast", {})],
        [("p", "a"), ("p", "b"), ("a", "bl"), ("b", "bl"), ("bl", "sh"), ("p", "v1"), ("p", "v2"), ("v1", "va"), ("v2", "va"),
         ("va", "vs"), ("sh", "g"), ("vs", "g"), ("vs", "c"), ("g", "o"), ("c", "o")])
    _no_lookahead(blend_gate)
    bt = m.backtest(blend_gate, _df())
    assert bt["report"] and bt["forecast"]


def test_shrink_zero_equals_no_change():
    g = _graph([("p", "price", {}), ("d", "momentum", {}), ("s", "shrink", {"factor": 0}), ("v", "ewma_vol", {}), ("o", "forecast", {})],
               [("p", "d"), ("d", "s"), ("p", "v"), ("s", "o"), ("v", "o")])
    f = m.evaluate(g, _df())
    assert np.nanmax(np.abs(f["mu"])) == 0


def test_every_block_has_learning_text_and_level():
    for key, t in m.NODE_TYPES.items():
        assert t["help"] and t["learn"] and t["level"] in ("Beginner", "Intermediate", "Advanced"), key


@pytest.mark.parametrize("name,graph", [p for p in m.PRESETS if "GARCH" not in p[0]][:6])
def test_track_record_matches_scorecard_and_uses_known_outcomes(name, graph):
    df = _df()
    bt = m.backtest(graph, df)
    t, rep = bt["track"], bt["report"]
    assert t["n"] == rep["n"] and len(t["time"]) == t["n"] == len(t["coverage68"])
    assert t["coverage68"][-1] == pytest.approx(rep["coverage68"], abs=1e-4)
    assert t["hits90"] / t["n"] == pytest.approx(rep["coverage90"])
    if rep["direction_hit"] is None:
        assert t["dir_n"] == 0 and t["direction"][-1] is None
    else:
        assert t["dir_hits"] / t["dir_n"] == pytest.approx(rep["direction_hit"])
    last_checked = pd.Timestamp(t["time"][-1], unit="s")
    assert last_checked <= df.index[-1]                       # every point already has its outcome
