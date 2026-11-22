import numpy as np
import pandas as pd
import pytest

from code.research import model_search as ms
from code.research import signals_lib as sl
from code.research import updown as u
from code.research import zoo
from tests.test_updown import _raw


def test_library_sizes():
    w = u.panel(_raw(0.0, n_sym=3))
    assert len(dict(sl.alpha158(w))) == 158
    assert len(dict(sl.alpha101(w))) == 25


def test_signals_never_change_when_later_days_are_removed():
    raw = _raw(0.1, n_sym=4)
    cut = pd.Timestamp("2014-06-30")
    full, short = u.panel(raw), u.panel(raw[raw["date"] <= cut])
    for lib in (sl.alpha158, sl.alpha101):
        a, b = dict(lib(full)), dict(lib(short))
        for name in a:
            x, y = a[name].loc[:cut].iloc[-200:], b[name].iloc[-200:]
            pd.testing.assert_frame_equal(x, y, check_names=False, rtol=1e-6, atol=1e-9, obj=name)


def test_rolling_fit_matches_numpy():
    y = pd.DataFrame({"s": [1.0, 3.0, 2.0, 5.0, 4.0, 7.0]})
    slope, rsq, resi = sl._slope_rsq_resi(y, 4)
    b, a = np.polyfit(np.arange(4), y["s"].iloc[2:6], 1)
    r = np.corrcoef(np.arange(4), y["s"].iloc[2:6])[0, 1]
    assert slope.iloc[-1, 0] == pytest.approx(b) and rsq.iloc[-1, 0] == pytest.approx(r * r)
    assert resi.iloc[-1, 0] == pytest.approx(7.0 - (a + b * 3))


def test_families_map_inputs_to_known_effects():
    assert zoo.family("ROC5") == "short-term" and zoo.family("ROC60") == "medium-term"
    assert zoo.family("KMID") == "candle" and zoo.family("VMA20") == "volume"
    assert zoo.family("mom_12_1") == "medium-term" and zoo.family("vol20") == "volatility"
    assert zoo.family("A101") == "candle" and zoo.family("gap") == "gap"


def test_zoo_finds_a_planted_edge_and_classic_signs_are_right(monkeypatch):
    monkeypatch.setattr(ms, "VALID", ("2014-07-01", "2015-07-01"))
    monkeypatch.setattr(ms, "LOCKED", "2015-07-01")
    df, a158, a101 = zoo.build(1, raw=_raw(0.5, n_sym=45))        # >= 30 liquid stocks a day to score
    res = zoo.run(1, log=lambda s: None, df=df, a158=a158, a101=a101, train_rows=30_000,
                  start="2014-07-01", keep_at="2014-10-01")
    by = {m["name"]: m for m in res["models"]}
    assert max(m["locked"]["ic"] for m in res["models"] if m["kind"] == "fitted") > 0.1
    # tomorrow leans the same way as today: short-term reversal must score negative here
    assert by["Short-term reversal"]["locked"]["ic"] < 0
    assert res["blends"][0]["locked"]["ic"] > 0.05
    assert res["common_pattern"] and res["agreement"]["NN3 neural net"]["NN3 neural net"] == pytest.approx(1.0)
