import numpy as np
import pandas as pd
import pytest

from code.research import updown as u


def _raw(autocorr: float, n_sym: int = 40, seed: int = 0) -> pd.DataFrame:
    """Bhavcopy-shaped rows. autocorr > 0 plants a real, causal edge: tomorrow's move leans the
    same way as today's. autocorr = 0 is pure noise."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2013-01-01", "2015-12-31")
    rows = []
    for s in range(n_sym):
        r = np.zeros(len(dates))
        e = rng.normal(0, 0.015, len(dates))
        for t in range(1, len(dates)):
            r[t] = autocorr * r[t - 1] + e[t]
        c = 100 * np.exp(np.cumsum(r))
        o = np.r_[c[0], c[:-1]]                               # opens at yesterday's close: no gaps
        rows.append(pd.DataFrame({"date": dates, "symbol": f"S{s:02d}", "series": "EQ", "open": o,
                                  "high": np.maximum(o, c) * 1.005, "low": np.minimum(o, c) * 0.995, "close": c,
                                  "prev_close": o, "volume": 1e6, "turnover": 1e9, "delivery_qty": 5e5}))
    return pd.concat(rows, ignore_index=True)


def _score(raw):
    df = u.features(u.panel(raw))
    return u.summarize(u.walk_forward(df, first_test="2015-01-01", train_rows=50_000))


def test_finds_a_real_planted_edge():
    d = _score(_raw(0.5))
    assert d["overall"]["auc"] > 0.55 and d["overall"]["edge"]
    r = d["ranking"]
    assert r["ic"] > 0.05 and r["spread"] > 0 and r["deciles"][9] > r["deciles"][0]
    ranks = [s["rank"] for s in d["stocks"].values() if s["rank"] is not None]
    assert d["ranked_today"] >= len(ranks) > 20 and min(ranks) > 0 and max(ranks) == 1.0


def test_pure_noise_is_labelled_no_edge():
    d = _score(_raw(0.0))
    assert d["overall"]["auc"] < u.NO_EDGE_AUC and not d["overall"]["edge"]
    assert abs(d["ranking"]["ic"]) < 0.03


def test_features_never_change_when_later_days_are_removed():
    raw = _raw(0.1, n_sym=8)
    cut = pd.Timestamp("2014-06-30")
    full = u.features(u.panel(raw)).set_index(["date", "symbol"])
    short = u.features(u.panel(raw[raw["date"] <= cut])).set_index(["date", "symbol"])
    both = short.index.intersection(full.index)
    assert len(both) > 1000
    cols = u.FEATURES + u.EXTRA + u.TREND
    pd.testing.assert_frame_equal(full.loc[both, cols], short.loc[both, cols])


def test_training_only_uses_outcomes_settled_before_the_test_month(monkeypatch):
    df = u.features(u.panel(_raw(0.1, n_sym=6)))
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(df.loc[X.index, "label_date"].max())
            return self

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

    monkeypatch.setattr(u, "_model", Spy)
    scored = u.walk_forward(df, first_test="2015-01-01")
    months = sorted(scored["date"].dt.to_period("M").unique())
    assert len(seen) == len(months)
    for latest_label, month in zip(seen, months):
        assert latest_label < month.start_time


def test_outcome_is_buy_next_open_hold_five_sessions():
    raw = _raw(0.0, n_sym=1)
    df = u.features(u.panel(raw)).set_index("date")
    c = raw.set_index("date")["close"]
    o = raw.set_index("date")["open"]
    d = df.index[300]
    i = c.index.get_loc(d)
    assert df.loc[d, "fwd"] == pytest.approx(c.iloc[i + 5] / o.iloc[i + 1] - 1)
    assert df.loc[d, "label_date"] == c.index[i + 5]


def test_longer_horizon_outcome_and_settle_date():
    raw = _raw(0.0, n_sym=1)
    df = u.features(u.panel(raw), horizon=20).set_index("date")
    c, o = raw.set_index("date")["close"], raw.set_index("date")["open"]
    d = df.index[300]
    i = c.index.get_loc(d)
    assert df.loc[d, "fwd"] == pytest.approx(c.iloc[i + 20] / o.iloc[i + 1] - 1)
    assert df.loc[d, "label_date"] == c.index[i + 20]


def test_trend_score_counts_minervini_rules():
    idx = pd.bdate_range("2015-01-01", periods=400)
    up = pd.DataFrame({"UP": 100 * 1.003 ** np.arange(400), "DOWN": 100 * 0.997 ** np.arange(400)}, index=idx)
    rs = pd.DataFrame({"UP": 0.9, "DOWN": 0.1}, index=idx)
    s = u.trend_score(up, rs)
    assert s["UP"].iloc[-1] == 8 and s["DOWN"].iloc[-1] == 0
    assert s.iloc[:199].isna().all().all()                       # needs 200 days of history


def test_split_does_not_look_like_a_crash():
    raw = _raw(0.0, n_sym=1)
    d = raw["date"].iloc[400]
    after = raw["date"] >= d
    raw.loc[after, ["open", "high", "low", "close"]] /= 2          # 1:1 bonus on day 400
    w = u.panel(raw)
    assert abs(w["close"]["S00"].pct_change().iloc[400]) < 0.06
