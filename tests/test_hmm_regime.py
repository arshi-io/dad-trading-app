import numpy as np
import pandas as pd
import pytest

from code.regime.hmm_regime import LABELS, features, fit, latest_regime, regime_probabilities

KW = dict(min_train=400, n_inits=2, seed=3)


@pytest.fixture(scope="module")
def close():
    """Alternating 200-day blocks of quiet (0.5%/day) and violent (3%/day) markets."""
    rng = np.random.default_rng(11)
    sig = np.tile(np.r_[np.full(200, 0.005), np.full(200, 0.03)], 3)
    return pd.Series(1000 * np.exp(np.cumsum(rng.normal(0.0003, sig))), index=pd.bdate_range("2015-01-01", periods=len(sig)))


@pytest.fixture(scope="module")
def probs(close):
    return regime_probabilities(close, **KW)


def test_deterministic(close, probs):
    pd.testing.assert_frame_equal(probs, regime_probabilities(close, **KW))


def test_labels_are_ordered_by_volatility(close):
    f = fit(features(close).to_numpy()[:600], n_inits=2, seed=3)
    vol_means = f.model.means_[f.order, 1]
    assert list(vol_means) == sorted(vol_means)


def test_obvious_regimes_are_found(close, probs):
    p = probs.dropna()
    quiet = p[np.isin(p.index, close.index[820:1000])]      # well inside a quiet block
    wild = p[np.isin(p.index, close.index[1020:1200])]       # well inside a violent block
    # Two true regimes, three states: the model must split one of them, so the violent block is
    # Choppy/Stress -- what matters is that quiet days are Calm and violent days never are.
    assert (quiet["label"] == "Calm trend").mean() > 0.8
    assert (wild["label"] == "Calm trend").mean() < 0.05
    assert (quiet["label"] == "Stress").mean() < 0.05 and (wild["label"] == "Stress").mean() > 0.3


def test_no_future_data(close, probs):
    cut = 1010                                               # mid-month, so the cut month is partial
    early = regime_probabilities(close.iloc[:cut], **KW)
    pd.testing.assert_frame_equal(probs.loc[early.index], early)


def test_latest_matches_the_series(close, probs):
    last = latest_regime(close, **KW)
    row = probs.iloc[-1]
    assert last["label"] == row["label"]
    assert last["probs"] == {lab: round(float(row[lab]), 3) for lab in LABELS}
    assert abs(sum(last["probs"].values()) - 1) < 0.01
