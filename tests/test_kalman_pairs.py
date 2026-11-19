import numpy as np
import pandas as pd

from code.signals.kalman_pairs import kalman_pair
from code.signals.pairs_trading import _ols_slope


def _synthetic(n=800, seed=7):
    """B is a random walk; A = beta_t * B + noise, with beta jumping 1.0 -> 1.5 halfway."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n)
    b = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx)
    beta = np.where(np.arange(n) < n // 2, 1.0, 1.5)
    a = pd.Series(beta * b.to_numpy() + rng.normal(0, 0.5, n), index=idx)
    return a, b


def test_kalman_tracks_a_beta_change_that_static_ols_misses():
    a, b = _synthetic()
    k = kalman_pair(a, b)
    first, second = k["beta"].iloc[200:400].mean(), k["beta"].iloc[-100:].mean()
    assert abs(first - 1.0) < 0.1 and abs(second - 1.5) < 0.1
    ols = _ols_slope(a, b)                       # one number for the whole period
    assert abs(ols - 1.0) > 0.1 and abs(ols - 1.5) > 0.1


def test_value_at_t_uses_only_data_up_to_t():
    a, b = _synthetic()
    full = kalman_pair(a, b)
    cut = kalman_pair(a.iloc[:500], b.iloc[:500])
    pd.testing.assert_frame_equal(full.iloc[:500], cut)


def test_zscore_is_standardised_forecast_error():
    a, b = _synthetic()
    z = kalman_pair(a, b)["zscore"].iloc[100:]
    assert abs(z.mean()) < 0.3 and 0.5 < z.std() < 2.0
