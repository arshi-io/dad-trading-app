import numpy as np
import pandas as pd

from code.signals.garch_volatility import MIN_HISTORY, classify_regime


def test_regime_cutoffs_use_only_past_volatility():
    vol = pd.Series(np.r_[np.linspace(10, 20, 100), np.full(50, 40.0)], index=pd.bdate_range("2024-01-01", periods=150))
    full = classify_regime(vol)
    early = classify_regime(vol.iloc[:100])
    # Appending a later volatility spike must not relabel earlier days.
    assert full.iloc[:100].equals(early)
    assert (full.iloc[:MIN_HISTORY - 1] == "MEDIUM").all()
    assert full.iloc[99] == "HIGH"          # top of its own history so far
    assert full.iloc[-1] == "HIGH"


def test_nan_stays_nan():
    vol = pd.Series([np.nan] + list(range(1, 40)), index=pd.bdate_range("2024-01-01", periods=40), dtype=float)
    out = classify_regime(vol)
    assert pd.isna(out.iloc[0]) and out.iloc[-1] == "HIGH"


def test_pairs_half_life_bounds_are_inclusive():
    from code.pipeline.nightly_pipeline import PAIRS_HALF_LIFE_MAX, PAIRS_HALF_LIFE_MIN, _half_life_ok
    assert _half_life_ok(PAIRS_HALF_LIFE_MIN) and _half_life_ok(PAIRS_HALF_LIFE_MAX) and _half_life_ok(12.0)
    assert not _half_life_ok(4.9) and not _half_life_ok(40.1) and not _half_life_ok(float("inf"))
    assert not _half_life_ok(float("nan"))
