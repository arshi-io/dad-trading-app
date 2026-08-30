from __future__ import annotations

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd

from code.signals.garch_volatility import classify_regime

logger = logging.getLogger(__name__)


def classify_market_regime(
    garch_vol: pd.Series,
    nifty_close: pd.Series,
    breadth_series: pd.Series,
) -> Dict[str, Any]:
    """Classify market regime using the P1 formulas.

    Formula per P1/P3 guidance:
    - use existing garch_volatility.classify_regime for vol state
    - compute NIFTY 200DMA slope from close series
    - compute breadth percent > 200DMA
    - map to TRENDING_UP / CHOPPY / TRENDING_DOWN
    """
    vol_state = classify_regime(garch_vol).astype(str)
    vol_state = vol_state.fillna("MEDIUM")

    close = pd.Series(nifty_close.squeeze() if hasattr(nifty_close, "squeeze") else nifty_close, dtype=float).dropna()
    if len(close) < 200:
        slope = np.nan
        breadth = np.nan
    else:
        sma200 = close.rolling(200, min_periods=200).mean()
        slope = (close.iloc[-1] - sma200.iloc[-1]) / sma200.iloc[-1]
        breadth = float((close > sma200).mean() * 100.0)

    breadth_series = pd.Series(breadth_series, dtype=float).dropna()
    if not breadth_series.empty:
        breadth = float(breadth_series.iloc[-1])

    if np.isnan(slope):
        slope = 0.0

    if np.isnan(breadth):
        breadth = 50.0

    if slope > 0.01 and breadth > 55 and vol_state.iloc[-1] in {"LOW", "MEDIUM"}:
        regime = "TRENDING_UP"
    elif slope < -0.01 and breadth < 45:
        regime = "TRENDING_DOWN"
    else:
        regime = "CHOPPY"

    return {
        "regime": regime,
        "vol_state": vol_state.iloc[-1] if len(vol_state) else "MEDIUM",
        "slope": float(slope),
        "breadth_pct": float(breadth),
        "details": {
            "vol_state_series": vol_state.to_dict(),
            "nifty_200dma_slope": float(slope),
            "breadth_pct_above_200dma": float(breadth),
        },
    }
