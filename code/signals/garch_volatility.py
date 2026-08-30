"""
GARCH volatility — regime detector for the ensemble router.

Why GARCH?
----------
Asset returns exhibit **volatility clustering** — large moves tend to be
followed by large moves (of either sign), and quiet periods persist.  An
unconditional sample-variance estimate smears these regimes together; a
*conditional* variance model — GARCH — tracks the cluster.

GARCH(1,1) says:

    σ²_t = ω + α · ε²_{t−1} + β · σ²_{t−1}

where

    ω (omega)   = long-run baseline variance (unconditional)
    α (alpha)   = how hard yesterday's shock kicks today's variance
    β (beta)    = how much yesterday's variance persists into today
    ε_{t−1}     = yesterday's residual return (mean-zero assumption here)
    σ²_{t−1}    = yesterday's conditional variance

For the process to be covariance-stationary we need ``α + β < 1``; equity
returns typically sit around ``α + β ≈ 0.95`` (high persistence — shocks
decay slowly but do decay).

This module is a *regime* tool, not a directional one.  It produces a
rolling conditional-volatility forecast, classifies each day into
``LOW / MEDIUM / HIGH`` by percentile of the vol history, and converts the
regime to a position-size scalar the rule engine can multiply onto any
other strategy's exposure.  A ``Signal`` is emitted only when the regime
changes — the ensemble router listens for these transitions, not for every
bar's vol reading.
"""
from __future__ import annotations

import logging
import os
import sys
from typing import List

import numpy as np
import pandas as pd

if __name__ == "__main__" and __package__ is None:
    _REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if _REPO_ROOT not in sys.path:
        sys.path.insert(0, _REPO_ROOT)

from code.signals.base import Signal

logger = logging.getLogger(__name__)

STRATEGY_NAME = "garch_volatility"
TRADING_DAYS = 252

# Percentile cutoffs for the three regimes.
LOW_Q, HIGH_Q = 0.33, 0.66

# Position-size scalars per regime.
SCALARS = {"LOW": 1.0, "MEDIUM": 0.75, "HIGH": 0.5}


# ---------------------------------------------------------------------------
# 1. One-shot fit
# ---------------------------------------------------------------------------
def fit_garch(returns: pd.Series, p: int = 1, q: int = 1) -> dict:
    """Fit a GARCH(p,q) on ``returns`` and return summary params.

    Uses ``arch.arch_model`` with ``mean='Zero'`` — we are modelling the
    volatility, not the drift.  Returns are rescaled to *percent* before
    fitting (``r * 100``) purely for numerical stability of the optimiser;
    the reported ``omega`` is therefore in percent-squared.  The downstream
    rolling ``forecast_volatility`` call denormalises back to decimal annual
    vol, so the scale inside this function never escapes.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns (e.g. ``close.pct_change()``).  NaNs are
        dropped.
    p, q : int
        Orders of the GARCH and ARCH terms respectively.

    Returns
    -------
    dict with ``omega``, ``alpha``, ``beta``, ``persistence`` (= α+β),
    ``aic``, ``bic``, ``converged`` (bool).  If there are too few
    observations or the fit fails, all numeric fields are NaN and
    ``converged = False``.
    """
    from arch import arch_model

    r = returns.dropna().astype(float) * 100.0
    empty = {
        "omega": float("nan"),
        "alpha": float("nan"),
        "beta": float("nan"),
        "persistence": float("nan"),
        "aic": float("nan"),
        "bic": float("nan"),
        "converged": False,
    }
    if len(r) < 50:
        return empty

    try:
        model = arch_model(r, vol="Garch", p=p, q=q, mean="Zero", dist="normal",
                           rescale=False)
        res = model.fit(disp="off", show_warning=False)
    except Exception as exc:
        logger.warning("fit_garch failed: %s", exc)
        return empty

    params = res.params
    alpha = float(params.get(f"alpha[{p}]", params.get("alpha[1]", float("nan"))))
    beta = float(params.get(f"beta[{q}]", params.get("beta[1]", float("nan"))))
    omega = float(params.get("omega", float("nan")))
    converged = (getattr(res, "convergence_flag", 0) == 0)

    return {
        "omega": omega,
        "alpha": alpha,
        "beta": beta,
        "persistence": alpha + beta,
        "aic": float(res.aic),
        "bic": float(res.bic),
        "converged": bool(converged),
    }


# ---------------------------------------------------------------------------
# 2. Rolling 1-day-ahead forecast
# ---------------------------------------------------------------------------
def forecast_volatility(
    returns: pd.Series,
    window: int = 252,
    refit_every: int = 5,
) -> pd.Series:
    """Rolling GARCH(1,1) 1-day-ahead volatility forecast.

    The loop fits on the most recent ``window`` returns and forecasts the
    next ``refit_every`` days from that fit before refitting.  For each
    forecast day we return the *annualised* conditional vol:

        σ_annual = sqrt(σ²_daily) · sqrt(252)

    Rolling GARCH is genuinely slow (thousands of optimisations on an
    equity series); ``refit_every=5`` is a practical compromise between
    freshness and speed — the literature typically refits weekly.  Pass
    ``refit_every=1`` if you need bar-by-bar freshness and can afford it.

    Parameters
    ----------
    returns : pd.Series
        Daily simple returns with a DatetimeIndex.  NaNs are dropped.
    window : int, default 252
        Rolling training window.  For speed-sensitive work use 126
        (≈ six months).
    refit_every : int, default 5
        Refit cadence in bars.

    Returns
    -------
    pd.Series named ``garch_vol`` on the *input* index (NaN before the
    first fit bar or on failed fits).
    """
    from arch import arch_model

    out = pd.Series(np.nan, index=returns.index, name="garch_vol", dtype=float)
    r = returns.dropna().astype(float) * 100.0
    if len(r) <= window + 1:
        return out

    i = window
    while i < len(r):
        window_r = r.iloc[i - window:i]
        try:
            model = arch_model(window_r, vol="Garch", p=1, q=1,
                               mean="Zero", dist="normal", rescale=False)
            res = model.fit(disp="off", show_warning=False)
            horizon = min(refit_every, len(r) - i)
            fc = res.forecast(horizon=horizon, reindex=False)
            # ``fc.variance`` has one row per origin, columns h.1..h.N.
            var_path = fc.variance.iloc[-1].values  # in percent²
            for j in range(horizon):
                v = var_path[j]
                if v > 0 and np.isfinite(v):
                    annual_vol = np.sqrt(v) / 100.0 * np.sqrt(TRADING_DAYS)
                    out.loc[r.index[i + j]] = float(annual_vol)
        except Exception as exc:
            logger.debug("rolling fit at %s failed: %s", r.index[i], exc)
        i += refit_every

    return out


# ---------------------------------------------------------------------------
# 3. Regime classification
# ---------------------------------------------------------------------------
def classify_regime(garch_vol: pd.Series) -> pd.Series:
    """Bucket each bar's vol into LOW / MEDIUM / HIGH by percentile.

    Uses the 33rd and 66th percentiles of the non-NaN series as the
    regime boundaries.  Because boundaries are computed from the full
    history, this is a *lookback-biased* classification when used on a
    historical series — which is acceptable for regime *analysis*
    (counting days per regime, aligning to macro events).  For live
    trading you would recompute the quantiles on a trailing expanding
    window; that refinement is deferred to the live engine.

    Returns
    -------
    pd.Series named ``regime`` with string values in
    {``LOW``, ``MEDIUM``, ``HIGH``}; NaN preserves NaN.
    """
    v = garch_vol.dropna()
    if len(v) < 3:
        return pd.Series("MEDIUM", index=garch_vol.index, name="regime")

    lo_cut = float(v.quantile(LOW_Q))
    hi_cut = float(v.quantile(HIGH_Q))

    def bucket(x: float) -> object:
        if pd.isna(x):
            return np.nan
        if x < lo_cut:
            return "LOW"
        if x > hi_cut:
            return "HIGH"
        return "MEDIUM"

    return garch_vol.apply(bucket).rename("regime")


# ---------------------------------------------------------------------------
# 4. Regime → position scalar
# ---------------------------------------------------------------------------
def compute_position_scalar(regime: pd.Series) -> pd.Series:
    """Map the regime label to a size scalar the rule engine consumes.

    * LOW    → 1.0  (full size; quiet market, signals reliable)
    * MEDIUM → 0.75 (moderate size; normal vol)
    * HIGH   → 0.5  (halved size; cluster of big moves, momentum off)

    NaN regimes map to NaN so the rule engine can short-circuit.
    """
    return regime.map(SCALARS).astype(float).rename("position_scalar")


# ---------------------------------------------------------------------------
# 5. Signal emission on regime change
# ---------------------------------------------------------------------------
def generate_vol_signals(
    df: pd.DataFrame,
    ticker: str,
    window: int = 252,
    refit_every: int = 5,
) -> List[Signal]:
    """Emit a Signal on every regime CHANGE (not every bar).

    * ``LOW  → MEDIUM`` or ``LOW  → HIGH``  → **SELL**  (vol up → reduce)
    * ``MEDIUM → HIGH``                       → **SELL**
    * ``HIGH → MEDIUM`` or ``HIGH → LOW``   → **BUY**   (vol down → lean in)
    * ``MEDIUM → LOW``                        → **BUY**

    The ``direction`` here is an *exposure-scaling* instruction for the
    router, not a directional bet on the underlying.  Confidence is the
    normalised distance from the crossed boundary (0 at the cutoff, 1
    deep into the new regime).

    ``metadata`` carries the full context:
    ``regime``, ``prev_regime``, ``garch_vol``, ``position_scalar``,
    ``persistence``.
    """
    if df.empty or "close" not in df.columns:
        return []

    close = df["close"].astype(float)
    ret = close.pct_change()

    vol = forecast_volatility(ret, window=window, refit_every=refit_every)
    regime = classify_regime(vol)
    scalar = compute_position_scalar(regime)

    fit = fit_garch(ret)
    persistence = fit["persistence"]

    v_clean = vol.dropna()
    lo_cut = float(v_clean.quantile(LOW_Q)) if len(v_clean) else float("nan")
    hi_cut = float(v_clean.quantile(HIGH_Q)) if len(v_clean) else float("nan")
    band = (hi_cut - lo_cut) if (pd.notna(hi_cut) and pd.notna(lo_cut) and hi_cut > lo_cut) else float("nan")

    signals: List[Signal] = []
    prev_regime = None

    for ts in df.index:
        r = regime.get(ts)
        if r is None or (isinstance(r, float) and pd.isna(r)):
            continue
        if prev_regime is None:
            prev_regime = r
            continue
        if r == prev_regime:
            continue

        v = float(vol.loc[ts]) if pd.notna(vol.loc[ts]) else float("nan")

        # Direction: vol-up events reduce, vol-down events re-engage.
        up = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
        if up[r] > up[prev_regime]:
            direction = "SELL"
        elif up[r] < up[prev_regime]:
            direction = "BUY"
        else:
            direction = "HOLD"

        # Confidence: distance past the crossed boundary, normalised by the band.
        if r == "LOW" and pd.notna(lo_cut) and pd.notna(band) and band > 0:
            dist = (lo_cut - v) / band
        elif r == "HIGH" and pd.notna(hi_cut) and pd.notna(band) and band > 0:
            dist = (v - hi_cut) / band
        elif r == "MEDIUM" and pd.notna(band) and band > 0:
            dist = min(abs(v - lo_cut), abs(v - hi_cut)) / band
        else:
            dist = 0.0
        confidence = float(np.clip(abs(dist), 0.0, 1.0))

        signals.append(Signal(
            asset=ticker,
            direction=direction,
            confidence=confidence,
            strategy=STRATEGY_NAME,
            timeframe="1d",
            timestamp=ts.to_pydatetime() if hasattr(ts, "to_pydatetime") else ts,
            metadata={
                "regime": r,
                "prev_regime": prev_regime,
                "garch_vol": v,
                "position_scalar": float(scalar.loc[ts]) if pd.notna(scalar.loc[ts]) else float("nan"),
                "persistence": float(persistence) if pd.notna(persistence) else float("nan"),
            },
        ))
        prev_regime = r

    logger.info("generate_vol_signals(%s): %d regime-change signals", ticker, len(signals))
    return signals
