"""Model Lab: compose quant blocks into a price forecast, backtest it, and rank models per stock.

A model is a small graph (the n8n-style canvas on /lab): a Price node feeds direction models
(expected drift) and volatility models (expected spread); optional Ensemble and Calibrate nodes
combine and correct them; a Forecast node turns them into a 5-day path with 68% and 90% bands.

Every block is computed for every day using only data up to that day, so a backtest is one
vectorised pass. Models are scored on two separate periods: the recommender picks on the earlier
one and reports the later one, so the score shown was not used to choose the model.

Honest baseline: "No change" (random walk). The app's own research found no reliable edge in
forecasting direction, and the scores here say whether any combination beats it for a stock.
"""
from __future__ import annotations

import json
import sqlite3
import warnings
from datetime import datetime
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS = 252
HORIZON = 5
SELECT_DAYS = REPORT_DAYS = 200     # forecasts used to choose a model, then to score it
CAL_WINDOW, CAL_MIN, ACI_GAMMA = 120, 60, 0.01
Z68, Z90 = 0.994, 1.645             # normal quantiles for |z| at 68% / 90%

# Node catalogue: what the canvas offers. kind = the type the node outputs; inputs = port types.
def _n(group, label, kind, inputs, params, level, help, learn):
    return {"group": group, "label": label, "kind": kind, "inputs": inputs, "params": params, "level": level, "help": help, "learn": learn}


NODE_TYPES: Dict[str, Dict[str, Any]] = {
    "price": _n("Data", "Price", "price", [], {}, "Beginner",
        "Daily open, high, low and close for the stock (3 years).",
        "Every model starts here. Connect Price into each direction and volatility block."),

    # ---- direction: where is the price heading? (expected daily log return) ----
    "no_change": _n("Direction", "No change", "dir", ["price"], {}, "Beginner",
        "Random walk: the best guess for any day ahead is today's price. The baseline to beat.",
        "Markets are close to a random walk, so 'no change' is very hard to beat. Any direction model should be compared with it: if it doesn't win, the extra idea is noise."),
    "drift": _n("Direction", "Drift", "dir", ["price"], {"window": 250}, "Beginner",
        "Average daily return over the window, carried forward.",
        "μ = mean of the last N daily log returns. Works when a stock has a steady long-run trend; fails at turning points."),
    "momentum": _n("Direction", "Momentum", "dir", ["price"], {"lookback": 20, "shrink": 0.5}, "Beginner",
        "Recent trend continues: the lookback's average return, shrunk toward zero.",
        "μ = shrink × mean return of the last N days. Shrinking (0–1) admits the signal is weak. Momentum is one of the best-documented effects in markets, but mostly over months, not days."),
    "tsmom": _n("Direction", "12-1 momentum", "dir", ["price"], {"lookback": 252, "skip": 21, "shrink": 0.5}, "Intermediate",
        "Last year's return, skipping the most recent month (the classic academic momentum signal).",
        "μ = shrink × (log price 1 month ago − log price 12 months ago) ÷ days. Skipping the last month avoids short-term reversal. From Jegadeesh & Titman (1993) and Moskowitz, Ooi & Pedersen (2012)."),
    "trend": _n("Direction", "Linear trend", "dir", ["price"], {"window": 60}, "Beginner",
        "Slope of a straight line fitted to log price over the window.",
        "μ = OLS slope of log price on time over N days. Simple and visual; reacts slowly to reversals."),
    "holt": _n("Direction", "Holt smoothing", "dir", ["price"], {"alpha": 0.2, "beta": 0.05}, "Intermediate",
        "Exponential smoothing of level and trend; the smoothed trend is the forecast.",
        "levelₜ = α·priceₜ + (1−α)(levelₜ₋₁ + trendₜ₋₁); trendₜ = β(levelₜ − levelₜ₋₁) + (1−β)trendₜ₋₁. Higher α/β react faster but are noisier. A forecasting workhorse since the 1950s."),
    "kalman_trend": _n("Direction", "Kalman trend", "dir", ["price"], {"level_noise": 1.0, "trend_noise": 0.01}, "Advanced",
        "A Kalman filter tracks the hidden level and slope of log price; the slope is the forecast.",
        "State = [level, slope], each drifting as a random walk. Each day: predict, compare with the actual price, correct by the Kalman gain. trend_noise sets how fast the slope may change. Same idea as the Kalman hedge ratio on the Pairs page."),
    "mean_rev": _n("Direction", "Mean reversion", "dir", ["price"], {"sma": 50, "half_life": 20}, "Intermediate",
        "Price is pulled back toward its moving average; half the gap closes in half_life days.",
        "μ = (log SMA − log price) × (1 − 0.5^(horizon/half-life)) ÷ horizon: an Ornstein-Uhlenbeck pull. Works in range-bound stocks, hurts in strong trends."),
    "bollinger": _n("Direction", "Bollinger reversion", "dir", ["price"], {"window": 20, "threshold": 1.0, "strength": 0.5}, "Intermediate",
        "When price is outside its Bollinger band, expect part of the stretch to snap back.",
        "z = (price − 20-day average) ÷ 20-day std. If |z| > threshold, μ = −strength × stretch ÷ horizon; otherwise 0. A classic mean-reversion trigger."),
    "rsi": _n("Direction", "RSI reversion", "dir", ["price"], {"period": 14, "strength": 0.3}, "Intermediate",
        "Overbought (RSI above 50) leans down, oversold leans up.",
        "RSI = 100 − 100 ÷ (1 + average gain ÷ average loss) over the period (Wilder). μ = −strength × (RSI − 50)/50 × daily σ. Popular with traders; test whether it helps on this stock."),
    "macd": _n("Direction", "MACD trend", "dir", ["price"], {"fast": 12, "slow": 26, "shrink": 0.5}, "Intermediate",
        "Fast minus slow exponential average of log price, read as the current trend.",
        "MACD = EMA(fast) − EMA(slow). The two averages lag by (span−1)/2 days, so MACD ÷ that lag gap ≈ the daily trend; μ = shrink × that. Trend-following, like momentum but smoother."),
    "ar1": _n("Direction", "AR(1)", "dir", ["price"], {"window": 250}, "Advanced",
        "Tomorrow's return = β × today's return, β fitted on the window.",
        "rₜ₊₁ = β·rₜ + ε with β from a rolling regression. β > 0: moves continue; β < 0: they bounce back. Usually β is tiny for stocks, which is the point."),
    "donchian": _n("Direction", "Channel breakout", "dir", ["price"], {"window": 20, "strength": 0.2}, "Intermediate",
        "Near the top of its recent high-low channel, lean up; near the bottom, lean down.",
        "position = (price − channel middle) ÷ channel width, from −½ to +½. μ = strength × 2·position × daily σ. The Turtle traders' Donchian channel, as a forecast."),

    # ---- volatility: how far might it move? (daily σ of log returns) ----
    "hist_vol": _n("Volatility", "Historical vol", "vol", ["price"], {"window": 60}, "Beginner",
        "Standard deviation of daily returns over the window.",
        "σ = std of the last N daily log returns. Simple; slow to react when markets get rough."),
    "ewma_vol": _n("Volatility", "EWMA vol", "vol", ["price"], {"lam": 0.94}, "Beginner",
        "RiskMetrics: recent days weigh more (λ = decay).",
        "σ²ₜ = λ·σ²ₜ₋₁ + (1−λ)·r²ₜ. λ = 0.94 is J.P. Morgan's RiskMetrics default; lower λ reacts faster."),
    "parkinson": _n("Volatility", "Parkinson (high-low)", "vol", ["price"], {"window": 20}, "Intermediate",
        "Uses each day's high-low range, which holds more information than the close alone.",
        "σ² = mean of (ln(High/Low))² ÷ (4 ln 2). About 5× more efficient than close-to-close when prices move continuously (Parkinson, 1980)."),
    "garman_klass": _n("Volatility", "Garman-Klass", "vol", ["price"], {"window": 20}, "Advanced",
        "Uses open, high, low and close together.",
        "σ² = mean of ½(ln H/L)² − (2 ln 2 − 1)(ln C/O)². Efficient, but ignores overnight gaps (Garman & Klass, 1980)."),
    "atr_vol": _n("Volatility", "ATR vol", "vol", ["price"], {"window": 14}, "Beginner",
        "Average true range as a % of price, converted to a daily σ (÷1.4).",
        "True range includes gaps from yesterday's close. Traders know ATR from stop-loss sizing; this turns it into a σ."),
    "garch_vol": _n("Volatility", "GARCH(1,1)", "vol", ["price"], {}, "Advanced",
        "σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁, refitted every 10 days on the last year.",
        "Volatility clusters: a shock raises tomorrow's variance, which then decays at rate β. Fitted by maximum likelihood (Bollerslev, 1986). Slower to compute."),
    "gjr_garch": _n("Volatility", "GJR-GARCH", "vol", ["price"], {}, "Advanced",
        "GARCH where falls raise volatility more than rises (the leverage effect).",
        "σ²ₜ = ω + (α + γ·1[εₜ₋₁<0])·ε²ₜ₋₁ + β·σ²ₜ₋₁. Captures that markets get jumpier after drops (Glosten, Jagannathan & Runkle, 1993)."),

    # ---- combine ----
    "ensemble": _n("Combine", "Ensemble", "dir", ["dir", "dir", "dir"], {}, "Beginner",
        "Average of up to three direction models.",
        "Averaging different forecasts usually beats picking one, because their errors partly cancel. One of the most reliable results in forecasting."),
    "blend": _n("Combine", "Weighted blend", "dir", ["dir", "dir"], {"weight": 0.5}, "Intermediate",
        "weight × first input + (1 − weight) × second.",
        "Lets you lean on the model you trust more. Try a few weights and watch the score."),
    "shrink": _n("Combine", "Shrink", "dir", ["dir"], {"factor": 0.5}, "Beginner",
        "Scales a direction forecast toward zero (no change).",
        "Forecasts are usually over-confident. Multiplying by a factor below 1 often improves accuracy: a cheap form of regularisation."),
    "vol_gate": _n("Combine", "Calm-market gate", "dir", ["dir", "vol"], {"window": 250}, "Advanced",
        "Uses the direction forecast only when volatility is below its usual level; otherwise 'no change'.",
        "A simple regime switch: when σ is above its rolling median, signals are noisier, so the gate stands aside. Same idea as the market grade and the GARCH regime elsewhere in the app."),
    "vol_avg": _n("Combine", "Vol average", "vol", ["vol", "vol", "vol"], {}, "Intermediate",
        "Average of up to three volatility models.",
        "Different volatility estimators make different mistakes; averaging them steadies the band."),
    "vol_scale": _n("Combine", "Vol scale", "vol", ["vol"], {"factor": 1.0}, "Beginner",
        "Multiplies volatility: above 1 widens the band, below 1 tightens it.",
        "Without calibration, scaling is how you trade off a tighter band against more misses. With calibration after it, the scale mostly cancels out."),
    "calibrate": _n("Combine", "Conformal calibration", "vol", ["vol"], {}, "Intermediate",
        "Adaptive conformal inference: widens the band after misses and tightens after hits so 68% means 68%.",
        "Keeps a rolling record of how wrong the model was, and sets the band to the quantile that would have caught 68% (and 90%). The level adapts after each miss (Gibbs & Candès, 2021)."),
    "forecast": _n("Output", "Forecast", "forecast", ["dir", "vol"], {}, "Beginner",
        "5-day path with 68% and 90% bands.",
        "Needs one direction input (where to centre the forecast) and one volatility input (how wide the bands are)."),
}


# ---------- blocks: each returns a numpy array aligned to the close series, known at day t -------

def _log_returns(close: pd.Series) -> pd.Series:
    return np.log(close).diff()


def _dir(node: str, p: Dict[str, Any], close: pd.Series) -> np.ndarray:
    """Expected daily log return over the next HORIZON days, as of each day."""
    r, logp = _log_returns(close), np.log(close)
    if node == "no_change":
        return np.zeros(len(close))
    if node == "drift":
        return r.rolling(int(p.get("window", 250)), min_periods=60).mean().to_numpy()
    if node == "momentum":
        return float(p.get("shrink", 0.5)) * r.rolling(int(p.get("lookback", 20))).mean().to_numpy()
    if node == "mean_rev":
        gap = np.log(close.rolling(int(p.get("sma", 50))).mean()) - logp
        closed = 1 - 0.5 ** (HORIZON / max(1.0, float(p.get("half_life", 20))))
        return (gap * closed / HORIZON).to_numpy()
    if node == "ar1":
        w = int(p.get("window", 250))
        lag = r.shift(1)
        beta = (r.rolling(w, min_periods=60).cov(lag) / lag.rolling(w, min_periods=60).var()).clip(-0.9, 0.9)
        b = beta.to_numpy()
        total = r.to_numpy() * b * (1 - b ** HORIZON) / np.where(np.abs(1 - b) < 1e-9, 1, 1 - b)
        return total / HORIZON
    if node == "trend":
        w = int(p.get("window", 60))
        t = pd.Series(np.arange(len(close), dtype=float), index=close.index)
        return (logp.rolling(w).cov(t) / t.rolling(w).var()).to_numpy()
    sd20 = r.rolling(20, min_periods=10).std()
    if node == "tsmom":
        lb, skip = int(p.get("lookback", 252)), int(p.get("skip", 21))
        return (float(p.get("shrink", 0.5)) * (logp.shift(skip) - logp.shift(lb)) / max(1, lb - skip)).to_numpy()
    if node == "holt":
        return _holt(logp.to_numpy(), float(p.get("alpha", 0.2)), float(p.get("beta", 0.05)))
    if node == "kalman_trend":
        return _kalman_slope(logp.to_numpy(), r.to_numpy(), float(p.get("level_noise", 1.0)), float(p.get("trend_noise", 0.01)))
    if node == "bollinger":
        w = int(p.get("window", 20))
        sma, sd = logp.rolling(w).mean(), logp.rolling(w).std()
        stretch = logp - sma
        out = np.where((stretch / sd).abs() > float(p.get("threshold", 1.0)), -float(p.get("strength", 0.5)) * stretch / HORIZON, 0.0)
        return np.where(sd.notna(), out, np.nan)
    if node == "rsi":
        n = int(p.get("period", 14))
        gain = r.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        loss = (-r).clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
        rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
        return (-float(p.get("strength", 0.3)) * (rsi - 50) / 50 * sd20).to_numpy()
    if node == "macd":
        fast, slow = int(p.get("fast", 12)), int(p.get("slow", 26))
        macd = logp.ewm(span=fast, adjust=False, min_periods=slow).mean() - logp.ewm(span=slow, adjust=False, min_periods=slow).mean()
        return (float(p.get("shrink", 0.5)) * macd / max(1.0, (slow - fast) / 2)).to_numpy()
    if node == "donchian":
        w = int(p.get("window", 20))
        hi, lo = close.rolling(w).max(), close.rolling(w).min()
        pos = ((close - (hi + lo) / 2) / (hi - lo).replace(0, np.nan)).fillna(0)
        return np.where(hi.notna(), float(p.get("strength", 0.2)) * 2 * pos * sd20, np.nan)
    raise ValueError(f"{node} is not a direction model")


def _holt(y: np.ndarray, alpha: float, beta: float) -> np.ndarray:
    """Holt's linear trend on log price; returns the trend (daily drift) known at each day."""
    out = np.full(len(y), np.nan)
    level, trend = y[0], 0.0
    for t in range(1, len(y)):
        prev = level
        level = alpha * y[t] + (1 - alpha) * (level + trend)
        trend = beta * (level - prev) + (1 - beta) * trend
        if t >= 20:
            out[t] = trend
    return out


def _kalman_slope(y: np.ndarray, r: np.ndarray, level_noise: float, trend_noise: float) -> np.ndarray:
    """Local linear trend: state [level, slope], observation = level + noise. Noise scales are
    relative to the observation variance, tracked from returns seen so far."""
    out = np.full(len(y), np.nan)
    x, P = np.array([y[0], 0.0]), np.eye(2)
    F, H = np.array([[1.0, 1.0], [0.0, 1.0]]), np.array([1.0, 0.0])
    var = 0.0004
    for t in range(1, len(y)):
        if t > 1 and not np.isnan(r[t - 1]):
            var = 0.97 * var + 0.03 * r[t - 1] ** 2
        Q = np.diag([level_noise * var * 0.1, trend_noise * var * 0.01])
        x, P = F @ x, F @ P @ F.T + Q
        K = P @ H / (H @ P @ H + var)
        x = x + K * (y[t] - H @ x)
        P = P - np.outer(K, H) @ P
        if t >= 20:
            out[t] = x[1]
    return out


def _vol(node: str, p: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    """Daily σ of log returns, as of each day."""
    close = df["close"]
    r = _log_returns(close)
    if node == "hist_vol":
        return r.rolling(int(p.get("window", 60))).std().to_numpy()
    if node == "ewma_vol":
        lam = float(p.get("lam", 0.94))
        return np.sqrt((r ** 2).ewm(alpha=1 - lam, adjust=False, min_periods=20).mean()).to_numpy()
    if node == "garch_vol":
        from code.signals.garch_volatility import forecast_volatility
        annual = forecast_volatility(close.pct_change(), window=250, refit_every=10)
        return (annual / np.sqrt(TRADING_DAYS)).to_numpy()
    if node == "atr_vol":
        prev = close.shift(1)
        tr = pd.concat([df["high"] - df["low"], (df["high"] - prev).abs(), (df["low"] - prev).abs()], axis=1).max(axis=1)
        return (tr.rolling(int(p.get("window", 14))).mean() / close / 1.4).to_numpy()
    lnhl = np.log(df["high"] / df["low"])
    if node == "parkinson":
        return np.sqrt((lnhl ** 2).rolling(int(p.get("window", 20))).mean() / (4 * np.log(2))).to_numpy()
    if node == "garman_klass":
        gk = 0.5 * lnhl ** 2 - (2 * np.log(2) - 1) * np.log(close / df["open"]) ** 2
        return np.sqrt(gk.clip(lower=0).rolling(int(p.get("window", 20))).mean()).to_numpy()
    if node == "gjr_garch":
        return _rolling_garch(close.pct_change())
    raise ValueError(f"{node} is not a volatility model")


def _rolling_garch(returns: pd.Series, window: int = 250, refit_every: int = 10) -> np.ndarray:
    """Daily σ from a GJR-GARCH fitted on the trailing window, refitted every refit_every days;
    the value at day t uses only returns before t."""
    from arch import arch_model
    r = returns.astype(float) * 100
    out = np.full(len(r), np.nan)
    for i in range(window + 1, len(r), refit_every):
        sample = r.iloc[i - window:i].dropna()
        try:
            res = arch_model(sample, vol="GARCH", p=1, o=1, q=1, mean="Zero", rescale=False).fit(disp="off", show_warning=False)
            h = min(refit_every, len(r) - i)
            var = res.forecast(horizon=h, reindex=False).variance.iloc[-1].to_numpy()
            out[i:i + h] = np.sqrt(np.clip(var, 0, None)) / 100
        except Exception:
            continue
    return out


def _calibrated_width(z: np.ndarray, sigma_h: np.ndarray, cov: float) -> np.ndarray:
    """Adaptive conformal width (log units) at each day. z[t] = |realised - forecast| / σ_h for the
    forecast made at t, known only HORIZON days later."""
    n = len(z)
    known = np.concatenate([np.full(HORIZON, np.nan), z[:-HORIZON]])
    out = np.full(n, np.nan)
    level, hits = cov, [None] * n
    for t in range(n):
        if t >= HORIZON and hits[t - HORIZON] is not None:
            level = min(0.999, max(0.5, level + ACI_GAMMA * ((0 if hits[t - HORIZON] else 1) - (1 - cov))))
        window = known[max(0, t - CAL_WINDOW + 1):t + 1]
        window = window[~np.isnan(window)]
        if len(window) >= CAL_MIN and not np.isnan(sigma_h[t]):
            q = float(np.quantile(window, level))
            out[t] = q * sigma_h[t]
            if not np.isnan(z[t]):
                hits[t] = z[t] <= q
    return out


# ---------- graph evaluation -----------------------------------------------------------------

def validate(graph: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, List[str]]]:
    """Return (nodes by id, inputs by node id) or raise ValueError with a message for the user."""
    nodes = {n["id"]: n for n in graph.get("nodes", [])}
    if len(nodes) > 40:
        raise ValueError("Too many blocks (40 max).")
    for n in nodes.values():
        if n.get("type") not in NODE_TYPES:
            raise ValueError(f"Unknown block type: {n.get('type')}")
    inputs: Dict[str, List[str]] = {i: [] for i in nodes}
    for e in graph.get("edges", []):
        if e.get("from") in nodes and e.get("to") in nodes:
            src, dst = NODE_TYPES[nodes[e["from"]]["type"]], NODE_TYPES[nodes[e["to"]]["type"]]
            if src["kind"] not in dst["inputs"]:
                raise ValueError(f"Can't connect {src['label']} into {dst['label']}.")
            inputs[e["to"]].append(e["from"])
    outs = [i for i, n in nodes.items() if n["type"] == "forecast"]
    if len(outs) != 1:
        raise ValueError("Add exactly one Forecast block.")
    return nodes, inputs


def evaluate(graph: Dict[str, Any], df: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Run a model over the whole history. Returns per-day arrays: mu (daily drift), w68/w90
    (half-widths of the HORIZON-day band in log units)."""
    nodes, inputs = validate(graph)
    close = df["close"].astype(float)
    cache: Dict[str, Any] = {}

    def run(nid: str, seen: Tuple[str, ...] = ()) -> Any:
        if nid in seen:
            raise ValueError("The model has a loop.")
        if nid in cache:
            return cache[nid]
        n = nodes[nid]
        t, p, spec = n["type"], n.get("params") or {}, NODE_TYPES[n["type"]]
        ins = [run(i, seen + (nid,)) for i in inputs[nid]]
        if spec["inputs"] and spec["inputs"][0] == "price" and not ins:
            raise ValueError(f"Connect Price into {spec['label']}.")
        if t == "price":
            val = ("price", None)
        elif spec["group"] == "Direction":
            val = ("dir", _dir(t, p, close))
        elif spec["group"] == "Volatility":
            val = ("vol", {"sigma": _vol(t, p, df), "calibrated": False})
        elif t == "ensemble":
            dirs = [v for k, v in ins if k == "dir"]
            if not dirs:
                raise ValueError("Connect at least one direction model into Ensemble.")
            with np.errstate(all="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # all-NaN warm-up days stay NaN
                val = ("dir", np.nanmean(np.vstack(dirs), axis=0) if len(dirs) > 1 else dirs[0])
        elif t == "blend":
            dirs = [v for k, v in ins if k == "dir"]
            if len(dirs) != 2:
                raise ValueError("Weighted blend needs two direction inputs.")
            w = min(1.0, max(0.0, float(p.get("weight", 0.5))))
            val = ("dir", w * dirs[0] + (1 - w) * dirs[1])
        elif t == "shrink":
            dirs = [v for k, v in ins if k == "dir"]
            if not dirs:
                raise ValueError("Connect a direction model into Shrink.")
            val = ("dir", float(p.get("factor", 0.5)) * dirs[0])
        elif t == "vol_gate":
            dirs = [v for k, v in ins if k == "dir"]
            vols = [v for k, v in ins if k == "vol"]
            if not dirs or not vols:
                raise ValueError("Calm-market gate needs a direction input and a volatility input.")
            sig = pd.Series(vols[0]["sigma"])
            usual = sig.rolling(int(p.get("window", 250)), min_periods=60).median().to_numpy()
            val = ("dir", np.where(np.isnan(usual), np.nan, np.where(sig.to_numpy() <= usual, dirs[0], 0.0)))
        elif t == "vol_avg":
            vols = [v for k, v in ins if k == "vol"]
            if not vols:
                raise ValueError("Connect at least one volatility model into Vol average.")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sigma = np.nanmean(np.vstack([v["sigma"] for v in vols]), axis=0)
            val = ("vol", {"sigma": sigma, "calibrated": any(v["calibrated"] for v in vols)})
        elif t == "vol_scale":
            vols = [v for k, v in ins if k == "vol"]
            if not vols:
                raise ValueError("Connect a volatility model into Vol scale.")
            val = ("vol", {**vols[0], "sigma": vols[0]["sigma"] * float(p.get("factor", 1.0))})
        elif t == "calibrate":
            vols = [v for k, v in ins if k == "vol"]
            if not vols:
                raise ValueError("Connect a volatility model into Conformal calibration.")
            val = ("vol", {**vols[0], "calibrated": True})
        else:  # forecast
            dirs = [v for k, v in ins if k == "dir"]
            vols = [v for k, v in ins if k == "vol"]
            if not dirs or not vols:
                raise ValueError("Forecast needs one direction input and one volatility input.")
            val = ("forecast", _forecast_arrays(close, dirs[0], vols[0]))
        cache[nid] = val
        return val

    out_id = next(i for i, n in nodes.items() if n["type"] == "forecast")
    return run(out_id)[1]


def _forecast_arrays(close: pd.Series, mu: np.ndarray, vol: Dict[str, Any]) -> Dict[str, np.ndarray]:
    logc = np.log(close.to_numpy())
    sigma_h = vol["sigma"] * np.sqrt(HORIZON)
    fwd = np.full(len(logc), np.nan)
    fwd[:-HORIZON] = logc[HORIZON:] - logc[:-HORIZON]
    err = np.abs(fwd - mu * HORIZON)
    if vol["calibrated"]:
        z = err / sigma_h
        w68, w90 = _calibrated_width(z, sigma_h, 0.68), _calibrated_width(z, sigma_h, 0.90)
    else:
        w68, w90 = Z68 * sigma_h, Z90 * sigma_h
    return {"mu": mu, "w68": w68, "w90": w90, "fwd": fwd}


# ---------- scoring --------------------------------------------------------------------------

def _score(f: Dict[str, np.ndarray], idx: slice) -> Optional[Dict[str, float]]:
    mu, w68, w90, fwd = (f[k][idx] for k in ("mu", "w68", "w90", "fwd"))
    ok = ~(np.isnan(mu) | np.isnan(w68) | np.isnan(w90) | np.isnan(fwd))
    if ok.sum() < 50:
        return None
    mu, w68, w90, fwd = mu[ok] * HORIZON, w68[ok], w90[ok], fwd[ok]
    lo, hi = mu - w68, mu + w68
    alpha = 0.32
    # Winkler interval score: width plus a penalty for each miss; lower is better. Proper for
    # intervals, so it rewards bands that are both tight and honest.
    interval = (hi - lo) + (2 / alpha) * (np.maximum(lo - fwd, 0) + np.maximum(fwd - hi, 0))
    mae, mae_rw = np.abs(fwd - mu).mean(), np.abs(fwd).mean()
    moved = mu != 0
    return {
        "n": int(ok.sum()),
        "coverage68": float((np.abs(fwd - mu) <= w68).mean()),
        "coverage90": float((np.abs(fwd - mu) <= w90).mean()),
        "interval_score": float(interval.mean() * 100),           # in % of price
        "skill_vs_no_change": float(1 - mae / mae_rw) if mae_rw else 0.0,
        "direction_hit": float((np.sign(mu[moved]) == np.sign(fwd[moved])).mean()) if moved.any() else None,
    }


def _track(f: Dict[str, np.ndarray], df: pd.DataFrame, idx: slice, past_days: int = 60) -> Optional[Dict[str, Any]]:
    """Running accuracy over the scored period, and the ranges the model gave for recent days.

    Uses the same rows as _score (only forecasts whose outcome is known), so the last running
    value equals the scorecard. Each point is dated on the day the forecast was checked (t + HORIZON).
    """
    mu, w68, w90, fwd = (f[k][idx] for k in ("mu", "w68", "w90", "fwd"))
    ok = ~(np.isnan(mu) | np.isnan(w68) | np.isnan(w90) | np.isnan(fwd))
    if ok.sum() < 50:
        return None
    rows = np.arange(len(df))[idx][ok]
    err = np.abs(fwd[ok] - mu[ok] * HORIZON)
    hit68, hit90 = err <= w68[ok], err <= w90[ok]
    moved = mu[ok] != 0
    right = (np.sign(mu[ok]) == np.sign(fwd[ok])) & moved
    dir_n = np.cumsum(moved)
    checked = df.index[rows + HORIZON]
    ts = [int(pd.Timestamp(d).normalize().timestamp()) for d in checked]
    close = df["close"].to_numpy()
    base, center = close[rows], close[rows] * np.exp(mu[ok] * HORIZON)
    k = slice(-past_days, None)
    return {
        "time": ts,
        "coverage68": np.round(np.cumsum(hit68) / np.arange(1, len(err) + 1), 4).tolist(),
        "direction": [round(float(r), 4) if n else None for r, n in zip(np.cumsum(right) / np.maximum(dir_n, 1), dir_n)],
        "n": int(len(err)), "hits68": int(hit68.sum()), "hits90": int(hit90.sum()),
        "dir_n": int(moved.sum()), "dir_hits": int(right.sum()),
        # ranges for recent days as given 5 sessions earlier; drawn behind the candles
        "past": {"time": ts[k], "lo68": np.round(center[k] * np.exp(-w68[ok][k]), 2).tolist(),
                 "hi68": np.round(center[k] * np.exp(w68[ok][k]), 2).tolist(),
                 "caught": hit68[k].tolist(), "base": np.round(base[k], 2).tolist()},
    }


def backtest(graph: Dict[str, Any], df: pd.DataFrame) -> Dict[str, Any]:
    f = evaluate(graph, df)
    n = len(df)
    last_known = n - HORIZON                 # forecasts made after this have no outcome yet
    report = slice(max(0, last_known - REPORT_DAYS), last_known)
    select = slice(max(0, last_known - REPORT_DAYS - SELECT_DAYS), max(0, last_known - REPORT_DAYS))
    dates = df.index
    span = lambda s: f"{dates[s.start].date():%b %Y}–{dates[max(s.start, s.stop - 1)].date():%b %Y}" if s.stop > s.start else ""
    return {"select": _score(f, select), "report": _score(f, report),
            "select_span": span(select), "report_span": span(report), "forecast": _path(f, df),
            "track": _track(f, df, report)}


def _path(f: Dict[str, np.ndarray], df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Today's forecast: centre path and bands for the next HORIZON sessions."""
    mu, w68, w90 = f["mu"][-1], f["w68"][-1], f["w90"][-1]
    if np.isnan(mu) or np.isnan(w68) or np.isnan(w90):
        return None
    last = float(df["close"].iloc[-1])
    day0 = pd.Timestamp(df.index[-1]).normalize()
    days = pd.bdate_range(day0 + pd.Timedelta(days=1), periods=HORIZON)
    pts = lambda fn: [[int(day0.timestamp()), round(last, 2)]] + [[int(d.timestamp()), round(fn(k), 2)] for k, d in enumerate(days, 1)]
    scale = lambda k: np.sqrt(k / HORIZON)
    return {
        "mid": pts(lambda k: last * np.exp(mu * k)),
        "lo68": pts(lambda k: last * np.exp(mu * k - w68 * scale(k))),
        "hi68": pts(lambda k: last * np.exp(mu * k + w68 * scale(k))),
        "lo90": pts(lambda k: last * np.exp(mu * k - w90 * scale(k))),
        "hi90": pts(lambda k: last * np.exp(mu * k + w90 * scale(k))),
        "end_mid": round(last * np.exp(mu * HORIZON), 2),
        "end_lo68": round(last * np.exp(mu * HORIZON - w68), 2), "end_hi68": round(last * np.exp(mu * HORIZON + w68), 2),
    }


# ---------- presets and the recommender -------------------------------------------------------

def _chain(direction: List[Tuple[str, Dict[str, Any]]], vol: Tuple[str, Dict[str, Any]], calibrate: bool = True) -> Dict[str, Any]:
    """Build a graph laid out left-to-right: Price → direction(s) [→ Ensemble] → Forecast ← vol [← Calibrate]."""
    nodes = [{"id": "price", "type": "price", "x": 20, "y": 160, "params": {}}]
    edges = []
    for i, (t, p) in enumerate(direction):
        nodes.append({"id": f"d{i}", "type": t, "params": dict(p), "x": 250, "y": 30 + 95 * i})
        edges.append({"from": "price", "to": f"d{i}"})
    dir_out = "d0"
    if len(direction) > 1:
        nodes.append({"id": "ens", "type": "ensemble", "params": {}, "x": 480, "y": 90})
        edges += [{"from": f"d{i}", "to": "ens"} for i in range(len(direction))]
        dir_out = "ens"
    vy = 50 + 95 * max(len(direction), 1)
    nodes.append({"id": "v", "type": vol[0], "params": dict(vol[1]), "x": 250, "y": vy})
    edges.append({"from": "price", "to": "v"})
    vol_out = "v"
    if calibrate:
        nodes.append({"id": "cal", "type": "calibrate", "params": {}, "x": 480, "y": vy})
        edges.append({"from": "v", "to": "cal"})
        vol_out = "cal"
    nodes.append({"id": "out", "type": "forecast", "params": {}, "x": 710, "y": 160})
    edges += [{"from": dir_out, "to": "out"}, {"from": vol_out, "to": "out"}]
    return {"nodes": nodes, "edges": edges}


PRESETS: List[Tuple[str, Dict[str, Any]]] = [
    ("No change + EWMA, calibrated (the app's default ranges)", _chain([("no_change", {})], ("ewma_vol", {}))),
    ("No change + GARCH, calibrated", _chain([("no_change", {})], ("garch_vol", {}))),
    ("No change + ATR, calibrated", _chain([("no_change", {})], ("atr_vol", {}))),
    ("No change + historical vol, uncalibrated", _chain([("no_change", {})], ("hist_vol", {}), calibrate=False)),
    ("Drift + EWMA, calibrated", _chain([("drift", {})], ("ewma_vol", {}))),
    ("Momentum 20d + EWMA, calibrated", _chain([("momentum", {})], ("ewma_vol", {}))),
    ("Mean reversion to 50-day + EWMA, calibrated", _chain([("mean_rev", {})], ("ewma_vol", {}))),
    ("AR(1) + EWMA, calibrated", _chain([("ar1", {})], ("ewma_vol", {}))),
    ("Linear trend 60d + EWMA, calibrated", _chain([("trend", {})], ("ewma_vol", {}))),
    ("Holt smoothing + EWMA, calibrated", _chain([("holt", {})], ("ewma_vol", {}))),
    ("Kalman trend + Parkinson, calibrated", _chain([("kalman_trend", {})], ("parkinson", {}))),
    ("Bollinger reversion + EWMA, calibrated", _chain([("bollinger", {})], ("ewma_vol", {}))),
    ("12-1 momentum + EWMA, calibrated", _chain([("tsmom", {})], ("ewma_vol", {}))),
    ("Ensemble: momentum + mean reversion + drift, GARCH, calibrated",
     _chain([("momentum", {}), ("mean_rev", {}), ("drift", {})], ("garch_vol", {}))),
]


def recommend(df: pd.DataFrame, extra: List[Tuple[str, Dict[str, Any]]] = ()) -> Dict[str, Any]:
    """Rank presets (plus the user's saved models) by interval score on the selection period;
    report each one's later, unseen period. Ties to the baseline go to the baseline."""
    rows = []
    for name, graph in list(PRESETS) + list(extra):
        try:
            bt = backtest(graph, df)
        except ValueError:
            continue
        if bt["select"] and bt["report"]:
            rows.append({"name": name, "graph": graph, **bt})
    rows.sort(key=lambda r: r["select"]["interval_score"])
    best = rows[0] if rows else None
    baseline = next((r for r in rows if r["name"].startswith("No change + EWMA")), None)
    return {"ranking": rows, "best": best, "baseline": baseline}


# ---------- data and saved models -------------------------------------------------------------

def load_prices(symbol: str) -> pd.DataFrame:
    from code.pipeline.nightly_pipeline import fetch_universe_closes
    df = fetch_universe_closes([symbol]).get(symbol)
    if df is None or df.empty:
        return pd.DataFrame()
    return df.dropna(subset=["open", "high", "low", "close"])


@lru_cache(maxsize=64)
def _cached_recommend(symbol: str, asof: str, saved_key: str) -> Dict[str, Any]:
    df = load_prices(symbol)
    return recommend(df, [(m["name"], m["graph"]) for m in json.loads(saved_key)])


def recommend_for(symbol: str) -> Optional[Dict[str, Any]]:
    """Cached per symbol, data date and saved-model set: a GARCH backtest takes a second or two."""
    df = load_prices(symbol)
    if len(df) < 2 * REPORT_DAYS + 120:
        return None
    saved = [{"name": f"Yours: {m['name']}", "graph": m["graph"]} for m in list_models()]
    return _cached_recommend(symbol, str(df.index[-1].date()), json.dumps(saved, sort_keys=True))


def _db() -> sqlite3.Connection:
    from code.trading.portal import DB_PATH
    conn = sqlite3.connect(DB_PATH)
    conn.execute("CREATE TABLE IF NOT EXISTS lab_models (id INTEGER PRIMARY KEY, name TEXT NOT NULL, graph TEXT NOT NULL, created_at TEXT NOT NULL)")
    return conn


def save_model(name: str, graph: Dict[str, Any]) -> int:
    validate(graph)
    name = name.strip()[:60] or "My model"
    conn = _db()
    try:
        cur = conn.execute("INSERT INTO lab_models (name, graph, created_at) VALUES (?, ?, ?)", (name, json.dumps(graph), datetime.now().isoformat()))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def list_models() -> List[Dict[str, Any]]:
    conn = _db()
    try:
        return [{"id": i, "name": n, "graph": json.loads(g)} for i, n, g in conn.execute("SELECT id, name, graph FROM lab_models ORDER BY id")]
    finally:
        conn.close()


def delete_model(model_id: int) -> None:
    conn = _db()
    try:
        conn.execute("DELETE FROM lab_models WHERE id = ?", (model_id,))
        conn.commit()
    finally:
        conn.close()
