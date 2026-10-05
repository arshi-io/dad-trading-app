"""Walk-forward evaluation of the setup-outcome model.

Each test year is scored by a model trained only on earlier years (no peeking). Target: did the
breakout reach 2R before the stop within 20 sessions. Probabilities are isotonic-calibrated inside
the training years. Reports what matters to a trader: calibration (does 40% mean 40%), ranking
(AUC), and expectancy of the setups the model would pick vs taking every breakout.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from code.research import bhavcopy

FEATURES = ["template", "rs", "base_len", "base_depth", "c1", "c2", "c3", "vcp", "vol_dry", "bo_vol",
            "close_vs_pivot", "close_loc", "day_ret", "ext50", "from_high", "atr_pct", "deliv_rel", "breadth",
            "mkt_above_200", "mkt_above_50", "mkt_from_high", "mkt_dist_days", "mkt_ret_20"]
MARKET_ONLY = ["breadth", "mkt_above_200", "mkt_above_50", "mkt_from_high", "mkt_dist_days", "mkt_ret_20"]
FIRST_TEST_YEAR = 2015


def models():
    return {
        "logistic": lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                          LogisticRegression(C=0.5, max_iter=2000)),
        "boosting": lambda: CalibratedClassifierCV(
            HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=250, min_samples_leaf=80,
                                           l2_regularization=1.0, random_state=0),
            method="isotonic", cv=3),
    }


def walk_forward(ev: pd.DataFrame, features, make) -> pd.DataFrame:
    ev = ev.copy()
    ev["year"] = ev.date.dt.year
    out = []
    for year in range(FIRST_TEST_YEAR, ev.year.max() + 1):
        train, test = ev[ev.year < year], ev[ev.year == year]
        if test.empty:
            continue
        m = make()
        X = train[features].replace([np.inf, -np.inf], np.nan)
        m.fit(X, train.win.astype(int))
        t = test.copy()
        t["p"] = m.predict_proba(test[features].replace([np.inf, -np.inf], np.nan))[:, 1]
        out.append(t)
    return pd.concat(out)


def report(scored: pd.DataFrame, name: str) -> None:
    y, p = scored.win.astype(int), scored.p
    base = np.full(len(y), y.mean())
    print(f"\n=== {name}: {len(scored):,} out-of-sample breakouts, {scored.year.min()}-{scored.year.max()} ===")
    print(f"AUC {roc_auc_score(y, p):.3f}   Brier {brier_score_loss(y, p):.4f} vs base-rate {brier_score_loss(y, base):.4f}")
    scored = scored.assign(q=pd.qcut(p.rank(method="first"), 5, labels=["Q1 lowest", "Q2", "Q3", "Q4", "Q5 highest"]))
    t = scored.groupby("q", observed=True).agg(n=("r", "size"), predicted=("p", "mean"), actual=("win", "mean"), avg_R=("r", "mean"))
    print("by predicted-probability fifth (calibration + expectancy):")
    print(t.round(3).to_string())
    top = scored[scored.q == "Q5 highest"]
    print(f"take everything: avg R {scored.r.mean():+.3f} | only top fifth: avg R {top.r.mean():+.3f} "
          f"| win {top.win.mean():.3f} vs {scored.win.mean():.3f}")
    yearly = scored.groupby("year").apply(lambda g: pd.Series({
        "all_R": g.r.mean(), "top_R": g[g.q == "Q5 highest"].r.mean(), "auc": roc_auc_score(g.win, g.p) if g.win.nunique() > 1 else np.nan}),
        include_groups=False)
    print("by year (top fifth vs all):")
    print(yearly.round(3).to_string())


if __name__ == "__main__":
    ev = pd.read_parquet(bhavcopy.ROOT / "breakout_events.parquet")
    for name, make in models().items():
        report(walk_forward(ev, FEATURES, make), f"{name}, all features")
    report(walk_forward(ev, MARKET_ONLY, models()["logistic"]), "logistic, market features only")
