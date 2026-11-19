"""Hidden Markov Model market regime on NIFTY 50 -- runs alongside the A/B/C grade and GARCH regime.

Features per day: log return and 20-day realised volatility (annualised), standardised with the
training window's own mean and std. A 3-state Gaussian HMM (hmmlearn) is fitted on an EXPANDING
window and refitted monthly: every day in a month is labelled by a model trained only on days
before that month started. Several fixed-seed inits are tried and the best log-likelihood kept.

Labels for a day come from FILTERED probabilities -- a forward pass over data up to that day only.
hmmlearn's predict_proba / Viterbi use the whole sequence (forward-backward), which would leak later
days into earlier labels, so they are never used here.

States are named by their mean volatility after each fit (lowest = "Calm trend", middle =
"Choppy", highest = "Stress"), so a label means the same thing across refits even though the HMM's
internal state numbers are arbitrary.
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

logger = logging.getLogger(__name__)

LABELS = ("Calm trend", "Choppy", "Stress")
VOL_WINDOW = 20
MIN_TRAIN = 750      # ~3 years of sessions before the first label
N_INITS = 5
SEED = 7


def features(close: pd.Series) -> pd.DataFrame:
    ret = np.log(close.astype(float)).diff()
    vol = ret.rolling(VOL_WINDOW).std() * np.sqrt(252)
    return pd.DataFrame({"ret": ret, "vol": vol}).dropna()


@dataclass
class Fitted:
    model: object
    mu: np.ndarray
    sd: np.ndarray
    order: np.ndarray      # HMM state index for each label, lowest volatility first
    loglik: float


def fit(train: np.ndarray, n_inits: int = N_INITS, seed: int = SEED) -> Fitted:
    from hmmlearn.hmm import GaussianHMM
    mu, sd = train.mean(axis=0), train.std(axis=0)
    xs = (train - mu) / sd
    best: Optional[Fitted] = None
    for i in range(n_inits):
        m = GaussianHMM(n_components=len(LABELS), covariance_type="full", n_iter=200, tol=1e-4, random_state=seed + i)
        with warnings.catch_warnings():  # hmmlearn 0.3.3 trips a NumPy 2.5 deprecation internally, thousands per fit
            warnings.filterwarnings("ignore", category=DeprecationWarning, module="hmmlearn")
            m.fit(xs)
            ll = m.score(xs)
        if best is None or ll > best.loglik:
            best = Fitted(m, mu, sd, np.argsort(m.means_[:, 1]), ll)
    return best


def filtered_probs(f: Fitted, x: np.ndarray) -> np.ndarray:
    """P(state_t | x_1..x_t) for every t, columns in LABELS order. Forward algorithm only."""
    m = f.model
    xs = (x - f.mu) / f.sd
    logb = np.column_stack([multivariate_normal(m.means_[k], m.covars_[k], allow_singular=True).logpdf(xs)
                            for k in range(m.n_components)])
    b = np.exp(logb - logb.max(axis=1, keepdims=True))   # per-row rescale; cancels in the normalisation
    out = np.empty_like(b)
    a = m.startprob_ * b[0]
    out[0] = a / a.sum()
    for t in range(1, len(b)):
        a = (out[t - 1] @ m.transmat_) * b[t]
        out[t] = a / a.sum()
    return out[:, f.order]


def regime_probabilities(close: pd.Series, min_train: int = MIN_TRAIN, n_inits: int = N_INITS, seed: int = SEED) -> pd.DataFrame:
    """Daily filtered state probabilities with monthly expanding-window refits (research use)."""
    feat = features(close)
    x, idx = feat.to_numpy(), feat.index
    month = idx.to_period("M")
    out = np.full((len(idx), len(LABELS)), np.nan)
    for mth in month.unique():
        rows = np.flatnonzero(month == mth)
        start = rows[0]
        if start < min_train:
            continue
        f = fit(x[:start], n_inits, seed)
        out[rows] = filtered_probs(f, x[:rows[-1] + 1])[rows]
    df = pd.DataFrame(out, index=idx, columns=list(LABELS))
    known = ~np.isnan(out[:, 0])
    df["label"] = None
    df.loc[known, "label"] = np.array(LABELS)[np.argmax(out[known], axis=1)]
    return df


def latest_regime(close: pd.Series, min_train: int = MIN_TRAIN, n_inits: int = N_INITS, seed: int = SEED) -> Optional[dict]:
    """Today's label and probabilities -- the same numbers regime_probabilities gives for the last day,
    without fitting every past month."""
    feat = features(close)
    if feat.empty:
        return None
    x, idx = feat.to_numpy(), feat.index
    start = int(np.flatnonzero(idx.to_period("M") == idx[-1].to_period("M"))[0])
    if start < min_train:
        return None
    f = fit(x[:start], n_inits, seed)
    p = filtered_probs(f, x)[-1]
    k = int(np.argmax(p))
    return {
        "label": LABELS[k],
        "prob": round(float(p[k]), 3),
        "probs": {lab: round(float(v), 3) for lab, v in zip(LABELS, p)},
        "asof": idx[-1].date().isoformat(),
        "trained_through": idx[start - 1].date().isoformat(),
    }
