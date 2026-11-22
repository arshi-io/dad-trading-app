"""Published signal libraries, computed from daily OHLCV wide frames (date x symbol).

- alpha158: Microsoft Qlib's Alpha158 feature set (qlib/contrib/data/handler.py): 9 candle shapes,
  4 price ratios, 29 rolling operators x windows 5/10/20/30/60.
- alpha101: 25 of WorldQuant's "101 Formulaic Alphas" (Kakushadze 2016) that need only OHLCV + VWAP.
Every value on a day uses that day's close and earlier only. VWAP = turnover / volume.
"""
from __future__ import annotations

from typing import Dict, Iterator, Tuple

import numpy as np
import pandas as pd

WINDOWS = (5, 10, 20, 30, 60)
EPS = 1e-12


# ---------- rolling helpers -----------------------------------------------------------------

def _slope_rsq_resi(y: pd.DataFrame, d: int):
    """Rolling linear fit of y on time 0..d-1: slope, R-squared, last point's residual."""
    t = pd.Series(np.arange(len(y), dtype=float), index=y.index)
    tt = pd.DataFrame(np.repeat(t.to_numpy()[:, None], y.shape[1], axis=1), index=y.index, columns=y.columns)
    r = lambda x: x.rolling(d, min_periods=d)
    mt, my = r(tt).mean(), r(y).mean()
    cov = r(tt * y).mean() - mt * my
    vt = (d * d - 1) / 12.0                                  # variance of 0..d-1
    vy = r(y * y).mean() - my * my
    slope = cov / vt
    rsq = (cov * cov) / (vt * vy.where(vy > EPS))
    resi = y - (my + slope * (tt - mt))
    return slope, rsq, resi


def _idx_arg(x: pd.DataFrame, d: int, fn) -> pd.DataFrame:
    """Rolling position (1..d) of the max/min, counted from the start of the window, in column chunks."""
    a = x.to_numpy(dtype=float)
    out = np.full(a.shape, np.nan)
    if len(a) < d:
        return pd.DataFrame(out, index=x.index, columns=x.columns)
    for c0 in range(0, a.shape[1], 300):
        blk = np.nan_to_num(a[:, c0:c0 + 300], nan=-np.inf if fn is np.argmax else np.inf)
        v = np.lib.stride_tricks.sliding_window_view(blk, d, axis=0)
        out[d - 1:, c0:c0 + 300] = fn(v, axis=-1) + 1
    out[np.isnan(a)] = np.nan
    return pd.DataFrame(out, index=x.index, columns=x.columns)


# ---------- Alpha158 ------------------------------------------------------------------------

def alpha158(w: Dict[str, pd.DataFrame]) -> Iterator[Tuple[str, pd.DataFrame]]:
    O, H, L, C, V = w["open"], w["high"], w["low"], w["close"], w["volume"]
    vwap = (w["turnover"] / V.replace(0, np.nan)) / w.get("cum", 1.0)       # same price basis as C
    f: Dict[str, pd.DataFrame] = {}
    mx, mn, rng = np.maximum(O, C), np.minimum(O, C), H - L + EPS
    f.update(KMID=(C - O) / O, KLEN=(H - L) / O, KMID2=(C - O) / rng, KUP=(H - mx) / O, KUP2=(H - mx) / rng,
             KLOW=(mn - L) / O, KLOW2=(mn - L) / rng, KSFT=(2 * C - H - L) / O, KSFT2=(2 * C - H - L) / rng)
    f.update(OPEN0=O / C, HIGH0=H / C, LOW0=L / C, VWAP0=vwap / C)
    yield from f.items()
    f.clear()
    ch = C - C.shift(1)
    ret = C / C.shift(1) - 1
    lv, dlv = np.log(V + 1), np.log(V / V.shift(1) + 1)
    av = ret.abs() * V
    up, dn, ab = ch.clip(lower=0), (-ch).clip(lower=0), ch.abs()
    vch = V - V.shift(1)
    for d in WINDOWS:
        r = lambda x: x.rolling(d, min_periods=d)
        slope, rsq, resi = _slope_rsq_resi(C, d)
        hmax, lmin = r(H).max(), r(L).min()
        f[f"ROC{d}"] = C.shift(d) / C
        f[f"MA{d}"] = r(C).mean() / C
        f[f"STD{d}"] = r(C).std() / C
        f[f"BETA{d}"] = slope / C
        f[f"RSQR{d}"] = rsq
        f[f"RESI{d}"] = resi / C
        f[f"MAX{d}"] = hmax / C
        f[f"MIN{d}"] = lmin / C
        f[f"QTLU{d}"] = r(C).quantile(0.8) / C
        f[f"QTLD{d}"] = r(C).quantile(0.2) / C
        f[f"RANK{d}"] = r(C).rank(pct=True)
        f[f"RSV{d}"] = (C - lmin) / (hmax - lmin + EPS)
        imax, imin = _idx_arg(H, d, np.argmax), _idx_arg(L, d, np.argmin)
        f[f"IMAX{d}"], f[f"IMIN{d}"], f[f"IMXD{d}"] = imax / d, imin / d, (imax - imin) / d
        f[f"CORR{d}"] = r(C).corr(lv)
        f[f"CORD{d}"] = r(C / C.shift(1)).corr(dlv)
        cp, cn = r((ch > 0).astype(float)).mean(), r((ch < 0).astype(float)).mean()
        f[f"CNTP{d}"], f[f"CNTN{d}"], f[f"CNTD{d}"] = cp, cn, cp - cn
        su, sd, sa = r(up).sum(), r(dn).sum(), r(ab).sum() + EPS
        f[f"SUMP{d}"], f[f"SUMN{d}"], f[f"SUMD{d}"] = su / sa, sd / sa, (su - sd) / sa
        f[f"VMA{d}"] = r(V).mean() / (V + EPS)
        f[f"VSTD{d}"] = r(V).std() / (V + EPS)
        f[f"WVMA{d}"] = r(av).std() / (r(av).mean() + EPS)
        vu, vd, va = r(vch.clip(lower=0)).sum(), r((-vch).clip(lower=0)).sum(), r(vch.abs()).sum() + EPS
        f[f"VSUMP{d}"], f[f"VSUMN{d}"], f[f"VSUMD{d}"] = vu / va, vd / va, (vu - vd) / va
        yield from f.items()                 # one window at a time keeps memory bounded
        f.clear()


# ---------- WorldQuant 101 (OHLCV-only subset) -------------------------------------------

def alpha101(w: Dict[str, pd.DataFrame]) -> Iterator[Tuple[str, pd.DataFrame]]:
    O, H, L, C, V = w["open"], w["high"], w["low"], w["close"], w["volume"]
    vwap = (w["turnover"] / V.replace(0, np.nan)) / w.get("cum", 1.0)
    ret = C / C.shift(1) - 1
    rank = lambda x: x.rank(axis=1, pct=True)
    ts = lambda x, d: x.rolling(d, min_periods=d)
    corr = lambda a, b, d: ts(a, d).corr(b)
    cov = lambda a, b, d: ts(a, d).cov(b)
    tsrank = lambda x, d: ts(x, d).rank(pct=True)
    delta = lambda x, d: x - x.shift(d)
    std = lambda x, d: ts(x, d).std()
    d1 = delta(C, 1)
    a: Dict[str, pd.DataFrame] = {}
    a["A002"] = -corr(rank(delta(np.log(V + 1), 2)), rank((C - O) / O), 6)
    a["A003"] = -corr(rank(O), rank(V), 10)
    a["A004"] = -tsrank(rank(L), 9)
    a["A006"] = -corr(O, V, 10)
    a["A009"] = d1.where((ts(d1, 5).min() > 0) | (ts(d1, 5).max() < 0), -d1)
    a["A012"] = np.sign(delta(V, 1)) * -d1
    a["A013"] = -rank(cov(rank(C), rank(V), 5))
    a["A014"] = -rank(delta(ret, 3)) * corr(O, V, 10)
    a["A015"] = -ts(rank(corr(rank(H), rank(V), 3)), 3).sum()
    a["A016"] = -rank(cov(rank(H), rank(V), 5))
    a["A018"] = -rank(std((C - O).abs(), 5) + (C - O) + corr(C, O, 10))
    a["A020"] = -rank(O - H.shift(1)) * rank(O - C.shift(1)) * rank(O - L.shift(1))
    a["A022"] = -delta(corr(H, V, 5), 5) * rank(std(C, 20))
    a["A023"] = (-delta(H, 2)).where(ts(H, 20).mean() < H, 0.0)
    a["A026"] = -ts(corr(tsrank(V, 5), tsrank(H, 5), 5), 3).max()
    a["A033"] = rank(-(1 - O / C))
    a["A034"] = rank((1 - rank(std(ret, 2) / std(ret, 5))) + (1 - rank(d1)))
    a["A038"] = -rank(tsrank(C, 10)) * rank(C / O)
    a["A040"] = -rank(std(H, 10)) * corr(H, V, 10)
    a["A041"] = np.sqrt(H * L) - vwap
    a["A042"] = rank(vwap - C) / rank(vwap + C)
    a["A044"] = -corr(H, rank(V), 5)
    a["A053"] = -delta(((C - L) - (H - C)) / (C - L).replace(0, np.nan), 9)
    a["A054"] = -((L - C) * O ** 5) / ((L - H).replace(0, np.nan) * C ** 5)
    a["A101"] = (C - O) / (H - L + 0.001)
    # price-level alphas (A041, A054) are made scale-free so stocks are comparable
    a["A041"] = a["A041"] / C
    yield from a.items()
