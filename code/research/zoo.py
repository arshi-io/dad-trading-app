"""Model Zoo: published models side by side, what they have in common, and a blend of them.

Models (each trained walk-forward, refit every quarter on outcomes settled before it):
  Qlib Alpha158 + gradient-boosted trees      (Microsoft Qlib benchmark recipe)
  WorldQuant 101 alphas (25, OHLCV-only) + trees (Kakushadze 2016)
  NN3 neural net on stock characteristics      (Gu, Kelly & Xiu, RFS 2020)
  Linear model on ranked characteristics       (Fama-MacBeth style)
  Everything + trees
and classic one-signal factors, used as published (no fitting): momentum 12-1, short-term
reversal, 52-week high, low volatility, illiquidity, MAX/lottery, Minervini trend.

Every input is ranked against that day's stocks; the target is the rank of the next move (Qlib's
CSRankNorm). Scored by rank IC on 2021-23 (choosing) and 2024+ (locked). Patterns: which signal
families each model leans on (shuffle a family within each day, measure the IC drop), how much
the models agree, and which classic factor each model's picks resemble most.

Run:  python -m code.research.zoo --horizon 1      (writes code/data/zoo_h1.json; laptop only)
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline

from code.research import model_search as ms
from code.research import signals_lib as sl
from code.research import updown as u

logger = logging.getLogger(__name__)
DATA = Path(__file__).resolve().parents[1] / "data"
SINCE = "2016-01-01"
START = "2021-01-01"
TRAIN_ROWS = 150_000

CHARS = [c for c in u.FEATURES + u.EXTRA + u.TREND]
CLASSIC = {   # name: (column, sign, paper)
    "Momentum 12-1": ("mom_12_1", 1, "Jegadeesh & Titman 1993"),
    "Short-term reversal": ("r5", -1, "Jegadeesh 1990; Lehmann 1990"),
    "52-week high": ("hi52", 1, "George & Hwang 2004"),
    "Low volatility": ("vol20", -1, "Ang, Hodrick, Xing & Zhang 2006"),
    "Illiquidity": ("amihud", 1, "Amihud 2002"),
    "MAX / lottery": ("maxret20", -1, "Bali, Cakici & Whitelaw 2011"),
    "Minervini trend": ("tt_score", 1, "Minervini 2013 trend template"),
}
FAMILY_NOTE = {
    "short-term": "Short-term reversal: last week's losers bounce (Jegadeesh 1990)",
    "medium-term": "Momentum and trend: 1-12 month winners keep winning (Jegadeesh & Titman 1993)",
    "volatility": "Low volatility / lottery: calm stocks beat jumpy ones (Ang et al. 2006, Bali et al. 2011)",
    "volume": "Volume and liquidity: how trading activity moves with price (Amihud 2002; Llorente et al. 2002)",
    "candle": "Candle shape: where the close sits in the day's range (Qlib kbar)",
    "gap": "Overnight gap: opening jumps partly reverse",
    "market": "Market state: breadth and trend of the whole market",
}


def family(col: str) -> str:
    c = col.removeprefix("x_")
    if c in ("breadth", "mkt_ret_20", "mkt_above_50"):
        return "market"
    if c in ("gap", "A020", "OPEN0"):
        return "gap"
    if c.startswith(("K", "HIGH0", "LOW0")) or c in ("close_loc", "A018", "A033", "A038", "A053", "A054", "A101"):
        return "candle"
    if c.startswith(("CORR", "CORD", "VMA", "VSTD", "WVMA", "VSUM", "VWAP0")) or c in (
            "vol_rel", "deliv_rel", "amihud", "A002", "A003", "A006", "A012", "A013", "A014", "A015", "A016",
            "A022", "A026", "A040", "A041", "A042", "A044"):
        return "volume"
    if c.startswith(("STD", "MAX", "MIN", "RSV", "IMAX", "IMIN", "IMXD")) or c in ("vol20", "atr_pct", "maxret20"):
        return "volatility"
    digits = "".join(ch for ch in c if ch.isdigit())
    if c.startswith(("ROC", "MA", "BETA", "RSQR", "RESI", "QTL", "RANK", "CNT", "SUM")) and digits:
        return "short-term" if int(digits) <= 10 else "medium-term"
    if c in ("r5", "day_ret", "z20", "A004", "A009", "A023", "A034"):
        return "short-term"
    return "medium-term"                                   # r20, r60, rs, mom_12_1, hi52, ext50/200, tt_score


# ---------- data ----------------------------------------------------------------------------

def build(horizon: int, since: str = SINCE, raw: Optional[pd.DataFrame] = None) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Long frame of liquid stock-days: characteristics, Alpha158, Alpha101 (all ranked within the
    day, except market-wide columns), the outcome and its settle date."""
    if raw is None:
        from code.research import bhavcopy
        raw = bhavcopy.load(since=since)
    w = u.panel(raw)
    base = u.features(w, horizon=horizon)
    liquid = base.set_index(["date", "symbol"]).index
    df = base.set_index(["date", "symbol"])
    names = {"alpha158": [], "alpha101": []}
    for lib, gen in (("alpha158", sl.alpha158(w)), ("alpha101", sl.alpha101(w))):
        for name, frame in gen:
            s = frame.replace([np.inf, -np.inf], np.nan).astype("float32").stack()
            df[name] = s.reindex(liquid).to_numpy()
            names[lib].append(name)
    df = df.reset_index()
    stock_cols = [c for c in CHARS + names["alpha158"] + names["alpha101"] if c not in ms.MARKET]
    ranked = df.groupby("date")[stock_cols].rank(pct=True).astype("float32")
    df[stock_cols] = ranked.to_numpy()
    df["rankret"] = df.groupby("date")["fwd"].rank(pct=True)
    return df, names["alpha158"], names["alpha101"]


# ---------- models --------------------------------------------------------------------------

def _hgb():
    return HistGradientBoostingRegressor(max_depth=6, learning_rate=0.05, max_iter=300, min_samples_leaf=500,
                                         l2_regularization=1.0, random_state=0)


def _nn3():
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.5),
                         MLPRegressor(hidden_layer_sizes=(32, 16, 8), alpha=1e-4, learning_rate_init=1e-3, batch_size=2048,
                                      max_iter=30, early_stopping=True, n_iter_no_change=4, random_state=0))


def _linear():
    return make_pipeline(SimpleImputer(strategy="constant", fill_value=0.5), Ridge(alpha=10.0))


def zoo(a158: List[str], a101: List[str]) -> List[Tuple[str, List[str], Callable, str]]:
    chars = CHARS
    return [("Qlib Alpha158 + trees", a158, _hgb, "Microsoft Qlib benchmark (Alpha158 + LightGBM)"),
            ("WorldQuant 101 alphas + trees", a101, _hgb, "Kakushadze 2016, 101 Formulaic Alphas"),
            ("NN3 neural net", chars, _nn3, "Gu, Kelly & Xiu 2020, Review of Financial Studies"),
            ("Linear (Fama-MacBeth style)", chars, _linear, "Fama & MacBeth 1973; Lewellen 2015"),
            ("Everything + trees", chars + a158 + a101, _hgb, "All inputs above, one tree model")]


def walk(df: pd.DataFrame, cols: List[str], make: Callable, start: str = START,
         train_rows: int = TRAIN_ROWS, keep_at: Optional[str] = None) -> Tuple[pd.Series, Any]:
    """Out-of-sample scores from `start` on, refit quarterly on outcomes settled before the quarter.
    Returns the scores and the model fitted at `keep_at` (for the pattern analysis)."""
    rng = np.random.default_rng(0)
    lab = df[df["rankret"].notna()]
    last = df["date"].max() + pd.Timedelta(days=1)
    quarters = list(pd.date_range(start, last, freq="QS"))
    out, kept = [], None
    for i, q in enumerate(quarters):
        end = quarters[i + 1] if i + 1 < len(quarters) else last
        test = df[(df["date"] >= q) & (df["date"] < end)]
        train = lab[lab["label_date"] < q]
        if test.empty or len(train) < 1000:
            continue
        if len(train) > train_rows:
            train = train.iloc[rng.choice(len(train), train_rows, replace=False)]
        use = [c for c in cols if train[c].notna().any()]         # a long-window signal can be blank early on
        m = make().fit(train[use], train["rankret"])
        m.used_cols = use
        out.append(pd.Series(m.predict(test[use]), index=test.index))
        if keep_at and q == pd.Timestamp(keep_at):
            kept = m
    return (pd.concat(out) if out else pd.Series(dtype=float)), kept


def score_frame(df: pd.DataFrame, score: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"date": df.loc[score.index, "date"], "fwd": df.loc[score.index, "fwd"], "score": score})


def split_eval(df: pd.DataFrame, score: pd.Series, horizon: int) -> Dict[str, Any]:
    f = score_frame(df, score)
    v = f[(f["date"] >= ms.VALID[0]) & (f["date"] < ms.VALID[1])]
    lk = f[f["date"] >= ms.LOCKED]
    return {"valid": ms.evaluate(v, horizon), "locked": ms.evaluate(lk, horizon)}


def family_importance(df: pd.DataFrame, m: Any, cols: List[str], q: str, horizon: int) -> Dict[str, float]:
    """IC drop when one family's inputs are shuffled among that day's stocks (one quarter's rows)."""
    qs, qe = pd.Timestamp(q), pd.Timestamp(q) + pd.offsets.QuarterBegin(1)
    rows = df[(df["date"] >= qs) & (df["date"] < qe) & df["fwd"].notna()]
    if m is None or rows.empty:
        return {}
    cols = m.used_cols
    base = ms.evaluate(pd.DataFrame({"date": rows["date"], "fwd": rows["fwd"], "score": m.predict(rows[cols])}), horizon)["ic"]
    rng = np.random.default_rng(1)
    out = {}
    fams: Dict[str, List[str]] = {}
    for c in cols:
        fams.setdefault(family(c), []).append(c)
    for fam, fc in fams.items():
        X = rows[cols].copy()
        for c in fc:                                               # shuffle within each day
            X[c] = X.groupby(rows["date"])[c].transform(lambda s: s.sample(frac=1.0, random_state=int(rng.integers(1e9))).to_numpy())
        ic = ms.evaluate(pd.DataFrame({"date": rows["date"], "fwd": rows["fwd"], "score": m.predict(X)}), horizon)["ic"]
        out[fam] = round(base - ic, 4)
    return out


def agreement(df: pd.DataFrame, scores: Dict[str, pd.Series], days: int = 120) -> Dict[str, Dict[str, float]]:
    """Average same-day rank correlation between every pair of models (on a sample of days)."""
    idx = pd.concat([s.rename(k) for k, s in scores.items()], axis=1).dropna()
    idx["date"] = df.loc[idx.index, "date"]
    pick = idx["date"].drop_duplicates().sample(min(days, idx["date"].nunique()), random_state=0)
    sub = idx[idx["date"].isin(pick)]
    names = list(scores)
    mat = {a: {} for a in names}
    corr = sub.groupby("date")[names].corr(method="spearman")
    for a in names:
        for b in names:
            mat[a][b] = round(float(corr.xs(a, level=1)[b].mean()), 3)
    return mat


def run(horizon: int, log: Callable[[str], None] = logger.info, df: Optional[pd.DataFrame] = None,
        a158: Optional[List[str]] = None, a101: Optional[List[str]] = None, train_rows: int = TRAIN_ROWS,
        start: str = START, keep_at: str = "2023-10-01") -> Dict[str, Any]:
    if df is None:
        df, a158, a101 = build(horizon)
    log(f"{len(df):,} stock-days, {len(a158)} Alpha158 + {len(a101)} Alpha101 + {len(CHARS)} characteristics")
    scores: Dict[str, pd.Series] = {}
    models: List[Dict[str, Any]] = []
    for name, cols, make, paper in zoo(a158, a101):
        s, m = walk(df, cols, make, start, train_rows, keep_at)
        scores[name] = s
        ev = split_eval(df, s, horizon)
        models.append({"name": name, "kind": "fitted", "paper": paper, "inputs": len(cols), **ev,
                       "families": family_importance(df, m, cols, keep_at, horizon)})
        log(f"{name}: IC choose {ev['valid']['ic']:+.4f}  locked {ev['locked']['ic']:+.4f}")
    test_rows = df[df["date"] >= start]
    for name, (col, sign, paper) in CLASSIC.items():
        s = (sign * test_rows[col]).astype(float)
        scores[name] = s
        ev = split_eval(df, s, horizon)
        models.append({"name": name, "kind": "classic", "paper": paper, "inputs": 1, **ev, "families": {family(col): 1.0}})
        log(f"{name}: IC choose {ev['valid']['ic']:+.4f}  locked {ev['locked']['ic']:+.4f}")

    # which classic factor each model resembles most (same-day rank correlation)
    agree = agreement(df, scores)
    for mdl in models:
        mdl["matches"] = {c: agree[mdl["name"]][c] for c in CLASSIC if c != mdl["name"]}

    # blends, rules fixed before 2024+ is looked at: weights = each model's positive IC on 2021-23
    w = {m["name"]: max(m["valid"]["ic"], 0.0) for m in models}
    tot = sum(w.values()) or 1.0
    ranks = {k: s.groupby(df.loc[s.index, "date"]).rank(pct=True) for k, s in scores.items()}
    common = pd.concat(ranks.values(), axis=1).dropna().index
    weighted = sum(ranks[k].reindex(common) * (w[k] / tot) for k in ranks)
    equal = sum(ranks[k].reindex(common) for k in ranks) / len(ranks)
    blends = [{"name": "Zoo blend, weighted by 2021-23 IC", "weights": {k: round(v / tot, 3) for k, v in w.items() if v > 0},
               **split_eval(df, weighted, horizon)},
              {"name": "Zoo blend, equal weights", **split_eval(df, equal, horizon)}]
    for b in blends:
        log(f"{b['name']}: IC choose {b['valid']['ic']:+.4f}  locked {b['locked']['ic']:+.4f}")
    fam_tot: Dict[str, float] = {}
    for m in models:
        if m["kind"] == "fitted":
            pos = {k: max(v, 0.0) for k, v in m["families"].items()}
            s = sum(pos.values()) or 1.0
            for k, v in pos.items():
                fam_tot[k] = fam_tot.get(k, 0.0) + v / s
    n_fit = sum(m["kind"] == "fitted" for m in models) or 1
    common_pattern = sorted(({"family": k, "share": round(v / n_fit, 3), "note": FAMILY_NOTE.get(k, "")}
                             for k, v in fam_tot.items()), key=lambda r: -r["share"])
    return {"generated": date.today().isoformat(), "horizon": horizon, "valid": list(ms.VALID), "locked_from": ms.LOCKED,
            "models": models, "blends": blends, "agreement": agree, "common_pattern": common_pattern,
            "scores": scores, "weighted": weighted}


def main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    h = int(sys.argv[sys.argv.index("--horizon") + 1]) if "--horizon" in sys.argv else 1
    res = run(h)
    res.pop("scores"), res.pop("weighted")
    (DATA / f"zoo_h{h}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("common pattern:", [(p["family"], p["share"]) for p in res["common_pattern"]])


if __name__ == "__main__":
    main()
