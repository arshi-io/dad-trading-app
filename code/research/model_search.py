"""Model search: rounds of model combinations for the 5-session stock ranking, best one kept.

Each try is one combination of model x target x inputs x scaling. It is trained walk-forward
(refit every quarter, only on outcomes settled before the quarter) and scored by rank IC on the
choosing period, 2021-2023. Each round tries every one-step change from the current best; the best
of the round is kept if it gains more than MIN_GAIN. Then blends of the top combinations are tried.
The winner and today's live model are then run once on 2024+, which no try has seen, and the result
is appended to a history so every look at 2024+ is on record.

Run:  python -m code.research.model_search      (laptop only; about 1-2 min per try)
"""
from __future__ import annotations

import json
import logging
from datetime import date
from itertools import product
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from code.research import updown as u

logger = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parents[1] / "data"
OUT = DATA / "model_search.json"
FEATURE_CACHE = DATA / "history" / "updown_features.parquet"
REPLAY = DATA / "history" / "replay.parquet"

VALID = ("2021-01-01", "2024-01-01")      # choosing period
LOCKED = "2024-01-01"                     # looked at once per search, never used to choose
MIN_GAIN = 0.002
MAX_TRIES = 30
TRAIN_ROWS = 200_000
MARKET = ["breadth", "mkt_ret_20", "mkt_above_50"]   # same for every stock on a day: never ranked

SPACE: Dict[str, List[str]] = {
    "model": ["hgb3", "hgb6", "linear"],
    "target": ["up", "beat", "rankret"],          # up: rose; beat: beat that day's median stock; rankret: rank of the move
    "inputs": ["base", "base+quant", "base+quant+trend"],
    "scaling": ["raw", "xsrank"],                  # xsrank: each input as a rank among that day's stocks
}
BASELINE = {"model": "hgb3", "target": "up", "inputs": "base", "scaling": "raw"}   # the live model


def paths(horizon: int = u.HORIZON) -> Tuple[Path, Path]:
    """Search result and feature cache for a horizon (5 sessions keeps the original names)."""
    if horizon == u.HORIZON:
        return OUT, FEATURE_CACHE
    return DATA / f"model_search_h{horizon}.json", DATA / "history" / f"updown_features_h{horizon}.parquet"


def key(spec: Dict[str, Any]) -> str:
    if "blend" in spec:
        return "blend(" + " + ".join(spec["blend"]) + ")"
    return " | ".join(spec[k] for k in SPACE)


# ---------- data ----------------------------------------------------------------------------

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Add the extra targets and the ranked copies of every stock input."""
    df = df.copy()
    day = df.groupby("date")["fwd"]
    df["beat"] = (df["fwd"] > day.transform("median")).astype(float).where(df["fwd"].notna())
    df["rankret"] = day.rank(pct=True)
    stock_cols = [c for c in u.FEATURES + u.EXTRA + u.TREND if c not in MARKET and c in df.columns]
    ranked = df.groupby("date")[stock_cols].rank(pct=True).astype("float32")
    df[["x_" + c for c in stock_cols]] = ranked.to_numpy()
    return df


def load_features(horizon: int = u.HORIZON) -> pd.DataFrame:
    cache = paths(horizon)[1]
    if cache.exists():
        df = pd.read_parquet(cache)
        if all(c in df.columns for c in u.TREND):
            return df
    from code.research import bhavcopy
    df = prepare(u.features(u.panel(bhavcopy.load()), horizon=horizon))
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache)
    return df


def columns(spec: Dict[str, str]) -> List[str]:
    cols = u.FEATURES + (u.EXTRA if "quant" in spec["inputs"] else []) + (u.TREND if "trend" in spec["inputs"] else [])
    if spec["scaling"] == "xsrank":
        cols = [c if c in MARKET else "x_" + c for c in cols]
    return cols


# ---------- models --------------------------------------------------------------------------

def estimator(spec: Dict[str, str]):
    classify = spec["target"] != "rankret"
    if spec["model"] == "linear":
        last = LogisticRegression(C=0.1, max_iter=500) if classify else Ridge(alpha=10.0)
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), last)
    deep = spec["model"] == "hgb6"
    kw = dict(max_depth=6 if deep else 3, learning_rate=0.05, max_iter=300 if deep else 200, min_samples_leaf=500,
              l2_regularization=1.0, random_state=0)
    return HistGradientBoostingClassifier(**kw) if classify else HistGradientBoostingRegressor(**kw)


def _score(m, X: pd.DataFrame) -> np.ndarray:
    return m.predict_proba(X)[:, 1] if hasattr(m, "predict_proba") else m.predict(X)


def walk(df: pd.DataFrame, spec: Dict[str, str], start: str, end: Optional[str] = None,
         train_rows: int = TRAIN_ROWS, make: Callable = estimator) -> pd.DataFrame:
    """Out-of-sample scores for every row dated [start, end), refitting each quarter on outcomes
    settled before that quarter began."""
    rng = np.random.default_rng(0)
    target, cols = spec["target"], columns(spec)
    last = df["date"].max() + pd.Timedelta(days=1)
    stop = min(pd.Timestamp(end), last) if end else last
    labelled = df[df[target].notna()]
    quarters = list(pd.date_range(pd.Timestamp(start), stop, freq="QS")) or [pd.Timestamp(start)]
    if quarters[0] > pd.Timestamp(start):
        quarters.insert(0, pd.Timestamp(start))
    out = []
    for i, q in enumerate(quarters):
        q_end = quarters[i + 1] if i + 1 < len(quarters) else stop
        test = df[(df["date"] >= q) & (df["date"] < q_end)]
        train = labelled[labelled["label_date"] < q]
        if test.empty or train[target].nunique() < 2:
            continue
        if len(train) > train_rows:
            train = train.iloc[rng.choice(len(train), train_rows, replace=False)]
        y = train[target] if target == "rankret" else train[target].astype(int)
        m = make(spec).fit(train[cols], y)
        out.append(pd.DataFrame({"date": test["date"].to_numpy(), "symbol": test["symbol"].to_numpy(),
                                 "fwd": test["fwd"].to_numpy(), "score": _score(m, test[cols])}, index=test.index))
    return pd.concat(out) if out else pd.DataFrame(columns=["date", "symbol", "fwd", "score"])


def evaluate(pred: pd.DataFrame, horizon: int = u.HORIZON) -> Dict[str, Any]:
    """Rank IC per day (score rank vs the move's rank), and top-minus-bottom tenth spread. The t-stat
    counts one independent observation per `horizon` days, since outcomes overlap."""
    p = pred[pred["fwd"].notna()]
    p = p[p.groupby("date")["fwd"].transform("size") >= 30]
    if p.empty:
        return {"ic": 0.0, "t": 0.0, "ic_days_positive": 0.0, "spread": 0.0, "days": 0}
    r = p.groupby("date")["score"].rank(pct=True)
    p = p.assign(r=r, f=p.groupby("date")["fwd"].rank(pct=True), dec=np.minimum((r * 10).astype(int), 9))
    ic = p.groupby("date")[["r", "f"]].corr().xs("r", level=1)["f"].dropna()
    dec = p.groupby(["date", "dec"])["fwd"].mean().unstack()
    spread = (dec.get(9) - dec.get(0)).dropna() if 9 in dec and 0 in dec else pd.Series(dtype=float)
    n = len(ic)
    return {"ic": round(float(ic.mean()), 4) if n else 0.0,
            "t": round(float(ic.mean() / ic.std() * np.sqrt(n / horizon)), 1) if n > 2 and ic.std() > 0 else 0.0,
            "ic_days_positive": round(float((ic > 0).mean()), 4) if n else 0.0,
            "spread": round(float(spread.mean()), 5) if len(spread) else 0.0, "days": n}


def blend(preds: List[pd.DataFrame]) -> pd.DataFrame:
    """Average of each member's same-day rank: scales differ between models, ranks don't."""
    ranks = [p.groupby("date")["score"].rank(pct=True) for p in preds]
    return preds[0].assign(score=sum(ranks) / len(ranks))


# ---------- search --------------------------------------------------------------------------

def neighbours(spec: Dict[str, str]) -> List[Dict[str, str]]:
    return [{**spec, k: v} for k in SPACE for v in SPACE[k] if v != spec[k]]


def search(df: pd.DataFrame, valid: Tuple[str, str] = VALID, max_tries: int = MAX_TRIES, min_gain: float = MIN_GAIN,
           train_rows: int = TRAIN_ROWS, log: Callable[[str], None] = logger.info,
           horizon: int = u.HORIZON) -> Dict[str, Any]:
    tried: Dict[str, Dict[str, Any]] = {}
    preds: Dict[str, pd.DataFrame] = {}

    def run(spec: Dict[str, Any], rnd: int) -> Dict[str, Any]:
        k = key(spec)
        if "blend" in spec:
            preds[k] = blend([preds[m] for m in spec["blend"]])
        else:
            preds[k] = walk(df, spec, valid[0], valid[1], train_rows)
        tried[k] = {"key": k, "spec": spec, "round": rnd, **evaluate(preds[k], horizon)}
        log(f"round {rnd} try {len(tried)}: {k}  IC {tried[k]['ic']:+.4f}  spread {tried[k]['spread']:+.2%}")
        return tried[k]

    best = run(BASELINE, 0)
    rounds = [{"round": 0, "best": best["key"], "ic": best["ic"]}]
    rnd = 0
    while len(tried) < max_tries:
        rnd += 1
        todo = [s for s in neighbours(best["spec"]) if key(s) not in tried][: max_tries - len(tried)]
        if not todo:
            break
        top = max((run(s, rnd) for s in todo), key=lambda r: r["ic"])
        improved = top["ic"] > best["ic"] + min_gain
        if improved:
            best = top
        rounds.append({"round": rnd, "best": best["key"], "ic": best["ic"], "improved": improved})
        if not improved:
            break

    singles = sorted((r for r in tried.values() if "blend" not in r["spec"]), key=lambda r: -r["ic"])
    rnd += 1
    for n in (2, 3, 5):
        if len(singles) >= n and len(tried) < max_tries + 3:
            b = run({"blend": [r["key"] for r in singles[:n]]}, rnd)
            if b["ic"] > best["ic"] + min_gain:
                best = b
    rounds.append({"round": rnd, "best": best["key"], "ic": best["ic"], "blends": True})
    return {"tried": sorted(tried.values(), key=lambda r: -r["ic"]), "rounds": rounds, "best": best,
            "baseline": tried[key(BASELINE)], "preds": preds}


def predict(df: pd.DataFrame, spec: Dict[str, Any], start: str, train_rows: int = TRAIN_ROWS) -> pd.DataFrame:
    if "blend" in spec:
        return blend([predict(df, _spec(k), start, train_rows) for k in spec["blend"]])
    return walk(df, spec, start, None, train_rows)


def _spec(k: str) -> Dict[str, str]:
    return dict(zip(SPACE, k.split(" | ")))


def locked_test(df: pd.DataFrame, best: Dict[str, Any], locked: str = LOCKED,
                train_rows: int = TRAIN_ROWS, horizon: int = u.HORIZON) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Winner and live model, once each, on the period no try has seen."""
    pb = predict(df, best["spec"], locked, train_rows)
    pl = pb if best["key"] == key(BASELINE) else predict(df, BASELINE, locked, train_rows)
    res = {"from": locked, "best": {"key": best["key"], **evaluate(pb, horizon)},
           "baseline": {"key": key(BASELINE), **evaluate(pl, horizon)}}
    return res, pb


# ---------- replay (read by the dashboard) ---------------------------------------------------

def replay_load(path: Path = REPLAY) -> pd.DataFrame:
    from code.research.setups import is_equity
    df = pd.read_parquet(path)
    df = df[df["fwd"].notna()]
    return df[df["symbol"].map(is_equity)]


def replay_weeks(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Per day: did the model's top tenth beat its bottom tenth over the next 5 sessions?"""
    d = df[df[model].notna()]
    d = d.assign(dec=np.minimum((d[model] * 10).astype(int), 9))
    g = d.groupby(["date", "dec"])["fwd"].mean().unstack()
    out = pd.DataFrame({"top": g.get(9), "bottom": g.get(0), "all": d.groupby("date")["fwd"].mean()}).dropna()
    return out.assign(right=out["top"] > out["bottom"])


def replay_day(df: pd.DataFrame, weeks: pd.DataFrame, model: str, day: str, picks: int = 10) -> Dict[str, Any]:
    days = weeks.index
    want = pd.Timestamp(day)
    d0 = days[days <= want].max() if (days <= want).any() else days.min()
    rows = df[(df["date"] == d0) & df[model].notna()].sort_values(model)
    med = rows["fwd"].median()
    top, bottom = rows[rows[model] >= 0.9], rows[rows[model] < 0.1]
    w = weeks.loc[d0]
    pick = lambda r: [{"symbol": s, "rank": round(float(k), 3), "move": round(float(f), 4),
                       "vs_median": round(float(f - med), 4)} for s, k, f in zip(r["symbol"], r[model], r["fwd"])]
    i = days.get_loc(d0)
    return {
        "date": d0.strftime("%Y-%m-%d"), "prev": days[max(i - 5, 0)].strftime("%Y-%m-%d"),
        "next": days[min(i + 5, len(days) - 1)].strftime("%Y-%m-%d"),
        "first": days.min().strftime("%Y-%m-%d"), "last": days.max().strftime("%Y-%m-%d"),
        "n": int(len(rows)), "top": round(float(w["top"]), 4), "bottom": round(float(w["bottom"]), 4),
        "all": round(float(w["all"]), 4), "median": round(float(med), 4), "right": bool(w["right"]),
        "ic": round(float(rows[model].corr(rows["fwd"].rank())), 4) if len(rows) > 2 else None,
        "top_beat_median": round(float((top["fwd"] > med).mean()), 4) if len(top) else None,
        "bottom_lagged_median": round(float((bottom["fwd"] < med).mean()), 4) if len(bottom) else None,
        "best_picks": pick(rows.tail(picks).iloc[::-1]), "worst_picks": pick(rows.head(picks)),
        "record": {"weeks": int(len(weeks)), "right": int(weeks["right"].sum()),
                   "avg_top": round(float(weeks["top"].mean()), 5), "avg_bottom": round(float(weeks["bottom"].mean()), 5),
                   "upto_weeks": int((days <= d0).sum()), "upto_right": int(weeks.loc[:d0, "right"].sum())},
    }


# ---------- outputs -------------------------------------------------------------------------

def write_replay(best_pred: pd.DataFrame, scored: Optional[pd.DataFrame]) -> None:
    """One row per stock-day: the 5-session move and each model's same-day rank (0..1)."""
    b = best_pred.assign(best=best_pred.groupby("date")["score"].rank(pct=True))[["date", "symbol", "fwd", "best"]]
    if scored is not None:
        c = scored.assign(current=scored.groupby("date")["p"].rank(pct=True))[["date", "symbol", "fwd", "current"]]
        b = c.merge(b.drop(columns="fwd"), on=["date", "symbol"], how="outer")
    b["date"] = pd.to_datetime(b["date"])
    REPLAY.parent.mkdir(parents=True, exist_ok=True)
    b.astype({c: "float32" for c in b.columns if c not in ("date", "symbol")}).to_parquet(REPLAY, index=False)


def add_rank_column(pred: pd.DataFrame, name: str) -> None:
    """Add (or replace) one model's same-day rank in the replay file, keeping everything else."""
    r = pred.assign(**{name: pred.groupby("date")["score"].rank(pct=True)})[["date", "symbol", name]]
    r["date"] = pd.to_datetime(r["date"])
    base = pd.read_parquet(REPLAY).drop(columns=[name], errors="ignore")
    out = base.merge(r, on=["date", "symbol"], how="outer")
    out[name] = out[name].astype("float32")
    out.to_parquet(REPLAY, index=False)


def save(result: Dict[str, Any], locked: Dict[str, Any], valid: Tuple[str, str] = VALID,
         horizon: int = u.HORIZON) -> Dict[str, Any]:
    out = paths(horizon)[0]
    prev = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
    history = prev.get("locked_history", []) + [{"date": date.today().isoformat(), **locked}]
    clean = lambda r: {k: v for k, v in r.items() if k != "spec"}
    data = {"generated": date.today().isoformat(), "horizon": horizon, "valid": list(valid), "min_gain": MIN_GAIN,
            "tried": [clean(r) for r in result["tried"]], "rounds": result["rounds"],
            "best": clean(result["best"]), "baseline": clean(result["baseline"]),
            "locked": locked, "locked_history": history}
    out.write_text(json.dumps(data, indent=1), encoding="utf-8")
    return data


def main() -> None:
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    horizon = int(sys.argv[sys.argv.index("--horizon") + 1]) if "--horizon" in sys.argv else u.HORIZON
    df = load_features(horizon)
    logger.info("%d stock-days loaded, horizon %d sessions", len(df), horizon)
    result = search(df, horizon=horizon)
    best = result["best"]
    # replay: the winner from the start of the choosing period on (2021-23 helped choose it; 2024+ did not)
    locked, pb = locked_test(df, best, horizon=horizon)
    choose = result["preds"][best["key"]]
    if horizon == u.HORIZON:
        scored = pd.read_parquet(u.SCORED) if u.SCORED.exists() else None
        write_replay(pd.concat([choose, pb]), scored)
    else:
        add_rank_column(pd.concat([choose, pb]), f"best_h{horizon}")
    d = save(result, locked, horizon=horizon)
    print(f"\nbest: {d['best']['key']}  IC {d['best']['ic']:+.4f} on {d['valid'][0][:4]}-2023 "
          f"(live model {d['baseline']['ic']:+.4f})")
    print(f"2024+ locked test: best IC {locked['best']['ic']:+.4f} spread {locked['best']['spread']:+.2%} | "
          f"live model IC {locked['baseline']['ic']:+.4f} spread {locked['baseline']['spread']:+.2%}")


if __name__ == "__main__":
    main()
