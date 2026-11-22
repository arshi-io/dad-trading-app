"""Chance a stock is higher 5 sessions from now: a walk-forward score with its own track record.

Question asked each day, at the close: if I buy at tomorrow's open, is the close 5 sessions after
today above that price? Features use only what is known at today's close. One model is trained on
all liquid NSE stocks together (survivorship-free bhavcopy 2011+) and refitted every month; each
month is scored by a model whose training labels were all settled before that month began. So every
number in the track record comes from days the model had not seen.

Run:  python -m code.research.updown        (writes code/data/updown.json; ~15 min, laptop only)
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from code.research import bhavcopy
from code.research.setups import MIN_PRICE, MIN_TURNOVER, is_equity

logger = logging.getLogger(__name__)

HORIZON = 5
FIRST_TEST = "2015-01-01"
TRAIN_ROWS = 400_000          # random sample of the past per refit; more adds time, not skill
NO_EDGE_AUC = 0.53            # at or below this on unseen data the score is labelled NO EDGE
BUCKETS = [0.0, 0.45, 0.50, 0.55, 0.60, 1.0]
OUT = Path(__file__).resolve().parents[1] / "data" / "updown.json"
SCORED = OUT.parent / "history" / "updown_scored.parquet"     # every scored stock-day, for the Replay page
FEATURES = ["r5", "r20", "r60", "rs", "z20", "ext50", "ext200", "vol20", "atr_pct", "vol_rel", "deliv_rel",
            "day_ret", "close_loc", "breadth", "mkt_ret_20", "mkt_above_50"]
# classic quant factors, tried by code/research/model_search.py: 12-1 momentum, distance from the
# 52-week high, lottery (max daily jump), Amihud illiquidity, overnight gap
EXTRA = ["mom_12_1", "hi52", "maxret20", "amihud", "gap"]
# Minervini trend template: how many of its 8 rules a stock meets that day (thresholds from
# code/signals/minervini/trend_template.py; RS rule uses the 120-day RS rank here)
TREND = ["tt_score"]


# ---------- data ----------------------------------------------------------------------------

def panel(raw: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Wide date x symbol frames, split/bonus-adjusted with past information only.

    NSE's bhavcopy is unadjusted (prev_close too), so a split shows up as an overnight gap. Most
    stocks have a 20% circuit, so a gap down of more than 25% is treated as a corporate action and
    removed; the day's own open-to-close move is kept (same rule as gatekeeper/gate.py).
    """
    raw = raw[raw["series"].isin(["EQ", "BE", "BZ"])]
    raw = raw.sort_values(["date", "symbol", "series"]).drop_duplicates(["date", "symbol"], keep="first")
    raw = raw[raw["symbol"].map(is_equity)]
    w = {c: raw.pivot(index="date", columns="symbol", values=c).astype("float64")
         for c in ("open", "high", "low", "close", "volume", "turnover", "delivery_qty")}
    gap = w["open"] / w["close"].ffill().shift(1)
    cum = gap.where(gap < 0.75, 1.0).fillna(1.0).cumprod()
    for c in ("open", "high", "low", "close"):
        w[c] = w[c] / cum
    w["cum"] = cum                       # adjusted x cum = the price actually traded that day
    return w


def trend_score(C: pd.DataFrame, rs: pd.DataFrame) -> pd.DataFrame:
    """Count of Minervini trend-template rules met (0-8), all from past and current closes."""
    from code.signals.minervini import trend_template as tt
    s50, s150, s200 = (C.rolling(n, min_periods=n).mean() for n in (50, 150, 200))
    lo = C.rolling(tt.TRADING_DAYS_52WK, min_periods=200).min()
    hi = C.rolling(tt.TRADING_DAYS_52WK, min_periods=200).max()
    rules = [(C > s150) & (C > s200), s150 > s200, s200 > s200.shift(tt.SMA_TRENDING_LOOKBACK_DAYS),
             (s50 > s150) & (s50 > s200), C > s50, C >= tt.LOW_BAND_MULTIPLIER * lo,
             C >= tt.HIGH_BAND_MULTIPLIER * hi, rs >= tt.RS_RANK_THRESHOLD / 100]
    return sum(r.astype(float) for r in rules).where(s200.notna())


def features(w: Dict[str, pd.DataFrame], horizon: int = HORIZON) -> pd.DataFrame:
    """One row per liquid stock-day: features known at that close, the outcome `horizon` sessions
    later, and the date that outcome becomes known (training may only use rows settled before)."""
    C, O, H, L, V = w["close"], w["open"], w["high"], w["low"], w["volume"]
    dates = C.index
    ret = C.pct_change(fill_method=None)
    liquid = (w["turnover"].rolling(50, min_periods=30).median() >= MIN_TURNOVER) & (C >= MIN_PRICE)
    sma20, sma50, sma200 = (C.rolling(n, min_periods=n).mean() for n in (20, 50, 200))
    tr = np.maximum(H - L, np.maximum((H - C.shift()).abs(), (L - C.shift()).abs()))
    breadth = (C > sma200).where(liquid).mean(axis=1)
    mkt = (1 + ret.where(liquid).mean(axis=1).fillna(0)).cumprod()       # equal-weight liquid market
    f = {
        "r5": C.pct_change(5, fill_method=None), "r20": C.pct_change(20, fill_method=None),
        "r60": C.pct_change(60, fill_method=None),
        "rs": C.pct_change(120, fill_method=None).where(liquid).rank(axis=1, pct=True),
        "z20": (C - sma20) / C.rolling(20, min_periods=20).std(),
        "ext50": C / sma50 - 1, "ext200": C / sma200 - 1,
        "vol20": ret.rolling(20, min_periods=20).std(),
        "atr_pct": tr.rolling(14, min_periods=14).mean() / C,
        "vol_rel": V / V.rolling(50, min_periods=30).mean().shift(1),
        "deliv_rel": (w["delivery_qty"] / V) / (w["delivery_qty"] / V).rolling(50, min_periods=30).mean().shift(1),
        "day_ret": ret, "close_loc": (C - L) / (H - L).replace(0, np.nan),
        "mom_12_1": C.shift(21) / C.shift(252) - 1,
        "hi52": C / C.rolling(252, min_periods=200).max() - 1,
        "maxret20": ret.rolling(20, min_periods=20).max(),
        "amihud": (ret.abs() / w["turnover"]).rolling(20, min_periods=20).mean() * 1e9,
        "gap": O / C.shift(1) - 1,
    }
    f["tt_score"] = trend_score(C, f["rs"])
    fwd = C.shift(-horizon) / O.shift(-1) - 1                              # buy next open, hold `horizon` sessions
    cols = {k: v.where(liquid).stack() for k, v in f.items()}
    df = pd.DataFrame(cols)
    df["fwd"] = fwd.where(liquid).stack().reindex(df.index)
    df = df.reset_index()
    df.columns = ["date", "symbol"] + list(cols) + ["fwd"]
    m = pd.DataFrame({"breadth": breadth, "mkt_ret_20": mkt.pct_change(20),
                      "mkt_above_50": (mkt > mkt.rolling(50).mean()).astype(float)})
    df = df.join(m, on="date")
    pos = pd.Series(np.arange(len(dates)), index=dates)
    settle = pos[df["date"]].to_numpy() + horizon
    df["label_date"] = pd.NaT
    ok = settle < len(dates)
    df.loc[ok, "label_date"] = dates[settle[ok]]
    df["up"] = (df["fwd"] > 0).astype(float).where(df["fwd"].notna())
    cols = FEATURES + EXTRA + TREND
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).astype("float32")
    return df.dropna(subset=["r60"])


# ---------- model ---------------------------------------------------------------------------

def _model() -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, min_samples_leaf=500,
                                          l2_regularization=1.0, random_state=0)


def walk_forward(df: pd.DataFrame, first_test: str = FIRST_TEST, train_rows: int = TRAIN_ROWS) -> pd.DataFrame:
    """Out-of-sample probability for every labelled row from first_test on, plus the latest
    unlabelled rows scored by a model trained on everything settled so far."""
    rng = np.random.default_rng(0)
    labelled = df[df["up"].notna()]
    months = pd.date_range(pd.Timestamp(first_test), df["date"].max(), freq="MS")
    out = []
    for i, start in enumerate(months):
        end = months[i + 1] if i + 1 < len(months) else df["date"].max() + pd.Timedelta(days=1)
        test = df[(df["date"] >= start) & (df["date"] < end)]
        train = labelled[labelled["label_date"] < start]          # every label settled before the month
        if test.empty or train["up"].nunique() < 2:
            continue
        if len(train) > train_rows:
            train = train.iloc[rng.choice(len(train), train_rows, replace=False)]
        m = _model().fit(train[FEATURES], train["up"].astype(int))
        out.append(test.assign(p=m.predict_proba(test[FEATURES])[:, 1]))
        logger.info("scored %s: %d rows (trained on %d)", start.strftime("%Y-%m"), len(test), len(train))
    return pd.concat(out, ignore_index=True)


# ---------- track record --------------------------------------------------------------------

def _auc(y: pd.Series, p: pd.Series) -> float | None:
    return float(roc_auc_score(y, p)) if y.nunique() == 2 and len(y) >= 100 else None


def _buckets(s: pd.DataFrame) -> List[Dict[str, Any]]:
    out = []
    for lo, hi in zip(BUCKETS[:-1], BUCKETS[1:]):
        g = s[(s["p"] >= lo) & (s["p"] < hi)]
        if len(g):
            out.append({"from": lo, "to": hi, "n": int(len(g)), "predicted": round(float(g["p"].mean()), 4),
                        "actual_up": round(float(g["up"].mean()), 4), "avg_return": round(float(g["fwd"].mean()), 5)})
    return out


def _record(s: pd.DataFrame) -> Dict[str, Any]:
    auc = _auc(s["up"], s["p"])
    hi, lo = s[s["p"] >= 0.55], s[s["p"] < 0.45]
    return {
        "n": int(len(s)), "base_up": round(float(s["up"].mean()), 4),
        "auc": None if auc is None else round(auc, 4),
        "edge": bool(auc is not None and auc > NO_EDGE_AUC),
        "high": {"n": int(len(hi)), "up": int(hi["up"].sum())},     # it said 55%+
        "low": {"n": int(len(lo)), "up": int(lo["up"].sum())},      # it said under 45%
        "buckets": _buckets(s),
    }


def ranking(done: pd.DataFrame) -> Dict[str, Any]:
    """The quant view: rank every stock against the others on the same day. Rank IC is the daily
    rank correlation of score vs outcome; deciles are 0 (least likely up) .. 9 (most likely)."""
    ic = done.groupby("date").apply(lambda g: g["rank"].corr(g["fwd"].rank()) if len(g) >= 30 else np.nan).dropna()
    by_dec = done.groupby(["date", "decile"])["fwd"].mean().unstack()
    spread = (by_dec[9] - by_dec[0]).dropna()
    return {
        "ic": round(float(ic.mean()), 4), "ic_days_positive": round(float((ic > 0).mean()), 4), "days": int(len(ic)),
        "deciles": [round(float(by_dec[d].mean()), 5) for d in range(10)],
        "spread": round(float(spread.mean()), 5),
        "spread_by_year": {str(y): round(float(v), 5) for y, v in spread.groupby(spread.index.year).mean().items()},
    }


def summarize(scored: pd.DataFrame) -> Dict[str, Any]:
    scored = scored.assign(rank=scored.groupby("date")["p"].rank(pct=True))
    scored["decile"] = np.minimum((scored["rank"] * 10).astype(int), 9)    # rank 1.0 -> decile 9, not 10
    done = scored[scored["up"].notna()]
    latest_day = scored["date"].max()
    today = scored[scored["date"] == latest_day].set_index("symbol")
    latest, latest_rank = today["p"], today["rank"]
    grade = np.where((done["breadth"] >= 0.6) & (done["mkt_above_50"] == 1), "A",
                     np.where((done["breadth"] >= 0.6) | (done["mkt_above_50"] == 1), "B", "C"))
    stocks = {}
    for sym, g in done.groupby("symbol"):
        if len(g) >= 250:
            bot, top = g[g["decile"] == 0], g[g["decile"] == 9]
            stocks[sym] = {**_record(g), "p": round(float(latest[sym]), 4) if sym in latest else None,
                           "rank": round(float(latest_rank[sym]), 4) if sym in latest_rank else None,
                           "bottom": {"n": int(len(bot)), "avg_return": round(float(bot["fwd"].mean()), 5) if len(bot) else None},
                           "top": {"n": int(len(top)), "avg_return": round(float(top["fwd"].mean()), 5) if len(top) else None}}
    return {
        "generated": date.today().isoformat(), "asof": latest_day.strftime("%Y-%m-%d"),
        "question": f"Close {HORIZON} sessions later above the next session's open",
        "period": f"{done['date'].min():%b %Y} to {done['date'].max():%b %Y}",
        "no_edge_auc": NO_EDGE_AUC, "overall": _record(done), "ranking": ranking(done),
        "ranked_today": int(len(today)),
        "by_grade": {k: _record(done[grade == k]) for k in ("A", "B", "C")},
        "by_year": {str(y): _auc(g["up"], g["p"]) for y, g in done.groupby(done["date"].dt.year)},
        "stocks": stocks,
    }


def build() -> Dict[str, Any]:
    df = features(panel(bhavcopy.load()))
    logger.info("%d stock-days, %d symbols", len(df), df["symbol"].nunique())
    scored = walk_forward(df)
    SCORED.parent.mkdir(parents=True, exist_ok=True)
    scored[["date", "symbol", "p", "fwd"]].to_parquet(SCORED, index=False)
    data = summarize(scored)
    OUT.write_text(json.dumps(data), encoding="utf-8")
    return data


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    d = build()
    o = d["overall"]
    print(f"unseen {d['period']}: {o['n']:,} stock-days, AUC {o['auc']}, base up {o['base_up']:.1%}, "
          f"edge={o['edge']}")
    for b in o["buckets"]:
        print(f"  said {b['from']:.0%}-{b['to']:.0%}: n={b['n']:,} predicted {b['predicted']:.1%} actual {b['actual_up']:.1%} "
              f"avg 5d {b['avg_return']:+.2%}")
    r = d["ranking"]
    print(f"rank IC {r['ic']} (positive {r['ic_days_positive']:.0%} of {r['days']} days), "
          f"top-minus-bottom {r['spread']:+.2%}; deciles:", [f"{x:+.2%}" for x in r["deciles"]])
    print("by market grade:", {k: (v["auc"], v["n"]) for k, v in d["by_grade"].items()})
    print("by year AUC:", d["by_year"])
