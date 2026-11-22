"""Daily prediction + paper-trading loop: predict tomorrow, plan ₹10,000, score it, keep a ledger.

Every evening, after NSE publishes the day's bhavcopy:
  1. settle   - yesterday's predictions get their outcome (crossed out when wrong), both books trade
                at today's real open/close prices with real Indian charges, the ledger gets a row.
  2. refit    - the 1-day and 20-session models are refitted on everything settled so far.
  3. predict  - every Nifty 100 stock gets tomorrow's call: chance to beat the middle stock,
                expected move and a 68% price range; both books get tomorrow's plan.

Book A (daily picks): top 5 Nifty 100 stocks by the 1-day model, ₹(book/5) each, bought at the open
and sold at the close (intraday charges). Book B (position book): the 20-session search winner over
all liquid stocks, 5 positions, rebalanced every 20 sessions, held between (delivery charges).
Predictions are written before the day they predict and can never be changed afterwards (SQLite
triggers). Decision support only: nothing here places orders.

Run:  python -m code.research.daily            (nightly; laptop only)
      python -m code.research.daily --replay 60 (rebuild the last 60 sessions as replayed history)
"""
from __future__ import annotations

import json
import logging
import math
import sqlite3
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from code.research import model_search as ms
from code.research import updown as u

logger = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parents[1] / "data"
DB = DATA / "papa.db"
NIFTY100 = DATA / "nifty100_constituents.csv"
SINCE = "2016-01-01"           # history loaded each night (features need a year; training uses the rest)
START_CASH = 10_000.0
N_A, N_B = 5, 5
B_REBAL, B_EXIT = 20, 0.70
SLIP_A, SLIP_B = 0.0005, 0.001  # per side: Nifty 100 at ₹2,000 orders vs all liquid stocks
CALIB_DAYS = 60                 # out-of-sample days used to turn scores into expected moves
TRAIN_ROWS = 300_000


# ---------- charges (Zerodha-style, per order value in ₹) ------------------------------------

def intraday_charges(buy: float, sell: float) -> float:
    brk = min(20.0, 0.0003 * buy) + min(20.0, 0.0003 * sell)
    exch, sebi = 0.0000297 * (buy + sell), 1e-6 * (buy + sell)
    return brk + exch + sebi + 0.18 * (brk + exch + sebi) + 0.00025 * sell + 0.00003 * buy


def delivery_charges(buy: float, sell: float) -> float:
    exch, sebi = 0.0000297 * (buy + sell), 1e-6 * (buy + sell)
    dp = 15.93 if sell > 0 else 0.0                     # depository charge per stock sold
    return 0.001 * (buy + sell) + exch + sebi + 0.18 * (exch + sebi) + 0.00015 * buy + dp


# ---------- storage -------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS daily_predictions (
  asof TEXT, symbol TEXT, replayed INTEGER, score REAL, rank REAL, call TEXT, exp_move REAL,
  close REAL, lo68 REAL, hi68 REAL, vol REAL, created_at TEXT,
  outcome_date TEXT, open_next REAL, close_next REAL, move REAL, median_move REAL, vs_median REAL,
  right INTEGER, in_range INTEGER, nifty_move REAL, gap REAL,
  PRIMARY KEY (asof, symbol));
CREATE TRIGGER IF NOT EXISTS daily_pred_frozen BEFORE UPDATE OF asof, symbol, score, rank, call, exp_move,
  close, lo68, hi68, vol ON daily_predictions BEGIN SELECT RAISE(ABORT, 'predictions are never changed'); END;
CREATE TRIGGER IF NOT EXISTS daily_pred_settled_once BEFORE UPDATE OF move, right, in_range ON daily_predictions
  WHEN OLD.move IS NOT NULL BEGIN SELECT RAISE(ABORT, 'outcomes are settled once'); END;
CREATE TRIGGER IF NOT EXISTS daily_pred_no_delete BEFORE DELETE ON daily_predictions
  WHEN OLD.replayed = 0 BEGIN SELECT RAISE(ABORT, 'live predictions are never deleted'); END;
CREATE TABLE IF NOT EXISTS daily_plan (
  asof TEXT, book TEXT, symbol TEXT, action TEXT, shares INTEGER, est_price REAL, rank REAL, replayed INTEGER,
  PRIMARY KEY (asof, book, symbol));
CREATE TABLE IF NOT EXISTS daily_trades (
  date TEXT, book TEXT, symbol TEXT, side TEXT, shares INTEGER, price REAL, charges REAL, replayed INTEGER);
CREATE TABLE IF NOT EXISTS daily_ledger (
  date TEXT, book TEXT, start_value REAL, end_value REAL, pnl REAL, charges REAL, nifty_move REAL,
  replayed INTEGER, PRIMARY KEY (date, book));
CREATE TABLE IF NOT EXISTS daily_meta (k TEXT PRIMARY KEY, v TEXT);
"""


def connect(path: Path = DB) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def clear_replayed(conn: sqlite3.Connection) -> None:
    for t in ("daily_predictions", "daily_plan", "daily_trades", "daily_ledger"):
        conn.execute(f"DELETE FROM {t} WHERE replayed = 1")
    conn.commit()


def holdings(conn: sqlite3.Connection, book: str) -> Tuple[Dict[str, int], float]:
    """Shares held and cash, rebuilt from the trade log (the log is the truth)."""
    pos: Dict[str, int] = {}
    cash = START_CASH
    for r in conn.execute("SELECT symbol, side, shares, price, charges FROM daily_trades WHERE book = ? ORDER BY rowid", (book,)):
        sign = -1 if r["side"] == "SELL" else 1              # BUY and SPLIT (price 0) add shares
        pos[r["symbol"]] = pos.get(r["symbol"], 0) + sign * r["shares"]
        cash -= sign * r["shares"] * r["price"] + r["charges"]
    return {s: n for s, n in pos.items() if n}, cash


def book_value(conn: sqlite3.Connection, book: str) -> float:
    row = conn.execute("SELECT end_value FROM daily_ledger WHERE book = ? ORDER BY date DESC LIMIT 1", (book,)).fetchone()
    return row["end_value"] if row else START_CASH


# ---------- data and models -----------------------------------------------------------------

def nifty100() -> List[str]:
    return pd.read_csv(NIFTY100)["Symbol"].str.strip().tolist()


def update_data() -> None:
    """Fetch any missing NSE days and Nifty closes up to today."""
    from code.research import bhavcopy
    last = max(f.stem for f in bhavcopy.DAYS.glob("*.parquet"))
    bhavcopy.backfill(start=datetime.strptime(last, "%Y%m%d").date() + timedelta(days=1), workers=2)
    try:
        import yfinance as yf
        from code.research.portfolio import NIFTY
        old = pd.read_parquet(NIFTY)
        new = yf.download("^NSEI", start=(old.index.max() - timedelta(days=10)).date().isoformat(), progress=False, auto_adjust=False)
        if len(new):
            add = pd.DataFrame({"nifty_open": new["Open"].squeeze(), "nifty": new["Close"].squeeze()})
            add.index = pd.to_datetime(add.index).tz_localize(None)
            merged = pd.concat([old[~old.index.isin(add.index)], add]).sort_index()
            merged["vix"] = merged["vix"].ffill() if "vix" in merged else np.nan
            merged.to_parquet(NIFTY)
    except Exception as exc:                                        # Nifty is context, never a blocker
        logger.warning("Nifty update failed: %s", exc)


def relabel(df: pd.DataFrame, w: Dict[str, pd.DataFrame], horizon: int) -> pd.DataFrame:
    """Same features, outcome `horizon` sessions out (buy next open, sell that close)."""
    C, O = w["close"], w["open"]
    fwd = (C.shift(-horizon) / O.shift(-1) - 1).stack()
    fwd.index.names = ["date", "symbol"]
    out = df.drop(columns=["fwd", "up", "label_date"]).join(fwd.rename("fwd"), on=["date", "symbol"])
    dates = C.index
    pos = pd.Series(np.arange(len(dates)), index=dates)
    settle = pos[out["date"]].to_numpy() + horizon
    out["label_date"] = pd.NaT
    ok = settle < len(dates)
    out.loc[ok, "label_date"] = dates[settle[ok]]
    out["up"] = (out["fwd"] > 0).astype(float).where(out["fwd"].notna())
    return out


def load_frames(since: str = SINCE) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    from code.research import bhavcopy
    w = u.panel(bhavcopy.load(since=since))
    d1 = u.features(w, horizon=1)
    d20 = relabel(d1, w, 20)
    return ms.prepare(d1), ms.prepare(d20), w


def best_spec(horizon: int) -> Dict[str, Any]:
    out = ms.paths(horizon)[0]
    key = json.loads(out.read_text(encoding="utf-8"))["best"]["key"]
    if key.startswith("blend("):
        return {"blend": key[len("blend("):-1].split(" + ")}
    return ms._spec(key)


def fit_score(df: pd.DataFrame, spec: Dict[str, Any], train_until: pd.Timestamp, rows: pd.DataFrame,
              train_rows: int = TRAIN_ROWS) -> pd.Series:
    """Fit on outcomes settled on or before train_until; score `rows`. Blends average member ranks."""
    if "blend" in spec:
        parts = [fit_score(df, ms._spec(k), train_until, rows, train_rows) for k in spec["blend"]]
        ranks = [p.groupby(rows["date"]).rank(pct=True) for p in parts]
        return sum(ranks) / len(ranks)
    target, cols = spec["target"], ms.columns(spec)
    train = df[df[target].notna() & (df["label_date"] <= train_until)]
    if len(train) > train_rows:                                     # recent rows more likely: the market changes
        wts = np.linspace(0.3, 1.0, len(train))
        train = train.iloc[np.sort(np.random.default_rng(0).choice(len(train), train_rows, replace=False, p=wts / wts.sum()))]
    y = train[target] if target == "rankret" else train[target].astype(int)
    m = ms.estimator(spec).fit(train[cols], y)
    return pd.Series(ms._score(m, rows[cols]), index=rows.index)


# ---------- one evening ---------------------------------------------------------------------

def predict_day(conn: sqlite3.Connection, d1: pd.DataFrame, d20: pd.DataFrame, asof: pd.Timestamp,
                replayed: int, spec1: Dict[str, Any], spec20: Dict[str, Any], universe: List[str]) -> None:
    """Tomorrow's calls for the corpus and both books' plans, using only what was known at asof's close."""
    dates = pd.Index(sorted(d1["date"].unique()))
    today = d1[d1["date"] == asof]
    corpus = today[today["symbol"].isin(universe)]
    if corpus.empty:
        return
    # expected move per score tenth, from the last CALIB_DAYS sessions scored by a model that never saw them
    i = dates.get_loc(asof)
    cal_from = dates[max(i - CALIB_DAYS, 0)]
    cal = d1[(d1["date"] > cal_from) & (d1["date"] <= dates[i - 1]) & d1["fwd"].notna()]
    cal_score = fit_score(d1, spec1, cal_from, cal)
    cal_dec = np.minimum((cal_score.groupby(cal["date"]).rank(pct=True) * 10).astype(int), 9)
    move_by_dec = cal["fwd"].groupby(cal_dec).mean().reindex(range(10)).fillna(0.0)

    s1 = fit_score(d1, spec1, asof, today)
    rank_all = s1.rank(pct=True)
    corpus_score = s1.loc[corpus.index]
    rk = corpus_score.rank(pct=True)
    created = datetime.now().isoformat(timespec="seconds")
    for idx, r in corpus.iterrows():
        dec = min(int(rank_all[idx] * 10), 9)
        exp = float(move_by_dec[dec])
        vol = float(r["vol20"]) if pd.notna(r["vol20"]) else 0.02
        close = float(w_close(r))
        conn.execute("INSERT OR IGNORE INTO daily_predictions (asof, symbol, replayed, score, rank, call, exp_move, close, "
                     "lo68, hi68, vol, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (asof.strftime("%Y-%m-%d"), r["symbol"], replayed, float(corpus_score[idx]), float(rk[idx]),
                      "BEAT" if rk[idx] >= 0.5 else "LAG", exp, close, close * (1 + exp - vol), close * (1 + exp + vol),
                      vol, created))

    # Book A: top 5 of the corpus, whole shares of ~book/5 each (skip a stock one share can't fit)
    value_a = book_value(conn, "A")
    picks = []
    for idx in corpus_score.sort_values(ascending=False).index:
        price = float(w_close(corpus.loc[idx]))
        sh = int((value_a / N_A) // price)
        if sh > 0:
            picks.append((corpus.loc[idx, "symbol"], sh, price, float(rk[idx])))
        if len(picks) == N_A:
            break
    for sym, sh, price, r_ in picks:
        conn.execute("INSERT OR REPLACE INTO daily_plan VALUES (?,?,?,?,?,?,?,?)",
                     (asof.strftime("%Y-%m-%d"), "A", sym, "BUY_OPEN_SELL_CLOSE", sh, price, r_, replayed))

    # Book B: rebalance every B_REBAL sessions (or when empty), keep stocks still in the top 30%
    held, cash = holdings(conn, "B")
    last = conn.execute("SELECT v FROM daily_meta WHERE k = 'b_last_rebalance'").fetchone()
    since = (i - dates.get_loc(pd.Timestamp(last["v"]))) if last and pd.Timestamp(last["v"]) in dates else B_REBAL
    if held and since < B_REBAL:
        for sym, sh in held.items():
            conn.execute("INSERT OR REPLACE INTO daily_plan VALUES (?,?,?,?,?,?,?,?)",
                         (asof.strftime("%Y-%m-%d"), "B", sym, "HOLD", sh, None, None, replayed))
        return
    t20 = d20[d20["date"] == asof]
    s20 = fit_score(d20, spec20, asof, t20)
    r20 = pd.Series(s20.rank(pct=True).to_numpy(), index=t20["symbol"].to_numpy())
    keep = [s for s in held if r20.get(s, 0.0) >= B_EXIT]
    value_b = book_value(conn, "B")
    close_of = dict(zip(t20["symbol"], [w_close(r) for _, r in t20.iterrows()]))
    adds = []
    for sym in r20.sort_values(ascending=False).index:
        if sym in keep or sym in held:
            continue
        price = close_of.get(sym)
        if price and (value_b / N_B) // price > 0:
            adds.append(sym)
        if len(keep) + len(adds) == N_B:
            break
    a = asof.strftime("%Y-%m-%d")
    for sym, sh in held.items():
        act = "HOLD" if sym in keep else "SELL_OPEN"
        conn.execute("INSERT OR REPLACE INTO daily_plan VALUES (?,?,?,?,?,?,?,?)",
                     (a, "B", sym, act, sh, close_of.get(sym), float(r20.get(sym, np.nan)), replayed))
    for sym in adds:
        sh = int((value_b / N_B) // close_of[sym])
        conn.execute("INSERT OR REPLACE INTO daily_plan VALUES (?,?,?,?,?,?,?,?)",
                     (a, "B", sym, "BUY_OPEN", sh, close_of[sym], float(r20[sym]), replayed))
    conn.execute("INSERT OR REPLACE INTO daily_meta VALUES ('b_last_rebalance', ?)", (a,))


def w_close(r: pd.Series) -> float:
    """The real (unadjusted) close on the row's date."""
    return float(r["close_px"])


def raw(w: Dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    """Prices as actually traded that day (the panel is split-adjusted for the models)."""
    return w[col] * w["cum"]


def settle_day(conn: sqlite3.Connection, w: Dict[str, pd.DataFrame], asof: pd.Timestamp, day: pd.Timestamp,
               replayed: int, nifty: Optional[pd.Series] = None) -> None:
    """`day` is the session after `asof`: score asof's predictions, trade both books at day's prices."""
    a, dstr = asof.strftime("%Y-%m-%d"), day.strftime("%Y-%m-%d")
    O, C = raw(w, "open").loc[day], raw(w, "close").loc[day]
    n_move = None
    if nifty is not None and day in nifty.index and asof in nifty.index:
        n_move = float(nifty[day] / nifty[asof] - 1)
    preds = conn.execute("SELECT symbol, call, close, lo68, hi68 FROM daily_predictions WHERE asof = ? AND move IS NULL", (a,)).fetchall()
    moves = {p["symbol"]: float(C[p["symbol"]] / O[p["symbol"]] - 1) for p in preds
             if pd.notna(O.get(p["symbol"])) and pd.notna(C.get(p["symbol"]))}
    if moves:
        med = float(np.median(list(moves.values())))
        for p in preds:
            if p["symbol"] not in moves:
                continue
            m, cl = moves[p["symbol"]], float(C[p["symbol"]])
            beat = m > med
            conn.execute("UPDATE daily_predictions SET outcome_date=?, open_next=?, close_next=?, move=?, median_move=?, "
                         "vs_median=?, right=?, in_range=?, nifty_move=?, gap=? WHERE asof=? AND symbol=?",
                         (dstr, float(O[p["symbol"]]), cl, m, med, m - med, int(beat == (p["call"] == "BEAT")),
                          int(p["lo68"] <= cl <= p["hi68"]), n_move, float(O[p["symbol"]] / p["close"] - 1), a, p["symbol"]))

    # Book A: buy at the open, sell at the close, whole shares sized on the actual open
    start = book_value(conn, "A")
    pnl = chg = 0.0
    plan = conn.execute("SELECT symbol FROM daily_plan WHERE asof=? AND book='A'", (a,)).fetchall()
    for p in plan:
        s = p["symbol"]
        if pd.isna(O.get(s)) or pd.isna(C.get(s)):
            continue
        buy_px, sell_px = float(O[s]) * (1 + SLIP_A), float(C[s]) * (1 - SLIP_A)
        sh = int((start / N_A) // buy_px)
        if sh <= 0:
            continue
        c = intraday_charges(sh * buy_px, sh * sell_px)
        pnl += sh * (sell_px - buy_px) - c
        chg += c
        conn.execute("INSERT INTO daily_trades VALUES (?,?,?,?,?,?,?,?)", (dstr, "A", s, "BUY", sh, buy_px, c, replayed))
        conn.execute("INSERT INTO daily_trades VALUES (?,?,?,?,?,?,?,?)", (dstr, "A", s, "SELL", sh, sell_px, 0.0, replayed))
    conn.execute("INSERT OR REPLACE INTO daily_ledger VALUES (?,?,?,?,?,?,?,?)",
                 (dstr, "A", start, start + pnl, pnl, chg, n_move, replayed))

    # Book B: a split or bonus overnight multiplies the shares held (no cost), then sells and buys at
    # the open (delivery), then everything is valued at the close
    start_b = book_value(conn, "B")
    held, _ = holdings(conn, "B")
    for s, n in held.items():
        ratio = float(w["cum"].loc[asof, s] / w["cum"].loc[day, s]) if s in w["cum"].columns else 1.0
        if pd.notna(ratio) and abs(ratio - 1) > 1e-6:
            extra = int(round(n * ratio)) - n
            conn.execute("INSERT INTO daily_trades VALUES (?,?,?,?,?,?,?,?)", (dstr, "B", s, "SPLIT", extra, 0.0, 0.0, replayed))
    held, _ = holdings(conn, "B")
    chg_b = 0.0
    for p in conn.execute("SELECT symbol, action, shares FROM daily_plan WHERE asof=? AND book='B'", (a,)).fetchall():
        s = p["symbol"]
        if p["action"] == "SELL_OPEN" and held.get(s) and pd.notna(O.get(s)):
            px = float(O[s]) * (1 - SLIP_B)
            c = delivery_charges(0.0, held[s] * px)
            chg_b += c
            conn.execute("INSERT INTO daily_trades VALUES (?,?,?,?,?,?,?,?)", (dstr, "B", s, "SELL", held[s], px, c, replayed))
    _, cash = holdings(conn, "B")
    buys = [p for p in conn.execute("SELECT symbol, shares FROM daily_plan WHERE asof=? AND book='B' AND action='BUY_OPEN'", (a,))]
    for p in buys:
        s = p["symbol"]
        if pd.isna(O.get(s)):
            continue
        px = float(O[s]) * (1 + SLIP_B)
        sh = min(int(p["shares"]), int(cash // (px * 1.002)))
        if sh <= 0:
            continue
        c = delivery_charges(sh * px, 0.0)
        cash -= sh * px + c
        chg_b += c
        conn.execute("INSERT INTO daily_trades VALUES (?,?,?,?,?,?,?,?)", (dstr, "B", s, "BUY", sh, px, c, replayed))
    held, cash = holdings(conn, "B")
    last_px = raw(w, "close").loc[:day].ffill().iloc[-1]
    end_b = cash + sum(n * float(last_px.get(s, 0.0)) for s, n in held.items())
    conn.execute("INSERT OR REPLACE INTO daily_ledger VALUES (?,?,?,?,?,?,?,?)",
                 (dstr, "B", start_b, end_b, end_b - start_b, chg_b, n_move, replayed))


# ---------- driver --------------------------------------------------------------------------

def _frames_with_close(since: str = SINCE):
    d1, d20, w = load_frames(since)
    px = raw(w, "close").stack()
    px.index.names = ["date", "symbol"]
    d1 = d1.join(px.rename("close_px"), on=["date", "symbol"])
    d20 = d20.join(px.rename("close_px"), on=["date", "symbol"])
    return d1, d20, w


def run(replay: int = 0, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    conn = conn or connect()
    if not replay:
        update_data()
    d1, d20, w = _frames_with_close()
    spec1, spec20 = best_spec(1), best_spec(20)
    universe = nifty100()
    from code.research.portfolio import NIFTY
    nifty = pd.read_parquet(NIFTY)["nifty"] if NIFTY.exists() else None
    dates = pd.Index(sorted(d1["date"].unique()))
    if replay:
        clear_replayed(conn)
        conn.execute("DELETE FROM daily_meta WHERE k = 'b_last_rebalance'")
        days = dates[-(replay + 1):]
        for asof, nxt in zip(days[:-1], days[1:]):
            predict_day(conn, d1, d20, asof, 1, spec1, spec20, universe)
            settle_day(conn, w, asof, nxt, 1, nifty)
            conn.commit()
            logger.info("replayed %s -> %s", asof.date(), nxt.date())
        predict_day(conn, d1, d20, days[-1], 1, spec1, spec20, universe)
        conn.commit()
        return summary(conn)
    # live: settle every predicted day that now has its next session, then predict from the latest close
    pending = [r["asof"] for r in conn.execute("SELECT DISTINCT asof FROM daily_predictions WHERE move IS NULL ORDER BY asof")]
    for a in pending:
        ts = pd.Timestamp(a)
        later = dates[dates > ts]
        if len(later):
            rep = conn.execute("SELECT MAX(replayed) r FROM daily_predictions WHERE asof = ?", (a,)).fetchone()["r"] or 0
            settle_day(conn, w, ts, later[0], int(rep), nifty)
    predict_day(conn, d1, d20, dates[-1], 0, spec1, spec20, universe)
    conn.commit()
    return summary(conn)


# ---------- report (read by the dashboard; the same payload is pushed to Railway) -------------

def report(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    rows = lambda q, *a: [dict(r) for r in conn.execute(q, a).fetchall()]
    latest = conn.execute("SELECT MAX(asof) FROM daily_predictions").fetchone()[0]
    if not latest:
        return None
    scored = conn.execute("SELECT MAX(asof) FROM daily_predictions WHERE move IS NOT NULL").fetchone()[0]
    return {
        "summary": summary(conn), "asof": latest, "scored_asof": scored,
        "predictions": rows("SELECT * FROM daily_predictions WHERE asof = ? ORDER BY score DESC", latest),
        "plan": rows("SELECT * FROM daily_plan WHERE asof = ? ORDER BY book, action, rank DESC", latest),
        "last_scored": rows("SELECT * FROM daily_predictions WHERE asof = ? ORDER BY vs_median DESC", scored) if scored else [],
        "last_plan": rows("SELECT * FROM daily_plan WHERE asof = ?", scored) if scored else [],
        "ledger": rows("SELECT * FROM daily_ledger ORDER BY date, book"),
        "daily_accuracy": rows("SELECT outcome_date AS date, AVG(right) AS accuracy, AVG(in_range) AS in_range, "
                               "COUNT(*) AS n, MAX(replayed) AS replayed FROM daily_predictions "
                               "WHERE move IS NOT NULL GROUP BY outcome_date ORDER BY outcome_date"),
        "generated": datetime.now().isoformat(timespec="seconds"),
    }


def push(conn: sqlite3.Connection) -> Optional[int]:
    """Send tonight's report to the deployed app (Railway can't run the loop: the NSE history lives
    here). Needs DAILY_SYNC_URL (https://<app>/api/daily/sync) and DAILY_SYNC_TOKEN in .env."""
    import os
    import requests
    url, token = os.getenv("DAILY_SYNC_URL"), os.getenv("DAILY_SYNC_TOKEN")
    data = report(conn)
    if not (url and token and data):
        return None
    r = requests.post(url, data=json.dumps(data, default=str), timeout=60,
                      headers={"Content-Type": "application/json", "X-Sync-Token": token})
    logger.info("pushed daily report to %s: HTTP %s", url, r.status_code)
    return r.status_code

def summary(conn: sqlite3.Connection) -> Dict[str, Any]:
    led = pd.read_sql("SELECT * FROM daily_ledger ORDER BY date", conn)
    pred = pd.read_sql("SELECT * FROM daily_predictions WHERE move IS NOT NULL", conn)
    books = {}
    for b in ("A", "B"):
        l = led[led["book"] == b]
        if len(l):
            books[b] = {"now": round(float(l["end_value"].iloc[-1]), 2), "pnl": round(float(l["end_value"].iloc[-1] - START_CASH), 2),
                        "days": int(len(l)), "up_days": int((l["pnl"] > 0).sum()), "charges": round(float(l["charges"].sum()), 2)}
    out: Dict[str, Any] = {"start": START_CASH, "books": books, "scored": int(len(pred))}
    if len(pred):
        out["accuracy"] = round(float(pred["right"].mean()), 4)
        out["in_range"] = round(float(pred["in_range"].mean()), 4)
        out["misses"] = misses(pred)
    return out


def misses(pred: pd.DataFrame) -> List[Dict[str, Any]]:
    """Where the calls go wrong: accuracy by market direction, opening gap and the stock's volatility."""
    p = pred.dropna(subset=["right"]).copy()
    out = []

    def add(name: str, mask: pd.Series) -> None:
        g = p[mask]
        if len(g) >= 20:
            out.append({"when": name, "n": int(len(g)), "accuracy": round(float(g["right"].mean()), 4)})
    if p["nifty_move"].notna().any():
        add("Nifty up that day", p["nifty_move"] > 0)
        add("Nifty down that day", p["nifty_move"] <= 0)
    add("Big opening gap (over 1%)", p["gap"].abs() > 0.01)
    add("Small opening gap", p["gap"].abs() <= 0.01)
    q = p["vol"].quantile([1 / 3, 2 / 3]).to_numpy()
    add("Calm stocks", p["vol"] <= q[0])
    add("Jumpy stocks", p["vol"] > q[1])
    add("Called BEAT", p["call"] == "BEAT")
    add("Called LAG", p["call"] == "LAG")
    return out


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    n = int(sys.argv[sys.argv.index("--replay") + 1]) if "--replay" in sys.argv else 0
    if "--push-only" not in sys.argv:
        print(json.dumps(run(replay=n), indent=1, default=str))
    try:
        push(connect())
    except Exception as exc:                       # the local results are safe either way
        logger.warning("push to the deployed app failed: %s", exc)
