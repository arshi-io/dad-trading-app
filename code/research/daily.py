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
CREATE TABLE IF NOT EXISTS daily_governor (
  date TEXT PRIMARY KEY, champion TEXT, switched INTEGER, reason TEXT, scores TEXT, retrain INTEGER, replayed INTEGER);
"""


def connect(path: Path = DB) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def clear_replayed(conn: sqlite3.Connection) -> None:
    for t in ("daily_predictions", "daily_plan", "daily_trades", "daily_ledger", "daily_governor"):
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


HOLIDAYS = DATA / "nse_holidays.json"


def missing_sessions(have: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> List[str]:
    """NSE trading days between start and end (weekdays minus exchange holidays) with no data.
    A missing day would make the next row of every feature, outcome and ledger span two sessions."""
    try:
        hol = set(json.loads(HOLIDAYS.read_text(encoding="utf-8"))["holidays"])
    except (OSError, ValueError, KeyError):
        hol = set()
    if hol:   # only judge years the holiday list covers; an older year's holidays would look like holes
        start = max(start, pd.Timestamp(f"{min(h[:4] for h in hol)}-01-01"))
    have_s = {d.strftime("%Y-%m-%d") for d in have}
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start, end)
            if d.strftime("%Y-%m-%d") not in hol and d.strftime("%Y-%m-%d") not in have_s]


def update_data() -> None:
    """Fetch any missing NSE days (new ones and holes in the last 400 days) and Nifty closes."""
    from code.research import bhavcopy
    bhavcopy.backfill(start=date.today() - timedelta(days=400), workers=2)   # skips days already on disk
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


def load_frames(since: str = SINCE) -> Tuple[Dict[int, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Feature frames for the 1-, 5- and 20-session outcomes (same features, different labels)."""
    from code.research import bhavcopy
    w = u.panel(bhavcopy.load(since=since))
    d1 = u.features(w, horizon=1)
    return {1: ms.prepare(d1), 5: ms.prepare(relabel(d1, w, 5)), 20: ms.prepare(relabel(d1, w, 20))}, w


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

def predict_day(conn: sqlite3.Connection, frames: Dict[int, pd.DataFrame], asof: pd.Timestamp,
                replayed: int, specs: Dict[int, Any], universe: List[str]) -> None:
    """Tomorrow's calls for the corpus and every book's plan, using only what was known at asof's close.
    frames / specs are keyed by horizon in sessions (1, 5, 20)."""
    d1, spec1 = frames[1], specs[1]
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

    # Intraday books, whole shares of ~book/5 per stock (a stock one share can't fit is skipped):
    #   A   buy the top 5 at the open, sell at the close
    #   AS  short the bottom 5 at the open, buy back at the close (intraday MIS short)
    #   ALS both: ~book/5 on each of 10 legs (2x gross, inside the ~20% intraday margin)
    def pick(book: str, order: List[Any]) -> List[Tuple[str, int, float, float]]:
        value = book_value(conn, book)
        out = []
        for idx in order:
            price = float(w_close(corpus.loc[idx]))
            sh = int((value / N_A) // price)
            if sh > 0:
                out.append((corpus.loc[idx, "symbol"], sh, price, float(rk[idx])))
            if len(out) == N_A:
                break
        return out
    best_first = list(corpus_score.sort_values(ascending=False).index)
    a = asof.strftime("%Y-%m-%d")
    for book, legs in (("A", [("BUY_OPEN_SELL_CLOSE", best_first)]), ("AS", [("SHORT_OPEN_COVER_CLOSE", best_first[::-1])]),
                       ("ALS", [("BUY_OPEN_SELL_CLOSE", best_first), ("SHORT_OPEN_COVER_CLOSE", best_first[::-1])]),
                       ("AX", [("SHORT_OPEN_COVER_CLOSE", best_first)])):          # contrarian: fade the top 5
        for action, order in legs:
            for sym, sh, price, r_ in pick(book, order):
                conn.execute("INSERT OR REPLACE INTO daily_plan VALUES (?,?,?,?,?,?,?,?)",
                             (a, book, sym, action, sh, price, r_, replayed))

    for book, (frame, spec, rebal, keep_rank) in holding_books(frames, specs).items():
        plan_holding(conn, book, frame, spec, asof, i, dates, rebal, keep_rank, replayed)


# holding books (delivery, held across days): B = 20-session position book, S = 5-session swing book
S_REBAL, S_EXIT = 5, 0.80


def holding_books(frames: Dict[int, pd.DataFrame], specs: Dict[int, Any]) -> Dict[str, Tuple]:
    out = {"B": (frames[20], specs[20], B_REBAL, B_EXIT)}
    if 5 in frames and specs.get(5):
        out["S"] = (frames[5], specs[5], S_REBAL, S_EXIT)
    return out


def plan_holding(conn: sqlite3.Connection, book: str, frame: pd.DataFrame, spec: Dict[str, Any], asof: pd.Timestamp,
                 i: int, dates: pd.Index, rebal: int, keep_rank: float, replayed: int) -> None:
    """Checked every evening: sell a holding that slipped to the middle of the ranking or fell B_STOP
    below its buy price; every `rebal` sessions also drop anything below `keep_rank`; fill to 5."""
    held, _ = holdings(conn, book)
    key = f"{book.lower()}_last_rebalance"
    last = conn.execute("SELECT v FROM daily_meta WHERE k = ?", (key,)).fetchone()
    since = (i - dates.get_loc(pd.Timestamp(last["v"]))) if last and pd.Timestamp(last["v"]) in dates else rebal
    t = frame[frame["date"] == asof]
    s = fit_score(frame, spec, asof, t)
    r = pd.Series(s.rank(pct=True).to_numpy(), index=t["symbol"].to_numpy())
    close_of = dict(zip(t["symbol"], t["close_px"].astype(float)))
    entry = {row["symbol"]: row["price"] for row in conn.execute(
        "SELECT symbol, price FROM daily_trades WHERE book = ? AND side = 'BUY' ORDER BY rowid", (book,))}
    full = since >= rebal or not held
    keep = b_keep(held, r, close_of, entry, full=full, keep_rank=keep_rank)
    value = book_value(conn, book)
    adds = []
    for sym in r.sort_values(ascending=False).index:
        if len(keep) + len(adds) >= N_B:
            break
        if sym in keep or sym in held:
            continue
        price = close_of.get(sym)
        if price and (value / N_B) // price > 0:
            adds.append(sym)
    a = asof.strftime("%Y-%m-%d")
    for sym, sh in held.items():
        act = "HOLD" if sym in keep else "SELL_OPEN"
        conn.execute("INSERT OR REPLACE INTO daily_plan VALUES (?,?,?,?,?,?,?,?)",
                     (a, book, sym, act, sh, close_of.get(sym), float(r.get(sym, np.nan)), replayed))
    for sym in adds:
        sh = int((value / N_B) // close_of[sym])
        conn.execute("INSERT OR REPLACE INTO daily_plan VALUES (?,?,?,?,?,?,?,?)",
                     (a, book, sym, "BUY_OPEN", sh, close_of[sym], float(r[sym]), replayed))
    if full:
        conn.execute("INSERT OR REPLACE INTO daily_meta VALUES (?, ?)", (key, a))


B_DAILY_EXIT, B_STOP = 0.50, 0.05


def b_keep(held: Dict[str, int], r20: pd.Series, close: Dict[str, float], entry: Dict[str, float], full: bool,
           keep_rank: float = B_EXIT) -> List[str]:
    """Which holdings stay tonight. Daily: out if ranked below the middle or down B_STOP from the
    buy price. On full-rebalance nights: out unless still ranked at keep_rank or better."""
    keep = []
    for s in held:
        rank, px, buy = r20.get(s, 0.0), close.get(s), entry.get(s)
        stopped = px is not None and buy and px <= buy * (1 - B_STOP)
        if rank >= (keep_rank if full else B_DAILY_EXIT) and not stopped:
            keep.append(s)
    return keep


def w_close(r: pd.Series) -> float:
    """The real (unadjusted) close on the row's date."""
    return float(r["close_px"])


def raw(w: Dict[str, pd.DataFrame], col: str) -> pd.DataFrame:
    """Prices as actually traded that day (the panel is split-adjusted for the models)."""
    return w[col] * w["cum"]


INTRADAY_BOOKS = ("A", "AS", "ALS", "AX")
BOOK_NAMES = {"A": "Buy top 5", "AS": "Short bottom 5", "ALS": "Buy top 5 + short bottom 5", "AX": "Short top 5 (contrarian)",
              "B": "20-session position book", "S": "5-session swing book", "N": "Nifty (NIFTYBEES)", "M": "Main book (governor)"}

# ---------- governor: judge every book against Nifty each evening, follow the best ----------
# score = sum over the last WINDOW sessions of (book return - Nifty return), minus PENALTY x every
# shortfall. Nifty itself (book N) always scores 0, so when every strategy trails Nifty the main
# book holds Nifty. A switch needs a lead of SWITCH_MARGIN and pays SWITCH_COST. When the main
# book trails Nifty by RETRAIN_BELOW over the window, a retrain is requested (search re-run on the
# most recent year), at most once every RETRAIN_GAP sessions. Decisions use settled days only.
GOVERNOR_ON = False            # paused 23 Nov 2026 at your request; Nifty book N still runs as the reference
CANDIDATES = ("A", "AS", "ALS", "AX", "B", "N")
WINDOW, MIN_HISTORY, PENALTY = 20, 10, 2.0
SWITCH_MARGIN, SWITCH_COST = 0.005, 0.002
RETRAIN_BELOW, RETRAIN_GAP = -0.02, 5


def settle_intraday(conn: sqlite3.Connection, book: str, a: str, dstr: str, O: pd.Series, C: pd.Series,
                    replayed: int, n_move: Optional[float]) -> None:
    """Open at the open, close at the close. Longs buy then sell; shorts sell first then buy back.
    Whole shares of ~book/5 per stock, sized on the actual open."""
    start = book_value(conn, book)
    pnl = chg = 0.0
    for p in conn.execute("SELECT symbol, action FROM daily_plan WHERE asof=? AND book=?", (a, book)).fetchall():
        s, short = p["symbol"], p["action"] == "SHORT_OPEN_COVER_CLOSE"
        if pd.isna(O.get(s)) or pd.isna(C.get(s)):
            continue
        if short:
            entry, exit_ = float(O[s]) * (1 - SLIP_A), float(C[s]) * (1 + SLIP_A)     # sell high-ish, buy back
        else:
            entry, exit_ = float(O[s]) * (1 + SLIP_A), float(C[s]) * (1 - SLIP_A)
        sh = int((start / N_A) // float(O[s]))
        if sh <= 0:
            continue
        buy_val, sell_val = (sh * exit_, sh * entry) if short else (sh * entry, sh * exit_)
        c = intraday_charges(buy_val, sell_val)
        pnl += sell_val - buy_val - c
        chg += c
        sides = ("SHORT", "COVER") if short else ("BUY", "SELL")
        conn.execute("INSERT INTO daily_trades VALUES (?,?,?,?,?,?,?,?)", (dstr, book, s, sides[0], sh, entry, c, replayed))
        conn.execute("INSERT INTO daily_trades VALUES (?,?,?,?,?,?,?,?)", (dstr, book, s, sides[1], sh, exit_, 0.0, replayed))
    conn.execute("INSERT OR REPLACE INTO daily_ledger VALUES (?,?,?,?,?,?,?,?)",
                 (dstr, book, start, start + pnl, pnl, chg, n_move, replayed))


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

    for book in INTRADAY_BOOKS:
        settle_intraday(conn, book, a, dstr, O, C, replayed, n_move)

    for book in HOLDING_BOOKS:
        if book == "B" or conn.execute("SELECT 1 FROM daily_plan WHERE book = ? LIMIT 1", (book,)).fetchone():
            settle_holding(conn, book, w, asof, day, a, dstr, O, n_move, replayed)
    settle_nifty_and_main(conn, a, dstr, n_move, replayed)
    from code.research import fund
    fund.settle(conn, dstr, replayed)


HOLDING_BOOKS = ("B", "S")


def settle_holding(conn: sqlite3.Connection, book: str, w: Dict[str, pd.DataFrame], asof: pd.Timestamp, day: pd.Timestamp,
                   a: str, dstr: str, O: pd.Series, n_move: Optional[float], replayed: int) -> None:
    """A split or bonus overnight multiplies the shares held (no cost), then sells and buys at the
    open (delivery charges), then everything is valued at the close."""
    start_b = book_value(conn, book)
    held, _ = holdings(conn, book)
    for s, n in held.items():
        ratio = float(w["cum"].loc[asof, s] / w["cum"].loc[day, s]) if s in w["cum"].columns else 1.0
        if pd.notna(ratio) and abs(ratio - 1) > 1e-6:
            extra = int(round(n * ratio)) - n
            conn.execute("INSERT INTO daily_trades VALUES (?,?,?,?,?,?,?,?)", (dstr, book, s, "SPLIT", extra, 0.0, 0.0, replayed))
    held, _ = holdings(conn, book)
    chg_b = 0.0
    for p in conn.execute("SELECT symbol, action, shares FROM daily_plan WHERE asof=? AND book=?", (a, book)).fetchall():
        s = p["symbol"]
        if p["action"] == "SELL_OPEN" and held.get(s) and pd.notna(O.get(s)):
            px = float(O[s]) * (1 - SLIP_B)
            c = delivery_charges(0.0, held[s] * px)
            chg_b += c
            conn.execute("INSERT INTO daily_trades VALUES (?,?,?,?,?,?,?,?)", (dstr, book, s, "SELL", held[s], px, c, replayed))
    _, cash = holdings(conn, book)
    buys = [p for p in conn.execute("SELECT symbol, shares FROM daily_plan WHERE asof=? AND book=? AND action='BUY_OPEN'", (a, book))]
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
        conn.execute("INSERT INTO daily_trades VALUES (?,?,?,?,?,?,?,?)", (dstr, book, s, "BUY", sh, px, c, replayed))
    held, cash = holdings(conn, book)
    last_px = raw(w, "close").loc[:day].ffill().iloc[-1]
    end_b = cash + sum(n * float(last_px.get(s, 0.0)) for s, n in held.items())
    conn.execute("INSERT OR REPLACE INTO daily_ledger VALUES (?,?,?,?,?,?,?,?)",
                 (dstr, book, start_b, end_b, end_b - start_b, chg_b, n_move, replayed))


def settle_nifty_and_main(conn: sqlite3.Connection, a: str, dstr: str, n_move: Optional[float], replayed: int) -> None:
    """Book N holds Nifty (NIFTYBEES, ~0.05%/yr cost ignored). Book M earns the day's return of the
    book the governor picked on the evening of `a` (never one picked after seeing `dstr`)."""
    nm = n_move or 0.0
    start_n = book_value(conn, "N")
    conn.execute("INSERT OR REPLACE INTO daily_ledger VALUES (?,?,?,?,?,?,?,?)",
                 (dstr, "N", start_n, start_n * (1 + nm), start_n * nm, 0.0, n_move, replayed))
    if not GOVERNOR_ON:
        return
    g = conn.execute("SELECT champion, switched FROM daily_governor WHERE date = ?", (a,)).fetchone()
    champ, switched = (g["champion"], g["switched"]) if g else ("N", 0)
    row = conn.execute("SELECT start_value, pnl FROM daily_ledger WHERE date = ? AND book = ?", (dstr, champ)).fetchone()
    r = (row["pnl"] / row["start_value"]) if row and row["start_value"] else nm
    cost = SWITCH_COST if switched else 0.0
    start_m = book_value(conn, "M")
    end_m = start_m * (1 + r - cost)
    conn.execute("INSERT OR REPLACE INTO daily_ledger VALUES (?,?,?,?,?,?,?,?)",
                 (dstr, "M", start_m, end_m, end_m - start_m, start_m * cost, n_move, replayed))


def scores(conn: sqlite3.Connection, upto: str) -> Dict[str, Dict[str, float]]:
    """Each candidate's last WINDOW settled sessions against Nifty, with penalties for trailing it."""
    led = pd.read_sql("SELECT date, book, start_value, pnl, nifty_move FROM daily_ledger WHERE date <= ?", conn, params=(upto,))
    out = {}
    for b in CANDIDATES:
        l = led[led["book"] == b].sort_values("date").tail(WINDOW)
        if len(l) < MIN_HISTORY:
            continue
        ex = (l["pnl"] / l["start_value"]).to_numpy() - l["nifty_move"].fillna(0.0).to_numpy()
        short = np.clip(-ex, 0, None)
        out[b] = {"excess": round(float(ex.sum()), 5), "penalty": round(float(PENALTY * short.sum()), 5),
                  "score": round(float(ex.sum() - PENALTY * short.sum()), 5), "days_behind": int((ex < -1e-9).sum()), "n": int(len(l))}
    return out


def decide(conn: sqlite3.Connection, day: str, replayed: int) -> Dict[str, Any]:
    """Tonight's champion for the main book, and whether to retrain."""
    sc = scores(conn, day)
    prev = conn.execute("SELECT champion FROM daily_governor WHERE date < ? ORDER BY date DESC LIMIT 1", (day,)).fetchone()
    current = prev["champion"] if prev else "N"
    champ, reason = current, "keeps the lead"
    if sc:
        best = max(sc, key=lambda b: sc[b]["score"])
        cur_score = sc.get(current, {}).get("score", -np.inf)
        if best != current and sc[best]["score"] > cur_score + SWITCH_MARGIN:
            champ = best
            reason = (f"switch: {BOOK_NAMES[best]} scores {sc[best]['score']:+.2%} vs {BOOK_NAMES.get(current, current)} "
                      f"{cur_score:+.2%}" if np.isfinite(cur_score) else f"start: {BOOK_NAMES[best]} leads")
        if champ == "N" and best == "N":
            reason = "every strategy trails Nifty after penalties: holding Nifty"
    m = conn.execute("SELECT start_value, pnl, nifty_move FROM daily_ledger WHERE book = 'M' AND date <= ? ORDER BY date DESC LIMIT ?",
                     (day, WINDOW)).fetchall()
    m_ex = sum(r["pnl"] / r["start_value"] - (r["nifty_move"] or 0.0) for r in m) if len(m) >= MIN_HISTORY else 0.0
    last_rt = conn.execute("SELECT COUNT(*) FROM daily_governor WHERE date > COALESCE((SELECT MAX(date) FROM daily_governor "
                           "WHERE retrain = 1 AND date < ?), '') AND date < ?", (day, day)).fetchone()[0]
    has_rt = conn.execute("SELECT 1 FROM daily_governor WHERE retrain = 1 AND date < ?", (day,)).fetchone()
    retrain = int(m_ex < RETRAIN_BELOW and (not has_rt or last_rt >= RETRAIN_GAP))
    if retrain:
        reason += f"; main book trails Nifty by {m_ex:+.2%} over {len(m)} sessions: retrain requested"
    conn.execute("INSERT OR REPLACE INTO daily_governor VALUES (?,?,?,?,?,?,?)",
                 (day, champ, int(champ != current), reason, json.dumps(sc), retrain, replayed))
    return {"champion": champ, "reason": reason, "retrain": retrain, "scores": sc}


# ---------- driver --------------------------------------------------------------------------

def _frames_with_close(since: str = SINCE):
    frames, w = load_frames(since)
    px = raw(w, "close").stack()
    px.index.names = ["date", "symbol"]
    return {h: f.join(px.rename("close_px"), on=["date", "symbol"]) for h, f in frames.items()}, w


def run(replay: int = 0, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    from code.research import fund
    conn = conn or connect()
    if not replay:
        update_data()
    frames, w = _frames_with_close()
    d1 = frames[1]
    universe = nifty100()
    specs = fund.current_specs(conn) if not replay else {}
    specs = {h: specs.get(h) or best_spec(h) for h in (1, 5, 20)}
    from code.research.portfolio import NIFTY
    nifty = pd.read_parquet(NIFTY)["nifty"] if NIFTY.exists() else None
    dates = pd.Index(sorted(d1["date"].unique()))
    gaps = missing_sessions(pd.DatetimeIndex(dates), dates[-1] - pd.Timedelta(days=400), dates[-1])
    if gaps:   # never settle or predict across a hole: the numbers would be silently wrong
        raise RuntimeError(f"NSE data missing for {len(gaps)} trading day(s): {', '.join(gaps[:10])}. "
                           "Run: python -m code.research.bhavcopy <first missing date>")
    if replay:
        clear_replayed(conn)
        conn.execute("DELETE FROM daily_meta WHERE k IN ('b_last_rebalance', 's_last_rebalance')")
        days = dates[-(replay + 1):]
        for asof, nxt in zip(days[:-1], days[1:]):
            predict_day(conn, frames, asof, 1, specs, universe)
            settle_day(conn, w, asof, nxt, 1, nifty)
            if GOVERNOR_ON:
                decide(conn, nxt.strftime("%Y-%m-%d"), 1)
            conn.commit()
            logger.info("replayed %s -> %s", asof.date(), nxt.date())
        predict_day(conn, frames, days[-1], 1, specs, universe)
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
            g = decide(conn, later[0].strftime("%Y-%m-%d"), 0) if GOVERNOR_ON else {"retrain": 0}
            conn.commit()
            if g["retrain"]:
                retrain(conn, dates[-1])
                specs[1] = best_spec(1)
    # the fund manager: born on the first live evening, replaced (with new model setups) when it dies
    fund.ensure_agent(conn, dates[-1].strftime("%Y-%m-%d"))
    specs = {h: fund.current_specs(conn).get(h) or specs[h] for h in (1, 5, 20)}
    predict_day(conn, frames, dates[-1], 0, specs, universe)
    conn.commit()
    return summary(conn)


def retrain(conn: sqlite3.Connection, today: pd.Timestamp) -> None:
    """Penalties piled up: re-run the 1-day model search choosing on the most recent year, so the
    combination of models / inputs / target can change. Runs in its own process (~15 min)."""
    import subprocess
    import sys as _sys
    frm = (today - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
    to = (today - pd.Timedelta(days=7)).strftime("%Y-%m-%d")       # last week's outcomes aren't all settled
    logger.info("retraining: model search on %s..%s", frm, to)
    r = subprocess.run([_sys.executable, "-m", "code.research.model_search", "--horizon", "1", "--valid", f"{frm}:{to}"],
                       cwd=str(DATA.parents[1]), capture_output=True, text=True, timeout=3 * 3600)
    conn.execute("INSERT OR REPLACE INTO daily_meta VALUES ('last_retrain', ?)",
                 (json.dumps({"date": today.strftime("%Y-%m-%d"), "window": [frm, to], "exit": r.returncode,
                              "result": r.stdout.strip().splitlines()[-2:] if r.stdout else []}),))
    conn.commit()


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
        "governor": rows("SELECT date, champion, switched, reason, scores, retrain, replayed FROM daily_governor ORDER BY date"),
        "fund": __import__("code.research.fund", fromlist=["report"]).report(conn),
        "last_retrain": (lambda r: json.loads(r[0]) if r else None)(conn.execute("SELECT v FROM daily_meta WHERE k = 'last_retrain'").fetchone()),
        "rules": {"governor_on": GOVERNOR_ON, "window": WINDOW, "penalty": PENALTY, "switch_margin": SWITCH_MARGIN, "switch_cost": SWITCH_COST,
                  "retrain_below": RETRAIN_BELOW, "book_names": BOOK_NAMES},
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
    for b in ("M", *INTRADAY_BOOKS, "S", "B", "N"):
        l = led[led["book"] == b]
        if len(l):
            books[b] = {"name": BOOK_NAMES[b], "now": round(float(l["end_value"].iloc[-1]), 2), "pnl": round(float(l["end_value"].iloc[-1] - START_CASH), 2),
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
