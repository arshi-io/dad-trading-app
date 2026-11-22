import sqlite3

import numpy as np
import pandas as pd
import pytest

from code.research import daily as dl
from code.research import model_search as ms

DAYS = pd.bdate_range("2026-01-05", periods=3)
SYMS = ["AAA", "BBB", "CCC", "DDD"]


def _w(opens, closes, cum=None):
    o = pd.DataFrame(opens, index=DAYS, columns=SYMS, dtype=float)
    c = pd.DataFrame(closes, index=DAYS, columns=SYMS, dtype=float)
    k = pd.DataFrame(1.0, index=DAYS, columns=SYMS) if cum is None else cum
    return {"open": o / k, "close": c / k, "cum": k}          # stored split-adjusted, like the real panel


@pytest.fixture
def conn(tmp_path):
    return dl.connect(tmp_path / "t.db")


def _predict(conn, asof, rows):
    for sym, call, close, lo, hi in rows:
        conn.execute("INSERT INTO daily_predictions (asof, symbol, replayed, score, rank, call, exp_move, close, lo68, hi68, vol)"
                     " VALUES (?,?,0,0,0,?,0,?,?,?,0.02)", (asof, sym, call, close, lo, hi))


def test_intraday_and_delivery_charges():
    c = dl.intraday_charges(2000, 2020)
    brk = 0.0003 * 4020
    exch = 0.0000297 * 4020 + 1e-6 * 4020
    assert c == pytest.approx(brk + exch + 0.18 * (brk + exch) + 0.00025 * 2020 + 0.00003 * 2000)
    assert dl.delivery_charges(0, 2000) - dl.delivery_charges(0, 0.0001) == pytest.approx(2.0 + 0.0000297 * 2000 * 1.18 + 1e-6 * 2000 * 1.18, rel=1e-3)
    assert dl.delivery_charges(0, 2000) > 15.93                 # the DP charge on every stock sold


def test_predictions_are_frozen_and_live_ones_never_deleted(conn):
    _predict(conn, "2026-01-05", [("AAA", "BEAT", 100, 98, 102)])
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE daily_predictions SET score = 9 WHERE symbol = 'AAA'")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("DELETE FROM daily_predictions WHERE symbol = 'AAA'")
    conn.execute("UPDATE daily_predictions SET move = 0.01 WHERE symbol = 'AAA'")
    with pytest.raises(sqlite3.DatabaseError):
        conn.execute("UPDATE daily_predictions SET move = 0.02 WHERE symbol = 'AAA'")   # settled once


def test_settle_crosses_out_wrong_calls_and_books_intraday_pnl(conn):
    w = _w([[100, 100, 100, 100], [100, 200, 50, 400], [100] * 4],
           [[100, 100, 100, 100], [110, 200, 50, 380], [100] * 4])
    a = "2026-01-05"
    _predict(conn, a, [("AAA", "BEAT", 100, 95, 105), ("BBB", "BEAT", 100, 150, 250),
                       ("CCC", "LAG", 100, 40, 60), ("DDD", "BEAT", 100, 390, 410)])
    conn.execute("INSERT INTO daily_plan VALUES (?,?,?,?,?,?,?,?)", (a, "A", "AAA", "BUY_OPEN_SELL_CLOSE", 20, 100, 1, 0))
    dl.settle_day(conn, w, DAYS[0], DAYS[1], 0)
    p = {r["symbol"]: r for r in conn.execute("SELECT * FROM daily_predictions")}
    # moves: AAA +10%, BBB 0, CCC 0, DDD -5%; middle stock 0
    assert p["AAA"]["right"] == 1 and p["DDD"]["right"] == 0           # DDD called BEAT, lagged: crossed out
    assert p["AAA"]["in_range"] == 0 and p["BBB"]["in_range"] == 1       # AAA closed 110, range 95-105
    led = conn.execute("SELECT * FROM daily_ledger WHERE book = 'A'").fetchone()
    sh = int((dl.START_CASH / dl.N_A) // (100 * (1 + dl.SLIP_A)))
    buy, sell = 100 * (1 + dl.SLIP_A), 110 * (1 - dl.SLIP_A)
    expect = sh * (sell - buy) - dl.intraday_charges(sh * buy, sh * sell)
    assert led["pnl"] == pytest.approx(expect) and led["end_value"] == pytest.approx(dl.START_CASH + expect)


def test_book_b_buys_at_open_and_marks_at_close(conn):
    w = _w([[100] * 4, [100, 100, 100, 100], [100] * 4], [[100] * 4, [120, 100, 100, 100], [100] * 4])
    a = "2026-01-05"
    conn.execute("INSERT INTO daily_plan VALUES (?,?,?,?,?,?,?,?)", (a, "B", "AAA", "BUY_OPEN", 19, 100, 1, 0))
    dl.settle_day(conn, w, DAYS[0], DAYS[1], 0)
    held, cash = dl.holdings(conn, "B")
    px = 100 * (1 + dl.SLIP_B)
    assert held == {"AAA": 19} and cash == pytest.approx(dl.START_CASH - 19 * px - dl.delivery_charges(19 * px, 0))
    led = conn.execute("SELECT * FROM daily_ledger WHERE book = 'B'").fetchone()
    assert led["end_value"] == pytest.approx(cash + 19 * 120)


def test_a_split_doubles_the_shares_held_without_a_loss(conn):
    cum = pd.DataFrame(1.0, index=DAYS, columns=SYMS)
    cum.loc[DAYS[2]:, "AAA"] = 0.5                                     # 1:1 bonus before day 3's open
    w = _w([[100] * 4, [100] * 4, [50, 100, 100, 100]], [[100] * 4, [100] * 4, [50, 100, 100, 100]], cum)
    conn.execute("INSERT INTO daily_trades VALUES (?,?,?,?,?,?,?,?)", ("2026-01-06", "B", "AAA", "BUY", 10, 100.0, 0.0, 0))
    conn.execute("INSERT INTO daily_ledger VALUES (?,?,?,?,?,?,?,?)", ("2026-01-06", "B", 10000, 10000, 0, 0, None, 0))
    dl.settle_day(conn, w, DAYS[1], DAYS[2], 0)
    assert dl.holdings(conn, "B")[0] == {"AAA": 20}
    assert conn.execute("SELECT end_value FROM daily_ledger WHERE date = ? AND book = 'B'",
                        (DAYS[2].strftime("%Y-%m-%d"),)).fetchone()[0] == pytest.approx(10000)


def test_sync_endpoint_needs_the_token_and_serves_the_pushed_report(monkeypatch, tmp_path, conn):
    from fastapi.testclient import TestClient
    from code.app import main
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "railway.db")
    monkeypatch.setattr(main, "APP_PIN_HASH", "$2b$12$notarealhashbutsetsoauthison")
    client = TestClient(main.app)
    _predict(conn, "2026-01-05", [("AAA", "BEAT", 100, 95, 105)])
    body = dl.report(conn)
    monkeypatch.delenv("DAILY_SYNC_TOKEN", raising=False)
    assert client.post("/api/daily/sync", json=body, headers={"X-Sync-Token": "x"}).status_code == 403   # no token set: off
    monkeypatch.setenv("DAILY_SYNC_TOKEN", "s3cret-token")
    assert client.post("/api/daily/sync", json=body, headers={"X-Sync-Token": "wrong"}).status_code == 403
    assert client.post("/api/daily/sync", json={"x": 1}, headers={"X-Sync-Token": "s3cret-token"}).status_code == 400
    assert client.post("/api/daily/sync", json=body, headers={"X-Sync-Token": "s3cret-token"}).json()["ok"]
    assert client.get("/api/daily").status_code == 401                       # reading still needs the PIN session
    monkeypatch.setattr(main, "APP_PIN_HASH", None)
    got = client.get("/api/daily").json()
    assert got["predictions"][0]["symbol"] == "AAA" and "synced_at" in got


def test_fit_only_uses_outcomes_settled_by_the_evening(monkeypatch):
    dates = pd.bdate_range("2026-01-01", periods=30)
    df = pd.DataFrame([(d, s) for d in dates for s in range(40)], columns=["date", "symbol"])
    for c in ms.columns(ms.BASELINE):
        df[c] = np.random.default_rng(0).normal(size=len(df))
    pos = {d: i for i, d in enumerate(dates)}
    df["label_date"] = [dates[pos[d] + 1] if pos[d] + 1 < len(dates) else pd.NaT for d in df["date"]]
    df["up"] = np.where(df["label_date"].notna(), (np.arange(len(df)) % 2).astype(float), np.nan)
    seen = []

    class Spy:
        def fit(self, X, y):
            seen.append(df.loc[X.index, "label_date"].max())
            return self

        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

    monkeypatch.setattr(ms, "estimator", lambda spec: Spy())
    asof = dates[20]
    dl.fit_score(df, ms.BASELINE, asof, df[df["date"] == asof])
    assert seen[0] <= asof
