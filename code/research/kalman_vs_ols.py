"""OLS vs Kalman hedge ratio on the same pairs, same rules, same costs -- walk-forward, no lookahead.

Pairs: the cached Engle-Granger survivors with a positive hedge ratio, plus ICICIBANK/SBIN.
Data: Yahoo adjusted daily closes from 2016. The first FORMATION days only warm the models up.

  OLS:    hedge = OLS slope of A on B over the trailing FORMATION days, refitted every REFIT days;
          z = today's spread vs its trailing WINDOW-day mean/std, using today's hedge.
  Kalman: code/signals/kalman_pairs.py; z = today's forecast error / its predicted std.

Both: enter when |z| > ENTRY (short the spread when z > 0), exit when |z| < EXIT or |z| > STOP.
Each trade keeps its entry hedge (1 share of A against beta shares of B, scaled to 1 rupee of gross
exposure) and pays COST of notional on entry and on exit. Sharpe is annualised from daily returns;
the portfolio row averages the pairs' daily returns (equal capital per pair).

Caveat: the pairs were picked by a cointegration scan over 2023-26, so that part of the test is
in-sample for both methods. The comparison between methods is fair; absolute numbers flatter both.

Run: python -m code.research.kalman_vs_ols
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from code.data.fetchers.equity import fetch_ohlcv
from code.signals.kalman_pairs import kalman_pair

OUT = Path(__file__).resolve().parents[1] / "data" / "kalman_vs_ols.json"
CACHE = Path(__file__).resolve().parents[1] / "data" / "pairs_cointegration_cache.json"
START = "2016-01-01"
FORMATION, REFIT, WINDOW = 252, 21, 60
ENTRY, EXIT, STOP = 2.0, 0.5, 3.5
COST = 0.001


def ols_signals(a: np.ndarray, b: np.ndarray):
    n = len(a)
    beta, z = np.full(n, np.nan), np.full(n, np.nan)
    hb = np.nan
    for t in range(FORMATION, n):
        if (t - FORMATION) % REFIT == 0:
            hb = np.polyfit(b[t - FORMATION + 1:t + 1], a[t - FORMATION + 1:t + 1], 1)[0]
        s = a[t - WINDOW + 1:t + 1] - hb * b[t - WINDOW + 1:t + 1]
        beta[t], z[t] = hb, (s[-1] - s.mean()) / s.std(ddof=1)
    return beta, z


def backtest(a: np.ndarray, b: np.ndarray, beta: np.ndarray, z: np.ndarray) -> dict:
    n = len(a)
    daily = np.zeros(n)
    trades, pos = [], None              # pos = (side, beta, gross, entry_index, running pnl)
    for t in range(FORMATION, n):
        if pos is not None:
            side, hb, gross, t0, pnl = pos
            r = side * ((a[t] - a[t - 1]) - hb * (b[t] - b[t - 1])) / gross
            daily[t] += r
            pnl += r
            if abs(z[t]) < EXIT or abs(z[t]) > STOP or t == n - 1:
                daily[t] -= COST
                trades.append((pnl - 2 * COST, t - t0))
                pos = None
            else:
                pos = (side, hb, gross, t0, pnl)
        if pos is None and t < n - 1 and abs(z[t]) > ENTRY and abs(z[t]) <= STOP and beta[t] > 0:
            side = -1 if z[t] > 0 else 1
            pos = (side, beta[t], a[t] + beta[t] * b[t], t, 0.0)
            daily[t] -= COST
    return {"daily": daily[FORMATION:], "trades": trades}


def stats(daily: np.ndarray, trades: list) -> dict:
    eq = np.cumsum(daily)
    sd = daily.std(ddof=1)
    return {
        "sharpe": round(float(daily.mean() / sd * np.sqrt(252)), 2) if sd > 0 else None,
        "max_dd_pct": round(float((eq - np.maximum.accumulate(eq)).min() * 100), 1),
        "total_return_pct": round(float(eq[-1] * 100), 1),
        "trades": len(trades),
        "win_rate": round(float(np.mean([p > 0 for p, _ in trades])), 2) if trades else None,
        "avg_hold_days": round(float(np.mean([d for _, d in trades])), 1) if trades else None,
    }


def build() -> dict:
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    pairs = [(p["ticker_a"], p["ticker_b"]) for p in cache["pairs"] if p["hedge_ratio"] > 0]
    if ("ICICIBANK.NS", "SBIN.NS") not in pairs:
        pairs.append(("ICICIBANK.NS", "SBIN.NS"))
    end = date.today().isoformat()
    rows, port = {}, {"OLS": [], "Kalman": []}
    for ta, tb in pairs:
        pa, pb = fetch_ohlcv(ta, START, end)["close"], fetch_ohlcv(tb, START, end)["close"]
        pa, pb = pa.align(pb, join="inner")
        keep = pa.notna() & pb.notna()
        pa, pb = pa[keep], pb[keep]
        if len(pa) < FORMATION + 120:
            continue
        a, b = pa.to_numpy(float), pb.to_numpy(float)
        k = kalman_pair(pa, pb)
        res = {"OLS": backtest(a, b, *ols_signals(a, b)), "Kalman": backtest(a, b, k["beta"].to_numpy(), k["zscore"].to_numpy())}
        name = f"{ta.removesuffix('.NS')}/{tb.removesuffix('.NS')}"
        rows[name] = {m: stats(r["daily"], r["trades"]) for m, r in res.items()}
        rows[name]["period"] = f"{pa.index[FORMATION]:%Y-%m-%d} to {pa.index[-1]:%Y-%m-%d}"
        for m, r in res.items():
            port[m].append(pd.Series(r["daily"], index=pa.index[FORMATION:]))
    portfolio = {}
    for m, series in port.items():
        df = pd.concat(series, axis=1).fillna(0.0)
        all_trades = sum((rows[p][m]["trades"] for p in rows), 0)
        s = stats(df.mean(axis=1).to_numpy(), [])
        s["trades"] = all_trades
        s["win_rate"] = round(float(np.average([rows[p][m]["win_rate"] or 0 for p in rows], weights=[rows[p][m]["trades"] or 0 for p in rows])), 2) if all_trades else None
        portfolio[m] = s
    return {"generated": end, "rules": {"entry_z": ENTRY, "exit_z": EXIT, "stop_z": STOP, "cost_per_side": COST,
            "formation_days": FORMATION, "ols_refit_days": REFIT, "ols_z_window": WINDOW},
            "pairs": rows, "portfolio": portfolio}


if __name__ == "__main__":
    data = build()
    OUT.write_text(json.dumps(data, indent=1), encoding="utf-8")
    print(f"{'pair':<26}{'':>4}{'Sharpe':>8}{'MaxDD%':>8}{'Ret%':>8}{'Trades':>8}{'Win':>6}{'Hold':>6}")
    for name, r in list(data["pairs"].items()) + [("PORTFOLIO", data["portfolio"])]:
        for m in ("OLS", "Kalman"):
            s = r[m]
            print(f"{name if m == 'OLS' else '':<26}{m:>6}{s['sharpe'] if s['sharpe'] is not None else '-':>8}{s['max_dd_pct']:>8}{s['total_return_pct']:>8}{s['trades']:>8}{s['win_rate'] if s['win_rate'] is not None else '-':>6}{s.get('avg_hold_days') or '-':>6}")
