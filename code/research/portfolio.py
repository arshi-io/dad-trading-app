"""Phase 1: turn the daily stock ranking into a weekly portfolio, with real costs and a Nifty hedge.

Rules (fixed before looking at results, never tuned on 2024+):
- Every 5 sessions, after the close, read the ranking. Buy at the next open.
- Hold N stocks equal-weight. A held stock stays while its rank is in the top EXIT_RANK; empty slots
  are filled with the best-ranked stocks not held. This buffer cuts needless trading.
- Costs per side: STT 0.1%, stamp duty 0.015% (buy), exchange fees, plus 0.1% slippage.
- Hedges: none; short Nifty futures sized to the portfolio's trailing 60-day beta (past days only);
  or monthly 5%-out-of-the-money Nifty puts priced from India VIX (+3 vol points for skew).

Run:  python -m code.research.portfolio      (writes code/data/portfolio.json; laptop only)
"""
from __future__ import annotations

import json
import logging
import math
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA = Path(__file__).resolve().parents[1] / "data"
HIST = DATA / "history"
OUT = DATA / "portfolio.json"
NIFTY = HIST / "nifty_vix.parquet"
PANEL = HIST / "panel_oc.parquet"

N = 20
EXIT_RANK = 0.80
REBAL = 5
FEES_BUY = 0.001 + 0.00015 + 0.00004          # STT + stamp + exchange/SEBI
FEES_SELL = 0.001 + 0.00004
SLIP = 0.001                                   # default slippage per side
BUY, SELL = FEES_BUY + SLIP, FEES_SELL + SLIP
RF = 0.065                                     # Indian T-bill, used in Sharpe and futures carry
FUT_ROLL = 0.0004                              # futures STT + slippage per monthly roll, on hedge notional
BETA_DAYS, BETA_MAX = 60, 1.5
PUT_STRIKE, PUT_DAYS, PUT_SKEW, PUT_SLIP = 0.95, 21, 0.03, 0.05
TOP_LIQUID = 200
# books: (name, ranking column in the replay file, rebalance every k sessions, keep while rank >= x).
# Position books (Phase 2) trade monthly with a looser buffer. "combined" = equal thirds of the three
# search winners, each sleeve trading by its own rules. All fixed before any Phase 2 result was seen.
BOOKS = [("current", "current", REBAL, EXIT_RANK), ("best", "best", REBAL, EXIT_RANK),
         ("best_h20", "best_h20", 20, 0.70), ("best_h60", "best_h60", 20, 0.70)]
COMBINED = ("best", "best_h20", "best_h60")
PERIODS = {"2015-20": ("2015-01-01", "2021-01-01"), "2021-23": ("2021-01-01", "2024-01-01"),
           "2024+": ("2024-01-01", "2100-01-01")}


# ---------- engine --------------------------------------------------------------------------

def simulate(ranks: pd.DataFrame, O: pd.DataFrame, C: pd.DataFrame, n: int = N, exit_rank: float = EXIT_RANK,
             rebal: int = REBAL, slip: float | pd.DataFrame = SLIP) -> Dict[str, Any]:
    """Daily net returns of the long book. ranks: date x symbol, 0..1 (NaN = not rankable that day).
    The ranking read after the close of day t is traded at the open of t+1: the overnight gap into
    t+1 belongs to the old book, t+1's open-to-close to the new one. slip: slippage per side, one
    number or a date x symbol frame (read on the signal day, so it uses past data only)."""
    dates = ranks.index.intersection(C.index)
    O, C = O.reindex(index=dates), C.reindex(index=dates)
    w = pd.Series(dtype=float)
    grow: Dict[str, float] = {}
    entry: Dict[str, float] = {}                   # buy cost paid per held stock
    out, turn, trades = [], [], []
    signal = {dates[i] for i in range(0, len(dates), rebal)}
    for i in range(1, len(dates)):
        d, prev = dates[i], dates[i - 1]
        if prev in signal and ranks.loc[prev].notna().sum() >= n:
            on = (O.loc[d, w.index] / C.loc[prev, w.index] - 1).fillna(0.0) if len(w) else pd.Series(dtype=float)
            r_on = float((w * on).sum())
            if len(w):
                w = w * (1 + on) / (1 + r_on)
                for s in w.index:
                    grow[s] *= 1 + on[s]
            r = ranks.loc[prev].dropna()
            keep = [s for s in w.index if r.get(s, 0.0) >= exit_rank]
            add = [s for s in r.sort_values(ascending=False).index if s not in keep][: n - len(keep)]
            hold = keep + add
            t = pd.Series(1.0 / len(hold), index=hold)
            delta = t.sub(w, fill_value=0.0)
            sl = (slip.loc[prev].reindex(delta.index).fillna(SLIP) if isinstance(slip, pd.DataFrame)
                  else pd.Series(slip, index=delta.index))
            bought, sold = delta.clip(lower=0), (-delta).clip(lower=0)
            buys, sells = float(bought.sum()), float(sold.sum())
            cost = float((bought * (FEES_BUY + sl)).sum() + (sold * (FEES_SELL + sl)).sum())
            for s in w.index.difference(t.index):
                trades.append({"symbol": s, "exit": d,
                               "ret": grow.pop(s) * (1 - FEES_SELL - sl[s]) / (1 + entry.pop(s)) - 1})
            for s in t.index.difference(w.index):
                grow[s] = 1.0
                entry[s] = FEES_BUY + sl[s]
            intra = (C.loc[d, t.index] / O.loc[d, t.index] - 1).fillna(0.0)
            r_id = float((t * intra).sum())
            for s in t.index:
                grow[s] *= 1 + intra[s]
            w = t * (1 + intra) / (1 + r_id)
            out.append((1 + r_on) * (1 + r_id) - 1 - cost)
            turn.append(buys + sells)
        else:
            if len(w):
                ret = (C.loc[d, w.index] / C.loc[prev, w.index] - 1).fillna(0.0)
                rp = float((w * ret).sum())
                w = w * (1 + ret) / (1 + rp)
                for s in w.index:
                    grow[s] *= 1 + ret[s]
            else:
                rp = 0.0
            out.append(rp)
            turn.append(0.0)
    idx = dates[1:]
    return {"ret": pd.Series(out, index=idx), "turnover": pd.Series(turn, index=idx),
            "trades": pd.DataFrame(trades, columns=["symbol", "exit", "ret"])}


def combine(sims: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Equal capital in each sleeve, re-split daily, over the days every sleeve exists."""
    rets = pd.concat([s["ret"] for s in sims], axis=1).dropna()
    turn = pd.concat([s["turnover"] for s in sims], axis=1).reindex(rets.index).fillna(0.0)
    k = len(sims)
    trades = pd.concat([s["trades"] for s in sims], ignore_index=True)
    return {"ret": rets.mean(axis=1), "turnover": turn.sum(axis=1) / k, "trades": trades}


def hedge_futures(port: pd.Series, nifty: pd.Series) -> pd.Series:
    """Short Nifty futures worth beta x the book. Beta from the previous 60 days only. A short future
    returns -(Nifty move - carry), carry ~ the risk-free rate."""
    nr = nifty.pct_change().reindex(port.index).fillna(0.0)
    beta = (port.rolling(BETA_DAYS).cov(nr) / nr.rolling(BETA_DAYS).var()).shift(1).clip(0, BETA_MAX).fillna(1.0)
    roll = pd.Series(np.where(port.index.to_series().dt.month.diff().fillna(1) != 0, FUT_ROLL, 0.0), index=port.index)
    return port - beta * (nr - RF / 252) - beta * roll


def bs_put(S: float, K: float, T: float, sigma: float, r: float = RF) -> float:
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    N_ = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    return K * math.exp(-r * T) * N_(-d2) - S * N_(-d1)


def hedge_puts(port: pd.Series, nifty: pd.Series, vix: pd.Series) -> pd.Series:
    """Each month buy Nifty puts 5% below spot, 21 sessions to expiry, on notional = trailing beta x
    book. Premium is paid on the buy day; the payoff (if any) lands on expiry day."""
    out = port.copy()
    nr = nifty.pct_change().reindex(port.index).fillna(0.0)
    beta = (port.rolling(BETA_DAYS).cov(nr) / nr.rolling(BETA_DAYS).var()).shift(1).clip(0, BETA_MAX).fillna(1.0)
    S, V = nifty.reindex(port.index).ffill(), vix.reindex(port.index).ffill()
    idx = port.index
    new_month = idx.to_series().dt.month.diff().fillna(1).ne(0).to_numpy()
    for i in np.flatnonzero(new_month):
        s0, k = float(S.iloc[i]), PUT_STRIKE * float(S.iloc[i])
        prem = bs_put(s0, k, PUT_DAYS / 252, float(V.iloc[i]) / 100 + PUT_SKEW) * (1 + PUT_SLIP)
        b = float(beta.iloc[i])
        out.iloc[i] -= b * prem / s0
        j = min(i + PUT_DAYS, len(idx) - 1)
        out.iloc[j] += b * max(k - float(S.iloc[j]), 0.0) / s0
    return out


# ---------- scoring -------------------------------------------------------------------------

def metrics(r: pd.Series, turnover: Optional[pd.Series] = None, trades: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    r = r.dropna()
    if len(r) < 20:
        return {}
    yrs = len(r) / 252
    eq = (1 + r).cumprod()
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    out = {"cagr": round(float(eq.iloc[-1] ** (1 / yrs) - 1), 4), "vol": round(float(r.std() * math.sqrt(252)), 4),
           "sharpe": round(float((r.mean() * 252 - RF) / (r.std() * math.sqrt(252))), 2) if r.std() > 0 else 0.0,
           "max_dd": round(float((eq / eq.cummax() - 1).min()), 4),
           "months_up": round(float((m > 0).mean()), 3), "worst_month": round(float(m.min()), 4)}
    if turnover is not None:
        out["turnover_yr"] = round(float(turnover.reindex(r.index).sum() / yrs), 2)
    if trades is not None and len(trades):
        t = trades[(trades["exit"] >= r.index[0]) & (trades["exit"] <= r.index[-1])]
        if len(t):
            out.update(trades=int(len(t)), avg_trade=round(float(t["ret"].mean()), 4),
                       trade_hit=round(float((t["ret"] > 0).mean()), 3))
    return out


def yearly(r: pd.Series) -> Dict[str, float]:
    return {str(y): round(float(v), 4) for y, v in ((1 + r).groupby(r.index.year).prod() - 1).items()}


def curve(r: pd.Series, step: int = 5) -> List[List[Any]]:
    eq = (1 + r).cumprod()
    pts = eq.iloc[::step]
    if pts.index[-1] != eq.index[-1]:
        pts = pd.concat([pts, eq.iloc[-1:]])
    return [[d.strftime("%Y-%m-%d"), round(float(v), 4)] for d, v in pts.items()]


# ---------- slippage stress -----------------------------------------------------------------

# half the bid-ask spread by 50-day median turnover: >= ₹100 cr, >= ₹20 cr, the rest (all >= ₹5 cr)
SPREAD_TIERS = [(1e9, 0.0003), (2e8, 0.0008), (0.0, 0.002)]
IMPACT_K = 0.5                 # square-root impact: K x daily vol x sqrt(order / daily turnover)
CAPITALS = [1e6, 5e6, 2e7]     # ₹10 lakh, ₹50 lakh, ₹2 crore books
FLAT = [0.001, 0.002, 0.003, 0.005]


def size_slippage(t50: pd.DataFrame, vol: pd.DataFrame, capital: float, n: int = N) -> pd.DataFrame:
    """Slippage per side for an order of capital/n in each stock: half-spread + market impact.
    Uses the stock's own past turnover and volatility."""
    half = pd.DataFrame(SPREAD_TIERS[-1][1], index=t50.index, columns=t50.columns)
    for lo, s in reversed(SPREAD_TIERS[:-1]):
        half = half.mask(t50 >= lo, s)
    impact = IMPACT_K * vol * np.sqrt((capital / n) / t50)
    return (half + impact).clip(upper=0.03)


def breakeven(points: List[Dict[str, Any]], period: str, target: float) -> Optional[float]:
    """Flat slippage per side at which the period's yearly return falls to target (linear between
    points). None if it never crosses within the tested range (check the last point to see which way)."""
    xs = [(p["slip"], p["periods"][period]["cagr"]) for p in points if p["periods"].get(period)]
    for (x0, y0), (x1, y1) in zip(xs, xs[1:]):
        if y0 >= target > y1:
            return round(x0 + (y0 - target) * (x1 - x0) / (y0 - y1), 4)
    return None


def stress(model: str = "best", hedge: str = "Nifty futures") -> Dict[str, Any]:
    from code.research import model_search as ms
    rp = pd.read_parquet(ms.REPLAY)
    p = load_panel()
    O, C, t50 = p["open"].astype("float64"), p["close"].astype("float64").ffill(), p["t50"].astype("float64")
    vol = C.pct_change(fill_method=None).rolling(20, min_periods=15).std().shift(1)
    nv = pd.read_parquet(NIFTY)
    ranks = rp.pivot(index="date", columns="symbol", values=model).dropna(how="all")
    rebal, exit_rank = next(((rb, ex) for m, _, rb, ex in BOOKS if m == model), (REBAL, EXIT_RANK))
    sim_ = lambda rk, s: simulate(rk, O, C, rebal=rebal, exit_rank=exit_rank, slip=s)
    t50, vol = t50.reindex(index=ranks.index, columns=ranks.columns), vol.reindex(index=ranks.index, columns=ranks.columns)
    top = t50.rank(axis=1, ascending=False) <= TOP_LIQUID
    hedged = (lambda r: hedge_futures(r, nv["nifty"])) if hedge == "Nifty futures" else (lambda r: r)
    out: Dict[str, Any] = {"model": model, "hedge": hedge, "spread_tiers": SPREAD_TIERS, "impact_k": IMPACT_K, "universes": {}}
    for uni, rk in (("all liquid", ranks), (f"top {TOP_LIQUID} by turnover", ranks.where(top).rank(axis=1, pct=True))):
        flat, sized = [], []
        for s in FLAT:
            r = hedged(sim_(rk, s)["ret"])
            flat.append({"slip": s, "periods": {k: metrics(r[(r.index >= a) & (r.index < b)]) for k, (a, b) in PERIODS.items()}})
            logger.info("%s flat %.1f%%: 2024+ %s", uni, s * 100, flat[-1]["periods"]["2024+"].get("cagr"))
        for cap in CAPITALS:
            sl = size_slippage(t50, vol, cap)
            sim = sim_(rk, sl)
            r = hedged(sim["ret"])
            sized.append({"capital": cap,
                          "median_slip": round(float(np.nanmedian(sl.where(rk >= EXIT_RANK).to_numpy())), 5),
                          "periods": {k: metrics(r[(r.index >= a) & (r.index < b)], sim["turnover"], sim["trades"])
                                      for k, (a, b) in PERIODS.items()}})
            logger.info("%s ₹%.0fL: median slip %.2f%%, 2024+ %s", uni, cap / 1e5, sized[-1]["median_slip"] * 100,
                        sized[-1]["periods"]["2024+"].get("cagr"))
        out["universes"][uni] = {"flat": flat, "sized": sized}
    nr = nv["nifty"].pct_change().dropna()
    n24 = metrics(nr[nr.index >= pd.Timestamp(PERIODS["2024+"][0])])["cagr"]
    for u in out["universes"].values():
        u["breakeven_vs_nifty_2024"] = breakeven(u["flat"], "2024+", n24)
        u["breakeven_zero_2024"] = breakeven(u["flat"], "2024+", 0.0)
    return out


# ---------- data and run --------------------------------------------------------------------

def load_panel() -> Dict[str, pd.DataFrame]:
    if PANEL.exists():
        long = pd.read_parquet(PANEL)
    else:
        from code.research import bhavcopy, updown
        w = updown.panel(bhavcopy.load())
        t50 = w["turnover"].rolling(50, min_periods=30).median().shift(1)
        long = pd.concat({"open": w["open"].stack(), "close": w["close"].stack(), "t50": t50.stack()}, axis=1)
        long.index.names = ["date", "symbol"]
        long = long.astype("float32").reset_index()
        long.to_parquet(PANEL, index=False)
    return {c: long.pivot(index="date", columns="symbol", values=c) for c in ("open", "close", "t50")}


def run() -> Dict[str, Any]:
    from code.research import model_search as ms
    rp = pd.read_parquet(ms.REPLAY)
    p = load_panel()
    # a missing close (no trade that day, or delisted) carries the last price: the stock is sold at the next rebalance
    O, C, t50 = p["open"].astype("float64"), p["close"].astype("float64").ffill(), p["t50"]
    nv = pd.read_parquet(NIFTY)
    nifty, vix = nv["nifty"], nv["vix"]
    configs, sleeves = [], {}
    unis = ("all liquid", f"top {TOP_LIQUID} by turnover")

    def add(model: str, uni: str, sim: Dict[str, Any]) -> None:
        base = sim["ret"]
        for hedge, r in (("none", base), ("Nifty futures", hedge_futures(base, nifty)),
                         ("Nifty puts", hedge_puts(base, nifty, vix))):
            logger.info("%s | %s | %s", model, uni, hedge)
            configs.append({
                "model": model, "universe": uni, "hedge": hedge,
                "periods": {k: metrics(r[(r.index >= a) & (r.index < b)], sim["turnover"], sim["trades"])
                            for k, (a, b) in PERIODS.items()},
                "all": metrics(r, sim["turnover"], sim["trades"]), "yearly": yearly(r), "curve": curve(r)})

    for model, col, rebal, exit_rank in BOOKS:
        if col not in rp.columns:
            continue
        ranks = rp.pivot(index="date", columns="symbol", values=col).dropna(how="all")
        top = t50.reindex(index=ranks.index, columns=ranks.columns).rank(axis=1, ascending=False) <= TOP_LIQUID
        for uni, rk in zip(unis, (ranks, ranks.where(top).rank(axis=1, pct=True))):
            sim = simulate(rk, O, C, rebal=rebal, exit_rank=exit_rank)
            sleeves[(model, uni)] = sim
            add(model, uni, sim)
    for uni in unis:
        parts = [sleeves.get((m, uni)) for m in COMBINED]
        if all(parts):
            add("combined", uni, combine(parts))
    nr = nifty.pct_change().dropna()
    nr = nr[nr.index >= pd.Timestamp("2015-01-01")]
    bench = {"periods": {k: metrics(nr[(nr.index >= a) & (nr.index < b)]) for k, (a, b) in PERIODS.items()},
             "all": metrics(nr), "yearly": yearly(nr), "curve": curve(nr)}
    data = {"generated": date.today().isoformat(),
            "rules": {"n": N, "exit_rank": EXIT_RANK, "rebalance_sessions": REBAL, "buy_cost": BUY, "sell_cost": SELL,
                      "rf": RF, "put_strike": PUT_STRIKE, "put_days": PUT_DAYS, "top_liquid": TOP_LIQUID,
                      "books": {m: [rb, ex] for m, _, rb, ex in BOOKS}},
            "configs": configs, "nifty": bench}
    OUT.write_text(json.dumps(data), encoding="utf-8")
    return data


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if "--stress" in sys.argv:            # python -m code.research.portfolio --stress  (adds to portfolio.json)
        d = json.loads(OUT.read_text(encoding="utf-8"))
        d["stress"] = stress()
        OUT.write_text(json.dumps(d), encoding="utf-8")
        for uni, u in d["stress"]["universes"].items():
            print(uni, "| breakeven slip vs Nifty 2024+:", u["breakeven_vs_nifty_2024"], "| to zero:", u["breakeven_zero_2024"])
            for s in u["sized"]:
                m = s["periods"]["2024+"]
                print(f"  ₹{s['capital'] / 1e5:.0f} lakh: median slip {s['median_slip']:.2%}  2024+ {m['cagr']:+.1%}/yr "
                      f"Sharpe {m['sharpe']:.2f} DD {m['max_dd']:.0%}")
        sys.exit()
    d = run()
    n = d["nifty"]["periods"]
    for c in d["configs"]:
        cells = []
        for k in PERIODS:
            m = c["periods"][k]
            if m:
                cells.append(f"{k}: {m['cagr']:+.1%}/yr S{m['sharpe']:.2f} DD{m['max_dd']:.0%}")
        print(f"{c['model']:7} | {c['universe']:22} | {c['hedge']:13} | " + " | ".join(cells))
    print("Nifty   | " + " | ".join(f"{k}: {m['cagr']:+.1%}/yr S{m['sharpe']:.2f} DD{m['max_dd']:.0%}" for k, m in n.items() if m))
