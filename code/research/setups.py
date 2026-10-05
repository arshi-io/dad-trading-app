"""Historical pivot-breakout events, labelled by what happened next — the training set for the
setup-outcome model.

Event: a liquid NSE stock trades through the high of a valid base (≥10 sessions, ≤35% deep, within
the last 65 sessions). Decision at that day's close, using only what is known then (including the
breakout day's volume and close). Entry: next session's open. Stop: 1.5 × ATR(14). Target: 2R.
Exits walk forward 20 sessions, stop first when a bar touches both, gaps fill at the open,
round-trip cost 0.25%.
"""
from __future__ import annotations

import logging
from typing import Dict

import numpy as np
import pandas as pd

from code.research import bhavcopy

logger = logging.getLogger(__name__)

BASE_LOOKBACK, MIN_BASE, MAX_DEPTH = 65, 10, 0.35
HOLD, TARGET_R, ATR_MULT, COST = 20, 2.0, 1.5, 0.0025
MIN_PRICE, MIN_TURNOVER = 30.0, 5e7          # ₹5 crore median daily turnover
MIN_ATR_PCT = 0.01                            # a 1.5×ATR stop under ~1.5% means it isn't a swing stock
ETF_MARKERS = ("BEES", "ETF", "LIQUID", "GILT", "SDL", "NIFTY", "SENSEX", "GOLD", "SILVER")


def is_equity(symbol: str) -> bool:
    """NSE's cash file includes ETFs and debt funds; their near-zero ranges make R meaningless."""
    s = symbol.upper()
    return not any(m in s for m in ETF_MARKERS)


def build_panel() -> Dict[str, pd.DataFrame]:
    """Wide date × symbol frames of split/bonus-adjusted OHLC, volume, turnover and delivery."""
    df = bhavcopy.load()
    df = df[df["series"].isin(["EQ", "BE", "BZ"])]
    df = df.sort_values(["date", "symbol", "series"]).drop_duplicates(["date", "symbol"], keep="first")
    for c in ("open", "high", "low", "close", "prev_close", "volume", "turnover", "delivery_qty"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    wide = {c: df.pivot(index="date", columns="symbol", values=c) for c in
            ("open", "high", "low", "close", "prev_close", "volume", "turnover", "delivery_qty")}
    # Adjustment: chain NSE's own ex-date-adjusted prev_close returns back from the last close.
    ret = (wide["close"] / wide["prev_close"]).where(lambda x: (x > 0.4) & (x < 2.5))
    growth = ret.fillna(1.0).cumprod()
    factor = (growth / wide["close"]).div((growth / wide["close"]).ffill().iloc[-1], axis=1)
    factor = factor.ffill()
    out = {k: wide[k] * factor for k in ("open", "high", "low", "close")}
    out.update({k: wide[k] for k in ("volume", "turnover", "delivery_qty")})
    return out


def _nifty_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    import yfinance as yf
    n = yf.download("^NSEI", start="2010-01-01", progress=False, auto_adjust=True)
    if isinstance(n.columns, pd.MultiIndex):
        n.columns = n.columns.get_level_values(0)
    c, v = n["Close"], n["Volume"]
    ret = c.pct_change()
    dist = ((ret <= -0.002) & (v > v.shift(1)) & (v > 0)).astype(int).rolling(25).sum()
    f = pd.DataFrame({
        "mkt_above_200": (c > c.rolling(200).mean()).astype(float),
        "mkt_above_50": (c > c.rolling(50).mean()).astype(float),
        "mkt_from_high": c / c.rolling(250).max() - 1,
        "mkt_dist_days": dist.where(v.rolling(25).min() > 0),       # no volume data → unknown, not 0
        "mkt_ret_20": c.pct_change(20),
    })
    return f.reindex(index).ffill()


def build_events() -> pd.DataFrame:
    p = build_panel()
    C, H, L, O, V = p["close"], p["high"], p["low"], p["open"], p["volume"]
    dates = C.index
    turnover50 = p["turnover"].rolling(50, min_periods=30).median()
    liquid = (turnover50 >= MIN_TURNOVER) & (C >= MIN_PRICE)

    sma50, sma150, sma200 = (C.rolling(n, min_periods=n).mean() for n in (50, 150, 200))
    hi250, lo250 = H.rolling(250, min_periods=200).max(), L.rolling(250, min_periods=200).min()
    tr = pd.concat([H - L, (H - C.shift()).abs(), (L - C.shift()).abs()]).groupby(level=0).max()
    atr = tr.reindex(dates).rolling(14, min_periods=14).mean()
    vol50 = V.rolling(50, min_periods=30).mean().shift(1)
    deliv_pct = (p["delivery_qty"] / V)
    deliv50 = deliv_pct.rolling(50, min_periods=30).mean().shift(1)
    # RS rank: IBD-style weighted 3/6/9/12-month performance, percentile across liquid stocks that day.
    perf = 0.4 * C.pct_change(63) + 0.2 * C.pct_change(126) + 0.2 * C.pct_change(189) + 0.2 * C.pct_change(252)
    rs = perf.where(liquid).rank(axis=1, pct=True) * 99
    breadth = (C > sma200).where(liquid).mean(axis=1)
    template = sum([
        (C > sma150) & (C > sma200), sma150 > sma200, sma200 > sma200.shift(21),
        (sma50 > sma150) & (sma50 > sma200), C > sma50, C >= lo250 * 1.3, C >= hi250 * 0.75, rs >= 70,
    ]).astype(float)
    pivot_now = H.shift(1).rolling(BASE_LOOKBACK, min_periods=BASE_LOOKBACK).max()
    # Trigger = the buy-stop being touched intraday, NOT a close above the pivot: requiring the close
    # would silently drop same-day reversals, which a real buy-stop fills and which mostly fail.
    trig = pivot_now * 1.001
    cand = (H >= trig) & (H.shift(1) < trig.shift(1)) & liquid.shift(1, fill_value=False)
    mkt = _nifty_features(dates)

    rows = []
    for sym in C.columns[cand.any(axis=0).to_numpy()]:
        if not is_equity(sym):
            continue
        c, h, l, o, v = (x[sym].to_numpy() for x in (C, H, L, O, V))
        a = atr[sym].to_numpy()
        for t in np.flatnonzero(cand[sym].to_numpy()):
            if t < 260 or t + 1 >= len(c):
                continue
            win_h = h[t - BASE_LOOKBACK:t]
            if np.isnan(win_h).any():
                continue
            peak = int(np.argmax(win_h))
            pivot = win_h[peak]
            base = slice(t - BASE_LOOKBACK + peak, t)
            base_len = t - (t - BASE_LOOKBACK + peak)
            depth = (pivot - np.nanmin(l[base])) / pivot
            if base_len < MIN_BASE or depth > MAX_DEPTH:
                continue
            cuts = np.linspace(t - BASE_LOOKBACK + peak, t, 4).astype(int)
            contr = [(np.nanmax(h[cuts[k]:cuts[k + 1]]) - np.nanmin(l[cuts[k]:cuts[k + 1]])) / np.nanmax(h[cuts[k]:cuts[k + 1]])
                     for k in range(3)]
            vol_dry = np.nanmean(v[cuts[2]:cuts[3]]) / vol50[sym].iat[t] if vol50[sym].iat[t] else np.nan
            # End-of-day decision: everything about the breakout day is known at its close; the
            # trade is entered at the next session's open. (Deciding intraday at the pivot would need
            # intraday volume history, which NSE's daily files don't have.)
            if t + 1 >= len(c) or np.isnan(o[t + 1]) or np.isnan(a[t]) or a[t] / c[t] < MIN_ATR_PCT:
                continue
            entry = o[t + 1]
            stop = entry - ATR_MULT * a[t]
            risk = entry - stop
            if risk <= 0:
                continue
            target = entry + TARGET_R * risk
            r_mult, outcome, days = None, "time", HOLD
            for k in range(t + 1, min(t + 1 + HOLD, len(c))):          # entry day included: fills at its open
                if np.isnan(l[k]):
                    continue
                if l[k] <= stop:
                    r_mult, outcome, days = (min(o[k], stop) - entry) / risk, "stop", k - t
                    break
                if h[k] >= target:
                    r_mult, outcome, days = (max(o[k], target) - entry) / risk, "target", k - t
                    break
            if r_mult is None:
                last = min(t + HOLD, len(c) - 1)
                if last - t < HOLD:
                    continue                                  # not enough future yet: unlabelled
                r_mult = (c[last] - entry) / risk
            r_mult -= COST * entry / risk
            if not np.isfinite(r_mult):
                continue
            d = dates[t]
            rows.append({
                "date": d, "symbol": sym, "entry": entry, "risk_pct": risk / entry,
                "r": r_mult, "outcome": outcome, "days": days, "win": outcome == "target",
                "template": template[sym].iat[t], "rs": rs[sym].iat[t],
                "base_len": base_len, "base_depth": depth,
                "c1": contr[0], "c2": contr[1], "c3": contr[2],
                "vcp": float(contr[0] > contr[1] > contr[2] and vol_dry < 1),
                "vol_dry": vol_dry, "bo_vol": v[t] / vol50[sym].iat[t] if vol50[sym].iat[t] else np.nan,
                "close_vs_pivot": c[t] / pivot - 1,
                "close_loc": (c[t] - l[t]) / (h[t] - l[t]) if h[t] > l[t] else np.nan,
                "day_ret": c[t] / c[t - 1] - 1, "ext50": c[t] / sma50[sym].iat[t] - 1,
                "from_high": c[t] / hi250[sym].iat[t] - 1, "atr_pct": a[t] / c[t],
                "deliv_rel": deliv_pct[sym].iat[t] / deliv50[sym].iat[t] if deliv50[sym].iat[t] else np.nan,
                "breadth": breadth.iat[t],
                **{k: mkt[k].iat[t] for k in mkt.columns},
            })
    ev = pd.DataFrame(rows)
    logger.info("built %d labelled breakout events", len(ev))
    return ev


if __name__ == "__main__":
    import time
    logging.basicConfig(level=logging.INFO)
    t0 = time.time()
    ev = build_events()
    out = bhavcopy.ROOT / "breakout_events.parquet"
    ev.to_parquet(out, index=False)
    print(f"{len(ev):,} events in {time.time() - t0:.0f}s -> {out}")
    print(ev.groupby(ev.date.dt.year).agg(n=("r", "size"), win=("win", "mean"), avg_r=("r", "mean")).round(3).to_string())
