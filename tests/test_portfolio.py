import numpy as np
import pandas as pd
import pytest

from code.research import portfolio as pf

DAYS = pd.bdate_range("2020-01-01", periods=60)
SYMS = [f"S{i}" for i in range(4)]


def _flat(price=100.0):
    f = pd.DataFrame(price, index=DAYS, columns=SYMS)
    return f.copy(), f.copy()


def _ranks(order):
    return pd.DataFrame([order] * len(DAYS), index=DAYS, columns=SYMS, dtype=float)


def test_flat_prices_lose_exactly_the_buy_cost_then_nothing():
    O, C = _flat()
    sim = pf.simulate(_ranks([0.9, 0.8, 0.5, 0.2]), O, C, n=2, exit_rank=0.7)
    r = sim["ret"]
    assert r.iloc[0] == pytest.approx(-pf.BUY)            # bought at the first open
    assert (r.iloc[1:] == 0).all()                          # ranks never change: no more trading


def test_overnight_gap_before_the_buy_is_not_earned():
    O, C = _flat()
    ranks = _ranks([0.1, 0.2, 0.3, 0.9])
    O.loc[DAYS[1], "S3"] = C.loc[DAYS[1], "S3"] = 150.0     # S3 gaps up at the open we buy at
    sim = pf.simulate(ranks, O, C, n=1)
    assert sim["ret"].iloc[0] == pytest.approx(-pf.BUY)    # bought at 150, closed at 150


def test_buffer_keeps_a_stock_that_slips_a_little_and_sells_one_that_slips_a_lot():
    O, C = _flat()
    ranks = _ranks([0.9, 0.85, 0.5, 0.2])
    ranks.iloc[5:, :] = [0.82, 0.95, 0.5, 0.2]             # S0 slips to 0.82: kept (exit at 0.80)
    ranks.iloc[10:, :] = [0.60, 0.95, 0.9, 0.2]            # S0 slips to 0.60: sold for S2
    sim = pf.simulate(ranks, O, C, n=2, exit_rank=0.8)
    t = sim["turnover"]
    assert t.iloc[5] == 0 and t.iloc[10] == pytest.approx(1.0)   # sell 0.5 + buy 0.5
    assert list(sim["trades"]["symbol"]) == ["S0"]
    assert sim["trades"]["ret"].iloc[0] == pytest.approx((1 - pf.SELL) / (1 + pf.BUY) - 1)


def test_per_stock_slippage_is_charged_on_each_stock_bought():
    O, C = _flat()
    slip = pd.DataFrame(0.0, index=DAYS, columns=SYMS)
    slip["S0"], slip["S1"] = 0.01, 0.03
    sim = pf.simulate(_ranks([0.9, 0.8, 0.5, 0.2]), O, C, n=2, exit_rank=0.7, slip=slip)
    assert sim["ret"].iloc[0] == pytest.approx(-(pf.FEES_BUY + 0.5 * 0.01 + 0.5 * 0.03))


def test_size_slippage_grows_with_order_size_and_shrinks_with_liquidity():
    t50 = pd.DataFrame({"big": [2e9], "small": [6e7]})
    vol = pd.DataFrame({"big": [0.02], "small": [0.02]})
    s1, s2 = pf.size_slippage(t50, vol, 1e6), pf.size_slippage(t50, vol, 2e7)
    assert s1.loc[0, "big"] < s1.loc[0, "small"] < s2.loc[0, "small"]
    assert s1.loc[0, "big"] == pytest.approx(0.0003 + 0.5 * 0.02 * np.sqrt(5e4 / 2e9))


def test_breakeven_interpolates_between_tested_slippages():
    pts = [{"slip": 0.001, "periods": {"2024+": {"cagr": 0.20}}}, {"slip": 0.003, "periods": {"2024+": {"cagr": 0.10}}}]
    assert pf.breakeven(pts, "2024+", 0.15) == pytest.approx(0.002)


def test_combined_book_splits_capital_equally_over_shared_days():
    idx = pd.bdate_range("2021-01-01", periods=4)
    a = {"ret": pd.Series([0.01, 0.02, 0.03, 0.04], index=idx), "turnover": pd.Series([1.0, 0, 0, 0], index=idx),
         "trades": pd.DataFrame(columns=["symbol", "exit", "ret"])}
    b = {"ret": pd.Series([0.03, -0.02, 0.01], index=idx[1:]), "turnover": pd.Series([2.0, 0, 0], index=idx[1:]),
         "trades": pd.DataFrame(columns=["symbol", "exit", "ret"])}
    c = pf.combine([a, b])
    assert list(c["ret"].index) == list(idx[1:])
    assert c["ret"].tolist() == pytest.approx([0.025, 0.005, 0.025]) and c["turnover"].iloc[0] == pytest.approx(1.0)


def test_futures_hedge_on_a_book_that_is_the_index_earns_the_risk_free_rate():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=300)
    nifty = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    port = nifty.pct_change().fillna(0.0)
    h = pf.hedge_futures(port, nifty).iloc[pf.BETA_DAYS + 5:]
    assert h.std() < 0.001 and h.mean() * 252 == pytest.approx(pf.RF, abs=0.02)


def test_puts_pay_in_a_crash_and_cost_premium_in_calm():
    idx = pd.bdate_range("2020-01-01", periods=120)
    calm = pd.Series(100.0, index=idx)
    crash = calm.copy()
    crash.iloc[70:] = 80.0                                    # -20% inside the third month's put
    vix = pd.Series(20.0, index=idx)
    port = pd.Series(0.0, index=idx)
    assert pf.hedge_puts(port, calm, vix).sum() < 0
    assert pf.hedge_puts(port, crash, vix).sum() > 0.1


def test_black_scholes_put_matches_a_known_value():
    assert pf.bs_put(100, 100, 1.0, 0.2, r=0.05) == pytest.approx(5.5735, abs=1e-3)


def test_metrics_on_a_known_series():
    r = pd.Series([0.001] * 252, index=pd.bdate_range("2020-01-01", periods=252))
    m = pf.metrics(r)
    assert m["cagr"] == pytest.approx(1.001 ** 252 - 1, abs=1e-4) and m["max_dd"] == 0 and m["months_up"] == 1
