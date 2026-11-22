import numpy as np
import pandas as pd
import pytest

from code.trading import portal
from code.trading.predict import predict_pair, predict_stock


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(portal, "DB_PATH", tmp_path / "papa.db")
    portal._init_tables()


def _ohlcv(closes, start="2026-01-01"):
    idx = pd.bdate_range(start, periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": 1000}, index=idx)


def test_predict_stock_levels_follow_risk_rule():
    rng = np.random.default_rng(0)
    df = _ohlcv(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 400))))
    p = predict_stock("TEST.NS", df)
    assert p["stop"] == pytest.approx(p["entry"] - 1.5 * p["atr"], abs=0.02)
    assert p["target"] - p["entry"] == pytest.approx(2 * (p["entry"] - p["stop"]), abs=0.02)
    assert p["next_low"] < p["last"] < p["next_high"]                     # centred: no direction claimed
    assert p["week_high"] - p["week_low"] > p["next_high"] - p["next_low"]
    assert "odds_target" not in p and "next_mid" not in p
    assert p == predict_stock("TEST.NS", df), "same stock, same day must give the same numbers"


def test_calibrated_band_covers_about_68_percent():
    from code.trading.ranges import latest_band
    rng = np.random.default_rng(1)
    r = rng.standard_t(4, 1600) * 0.012
    vol = np.where(np.arange(1600) % 400 < 120, 2.5, 1.0)               # volatile spells
    close = pd.Series(100 * np.exp(np.cumsum(r * vol)))
    hits = []
    for t in range(700, 1599, 3):
        b = latest_band(close.iloc[:t + 1], 1)
        hits.append(b["lo"] <= close.iat[t + 1] <= b["hi"])
    assert 0.62 <= np.mean(hits) <= 0.74


def test_trailing_exit_sells_on_close_below_21_day_average():
    portal.place_trade("RUN.NS", "BUY", 100.0, "minervini", qty=10, target=110.0, stop=90.0, exit_rule="trail21")
    future = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    hist = _ohlcv([100.0] * 30, start=future - pd.Timedelta(days=60))
    fwd = _ohlcv([112, 120, 125, 99], start=future)                       # through 2R, then a close below the average
    df = pd.concat([hist[hist.index < future], fwd])
    portal.settle_open_trades(lambda t: df)
    t = portal.get_trades(status="closed")[0]
    assert t["exit_reason"] == "Closed below 21-day average" and t["exit_price"] == 99.0


def test_fixed_target_rule_unchanged():
    portal.place_trade("FIX.NS", "BUY", 100.0, "minervini", qty=10, target=110.0, stop=90.0, exit_rule="target")
    future = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    portal.settle_open_trades(lambda t: _ohlcv([105, 112], start=future))
    assert portal.get_trades(status="closed")[0]["exit_reason"] == "Target hit"


def test_market_grade_and_grade_c_is_defensive():
    from code.regime.market_health import breadth_200, market_grade, risk_posture
    assert market_grade(0.65, True) == "A" and market_grade(0.65, False) == "B" and market_grade(0.40, False) == "C"
    assert breadth_200([{"above_200": True}, {"above_200": False}, {"above_200": None}]) == 0.5
    up = {"distribution_days": 1, "rally": {"state": "CONFIRMED"}}
    assert risk_posture("TRENDING_UP", up, "C")["level"] == "DEFENSIVE"
    assert risk_posture("TRENDING_UP", up, "A")["level"] == "PRESS"
    assert risk_posture("TRENDING_UP", up, "B")["level"] == "CAUTIOUS"


def test_prediction_log_accepts_plans_without_a_midpoint():
    portal.log_predictions([{"kind": "stock", "asof": "2026-01-01", "symbol": "Z.NS", "entry": 100, "last": 100,
                             "next_low": 98, "next_high": 102}])
    portal.score_predictions(lambda t: _ohlcv([101], start="2026-01-02"))
    card = portal.get_scorecard()
    assert card["scored"] == 1 and card["inside_pct"] == 100 and card["direction_pct"] is None


def test_predict_pair_targets_mean_side():
    pts = [[1_700_000_000_000 + i * 86_400_000, 100.0 - i * 0.1] for i in range(60)]
    item = {
        "pair": "A.NS/B.NS", "ticker_a": "A.NS", "ticker_b": "B.NS", "hedge_ratio": 2.0, "qty_a": 100, "qty_b": 200,
        "zscore": -2.5, "status": "BUY", "entry_z": 2.0, "exit_z": 0.5, "half_life_days": 8,
        "spread_series": pts, "upper_band_series": [[0, 110.0]], "lower_band_series": [[0, 90.0]],
    }
    p = predict_pair(item)
    assert p["stop"] < p["entry"] < p["target"] < p["mean"]


def test_settle_hits_target_and_marks_others():
    portal.place_trade("WIN.NS", "BUY", 100.0, "minervini", qty=10, target=110.0, stop=95.0)
    portal.place_trade("FLAT.NS", "BUY", 100.0, "minervini", qty=10, target=110.0, stop=95.0)
    future = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    data = {"WIN.NS": _ohlcv([105, 112], start=future), "FLAT.NS": _ohlcv([101, 102], start=future)}
    result = portal.settle_open_trades(lambda t: data[t])
    assert result["closed"] == 1 and result["marked"] == 1

    closed = portal.get_trades(status="closed")[0]
    # day 2 gaps open above the target, so the fill is the open (112), not the target
    assert closed["exit_reason"] == "Target hit" and closed["pnl"] == pytest.approx(120.0)
    open_trade = portal.get_trades(status="open")[0]
    assert open_trade["last_price"] == 102 and open_trade["mtm_pnl"] == pytest.approx(20.0)


def test_settle_same_bar_assumes_stop_first():
    portal.place_trade("WILD.NS", "BUY", 100.0, "minervini", qty=1, target=101.0, stop=99.5)
    future = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    portal.settle_open_trades(lambda t: _ohlcv([100], start=future))
    assert portal.get_trades(status="closed")[0]["exit_reason"] == "Stop hit"


def test_close_at_market_uses_last_price():
    tid = portal.place_trade("X.NS", "BUY", 50.0, "minervini", qty=4)["id"]
    future = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    portal.settle_open_trades(lambda t: _ohlcv([55], start=future))
    assert portal.close_trade(tid)["pnl"] == pytest.approx(20.0)


def test_prediction_scorecard():
    portal.log_predictions([{"kind": "stock", "asof": "2026-01-01", "symbol": "S.NS", "entry": 100,
                             "next_low": 98, "next_mid": 101, "next_high": 103}])
    portal.score_predictions(lambda t: _ohlcv([102], start="2026-01-02"))
    card = portal.get_scorecard()
    assert card["scored"] == 1 and card["inside_pct"] == 100 and card["direction_pct"] == 100


def test_delete_only_closed_trades():
    open_id = portal.place_trade("A.NS", "BUY", 10.0, "minervini")["id"]
    closed_id = portal.place_trade("B.NS", "BUY", 10.0, "minervini")["id"]
    portal.close_trade(closed_id, 11.0)
    assert portal.delete_trade(open_id)["status"] == "error"
    assert portal.delete_trade(closed_id)["status"] == "deleted"
    assert [t["id"] for t in portal.get_trades()] == [open_id]


def test_candidates_cover_passes_entry_pairs_and_watchlist():
    from code.trading.predict import prediction_candidates
    screen = {"items": [
        {"symbol": "P1.NS", "passed": True, "rs_rank": 90, "conditions_passed": 8},
        {"symbol": "P2.NS", "passed": True, "rs_rank": 95, "conditions_passed": 8},
        {"symbol": "W.NS", "passed": False, "rs_rank": 40, "conditions_passed": 5},
    ]}
    pairs = {"items": [{"pair": "X.NS/Y.NS", "status": "BUY"}, {"pair": "Q.NS/R.NS", "status": "NEUTRAL"}]}
    out = prediction_candidates(screen, pairs, ["W.NS", "P1.NS"])
    assert [c["symbol"] for c in out] == ["P2.NS", "P1.NS", "X.NS/Y.NS", "W.NS"]
    assert next(c for c in out if c["symbol"] == "P1.NS")["on_watchlist"] is True


def test_candidates_always_include_todays_setups(monkeypatch):
    from code.trading import predict
    monkeypatch.setattr(predict, "MAX_SCREEN_PREDICTIONS", 1)
    screen = {"items": [{"symbol": "HI.NS", "passed": True, "rs_rank": 99}, {"symbol": "LO.NS", "passed": True, "rs_rank": 70}]}
    assert [c["symbol"] for c in predict.prediction_candidates(screen, {}, [])] == ["HI.NS"]
    assert [c["symbol"] for c in predict.prediction_candidates(screen, {}, [], ["LO.NS"])] == ["HI.NS", "LO.NS"]


def test_breakout_order_fills_through_trigger_then_hits_target():
    portal.place_trade("BO.NS", "BUY", 100.0, "minervini", qty=10, target=110.0, stop=95.0, pending=True)
    future = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    # day 1 stays under the trigger, day 2 gaps through it, day 3 hits target
    df = _ohlcv([98, 101, 111], start=future)
    df.loc[df.index[1], "open"] = 101.0
    result = portal.settle_open_trades(lambda t: df)
    assert result["filled"] == 1 and result["closed"] == 1
    t = portal.get_trades(status="closed")[0]
    assert t["entry_price"] == 101.0 and t["exit_reason"] == "Target hit"


def test_breakout_order_cancels_if_stop_breaks_first():
    portal.place_trade("FAIL.NS", "BUY", 100.0, "minervini", qty=10, target=110.0, stop=95.0, pending=True)
    future = pd.Timestamp.today().normalize() + pd.Timedelta(days=1)
    portal.settle_open_trades(lambda t: _ohlcv([94], start=future))
    t = portal.get_trades(status="expired")[0]
    assert t["exit_reason"].startswith("Cancelled")


def test_position_cap_refuses_extra_trades():
    portal.place_trade("A.NS", "BUY", 10.0, "minervini", max_positions=2)
    portal.place_trade("B.NS", "BUY", 10.0, "minervini", pending=True, max_positions=2)
    assert portal.place_trade("C.NS", "BUY", 10.0, "minervini", max_positions=2)["status"] == "error"


def test_setup_detects_base_breakout_and_extension():
    from code.signals.minervini.setup import detect_setup
    up = list(np.linspace(50, 100, 60))
    base = [100 - 8 * abs(np.sin(i / 3)) * (1 - i / 30) for i in range(30)]  # tightening pullbacks under 100
    basing = _ohlcv(up + base + [97])
    assert detect_setup(basing)["status"] == "BASING"
    assert detect_setup(_ohlcv(up + base + [103]))["status"] == "IN BUY RANGE"
    assert detect_setup(_ohlcv(up + base + [112]))["status"] == "EXTENDED"


def test_market_health_counts_distribution_and_follow_through():
    from code.regime.market_health import index_health, risk_posture
    closes = [100.0] * 30 + [99.0, 98.0, 97.0, 96.0, 96.5, 96.8, 97.0, 99.0]
    df = _ohlcv(closes)
    df["volume"] = [1000 + 10 * i for i in range(len(closes))]  # rising volume every day
    h = index_health(df)
    assert h["distribution_days"] == 4            # the four down days of >= 0.2%
    assert h["rally"]["state"] == "CONFIRMED"     # +2.1% on day 4 of the rally, volume up
    assert risk_posture("TRENDING_DOWN", {**h, "rally": {"state": "ATTEMPT"}})["max_positions"] == 2


def test_futures_legs_round_to_whole_lots():
    from code.pipeline.nightly_pipeline import _futures_legs
    legs = _futures_legs("A.NS", "B.NS", 1.5, 1000.0, 400.0, {"A.NS": 100, "B.NS": 140})
    assert legs["lots_b"] == 1 and legs["qty_b"] == 140 and legs["notional_a"] == 100000
    assert _futures_legs("A.NS", "C.NS", 1.0, 1.0, 1.0, {"A.NS": 100})["fno"] is False


def test_headlines_must_name_the_company():
    from code.evaluator.memo import match_headlines
    news = [{"title": "Buy HDFC Bank; target 1850", "ts": ""}, {"title": "Cupid Ltd hits record high", "ts": ""}]
    assert [h["title"] for h in match_headlines("CUPID.NS", news)] == ["Cupid Ltd hits record high"]


def test_plan_uses_the_screens_setup():
    from code.trading.predict import predict_stock
    rng = np.random.default_rng(0)
    df = _ohlcv(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 400))))
    setup = {"status": "BASING", "pivot": 123.45, "pct_to_pivot": 2.0, "vcp": False}
    plan = predict_stock("X.NS", df, setup=setup)
    assert plan["setup"] is setup and plan["order"] == "STOP" and plan["entry"] == round(123.45 * 1.001, 2)


def test_position_view_has_chart_ranges_and_plan():
    from code.trading.predict import position_view
    rng = np.random.default_rng(1)
    df = _ohlcv(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 400))))
    last = float(df["close"].iloc[-1])
    trade = {"ticker": "X.NS", "entry_price": last * 0.9, "stop": last * 0.85, "target": last * 1.0, "qty": 10, "exit_rule": "target"}
    v = position_view(trade, df)
    assert v["candles"] and v["path_high"] and v["week_low"] < v["week_high"]
    titles = [s["title"] for s in v["steps"]]
    assert titles[:3] == ["Stop", "Protect", "Take profit"] and titles[3].startswith("Tonight:")
    assert v["r_now"] == round((last - trade["entry_price"]) / (trade["entry_price"] - trade["stop"]), 2)
    assert v["steps"][1]["state"] == "done"            # price is past +1R, so the break-even step is done
    assert any("next week's likely range" in o for o in v["outlook"])
    assert position_view({**trade, "stop": None}, df)["steps"][0]["title"] == "Set a stop"


def _q(ltp, high, low, open_, stale=False, simulated=False):
    return {"ltp": ltp, "day_high": high, "day_low": low, "day_open": open_, "stale": stale, "simulated": simulated}


def test_live_fill_rules():
    today = pd.Timestamp.today().date().isoformat()
    old = portal.place_trade("OLD.NS", "BUY", 100.0, "minervini", qty=10, target=110.0, stop=95.0, pending=True)["id"]
    new = portal.place_trade("NEW.NS", "BUY", 100.0, "minervini", qty=10, target=110.0, stop=95.0, pending=True)["id"]
    stopped = portal.place_trade("STP.NS", "BUY", 100.0, "minervini", qty=10, target=110.0, stop=95.0, pending=True)["id"]
    conn = portal._conn()
    conn.execute("UPDATE paper_trades SET created_at='2026-01-01T10:00:00' WHERE id IN (?, ?)", (old, stopped))
    conn.commit(); conn.close()
    quotes = {"OLD.NS": _q(103, 104, 101, 102), "NEW.NS": _q(99, 101, 98, 99), "STP.NS": _q(101, 102, 94, 96)}
    assert portal.fill_pending_live({k: {**v, "stale": True} for k, v in quotes.items()}, today) == []
    assert portal.fill_pending_live({k: {**v, "simulated": True} for k, v in quotes.items()}, today) == []
    assert portal.fill_pending_live(quotes, today) == [old]          # NEW is back under its trigger; STP touched the stop
    t = next(x for x in portal.get_trades(status="open") if x["id"] == old)
    assert t["entry_price"] == 102.0 and t["entry_date"] == today       # gapped through: fills at the open
    assert portal.fill_pending_live({"NEW.NS": _q(100.5, 101, 98, 99)}, today) == [new]
    assert next(x for x in portal.get_trades(status="open") if x["id"] == new)["entry_price"] == 100.0


def test_touch_probability_and_pending_view():
    from code.trading.predict import position_view, touch_probability
    assert touch_probability(100, 100, 0.02) == 1.0
    assert 0.3 < touch_probability(100, 102, 0.02) < 0.33        # 1σ away: 2·(1 − Φ(1)) ≈ 0.317
    assert touch_probability(100, 110, 0.02) < 0.001
    rng = np.random.default_rng(3)
    df = _ohlcv(100 * np.exp(np.cumsum(rng.normal(0, 0.015, 400))))
    last = float(df["close"].iloc[-1])
    v = position_view({"ticker": "X.NS", "status": "pending", "entry_price": last * 1.02, "stop": last * 0.95, "target": last * 1.15,
                       "qty": 10, "created_at": str(df.index[-3].date())}, df)
    assert v["pending"] and v["steps"][0]["title"] == "Waiting" and 0 < v["trigger_prob"] < 1
    assert v["tomorrow"]["action"] == "WAIT FOR BREAKOUT" and "8 sessions left" in v["tomorrow"]["text"]
    o = position_view({"ticker": "X.NS", "status": "open", "entry_price": last, "stop": last * 0.95, "target": last * 1.1, "qty": 10}, df)
    assert o["tomorrow"]["action"] and "chance" in o["tomorrow"]["text"] or o["tomorrow"]["action"] in ("SELL", "SELL HALF", "RAISE STOP")


def test_portfolio_outlook_replays_history():
    from code.trading.predict import portfolio_outlook
    rng = np.random.default_rng(4)
    a = _ohlcv(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 300))))
    b = _ohlcv(50 * np.exp(np.cumsum(rng.normal(0, 0.02, 300))))
    ta = {"id": 1, "qty": 100, "entry_price": float(a["close"].iloc[-1]) * 0.9, "stop": float(a["close"].iloc[-1]) * 0.95}
    tb = {"id": 2, "qty": 50, "entry_price": float(b["close"].iloc[-1]), "stop": None}
    o = portfolio_outlook([(ta, a), (tb, b)], 1_000_000)
    assert o["n"] == 2 and o["pnl"] > 0 and o["heat"] > 0 and o["days_used"] == 250
    assert o["day_bad"] < o["day_lo"] < 0 < o["day_hi"] and o["week_lo"] < o["day_lo"] and o["week_hi"] > o["day_hi"]
    assert portfolio_outlook([], 1_000_000) == {"n": 0, "invested": 0.0, "pnl": 0.0, "heat": 0.0}
