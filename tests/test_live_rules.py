from datetime import datetime

from code.signals.live_rules import day_phase, position_signal, setup_signal


def test_live_rules():
    ist = datetime(2026, 11, 20, 11, 0)
    plan = {"entry": 101.0, "stop": 95.0, "target": 113.0, "qty": 50, "setup": {"pivot": 100.0}, "metrics": {"avg_vol50": 100_000}}
    q = lambda ltp, high=None, vol=200_000: {"ltp": ltp, "day_high": high or ltp, "volume": vol}
    assert setup_signal(plan, q(102), ist)["state"] == "BUY"
    assert setup_signal(plan, {**q(102, vol=20_000), "prev_close": 101.5}, ist)["state"] == "BUY RANGE"  # broke out yesterday            # 105 min in, 200k vs 100k avg -> pace 7×
    assert setup_signal(plan, q(102, vol=20_000), ist)["state"] == "WEAK"
    assert setup_signal({**plan, "metrics": {}}, q(102), ist)["state"] == "CHECK"
    assert setup_signal(plan, q(106), ist)["state"] == "EXTENDED"
    assert setup_signal(plan, q(99, high=101), ist)["state"] == "FAILED"
    assert setup_signal(plan, q(98), ist)["state"] == "WAIT"
    assert setup_signal(plan, q(94), ist)["state"] == "AVOID"
    assert setup_signal(plan, q(102), datetime(2026, 11, 20, 9, 20))["state"] == "WAIT"
    t = {"entry_price": 100.0, "stop": 95.0, "target": 110.0, "side": "BUY"}
    assert [position_signal(t, p)["state"] for p in (94, 98, 102, 106, 111)] == ["EXIT", "HOLD", "HOLD", "PROTECT", "TAKE PROFIT"]
    assert day_phase(datetime(2026, 11, 20, 9, 20)) == "settle" and day_phase(ist, trading_day=False) == "after"
