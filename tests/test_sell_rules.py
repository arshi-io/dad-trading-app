from datetime import datetime

import numpy as np
import pandas as pd

from code.signals.live_rules import sell_level_signal, short_signal
from code.signals.sell_rules import breakdown_setup, sell_check, short_plan, stage4


def _df(closes, vol=1000.0, last_vol=None):
    idx = pd.bdate_range("2025-01-01", periods=len(closes))
    c = pd.Series(closes, index=idx, dtype=float)
    v = pd.Series(vol, index=idx)
    if last_vol is not None:
        v.iloc[-1] = last_vol
    return pd.DataFrame({"open": c, "high": c * 1.01, "low": c * 0.99, "close": c, "volume": v})


def test_sell_check_rules():
    up = list(np.linspace(100, 150, 120))
    assert sell_check(_df(up))["action"] == "HOLD"
    assert sell_check(_df(up), stop=151)["action"] == "SELL"                                  # closed under the stop
    assert sell_check(_df(up + [125], last_vol=3000))["action"] == "SELL"                    # 50-day break on 3x volume
    assert sell_check(_df(up + [125]))["action"] == "WEAK"                                   # same break, light volume
    assert sell_check(_df(up[:100] + list(np.linspace(140, 190, 16))))["action"] == "SELL HALF"  # +35% in 15 sessions
    assert sell_check(_df(up))["sell_below"] > 0


def test_stage4_and_breakdown():
    down = list(np.linspace(300, 100, 260))
    df = _df(down)
    assert stage4(df, rs_rank=10) and not stage4(df, rs_rank=80) and not stage4(_df(down[::-1]), rs_rank=10)
    flat = _df([100.0] * 40 + [101.0])
    s = breakdown_setup(flat)
    assert s["status"] == "NEAR BREAKDOWN" and s["support"] == 99.0
    assert breakdown_setup(_df([100.0] * 40 + [97.0]))["status"] == "IN SHORT RANGE"
    assert breakdown_setup(_df([100.0] * 40 + [90.0]))["status"] == "EXTENDED DOWN"
    plan = short_plan(flat, lot=500, risk_rupees=10_000)
    assert plan["order"] == "STOP" and plan["stop"] > plan["entry"] > plan["target"]
    assert plan["lots"] == int(10_000 // (plan["risk_per_lot"]))


def test_live_short_and_sell_level():
    now = datetime(2026, 11, 20, 11, 0)
    plan = {"support": 100.0, "entry": 99.9, "stop": 104.0, "target": 91.7, "lots": 2, "avg_vol50": 100_000}
    q = lambda ltp, low=None, vol=200_000: {"ltp": ltp, "day_low": low or ltp, "volume": vol}
    assert short_signal(plan, q(98.5), now)["state"] == "SHORT"
    assert short_signal(plan, q(98.5, vol=10_000), now)["state"] == "WEAK"
    assert short_signal(plan, q(101), now)["state"] == "WAIT"
    assert short_signal(plan, q(101, low=99.5), now)["state"] == "FAILED"
    assert short_signal(plan, q(95), now)["state"] == "EXTENDED"
    assert short_signal(plan, q(105), now)["state"] == "AVOID"
    assert sell_level_signal({"sell_below": 100.0}, 99.0)["state"] == "AT SELL LEVEL"
    assert sell_level_signal({"sell_below": 100.0}, 101.0) is None
