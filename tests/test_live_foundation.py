import sqlite3
from datetime import datetime

import pytest
from fastapi.testclient import TestClient

from code.live import audit, db, state
from code.live.market_data import SimulatedProvider
from code.live.session import IST, MarketSession


@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "live.db")
    db.migrate()
    return tmp_path / "live.db"


def test_migrations_apply_once(temp_db):
    assert db.migrate() == []
    tables = {r[0] for r in sqlite3.connect(temp_db).execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"audit_log", "app_state", "alerts", "proposed_orders", "broker_orders", "risk_events", "push_subscriptions"} <= tables


def test_audit_chain_verifies_and_detects_tampering(temp_db):
    for i in range(3):
        audit.record("test_event", "tester", "thing", i, {"n": i})
    assert audit.verify() == {"ok": True, "rows": 3}
    conn = sqlite3.connect(temp_db)
    conn.execute("UPDATE audit_log SET detail='{\"n\": 99}' WHERE id=2")
    conn.commit()
    result = audit.verify()
    assert result["ok"] is False and result["broken_at"] == 2


def test_default_mode_is_paper_and_halt_is_off():
    assert state.mode() == "PAPER_TRADING"
    assert state.is_halted() is False


def test_halt_and_mode_changes_are_audited():
    state.set_halt(True, "papa", "testing")
    state.set_mode("ALERT_ONLY", "papa")
    assert state.snapshot() == {"mode": "ALERT_ONLY", "halted": True}
    actions = [r["action"] for r in audit.recent()]
    assert actions[:2] == ["mode_changed", "halt_on"]


def test_live_execution_is_not_a_mode():
    with pytest.raises(ValueError):
        state.set_mode("LIVE_EXECUTION", "papa")


HOLIDAYS = {"2026-10-02": "Mahatma Gandhi Jayanti"}


@pytest.mark.parametrize("when, phase, is_open", [
    (datetime(2026, 10, 5, 9, 3, tzinfo=IST), "PRE_OPEN", False),
    (datetime(2026, 10, 5, 9, 10, tzinfo=IST), "CLOSED", False),     # gap between pre-open and open
    (datetime(2026, 10, 5, 9, 15, tzinfo=IST), "OPEN", True),
    (datetime(2026, 10, 5, 15, 29, tzinfo=IST), "OPEN", True),
    (datetime(2026, 10, 5, 15, 30, tzinfo=IST), "CLOSED", False),
    (datetime(2026, 10, 3, 11, 0, tzinfo=IST), "WEEKEND", False),
    (datetime(2026, 10, 2, 11, 0, tzinfo=IST), "HOLIDAY", False),
])
def test_session_phases(when, phase, is_open):
    s = MarketSession(HOLIDAYS).state(when)
    assert (s.phase, s.is_open) == (phase, is_open)


def test_next_open_skips_weekend_and_holiday():
    s = MarketSession({"2026-10-05": "Test holiday"}).state(datetime(2026, 10, 2, 16, 0, tzinfo=IST))
    assert s.next_open == datetime(2026, 10, 6, 9, 15, tzinfo=IST)
    assert MarketSession(HOLIDAYS).state(datetime(2026, 10, 2, 11, 0, tzinfo=IST)).holiday_name == "Mahatma Gandhi Jayanti"


def test_simulated_quotes_are_labelled_and_deterministic():
    p = SimulatedProvider({"DEMO.NS": 100.0})
    when = datetime(2026, 10, 5, 11, 0, tzinfo=IST)
    a, b = p.quotes(["DEMO.NS", "MISSING.NS"], now=when), p.quotes(["DEMO.NS"], now=when)
    q = a["DEMO.NS"]
    assert "MISSING.NS" not in a and q == b["DEMO.NS"]
    assert q.simulated is True and q.source == "simulated"
    assert q.day_low <= q.ltp <= q.day_high and 80 < q.ltp < 120


def test_settings_defaults_are_safe(monkeypatch):
    from code.app import settings
    assert settings.APP_MODE in settings.APP_MODES and "LIVE" not in " ".join(settings.APP_MODES)
    assert settings.TELEGRAM_DRY_RUN is True or (settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)


def test_healthz_is_public_and_halt_requires_login(monkeypatch):
    import bcrypt
    from code.app import main
    monkeypatch.setattr(main, "APP_PIN_HASH", bcrypt.hashpw(b"123456", bcrypt.gensalt()).decode())
    monkeypatch.setattr(main, "SESSION_SECRET", "test-secret")
    monkeypatch.setattr(main, "market_session", lambda: MarketSession(HOLIDAYS))
    client = TestClient(main.app)
    r = client.get("/healthz")
    assert r.status_code == 200 and r.json()["checks"]["mode"] == "PAPER_TRADING"
    assert client.post("/api/halt", data={"on": "1"}, headers={"Origin": "http://testserver"}).status_code == 401


def test_nifty_gap_filled_from_nse_with_rescaled_volume(monkeypatch):
    import pandas as pd
    from code.pipeline import nightly_pipeline as np_
    nse = {pd.Timestamp("2026-10-01").date(): {"open": 1, "high": 1, "low": 1, "close": 100.0, "volume": 1000.0},
           pd.Timestamp("2026-10-05").date(): {"open": 1, "high": 1, "low": 1, "close": 101.0, "volume": 2000.0}}
    monkeypatch.setattr(np_, "_nse_nifty_row", lambda d, s: dict(nse[d]) if d in nse else None)
    monkeypatch.setattr("code.research.bhavcopy._session", lambda: None)
    nifty = pd.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [100.0], "volume": [10.0]},
                         index=pd.DatetimeIndex(["2026-10-01"]))
    ref = pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(["2026-10-05"]))
    out = np_._patch_nifty_from_nse(nifty, ref)
    assert list(out.index.date.astype(str)) == ["2026-10-01", "2026-10-05"]   # 2 Oct holiday skipped
    assert out["close"].iloc[-1] == 101.0 and out["volume"].iloc[-1] == 20.0  # 2000 × (10 / 1000)


def test_nan_yahoo_bar_filled_from_bhavcopy(monkeypatch):
    import pandas as pd
    from code.pipeline import nightly_pipeline as np_
    bhav = pd.DataFrame({"symbol": ["CUPID"], "date": [pd.Timestamp("2026-10-05")], "open": [313.0], "high": [320.0],
                         "low": [310.0], "close": [318.0], "volume": [9.0]}).set_index(["symbol", "date"])
    monkeypatch.setattr(np_, "_recent_bhav", lambda key: bhav)
    nan = float("nan")
    df = pd.DataFrame({"open": [313.5, nan], "high": [314.3, nan], "low": [308.5, nan], "close": [312.5, nan],
                       "volume": [25.0, 15.0]}, index=pd.DatetimeIndex(["2026-10-01", "2026-10-05"]))
    out = np_._patch_from_bhavcopy("CUPID.NS", df.copy())
    assert out.loc["2026-10-05", "close"] == 318.0 and out.loc["2026-10-05", "volume"] == 15.0   # Yahoo volume kept
    assert np_._patch_from_bhavcopy("^NSEI", df.copy())["close"].isna().sum() == 1              # non-.NS untouched
