import hashlib
import inspect
import json
from datetime import datetime, timedelta

import pytest

from code.live.brokers import motilal
from code.live.brokers.motilal import MotilalClient, MotilalError, parse_nse_equity_codes, totp_now
from code.live.feed import LiveFeed, valid
from code.live.market_data import IST, MotilalProvider, Quote, SimulatedProvider
from code.live.session import MarketSession


class FakeResp:
    def __init__(self, data):
        self._data = data
        self.status_code = 200

    def json(self):
        return self._data


class FakeMotilal:
    """Records every call; answers like the documented API."""

    def __init__(self, dealer=False):
        self.calls = []
        self.dealer = dealer

    def __call__(self, url, headers=None, data=None, timeout=None):
        body = json.loads(data)
        self.calls.append((url, headers, body))
        if url.endswith("authdirectapi"):
            return FakeResp({"status": "SUCCESS", "AuthToken": "AUTH123", "isAuthTokenVerified": "TRUE"})
        if url.endswith("getaccesstoken"):
            return FakeResp({"status": "SUCCESS", "accesstoken": "ACC456"})
        if url.endswith("getltpdata"):
            if self.dealer and "clientcode" not in body:
                return FakeResp({"status": "FAILED", "errorcode": "MO1062", "message": "Please Provide Client Code"})
            return FakeResp({"status": "SUCCESS", "data": {"ltp": 31250, "open": 31350, "high": 31430, "low": 30850, "close": 31165, "volume": 25539188}})
        if url.endswith("getscripsbyexchangename"):
            return FakeResp({"status": "SUCCESS", "data": [
                {"scripshortname": "CUPID", "scripcode": 9999, "optiontype": "BE"},
                {"scripshortname": "CUPID", "scripcode": 1234, "optiontype": "EQ"},
                {"scripshortname": "RELIANCE", "scripcode": 2885, "optiontype": "EQ"},
            ]})
        return FakeResp({"status": "FAILED", "message": "unknown"})


@pytest.fixture
def fake():
    return FakeMotilal()


@pytest.fixture
def client(fake):
    return MotilalClient("KEY", "SECRET", env="uat", post=fake)


def test_login_hashes_password_with_api_key_and_omits_auth_header(client, fake):
    client.login("AB1234", "hunter2", "01/01/1960", "123456")
    url, headers, body = fake.calls[0]
    assert url == "https://openapi.motilaloswaluat.com/rest/login/v7/authdirectapi"
    assert body == {"userid": "AB1234", "password": hashlib.sha256(b"hunter2KEY").hexdigest(), "2FA": "01/01/1960", "totp": "123456"}
    assert "Authorization" not in headers and headers["ApiKey"] == "KEY" and headers["apisecretkey"] == "SECRET"
    assert client.connected and client.auth_token == "AUTH123" and client.access_token == "ACC456"


def test_failed_login_raises_with_errorcode(fake):
    bad = MotilalClient("KEY", post=lambda *a, **k: FakeResp({"status": "FAILED", "message": "Invalid TOTP", "errorcode": "MO1093"}))
    with pytest.raises(MotilalError) as e:
        bad.login("AB1234", "x", "01/01/1960", "000000")
    assert e.value.errorcode == "MO1093" and not bad.connected


def test_ltp_converts_paise_to_rupees_and_sends_auth(client, fake):
    client.login("AB1234", "pw", "dob", "123456")
    q = client.ltp(1234)
    assert q == {"ltp": 312.5, "open": 313.5, "high": 314.3, "low": 308.5, "prev_close": 311.65, "volume": 25539188}
    url, headers, body = fake.calls[-1]
    assert body == {"exchange": "NSE", "scripcode": 1234} and headers["Authorization"] == "AUTH123"


def test_dealer_login_retries_ltp_with_clientcode():
    fake = FakeMotilal(dealer=True)
    c = MotilalClient("KEY", post=fake)
    c.login("DEALER1", "pw", "dob")
    assert c.ltp(1234)["ltp"] == 312.5
    assert fake.calls[-1][2]["clientcode"] == "DEALER1"


def test_instrument_codes_prefer_eq_series(client, tmp_path):
    client.login("AB1234", "pw", "dob")
    codes = client.nse_equity_codes(cache_path=tmp_path / "codes.json")
    assert codes == {"CUPID": 1234, "RELIANCE": 2885}
    assert parse_nse_equity_codes([{"scripshortname": "X", "scripcode": 1, "optiontype": "EQ"},
                                   {"scripshortname": "X", "scripcode": 2, "optiontype": "BE"}]) == {"X": 1}


def test_client_cannot_place_orders():
    src = inspect.getsource(motilal).lower()
    assert "placeorder" not in src and "modifyorder" not in src and "cancelorder" not in src


def test_token_expires_at_six_am():
    c = MotilalClient("KEY")
    c.auth_token = "T"
    c.logged_in_at = datetime.now() - timedelta(days=2)
    assert c.connected is False


def test_totp_matches_rfc6238_vector():
    # RFC 6238 SHA1 secret "12345678901234567890", T=59 -> 94287082 (8 digits)
    secret = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ"
    assert totp_now(secret, at=59, digits=8) == "94287082"
    assert len(totp_now(secret)) == 6


def test_motilal_provider_returns_real_quotes(client):
    client.login("AB1234", "pw", "dob")
    provider = MotilalProvider(client)
    provider._codes = {"CUPID": 1234}
    q = provider.quotes(["CUPID.NS", "UNKNOWN.NS"])
    assert list(q) == ["CUPID.NS"] and q["CUPID.NS"].simulated is False and q["CUPID.NS"].ltp == 312.5


def test_provider_returns_nothing_when_not_connected():
    assert MotilalProvider(MotilalClient("KEY")).quotes(["CUPID.NS"]) == {}


def _q(ltp, prev=100.0, lo=95.0, hi=105.0, ts=None):
    return Quote("A.NS", ltp, prev, 100.0, hi, lo, 1000, ts or datetime.now(IST), "test", False)


def test_quote_validation_rejects_bad_prints():
    assert valid(_q(101))
    assert not valid(_q(0))                 # zero price
    assert not valid(_q(120))               # outside the day's range
    assert not valid(_q(140, lo=90, hi=150))  # beyond any NSE price band


def test_feed_keeps_valid_quotes_and_flags_staleness():
    old = datetime.now(IST) - timedelta(minutes=10)
    good, bad = _q(101, ts=old), Quote("B.NS", 0, 100, 0, 0, 0, 0, old, "test", False)

    class P:
        name = "test"

        def quotes(self, symbols):
            return {"A.NS": good, "B.NS": bad}

    feed = LiveFeed(lambda: P(), lambda: ["A.NS", "B.NS"], session=lambda: MarketSession({}), stale_seconds=90)
    assert feed.poll_once() == 2
    view = feed.view()
    assert list(view) == ["A.NS"] and view["A.NS"]["stale"] is True and "B.NS" in feed.rejected


def test_feed_survives_provider_errors():
    class Boom:
        name = "boom"

        def quotes(self, symbols):
            raise ConnectionError("network down")

    feed = LiveFeed(lambda: Boom(), lambda: ["A.NS"], session=lambda: MarketSession({}))
    assert feed.poll_once() == 0 and "network down" in feed.last_error


def test_dotenv_strips_inline_comments(tmp_path, monkeypatch):
    from code.app import settings
    env = tmp_path / ".env"
    env.write_text('MOTILAL_ENV=uat   # test env\nQUOTED="a # b"\n', encoding="utf-8")
    monkeypatch.delenv("MOTILAL_ENV", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)
    settings._load_dotenv(env)
    import os
    assert os.environ["MOTILAL_ENV"] == "uat" and os.environ["QUOTED"] == "a # b"


def test_simulated_provider_used_for_demo():
    assert SimulatedProvider({"A.NS": 10.0}).quotes(["A.NS"])["A.NS"].simulated is True


def _jwt(payload: dict) -> str:
    import base64
    enc = lambda d: base64.urlsafe_b64encode(json.dumps(d).encode()).decode().rstrip("=")  # noqa: E731
    return f"{enc({'alg': 'RS256'})}.{enc(payload)}.sig"


def test_choice_key_info_reads_expiry_and_locked_ip():
    from code.live.brokers.choice import key_info
    info = key_info(_jwt({"UserId": "ABC1", "iss": "FINX", "iat": 1791081962, "exp": 1793673962, "CliIPAddress": "203.0.113.7"}))
    assert info["user"] == "ABC1" and info["locked_ip"] == "203.0.113.7"
    assert info["expires"].strftime("%Y-%m-%d") == "2026-11-03"
    assert key_info("not-a-jwt") is None


def test_choice_ip_match_compares_like_with_like():
    from code.live.brokers.choice import ip_matches
    ips = {"ipv4": "203.0.113.7", "ipv6": "2405:201::1"}
    assert ip_matches("203.0.113.7", ips) is True
    assert ip_matches("2405:201:0:0::1", ips) is True          # same address, different spelling
    assert ip_matches("2405:201::2", ips) is False
    assert ip_matches("2405:201::2", {"ipv4": None, "ipv6": None}) is None   # couldn't check
