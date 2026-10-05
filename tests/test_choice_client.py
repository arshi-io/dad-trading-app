import base64
import inspect
import json

import pytest

from code.live.brokers import choice_client
from code.live.brokers.choice_client import (ChoiceClient, ChoiceError, encrypt_mobile, parse_scrip_master,
                                             parse_touchline)
from code.live.market_data import ChoiceProvider

SCRIPS = (
    "Exchange,Segment,ISIN,Token,UnderlyingToken,Symbol,SecName,SecDesc,Series,Instrument,Expiry,OptionType,StrikePrice,PriceDivisor,MarketLot\n"
    "NSE,1,INE002A01018,99999,0,RELIANCE,RELIANCE-BE,RELIANCE IND,BE,      ,,  ,0,100,1\n"
    "NSE,1,INE002A01018,2885,0,RELIANCE,RELIANCE-EQ,RELIANCE IND,EQ,      ,,  ,0,100,1\n"
    "NSE,2,,35001,2885,RELIANCE,RELIANCE26OCTFUT,FUT,XX,FUTSTK,,  ,0,100,500\n"
    "BSE,3,INE002A01018,500325,0,RELIANCE,RELIANCE,RELIANCE IND,A,      ,,  ,0,100,1\n"
)


class Resp:
    def __init__(self, data, status=200):
        self._data, self.status_code = data, status

    def json(self):
        return self._data


class FakeChoice:
    def __init__(self, touch_rows=None):
        self.calls = []
        self.touch_rows = touch_rows if touch_rows is not None else [
            {"Token": 2885, "LTP": 141050, "Open": 140000, "High": 141500, "Low": 139800, "Close": 139900, "Volume": 1250000}]

    def __call__(self, method, url, headers=None, data=None, timeout=None):
        body = json.loads(data) if data else None
        self.calls.append((method, url, headers, body))
        if url.endswith("LoginTOTP") and "Validate" not in url and "GetClient" not in url:
            return Resp({"Status": "Success", "Response": "OTP sent"})
        if url.endswith("GetClientLoginTOTP"):
            return Resp({"Status": "Success", "Response": "123456"})
        if url.endswith("ValidateTOTP"):
            return Resp({"Status": "Success", "Response": {"SessionId": "SID-1", "BaseURL": "https://node7.example.in/"}})
        if url.endswith("MultipleTouchline"):
            return Resp({"Status": "Success", "Response": self.touch_rows})
        return Resp({"Status": "Fail", "Reason": "unknown"})


def make(fake=None, **kw):
    c = ChoiceClient("APIKEY", "MYVENDOR", "9876543210", http=fake or FakeChoice(), **kw)
    c.scrips = parse_scrip_master(SCRIPS)
    return c


def test_login_is_automatic_and_switches_to_returned_base_url():
    fake = FakeChoice()
    c = make(fake)
    c.login()
    paths = [u.split("/api/")[1] for _, u, _, _ in fake.calls]
    assert paths == ["OpenAPIV1/LoginTOTP", "OpenAPIV1/GetClientLoginTOTP", "OpenAPIV1/ValidateTOTP"]
    assert fake.calls[2][3] == {"MobileNo": base64.b64encode(b"9876543210").decode(), "OTP": "123456"}
    _, _, h, _ = fake.calls[0]
    assert h["Bearer"] == "APIKEY" and h["VendorId"] == "MYVENDOR" and "Authorization" not in h
    assert c.connected and c.session_id == "SID-1" and c.active_base == "https://node7.example.in"


def test_session_header_and_paise_conversion():
    fake = FakeChoice()
    c = make(fake)
    c.login()
    q = c.touchline(["RELIANCE.NS", "NOTLISTED.NS"])
    method, url, h, body = fake.calls[-1]
    assert url == "https://node7.example.in/api/OpenAPI/MultipleTouchline"
    assert h["Authorization"] == "SessionId SID-1" and body == {"MultipleSegToken": "1@2885"}
    assert q == {"RELIANCE.NS": {"ltp": 1410.5, "open": 1400.0, "high": 1415.0, "low": 1398.0, "prev_close": 1399.0, "volume": 1250000.0}}


def test_unrecognised_touchline_keeps_raw_sample_and_returns_nothing():
    c = make(FakeChoice(touch_rows=[{"Token": 2885, "SomethingElse": 1}]))
    c.login()
    assert c.touchline(["RELIANCE.NS"]) == {} and c.last_raw is not None


def test_failed_step_names_the_step():
    class Bad(FakeChoice):
        def __call__(self, method, url, **kw):
            if url.endswith("GetClientLoginTOTP"):
                return Resp({"Status": "Fail", "Reason": "Invalid Vendor"})
            return super().__call__(method, url, **kw)
    with pytest.raises(ChoiceError, match="GetClientLoginTOTP: Invalid Vendor"):
        make(Bad()).login()


def test_http_401_explains_ip_or_key():
    c = make(lambda *a, **k: Resp("", status=401))
    with pytest.raises(ChoiceError, match="IP"):
        c.login()


def test_scrip_master_keeps_nse_cash_eq_only():
    s = parse_scrip_master(SCRIPS)
    assert s == {"RELIANCE": {"token": 2885, "divisor": 100.0, "series": "EQ"}}


def test_mobile_encoding_aes_when_key_issued():
    from cryptography.hazmat.primitives import padding
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    key, iv = "k" * 32, "i" * 16
    enc = encrypt_mobile("9876543210", key, iv)
    dec = Cipher(algorithms.AES(key.encode()), modes.CBC(iv.encode())).decryptor()
    raw = dec.update(base64.b64decode(enc)) + dec.finalize()
    unpad = padding.PKCS7(128).unpadder()
    assert unpad.update(raw) + unpad.finalize() == b"9876543210"
    assert encrypt_mobile("9876543210") == base64.b64encode(b"9876543210").decode()
    with pytest.raises(ChoiceError):
        encrypt_mobile("1", "short", "short")


def test_touchline_parser_handles_wrapped_lists():
    rows = {"Status": "Success", "Response": {"lstTouchline": [{"Token": "2885", "LTP": "50000"}]}}
    assert parse_touchline(rows, {2885: 100.0}) == {2885: {"ltp": 500.0}}


def test_choice_client_cannot_trade_or_move_money():
    src = inspect.getsource(choice_client).lower()
    for forbidden in ("neworder", "modifyorder", "cancelorder", "payout", "debitfunds", "payment", "verifydis"):
        assert forbidden not in src.replace("no order, payment, payout or dis endpoints", "")


def test_provider_wraps_choice_quotes():
    c = make()
    c.login()
    q = ChoiceProvider(c).quotes(["RELIANCE.NS"])["RELIANCE.NS"]
    assert q.ltp == 1410.5 and q.prev_close == 1399.0 and q.simulated is False and q.source == "choice"
    assert ChoiceProvider(make()).quotes(["RELIANCE.NS"]) == {}    # not logged in -> nothing
