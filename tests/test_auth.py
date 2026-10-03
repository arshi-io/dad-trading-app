import time

import bcrypt
import pytest

from code.app import main


@pytest.fixture(autouse=True)
def _reset_rate_limit_state():
    main._pin_attempts.clear()
    yield
    main._pin_attempts.clear()


@pytest.fixture
def pin_hash(monkeypatch):
    hashed = bcrypt.hashpw(b"255202", bcrypt.gensalt()).decode("utf-8")
    monkeypatch.setattr(main, "APP_PIN_HASH", hashed)
    return hashed


@pytest.fixture
def session_secret(monkeypatch):
    monkeypatch.setattr(main, "SESSION_SECRET", "test-only-secret")


def test_check_pin_accepts_the_correct_pin(pin_hash):
    assert main._check_pin("255202") is True


def test_check_pin_rejects_a_wrong_pin(pin_hash):
    assert main._check_pin("000000") is False


def test_check_pin_rejects_when_app_pin_unset(monkeypatch):
    monkeypatch.setattr(main, "APP_PIN_HASH", None)
    assert main._check_pin("255202") is False


def test_check_pin_never_crashes_on_a_malformed_hash(monkeypatch):
    """If someone sets APP_PIN to plaintext by mistake, bcrypt.checkpw raises
    -- must degrade to "wrong PIN", never a 500."""
    monkeypatch.setattr(main, "APP_PIN_HASH", "255202")
    assert main._check_pin("255202") is False


def test_session_round_trips(session_secret):
    token = main._sign_session(int(time.time()) + 3600)
    assert main._verify_session(token) is True


def test_session_rejects_a_tampered_payload(session_secret):
    token = main._sign_session(int(time.time()) + 3600)
    payload, _, sig = token.partition(".")
    forged = f"{int(payload) + 999999}.{sig}"
    assert main._verify_session(forged) is False


def test_session_rejects_an_expired_token(session_secret):
    token = main._sign_session(int(time.time()) - 10)
    assert main._verify_session(token) is False


def test_session_rejects_garbage(session_secret):
    assert main._verify_session("not-a-real-cookie") is False
    assert main._verify_session(None) is False


def test_session_rejects_everything_without_a_secret(monkeypatch):
    monkeypatch.setattr(main, "SESSION_SECRET", None)
    assert main._verify_session("anything.anything") is False


def test_rate_limit_locks_out_after_five_failures():
    key = "1.2.3.4"
    for _ in range(main.RATE_LIMIT_MAX_ATTEMPTS - 1):
        assert main._is_locked_out(key) is None
        main._record_failed_attempt(key)
    assert main._is_locked_out(key) is None
    main._record_failed_attempt(key)  # 5th failure
    locked_seconds = main._is_locked_out(key)
    assert locked_seconds is not None
    assert 0 < locked_seconds <= main.RATE_LIMIT_LOCKOUT_SECONDS + 1


def test_rate_limit_is_scoped_per_key():
    for _ in range(main.RATE_LIMIT_MAX_ATTEMPTS):
        main._record_failed_attempt("attacker-ip")
    assert main._is_locked_out("attacker-ip") is not None
    assert main._is_locked_out("dads-ip") is None


def test_successful_login_clears_the_attempt_count():
    key = "5.6.7.8"
    main._record_failed_attempt(key)
    main._record_failed_attempt(key)
    main._clear_attempts(key)
    assert main._is_locked_out(key) is None
    assert key not in main._pin_attempts


def test_safe_next_allows_a_local_path():
    assert main._safe_next("/screen") == "/screen"


@pytest.mark.parametrize("bad", ["//evil.com", "http://evil.com", "evil.com", ""])
def test_safe_next_rejects_open_redirect_attempts(bad):
    assert main._safe_next(bad) == "/"
