import json
from unittest.mock import patch

import pytest

from code.evaluator import memo as memo_mod
from code.evaluator.memo import build_bundle, generate_memo


@pytest.fixture(autouse=True)
def _isolated_memo_cache(tmp_path, monkeypatch):
    """Point the memo cache at a throwaway DB for the duration of each test.

    Two reasons this can't just flush the real one (which is what it used to
    do): a test reusing a ticker/date would otherwise pick up a stale payload
    from an earlier run, AND -- the one that actually bit -- flushing the
    shared code/data/papa.db wiped the running app's real memo cache, so the
    next Stock Room view paid a fresh 8-10s OpenAI call. Tests must not
    reach into live state.
    """
    monkeypatch.setattr(memo_mod, "DB_PATH", tmp_path / "test_memos.db")


def test_valid_json_and_fenced_json():
    fake = '```json\n{"score": 78, "verdict": "WAIT", "reasoning": "Plain English", "risk_flags": ["RBI"]}\n```'
    assert json.loads(fake.split("```json", 1)[1].split("```", 1)[0].strip())["score"] == 78


def test_garbage_falls_back_to_sentinel():
    bundle = build_bundle("RELIANCE", headlines=[{"title": "RBI policy", "source": "ET", "ts": "now"}], diary=["watch event"])
    with patch("code.evaluator.memo._call_llm", side_effect=["not json", "not json"]):
        memo = generate_memo(bundle, "RELIANCE", snapshot_date="2026-07-22", provider="claude")
    assert memo["verdict"] == "AVOID"
    assert memo["sentinel"] is True
    assert "couldn't be parsed as JSON" in memo["reasoning"]


def test_score_below_60_is_avoid():
    bundle = build_bundle("TCS", headlines=[{"title": "results", "source": "ET", "ts": "now"}], diary=["results"])
    with patch("code.evaluator.memo._call_llm", return_value='{"score": 55, "verdict": "WAIT", "reasoning": "Needs caution", "risk_flags": []}'):
        memo = generate_memo(bundle, "TCS", snapshot_date="2026-07-22", provider="claude")
    assert memo["verdict"] == "AVOID"


def test_no_clear_edge_is_not_forced_to_avoid_below_60():
    """NO_CLEAR_EDGE is a valid abstention, not a synonym for AVOID -- the
    score<60 enforcement must leave it standing, unlike TRADE_VALID/WAIT."""
    bundle = build_bundle("INFY", headlines=[], diary=[])
    with patch(
        "code.evaluator.memo._call_llm",
        return_value='{"score": 50, "verdict": "NO_CLEAR_EDGE", "reasoning": "Mixed signals", "risk_flags": [], "invalidation": []}',
    ):
        memo = generate_memo(bundle, "INFY", snapshot_date="2026-07-22", provider="claude")
    assert memo["verdict"] == "NO_CLEAR_EDGE"


def test_missing_openai_key_message_names_the_provider():
    with patch("code.evaluator.memo.os.getenv", return_value=None):
        payload = json.loads(memo_mod._call_openai("prompt"))
    assert payload["sentinel"] is True
    assert "MODEL_PROVIDER=openai" in payload["reasoning"]
    assert "OPENAI_API_KEY" in payload["reasoning"]


def test_missing_anthropic_key_message_names_the_provider():
    with patch("code.evaluator.memo.os.getenv", return_value=None):
        payload = json.loads(memo_mod._call_claude("prompt"))
    assert payload["sentinel"] is True
    assert "MODEL_PROVIDER=claude" in payload["reasoning"]
    assert "ANTHROPIC_API_KEY" in payload["reasoning"]


def test_openai_call_failure_message_is_not_the_missing_key_message():
    with patch("code.evaluator.memo.os.getenv", return_value="fake-key"), \
         patch("code.evaluator.memo.requests.post", side_effect=RuntimeError("connection reset")):
        payload = json.loads(memo_mod._call_openai("prompt"))
    assert payload["sentinel"] is True
    assert "OpenAI call failed" in payload["reasoning"]
    assert "connection reset" in payload["reasoning"]
    assert "API key" not in payload["reasoning"]


def test_successful_memo_is_not_flagged_sentinel():
    with patch("code.evaluator.memo._call_llm", return_value='{"score": 78, "verdict": "WAIT", "reasoning": "Plain English", "risk_flags": []}'):
        memo = generate_memo(build_bundle("HDFC"), "HDFC", snapshot_date="2026-07-22", provider="claude")
    assert memo["sentinel"] is False


def test_invalidation_field_passes_through():
    bundle = build_bundle("WIPRO", headlines=[], diary=[])
    fake = (
        '{"score": 72, "verdict": "TRADE_VALID", "reasoning": "Solid setup", "risk_flags": [], '
        '"invalidation": ["Close below 50 DMA at 450", "RS rank below 70"]}'
    )
    with patch("code.evaluator.memo._call_llm", return_value=fake):
        memo = generate_memo(bundle, "WIPRO", snapshot_date="2026-07-22", provider="claude")
    assert memo["invalidation"] == ["Close below 50 DMA at 450", "RS rank below 70"]
