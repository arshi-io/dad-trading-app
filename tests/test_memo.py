import json
from unittest.mock import patch

from code.evaluator.memo import build_bundle, generate_memo


def test_valid_json_and_fenced_json():
    fake = '```json\n{"score": 78, "verdict": "WAIT", "reasoning": "Plain English", "risk_flags": ["RBI"]}\n```'
    assert json.loads(fake.split("```json", 1)[1].split("```", 1)[0].strip())["score"] == 78


def test_garbage_falls_back_to_sentinel():
    bundle = build_bundle("RELIANCE", headlines=[{"title": "RBI policy", "source": "ET", "ts": "now"}], diary=["watch event"])
    with patch("code.evaluator.memo._call_llm", side_effect=["not json", "not json"]):
        memo = generate_memo(bundle, "RELIANCE", snapshot_date="2026-07-22", provider="claude")
    assert memo["verdict"] == "AVOID"
    assert memo["reasoning"] == "Analysis unavailable"


def test_score_below_60_is_avoid():
    bundle = build_bundle("TCS", headlines=[{"title": "results", "source": "ET", "ts": "now"}], diary=["results"])
    with patch("code.evaluator.memo._call_llm", return_value='{"score": 55, "verdict": "WAIT", "reasoning": "Needs caution", "risk_flags": []}'):
        memo = generate_memo(bundle, "TCS", snapshot_date="2026-07-22", provider="claude")
    assert memo["verdict"] == "AVOID"
