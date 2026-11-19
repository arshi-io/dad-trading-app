"""One-off check: can this environment reach the LLM the setup review uses?

Reads the same settings as the app (MODEL_PROVIDER, OPENAI_API_KEY / ANTHROPIC_API_KEY, the model
names in code/evaluator/memo.py), makes one tiny request and prints what came back. Never prints
the key: only its first 7 characters, its length and whether it has stray quotes or spaces.

    python scripts/check_llm.py            # locally (reads .env like the app)
    railway run python scripts/check_llm.py   # with Railway's variables, from this laptop
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests  # noqa: E402

from code.app import settings  # noqa: E402,F401  -- loads .env exactly like the app does
from code.app import config  # noqa: E402
from code.evaluator import memo  # noqa: E402

KEY_PATTERN = re.compile(r"sk-[A-Za-z0-9_\-*]{4,}")


def describe_key(raw):
    if raw is None:
        return "NOT SET", None
    problems = []
    if raw != raw.strip():
        problems.append("leading/trailing whitespace")
    if raw.strip()[:1] in ("'", '"') or raw.strip()[-1:] in ("'", '"'):
        problems.append("wrapped in quotes")
    if "\n" in raw or "\r" in raw:
        problems.append("contains a line break")
    shown = f"{raw.strip()[:7]}… (length {len(raw)})"
    return shown + (f"  PROBLEM: {', '.join(problems)}" if problems else ""), raw.strip().strip("'\"")


def redact(text):
    return KEY_PATTERN.sub("sk-…[redacted]", text)[:300]


def main():
    provider = config.MODEL_PROVIDER
    if provider == "openai":
        name, model = "OPENAI_API_KEY", memo.OPENAI_MODEL
        url = "https://api.openai.com/v1/chat/completions"
    else:
        name, model = "ANTHROPIC_API_KEY", "claude-sonnet-4-6"  # as hard-coded in memo._call_claude
        url = "https://api.anthropic.com/v1/messages"
    shown, key = describe_key(os.environ.get(name))
    print(f"provider     : {provider}  (MODEL_PROVIDER={os.environ.get('MODEL_PROVIDER', '<unset, app defaults to claude>')})")
    print(f"model        : {model}")
    print(f"{name:<13}: {shown}")
    if not key:
        print("result       : no key -- the app shows 'Analysis unavailable ... no key is configured'")
        return 1

    if provider == "openai":
        r = requests.post(url, timeout=30, headers={"Authorization": f"Bearer {key}"},
                          json={"model": model, "messages": [{"role": "user", "content": "Reply with OK"}],
                                "max_completion_tokens": 16})
        ids = {h: r.headers.get(h) for h in ("openai-organization", "openai-project", "x-request-id")}
    else:
        r = requests.post(url, timeout=30, headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
                          json={"model": model, "max_tokens": 5, "messages": [{"role": "user", "content": "Reply with OK"}]})
        ids = {"request-id": r.headers.get("request-id")}

    print(f"HTTP status  : {r.status_code} {r.reason}")
    for k, v in ids.items():
        if v:
            print(f"{k:<13}: {v}")
    if r.ok:
        print("result       : OK -- the key works for this model")
        return 0
    try:
        err = r.json().get("error", {})
        print(f"error type   : {err.get('type')} / code: {err.get('code')}")
        print(f"error message: {redact(str(err.get('message')))}")
    except ValueError:
        print(f"error body   : {redact(r.text)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
