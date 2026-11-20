"""Roles from roles.yaml and the calls that drive them (Ollama locally, Anthropic in the cloud)."""
from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict

import requests
import yaml

ROLES_PATH = Path(__file__).with_name("roles.yaml")
ROLE_NAMES = ("librarian", "idea_maker", "coder", "critic")


def load_config(path: Path = ROLES_PATH) -> Dict[str, Any]:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    for name, role in cfg["roles"].items():
        role["role"] = name
        for k in ("model", "family", "provider", "temperature", "system"):
            if k not in role:
                raise ValueError(f"roles.yaml: {name} is missing {k}")
        if role["provider"] not in ("ollama", "cloud"):
            raise ValueError(f"roles.yaml: {name} provider must be ollama or cloud")
    missing = [r for r in ROLE_NAMES if r not in cfg["roles"]]
    if missing:
        raise ValueError(f"roles.yaml: missing roles {missing}")
    r = cfg["roles"]
    if r["critic"]["family"] == r["idea_maker"]["family"] and not cfg.get("allow_same_family"):
        raise ValueError("roles.yaml: the critic must be a different model family than the idea_maker "
                         "(or set allow_same_family: true)")
    return cfg


def chat(cfg: Dict[str, Any], role: Dict[str, Any], system: str, user: str) -> str:
    if role["provider"] == "cloud":
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=120,
                          headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                          json={"model": role["model"], "max_tokens": 2000, "temperature": role["temperature"],
                                "system": system, "messages": [{"role": "user", "content": user}]})
        r.raise_for_status()
        return "".join(b.get("text", "") for b in r.json()["content"])
    body = {"model": role["model"], "stream": False, "keep_alive": "30m",
            "options": {"temperature": role["temperature"], "seed": cfg.get("seed", 7)},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if "think" in role:      # reasoning models (qwen3): false skips the long hidden reasoning
        body["think"] = role["think"]
    for attempt in range(3):
        r = requests.post(f"{cfg['ollama_host']}/api/chat", timeout=900, json=body)
        # A model load can fail when RAM is nearly full (Ollama answers 500); memory is often freed
        # a few seconds later, so wait and retry before giving up.
        if r.status_code != 500 or attempt == 2:
            break
        time.sleep(15)
    if r.status_code == 500:
        raise RuntimeError(f"Ollama could not run {role['model']} (usually not enough free RAM; close other apps): {r.text[:200]}")
    r.raise_for_status()
    return r.json()["message"]["content"]


def unload(cfg: Dict[str, Any], role: Dict[str, Any]) -> None:
    """Free the model's RAM now instead of after Ollama's idle timeout."""
    if role["provider"] == "ollama":
        requests.post(f"{cfg['ollama_host']}/api/generate", json={"model": role["model"], "keep_alive": 0}, timeout=60)


def parse_json(text: str) -> Any:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)      # reasoning models
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start = min([i for i in (text.find("["), text.find("{")) if i >= 0], default=0)
    return json.loads(text[start:])


def parse_code(text: str) -> str:
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, flags=re.S)
    code = (m.group(1) if m else text).strip() + "\n"
    if "def positions" not in code:
        raise ValueError("no positions() function in the coder's reply")
    return code
