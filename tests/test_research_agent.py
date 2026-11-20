import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from gatekeeper import gate
from research.agent import critic, journal, lab, session, team

CYCLE = """```python
import numpy as np, pandas as pd
def positions(df):
    i = np.arange(len(df))
    return pd.Series(((i // 5) % 2 == 0).astype(int), index=df.index)
```"""
PEEK = """```python
def positions(df):
    return (df["close"].shift(-1) > df["close"]).astype(int)
```"""


def _df():
    idx = pd.bdate_range("2011-01-03", "2026-06-30")
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(0.002 + rng.normal(0, 0.002, len(idx))))
    return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "_load", lambda s: _df())
    monkeypatch.setattr(gate, "DB_PATH", tmp_path / "attempts.db")
    lab._data.cache_clear()
    return {"conn": journal.connect(tmp_path / "journal.db"), "cfg": team.load_config(), "lib": tmp_path / "lib"}


class FakeTeam:
    """Stands in for Ollama: answers per role and fails if two local models are ever loaded at once."""

    def __init__(self, code=(CYCLE, PEEK), critic="pass", ensembles="[]"):
        self.code, self.critic_verdict, self.ensembles = list(code), critic, ensembles
        self.n_ideas = len(self.code)
        self.loaded, self.events = None, []

    def llm(self, cfg, role, system, user):
        if role["provider"] == "ollama":
            assert self.loaded in (None, role["model"]), f"{role['model']} loaded while {self.loaded} still in RAM"
            self.loaded = role["model"]
        self.events.append(("call", role["role"]))
        r = role["role"]
        if r == "librarian":
            return json.dumps([{"idea": f"idea {i}", "source": "general knowledge"} for i in range(self.n_ideas)])
        if r == "idea_maker":
            k = int(re.search(r"idea (\d+)", user).group(1))
            return json.dumps({"family": f"fam{k}", "hypothesis": f"h{k}", "rule": f"r{k}"})
        if r == "coder":
            return self.ensembles if system == role.get("ensemble_system") else self.code.pop(0)
        return json.dumps({"verdict": self.critic_verdict, "reasons": ["looked fine"]})

    def free(self, cfg, role):
        self.events.append(("unload", role["role"]))
        if role["model"] == self.loaded:
            self.loaded = None


def _run(env, fake, symbols=("X",), submitted=None, **cfg):
    submitted = [] if submitted is None else submitted
    sub = lambda code, sym, name, att: submitted.append((sym, name, att)) or {"passed": True}
    return session.run_session(list(symbols), {**env["cfg"], **cfg}, env["conn"], fake.llm, fake.free, sub,
                               library=env["lib"]), submitted


# ---------- roles.yaml ----------------------------------------------------------------------

def test_swapping_a_model_is_a_yaml_edit(tmp_path):
    cfg = yaml.safe_load(team.ROLES_PATH.read_text())
    cfg["roles"]["coder"]["model"] = "deepseek-coder-v2:16b"
    p = tmp_path / "roles.yaml"
    p.write_text(yaml.safe_dump(cfg))
    assert team.load_config(p)["roles"]["coder"]["model"] == "deepseek-coder-v2:16b"


def test_critic_must_differ_from_idea_maker_family(tmp_path):
    cfg = yaml.safe_load(team.ROLES_PATH.read_text())
    assert cfg["roles"]["critic"]["family"] != cfg["roles"]["idea_maker"]["family"]   # the default
    cfg["roles"]["critic"]["family"] = cfg["roles"]["idea_maker"]["family"]
    p = tmp_path / "roles.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="different model family"):
        team.load_config(p)


# ---------- batching and memory -------------------------------------------------------------

def test_roles_run_in_batches_one_model_at_a_time(env):
    fake = FakeTeam()
    _run(env, fake)
    calls = [r for e, r in fake.events if e == "call"]
    order = [r for i, r in enumerate(calls) if i == 0 or calls[i - 1] != r]
    assert order == ["librarian", "idea_maker", "coder", "critic"]   # each role exactly once, in order
    for role in ("librarian", "idea_maker", "coder", "critic"):
        last_call = max(i for i, ev in enumerate(fake.events) if ev == ("call", role))
        assert fake.events[last_call + 1] == ("unload", role)        # unloaded right after its batch
    assert fake.loaded is None


# ---------- critic --------------------------------------------------------------------------

def _bt(**kw):
    s = {"n_trades": 40, "expectancy": 0.004, "win_rate": 0.55, "avg_win": 0.02, "avg_loss": -0.01,
         "max_drawdown": -0.1, "total_return": 0.2}
    bt = {"A": dict(s), "B": dict(s), "years": {"2019": 0.05, "2020": 0.06, "2021": 0.04}, "lookahead": [], "cost": 0.0025}
    bt["B"].update(kw.pop("B", {}))
    bt.update(kw)
    return bt


@pytest.mark.parametrize("bt,attempt,needle", [
    (_bt(lookahead=["2022-03-01"]), 1, "lookahead"),
    (_bt(B={"n_trades": 12}), 1, "too few trades"),
    (_bt(), 6, "too many attempts"),
    (_bt(years={"2019": -0.05, "2020": 0.30, "2021": -0.04}), 1, "one lucky period"),
    (_bt(cost=0.0), 1, "missing costs"),
    (_bt(B={"expectancy": -0.001}), 1, "no edge"),
])
def test_critic_catches(bt, attempt, needle):
    failed = critic.code_checks(bt, attempt, max_attempts=5)
    assert any(needle in f for f in failed), failed


def test_clean_result_passes_code_checks():
    assert critic.code_checks(_bt(), 1, 5) == []


def test_only_critic_passed_ideas_reach_the_gate(env):
    out, submitted = _run(env, FakeTeam())
    by_fam = {r["family"]: r for r in out["rows"]}
    assert by_fam["fam1"]["status"] == "critic_failed"                     # PEEK: lookahead
    assert any("lookahead" in c for c in by_fam["fam1"]["checks"])
    assert by_fam["fam0"]["status"] == "gated" and by_fam["fam0"]["gate_passed"] == 1
    assert [s[1] for s in submitted] == [f"journal #{by_fam['fam0']['id']} fam0"]


def test_coder_gets_one_repair_attempt(env):
    broken = "```python\ndef positions(df):\n    return df['close'].nope()\n```"
    fake = FakeTeam(code=[broken])                      # librarian offers one idea
    fake.code.append(CYCLE)                             # the coder's second reply is the fix
    out, _ = _run(env, fake)
    assert len(out["rows"]) == 1 and out["rows"][0]["status"] == "gated"   # one idea, repaired


def test_critic_model_can_veto(env):
    out, submitted = _run(env, FakeTeam(code=[CYCLE], critic="fail"))
    assert submitted == [] and out["rows"][0]["status"] == "critic_failed"


def test_repeated_attempts_are_counted_and_blocked(env):
    for _ in range(3):
        _run(env, FakeTeam(code=[CYCLE.replace("// 5", f"// {5 + _}")]), max_attempts=2)
    rows = journal.rows(env["conn"])
    assert [r["attempt_no"] for r in rows] == [1, 2, 3]
    assert any("too many attempts" in c for c in rows[-1]["checks"])


# ---------- journal -------------------------------------------------------------------------

def test_journal_records_models_per_role_and_never_deletes(env):
    _run(env, FakeTeam(), symbols=("X", "Y"))
    rows = journal.rows(env["conn"])
    assert len(rows) == 4                                                  # 2 ideas x 2 stocks
    assert rows[0]["models"]["critic"] == {"model": "gemma3:12b", "provider": "ollama"}
    with pytest.raises(Exception, match="never deleted"):
        env["conn"].execute("DELETE FROM ideas")
    rate = journal.pass_rate_by_model(env["conn"], "idea_maker")
    assert rate[0]["model"] == "qwen3:14b" and rate[0]["ideas"] == 4 and rate[0]["gate_passed"] == 2


# ---------- ensembles -----------------------------------------------------------------------

def test_vote_ensemble_is_logged_and_gated_like_any_idea(env):
    three = [CYCLE.replace("// 5", f"// {k}") for k in (4, 5, 6)]
    _run(env, FakeTeam(code=three))
    ids = [r["id"] for r in journal.rows(env["conn"])]
    spec = json.dumps([{"method": "vote", "members": ids}])
    out, submitted = _run(env, FakeTeam(code=[], ensembles=spec))
    ens = out["rows"][0]
    assert ens["kind"] == "ensemble" and ens["parents"] == ids and ens["status"] == "gated"
    assert ens["backtest"]["lookahead"] == []


def test_regime_ensemble_has_no_lookahead(env):
    code = lab.build_ensemble({"method": "regime", "map": {"Calm trend": 1, "Choppy": 2}},
                              {1: {"code": team.parse_code(CYCLE), "family": "a"},
                               2: {"code": team.parse_code(CYCLE.replace("== 0", "== 1")), "family": "b"}})["code"]
    bt = lab.backtest(code, "X")
    assert bt["lookahead"] == [] and bt["B"]["n_trades"] > 0


def test_ensemble_of_untested_strategies_is_rejected():
    with pytest.raises(ValueError, match="not tested"):
        lab.build_ensemble({"method": "vote", "members": [1, 2, 99]}, {1: {}, 2: {}})


# ---------- cloud review --------------------------------------------------------------------

def test_cloud_review_holds_ideas_then_sends_only_the_top_ones(env):
    for k in range(3):
        _run(env, FakeTeam(code=[CYCLE.replace("// 5", f"// {5 + k}")]), cloud_review=True)
    assert all(r["status"] == "awaiting_cloud" for r in journal.rows(env["conn"]))
    seen, submitted = [], []
    def cloud(cfg, role, system, user):
        assert role["provider"] == "cloud"
        seen.append(user)
        return '{"verdict": "pass", "reasons": []}'
    sub = lambda code, sym, name, att: submitted.append(name) or {"passed": False}
    session.cloud_review({**env["cfg"], "cloud_review_top": 2}, env["conn"], cloud, sub)
    rows = journal.rows(env["conn"])
    assert len(seen) == 2 and len(submitted) == 2
    assert sum(r["status"] == "awaiting_cloud" for r in rows) == 1
    assert all(r["models"]["cloud_critic"]["model"] == "claude-opus-5-5" for r in rows if r["cloud_verdict"])


# ---------- the gatekeeper stays outside ----------------------------------------------------

def test_agent_never_touches_gate_internals():
    """The agent uses get_data/submit plus the gate's data-free runner and trade maths; nothing that
    reads Period C, the attempt log or the scorer."""
    import ast
    allowed = {"get_data", "submit", "_run", "trades", "COST", "PERIODS"}
    for p in Path(session.__file__).parent.glob("*.py"):
        src = p.read_text(encoding="utf-8")
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("gatekeeper"):
                assert {a.name for a in node.names} <= allowed, (p.name, [a.name for a in node.names])
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id in ("gate", "gatekeeper"):
                assert node.attr in allowed, (p.name, node.attr)
        assert 'PERIODS["C"]' not in src and "attempts.db" not in src, p.name
