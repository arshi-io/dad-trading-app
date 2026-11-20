"""Backtests on Periods A and B, and ensemble strategies built from already-tested ones.

Uses the gatekeeper's own runner and trade maths so the agent's numbers are computed exactly the
way the gate will compute them; data comes only through gatekeeper.get_data (A and B).
"""
from __future__ import annotations

import inspect
from functools import lru_cache
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import gatekeeper
from gatekeeper.gate import COST, PERIODS, _run, trades

LOOKAHEAD_CHECKS = 8


@lru_cache(maxsize=16)
def _data(symbol: str) -> pd.DataFrame:
    return gatekeeper.get_data(symbol, "B")


def backtest(source: str, symbol: str) -> Dict[str, Any]:
    df = _data(symbol)
    if len(df) < 600:
        raise ValueError(f"{symbol}: not enough history")
    cuts = list(df.index[np.linspace(300, len(df) - 2, LOOKAHEAD_CHECKS).astype(int)])
    run = _run(source, df, cuts)
    pos, close = run["pos"], df["close"]
    leaked = [str(t.date()) for t, v in run["cut"].items()
              if round(float(np.clip(np.nan_to_num(v), -1, 1))) != pos.get(t, 0)]

    def window(start: str, end: str) -> Dict[str, Any]:
        w = (pos.index >= pd.Timestamp(start)) & (pos.index <= pd.Timestamp(end))
        return trades(pos[w], close[w])

    out = {p: window(*PERIODS[p]) for p in ("A", "B")}
    years = {str(y): window(f"{y}-01-01", f"{y}-12-31")["total_return"] for y in sorted(set(pos.index.year))}
    for p in ("A", "B"):
        out[p].pop("trade_returns")
    return {**out, "years": years, "lookahead": leaked, "cost": COST}


# ---------- ensembles -----------------------------------------------------------------------

_MEMBER_LOADER = '''
def _member(src):
    ns = {}
    exec(src, ns)
    return ns["positions"]
'''


def vote_source(members: Dict[int, str]) -> str:
    """Majority vote: long if more members are long than short, short if the reverse."""
    return ("import numpy as np\nimport pandas as pd\n"
            f"_MEMBERS = {members!r}\n{_MEMBER_LOADER}\n"
            "def positions(df):\n"
            "    total = sum(pd.Series(_member(s)(df), index=df.index).fillna(0) for s in _MEMBERS.values())\n"
            "    return np.sign(total)\n")


def regime_source(mapping: Dict[str, int], members: Dict[int, str]) -> str:
    """Use the member mapped to today's HMM state (filtered, forward-only); flat in unmapped states.

    The HMM is fitted once on the stock's first MIN_TRAIN feature days and no trades are taken
    inside that window, so no day is labelled by a model that saw its future.
    """
    from code.regime import hmm_regime
    # ponytail: regimes come from the stock's own prices; feed NIFTY into df to use the market regime.
    return ("import numpy as np\nimport pandas as pd\n"
            f"_HMM = {inspect.getsource(hmm_regime)!r}\n"
            "import sys, types\n_m = sys.modules['hmm_regime'] = types.ModuleType('hmm_regime')\n"
            "exec(_HMM, _m.__dict__)\n_h = _m.__dict__\n"
            f"_MAP = {mapping!r}\n_MEMBERS = {members!r}\n{_MEMBER_LOADER}\n"
            "def positions(df):\n"
            "    out = pd.Series(0.0, index=df.index)\n"
            "    f = _h['features'](df['close'])\n"
            "    n = _h['MIN_TRAIN']\n"
            "    if len(f) <= n:\n"
            "        return out\n"
            "    fit = _h['fit'](f.values[:n])\n"
            "    state = np.array(_h['LABELS'])[_h['filtered_probs'](fit, f.values).argmax(axis=1)]\n"
            "    for label, mid in _MAP.items():\n"
            "        pos = pd.Series(_member(_MEMBERS[mid])(df), index=df.index).fillna(0)\n"
            "        days = f.index[(state == label) & (np.arange(len(f)) >= n)]\n"
            "        out.loc[days] = pos.loc[days].values\n"
            "    return out\n")


def build_ensemble(spec: Dict[str, Any], tested: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """Turn the coder's ensemble spec into a strategy row. Members must be journal ids already tested."""
    method = spec.get("method")
    if method == "vote":
        ids: List[int] = [int(i) for i in spec["members"]]
        if len(ids) < 3 or len(ids) % 2 == 0:
            raise ValueError("a vote needs an odd number of members, at least 3")
    elif method == "regime":
        mapping = {k: int(v) for k, v in spec["map"].items()}
        ids = sorted(set(mapping.values()))
    else:
        raise ValueError(f"unknown ensemble method {method!r}")
    unknown = [i for i in ids if i not in tested]
    if unknown:
        raise ValueError(f"ensemble members not tested yet: {unknown}")
    members = {i: tested[i]["code"] for i in ids}
    code = vote_source(members) if method == "vote" else regime_source(mapping, members)
    family = f"ensemble_{spec['method']}_" + "_".join(map(str, sorted(ids)))
    hyp = f"{spec['method']} ensemble of " + ", ".join(f"#{i} ({tested[i]['family']})" for i in ids)
    return {"family": family, "kind": "ensemble", "parents": ids, "code": code, "hypothesis": hyp,
            "rule": str(spec), "idea": hyp, "source": "journal " + ", ".join(f"#{i}" for i in ids)}
