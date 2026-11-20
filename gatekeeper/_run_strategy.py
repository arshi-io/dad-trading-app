"""Runs a submitted strategy in its own isolated process (python -I). Reads strategy.py,
data.parquet and cuts.json from the folder given; writes out.json with the full positions and,
for each cut date, the position computed from data up to and including that date only."""
import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

folder = Path(sys.argv[1])
spec = importlib.util.spec_from_file_location("strategy", folder / "strategy.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

df = pd.read_parquet(folder / "data.parquet")


def run(d: pd.DataFrame) -> pd.Series:
    return pd.Series(mod.positions(d.copy()), index=d.index, dtype=float)


full = run(df)
cut = {}
for c in json.loads((folder / "cuts.json").read_text()):
    t = pd.Timestamp(c)
    cut[c] = float(run(df[df.index <= t]).iloc[-1])
(folder / "out.json").write_text(json.dumps({"full": full.tolist(), "cut": cut}))
