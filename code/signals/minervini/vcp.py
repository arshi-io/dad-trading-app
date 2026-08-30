"""VCP (Volatility Contraction Pattern) detection — deferred to P6 per P2.md."""
from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd


def detect_vcp(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Stub — VCP Watch is a P6 stretch goal. Always returns no candidates."""
    return []
