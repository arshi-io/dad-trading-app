"""Signal dataclass — the universal contract every strategy emits."""
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Signal:
    asset: str
    direction: str        # "BUY", "SELL", "HOLD"
    confidence: float     # 0.0 to 1.0
    strategy: str
    timeframe: str
    timestamp: datetime
    metadata: dict = field(default_factory=dict)
