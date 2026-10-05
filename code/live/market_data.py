"""Live-quote providers behind one interface. The broker provider plugs in here once a broker
is chosen; until then the simulated provider makes every market-hours feature testable."""
from __future__ import annotations

import math
import zlib
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, Iterable, Optional, Protocol
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")


@dataclass(frozen=True)
class Quote:
    symbol: str
    ltp: float
    prev_close: float
    day_open: float
    day_high: float
    day_low: float
    volume: int
    ts: datetime
    source: str
    simulated: bool

    @property
    def chg_pct(self) -> float:
        return (self.ltp / self.prev_close - 1) * 100 if self.prev_close else 0.0

    def age_seconds(self, now: Optional[datetime] = None) -> float:
        return ((now or datetime.now(IST)) - self.ts).total_seconds()

    def as_dict(self) -> dict:
        d = asdict(self)
        d["ts"] = self.ts.isoformat()
        d["chg_pct"] = round(self.chg_pct, 2)
        return d


class MarketDataProvider(Protocol):
    name: str

    def quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        """Latest quote per symbol; symbols the provider can't price are simply absent."""
        ...


class SimulatedProvider:
    """SIMULATED prices for demos and tests -- never real market data, and labelled as such.

    Each symbol drifts around its last real close with a deterministic intraday path (seeded by
    symbol and date), so a page refresh shows the same price for the same minute.
    """

    name = "simulated"

    def __init__(self, prev_closes: Dict[str, float], daily_vol: float = 0.018):
        self.prev_closes = prev_closes
        self.daily_vol = daily_vol

    def quotes(self, symbols: Iterable[str], now: Optional[datetime] = None) -> Dict[str, Quote]:
        now = (now or datetime.now(IST)).astimezone(IST)
        minutes = max(0, min(375, (now.hour * 60 + now.minute) - (9 * 60 + 15)))  # 09:15 → 15:30
        out: Dict[str, Quote] = {}
        for sym in symbols:
            prev = self.prev_closes.get(sym)
            if not prev:
                continue
            seed = zlib.crc32(f"{sym}|{now.date().isoformat()}".encode())
            path = [self._move(seed, m) for m in range(minutes + 1)]
            level = [prev * math.exp(self.daily_vol * p) for p in path]
            out[sym] = Quote(
                symbol=sym, ltp=round(level[-1], 2), prev_close=prev, day_open=round(level[0], 2),
                day_high=round(max(level), 2), day_low=round(min(level), 2),
                volume=int(1000 + (seed % 9000) * (minutes + 1)), ts=now, source=self.name, simulated=True,
            )
        return out

    @staticmethod
    def _move(seed: int, minute: int) -> float:
        # Smooth pseudo-random walk in "daily sigma" units: a few overlapping sine waves per symbol.
        a, b, c = (seed % 97) / 97, (seed % 89) / 89, (seed % 83) / 83
        t = minute / 375
        return 0.6 * math.sin(6.28 * (t * (1 + a) + b)) * t + 0.3 * math.sin(6.28 * (t * 4 + c)) * 0.5 + (a - 0.5) * t


class MotilalProvider:
    """Live last-traded prices from Motilal Oswal (requires a connected session)."""

    name = "motilal"

    def __init__(self, client, max_workers: int = 4):
        self.client = client
        self.max_workers = max_workers
        self._codes: Dict[str, int] = {}

    def quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        from concurrent.futures import ThreadPoolExecutor
        if not self.client.connected:
            return {}
        if not self._codes:
            self._codes = self.client.nse_equity_codes()
        wanted = [(s, self._codes.get(s.replace(".NS", "").upper())) for s in symbols]
        wanted = [(s, c) for s, c in wanted if c]
        now = datetime.now(IST)

        def one(item):
            sym, code = item
            try:
                d = self.client.ltp(code)
            except Exception:
                return None
            if d["ltp"] <= 0:
                return None
            return Quote(sym, d["ltp"], d["prev_close"], d["open"], d["high"], d["low"], d["volume"], now, self.name, False)

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return {q.symbol: q for q in pool.map(one, wanted) if q}


class YFinanceDelayedProvider:
    """Free fallback: Yahoo's intraday bars. Delayed (often ~15 min) -- flagged by its timestamp,
    which is the bar time, so the staleness check treats it honestly."""

    name = "yfinance_delayed"

    def quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        import yfinance as yf
        symbols = list(symbols)
        if not symbols:
            return {}
        raw = yf.download(symbols, period="2d", interval="1m", progress=False, auto_adjust=False, group_by="ticker", threads=True)
        out: Dict[str, Quote] = {}
        for sym in symbols:
            try:
                df = (raw[sym] if getattr(raw.columns, "nlevels", 1) > 1 else raw).dropna(subset=["Close"])
            except KeyError:
                continue
            if df.empty:
                continue
            last_day = df.index[-1].date()
            today = df[df.index.date == last_day]
            prev = df[df.index.date < last_day]
            ts = df.index[-1].to_pydatetime()
            ts = ts.astimezone(IST) if ts.tzinfo else ts.replace(tzinfo=IST)
            out[sym] = Quote(sym, round(float(today["Close"].iloc[-1]), 2),
                             round(float(prev["Close"].iloc[-1]), 2) if not prev.empty else round(float(today["Open"].iloc[0]), 2),
                             round(float(today["Open"].iloc[0]), 2), round(float(today["High"].max()), 2),
                             round(float(today["Low"].min()), 2), int(today["Volume"].sum()), ts, self.name, False)
        return out


class ChoiceProvider:
    """Live NSE quotes from Choice FinX's touchline (requires a connected session)."""

    name = "choice"

    def __init__(self, client):
        self.client = client

    def quotes(self, symbols: Iterable[str]) -> Dict[str, Quote]:
        if not self.client.connected:
            return {}
        now = datetime.now(IST)
        out: Dict[str, Quote] = {}
        for sym, d in self.client.touchline(symbols).items():
            out[sym] = Quote(sym, round(d["ltp"], 2), round(d.get("prev_close", 0.0), 2), round(d.get("open", 0.0), 2),
                             round(d.get("high", 0.0), 2), round(d.get("low", 0.0), 2), int(d.get("volume", 0)),
                             now, self.name, False)
        return out
