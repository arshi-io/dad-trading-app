"""Market-hours quote loop: polls the configured provider for the symbols that matter
(watchlist, tonight's buy-ready setups, open positions and waiting orders) and keeps the latest
validated quote per symbol in memory. Nothing here trades."""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional

from code.app import settings
from code.live.market_data import IST, MarketDataProvider, Quote
from code.live.session import MarketSession, market_session

logger = logging.getLogger(__name__)

MAX_SYMBOLS = 60


def valid(q: Quote) -> bool:
    """Reject obviously broken prints before anything can act on them."""
    if q.ltp <= 0 or q.prev_close <= 0:
        return False
    if q.day_high and q.day_low and not (q.day_low * 0.995 <= q.ltp <= q.day_high * 1.005):
        return False
    return abs(q.ltp / q.prev_close - 1) < 0.25   # beyond NSE's widest price band = bad data


class LiveFeed:
    def __init__(self, provider_fn: Callable[[], MarketDataProvider], symbols_fn: Callable[[], Iterable[str]],
                 session: Callable[[], MarketSession] = market_session, stale_seconds: int = settings.QUOTE_STALE_SECONDS):
        self.provider_fn = provider_fn
        self.symbols_fn = symbols_fn
        self.session = session
        self.stale_seconds = stale_seconds
        self.quotes: Dict[str, Quote] = {}
        self.rejected: Dict[str, str] = {}
        self.last_poll: Optional[datetime] = None
        self.last_error = ""
        self.provider_name = ""

    def poll_once(self, now: Optional[datetime] = None) -> int:
        provider = self.provider_fn()
        self.provider_name = provider.name
        symbols: List[str] = list(dict.fromkeys(self.symbols_fn()))[:MAX_SYMBOLS]
        try:
            got = provider.quotes(symbols)
            self.last_error = ""
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"[:200]
            logger.warning("live poll failed (%s): %s", provider.name, exc)
            return 0
        for sym, q in got.items():
            if valid(q):
                self.quotes[sym] = q
                self.rejected.pop(sym, None)
            else:
                self.rejected[sym] = f"invalid print ₹{q.ltp}"
        self.last_poll = now or datetime.now(IST)
        return len(got)

    def is_stale(self, q: Quote, now: Optional[datetime] = None) -> bool:
        return q.age_seconds(now) > self.stale_seconds

    def view(self, now: Optional[datetime] = None) -> Dict[str, dict]:
        return {s: {**q.as_dict(), "stale": self.is_stale(q, now)} for s, q in self.quotes.items()}

    async def run(self) -> None:
        """Poll while the market is open (and once just after the close for the final print)."""
        polled_close_for = None
        while True:
            try:
                st = self.session().state()
                if st.phase == "OPEN":
                    await asyncio.to_thread(self.poll_once)
                elif st.phase == "CLOSED" and st.now.hour >= 15 and polled_close_for != st.now.date():
                    await asyncio.to_thread(self.poll_once)
                    polled_close_for = st.now.date()
            except Exception:
                logger.exception("live feed loop error")
            await asyncio.sleep(settings.LIVE_POLL_SECONDS)
