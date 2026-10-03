"""Covers the caching + labelling work: the things that made pages slow, and
the internal identifiers that were leaking into the UI."""
from unittest.mock import patch

import pandas as pd
import pytest

from code.app import main
from code.pipeline import news_rss


@pytest.fixture(autouse=True)
def _clear_news_cache():
    news_rss.clear_news_cache()
    yield
    news_rss.clear_news_cache()


def _fake_feed_response():
    class _Resp:
        content = b"""<?xml version="1.0"?><rss><channel>
            <item><title>Market moves</title><link>http://x/1</link><pubDate>Tue, 02 Sep 2026 10:00:00 +0530</pubDate></item>
        </channel></rss>"""

        def raise_for_status(self):
            return None

    return _Resp()


def test_news_is_fetched_once_then_served_from_cache():
    """The RSS pull was 85-95% of a Stock Room request because it ran per
    view -- a second call in the same window must not touch the network."""
    with patch("code.pipeline.news_rss.requests.get", return_value=_fake_feed_response()) as mock_get:
        first = news_rss.fetch_news_items(limit=20)
        calls_after_first = mock_get.call_count
        second = news_rss.fetch_news_items(limit=20)

    assert calls_after_first > 0          # first call really fetched
    assert mock_get.call_count == calls_after_first  # second call fetched nothing
    assert second == first


def test_clear_news_cache_forces_a_refetch():
    with patch("code.pipeline.news_rss.requests.get", return_value=_fake_feed_response()) as mock_get:
        news_rss.fetch_news_items(limit=20)
        after_first = mock_get.call_count
        news_rss.clear_news_cache()
        news_rss.fetch_news_items(limit=20)
        assert mock_get.call_count > after_first


@pytest.mark.parametrize(
    "strategy,expected",
    [
        ("mean_reversion_zscore", "Mean reversion"),
        ("pairs_trading_zscore", "Pair spread"),
        ("garch_volatility", "Volatility"),
    ],
)
def test_strategy_ids_render_as_trader_language(strategy, expected):
    assert main._strategy_label(strategy) == expected


def test_unknown_strategy_id_is_still_never_shown_raw_with_underscores():
    assert main._strategy_label("some_new_engine") == "Some new engine"


def test_missing_strategy_reads_as_no_signal():
    assert main._strategy_label(None) == "No signal"


def _synthetic_ohlcv(bars: int) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=bars, freq="B", name="timestamp")
    close = pd.Series(range(1, bars + 1), index=idx, dtype=float)
    return pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99, "close": close, "volume": 1000.0},
        index=idx,
    )


def test_off_universe_ticker_gets_a_live_checklist():
    """KRBL/RISHABH aren't in the NIFTY 500 snapshot -- the 8-point template
    still has to be computed from their own bars, not skipped."""
    item = main._evaluate_screen_item_live("KRBL.NS", _synthetic_ohlcv(400), "CHOPPY")
    assert item is not None
    assert item["symbol"] == "KRBL.NS"
    assert len(item["checklist"]) == 8
    assert item["computed_live"] is True
    assert item["rs_rank"] is None  # can't rank against a universe this stock isn't in


def test_live_checklist_declines_on_thin_history_rather_than_faking_it():
    assert main._evaluate_screen_item_live("NEWLISTING.NS", _synthetic_ohlcv(40), "CHOPPY") is None


def test_live_checklist_handles_an_empty_frame():
    assert main._evaluate_screen_item_live("NOPE.NS", pd.DataFrame(), "CHOPPY") is None
