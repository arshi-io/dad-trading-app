import pandas as pd

from code.pipeline import nightly_pipeline as n


def _bars(start, end):
    idx = pd.bdate_range(start, end)
    return pd.DataFrame({"Open": 1.0, "High": 1.0, "Low": 1.0, "Close": 1.0, "Volume": 1}, index=idx)


def test_historical_fetch_cannot_poison_the_live_cache(tmp_path, monkeypatch):
    """A 2021-2024 fetch (the golden test) plus bhavcopy's recent bars used to leave a file that
    looked like it covered today, with a two-year hole the live screen then read."""
    monkeypatch.setattr(n, "PRICES_DIR", tmp_path)
    monkeypatch.setattr(n, "_patch_from_bhavcopy",
                        lambda t, df: pd.concat([df, _bars("2026-11-16", "2026-11-19").rename(columns=str.lower)]))
    calls = []
    monkeypatch.setattr(n.yf, "download", lambda **kw: calls.append(kw) or _bars(kw["start"], pd.Timestamp(kw["end"]) - pd.Timedelta(days=1)))

    old = n.fetch_universe_closes(["X.NS"], start="2021-06-01", end="2024-10-01")["X.NS"]
    assert old.index.max() < pd.Timestamp("2024-10-02")          # nothing past the window

    n.fetch_universe_closes(["X.NS"], start="2023-11-19", end="2026-11-19")
    assert len(calls) == 2                                         # live window refetched, not served the old file

    gappy = pd.concat([_bars("2023-11-20", "2024-09-30"), _bars("2026-11-13", "2026-11-19")]).rename(columns=str.lower)
    gappy.index.name = "timestamp"
    gappy.to_parquet(n._universe_cache_path("X.NS"))
    n.fetch_universe_closes(["X.NS"], start="2023-11-19", end="2026-11-19")
    assert len(calls) == 3                                         # a file with a hole is never reused
