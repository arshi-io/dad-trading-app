"""Golden-file test for pairs_trading.py (P3), pinned against the v1 backtest.

raw/data/backtest_pairs_trading_v1.csv is aggregate backtest output (hedge
ratio, cointegration p-value, half-life -- no per-date z-score series exists
in that file), so this validates find_cointegrated_pairs (unchanged from the
research repo) reproduces those aggregate values over the same date range
the CSV was generated from (2020-01-01..2024-12-31).

pvalue and half_life match to ~1e-5 (both are scale-invariant under a
per-leg price rescale). hedge_ratio does NOT: yfinance's auto_adjust=True
retroactively rescales an entire price history every time a new dividend
posts, so a hedge ratio computed today legitimately drifts a few percent
from one computed when the CSV was generated -- confirmed against each
ticker's dividend history (both ICICIBANK.NS and SBIN.NS posted additional
dividends after the backtest's 2024-12-31 end date). ±0.01 (the tolerance
used for pvalue/half_life) is too tight for a price-scale-dependent
quantity that drifts with every future dividend; hedge_ratio uses a 5%
relative tolerance instead.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from code.signals.pairs_trading import find_cointegrated_pairs

CSV_PATH = Path(__file__).resolve().parents[1] / "raw" / "data" / "backtest_pairs_trading_v1.csv"
BACKTEST_START = "2020-01-01"
BACKTEST_END = "2024-12-31"

ABS_TOL = 0.01


@pytest.fixture(scope="module")
def golden_rows():
    df = pd.read_csv(CSV_PATH)
    return df


@pytest.mark.parametrize("ticker_a,ticker_b", [
    ("ICICIBANK.NS", "SBIN.NS"),
    ("AXISBANK.NS", "MARUTI.NS"),
    ("RELIANCE.NS", "MARUTI.NS"),
])
def test_pair_matches_backtest_csv(golden_rows, ticker_a, ticker_b):
    row = golden_rows[(golden_rows["ticker_a"] == ticker_a) & (golden_rows["ticker_b"] == ticker_b)]
    assert not row.empty, f"{ticker_a}/{ticker_b} missing from {CSV_PATH.name}"
    row = row.iloc[0]

    scan = find_cointegrated_pairs(
        [ticker_a, ticker_b], start=BACKTEST_START, end=BACKTEST_END, pvalue_threshold=1.0,
    )
    assert scan, f"find_cointegrated_pairs returned nothing for {ticker_a}/{ticker_b}"
    result = scan[0]

    assert result["pvalue"] == pytest.approx(row["coint_pvalue"], abs=ABS_TOL), (
        f"coint p-value drifted beyond ±{ABS_TOL}: got {result['pvalue']}, "
        f"csv had {row['coint_pvalue']}"
    )
    assert result["half_life"] == pytest.approx(row["half_life_days"], abs=ABS_TOL), (
        f"half-life (days) drifted: got {result['half_life']}, csv had {row['half_life_days']}"
    )
    assert result["hedge_ratio"] == pytest.approx(row["hedge_ratio"], rel=0.05), (
        f"hedge ratio drifted beyond 5% (see module docstring re: dividend "
        f"auto-adjust drift): got {result['hedge_ratio']}, csv had {row['hedge_ratio']}"
    )
