"""Survivorship-free NSE equity history from the exchange's own daily files.

Every trading day since 2011: OHLC, previous close, volume, turnover (bhavcopy) and delivery
quantity (MTO). Includes stocks that were later delisted, so backtests don't only see survivors.
Daily returns use NSE's PREVCLOSE, which the exchange adjusts on corporate-action ex-dates, so a
return series chained from it is split/bonus-adjusted without a separate corporate-actions table.

Resumable: each day is stored once under code/data/history/days/; re-running fetches only gaps.
Run:  python -m code.research.bhavcopy            (backfill from 2011)
"""
from __future__ import annotations

import io
import logging
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1] / "data" / "history"
DAYS = ROOT / "days"
NEW_FORMAT_FROM = date(2024, 7, 8)   # NSE switched cash bhavcopy to the UDiFF layout
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
           "Referer": "https://www.nseindia.com/"}
SERIES_KEEP = {"EQ", "BE", "BZ", "SM", "ST"}


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get("https://www.nseindia.com/", timeout=15)
    except requests.RequestException:
        pass
    return s


def _old_url(d: date) -> str:
    mon = d.strftime("%b").upper()
    return f"https://nsearchives.nseindia.com/content/historical/EQUITIES/{d.year}/{mon}/cm{d.day:02d}{mon}{d.year}bhav.csv.zip"


def _new_url(d: date) -> str:
    return f"https://nsearchives.nseindia.com/content/cm/BhavCopy_NSE_CM_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"


def _mto_url(d: date) -> str:
    return f"https://nsearchives.nseindia.com/archives/equities/mto/MTO_{d:%d%m%Y}.DAT"


def _read_zip_csv(content: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        return pd.read_csv(z.open(z.namelist()[0]))


def parse_old(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().upper() for c in df.columns]
    return pd.DataFrame({
        "symbol": df["SYMBOL"].str.strip(), "series": df["SERIES"].str.strip(),
        "open": df["OPEN"], "high": df["HIGH"], "low": df["LOW"], "close": df["CLOSE"],
        "prev_close": df["PREVCLOSE"], "volume": df["TOTTRDQTY"], "turnover": df["TOTTRDVAL"],
    })


def parse_new(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["Sgmt"].astype(str).str.strip() == "CM"]
    return pd.DataFrame({
        "symbol": df["TckrSymb"].astype(str).str.strip(), "series": df["SctySrs"].astype(str).str.strip(),
        "open": df["OpnPric"], "high": df["HghPric"], "low": df["LwPric"], "close": df["ClsPric"],
        "prev_close": df["PrvsClsgPric"], "volume": df["TtlTradgVol"], "turnover": df["TtlTrfVal"],
    })


def parse_mto(text: str) -> pd.DataFrame:
    """Delivery file: data rows start with record type 20: 20,sr,SYMBOL,SERIES,traded,deliverable,%."""
    rows = []
    for line in text.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 6 and parts[0] == "20":
            try:
                rows.append((parts[2], parts[3], float(parts[5])))
            except ValueError:
                continue
    return pd.DataFrame(rows, columns=["symbol", "series", "delivery_qty"])


def fetch_day(d: date, s: requests.Session) -> Optional[pd.DataFrame]:
    """One trading day, or None if NSE has no file (weekend/holiday)."""
    out = DAYS / f"{d:%Y%m%d}.parquet"
    if out.exists():
        return None
    urls = [_new_url(d), _old_url(d)] if d >= NEW_FORMAT_FROM else [_old_url(d), _new_url(d)]
    bhav = None
    for url in urls:
        r = s.get(url, timeout=30)
        if r.status_code == 200 and r.content[:2] == b"PK":
            raw = _read_zip_csv(r.content)
            bhav = parse_new(raw) if "TckrSymb" in raw.columns else parse_old(raw)
            break
    if bhav is None:
        return None
    bhav = bhav[bhav["series"].isin(SERIES_KEEP)].copy()
    try:
        m = s.get(_mto_url(d), timeout=30)
        if m.status_code == 200:
            bhav = bhav.merge(parse_mto(m.text), on=["symbol", "series"], how="left")
    except requests.RequestException:
        pass
    if "delivery_qty" not in bhav:
        bhav["delivery_qty"] = float("nan")
    bhav.insert(0, "date", pd.Timestamp(d))
    DAYS.mkdir(parents=True, exist_ok=True)
    bhav.to_parquet(out, index=False)
    return bhav


def backfill(start: date = date(2011, 1, 3), end: Optional[date] = None, workers: int = 4) -> dict:
    end = end or date.today()
    days = [start + timedelta(n) for n in range((end - start).days + 1)]
    days = [d for d in days if d.weekday() < 5 and not (DAYS / f"{d:%Y%m%d}.parquet").exists()]
    t0, done, empty = time.time(), 0, 0
    sessions = [_session() for _ in range(workers)]

    def work(i_d):
        i, d = i_d
        for attempt in range(3):
            try:
                return fetch_day(d, sessions[i % workers])
            except Exception as exc:
                logger.warning("%s attempt %d failed: %s", d, attempt + 1, exc)
                time.sleep(2 * (attempt + 1))
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for n, res in enumerate(pool.map(work, enumerate(days)), 1):
            done += res is not None
            empty += res is None
            if n % 100 == 0:
                print(f"{n}/{len(days)} days checked, {done} saved, {time.time() - t0:.0f}s", flush=True)
    return {"checked": len(days), "saved": done, "no_file": empty, "seconds": round(time.time() - t0)}


def load(symbols: Optional[list] = None, since: Optional[str] = None) -> pd.DataFrame:
    files = sorted(DAYS.glob("*.parquet"))
    if since:
        files = [f for f in files if f.stem >= since.replace("-", "")]
    df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    return df[df["symbol"].isin(symbols)] if symbols else df


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    start = date.fromisoformat(sys.argv[1]) if len(sys.argv) > 1 else date(2011, 1, 3)
    print(backfill(start))
