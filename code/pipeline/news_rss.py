from __future__ import annotations

import html
import re
import logging
import threading
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# All five verified 2026-07: 200 + parseable <item>s, checked 3x each for
# stability. The old market.xml / marketsrssfeed.xml URLs (503/404) are gone
# for good -- replaced outright, not kept as dead entries.
SOURCES = [
    ("Moneycontrol", "https://www.moneycontrol.com/rss/marketreports.xml"),
    ("Moneycontrol", "https://www.moneycontrol.com/rss/latestnews.xml"),
    ("ET Markets", "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms"),
    ("Business Standard", "https://www.business-standard.com/rss/markets-106.rss"),
    ("Livemint", "https://www.livemint.com/rss/markets"),
]

# A plain browser User-Agent is enough; an explicit Accept header appeared to
# trigger WAF blocks on some of these (403s that vanished once it was
# dropped) -- kept minimal deliberately.
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
}

# Per-feed timeout. Deliberately short: five feeds are fetched in sequence, so
# a 10s timeout meant one hung publisher could stall a page load for 10s+ (and
# all five for ~50s). Headlines are supporting context on an EOD dashboard --
# dropping a slow feed costs far less than making the user wait for it.
_FEED_TIMEOUT_SECONDS = 4

# These are market-wide headlines, identical for every ticker and every
# viewer, so re-fetching them per page view was pure waste: measured at
# 4.3-29.5s per Stock Room load, 85-95% of the whole request. One shared
# TTL cache turns that into a single fetch per window.
_CACHE_TTL = timedelta(minutes=15)
_cache: dict[int, Tuple[datetime, List[Dict[str, Any]]]] = {}
_cache_lock = threading.Lock()


def _cached(limit: int) -> Optional[List[Dict[str, Any]]]:
    with _cache_lock:
        entry = _cache.get(limit)
        if entry and datetime.now() - entry[0] < _CACHE_TTL:
            return entry[1]
    return None


def _store(limit: int, items: List[Dict[str, Any]]) -> None:
    with _cache_lock:
        _cache[limit] = (datetime.now(), items)


def clear_news_cache() -> None:
    """Drop the cached feed pull (tests, and the nightly job which should
    always see fresh headlines rather than a 15-minute-old slice)."""
    with _cache_lock:
        _cache.clear()


def _parse_pubdate(raw: str) -> str:
    """Best-effort RFC-822 pubDate -> ISO string. Falls back to the raw text."""
    if not raw:
        return ""
    try:
        return parsedate_to_datetime(raw).isoformat()
    except Exception:
        return raw


def fetch_news_items(limit: int = 6) -> List[Dict[str, Any]]:
    """Fetch a small, deterministic set of market-news items from RSS sources.

    Pulls from every source in SOURCES independently -- one dead feed never
    blocks the others, since each fetch is wrapped in its own try/except.
    Items are deduped by normalised title (some stories get syndicated
    across publishers) before the per-source slice count is applied.
    """
    hit = _cached(limit)
    if hit is not None:
        logger.debug("fetch_news_items(limit=%d) served from cache (%d items)", limit, len(hit))
        return hit

    items: List[Dict[str, Any]] = []
    seen_titles = set()
    per_source = max(1, limit // len(SOURCES))

    for source_name, url in SOURCES:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=_FEED_TIMEOUT_SECONDS)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.content, "xml")
            taken = 0
            for entry in soup.find_all("item"):
                if taken >= per_source:
                    break
                # Some publishers double-encode entities in their XML (e.g. a
                # literal "&amp;amp;" in the feed), which the XML parser only
                # unwinds one level, leaving a visible "&amp;" in the title --
                # html.unescape() cleans up whatever's left.
                title = html.unescape((entry.title.get_text(strip=True) if entry.title else "").strip())
                if not title:
                    continue
                key = title.lower()
                if key in seen_titles:
                    continue
                seen_titles.add(key)
                link = (entry.link.get_text(strip=True) if entry.link else "").strip()
                pubdate_tag = entry.find("pubDate")
                ts = _parse_pubdate(pubdate_tag.get_text(strip=True) if pubdate_tag else "")
                items.append({"source": source_name, "title": title, "link": link or url, "ts": ts})
                taken += 1
        except Exception as exc:
            logger.warning("news fetch failed for %s: %s", source_name, exc)

    if not items:
        items.append({"source": "System", "title": "Markets data refreshed; no headline feed available.", "link": "", "ts": ""})
    result = items[:limit]
    _store(limit, result)
    return result


_COMPANY_TTL = timedelta(hours=2)
_company_cache: dict[str, Tuple[datetime, List[Dict[str, Any]]]] = {}
BING_NEWS_RSS = "https://www.bing.com/news/search"
COMPANY_NEWS_MAX_AGE = timedelta(days=30)
_NAME_NOISE = {"ltd", "ltd.", "limited", "the", "india", "of", "and", "co", "co.", "company", "corporation"}

# Where publishers put the real publish time. Feed dates are not trusted on their own:
# aggregators re-date old stories when they re-index them, so a March article can arrive
# stamped "October".
_DATE_META = (
    ("meta", {"property": "article:published_time"}),
    ("meta", {"name": "article:published_time"}),
    ("meta", {"itemprop": "datePublished"}),
    ("meta", {"name": "publish-date"}),
    ("meta", {"name": "pubdate"}),
    ("meta", {"name": "Last-Modified"}),
    ("meta", {"property": "og:article:published_time"}),
)


def _real_url(link: str) -> str:
    """Bing wraps every result in a click-tracking redirect; the publisher URL is its url= parameter."""
    from urllib.parse import parse_qs, urlsplit
    qs = parse_qs(urlsplit(link).query)
    return qs.get("url", [link])[0]


def _parse_when(raw: str) -> Optional[datetime]:
    try:
        when = pd.Timestamp(raw)
    except (ValueError, TypeError):
        return None
    if pd.isna(when):
        return None
    if when.tzinfo is None:
        when = when.tz_localize("Asia/Kolkata")
    return when.to_pydatetime()


def _published_at(url: str) -> Optional[datetime]:
    """The publisher's own publish timestamp, from page metadata or JSON-LD. None if the page doesn't say."""
    import json as _json
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_FEED_TIMEOUT_SECONDS)
        resp.raise_for_status()
    except Exception:
        return None
    soup = BeautifulSoup(resp.content[:600_000], "lxml")
    for tag, attrs in _DATE_META:
        el = soup.find(tag, attrs=attrs)
        if el and el.get("content") and (when := _parse_when(el["content"])):
            return when
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = _json.loads(script.string or "")
        except ValueError:
            continue
        stack = data if isinstance(data, list) else [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                if node.get("datePublished") and (when := _parse_when(str(node["datePublished"]))):
                    return when
                stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
            elif isinstance(node, list):
                stack.extend(node)
    el = soup.find("time", attrs={"datetime": True})
    return _parse_when(el["datetime"]) if el else None


# Quote pages and live blogs re-date themselves daily; they aren't news.
_EVERGREEN = re.compile(r"(share|stock) price( today| live| highlights)?$|price history|/topic/|live updates", re.I)


def fetch_company_news(company_name: str, limit: int = 5, ticker: Optional[str] = None) -> List[Dict[str, Any]]:
    """Last month's headlines that name this company, newest first. Dated by the publisher's
    own page; Bing's date only when the page doesn't say (Bing's dates match publishers', unlike
    Google News, which re-dates old stories when it re-indexes them)."""
    from concurrent.futures import ThreadPoolExecutor

    key = company_name.lower()
    with _cache_lock:
        hit = _company_cache.get(key)
        if hit and datetime.now() - hit[0] < _COMPANY_TTL:
            return hit[1][:limit]
    query = company_name.replace(" Ltd.", "").replace(" Limited", "").replace(" Ltd", "").strip()
    words = [w for w in query.lower().split() if w not in _NAME_NOISE]
    must = words[0] if words else query.lower()
    # "Titan Securities" isn't Titan Company: with a two-word name, the second word or the ticker must appear too.
    also = {words[1]} if len(words) > 1 else set()
    if ticker:
        also.add(ticker.replace(".NS", "").lower())
    entries = []
    for q in (f"{query} share", query):  # the bare name catches results/deal news without "share" in it
        try:
            resp = requests.get(BING_NEWS_RSS, params={"q": q, "format": "rss"}, headers=_HEADERS, timeout=_FEED_TIMEOUT_SECONDS)
            resp.raise_for_status()
            entries += BeautifulSoup(resp.content, "xml").find_all("item")
        except Exception as exc:
            logger.warning("company news fetch failed for %s (%s): %s", company_name, q, exc)

    candidates, seen = [], set()
    for entry in entries:
        title = html.unescape(entry.title.get_text(strip=True)) if entry.title else ""
        desc = html.unescape(entry.description.get_text(strip=True)) if entry.description else ""
        text = (title + " " + desc).lower()
        if not title or title.lower() in seen or must not in text or (also and not any(a in text for a in also)):
            continue
        seen.add(title.lower())
        url = _real_url(entry.link.get_text(strip=True)) if entry.link else ""
        if (not url.startswith(("https://", "http://")) or _EVERGREEN.search(title) or _EVERGREEN.search(url)
                or len(title.split()) < 4):  # a bare company name is a quote page, not a headline
            continue
        pub = entry.find("pubDate")
        feed_when = _parse_when(pub.get_text(strip=True)) if pub else None
        candidates.append({"title": title, "link": url, "source": urlsplit_host(url), "feed_when": feed_when})
        if len(candidates) >= 14:
            break

    with ThreadPoolExecutor(max_workers=8) as pool:
        dates = list(pool.map(lambda c: _published_at(c["link"]), candidates))
    now = datetime.now().astimezone()
    items = []
    for c, when in zip(candidates, dates):
        when = when or c.pop("feed_when")
        c.pop("feed_when", None)
        if when is None or not (timedelta(hours=-6) <= now - when <= COMPANY_NEWS_MAX_AGE):
            continue
        items.append({**c, "ts": when.isoformat()})
    items.sort(key=lambda i: i["ts"], reverse=True)
    with _cache_lock:
        _company_cache[key] = (datetime.now(), items)
    return items[:limit]


def urlsplit_host(url: str) -> str:
    from urllib.parse import urlsplit
    host = urlsplit(url).netloc.lower()
    return host[4:] if host.startswith("www.") else host
