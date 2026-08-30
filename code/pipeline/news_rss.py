from __future__ import annotations

import html
import logging
from email.utils import parsedate_to_datetime
from typing import List, Dict, Any
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
    items: List[Dict[str, Any]] = []
    seen_titles = set()
    per_source = max(1, limit // len(SOURCES))

    for source_name, url in SOURCES:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=10)
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
    return items[:limit]
