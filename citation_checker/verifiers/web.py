"""Web URL verifier — fetches a page and extracts the article/page title.

Used for news articles, industry reports, and other web resources where the
citation URL points directly to the referenced content. Title is extracted
from og:title, twitter:title, or the HTML <title> tag (in that priority
order) and compared against the local citation title via fuzzy matching.
"""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from typing import Optional

from ..http_client import CitationHttpClient, CitationHttpError
from ..models import RemoteRecord

log = logging.getLogger(__name__)

# Site-name suffixes that news outlets append to page titles, e.g.
# "AI Needs Power — Bloomberg" or "Some Article | The New York Times"
_SITE_SUFFIX_RE = re.compile(
    r'\s*[|\-–—]\s*(?:'
    r'Bloomberg|Financial Times|The New York Times|The NYT|'
    r'Reuters|The Wall Street Journal|WSJ|'
    r'The Guardian|BBC News|BBC|CNN|'
    r'The Atlantic|Vox|Axios|The Economist|NPR|'
    r'TechCrunch|Wired|Ars Technica|The Verge|'
    r'MIT Technology Review|Scientific American|'
    r'New Scientist|Science News|Utility Dive|'
    r'Fortune|Forbes|Business Insider|CNBC|'
    r'[A-Z][A-Za-z\s&]{2,30}'  # catch remaining "Site Name" patterns
    r')\s*$',
    re.IGNORECASE,
)


class _MetaExtractor(HTMLParser):
    """Extract og:title / twitter:title / <title> from an HTML document."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.og_title: Optional[str] = None
        self.page_title: Optional[str] = None
        self._in_title = False
        self._title_buf: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attrs_d = dict(attrs)
        if tag == "meta":
            prop = (attrs_d.get("property") or attrs_d.get("name") or "").lower()
            if prop in ("og:title", "twitter:title"):
                val = (attrs_d.get("content") or "").strip()
                if val and self.og_title is None:
                    self.og_title = val
        if tag == "title":
            self._in_title = True
            self._title_buf = []

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title_buf.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title" and self._in_title:
            self._in_title = False
            self.page_title = "".join(self._title_buf).strip()

    @property
    def best_title(self) -> Optional[str]:
        return self.og_title or self.page_title


def _clean_page_title(raw: str) -> str:
    """Strip trailing site-name suffix from a page title."""
    cleaned = _SITE_SUFFIX_RE.sub("", raw).strip()
    return cleaned or raw  # fall back to raw if stripping left nothing


_BOT_CHALLENGE_RE = re.compile(
    r'are you a robot|just a moment|access denied|'
    r'robot check|verify you are human|please verify|'
    r'attention required|security check|'
    r'please enable javascript|403 forbidden|'
    r'service unavailable|'
    r'checking your browser|cf-chl-bypass|enable cookies|'
    r'cloudflare|unusual traffic',
    re.IGNORECASE,
)


def _is_usable_title(title: str) -> bool:
    """Return False if the title looks like a bot-challenge or error page."""
    # Use search (not match) so we catch "Bloomberg - Are you a robot?" too.
    if _BOT_CHALLENGE_RE.search(title):
        return False
    stripped = title.strip()
    # Reject bare domain names (no spaces, contains a dot): "reuters.com", "ft.com"
    if '.' in stripped and ' ' not in stripped:
        return False
    # Titles under 10 characters are almost certainly not article titles
    if len(stripped) < 10:
        return False
    return True


async def lookup_by_url(url: str, client: CitationHttpClient) -> Optional[RemoteRecord]:
    """Fetch a URL and extract the page title as a RemoteRecord.

    Returns None if the page is unreachable, returns a non-2xx status, or
    no usable title can be extracted. The returned record has an empty
    author list and no year — the caller should skip author/year scoring
    and compare titles only.
    """
    try:
        response = await client.get_html(url)
    except CitationHttpError as exc:
        log.debug("Web fetch failed for %s: %s", url, exc)
        return None

    if response.status_code >= 400:
        log.debug("Web fetch returned %d for %s; refusing to scrape body",
                  response.status_code, url)
        return None

    extractor = _MetaExtractor()
    try:
        extractor.feed(response.text)
    except Exception as exc:
        log.debug("HTML parse error for %s: %s", url, exc)
        return None

    raw_title = extractor.best_title
    if not raw_title:
        log.debug("No title found at %s", url)
        return None

    title = _clean_page_title(raw_title)

    if not _is_usable_title(title):
        log.debug("Bot-challenge or unusable title at %s: %r", url, title)
        return None

    log.debug("Extracted title from %s: %r", url, title)

    return RemoteRecord(
        title=title,
        authors=[],   # web pages don't expose structured author metadata
        year=None,    # ditto for year
        source="web",
        raw_response={"url": url, "raw_title": raw_title},
    )
