"""CrossRef API verifier — DOI lookup and bibliographic title/author search."""

from __future__ import annotations

import logging
from typing import Optional
from urllib.parse import quote

from ..http_client import CitationHttpClient, CitationHttpError
from ..models import RemoteRecord

log = logging.getLogger(__name__)

_BASE = "https://api.crossref.org"


async def lookup_by_doi(doi: str, client: CitationHttpClient) -> Optional[RemoteRecord]:
    """Look up a work by DOI. Returns None on 404, raises CitationHttpError otherwise."""
    url = f"{_BASE}/works/{quote(doi, safe='/')}"
    try:
        data = await client.get_json(url)
    except CitationHttpError as exc:
        if exc.status_code == 404:
            log.debug("DOI not found in CrossRef: %s", doi)
            return None
        raise
    return _parse_message(data.get("message", {}), source="crossref")


async def search_by_title_author(
    title: str,
    authors: list[str],
    client: CitationHttpClient,
    rows: int = 5,
) -> Optional[RemoteRecord]:
    """Search CrossRef by bibliographic fields. Returns the best-matching record or None."""
    from ..fuzzy import normalize_string, TITLE_THRESHOLD

    params: dict = {
        "query.bibliographic": title,
        "rows": rows,
        "select": "DOI,title,author,published,published-print,published-online",
    }
    if authors:
        params["query.author"] = authors[0]

    url = f"{_BASE}/works"
    try:
        data = await client.get_json(url, params=params)
    except CitationHttpError as exc:
        log.debug("CrossRef search failed: %s", exc)
        return None

    items = data.get("message", {}).get("items", [])
    if not items:
        return None

    # Pick the result whose title best matches ours
    best_record: Optional[RemoteRecord] = None
    best_score: float = -1.0
    norm_local = normalize_string(title)

    for item in items:
        record = _parse_message(item, source="crossref")
        if record.title is None:
            continue
        score = __import__('rapidfuzz').fuzz.ratio(
            norm_local, normalize_string(record.title)
        )
        if score > best_score:
            best_score = score
            best_record = record

    if best_score >= TITLE_THRESHOLD and best_record is not None:
        log.debug("CrossRef search matched (score=%.1f): %s", best_score, best_record.title)
        return best_record

    log.debug("CrossRef search: no match above threshold (best=%.1f) for: %s", best_score, title)
    return None


def _parse_message(msg: dict, source: str) -> RemoteRecord:
    """Parse a CrossRef 'message' or 'item' dict into a RemoteRecord."""
    # Title is a list in CrossRef responses
    raw_title = msg.get("title") or msg.get("short-title")
    if isinstance(raw_title, list):
        title = raw_title[0] if raw_title else None
    else:
        title = raw_title or None

    # Authors
    author_list = msg.get("author", [])
    authors: list[str] = []
    for a in author_list:
        given = a.get("given", "")
        family = a.get("family", "")
        name = f"{given} {family}".strip() if given else family
        if name:
            authors.append(name)

    # Year — prefer published-print, then published-online, then published
    year: Optional[int] = None
    for date_key in ("published-print", "published-online", "published"):
        date_parts = msg.get(date_key, {}).get("date-parts", [[]])[0]
        if date_parts:
            try:
                year = int(date_parts[0])
                break
            except (ValueError, TypeError):
                continue

    return RemoteRecord(
        title=title,
        authors=authors,
        year=year,
        source=source,
        raw_response=msg,
    )


def _normalize_string(text: str) -> str:
    from ..fuzzy import normalize_string
    return normalize_string(text)
