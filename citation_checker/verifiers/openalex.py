"""OpenAlex API verifier — title search fallback."""

from __future__ import annotations

import logging
from typing import Optional

from ..http_client import CitationHttpClient, CitationHttpError
from ..models import RemoteRecord

log = logging.getLogger(__name__)

_BASE = "https://api.openalex.org/works"


async def search_by_title(
    title: str,
    client: CitationHttpClient,
    api_key: Optional[str] = None,
) -> Optional[RemoteRecord]:
    """Search OpenAlex by title. Returns best-matching record or None."""
    from ..fuzzy import normalize_string, TITLE_THRESHOLD

    params: dict = {
        "search": title,
        "per-page": 5,
        "select": "title,authorships,publication_year",
    }
    if api_key:
        params["api_key"] = api_key

    try:
        data = await client.get_json(_BASE, params=params)
    except CitationHttpError as exc:
        log.debug("OpenAlex search failed: %s", exc)
        return None

    results = data.get("results", [])
    if not results:
        return None

    norm_local = normalize_string(title)
    best_record: Optional[RemoteRecord] = None
    best_score: float = -1.0

    for work in results:
        record = _parse_work(work)
        if record.title is None:
            continue
        score = __import__('rapidfuzz').fuzz.ratio(
            norm_local, normalize_string(record.title)
        )
        if score > best_score:
            best_score = score
            best_record = record

    if best_score >= TITLE_THRESHOLD and best_record is not None:
        log.debug("OpenAlex matched (score=%.1f): %s", best_score, best_record.title)
        return best_record

    log.debug("OpenAlex: no match above threshold (best=%.1f) for: %s", best_score, title)
    return None


def _parse_work(work: dict) -> RemoteRecord:
    title = work.get("title") or None

    authors: list[str] = []
    for authorship in work.get("authorships", []):
        author = authorship.get("author", {})
        name = author.get("display_name")
        if name:
            authors.append(name)

    year: Optional[int] = None
    raw_year = work.get("publication_year")
    if raw_year is not None:
        try:
            year = int(raw_year)
        except (ValueError, TypeError):
            pass

    return RemoteRecord(
        title=title,
        authors=authors,
        year=year,
        source="openalex",
        raw_response=work,
    )
