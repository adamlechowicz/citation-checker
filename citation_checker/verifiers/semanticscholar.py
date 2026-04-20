"""Semantic Scholar paper search verifier.

Used as a final fallback for papers not found in CrossRef or OpenAlex.
Particularly effective for:
  - PMLR proceedings (AISTATS, ICML, COLT, UAI, etc.) which often lack DOIs
  - Recent ML/CS conference papers
  - Papers with non-standard metadata in CrossRef

API: https://api.semanticscholar.org/graph/v1/paper/search
Rate limit: ~100 req/5 min without an API key (enforced via http_client config).
"""

from __future__ import annotations

import logging
from typing import Optional

from ..http_client import CitationHttpClient, CitationHttpError
from ..models import RemoteRecord

log = logging.getLogger(__name__)

_BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
_FIELDS = "title,authors,year,venue"


async def search_by_title_author(
    title: str,
    authors: list[str],
    client: CitationHttpClient,
) -> Optional[RemoteRecord]:
    """Search Semantic Scholar by title (+ first author for disambiguation).

    Returns the top result as a RemoteRecord, or None if nothing was found.
    """
    # Build query: title + first author's last name to reduce false positives.
    query = title
    if authors:
        last_name = authors[0].split()[-1]
        query = f"{title} {last_name}"

    try:
        data = await client.get_json(
            _BASE_URL,
            params={"query": query, "fields": _FIELDS, "limit": 1},
        )
    except CitationHttpError as exc:
        log.debug("Semantic Scholar search error: %s", exc)
        return None

    papers = data.get("data", [])
    if not papers:
        return None

    p = papers[0]
    return RemoteRecord(
        title=p.get("title"),
        authors=[a["name"] for a in p.get("authors", [])],
        year=p.get("year"),
        source="semanticscholar",
        raw_response=p,
    )
