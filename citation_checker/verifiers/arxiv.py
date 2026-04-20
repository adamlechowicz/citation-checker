"""arXiv API verifier — looks up papers by eprint ID via the Atom XML feed."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Optional

from ..http_client import CitationHttpClient, CitationHttpError
from ..models import RemoteRecord

log = logging.getLogger(__name__)

_BASE = "http://export.arxiv.org/api/query"
_NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}


async def lookup_by_eprint(
    eprint: str, client: CitationHttpClient
) -> Optional[RemoteRecord]:
    """Look up an arXiv paper by eprint ID. Returns None if not found."""
    try:
        xml_text = await client.get_xml(_BASE, params={"id_list": eprint, "max_results": 1})
    except CitationHttpError as exc:
        log.debug("arXiv lookup failed for %s: %s", eprint, exc)
        return None

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        log.debug("arXiv XML parse error for %s: %s", eprint, exc)
        return None

    # Check total results
    total_el = root.find("opensearch:totalResults", _NS)
    if total_el is not None and total_el.text and int(total_el.text) == 0:
        log.debug("arXiv: no results for eprint %s", eprint)
        return None

    entry = root.find("atom:entry", _NS)
    if entry is None:
        log.debug("arXiv: no <entry> element for eprint %s", eprint)
        return None

    return _parse_entry(entry)


def _parse_entry(entry: ET.Element) -> RemoteRecord:
    title_el = entry.find("atom:title", _NS)
    title = _clean(title_el.text) if title_el is not None and title_el.text else None

    authors: list[str] = []
    for author_el in entry.findall("atom:author", _NS):
        name_el = author_el.find("atom:name", _NS)
        if name_el is not None and name_el.text:
            authors.append(_clean(name_el.text))

    year: Optional[int] = None
    published_el = entry.find("atom:published", _NS)
    if published_el is not None and published_el.text:
        try:
            year = int(published_el.text[:4])
        except (ValueError, IndexError):
            pass

    return RemoteRecord(
        title=title,
        authors=authors,
        year=year,
        source="arxiv",
        raw_response={"xml_entry": ET.tostring(entry, encoding="unicode")},
    )


def _clean(text: str) -> str:
    """Strip leading/trailing whitespace and collapse internal whitespace."""
    import re
    return re.sub(r'\s+', ' ', text).strip()
