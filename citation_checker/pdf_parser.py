"""Parse bibliographies from PDF files into BibEntry objects.

Pipeline:
  extract text (pdfplumber) → locate references section →
  split into numbered blocks → parse each block with regex heuristics →
  return list[BibEntry]

Supports ACM-style numbered references ([1], [2], …) which is the most
common format in CS/engineering papers. Other formats (IEEE, APA) are
parsed best-effort.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import fitz  # pymupdf

from .models import BibEntry
from .utils import clean_doi, clean_arxiv_id

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level for performance)
# ---------------------------------------------------------------------------

# Matches "References", "REFERENCES", "Bibliography", "BIBLIOGRAPHY" as a
# standalone line (the section heading).
_REFS_HEADING_RE = re.compile(
    r'^\s*(?:REFERENCES|References|Bibliography|BIBLIOGRAPHY)\s*$',
    re.MULTILINE,
)

# Matches the start of a numbered reference block: [N] at line start.
_REF_START_RE = re.compile(r'^\[(\d+)\]\s+', re.MULTILINE)

# DOI: doi:10.xxx/yyy or https://doi.org/10.xxx/yyy
_DOI_RE = re.compile(
    r'(?:doi:\s*|https?://(?:dx\.)?doi\.org/)'
    r'(10\.\d{4,9}/[^\s,;\]]+)',
    re.IGNORECASE,
)

# arXiv: arXiv:NNNN.NNNNN or arXiv:category/NNNNNNN, optional version + category tag
_ARXIV_RE = re.compile(
    r'arXiv:([a-zA-Z\-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\s*\[[^\]]+\])?',
    re.IGNORECASE,
)

# Generic URL (matched after DOI/arXiv to avoid duplicates)
_URL_RE = re.compile(r'https?://[^\s,;\]]+', re.IGNORECASE)

# 4-digit year in the range 1000–2099
_YEAR_RE = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')

# ACM format anchor: "Authors. YEAR. Title. Venue."
# Non-greedy match on group 1 to catch the FIRST ". YEAR." occurrence.
_ACM_YEAR_SPLIT_RE = re.compile(
    r'^(.*?)\.\s+(1[0-9]{3}|20[0-9]{2})\.\s+(.*)',
    re.DOTALL,
)

# Where a title likely ends: a '. ' followed by a venue-like token.
_TITLE_END_RE = re.compile(
    r'\.\s+(?:In\b|Proc\.|Proceedings|IEEE|ACM|Journal|arXiv:|'
    r'https?://|doi:|Vol\.\s*\d+|\d{1,3},\s*\d+\s*\(|Springer|'
    r'Elsevier|PMLR|Advances in|[A-Z]{2,4}\s+\d{4})'
)

# Noise lines inserted by pdfplumber between reference text:
# bare page numbers, running heads (e.g. "Smith et al.", "ONLINE SMOOTHED…")
_NOISE_LINE_RE = re.compile(
    r'^\s*(?:\d{1,4}|[A-Z][a-z][\w\s,\-\.]+ et al\.|[A-Z][A-Z\s\-]{5,})\s*$'
)

# Heals URLs split across lines by PDF line-breaking: "https:\n//"
_SPLIT_URL_RE = re.compile(r'(https?):\s*\n\s*//')

# Split author lists on " and " (case-insensitive)
_AUTHOR_AND_RE = re.compile(r'\s+and\s+', re.IGNORECASE)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_pdf_file(path: str) -> list[BibEntry]:
    """Extract and parse the bibliography from a PDF file.

    Returns a list of BibEntry objects with cite keys ref1, ref2, …

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if no References/Bibliography section is found.
    """
    text = _extract_text(path)
    refs_text = _find_references_section(text)
    blocks = _split_into_blocks(refs_text)

    if not blocks:
        raise ValueError("Found a References section but could not split it into entries.")

    entries = []
    for num, raw in blocks:
        try:
            entry = _parse_block(num, raw)
            entries.append(entry)
        except Exception as exc:
            log.warning("Failed to parse reference [%d]: %s", num, exc)

    log.info("Parsed %d references from %s", len(entries), path)
    return entries


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text(path: str) -> str:
    """Extract full text from a PDF using pymupdf."""
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Section detection and block splitting
# ---------------------------------------------------------------------------

def _find_references_section(text: str) -> str:
    """Return the text from the References heading to end of document.

    Raises ValueError if no heading is found.
    """
    m = _REFS_HEADING_RE.search(text)
    if m is None:
        raise ValueError(
            "Could not find a References or Bibliography section in the PDF. "
            "The heading must appear as a standalone line."
        )
    return text[m.end():]


def _split_into_blocks(refs_text: str) -> list[tuple[int, str]]:
    """Split the references section into (number, raw_text) pairs."""
    matches = list(_REF_START_RE.finditer(refs_text))
    if not matches:
        return []

    blocks = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(refs_text)
        raw = refs_text[start:end]
        blocks.append((num, raw))
    return blocks


# ---------------------------------------------------------------------------
# Block-level parsing
# ---------------------------------------------------------------------------

def _parse_block(num: int, raw: str) -> BibEntry:
    """Parse a single reference block into a BibEntry."""
    text = _collapse_block(raw)

    doi = _extract_doi(text)
    eprint = _extract_arxiv(text)
    archiveprefix = "arXiv" if eprint else None
    url = _extract_url(text, doi, eprint)

    year: Optional[int] = None
    authors: list[str] = []
    title: Optional[str] = None

    m = _ACM_YEAR_SPLIT_RE.match(text)
    if m:
        author_str = m.group(1).strip()
        year = int(m.group(2))
        remainder = m.group(3).strip()
        authors = _parse_pdf_authors(author_str)
        title = _extract_title(remainder)
    else:
        # Fallback: best-effort year extraction
        ym = _YEAR_RE.search(text)
        if ym:
            year = int(ym.group(1))
        log.debug("Reference [%d] did not match ACM year pattern; partial parse only.", num)

    return BibEntry(
        key=f"ref{num}",
        entry_type="article",
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        url=url,
        eprint=eprint,
        archiveprefix=archiveprefix,
        raw_fields={"raw_text": text},
    )


def _collapse_block(raw: str) -> str:
    """Collapse a multi-line block into a single clean string.

    Drops noise lines (page numbers, running heads) and heals split URLs.
    """
    # Heal split URLs before splitting on newlines
    raw = _SPLIT_URL_RE.sub(r'\1://', raw)
    lines = raw.split('\n')
    lines = [l for l in lines if not _NOISE_LINE_RE.match(l)]
    text = ' '.join(lines)
    return re.sub(r'\s+', ' ', text).strip()


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def _extract_doi(text: str) -> Optional[str]:
    m = _DOI_RE.search(text)
    if not m:
        return None
    doi = m.group(1).rstrip('.,;')
    return clean_doi(doi)


def _extract_arxiv(text: str) -> Optional[str]:
    m = _ARXIV_RE.search(text)
    if not m:
        return None
    return clean_arxiv_id(m.group(1))


def _extract_url(text: str, doi: Optional[str], eprint: Optional[str]) -> Optional[str]:
    """Return the first URL in text that is not a DOI or arXiv link."""
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip('.,;)')
        lower = url.lower()
        if 'doi.org' in lower:
            continue
        if 'arxiv.org' in lower:
            continue
        return url
    return None


def _extract_title(remainder: str) -> Optional[str]:
    """Extract the title from the text after '. YEAR. '.

    The title ends at the first venue-like token. If no venue token is found,
    the entire remainder (minus trailing identifiers) is the title.
    """
    # Strip trailing DOI/arXiv/URL noise so it doesn't end up in the title
    remainder = _DOI_RE.sub('', remainder)
    remainder = _ARXIV_RE.sub('', remainder)
    remainder = _URL_RE.sub('', remainder)
    remainder = re.sub(r'\s+', ' ', remainder).strip().rstrip('.,;')

    m = _TITLE_END_RE.search(remainder)
    if m:
        title = remainder[:m.start()].strip().rstrip('.,;')
    else:
        title = remainder

    return title if title else None


def _parse_pdf_authors(raw: str) -> list[str]:
    """Parse an ACM-style author string into a list of individual names.

    ACM references use "First Last" order, comma-separated within a group
    and " and " between the last two authors.

    Examples:
        "Abdul Afram and Farrokh Janabi-Sharifi"
        "A. Mamun, I. Narayanan, D. Wang, A. Sivasubramaniam, and H.K. Fathy"
    """
    # First split on " and "
    parts = _AUTHOR_AND_RE.split(raw.strip())
    names: list[str] = []
    for part in parts:
        # Within each part, split on ", " — but only where the token after the
        # comma looks like a personal name (starts with capital letter or initial)
        # to avoid splitting "Gurobi Optimization, LLC" incorrectly.
        # Split on ", " only when the next token looks like a personal name:
        # a capital letter followed by a lowercase letter or period (e.g. "I."
        # or "John"). This avoids splitting corporate suffixes like "LLC".
        sub_parts = re.split(r',\s+(?=[A-Z][a-z.])', part)
        for sub in sub_parts:
            name = sub.strip().rstrip(',')
            if name:
                names.append(name)
    return names
