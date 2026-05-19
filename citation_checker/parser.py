"""Parse .bib files into BibEntry dataclasses using bibtexparser v1."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Optional

import bibtexparser
from bibtexparser.bparser import BibTexParser
from bibtexparser.customization import convert_to_unicode

from .models import BibEntry
from .utils import clean_doi, clean_arxiv_id

log = logging.getLogger(__name__)

_LATEX_CMD_RE = re.compile(r'\\[a-zA-Z]+\{([^}]*)\}|\\[a-zA-Z]+\s*')
_BRACE_RE = re.compile(r'[{}]')
_WHITESPACE_RE = re.compile(r'\s+')

# Trailing truncation markers in a bib author field.
_BIB_AUTHOR_TRUNC_RE = re.compile(
    r'\s+(?:and\s+others|et\s+al\.?)\s*$',
    re.IGNORECASE,
)

# Comma-separated author chunk shape: starts with a Title-cased token
# (capital + word chars) OR an initial (capital + period). Used to decide
# whether a comma-only list is a safe-to-split author list.
_AUTHOR_CHUNK_SHAPE_RE = re.compile(r"^[A-Z](?:[\w'À-ſ\-]+|\.)")


def parse_bib_file(path: str) -> list[BibEntry]:
    """Parse a .bib file and return a list of BibEntry objects.

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if the file cannot be parsed.
    """
    parser = BibTexParser(common_strings=True)
    parser.customization = convert_to_unicode
    parser.ignore_nonstandard_types = False

    try:
        with open(path, encoding="utf-8") as fh:
            library = bibtexparser.load(fh, parser=parser)
    except UnicodeDecodeError:
        with open(path, encoding="latin-1") as fh:
            library = bibtexparser.load(fh, parser=parser)

    entries = []
    for raw in library.entries:
        try:
            entry = _normalize_entry(raw)
            entries.append(entry)
        except Exception as exc:
            log.warning("Skipping entry %r due to error: %s", raw.get("ID", "?"), exc)

    log.info("Parsed %d entries from %s", len(entries), path)
    return entries


def _normalize_entry(raw: dict) -> BibEntry:
    """Convert a bibtexparser v1 entry dict to BibEntry."""
    key = raw.get("ID", "unknown")
    entry_type = raw.get("ENTRYTYPE", "misc").lower()

    title = raw.get("title")
    if title:
        title = _strip_latex(title).strip() or None

    authors, truncated_authors = _parse_authors(raw.get("author", ""))
    year = _parse_year(raw.get("year"))

    doi_raw = raw.get("doi") or raw.get("DOI")
    doi = clean_doi(doi_raw) if doi_raw else None

    eprint_raw = raw.get("eprint")
    eprint = clean_arxiv_id(eprint_raw) if eprint_raw else None

    archiveprefix = raw.get("archiveprefix") or raw.get("archivePrefix")
    url = raw.get("url") or raw.get("URL")

    # Exclude bibtexparser internal keys from raw_fields
    internal_keys = {"ID", "ENTRYTYPE"}
    raw_fields = {k: v for k, v in raw.items() if k not in internal_keys}

    return BibEntry(
        key=key,
        entry_type=entry_type,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        url=url,
        eprint=eprint,
        archiveprefix=archiveprefix,
        raw_fields=raw_fields,
        truncated_authors=truncated_authors,
    )


def _parse_authors(raw: str) -> tuple[list[str], bool]:
    """Split a bib author field into canonical 'First Last' names.

    Returns ``(authors, truncated_authors)`` where ``truncated_authors`` is
    True when the raw field ended with a "and others" or "et al." marker
    (BibTeX's standard way of saying the list was truncated).

    Splitting:
      - Primary: split on ``\\s+and\\s+`` (the BibTeX-standard separator).
      - Fallback: if the field has zero ``and`` separators but ≥2 commas
        AND every comma-delimited chunk looks like a Title-cased name,
        split on commas. This recovers entries that humans / non-BibTeX
        tools formatted with commas only ("Smith, J., Jones, K.").
      - Single "Last, First" still parses as one author (the chunk shape
        check is per-chunk; "Last, First" has 1 comma so it doesn't enter
        the comma-split branch).
    """
    raw = raw.strip()
    if not raw:
        return [], False

    truncated = bool(_BIB_AUTHOR_TRUNC_RE.search(raw))
    if truncated:
        raw = _BIB_AUTHOR_TRUNC_RE.sub('', raw).strip()
        if not raw:
            return [], True

    if re.search(r'\s+and\s+', raw, flags=re.IGNORECASE):
        parts = re.split(r'\s+and\s+', raw, flags=re.IGNORECASE)
    else:
        parts = _split_comma_only_authors(raw)

    result = []
    for part in parts:
        part = _strip_latex(part).strip()
        if not part:
            continue
        # Handle "Last, First Middle" → "First Middle Last"
        if ',' in part:
            last, _, rest = part.partition(',')
            name = f"{rest.strip()} {last.strip()}"
        else:
            name = part
        name = _whitespace_norm(name)
        if name:
            result.append(name)
    return result, truncated


def _split_comma_only_authors(raw: str) -> list[str]:
    """Decide how to split a comma-only author string.

    Returns the list of chunks to feed into the normaliser. Falls back to
    a single-element list (existing behaviour) when the comma-split shape
    is ambiguous.

    Heuristics:
      - 1 comma: ambiguous between "Last, First" (1 author) and
        "First1 Last1, First2 Last2" (2 authors). Split only when BOTH
        chunks contain an internal space, signaling each is already a
        multi-word "First Last".
      - ≥2 commas: split when every chunk starts with a Title-cased
        token. A bare "Smith, John" trailing fragment would fail this
        check and the whole string falls back to the existing logic.
    """
    chunks = [c.strip() for c in raw.split(',')]
    chunks = [c for c in chunks if c]
    if len(chunks) < 2:
        return [raw]

    # Every chunk must start with a Title-cased token; otherwise the
    # comma may be a "Last, First" separator we should not break apart.
    if not all(_AUTHOR_CHUNK_SHAPE_RE.match(c) for c in chunks):
        return [raw]

    # Every chunk must contain an internal space — i.e., look like a
    # multi-word "First Last" name. This rejects:
    #   - "Smith, John"           (chunk2 has no space)
    #   - "Smith, John, Jones, K" (chunks 1,2,3,4 single-word)
    # while accepting:
    #   - "Alice Smith, Bob Jones, Carol White"
    if not all(' ' in c for c in chunks):
        return [raw]

    return chunks


def _parse_year(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    m = re.search(r'\b(1[5-9]\d{2}|2[01]\d{2})\b', raw)
    if m:
        return int(m.group(1))
    return None


def _strip_latex(text: str) -> str:
    """Remove common LaTeX markup, leaving plain text."""
    # Replace \cmd{content} with content
    text = _LATEX_CMD_RE.sub(lambda m: m.group(1) if m.group(1) is not None else '', text)
    # Remove remaining braces
    text = _BRACE_RE.sub('', text)
    # Normalise unicode
    text = unicodedata.normalize('NFC', text)
    return text


def _whitespace_norm(text: str) -> str:
    return _WHITESPACE_RE.sub(' ', text).strip()
