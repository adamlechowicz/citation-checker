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

    authors = _parse_authors(raw.get("author", ""))
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
    )


def _parse_authors(raw: str) -> list[str]:
    """Split on ' and ' (case-insensitive) and return canonical 'First Last' names."""
    if not raw.strip():
        return []
    parts = re.split(r'\s+and\s+', raw.strip(), flags=re.IGNORECASE)
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
    return result


def _parse_year(raw: Optional[str]) -> Optional[int]:
    if not raw:
        return None
    m = re.search(r'\b(1[5-9]\d{2}|20\d{2})\b', raw)
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
