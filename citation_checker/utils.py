import re


_DOI_URL_PREFIX_RE = re.compile(r'^https?://(?:dx\.)?doi\.org/', re.IGNORECASE)
_DOI_LABEL_PREFIX_RE = re.compile(r'^doi\s*:\s*', re.IGNORECASE)
_DOI_TRAILING_PUNCT_RE = re.compile(r'[;.,)\]>]+$')

_ARXIV_LABEL_PREFIX_RE = re.compile(r'^arxiv\s*:\s*', re.IGNORECASE)
_ARXIV_CATEGORY_SUFFIX_RE = re.compile(r'\s*\[[^\]]+\]\s*$')


def clean_doi(doi: str) -> str:
    """Strip URL/label prefixes and trailing punctuation, leaving a bare DOI."""
    doi = doi.strip()
    # Repeatedly strip prefixes so "DOI: https://doi.org/10..." works.
    while True:
        new = _DOI_URL_PREFIX_RE.sub('', doi)
        new = _DOI_LABEL_PREFIX_RE.sub('', new)
        if new == doi:
            break
        doi = new.strip()
    doi = _DOI_TRAILING_PUNCT_RE.sub('', doi)
    return doi.strip()


def clean_arxiv_id(eprint: str) -> str:
    """Normalise an arXiv eprint ID.

    Handles:
    - 'arXiv:2301.00001' -> '2301.00001'
    - 'arXiv:cs/0501001' -> 'cs/0501001'
    - '1234.5678 [cs.LG]' -> '1234.5678'
    - 'arxiv: 1234.5678v3' -> '1234.5678v3'  (version suffix preserved)
    - Already bare IDs returned unchanged.
    """
    eprint = eprint.strip()
    eprint = _ARXIV_LABEL_PREFIX_RE.sub('', eprint)
    eprint = _ARXIV_CATEGORY_SUFFIX_RE.sub('', eprint)
    return eprint.strip()


def truncate(text: str, max_len: int = 60) -> str:
    """Truncate a string for display, appending '...' if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
