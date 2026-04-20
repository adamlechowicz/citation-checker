import re


def clean_doi(doi: str) -> str:
    """Strip URL prefix leaving a bare DOI, and strip whitespace."""
    doi = doi.strip()
    doi = re.sub(r'^https?://(dx\.)?doi\.org/', '', doi, flags=re.IGNORECASE)
    return doi.strip()


def clean_arxiv_id(eprint: str) -> str:
    """Normalise an arXiv eprint ID.

    Handles:
    - 'arXiv:2301.00001' -> '2301.00001'
    - 'arXiv:cs/0501001' -> 'cs/0501001'
    - Already bare IDs returned unchanged.
    """
    eprint = eprint.strip()
    eprint = re.sub(r'^arxiv:', '', eprint, flags=re.IGNORECASE)
    return eprint.strip()


def truncate(text: str, max_len: int = 60) -> str:
    """Truncate a string for display, appending '...' if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."
