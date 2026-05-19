from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VerificationStatus(str, Enum):
    VERIFIED = "VERIFIED"
    MISMATCH = "MISMATCH"
    NOT_FOUND = "NOT_FOUND"
    GREY_LITERATURE = "GREY_LITERATURE"
    UNVERIFIABLE = "UNVERIFIABLE"
    ERROR = "ERROR"


class VerificationStrategy(str, Enum):
    DOI_CROSSREF = "doi_crossref"
    ARXIV = "arxiv"
    CROSSREF_SEARCH = "crossref_search"
    OPENALEX_SEARCH = "openalex_search"
    SEMANTICSCHOLAR = "semanticscholar"
    URL_WEB = "url_web"
    NONE = "none"


@dataclass
class BibEntry:
    key: str
    entry_type: str
    title: Optional[str]
    authors: list[str]
    year: Optional[int]
    doi: Optional[str]
    url: Optional[str]
    eprint: Optional[str]
    archiveprefix: Optional[str]
    raw_fields: dict
    # True when the original citation listed authors as "First Author et al."
    # (or similar truncation). The parsed `authors` list contains only the
    # names explicitly written before the "et al." marker, so a strict
    # local-vs-remote count comparison would over-penalise these entries.
    truncated_authors: bool = False


@dataclass
class RemoteRecord:
    title: Optional[str]
    authors: list[str]
    year: Optional[int]
    source: str
    raw_response: dict
    container_title: Optional[str] = None


@dataclass
class FieldScores:
    title_score: float
    # None when the local entry has no parsed authors — comparison is
    # not meaningful, and the verifier flags the entry with a warning.
    author_score: Optional[float]
    year_match: Optional[bool]


@dataclass
class VerificationResult:
    entry_key: str
    status: VerificationStatus
    strategy: VerificationStrategy
    remote_record: Optional[RemoteRecord]
    scores: Optional[FieldScores]
    url_reachable: Optional[bool]
    error_message: Optional[str]
    warnings: list[str] = field(default_factory=list)
