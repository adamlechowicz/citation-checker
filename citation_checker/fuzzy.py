"""Deterministic fuzzy-matching logic for comparing local bib entries to remote records."""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

from rapidfuzz import fuzz

from .models import BibEntry, FieldScores, RemoteRecord, VerificationStatus

# Hard thresholds — failures below these become MISMATCH
TITLE_THRESHOLD: float = 85.0
AUTHOR_THRESHOLD: float = 80.0

# Soft threshold for authors — scores between this and AUTHOR_THRESHOLD are
# reported as warnings (e.g. initials vs. full names) but do not cause MISMATCH
AUTHOR_SOFT_THRESHOLD: float = 55.0

_LATEX_ARTIFACT_RE = re.compile(r'\\[a-zA-Z]+|[{}~]|--+')
_PUNCT_RE = re.compile(r"[^\w\s\-']")
_WHITESPACE_RE = re.compile(r'\s+')


def compare_records(
    local: BibEntry, remote: RemoteRecord, *, skip_author_year: bool = False
) -> tuple[FieldScores, VerificationStatus, list[str]]:
    """Compare a local BibEntry against a RemoteRecord.

    Returns (FieldScores, VerificationStatus, soft_warnings).

    Soft warnings are issues that do not change the status to MISMATCH:
      - Year mismatch (common for conference paper vs. journal version)
      - Author score between AUTHOR_SOFT_THRESHOLD and AUTHOR_THRESHOLD
        (common for initials vs. full names, hyphen encoding, etc.)

    Hard failures that produce MISMATCH:
      - Title score below TITLE_THRESHOLD
      - Author score below AUTHOR_SOFT_THRESHOLD (genuinely different people)
    """
    soft_warnings: list[str] = []

    title_score = _score_title(local.title, remote.title)

    if skip_author_year:
        # Web records have no structured author/year metadata — score title only.
        author_score = 100.0
        year_match = None
    else:
        author_score = _score_authors(local.authors, remote.authors)
        year_match = _score_year(local.year, remote.year)

        # Soft warning: year mismatch (conference vs. journal version)
        if year_match is False:
            soft_warnings.append(
                f"Year mismatch: bib={local.year}, database={remote.year}"
                f" (may be conference vs. journal/final version)"
            )

        # Soft warning: author formatting difference (initials vs. full names, etc.)
        if AUTHOR_SOFT_THRESHOLD <= author_score < AUTHOR_THRESHOLD:
            soft_warnings.append(
                f"Author name formatting difference (score {author_score:.0f}/100)"
                f"; may be initials vs. full names or encoding variation"
            )

    # Hard failures → MISMATCH
    title_ok = title_score >= TITLE_THRESHOLD
    author_hard_ok = author_score >= AUTHOR_SOFT_THRESHOLD  # hard floor is the soft threshold

    if title_ok and author_hard_ok:
        status = VerificationStatus.VERIFIED
    else:
        status = VerificationStatus.MISMATCH

    return FieldScores(
        title_score=title_score,
        author_score=author_score,
        year_match=year_match,
    ), status, soft_warnings


def _score_title(local: Optional[str], remote: Optional[str]) -> float:
    """Compute title similarity (0–100). Returns 100 if either side is None (skip).

    Subtitle prefix rule: if CrossRef stores an abbreviated title (everything
    before the first ':'), e.g. remote="DACF" for local="DACF: Day-Ahead...",
    that counts as a full match.
    """
    if local is None or remote is None:
        return 100.0

    norm_local = normalize_string(local)
    norm_remote = normalize_string(remote)

    score = fuzz.ratio(norm_local, norm_remote)
    if score >= TITLE_THRESHOLD:
        return score

    # Subtitle prefix check (split before normalizing to preserve ':')
    local_prefix = normalize_string(local.split(':', 1)[0])
    remote_prefix = normalize_string(remote.split(':', 1)[0])

    # CrossRef stores local's prefix (e.g. "DACF" == "DACF: Day-Ahead..."[:colon])
    if local_prefix and norm_remote == local_prefix:
        return 100.0
    # Reverse: local is the abbreviated form, remote has the subtitle
    if remote_prefix and norm_local == remote_prefix:
        return 100.0

    # PDF-parsed titles often include trailing venue info (e.g. ". Nature Energy 2025").
    # If the local title starts with the full remote title, count as a match.
    # Require remote to be substantial (>20 chars) to avoid spurious short matches.
    if len(norm_remote) > 20 and norm_local.startswith(norm_remote):
        return 100.0
    # Reverse: if remote starts with local (remote has a subtitle we don't)
    if len(norm_local) > 20 and norm_remote.startswith(norm_local):
        return 100.0

    return score


def _score_authors(local: list[str], remote: list[str]) -> float:
    """Compute author set similarity (0–100).

    For each local author, find the best match among remote authors.
    Average those best-match scores.
    Returns 100.0 if local list is empty (cannot penalise).
    """
    if not local:
        return 100.0
    if not remote:
        # Local has authors but remote has none — suspicious, penalise lightly.
        return 50.0

    norm_remote = [normalize_string(a) for a in remote]
    scores = []
    for la in local:
        norm_la = normalize_string(la)
        best = max(
            fuzz.token_sort_ratio(norm_la, nr) for nr in norm_remote
        )
        scores.append(best)
    return sum(scores) / len(scores)


def _score_year(local: Optional[int], remote: Optional[int]) -> Optional[bool]:
    """Exact year comparison.

    Returns True if equal, False if both present and differ, None if either missing.
    """
    if local is None or remote is None:
        return None
    return local == remote


_SMART_QUOTE_TABLE = str.maketrans({
    '\u2018': "'",  # LEFT SINGLE QUOTATION MARK
    '\u2019': "'",  # RIGHT SINGLE QUOTATION MARK
    '\u201a': "'",  # SINGLE LOW-9 QUOTATION MARK
    '\u201b': "'",  # SINGLE HIGH-REVERSED-9 QUOTATION MARK
    '\u02bc': "'",  # MODIFIER LETTER APOSTROPHE
    '\u2032': "'",  # PRIME
    '\u201c': '"',  # LEFT DOUBLE QUOTATION MARK
    '\u201d': '"',  # RIGHT DOUBLE QUOTATION MARK
})


def normalize_string(text: str) -> str:
    """Canonical normalisation for fuzzy comparison.

    Steps: unify smart quotes → NFKD unicode → lowercase →
           strip LaTeX artifacts → strip punctuation (except hyphens) →
           collapse whitespace.
    """
    text = text.translate(_SMART_QUOTE_TABLE)
    text = unicodedata.normalize('NFKD', text)
    text = text.lower()
    text = _LATEX_ARTIFACT_RE.sub(' ', text)
    text = _PUNCT_RE.sub(' ', text)
    text = _WHITESPACE_RE.sub(' ', text)
    return text.strip()
