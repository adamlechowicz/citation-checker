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

# Venue comparison thresholds (journal / booktitle vs. container-title)
VENUE_MISMATCH_THRESHOLD: float = 45.0   # below → MISMATCH
VENUE_SOFT_THRESHOLD: float = 72.0       # 45–72 → soft warning only

# Per-author greedy match floor: assigned scores below this are treated as
# no match (zeroed). Prevents a remote "Jie Xu" from giving "Jie Zhang" a
# score of 40 just because it's the only author left to claim.
AUTHOR_MIN_MATCH: float = 45.0

_LATEX_ARTIFACT_RE = re.compile(r'\\[a-zA-Z]+|[{}~]|--+')
_PUNCT_RE = re.compile(r"[^\w\s\-']")
_WHITESPACE_RE = re.compile(r'\s+')


# Title score above which we trust a title-only match even with no author
# data — used for entries whose local author list failed to parse and for
# web-source verifications.
TITLE_ONLY_STRONG_THRESHOLD: float = 95.0


def compare_records(
    local: BibEntry, remote: RemoteRecord, *, skip_author_year: bool = False
) -> tuple[FieldScores, VerificationStatus, list[str]]:
    """Compare a local BibEntry against a RemoteRecord.

    Returns (FieldScores, VerificationStatus, soft_warnings).

    Soft warnings are issues that do not change the status to MISMATCH:
      - Author score between AUTHOR_SOFT_THRESHOLD and AUTHOR_THRESHOLD
        (common for initials vs. full names, hyphen encoding, etc.)

    Hard failures that produce MISMATCH:
      - Title score below TITLE_THRESHOLD
      - Author score below AUTHOR_SOFT_THRESHOLD (genuinely different people)
      - Year mismatch (when both sides have a year)
    """
    soft_warnings: list[str] = []

    title_score = _score_title(local.title, remote.title)

    venue_score: Optional[float] = None
    effective_author_floor = AUTHOR_SOFT_THRESHOLD
    author_score: Optional[float]
    year_match: Optional[bool]
    if skip_author_year:
        # Web records have no structured author/year metadata — score title only.
        author_score = None
        year_match = None
    else:
        author_score = _score_authors(local.authors, remote.authors)
        year_match = _score_year(local.year, remote.year)

        # Soft warning: author formatting difference (initials vs. full names, etc.)
        if author_score is not None and AUTHOR_SOFT_THRESHOLD <= author_score < AUTHOR_THRESHOLD:
            soft_warnings.append(
                f"Author name formatting difference (score {author_score:.0f}/100)"
                f"; may be initials vs. full names or encoding variation"
            )

        # When the remote record has many more authors than the local entry,
        # a shared surname can inflate the score without a real first-name match.
        # Require a solid match (≥ AUTHOR_THRESHOLD) in that case.
        #
        # Exception: when the local entry's author list was explicitly
        # truncated with "et al.", the size disparity is intentional — the
        # listed local authors are the FIRST N of the remote list, not a
        # coincidence. Keep the lax floor so an initial-vs-full-name mismatch
        # on the first author doesn't turn a real verify into a MISMATCH.
        remote_count = len(remote.authors)
        local_count = len(local.authors)
        size_skew = remote_count > 3 and local_count > 0 and remote_count / local_count >= 3
        if size_skew and not local.truncated_authors:
            effective_author_floor = AUTHOR_THRESHOLD
        else:
            effective_author_floor = AUTHOR_SOFT_THRESHOLD

        # Venue comparison (journal / booktitle vs. remote container-title)
        local_venue = local.raw_fields.get("journal") or local.raw_fields.get("booktitle")
        venue_score = _score_venue(local_venue, remote.container_title)
        if venue_score is not None:
            if venue_score < VENUE_MISMATCH_THRESHOLD:
                soft_warnings.append(
                    f"Venue mismatch: bib='{local_venue}', database='{remote.container_title}'"
                    f" (score {venue_score:.0f}/100)"
                )
            elif venue_score < VENUE_SOFT_THRESHOLD:
                soft_warnings.append(
                    f"Venue difference: bib='{local_venue}', database='{remote.container_title}'"
                    f" (score {venue_score:.0f}/100; may be abbreviation or alternate name)"
                )

    # Hard failures → MISMATCH
    title_ok = title_score >= TITLE_THRESHOLD
    if author_score is None:
        # Local authors not parsed (or skipped for web sources). Require a
        # very-strong title match to trust the verification.
        author_hard_ok = title_score >= TITLE_ONLY_STRONG_THRESHOLD
        if not skip_author_year:
            soft_warnings.append(
                "Local authors not parsed; verification relies on title alone"
            )
    else:
        author_hard_ok = author_score >= effective_author_floor
    year_ok = year_match is not False  # None (missing) is OK; False forces MISMATCH
    venue_ok = venue_score is None or venue_score >= VENUE_MISMATCH_THRESHOLD

    if not year_ok:
        soft_warnings.append(
            f"Year mismatch: bib={local.year}, database={remote.year}"
        )

    if title_ok and author_hard_ok and year_ok and venue_ok:
        status = VerificationStatus.VERIFIED
    else:
        status = VerificationStatus.MISMATCH

    return FieldScores(
        title_score=title_score,
        author_score=author_score,
        year_match=year_match,
    ), status, soft_warnings


# Metadata suffix shape — what should follow a real title that has been
# concatenated with venue/year/publisher text by a PDF extractor. Only
# accept the prefix-rescue when the surplus looks like one of these.
_TITLE_METADATA_SUFFIX_RE = re.compile(
    r'^[\s.,:;\-—]*'
    r'(?:in\b|proc\.?|proceedings\b|journal\b|vol\.?|arxiv\b|doi\b|'
    r'\d{4}\b|springer\b|elsevier\b|ieee\b|acm\b|pmlr\b|nature\b|'
    r'advances\s+in\b)',
    re.IGNORECASE,
)


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
    # Accept the prefix rescue ONLY when the surplus suffix looks like
    # citation metadata — otherwise a long fabricated title that simply
    # appends extra words to a real title would silently verify.
    if len(norm_remote) > 20 and norm_local.startswith(norm_remote):
        surplus = norm_local[len(norm_remote):]
        if _TITLE_METADATA_SUFFIX_RE.match(surplus):
            return 100.0
    if len(norm_local) > 20 and norm_remote.startswith(norm_local):
        surplus = norm_remote[len(norm_local):]
        if _TITLE_METADATA_SUFFIX_RE.match(surplus):
            return 100.0

    return score


def _score_authors(local: list[str], remote: list[str]) -> Optional[float]:
    """Compute author set similarity (0–100).

    Uses greedy one-to-one matching so that a single remote author cannot be
    claimed by multiple local authors (e.g. 'Jie Xu' and 'Jie Zhang' both
    matching the same remote 'Jie Xu' and inflating the score).

    Returns None when the local list is empty — caller must handle the
    "uncomparable" case explicitly rather than silently passing.
    """
    if not local:
        return None
    if not remote:
        return 50.0

    norm_local = [normalize_string(a) for a in local]
    norm_remote = [normalize_string(a) for a in remote]

    # Build full score matrix once
    matrix = [
        [fuzz.token_sort_ratio(nl, nr) for nr in norm_remote]
        for nl in norm_local
    ]

    # Greedy assignment: each remote slot can be claimed at most once
    used: set[int] = set()
    total = 0.0
    for i, row in enumerate(matrix):
        best_score = 0.0
        best_j = -1
        for j, s in enumerate(row):
            if j not in used and s > best_score:
                best_score = s
                best_j = j
        if best_j >= 0:
            used.add(best_j)
        # Scores below the minimum floor count as no match (the forced
        # assignment to a "least-bad" remote author is not meaningful).
        if best_score >= AUTHOR_MIN_MATCH:
            total += best_score

    return total / len(norm_local)


def _score_venue(local_venue: Optional[str], remote_venue: Optional[str]) -> Optional[float]:
    """Compare journal/booktitle against remote container-title.

    Returns a 0–100 score, or None if either side is absent (skip comparison).
    Uses token_sort_ratio so abbreviations like 'IEEE Trans.' still match
    'IEEE Transactions on Wireless Communications' well.
    """
    if not local_venue or not remote_venue:
        return None
    return fuzz.token_sort_ratio(normalize_string(local_venue), normalize_string(remote_venue))


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
