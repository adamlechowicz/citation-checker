"""Tests for the fuzzy matching module."""

import pytest

from citation_checker.fuzzy import (
    compare_records,
    normalize_string,
    _score_title,
    _score_authors,
    _score_year,
    TITLE_THRESHOLD,
    AUTHOR_THRESHOLD,
    AUTHOR_SOFT_THRESHOLD,
)
from citation_checker.models import BibEntry, RemoteRecord, VerificationStatus


def _make_entry(**kwargs) -> BibEntry:
    defaults = dict(
        key="test",
        entry_type="article",
        title="Test Paper",
        authors=["Alice Smith"],
        year=2020,
        doi=None,
        url=None,
        eprint=None,
        archiveprefix=None,
        raw_fields={},
    )
    defaults.update(kwargs)
    return BibEntry(**defaults)


def _make_remote(**kwargs) -> RemoteRecord:
    defaults = dict(
        title="Test Paper",
        authors=["Alice Smith"],
        year=2020,
        source="crossref",
        raw_response={},
    )
    defaults.update(kwargs)
    return RemoteRecord(**defaults)


class TestNormalizeString:
    def test_lowercases(self):
        assert normalize_string("Hello World") == "hello world"

    def test_strips_latex_braces(self):
        result = normalize_string("{Gradient}-Based Learning")
        assert "{" not in result
        assert "}" not in result

    def test_collapses_whitespace(self):
        assert normalize_string("a   b  c") == "a b c"

    def test_unicode_normalisation(self):
        # Both forms of é should normalise the same way
        result1 = normalize_string("caf\u00e9")   # precomposed
        result2 = normalize_string("cafe\u0301")   # decomposed
        assert result1 == result2


class TestScoreTitle:
    def test_identical_titles(self):
        assert _score_title("Attention Is All You Need", "Attention Is All You Need") == 100.0

    def test_case_insensitive(self):
        score = _score_title("attention is all you need", "Attention Is All You Need")
        assert score >= TITLE_THRESHOLD

    def test_minor_difference(self):
        score = _score_title(
            "Gradient-based learning applied to document recognition",
            "Gradient-Based Learning Applied to Document Recognition",
        )
        assert score >= TITLE_THRESHOLD

    def test_completely_different(self):
        score = _score_title("Neural Networks", "Competitive Analysis of Online Algorithms")
        assert score < TITLE_THRESHOLD

    def test_none_local(self):
        assert _score_title(None, "Something") == 100.0

    def test_none_remote(self):
        assert _score_title("Something", None) == 100.0


class TestScoreAuthors:
    def test_identical(self):
        assert _score_authors(["Alice Smith"], ["Alice Smith"]) == 100.0

    def test_empty_local(self):
        # Empty local author list returns None (uncomparable) — callers must
        # decide how to handle the missing data rather than silently passing.
        assert _score_authors([], ["Alice Smith"]) is None

    def test_empty_remote(self):
        score = _score_authors(["Alice Smith"], [])
        assert score < AUTHOR_THRESHOLD

    def test_last_first_vs_first_last(self):
        score = _score_authors(["Smith, Alice"], ["Alice Smith"])
        assert score >= AUTHOR_THRESHOLD

    def test_partial_author_list(self):
        # Local only has first author but remote has all 8
        score = _score_authors(
            ["Ashish Vaswani"],
            ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit"],
        )
        assert score >= AUTHOR_THRESHOLD


class TestScoreYear:
    def test_match(self):
        assert _score_year(2020, 2020) is True

    def test_mismatch(self):
        assert _score_year(2020, 2021) is False

    def test_none_local(self):
        assert _score_year(None, 2020) is None

    def test_none_remote(self):
        assert _score_year(2020, None) is None


class TestCompareRecords:
    def test_verified_exact_match(self):
        entry = _make_entry()
        remote = _make_remote()
        scores, status, warnings = compare_records(entry, remote)
        assert status == VerificationStatus.VERIFIED
        assert scores.title_score == 100.0
        assert warnings == []

    def test_year_mismatch_forces_mismatch(self):
        # Year mismatch always forces MISMATCH — the README documents this
        # as the authoritative behaviour, and a fabricated year on an
        # otherwise-real paper is exactly the LLM-hallucination pattern we
        # want to catch.
        entry = _make_entry(year=2020)
        remote = _make_remote(year=2021)
        _, status, warnings = compare_records(entry, remote)
        assert status == VerificationStatus.MISMATCH
        assert any("Year mismatch" in w for w in warnings)

    def test_mismatch_title(self):
        entry = _make_entry(title="Neural Networks for Classification")
        remote = _make_remote(title="Attention Is All You Need")
        _, status, _ = compare_records(entry, remote)
        assert status == VerificationStatus.MISMATCH

    def test_year_none_does_not_block_verified(self):
        entry = _make_entry(year=None)
        remote = _make_remote(year=2020)
        _, status, warnings = compare_records(entry, remote)
        assert status == VerificationStatus.VERIFIED
        assert warnings == []

    def test_author_formatting_is_soft_warning(self):
        # Abbreviated first name ("A. Dobos") vs full name ("Aron P Dobos")
        # scores in the 55-80 soft zone — should be VERIFIED with a warning
        entry = _make_entry(authors=["Aron P Dobos"])
        remote = _make_remote(authors=["A. Dobos"])
        _, status, warnings = compare_records(entry, remote)
        assert status == VerificationStatus.VERIFIED
        assert any("formatting" in w for w in warnings)

    def test_genuinely_wrong_author_is_mismatch(self):
        # Completely different authors should still be MISMATCH
        entry = _make_entry(authors=["Bob Jones"])
        remote = _make_remote(authors=["Alice Smith", "Carol White"])
        _, status, _ = compare_records(entry, remote)
        assert status == VerificationStatus.MISMATCH

    def test_subtitle_prefix_match(self):
        # CrossRef stores "DACF"; local has "DACF: Day-Ahead Carbon Intensity..."
        entry = _make_entry(title="DACF: Day-Ahead Carbon Intensity Forecasting")
        remote = _make_remote(title="DACF")
        scores, status, _ = compare_records(entry, remote)
        assert status == VerificationStatus.VERIFIED
        assert scores.title_score == 100.0

    def test_subtitle_prefix_match_reverse(self):
        # Local is abbreviated, remote has subtitle
        entry = _make_entry(title="ACN-Data")
        remote = _make_remote(title="ACN-Data: Analysis and Applications of an Open EV Charging Dataset")
        scores, status, _ = compare_records(entry, remote)
        assert status == VerificationStatus.VERIFIED
        assert scores.title_score == 100.0


class TestPrefixRescueMetadataGate:
    """A3: prefix rescue only fires when surplus looks like venue metadata."""

    def test_venue_suffix_rescues(self):
        score = _score_title(
            "Carbon-aware load shifting for data centers. Nature Energy 2025",
            "Carbon-aware load shifting for data centers",
        )
        assert score == 100.0

    def test_proceedings_suffix_rescues(self):
        score = _score_title(
            "Carbon-aware load shifting for data centers. Proceedings of FOO 2024",
            "Carbon-aware load shifting for data centers",
        )
        assert score == 100.0

    def test_non_metadata_suffix_does_not_rescue(self):
        # Adversarial: appending plausible English words to a real title.
        score = _score_title(
            "Carbon-aware load shifting for data centers applications and best practices",
            "Carbon-aware load shifting for data centers",
        )
        assert score < TITLE_THRESHOLD


class TestEmptyLocalAuthors:
    """A2: empty local author list flows through as None, not 100."""

    def test_strong_title_with_empty_authors_verifies_with_warning(self):
        entry = _make_entry(authors=[])
        remote = _make_remote()  # same title 100%
        scores, status, warnings = compare_records(entry, remote)
        assert scores.author_score is None
        assert status == VerificationStatus.VERIFIED
        assert any("authors not parsed" in w.lower() for w in warnings)

    def test_weak_title_with_empty_authors_is_mismatch(self):
        # Title score of ~89 — passes TITLE_THRESHOLD (85) but below the
        # title-only strong threshold (95).
        entry = _make_entry(authors=[], title="Gradient learning applied to document recognition")
        remote = _make_remote(title="Gradient-Based Learning Applied to Document Recognition")
        scores, status, _ = compare_records(entry, remote)
        assert scores.author_score is None
        # 89 < 95 strong threshold → MISMATCH
        assert status == VerificationStatus.MISMATCH
