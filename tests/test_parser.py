"""Tests for the bib parser."""

import pytest
from pathlib import Path

from citation_checker.parser import parse_bib_file, _parse_authors, _parse_year, _strip_latex

FIXTURE = Path(__file__).parent / "fixtures" / "sample.bib"


def test_parse_file_returns_all_entries():
    entries = parse_bib_file(str(FIXTURE))
    # bibtexparser v1 drops entries with no fields; expect 4
    assert len(entries) == 4


def test_doi_cleaned():
    entries = {e.key: e for e in parse_bib_file(str(FIXTURE))}
    # Should be bare DOI, not URL
    assert entries["lecun1998"].doi == "10.1109/5.726791"


def test_arxiv_fields():
    entries = {e.key: e for e in parse_bib_file(str(FIXTURE))}
    e = entries["vaswani2017"]
    assert e.eprint == "1706.03762"
    assert e.archiveprefix is not None
    assert e.archiveprefix.lower() == "arxiv"


def test_authors_parsed():
    entries = {e.key: e for e in parse_bib_file(str(FIXTURE))}
    authors = entries["lecun1998"].authors
    assert len(authors) == 4
    # Should be in "First Last" canonical form
    assert any("LeCun" in a or "Lecun" in a or "lecun" in a.lower() for a in authors)


def test_year_parsed():
    entries = {e.key: e for e in parse_bib_file(str(FIXTURE))}
    assert entries["lecun1998"].year == 1998
    assert entries["vaswani2017"].year == 2017


def test_fake_entry_fields():
    entries = {e.key: e for e in parse_bib_file(str(FIXTURE))}
    # The fake/hallucinated entry should still parse with correct fields
    e = entries["fake2099"]
    assert e.title is not None
    assert "Fabricated" in e.title
    assert e.year == 2099
    assert e.doi == "10.9999/fake.doi.000000"


class TestParseAuthors:
    def test_last_first_format(self):
        result, truncated = _parse_authors("LeCun, Yann and Bengio, Yoshua")
        assert len(result) == 2
        assert any("Yann" in a for a in result)
        assert truncated is False

    def test_first_last_format(self):
        result, truncated = _parse_authors("Yann LeCun and Yoshua Bengio")
        assert len(result) == 2
        assert truncated is False

    def test_single_author(self):
        result, truncated = _parse_authors("Turing, Alan")
        assert len(result) == 1
        assert truncated is False

    def test_empty(self):
        assert _parse_authors("") == ([], False)

    def test_strips_latex(self):
        result, truncated = _parse_authors(r"Sch\"{o}lkopf, Bernhard")
        assert len(result) == 1
        assert result[0]  # non-empty
        assert truncated is False


class TestParseAuthorsCommaOnly:
    """A1: split comma-separated author lists that lack 'and' separators."""

    def test_two_full_name_authors_split(self):
        result, _ = _parse_authors("Alice Smith, Bob Jones")
        assert result == ["Alice Smith", "Bob Jones"]

    def test_commas_no_and_splits_three_authors(self):
        result, _ = _parse_authors("Alice Smith, Bob Jones, Carol White")
        assert result == ["Alice Smith", "Bob Jones", "Carol White"]

    def test_initial_form_authors_split(self):
        result, _ = _parse_authors("A. Smith, B. Jones, C. White")
        assert len(result) == 3
        assert "A. Smith" in result

    def test_single_last_first_stays_one_author(self):
        # "Smith, John" must NOT trigger comma-splitting.
        result, _ = _parse_authors("Smith, John")
        assert result == ["John Smith"]

    def test_multi_last_first_falls_back(self):
        # "Smith, John, Jones, Kate" is malformed (BibTeX would use 'and');
        # the safe behaviour is to keep it as one chunk rather than misparse.
        result, _ = _parse_authors("Smith, John, Jones, Kate")
        # Single-token chunks (no internal space) → fallback.
        assert len(result) == 1

    def test_comma_only_with_and_takes_and_path(self):
        # If 'and' is present, the comma-fallback must not fire.
        result, _ = _parse_authors("Smith, John and Jones, Kate")
        assert len(result) == 2

    def test_org_acronyms_stay_one(self):
        # Single-token chunks ("NREL", "EPA") — fallback rejects split.
        result, _ = _parse_authors("NREL, EPA, NASA")
        assert len(result) == 1


class TestParseAuthorsTruncation:
    """A5: detect 'and others' / 'et al.' truncation markers."""

    def test_and_others_marker(self):
        result, truncated = _parse_authors("Alice Smith and Bob Jones and others")
        assert truncated is True
        assert "others" not in " ".join(result).lower()

    def test_et_al_marker(self):
        result, truncated = _parse_authors("Alice Smith and Bob Jones et al.")
        assert truncated is True

    def test_no_marker(self):
        _, truncated = _parse_authors("Alice Smith and Bob Jones")
        assert truncated is False


class TestParseYear:
    def test_four_digit_year(self):
        assert _parse_year("2023") == 2023

    def test_year_in_braces(self):
        assert _parse_year("{2023}") == 2023

    def test_none_input(self):
        assert _parse_year(None) is None

    def test_invalid(self):
        assert _parse_year("forthcoming") is None

    # C4: extended year range
    def test_year_2100_in_range(self):
        assert _parse_year("2100") == 2100

    def test_year_2150_in_range(self):
        assert _parse_year("2150") == 2150

    def test_year_2200_out_of_range(self):
        assert _parse_year("2200") is None

    def test_year_1499_out_of_range(self):
        assert _parse_year("1499") is None


class TestStripLatex:
    def test_removes_braces(self):
        assert "{Hello}" not in _strip_latex("{Hello}")

    def test_removes_textit(self):
        result = _strip_latex(r"\textit{Nature}")
        assert r"\textit" not in result
        assert "Nature" in result
