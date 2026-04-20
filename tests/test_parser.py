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
        result = _parse_authors("LeCun, Yann and Bengio, Yoshua")
        assert len(result) == 2
        assert any("Yann" in a for a in result)

    def test_first_last_format(self):
        result = _parse_authors("Yann LeCun and Yoshua Bengio")
        assert len(result) == 2

    def test_single_author(self):
        result = _parse_authors("Turing, Alan")
        assert len(result) == 1

    def test_empty(self):
        assert _parse_authors("") == []

    def test_strips_latex(self):
        result = _parse_authors(r"Sch\"{o}lkopf, Bernhard")
        assert len(result) == 1
        assert result[0]  # non-empty


class TestParseYear:
    def test_four_digit_year(self):
        assert _parse_year("2023") == 2023

    def test_year_in_braces(self):
        assert _parse_year("{2023}") == 2023

    def test_none_input(self):
        assert _parse_year(None) is None

    def test_invalid(self):
        assert _parse_year("forthcoming") is None


class TestStripLatex:
    def test_removes_braces(self):
        assert "{Hello}" not in _strip_latex("{Hello}")

    def test_removes_textit(self):
        result = _strip_latex(r"\textit{Nature}")
        assert r"\textit" not in result
        assert "Nature" in result
