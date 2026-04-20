"""Tests for the PDF bibliography parser."""

from __future__ import annotations

import pytest
from pathlib import Path

from citation_checker.pdf_parser import (
    _collapse_block,
    _extract_doi,
    _extract_arxiv,
    _extract_url,
    _extract_title,
    _find_references_section,
    _parse_pdf_authors,
    _split_into_blocks,
    parse_pdf_file,
    _ACM_YEAR_SPLIT_RE,
)

# Path to the real PDF used for integration tests
PDF_PATH = Path(__file__).parent.parent / "2511.18554v2.pdf"


# ---------------------------------------------------------------------------
# Unit tests — no PDF file required
# ---------------------------------------------------------------------------

class TestExtractDoi:
    def test_doi_prefix(self):
        assert _extract_doi("doi:10.1145/3469379.3469382") == "10.1145/3469379.3469382"

    def test_doi_url(self):
        assert _extract_doi("https://doi.org/10.1016/j.apenergy.2017.07.034") == "10.1016/j.apenergy.2017.07.034"

    def test_dx_doi_url(self):
        assert _extract_doi("http://dx.doi.org/10.1109/tsg.2014.2320261") == "10.1109/tsg.2014.2320261"

    def test_trailing_punctuation_stripped(self):
        doi = _extract_doi("doi:10.1145/146585.146588.")
        assert doi is not None
        assert not doi.endswith(".")

    def test_no_doi(self):
        assert _extract_doi("No identifier here, just text.") is None


class TestExtractArxiv:
    def test_new_format(self):
        assert _extract_arxiv("arXiv:2206.13606 [cs.DS]") == "2206.13606"

    def test_old_format(self):
        assert _extract_arxiv("arXiv:math/0501328 [math.CA]") == "math/0501328"

    def test_version_suffix_stripped(self):
        assert _extract_arxiv("arXiv:1912.01703v1") == "1912.01703"

    def test_case_insensitive(self):
        assert _extract_arxiv("arxiv:2206.13606") == "2206.13606"

    def test_no_arxiv(self):
        assert _extract_arxiv("No identifier here.") is None


class TestExtractUrl:
    def test_plain_url(self):
        url = _extract_url("See https://example.com/paper for details.", None, None)
        assert url == "https://example.com/paper"

    def test_doi_url_skipped(self):
        url = _extract_url("https://doi.org/10.1145/123", "10.1145/123", None)
        assert url is None

    def test_arxiv_url_skipped(self):
        url = _extract_url("https://arxiv.org/abs/2206.13606", None, "2206.13606")
        assert url is None

    def test_trailing_punctuation_stripped(self):
        url = _extract_url("Visit https://example.com.", None, None)
        assert url is not None
        assert not url.endswith(".")


class TestCollapseBlock:
    def test_joins_lines(self):
        result = _collapse_block("line one\nline two\nline three")
        assert result == "line one line two line three"

    def test_removes_page_numbers(self):
        result = _collapse_block("before\n23\nafter")
        assert "23" not in result.split()  # "23" should not appear as standalone token

    def test_heals_split_url(self):
        raw = "See https:\n//arxiv.org/abs/1706.03762 for details."
        result = _collapse_block(raw)
        assert "https://arxiv.org/abs/1706.03762" in result

    def test_removes_et_al_running_head(self):
        result = _collapse_block("reference text\nSmith et al.\nmore text")
        assert "Smith et al." not in result


class TestAcmYearSplit:
    def test_basic_split(self):
        text = "Abdul Afram and Farrokh Janabi-Sharifi. 2014. Theory and applications of HVAC. Building."
        m = _ACM_YEAR_SPLIT_RE.match(text)
        assert m is not None
        assert m.group(2) == "2014"
        assert "Afram" in m.group(1)
        assert "Theory" in m.group(3)

    def test_year_in_range(self):
        text = "Author Name. 1998. A Classic Paper. Some Journal."
        m = _ACM_YEAR_SPLIT_RE.match(text)
        assert m is not None
        assert m.group(2) == "1998"

    def test_no_match_without_period_year_period(self):
        text = "No year in this text."
        m = _ACM_YEAR_SPLIT_RE.match(text)
        assert m is None


class TestParsePdfAuthors:
    def test_two_authors_and(self):
        result = _parse_pdf_authors("Abdul Afram and Farrokh Janabi-Sharifi")
        assert len(result) == 2
        assert "Abdul Afram" in result
        assert "Farrokh Janabi-Sharifi" in result

    def test_comma_separated_with_and(self):
        result = _parse_pdf_authors("A. Mamun, I. Narayanan, D. Wang, and H.K. Fathy")
        assert len(result) == 4

    def test_single_author(self):
        result = _parse_pdf_authors("Isi Mitrani")
        assert result == ["Isi Mitrani"]

    def test_corporate_author_not_split(self):
        # "LLC" after comma should not be treated as a new author name
        result = _parse_pdf_authors("Gurobi Optimization, LLC")
        # Should be one name, not two
        assert len(result) == 1


class TestFindReferencesSection:
    def test_finds_references_heading(self):
        text = "Introduction\n...\nReferences\n[1] Some paper.\n"
        result = _find_references_section(text)
        assert "[1]" in result

    def test_finds_bibliography_heading(self):
        text = "Body text.\nBibliography\n[1] A paper."
        result = _find_references_section(text)
        assert "[1]" in result

    def test_all_caps_heading(self):
        text = "Conclusion.\nREFERENCES\n[1] A paper."
        result = _find_references_section(text)
        assert "[1]" in result

    def test_raises_if_not_found(self):
        with pytest.raises(ValueError, match="References"):
            _find_references_section("No bibliography here at all.")


class TestSplitIntoBlocks:
    def test_splits_three_blocks(self):
        text = "[1] First reference.\n[2] Second reference.\n[3] Third."
        blocks = _split_into_blocks(text)
        assert len(blocks) == 3
        assert blocks[0][0] == 1
        assert blocks[1][0] == 2
        assert blocks[2][0] == 3

    def test_block_content(self):
        text = "[1] First paper. 2020. Title.\n[2] Second."
        blocks = _split_into_blocks(text)
        assert "First paper" in blocks[0][1]

    def test_empty_text(self):
        assert _split_into_blocks("No numbered references here.") == []


# ---------------------------------------------------------------------------
# Integration tests against the real PDF
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF fixture not present")
class TestRealPdf:
    @pytest.fixture(scope="class")
    def entries(self):
        return parse_pdf_file(str(PDF_PATH))

    @pytest.fixture(scope="class")
    def by_key(self, entries):
        return {e.key: e for e in entries}

    def test_parses_nonzero_entries(self, entries):
        assert len(entries) > 50

    def test_all_keys_are_refN(self, entries):
        for e in entries:
            assert e.key.startswith("ref")
            assert e.key[3:].isdigit()

    def test_all_entry_types_are_article(self, entries):
        for e in entries:
            assert e.entry_type == "article"

    def test_ref1_fields(self, by_key):
        e = by_key["ref1"]
        assert e.year == 2014
        assert any("Afram" in a for a in e.authors)
        assert e.title is not None
        assert "HVAC" in e.title or "control" in e.title.lower()

    def test_arxiv_entry_has_eprint(self, entries):
        arxiv_entries = [e for e in entries if e.eprint]
        assert len(arxiv_entries) > 0

    def test_doi_entry_has_doi(self, entries):
        doi_entries = [e for e in entries if e.doi]
        assert len(doi_entries) > 0

    def test_no_entry_is_completely_empty(self, entries):
        for e in entries:
            has_something = any([e.title, e.authors, e.year, e.doi, e.eprint])
            assert has_something, f"{e.key} has no parsed fields at all"
