"""Tests for the grey-literature heuristic classifier."""

import pytest
from citation_checker.classifier import is_grey_literature, likely_book_publisher
from citation_checker.models import BibEntry


def _entry(**kwargs) -> BibEntry:
    defaults = dict(
        key="test", entry_type="misc", title=None, authors=[],
        year=None, doi=None, url=None, eprint=None,
        archiveprefix=None, raw_fields={},
    )
    defaults.update(kwargs)
    return BibEntry(**defaults)


# ---------------------------------------------------------------------------
# Should be classified as grey literature
# ---------------------------------------------------------------------------

class TestGreyLiterature:
    def test_misc_with_github_url(self):
        e = _entry(title="My Tool", url="https://github.com/user/repo")
        assert is_grey_literature(e)

    def test_misc_with_government_url(self):
        e = _entry(title="PVWatts Manual", url="https://pvwatts.nrel.gov/")
        assert is_grey_literature(e)

    def test_misc_with_org_author(self):
        e = _entry(title="Gurobi Optimizer", authors=["Gurobi Optimization, LLC"])
        assert is_grey_literature(e)

    def test_misc_with_allcaps_acronym_author(self):
        e = _entry(title="Emissions Report", authors=["EPA"])
        assert is_grey_literature(e)

    def test_misc_with_multiword_org_author(self):
        e = _entry(authors=["National Renewable Energy Laboratory"])
        assert is_grey_literature(e)

    def test_misc_with_software_keyword_in_title(self):
        e = _entry(title="PyTorch: An imperative deep learning framework")
        assert is_grey_literature(e)

    def test_misc_with_dataset_keyword(self):
        e = _entry(title="Cluster trace dataset from Alibaba")
        assert is_grey_literature(e)

    def test_misc_with_version_in_title(self):
        e = _entry(title="PVWatts Version 5 Manual")
        assert is_grey_literature(e)

    def test_online_entry_type_with_url(self):
        e = _entry(entry_type="online", url="https://watttime.org/")
        assert is_grey_literature(e)

    def test_article_with_clearly_grey_lit_domain(self):
        # Even a @article type pointing to a grey-lit URL should be flagged
        e = _entry(entry_type="article", url="https://zenodo.org/record/12345")
        assert is_grey_literature(e)


# ---------------------------------------------------------------------------
# Should NOT be classified as grey literature
# ---------------------------------------------------------------------------

class TestScholarly:
    def test_entry_with_doi(self):
        e = _entry(doi="10.1145/1234567.1234568")
        assert not is_grey_literature(e)

    def test_entry_with_arxiv(self):
        e = _entry(entry_type="misc", eprint="2206.13606", archiveprefix="arXiv")
        assert not is_grey_literature(e)

    def test_inproceedings_no_url(self):
        e = _entry(
            entry_type="inproceedings",
            title="Online Algorithms via ML Predictions",
            authors=["Ravi Kumar", "Manish Purohit"],
            year=2018,
        )
        assert not is_grey_literature(e)

    def test_article_with_doi_url(self):
        e = _entry(
            entry_type="article",
            url="https://doi.org/10.1016/j.energy.2021.01.001",
        )
        assert not is_grey_literature(e)

    def test_misc_no_signals(self):
        # A @misc with no URL, no org author, no keywords — don't classify
        e = _entry(
            entry_type="misc",
            title="Online scheduling with predictions",
            authors=["Jane Smith", "John Doe"],
            year=2021,
        )
        assert not is_grey_literature(e)

    def test_multiple_person_authors(self):
        e = _entry(
            entry_type="misc",
            authors=["Adam Lechowicz", "Cameron Musco"],
        )
        assert not is_grey_literature(e)

    def test_arxiv_url_not_grey(self):
        e = _entry(
            entry_type="misc",
            url="https://arxiv.org/abs/2206.13606",
        )
        assert not is_grey_literature(e)


# ---------------------------------------------------------------------------
# likely_book_publisher
# ---------------------------------------------------------------------------

class TestLikelyBookPublisher:
    def test_publisher_field_match(self):
        e = _entry(
            entry_type="book",
            title="Analytic Combinatorics",
            raw_fields={"publisher": "Cambridge University Press"},
        )
        assert likely_book_publisher(e) == "Cambridge University Press"

    def test_publisher_bled_into_title(self):
        # PDF parser sometimes includes publisher text in the extracted title
        e = _entry(
            title="Analytic Combinatorics. Cambridge University Press, Cambridge, England",
        )
        assert likely_book_publisher(e) == "Cambridge University Press"

    def test_springer_verlag(self):
        e = _entry(raw_fields={"publisher": "Springer-Verlag"})
        assert likely_book_publisher(e) == "Springer-Verlag"

    def test_mit_press(self):
        e = _entry(raw_fields={"publisher": "MIT Press"})
        assert likely_book_publisher(e) == "MIT Press"

    def test_no_publisher_signals(self):
        e = _entry(
            entry_type="article",
            title="Learning-augmented algorithms for online scheduling",
            raw_fields={},
        )
        assert likely_book_publisher(e) is None

    def test_publisher_field_takes_priority_over_title(self):
        # Both match; publisher field is checked first
        e = _entry(
            title="Some title with MIT Press in it",
            raw_fields={"publisher": "Cambridge University Press"},
        )
        assert likely_book_publisher(e) == "Cambridge University Press"
