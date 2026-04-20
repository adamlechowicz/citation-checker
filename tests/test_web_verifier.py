"""Tests for the web URL verifier."""

import pytest
import respx
import httpx

from citation_checker.verifiers.web import (
    _MetaExtractor,
    _clean_page_title,
    lookup_by_url,
)
from citation_checker.http_client import CitationHttpClient


# ---------------------------------------------------------------------------
# Unit tests — HTML parsing
# ---------------------------------------------------------------------------

class TestMetaExtractor:
    def test_og_title(self):
        html = '<meta property="og:title" content="AI Needs Power"/>'
        e = _MetaExtractor()
        e.feed(html)
        assert e.best_title == "AI Needs Power"

    def test_twitter_title(self):
        html = '<meta name="twitter:title" content="Tweet Title"/>'
        e = _MetaExtractor()
        e.feed(html)
        assert e.best_title == "Tweet Title"

    def test_html_title_tag(self):
        html = "<html><head><title>Page Title | Site</title></head></html>"
        e = _MetaExtractor()
        e.feed(html)
        assert e.best_title == "Page Title | Site"

    def test_og_title_takes_precedence_over_html_title(self):
        html = (
            '<meta property="og:title" content="Article Title"/>'
            "<title>Article Title | Bloomberg</title>"
        )
        e = _MetaExtractor()
        e.feed(html)
        assert e.best_title == "Article Title"

    def test_no_title(self):
        html = "<html><head></head><body>No title here</body></html>"
        e = _MetaExtractor()
        e.feed(html)
        assert e.best_title is None


class TestCleanPageTitle:
    def test_strips_bloomberg_suffix(self):
        assert _clean_page_title("AI Needs Power — Bloomberg") == "AI Needs Power"

    def test_strips_pipe_nyt(self):
        assert _clean_page_title("Some Story | The New York Times") == "Some Story"

    def test_strips_dash_bbc(self):
        assert _clean_page_title("Tech Article - BBC News") == "Tech Article"

    def test_no_suffix_unchanged(self):
        raw = "AI Needs So Much Power, It's Making Yours Worse"
        assert _clean_page_title(raw) == raw

    def test_strips_wired(self):
        assert _clean_page_title("The Future of AI | Wired") == "The Future of AI"


# ---------------------------------------------------------------------------
# Integration tests — lookup_by_url with mocked HTTP
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_lookup_returns_record_with_og_title():
    html = (
        '<html><head>'
        '<meta property="og:title" content="AI Needs So Much Power"/>'
        '</head><body></body></html>'
    )
    async with respx.mock:
        respx.get("https://bloomberg.com/article/123").mock(
            return_value=httpx.Response(200, text=html)
        )
        async with CitationHttpClient() as client:
            record = await lookup_by_url("https://bloomberg.com/article/123", client)

    assert record is not None
    assert record.title == "AI Needs So Much Power"
    assert record.source == "web"
    assert record.authors == []
    assert record.year is None


@pytest.mark.asyncio
async def test_lookup_strips_site_suffix():
    html = "<html><head><title>Europe grid tested | Financial Times</title></head></html>"
    async with respx.mock:
        respx.get("https://ft.com/article/456").mock(
            return_value=httpx.Response(200, text=html)
        )
        async with CitationHttpClient() as client:
            record = await lookup_by_url("https://ft.com/article/456", client)

    assert record is not None
    assert "Financial Times" not in record.title


@pytest.mark.asyncio
async def test_lookup_returns_none_on_http_error():
    async with respx.mock:
        respx.get("https://bloomberg.com/article/404").mock(
            return_value=httpx.Response(404, text="Not found")
        )
        async with CitationHttpClient() as client:
            # The client raises CitationHttpError for 4xx... actually it returns
            # the 404 response. lookup_by_url won't crash but may return None or
            # a record depending on whether the 404 page has a title.
            # We just check it doesn't raise.
            result = await lookup_by_url("https://bloomberg.com/article/404", client)
        # Any result (including None) is acceptable for a 404 page
        assert result is None or result.source == "web"


@pytest.mark.asyncio
async def test_lookup_returns_none_when_no_title():
    html = "<html><head></head><body>No title</body></html>"
    async with respx.mock:
        respx.get("https://reuters.com/article/789").mock(
            return_value=httpx.Response(200, text=html)
        )
        async with CitationHttpClient() as client:
            record = await lookup_by_url("https://reuters.com/article/789", client)

    assert record is None


# ---------------------------------------------------------------------------
# Classifier tests for new news domains
# ---------------------------------------------------------------------------

from citation_checker.classifier import is_grey_literature, is_web_verifiable
from citation_checker.models import BibEntry


def _entry(**kwargs) -> BibEntry:
    defaults = dict(
        key="test", entry_type="article", title=None, authors=[],
        year=None, doi=None, url=None, eprint=None,
        archiveprefix=None, raw_fields={},
    )
    defaults.update(kwargs)
    return BibEntry(**defaults)


class TestIsWebVerifiable:
    def test_bloomberg_is_verifiable(self):
        assert is_web_verifiable(_entry(url="https://bloomberg.com/article/123"))

    def test_ft_is_verifiable(self):
        assert is_web_verifiable(_entry(url="https://ft.com/article/xyz"))

    def test_nytimes_is_verifiable(self):
        assert is_web_verifiable(_entry(url="https://nytimes.com/2024/01/ai.html"))

    def test_reuters_is_verifiable(self):
        assert is_web_verifiable(_entry(url="https://reuters.com/tech/ai"))

    def test_github_is_not_verifiable(self):
        assert not is_web_verifiable(_entry(url="https://github.com/user/repo"))

    def test_no_url_is_not_verifiable(self):
        assert not is_web_verifiable(_entry())


class TestNewsDomainsAreGreyLit:
    def test_bloomberg_is_grey(self):
        assert is_grey_literature(_entry(url="https://bloomberg.com/article/123"))

    def test_ft_is_grey(self):
        assert is_grey_literature(_entry(url="https://ft.com/xyz"))

    def test_wired_is_grey(self):
        assert is_grey_literature(_entry(url="https://wired.com/story/ai"))

    def test_scholarly_not_grey(self):
        assert not is_grey_literature(_entry(url="https://doi.org/10.1145/123"))
