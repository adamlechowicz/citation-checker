"""Tests for API verifiers using respx mock transport."""

import pytest
import httpx
import respx
import json

from citation_checker.http_client import CitationHttpClient
from citation_checker.verifiers import crossref, arxiv, openalex


CROSSREF_DOI_RESPONSE = {
    "status": "ok",
    "message": {
        "title": ["Gradient-Based Learning Applied to Document Recognition"],
        "author": [
            {"given": "Yann", "family": "LeCun"},
            {"given": "Leon", "family": "Bottou"},
            {"given": "Yoshua", "family": "Bengio"},
            {"given": "Patrick", "family": "Haffner"},
        ],
        "published-print": {"date-parts": [[1998, 11]]},
    },
}

CROSSREF_SEARCH_RESPONSE = {
    "status": "ok",
    "message": {
        "items": [
            {
                "title": ["Gradient-Based Learning Applied to Document Recognition"],
                "author": [{"given": "Yann", "family": "LeCun"}],
                "published-print": {"date-parts": [[1998]]},
            }
        ]
    },
}

ARXIV_ATOM_RESPONSE = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <opensearch:totalResults>1</opensearch:totalResults>
  <entry>
    <title>Attention Is All You Need</title>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <published>2017-06-12T00:00:00Z</published>
  </entry>
</feed>
"""

OPENALEX_RESPONSE = {
    "results": [
        {
            "title": "Attention Is All You Need",
            "authorships": [
                {"author": {"display_name": "Ashish Vaswani"}},
            ],
            "publication_year": 2017,
        }
    ]
}


@pytest.fixture
def client():
    return CitationHttpClient(timeout=5.0, max_retries=1)


@pytest.mark.asyncio
async def test_crossref_doi_lookup_found(client):
    with respx.mock:
        respx.get("https://api.crossref.org/works/10.1109/5.726791").mock(
            return_value=httpx.Response(200, json=CROSSREF_DOI_RESPONSE)
        )
        async with client:
            record = await crossref.lookup_by_doi("10.1109/5.726791", client)

    assert record is not None
    assert "LeCun" in record.title or "Gradient" in record.title
    assert record.year == 1998
    assert len(record.authors) == 4


@pytest.mark.asyncio
async def test_crossref_doi_lookup_not_found(client):
    with respx.mock:
        respx.get("https://api.crossref.org/works/10.9999/fake").mock(
            return_value=httpx.Response(404, text="Not found")
        )
        async with client:
            record = await crossref.lookup_by_doi("10.9999/fake", client)

    assert record is None


@pytest.mark.asyncio
async def test_crossref_search(client):
    with respx.mock:
        respx.get("https://api.crossref.org/works").mock(
            return_value=httpx.Response(200, json=CROSSREF_SEARCH_RESPONSE)
        )
        async with client:
            record = await crossref.search_by_title_author(
                "Gradient-based learning applied to document recognition",
                ["LeCun, Yann"],
                client,
            )

    assert record is not None
    assert "Gradient" in record.title


@pytest.mark.asyncio
async def test_arxiv_lookup_found(client):
    with respx.mock:
        respx.get("http://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=ARXIV_ATOM_RESPONSE)
        )
        async with client:
            record = await arxiv.lookup_by_eprint("1706.03762", client)

    assert record is not None
    assert record.title == "Attention Is All You Need"
    assert record.year == 2017
    assert "Ashish Vaswani" in record.authors


@pytest.mark.asyncio
async def test_arxiv_lookup_not_found(client):
    empty_atom = """\
<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">
  <opensearch:totalResults>0</opensearch:totalResults>
</feed>
"""
    with respx.mock:
        respx.get("http://export.arxiv.org/api/query").mock(
            return_value=httpx.Response(200, text=empty_atom)
        )
        async with client:
            record = await arxiv.lookup_by_eprint("9999.99999", client)

    assert record is None


@pytest.mark.asyncio
async def test_openalex_search_found(client):
    with respx.mock:
        respx.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(200, json=OPENALEX_RESPONSE)
        )
        async with client:
            record = await openalex.search_by_title("Attention Is All You Need", client)

    assert record is not None
    assert "Attention" in record.title
    assert record.year == 2017


@pytest.mark.asyncio
async def test_openalex_search_no_results(client):
    with respx.mock:
        respx.get("https://api.openalex.org/works").mock(
            return_value=httpx.Response(200, json={"results": []})
        )
        async with client:
            record = await openalex.search_by_title("Nonexistent Paper Title XYZ", client)

    assert record is None
