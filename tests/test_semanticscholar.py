"""Tests for the Semantic Scholar verifier."""

import pytest
import respx
import httpx

from citation_checker.verifiers.semanticscholar import search_by_title_author
from citation_checker.http_client import CitationHttpClient

_BASE = "https://api.semanticscholar.org/graph/v1/paper/search"


@pytest.mark.asyncio
async def test_returns_record_on_match():
    payload = {
        "data": [{
            "title": "Optimal robustness-consistency tradeoffs for learning-augmented metrical task systems",
            "authors": [{"name": "Nicolas Christianson"}, {"name": "Junxuan Shen"}, {"name": "A. Wierman"}],
            "year": 2023,
            "venue": "AISTATS",
        }]
    }
    async with respx.mock:
        respx.get(_BASE).mock(return_value=httpx.Response(200, json=payload))
        async with CitationHttpClient() as client:
            record = await search_by_title_author(
                "Optimal robustness-consistency tradeoffs for learning-augmented metrical task systems",
                ["Nicolas Christianson", "Junxuan Shen", "Adam Wierman"],
                client,
            )

    assert record is not None
    assert record.source == "semanticscholar"
    assert "metrical task systems" in record.title.lower()
    assert record.year == 2023
    assert any("Christianson" in a for a in record.authors)


@pytest.mark.asyncio
async def test_returns_none_on_empty_results():
    async with respx.mock:
        respx.get(_BASE).mock(return_value=httpx.Response(200, json={"data": []}))
        async with CitationHttpClient() as client:
            record = await search_by_title_author("Nonexistent Paper Title", [], client)
    assert record is None


@pytest.mark.asyncio
async def test_includes_first_author_in_query():
    """Verify the query includes the first author's last name."""
    captured = {}

    async with respx.mock:
        def capture(request):
            captured["query"] = request.url.params.get("query", "")
            return httpx.Response(200, json={"data": []})
        respx.get(_BASE).mock(side_effect=capture)
        async with CitationHttpClient() as client:
            await search_by_title_author("Some Title", ["Adam Wierman"], client)

    assert "Wierman" in captured["query"]
    assert "Some Title" in captured["query"]
