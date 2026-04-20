"""Async orchestrator — coordinates verification of all bib entries."""

from __future__ import annotations

import asyncio
import logging
from typing import Callable, Optional

from .classifier import is_grey_literature, is_web_verifiable
from .fuzzy import compare_records
from .http_client import CitationHttpClient, CitationHttpError
from .models import (
    BibEntry,
    RemoteRecord,
    VerificationResult,
    VerificationStatus,
    VerificationStrategy,
)
from .verifiers import arxiv, crossref, openalex, semanticscholar, web

log = logging.getLogger(__name__)


async def run_checks(
    entries: list[BibEntry],
    client: CitationHttpClient,
    check_urls: bool = True,
    openalex_key: Optional[str] = None,
    concurrency: int = 10,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> list[VerificationResult]:
    """Verify all entries concurrently.

    Returns results in the same order as the input list.
    """
    sem = asyncio.Semaphore(concurrency)
    total = len(entries)
    completed = 0

    async def _bounded(entry: BibEntry) -> VerificationResult:
        nonlocal completed
        async with sem:
            result = await check_entry(entry, client, check_urls, openalex_key)
        completed += 1
        if progress_callback:
            progress_callback(completed, total)
        return result

    tasks = [asyncio.create_task(_bounded(e)) for e in entries]
    results = await asyncio.gather(*tasks)
    return list(results)


async def check_entry(
    entry: BibEntry,
    client: CitationHttpClient,
    check_urls: bool,
    openalex_key: Optional[str],
) -> VerificationResult:
    """Verify a single BibEntry through the prioritised strategy chain."""
    try:
        return await _check(entry, client, check_urls, openalex_key)
    except Exception as exc:
        log.exception("Unexpected error verifying %s", entry.key)
        return VerificationResult(
            entry_key=entry.key,
            status=VerificationStatus.ERROR,
            strategy=VerificationStrategy.NONE,
            remote_record=None,
            scores=None,
            url_reachable=None,
            error_message=str(exc),
        )


async def _check(
    entry: BibEntry,
    client: CitationHttpClient,
    check_urls: bool,
    openalex_key: Optional[str],
) -> VerificationResult:
    warnings: list[str] = []

    # Guard: not enough metadata to search
    if not entry.title and not entry.authors:
        return VerificationResult(
            entry_key=entry.key,
            status=VerificationStatus.UNVERIFIABLE,
            strategy=VerificationStrategy.NONE,
            remote_record=None,
            scores=None,
            url_reachable=None,
            error_message=None,
            warnings=["No title or authors — cannot verify"],
        )

    remote: Optional[RemoteRecord] = None
    strategy = VerificationStrategy.NONE

    # ------------------------------------------------------------------ #
    # Strategy A: DOI lookup via CrossRef                                 #
    # ------------------------------------------------------------------ #
    if entry.doi:
        strategy = VerificationStrategy.DOI_CROSSREF
        try:
            remote = await crossref.lookup_by_doi(entry.doi, client)
        except CitationHttpError as exc:
            return _error_result(entry.key, strategy, exc)

        if remote is None:
            # DOI returned 404 — fall through to title search
            warnings.append(f"DOI {entry.doi!r} not found in CrossRef; trying title search")
            strategy = VerificationStrategy.NONE
            remote = await _title_search(entry, client, openalex_key, warnings)
            strategy = _last_title_strategy(remote, warnings)

    # ------------------------------------------------------------------ #
    # Strategy B: arXiv eprint lookup                                     #
    # ------------------------------------------------------------------ #
    elif _is_arxiv(entry):
        strategy = VerificationStrategy.ARXIV
        try:
            remote = await arxiv.lookup_by_eprint(entry.eprint, client)  # type: ignore[arg-type]
        except CitationHttpError as exc:
            return _error_result(entry.key, strategy, exc)

        if remote is None:
            warnings.append(f"arXiv eprint {entry.eprint!r} not found; trying title search")
            strategy = VerificationStrategy.NONE
            remote = await _title_search(entry, client, openalex_key, warnings)
            strategy = _last_title_strategy(remote, warnings)

    # ------------------------------------------------------------------ #
    # Strategy C: title + author search                                   #
    # ------------------------------------------------------------------ #
    elif entry.title:
        remote = await _title_search(entry, client, openalex_key, warnings)
        strategy = _last_title_strategy(remote, warnings)

    # ------------------------------------------------------------------ #
    # Strategy D: web title verification for news / media URLs            #
    # ------------------------------------------------------------------ #
    # Only attempted when all academic strategies have failed and the     #
    # entry has a URL on a known news/media domain.                       #
    url_reachable: Optional[bool] = None
    if remote is None and is_web_verifiable(entry):
        strategy = VerificationStrategy.URL_WEB
        web_record = await web.lookup_by_url(entry.url, client)  # type: ignore[arg-type]
        if web_record is not None:
            remote = web_record
            url_reachable = True  # we successfully fetched the page

    # Supplementary: URL reachability check (only if we haven't already
    # fetched the page via web verification above).
    if url_reachable is None and check_urls and entry.url:
        status_code = await client.head_url(entry.url)
        url_reachable = status_code is not None and status_code < 400

    # ------------------------------------------------------------------ #
    # Determine final status                                              #
    # ------------------------------------------------------------------ #
    if remote is None:
        if is_grey_literature(entry):
            warnings.append(
                "Likely software, dataset, or web resource — not expected in academic databases"
            )
            status = VerificationStatus.GREY_LITERATURE
        else:
            status = VerificationStatus.NOT_FOUND
        return VerificationResult(
            entry_key=entry.key,
            status=status,
            strategy=strategy,
            remote_record=None,
            scores=None,
            url_reachable=url_reachable,
            error_message=None,
            warnings=warnings,
        )

    # Warn if author list is drastically truncated in bib file
    if entry.authors and remote.authors:
        if len(remote.authors) > len(entry.authors) * 2 and len(remote.authors) > 3:
            warnings.append(
                f"Only {len(entry.authors)} of {len(remote.authors)} authors listed in bib entry"
            )

    scores, status, fuzzy_warnings = compare_records(
        entry, remote, skip_author_year=(remote.source == "web")
    )
    warnings.extend(fuzzy_warnings)

    return VerificationResult(
        entry_key=entry.key,
        status=status,
        strategy=strategy,
        remote_record=remote,
        scores=scores,
        url_reachable=url_reachable,
        error_message=None,
        warnings=warnings,
    )


async def _title_search(
    entry: BibEntry,
    client: CitationHttpClient,
    openalex_key: Optional[str],
    warnings: list[str],
) -> Optional[RemoteRecord]:
    """Try CrossRef search, then OpenAlex search."""
    if not entry.title:
        return None

    try:
        remote = await crossref.search_by_title_author(entry.title, entry.authors, client)
    except CitationHttpError as exc:
        log.debug("CrossRef search error for %s: %s", entry.key, exc)
        remote = None

    if remote is not None:
        return remote

    # Fallback to OpenAlex
    try:
        remote = await openalex.search_by_title(entry.title, client, openalex_key)
    except CitationHttpError as exc:
        log.debug("OpenAlex search error for %s: %s", entry.key, exc)

    if remote is not None:
        return remote

    # Final fallback: Semantic Scholar (strong coverage of PMLR / ML venues).
    # Skip for grey-literature entries — their generic titles produce false positives.
    if not is_grey_literature(entry):
        try:
            remote = await semanticscholar.search_by_title_author(
                entry.title, entry.authors, client
            )
        except CitationHttpError as exc:
            log.debug("Semantic Scholar search error for %s: %s", entry.key, exc)

    return remote


def _last_title_strategy(
    remote: Optional[RemoteRecord], warnings: list[str]
) -> VerificationStrategy:
    if remote is None:
        return VerificationStrategy.CROSSREF_SEARCH
    if remote.source == "openalex":
        return VerificationStrategy.OPENALEX_SEARCH
    if remote.source == "semanticscholar":
        return VerificationStrategy.SEMANTICSCHOLAR
    return VerificationStrategy.CROSSREF_SEARCH


def _is_arxiv(entry: BibEntry) -> bool:
    return bool(
        entry.eprint
        and entry.archiveprefix
        and entry.archiveprefix.lower() == "arxiv"
    )


def _error_result(
    key: str, strategy: VerificationStrategy, exc: CitationHttpError
) -> VerificationResult:
    return VerificationResult(
        entry_key=key,
        status=VerificationStatus.ERROR,
        strategy=strategy,
        remote_record=None,
        scores=None,
        url_reachable=None,
        error_message=str(exc),
    )
