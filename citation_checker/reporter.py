"""Rich CLI table renderer and JSON report writer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich import box
from rich.text import Text

from .fuzzy import AUTHOR_SOFT_THRESHOLD, TITLE_THRESHOLD
from .models import BibEntry, VerificationResult, VerificationStatus, VerificationStrategy
from .utils import truncate

_CONSOLE = Console()

_STATUS_STYLE: dict[VerificationStatus, str] = {
    VerificationStatus.VERIFIED:        "green",
    VerificationStatus.MISMATCH:        "yellow",
    VerificationStatus.NOT_FOUND:       "red",
    VerificationStatus.GREY_LITERATURE: "dim",
    VerificationStatus.UNVERIFIABLE:    "dim",
    VerificationStatus.ERROR:           "bold red",
}

_STATUS_ORDER = [
    VerificationStatus.MISMATCH,
    VerificationStatus.NOT_FOUND,
    VerificationStatus.ERROR,
    VerificationStatus.VERIFIED,
    VerificationStatus.GREY_LITERATURE,
    VerificationStatus.UNVERIFIABLE,
]

_STRATEGY_LABELS: dict[VerificationStrategy, str] = {
    VerificationStrategy.DOI_CROSSREF:    "DOI/CrossRef",
    VerificationStrategy.ARXIV:           "arXiv",
    VerificationStrategy.CROSSREF_SEARCH: "CrossRef search",
    VerificationStrategy.OPENALEX_SEARCH: "OpenAlex search",
    VerificationStrategy.SEMANTICSCHOLAR: "Semantic Scholar",
    VerificationStrategy.URL_WEB:         "URL/web title",
    VerificationStrategy.NONE:            "—",
}


def print_table(
    results: list[VerificationResult],
    entries_by_key: dict[str, BibEntry],
    show_scores: bool = False,
    show_remote: bool = False,
    filter_status: Optional[list[VerificationStatus]] = None,
) -> None:
    """Render a Rich table of verification results to stdout."""
    filtered = _filter_and_sort(results, filter_status)

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        expand=False,
    )

    table.add_column("Cite Key", style="cyan", no_wrap=True)
    table.add_column("Type", style="dim", no_wrap=True)
    table.add_column("Strategy", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    if show_scores:
        table.add_column("Title", justify="right", no_wrap=True)
        table.add_column("Author", justify="right", no_wrap=True)
        table.add_column("Year", justify="center", no_wrap=True)
    if show_remote:
        table.add_column("DB Title", max_width=50, overflow="ellipsis")
        table.add_column("DB Authors", max_width=40, overflow="ellipsis")
    table.add_column("URL", justify="center", no_wrap=True)
    table.add_column("Notes")

    for r in filtered:
        entry = entries_by_key.get(r.entry_key)
        status_text = Text(r.status.value, style=_STATUS_STYLE[r.status])
        strategy_label = _STRATEGY_LABELS.get(r.strategy, r.strategy.value)
        url_cell = _url_cell(r.url_reachable)
        notes = _notes(r)
        entry_type = entry.entry_type if entry else "?"

        row: list = [
            r.entry_key,
            entry_type,
            strategy_label,
            status_text,
        ]

        if show_scores:
            if r.scores:
                row.append(f"{r.scores.title_score:.0f}")
                if r.scores.author_score is None:
                    row.append("—")
                else:
                    row.append(f"{r.scores.author_score:.0f}")
                year_sym = {True: "✓", False: "✗", None: "—"}.get(r.scores.year_match, "—")
                row.append(year_sym)
            else:
                row += ["—", "—", "—"]

        if show_remote:
            if r.remote_record:
                db_title = r.remote_record.title or "—"
                db_authors = ", ".join(r.remote_record.authors) if r.remote_record.authors else "—"
            else:
                db_title = "—"
                db_authors = "—"
            row.append(db_title)
            row.append(db_authors)

        row.append(url_cell)
        row.append(notes)
        table.add_row(*row)

    _CONSOLE.print(table)


def print_summary(results: list[VerificationResult]) -> None:
    """Print a one-line summary of counts."""
    counts = _count_by_status(results)
    total = len(results)
    parts = []
    for status in _STATUS_ORDER:
        n = counts[status]
        style = _STATUS_STYLE[status]
        parts.append(f"[{style}]{n} {status.value}[/{style}]")

    _CONSOLE.print(f"\n[bold]{total} entries checked:[/bold]  " + "  ".join(parts))


def write_json_report(
    results: list[VerificationResult],
    entries_by_key: dict[str, BibEntry],
    output_path: str,
    elapsed_seconds: float,
    bib_file: str,
    title_threshold: float,
    author_threshold: float,
) -> None:
    """Serialize results to a JSON file."""
    counts = {status.value: count for status, count in _count_by_status(results).items()}

    report = {
        "meta": {
            "tool_version": "1.0.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "bib_file": bib_file,
            "total_entries": len(results),
            "elapsed_seconds": round(elapsed_seconds, 2),
            "thresholds": {
                "title_score": title_threshold,
                "author_score": author_threshold,
            },
            "counts": counts,
        },
        "results": [
            _result_to_dict(r, entries_by_key.get(r.entry_key))
            for r in results
        ],
    }

    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------- #
# Helpers                                                                 #
# ---------------------------------------------------------------------- #

def _filter_and_sort(
    results: list[VerificationResult],
    filter_status: Optional[list[VerificationStatus]],
) -> list[VerificationResult]:
    if filter_status:
        results = [r for r in results if r.status in filter_status]
    return sorted(results, key=lambda r: _STATUS_ORDER.index(r.status))


def _count_by_status(results: list[VerificationResult]) -> dict[VerificationStatus, int]:
    counts = {status: 0 for status in VerificationStatus}
    for result in results:
        counts[result.status] += 1
    return counts


def _url_cell(url_reachable: Optional[bool]) -> str:
    if url_reachable is True:
        return "[green]✓[/green]"
    if url_reachable is False:
        return "[red]✗[/red]"
    return "—"


def _notes(r: VerificationResult) -> str:
    parts: list[str] = []
    if r.error_message:
        parts.append(truncate(r.error_message, 80))
    if r.warnings:
        parts.extend(r.warnings)
    if r.scores and r.status == VerificationStatus.MISMATCH:
        if r.scores.title_score < TITLE_THRESHOLD:
            parts.append(f"title score {r.scores.title_score:.0f}")
        if r.scores.author_score is not None and r.scores.author_score < AUTHOR_SOFT_THRESHOLD:
            parts.append(f"author score {r.scores.author_score:.0f}")
    return "; ".join(parts)


def _result_to_dict(r: VerificationResult, entry: Optional[BibEntry]) -> dict:
    local: dict = {}
    if entry:
        local = {
            "title": entry.title,
            "authors": entry.authors,
            "year": entry.year,
            "doi": entry.doi,
            "url": entry.url,
            "eprint": entry.eprint,
        }

    remote = None
    if r.remote_record:
        rec = r.remote_record
        remote = {
            "title": rec.title,
            "authors": rec.authors,
            "year": rec.year,
            "source": rec.source,
        }

    scores = None
    if r.scores:
        author_score = (
            None if r.scores.author_score is None
            else round(r.scores.author_score, 2)
        )
        scores = {
            "title_score": round(r.scores.title_score, 2),
            "author_score": author_score,
            "year_match": r.scores.year_match,
        }

    return {
        "cite_key": r.entry_key,
        "entry_type": entry.entry_type if entry else None,
        "status": r.status.value,
        "strategy": r.strategy.value,
        "local": local,
        "remote": remote,
        "scores": scores,
        "url_reachable": r.url_reachable,
        "error_message": r.error_message,
        "warnings": r.warnings,
    }
