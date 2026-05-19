"""CLI entry point for citation-checker."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from .checker import run_checks
from .fuzzy import TITLE_THRESHOLD, AUTHOR_THRESHOLD
from .http_client import CitationHttpClient
from .models import VerificationStatus
from .parser import parse_bib_file
from .pdf_parser import parse_pdf_file
from .reporter import print_table, print_summary, write_json_report

console = Console()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="citation-checker",
        description="Deterministic bibliography verifier for .bib files.",
    )
    p.add_argument("bib_file", help="Path to the .bib or .pdf file to check")
    p.add_argument(
        "--output", "-o", metavar="PATH",
        help="Write JSON report to this file",
    )
    p.add_argument(
        "--mailto", metavar="EMAIL",
        help="Your email for the CrossRef polite pool User-Agent (strongly recommended)",
    )
    p.add_argument(
        "--openalex-key", metavar="KEY",
        help="OpenAlex API key for higher rate limits",
    )
    p.add_argument(
        "--timeout", type=float, default=10.0, metavar="SECS",
        help="Per-request HTTP timeout in seconds (default: 10)",
    )
    p.add_argument(
        "--retries", type=int, default=3, metavar="N",
        help="Max retries per request (default: 3)",
    )
    p.add_argument(
        "--concurrency", type=int, default=10, metavar="N",
        help="Max simultaneous entry checks (default: 10)",
    )
    url_group = p.add_mutually_exclusive_group()
    url_group.add_argument(
        "--check-urls", dest="check_urls", action="store_true", default=True,
        help="Perform supplementary URL HEAD checks (default: on)",
    )
    url_group.add_argument(
        "--no-check-urls", dest="check_urls", action="store_false",
        help="Disable URL checks",
    )
    p.add_argument(
        "--allow-local-urls", action="store_true",
        help="Allow URL checks to hit private / loopback / link-local IPs "
             "(off by default to avoid SSRF on hostile .bib files)",
    )
    p.add_argument(
        "--show-scores", action="store_true",
        help="Show title/author fuzzy scores in the table",
    )
    p.add_argument(
        "--show-remote", action="store_true",
        help="Show the title and authors returned by the database for each matched entry",
    )
    p.add_argument(
        "--filter-keys", nargs="+", metavar="KEY",
        help="Only check these specific cite keys",
    )
    p.add_argument(
        "--filter-status", nargs="+", metavar="STATUS",
        choices=[s.value for s in VerificationStatus],
        help="Only display entries with these statuses in the table",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="Suppress table output; only print summary",
    )
    p.add_argument(
        "--json-only", action="store_true",
        help="Suppress all terminal output; only write JSON (requires --output)",
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Enable debug logging to stderr",
    )
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate
    if args.json_only and not args.output:
        console.print("[bold red]Error:[/bold red] --json-only requires --output", style="red")
        sys.exit(3)

    # Logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        stream=sys.stderr,
        level=log_level,
        format="%(levelname)s %(name)s: %(message)s",
    )

    # Warn if no mailto
    if not args.mailto and not args.json_only:
        console.print(
            "[yellow]Warning:[/yellow] No --mailto provided. CrossRef polite pool "
            "gives higher rate limits when you identify yourself. "
            "Consider adding --mailto your@email.com"
        )

    # Parse bib file
    bib_path = Path(args.bib_file)
    if not bib_path.exists():
        console.print(f"[bold red]Error:[/bold red] File not found: {bib_path}")
        sys.exit(2)

    try:
        if bib_path.suffix.lower() == ".pdf":
            entries = parse_pdf_file(str(bib_path))
        else:
            entries = parse_bib_file(str(bib_path))
    except Exception as exc:
        console.print(f"[bold red]Parse error:[/bold red] {exc}")
        sys.exit(2)

    # Apply key filter
    if args.filter_keys:
        key_set = set(args.filter_keys)
        entries = [e for e in entries if e.key in key_set]
        if not entries:
            console.print(f"[yellow]No entries matched --filter-keys {args.filter_keys}[/yellow]")
            sys.exit(2)

    entries_by_key = {e.key: e for e in entries}

    if not args.json_only:
        console.print(
            f"[bold]Checking [cyan]{len(entries)}[/cyan] entries from "
            f"[cyan]{bib_path.name}[/cyan]...[/bold]"
        )

    # Run checks with progress bar
    start = time.monotonic()
    results = asyncio.run(
        _run_with_progress(args, entries, bib_path)
    )
    elapsed = time.monotonic() - start

    # Output
    filter_status: Optional[list[VerificationStatus]] = None
    if args.filter_status:
        filter_status = [VerificationStatus(s) for s in args.filter_status]

    if not args.json_only:
        if not args.quiet:
            print_table(results, entries_by_key, show_scores=args.show_scores, show_remote=args.show_remote, filter_status=filter_status)
        print_summary(results)
        console.print(f"[dim]Completed in {elapsed:.1f}s[/dim]")

    if args.output:
        write_json_report(
            results=results,
            entries_by_key=entries_by_key,
            output_path=args.output,
            elapsed_seconds=elapsed,
            bib_file=str(bib_path.resolve()),
            title_threshold=TITLE_THRESHOLD,
            author_threshold=AUTHOR_THRESHOLD,
        )
        if not args.json_only:
            console.print(f"[dim]JSON report written to {args.output}[/dim]")

    # Exit code
    bad = {VerificationStatus.MISMATCH, VerificationStatus.ERROR}
    if any(r.status in bad for r in results):
        sys.exit(1)
    sys.exit(0)


async def _run_with_progress(args, entries, bib_path) -> list:
    total = len(entries)

    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=Console(stderr=True),
        transient=True,
        disable=args.json_only,
    ) as progress:
        task_id = progress.add_task("Verifying...", total=total)

        def on_progress(completed: int, _total: int) -> None:
            progress.update(task_id, completed=completed)

        async with CitationHttpClient(
            timeout=args.timeout,
            max_retries=args.retries,
            mailto=args.mailto,
            allow_local_urls=args.allow_local_urls,
        ) as client:
            return await run_checks(
                entries=entries,
                client=client,
                check_urls=args.check_urls,
                openalex_key=args.openalex_key,
                concurrency=args.concurrency,
                progress_callback=on_progress,
            )


if __name__ == "__main__":
    main()
