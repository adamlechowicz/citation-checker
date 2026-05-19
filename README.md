# citation-checker

A deterministic bibliography verifier for `.bib` files and PDFs. Checks whether cited works exist in authoritative academic databases and whether core metadata (title, authors, year) actually matches what you cited — no LLMs involved.

## How It Works

Each bibliography entry is verified through a priority-ordered strategy chain:

1. **DOI → CrossRef** — direct lookup by DOI; the most reliable path
2. **arXiv eprint → arXiv Atom XML API** — for preprints with an `eprint` field
3. **Title + author → CrossRef bibliographic search** — for entries without a DOI
4. **Title → OpenAlex** — fallback when CrossRef search finds nothing
5. **Title + author → Semantic Scholar** — additional fallback with strong ML/CS venue coverage (PMLR, NeurIPS, ICML, etc.)
6. **URL → web title extraction** — for news and media sources (Bloomberg, NYT, Reuters, etc.)

All entries are checked concurrently with per-host rate limiting to stay within API guidelines.

## Installation

Requires Python ≥ 3.11.

```bash
pip install -e .
```

Dependencies: `bibtexparser`, `pymupdf`, `httpx`, `rapidfuzz`, `rich`

## Quick Start

```bash
# Check a .bib file
citation-checker refs.bib --mailto you@email.com

# Check a PDF's bibliography
citation-checker paper.pdf --mailto you@email.com

# Save a JSON report and show fuzzy match scores
citation-checker refs.bib --mailto you@email.com --output report.json --show-scores

# Show what the database actually found (DB title + authors columns)
citation-checker refs.bib --show-remote

# Only show problems
citation-checker refs.bib --filter-status MISMATCH NOT_FOUND ERROR

# Check specific cite keys
citation-checker refs.bib --filter-keys Vaswani17 LeCun89
```

> **Tip:** Pass `--mailto your@email.com` to join the CrossRef polite pool and get higher rate limits.

## CLI Reference

| Flag | Default | Description |
|------|---------|-------------|
| `bib_file` | required | Path to a `.bib` or `.pdf` file |
| `--output, -o PATH` | — | Write a JSON report to this file |
| `--mailto EMAIL` | — | Your email for the CrossRef polite pool |
| `--openalex-key KEY` | — | OpenAlex API key for higher rate limits |
| `--timeout SECS` | `10.0` | Per-request HTTP timeout |
| `--retries N` | `3` | Max retries per request |
| `--concurrency N` | `10` | Max simultaneous entry checks |
| `--no-check-urls` | off | Disable supplementary URL reachability checks |
| `--show-scores` | off | Show title/author fuzzy scores in the table |
| `--show-remote` | off | Show the title and authors returned by the matched database |
| `--filter-keys KEY...` | — | Only check these cite keys |
| `--filter-status S...` | — | Only display entries with these statuses |
| `--quiet` | off | Print summary only; suppress table |
| `--json-only` | off | No terminal output; write JSON only (requires `--output`) |
| `--verbose` | off | Enable debug logging to stderr |

**Exit codes:** `0` = all OK · `1` = MISMATCH or ERROR found · `2` = parse/file error · `3` = config error

## Verification Statuses

| Status | Meaning |
|--------|---------|
| `VERIFIED` | Found in an external database; title, authors, and year match |
| `MISMATCH` | Found, but one or more core fields differ significantly |
| `NOT_FOUND` | Not found in any database |
| `GREY_LITERATURE` | Software, dataset, government report, or news article — not expected in academic DBs |
| `UNVERIFIABLE` | Too little metadata (no title or authors) to search with |
| `ERROR` | Network or API failure for this entry |

## Fuzzy Matching

Field comparison uses [RapidFuzz](https://github.com/rapidfuzz/RapidFuzz):

- **Title**: `fuzz.ratio ≥ 85` after NFKD normalization and LaTeX artifact stripping
- **Authors**: per-author best-match `token_sort_ratio ≥ 80` — handles "Last, First" vs "First Last" ordering; abbreviated first names (e.g., "R. Smith" vs "Robert Smith") are tolerated as soft warnings
- **Year**: exact integer match; a mismatch always forces `MISMATCH` regardless of other scores

Author scores between 55 and 80 produce a warning but do not by themselves trigger `MISMATCH`.

## PDF Input

When given a `.pdf` file, citation-checker extracts the bibliography section using PyMuPDF and auto-detects the reference list format:

- **Numbered** (`[1] Author, A. Title. Venue, year.`) — brackets or parenthesised numbers
- **Author–year** (`Surname, A. (year). Title. Venue.`) — common in economics and some CS venues

Cite keys are derived from the **first author's surname + year** (e.g., `Vaswani2017`, `LeCun1989`). When two entries share the same base key a letter suffix is appended to the second and later occurrences (`Chin2015`, `Chin2015a`).

Extracted entries go through the same verification pipeline as `.bib` entries. No DOI or arXiv eprint is assumed unless one is found in the text.

## Grey Literature

Entries on code/data hosting sites (GitHub, Zenodo, Hugging Face), government and national lab sites (`nrel.gov`, `epa.gov`, `eia.gov`, etc.), or corporate technical resources are automatically classified as `GREY_LITERATURE` and skipped in academic database searches — they are not expected to appear in CrossRef or Semantic Scholar.

Entries whose URL points to a supported news or media domain (Bloomberg, Financial Times, NYT, Reuters, WSJ, The Guardian, BBC, Wired, MIT Technology Review, and more) are verified by fetching the page and comparing the article title. A match counts as `VERIFIED`.

## JSON Report

```json
{
  "meta": {
    "tool_version": "1.0.0",
    "generated_at": "...",
    "bib_file": "refs.bib",
    "total_entries": 218,
    "elapsed_seconds": 61.4,
    "thresholds": { "title_score": 85.0, "author_score": 80.0 },
    "counts": { "VERIFIED": 190, "MISMATCH": 5, "NOT_FOUND": 8, ... }
  },
  "results": [
    {
      "cite_key": "Vaswani17",
      "entry_type": "article",
      "status": "VERIFIED",
      "strategy": "doi_crossref",
      "local":  { "title": "...", "authors": [...], "year": 2017, "doi": "...", "url": null, "eprint": null },
      "remote": { "title": "...", "authors": [...], "year": 2017, "source": "crossref" },
      "scores": { "title_score": 100.0, "author_score": 95.3, "year_match": true },
      "url_reachable": null,
      "error_message": null,
      "warnings": []
    }
  ]
}
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest tests/ -v             # fast unit suite
pytest -m slow -v            # real-PDF corpus regression suite
```

Unit tests use [respx](https://github.com/lundberg/respx) to mock all HTTP calls — no real API requests are made during testing.

The `slow`-marked suite (`tests/test_pdf_parser_real.py`) runs the PDF parser
against a corpus of ~30 real open-access papers in `tests/fixtures/pdfs/`
spanning every supported citation style plus intentionally out-of-distribution
samples (Bluebook legal, headings-absent layouts) that document known gaps.
