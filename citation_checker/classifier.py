"""Heuristic classifier for non-scholarly references.

Detects software tools, datasets, websites, government reports, and other
grey literature that is not expected to appear in academic databases.
The classification is intentionally conservative: when in doubt, we return
False so that a genuine missing paper is never silently dismissed.
"""

from __future__ import annotations

import re
from urllib.parse import urlparse

from .models import BibEntry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Entry types that are inherently non-scholarly when they also lack a DOI/arXiv
_GREY_LIT_TYPES = frozenset({"misc", "online", "software"})

# URL domains that are clearly non-academic (grey literature, software, data)
_GREY_LIT_DOMAINS = frozenset({
    # Code hosting
    "github.com", "gitlab.com", "bitbucket.org",
    # Data repositories
    "zenodo.org", "figshare.com", "kaggle.com", "data.gov", "dataverse.org",
    "huggingface.co", "paperswithcode.com", "osf.io",
    # Package registries
    "pypi.org", "cran.r-project.org", "anaconda.org", "conda-forge.org",
    # Government / national labs
    "nrel.gov", "epa.gov", "eia.gov", "energy.gov", "doe.gov",
    "dot.gov", "bls.gov", "census.gov", "nasa.gov", "noaa.gov",
    # Corporate
    "gurobi.com", "cplex.ibm.com", "mosek.com",
    "aws.amazon.com", "cloud.google.com", "azure.microsoft.com",
    "openai.com", "anthropic.com",
    # Web resources frequently cited in technical papers
    "gridstatus.io", "electricitymap.org", "watttime.org",
    "chargepoint.com", "ladwp.com", "moxionpower.com",
    "goldmansachs.com", "waymo.com",
})

# News and media domains where URL verification is meaningful:
# if a citation URL points here and the page title matches, the citation
# is considered verified as a web source.
_NEWS_DOMAINS = frozenset({
    # Business / financial press
    "bloomberg.com", "ft.com", "wsj.com", "economist.com",
    "reuters.com", "apnews.com", "fortune.com", "forbes.com",
    "businessinsider.com", "cnbc.com",
    # General news
    "nytimes.com", "washingtonpost.com", "theguardian.com",
    "bbc.com", "bbc.co.uk", "cnn.com", "nbcnews.com",
    "theatlantic.com", "vox.com", "axios.com", "politico.com",
    "nationalgeographic.com", "npr.org",
    # Tech / science press
    "techcrunch.com", "wired.com", "arstechnica.com", "theverge.com",
    "technologyreview.com", "venturebeat.com",
    "scientificamerican.com", "newscientist.com", "sciencenews.org",
    # Energy / industry trade press
    "utilitydive.com", "greentechmedia.com", "spglobal.com",
    "powermag.com", "energymonitor.ai",
})

# URL domains that definitively indicate a scholarly source
_SCHOLARLY_DOMAINS = frozenset({
    "arxiv.org", "doi.org", "semanticscholar.org",
    "acm.org", "ieee.org", "springer.com", "springerlink.com",
    "elsevier.com", "sciencedirect.com", "nature.com", "science.org",
    "jstor.org", "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov",
    "dl.acm.org", "ieeexplore.ieee.org", "link.springer.com",
})

# Keywords in titles that strongly suggest software, datasets, or reports
_GREY_LIT_TITLE_RE = re.compile(
    r'\b(?:software|dataset|data\s+set|package|library|framework|'
    r'documentation|manual|handbook|technical\s+report|white\s*paper|'
    r'zenodo|github|gitlab|v\d+\.\d+|version\s+\d)\b',
    re.IGNORECASE,
)

# Patterns that suggest a single-entity (organization) author:
# all-uppercase abbreviation (e.g. "NREL", "EPA"), or no period/comma in a
# short name (typical of "Gurobi Optimization" style), or ends with
# corporate/gov suffixes.
_ORG_SUFFIX_RE = re.compile(
    r'\b(?:LLC|Inc\.?|Ltd\.?|Corp\.?|Co\.?|GmbH|Group|Institute|'
    r'Foundation|Agency|Department|Bureau|Office|Authority|Commission|'
    r'Administration|Laboratory|Center|Centre|University)\b',
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def is_web_verifiable(entry: BibEntry) -> bool:
    """Return True if this entry has a URL on a news/media domain.

    These entries can be verified by fetching the page and comparing the
    extracted title against the citation title.
    """
    if not entry.url:
        return False
    domain = _get_domain(entry.url)
    return any(domain == d or domain.endswith("." + d) for d in _NEWS_DOMAINS)


def is_grey_literature(entry: BibEntry) -> bool:
    """Return True if the entry looks like grey literature.

    Grey literature includes: software tools, datasets, websites, government
    reports, corporate white papers, and similar non-scholarly resources that
    are not expected to appear in CrossRef or OpenAlex.

    Conservative by design: returns False when uncertain.
    """
    # Entries with a DOI or arXiv ID are scholarly by definition.
    if entry.doi or entry.eprint:
        return False

    # --- Signal 1: entry type ---
    grey_type = entry.entry_type.lower() in _GREY_LIT_TYPES

    # --- Signal 2: URL domain ---
    url_signal = _url_is_grey(entry.url)
    url_scholarly = _url_is_scholarly(entry.url)

    # A scholarly-looking URL overrides everything.
    if url_scholarly:
        return False

    # --- Signal 3: author looks like an organization ---
    org_author = _authors_look_like_org(entry.authors)

    # --- Signal 4: title contains grey-lit keywords ---
    title_signal = bool(entry.title and _GREY_LIT_TITLE_RE.search(entry.title))

    # --- Signal 5: news/media domain (also grey lit, verified separately) ---
    news_domain = is_web_verifiable(entry)

    # Classification rules (ordered by confidence):
    #   - @online + any URL that isn't scholarly → grey literature by definition
    #   - Other grey-lit types + any positive signal → grey literature
    #   - Non-grey type + URL on known grey domain  → grey literature (e.g.
    #     a GitHub link as the primary reference for a dataset)
    #   - Otherwise → not classified
    if entry.entry_type.lower() == "online" and entry.url and not url_scholarly:
        return True

    if grey_type and (url_signal or org_author or title_signal):
        return True

    if not grey_type and url_signal and not url_scholarly:
        return True

    # News/media URLs are grey literature (verified separately via web fetch)
    if news_domain:
        return True

    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_domain(url: str) -> str:
    """Return the bare domain from a URL (no www. prefix)."""
    try:
        netloc = urlparse(url).netloc.lower()
        return netloc.removeprefix("www.")
    except Exception:
        return ""


def _url_is_grey(url: str | None) -> bool:
    if not url:
        return False
    domain = _get_domain(url)
    return any(domain == d or domain.endswith("." + d) for d in _GREY_LIT_DOMAINS)


def _url_is_scholarly(url: str | None) -> bool:
    if not url:
        return False
    domain = _get_domain(url)
    return any(domain == d or domain.endswith("." + d) for d in _SCHOLARLY_DOMAINS)


def _authors_look_like_org(authors: list[str]) -> bool:
    """Return True if the author list looks like a single organization."""
    if len(authors) != 1:
        return False
    name = authors[0].strip()
    # All-uppercase abbreviation (e.g. "NREL", "EPA", "AWS")
    if name.isupper() and len(name) <= 10:
        return True
    # Contains a known corporate/government suffix
    if _ORG_SUFFIX_RE.search(name):
        return True
    # BibTeX personal names use "Last, First" — a name with no comma and no
    # period (period = initial) in a multi-word string is likely an org name.
    # e.g. "National Center for Biotechnology Information"
    if ',' not in name and '.' not in name and len(name.split()) >= 3:
        return True
    return False
