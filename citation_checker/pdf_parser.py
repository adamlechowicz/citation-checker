"""Parse bibliographies from PDF files into BibEntry objects.

Pipeline:
  extract text (pymupdf) -> locate references section ->
  detect citation format -> split into blocks -> parse each block ->
  return list[BibEntry]

Supports eight citation format families, auto-detected per PDF
(consistent format within a single file is assumed):

  - ACM bracketed:    [1] Authors. YEAR. Title. Venue.
  - Plain numbered:    1. Authors. YEAR. Title. Venue.   (or "1) Authors...")
  - Author-year:       Authors. (YEAR). Title. Venue, vol(issue), pages.
  - LaTeX alpha:      [AGS20] Authors. Title. Venue, YEAR.
  - AY bracketed:     [Surname et al., YEAR] Authors. (YEAR). Title. Venue.
  - ACM name-year:    First Last, ..., and First Last. YEAR. Title. Venue.
  - Chicago:          Last, First, and First Last. "Title." Venue vol (Date): pp.
                      (or "Last, First. Book Title. City: Publisher, YEAR.")
  - ICML:             Surname, F., Surname, F., and Surname, F. Title. Venue, Year.
  - NeurIPS/ICLR:     F. Surname, F. Surname, and F. Surname. Title. Venue, Year.
"""

from __future__ import annotations

import logging
import re
from enum import Enum
from typing import Callable, Optional

import fitz  # pymupdf

from .models import BibEntry
from .utils import clean_doi, clean_arxiv_id

log = logging.getLogger(__name__)


class CitationFormat(Enum):
    ACM_BRACKETED = "acm_bracketed"
    PLAIN_NUMBERED = "plain_numbered"
    AUTHOR_YEAR = "author_year"
    LATEX_ALPHA = "latex_alpha"
    AY_BRACKETED = "ay_bracketed"
    ACM_NAME_YEAR = "acm_name_year"
    CHICAGO = "chicago"
    NEURIPS = "neurips"
    ICML = "icml"
    # Full-firstname authors, no entry numbering, year at end of citation.
    # Example: "Daron Acemoglu and Pascual Restrepo. Artificial intelligence,
    # automation, and work. In The economics of artificial intelligence: An
    # agenda, pages 197–236. University of Chicago Press, 2018."
    # Common in recent large-LM tech reports (Llama 2, PaLM, etc.).
    NAME_YEAR_END = "name_year_end"


_GENERIC_FORMATS = frozenset({
    CitationFormat.ACM_BRACKETED,
    CitationFormat.PLAIN_NUMBERED,
    CitationFormat.AUTHOR_YEAR,
})


_BLOCK_GRAMMAR_ORDER: dict[str, list[str]] = {
    'acm':       ['acm', 'bare_year'],
    'elsevier':  ['elsevier', 'bare_year'],
    'ay_inline': ['ay_inline', 'bare_year'],
    'vancouver': ['vancouver', 'vancouver_corp'],
    'nature':    ['nature', 'bare_year'],
    'bare_year': ['bare_year', 'elsevier'],
    'unknown':   ['acm', 'elsevier', 'ay_inline', 'vancouver', 'nature',
                  'vancouver_corp', 'bare_year'],
    # 'n/a' from non-generic formats; treat as unknown.
    'n/a':       ['acm', 'elsevier', 'ay_inline', 'vancouver', 'nature',
                  'vancouver_corp', 'bare_year'],
}


# ---------------------------------------------------------------------------
# Compiled regex patterns (module-level for performance)
# ---------------------------------------------------------------------------

# Section heading: "References" / "Bibliography" / "Works Cited" as a
# standalone line. The "Works Cited" variant is MLA's section name; PDFs in
# this style may have MULTIPLE such sections (one per essay in a collection).
_REFS_HEADING_RE = re.compile(
    r'^\s*(?:'
    r'REFERENCES|References'
    r'|BIBLIOGRAPHY|Bibliography'
    r'|WORKS\s+CITED|Works\s+Cited'
    r')\s*$',
    re.MULTILINE,
)

# Start markers used by splitters
_REF_START_RE = re.compile(r'^\[(\d+)\]\s+', re.MULTILINE)
_PLAIN_NUM_START_RE = re.compile(r'^\s{0,3}(\d{1,3})[.\)]\s+(?=[A-Z])', re.MULTILINE)
# LaTeX alpha key: "[AGS20]", "[DBB+14]", "[DeG74]", "[NHH+14]". Letters
# (mixed case) + optional "+" + 2-or-4-digit year + optional letter suffix.
# Followed by whitespace (either a newline or a space before the entry text).
_REF_ALPHA_START_RE = re.compile(
    r'^\[([A-Za-z]+\+?\d{2,4}[a-z]?)\]\s+', re.MULTILINE
)

# Patterns used by format detection (count occurrences; tie-break by priority)
_FMT_BRACKETED_RE = re.compile(r'^\[\d+\]\s+', re.MULTILINE)
_FMT_PLAIN_NUM_RE = re.compile(r'^\s{0,3}\d{1,3}[.\)]\s+[A-Z]', re.MULTILINE)
_FMT_AUTHOR_YEAR_RE = re.compile(
    r"^[A-Z][\w\-'À-ſ]+"
    r"[^\n]{0,250}?\(\s*(?:1[89]\d{2}|20\d{2})\s*[a-z]?\s*\)",
    re.MULTILINE,
)
_FMT_ALPHA_RE = re.compile(r'^\[[A-Za-z]+\+?\d{2,4}[a-z]?\]', re.MULTILINE)
# Author-year bracketed: "[Surname et al., YEAR]" or "[Surname and Surname, YEAR]".
# Key starts with a capital letter + word chars (no digits-only or short alpha keys).
_FMT_AY_BRACKETED_RE = re.compile(
    r"^\[[A-Z][\wÀ-ÿ'\-]+[^\[\]\n]*,\s+(?:1[89]\d{2}|20\d{2})[a-z]?\]",
    re.MULTILINE,
)
# Entry-start anchor for the splitter: captures the key inside the brackets.
_REF_AY_BRACKETED_START_RE = re.compile(
    r"^\[([A-Z][\wÀ-ÿ'\-]+[^\[\]\n]*,\s+(?:1[89]\d{2}|20\d{2})[a-z]?)\]\s*",
    re.MULTILINE,
)
# ACM name-year: un-numbered entries where authors are in "First Last" order
# followed by a bare year — "First Last, ..., and First Last. YEAR. Title."
# Requires `\S*\s+\S` so there is at least one internal space before ". YEAR.",
# which prevents matching bare-surname lines like "Yokoo. 2024." (where `\S*`
# consumes the trailing period, placing "2024" in the next token position —
# after which no second ". YEAR. " can be found). `[a-z]?` handles
# disambiguation suffixes such as "2021a".
_FMT_ACM_NAME_YEAR_RE = re.compile(
    r'^[A-Z][a-z]\S*\s+\S.+?\.\s+(?:1[89]\d{2}|20\d{2})[a-z]?\.\s+',
    re.MULTILINE,
)
# Lookahead scanner: checks whether ". YEAR. " appears in a short text window.
_ACM_NAME_YEAR_IN_WINDOW_RE = re.compile(
    r'\.\s+(?:1[89]\d{2}|20\d{2})[a-z]?\.\s+'
)
# Venue/institution words that can start continuation lines — used in the
# secondary (lookahead) splitter pass to reject false entry-start candidates.
_ACM_NAME_YEAR_VENUE_START_RE = re.compile(
    r'^(?:In\s|Proceedings\b|Conference\b|Journal\b|Workshop\b|'
    r'Symposium\b|University\b|Systems\b|Dissertation\b|arXiv\b)',
    re.IGNORECASE,
)
# Chicago format signal: line begins with "Surname, Firstname" (full first
# name, not just an initial — that's what distinguishes from the author-year
# "Smith, J." form) OR the em-dash "ditto authors" line that Chicago uses for
# multiple works by the same author.
# The first-name token after the comma must be Title-cased — a capital
# followed by at least one LOWERCASE letter. This rules out "City, ST"
# fragments like "Washington, DC" or "Princeton, NJ" that wrap inside
# citation venue/publisher clauses and would otherwise be mis-detected as
# new entries.
#
# Negative lookahead `(?!\s+[A-Z][a-z]{2,},)` rules out *wrapped author list*
# continuations like ``Shakeri, Emanuel Taropa, Paige Bailey, ...`` — where
# `Emanuel` is followed by ` Taropa,` (space + multi-letter surname + comma),
# i.e. the previous author's surname wrapping to a new line. Real Chicago
# entries never have a full-length surname-with-comma directly after the
# firstname; they have either a `.`, a `,` (multi-author list with surname
# next), or a single-letter middle initial like ` E,`.
_FMT_CHICAGO_RE = re.compile(
    r"^(?:[A-Z][\w'À-ſ\-]+,\s+[A-Z][a-zÀ-ſ][\w'À-ſ\-]*\b"
    r"(?!\s+[A-Z][a-z]{2,},)"
    r"|———)",
    re.MULTILINE,
)
# ICML/many-ML-venues style: entries start with "Surname, F." — surname-first
# with comma then initial, alphabetically ordered, no numbering prefix.
# Requires ≥3-character surname (capital + at least 2 more) so that stray
# initial lines like "U., Chakkaravarthy" (where "U" is just one letter)
# don't fire.
_FMT_ICML_RE = re.compile(
    r"^[A-Z][a-z][\w\-'À-ÿ]+,\s+[A-Z]\.",
    re.MULTILINE,
)
_DETECT_MIN_MATCHES = 3

# NeurIPS/ICLR style: entries start with "Initial. Surname" where Initial is a
# single capital letter followed by a period, and Surname is a multi-letter word.
# Each such line at the START of a line (^) is a strong signal.
_FMT_NEURIPS_RE = re.compile(r'^[A-Z]\.\s+[A-Z][a-z]', re.MULTILINE)

# Name-year-end style entry start: line begins with a full first name followed
# by a capitalised surname (e.g. "Daron Acemoglu" or "Yuntao Bai"). Used as a
# raw-count detection signal AND as the entry-start anchor in the splitter.
# Distinguished from Chicago (which is "Last, First") and from NEURIPS (which
# is "F. Last" with an initial-period). The character class allows hyphenated
# and diacritic surnames; ``[A-Z][a-z]+`` on the first token forces a real
# first name rather than an initial.
_FMT_NAME_YEAR_END_RE = re.compile(
    r"^[A-Z][a-z]+\s+[A-Z][\w'À-ÿ\-]+",
    re.MULTILINE,
)

# Appendix/supplementary section heading — truncated from the refs section
# before splitting to avoid in-text citations contaminating the entry list.
_APPENDIX_HEADING_RE = re.compile(
    r'^\s*(?:SUPPLEMENTARY\s+MATERIAL|SUPPLEMENTARY\s+APPENDIX|'
    r'APPENDIX|Appendix(?:es|\s+[A-Z])?|ADDITIONAL\s+MATERIAL|'
    r'PROOFS|APPENDIX\s+A\b|'
    r'[A-Z]\s*\n\s*(?:MODEL\s+ARCHITECTURE|EXPERIMENTAL\s+DETAILS|'
    r'IMPLEMENTATION\s+DETAILS|ADDITIONAL\s+RESULTS))\s*$',
    re.MULTILINE | re.IGNORECASE,
)

# DOI: doi:10.xxx/yyy or https://doi.org/10.xxx/yyy
_DOI_RE = re.compile(
    r'(?:doi:\s*|https?://(?:dx\.)?doi\.org/)'
    r'(10\.\d{4,9}/[^\s,;\]]+)',
    re.IGNORECASE,
)

# arXiv: arXiv:NNNN.NNNNN or arXiv:category/NNNNNNN, optional version + category tag
_ARXIV_RE = re.compile(
    r'arXiv:\s*(?:arxiv\s*[-:]?\s*)?'
    r'([a-zA-Z\-]+/\d{7}|\d{4}\.\d{4,5})(?:v\d+)?(?:\s*\[[^\]]+\])?',
    re.IGNORECASE,
)

# Generic URL (matched after DOI/arXiv to avoid duplicates)
_URL_RE = re.compile(r'https?://[^\s,;\]]+', re.IGNORECASE)

# 4-digit year in the range 1000-2099
_YEAR_RE = re.compile(r'\b(1[0-9]{3}|20[0-9]{2})\b')

# ACM-style block anchor: "Authors. YEAR. Title. Venue."
_ACM_YEAR_SPLIT_RE = re.compile(
    r'^(.*?)\.\s+(1[0-9]{3}|20[0-9]{2})[a-z]?\.\s+(.*)',
    re.DOTALL,
)

# Author-year block anchor: "Authors. (YEAR)[.] Title. ..."
# Requires a literal "." immediately (with optional whitespace) before the
# year paren. This is what distinguishes the author-year-inline grammar from
# Elsevier-style refs where (YYYY) is preceded by a volume number digit
# (e.g. "Renew. Sustain. Energy Rev. 39 (2014) 748-764.") — there the char
# before "(" is a space following a digit, not a period.
# `rest` requires a word character so this regex doesn't swallow Nature-style
# blocks where (YYYY) is at the very end with only a trailing "." after it.
_AY_BLOCK_RE = re.compile(
    r"^(?P<authors>.+?)"
    r"[.,]\s*"
    r"\(\s*(?:(?P<year>1[89]\d{2}|20\d{2})\s*[a-z]?|(?P<nodate>n\.\s*d\.))\s*\)"
    r"\s*\.?\s*"
    r"(?P<rest>\w.+)$",
    re.DOTALL,
)

# Parenthesized year token (used in the author-year splitter lookahead).
# Also matches APA's "(n.d.)" no-date marker.
_AY_YEAR_PAREN_RE = re.compile(
    r"\(\s*(?:1[89]\d{2}|20\d{2})\s*[a-z]?\s*\)|\(\s*n\.\s*d\.\s*\)"
)

# Line begins with a surname-shaped capitalized token (allows hyphens, diacritics)
_AY_NEW_ENTRY_START_RE = re.compile(r"^[A-Z][\w\-'À-ſ]+")

# Number of trailing lines to consider when deciding if a candidate start
# line truly belongs to a new author-year entry (entry may wrap before year).
_AY_LOOKAHEAD_LINES = 2

# A single-letter capital that looks like an initial: "F." or "J," or "A ".
# Used to distinguish author-list lines (have initials) from venue continuation
# lines (don't), when the parenthesized year is on a wrap line below.
_AY_AUTHOR_INITIAL_RE = re.compile(r'\b[A-Z](?:\.|(?=[\s,]))')

# Only the first ~50 chars of a candidate line are scanned for an initial,
# to avoid matching incidental single-letter capitals deep inside venue names.
_AY_AUTHOR_LIST_WINDOW = 50

# Where a title likely ends: a '. ' followed by a venue-like token.
# The set covers the most common ACM/CS/operations-research journals; entries
# whose venue is not in this list fall through to `_VENUE_TRAILER_RE` (below)
# which finds the volume/issue/pages signature instead.
_TITLE_END_RE = re.compile(
    r'[.!?]\s+(?:'
    # Generic venue markers
    r'In\b|Proc\.|Proceedings|arXiv\s+preprint|arXiv:|https?://|doi:|Vol\.\s*\d+|'
    r'\d{1,3},\s*\d+\s*\(|Springer|Elsevier|PMLR|Advances\s+in|'
    r'[A-Z]{2,4}\s+\d{4}|'
    # Common ML/CS conference abbreviations (ICML, NeurIPS, ICLR, AAAI, etc.)
    r'NeurIPS\b|ICML\b|ICLR\b|AAAI\b|IJCAI\b|CVPR\b|ICCV\b|ECCV\b|'
    r'SODA\b|STOC\b|FOCS\b|ICALP\b|COLT\b|OSDI\b|ISCA\b|SOSP\b|NSDI\b|'
    # Publisher / society prefixes
    r'IEEE|ACM|'
    # "Journal" / "The Journal" and compounds
    r'Journal|The\s+Journal|European\s+Journal|International\s+Journal|'
    # Common specific journals seen in this domain
    r'Algorithmica\b|American\b|Communications\s+of|'
    r'Computers\s+(?:and|&)|CoRR\b|Handbook\b|'
    r'Information\s+Processing|Mathematical\s+Foundations|'
    r'Mathematics\s+of|Nature\b|Operational\s+Research|Operations\s+Research|'
    r'RAIRO|Surveys\s+in|Systems\s+Engineering|Theoretical\s+Computer'
    r')'
)

# Numeric venue-trailer fallback: the volume/issue/pages signature that almost
# every academic citation ends with. Matches things like:
#   "30(1), 101-139"   "607, 35-48"   "76(5), 2249-2305"   "2015, 593: 139-145"
#   "Article 108437"
# If `_TITLE_END_RE` finds no explicit venue token, `_extract_title` walks back
# from the trailer to the previous period and cuts the title there.
_VENUE_TRAILER_RE = re.compile(
    # Volume(issue), page-page  /  volume, page-page
    # (issue may itself be a range like "1–2" for combined-issue journals)
    r'\b\d{1,4}\s*(?:\(\s*\d+(?:\s*[-–]\s*\d+)?[a-z]?\s*\))?\s*[,:]\s*\d{1,4}\s*[-–]\s*\d'
    # Article number only — no page range. APA journals increasingly cite
    # papers by article ID alone (PLoS One e0249751, Asian J Psychiatry 101909).
    r'|\b\d{1,4}\s*(?:\(\s*\d+(?:\s*[-–]\s*\d+)?[a-z]?\s*\))?\s*,\s*e?\d{3,}\b'
    r'|\b\d{1,4}\s*\(\s*\d+\s*\)\s*[,:]\s*\d'
    r'|\bArticle\s+\d+\b'
)

# Noise lines inserted by pdf extraction between reference text:
# bare page numbers, running heads ("Smith et al.", "ONLINE SMOOTHED...",
# single short capitalized surnames like "Rhode" used as page-top indicators).
# The "et al." branch must NOT contain commas — running headers are single
# surnames like "Smith et al.", whereas Vancouver-style author lines that
# *legitimately* end in "et al." contain commas between authors and must be
# preserved (e.g., "Celli B, Fabbri L, ..., Vogelmeier C, et al.").
# The single-Surname branch is limited to ≤8 chars so content words that wrap
# alone onto a line ("Optimal", "Balanced", "Learning") are not discarded.
_NOISE_LINE_RE = re.compile(
    r'^\s*(?:'
    r'\d{1,4}'
    r'|[A-Z][a-z][\w\s\-\.]+ et al\.'
    r'|[A-Z][A-Z\s\-]{5,}'
    r')\s*$'
)

# Heals URLs split across lines by PDF line-breaking: "https:\n//"
_SPLIT_URL_RE = re.compile(r'(https?):\s*\n\s*//')

# URL-path continuation healer: when a URL was wrapped mid-path in the PDF
# (e.g. ``https://example.com/path-\nand-more``), after the line collapse
# joins lines with spaces we end up with ``https://example.com/path- and-more``.
# This regex finds an ``https?://...`` URL prefix followed by whitespace and
# a URL-shaped continuation token (one containing at least one of /, ., %,
# =, #, _, ~, or trailing -) and glues them together. Plain words ("Press",
# "Some") lack URL-shaped chars and are *not* absorbed.
_URL_REJOIN_RE = re.compile(r'(https?://\S*)\s+(/?\S*[/.%=#_~-]\S*)')

# Stray glyph-encoding noise emitted by pymupdf when a font's mapping is
# incomplete: ASCII control chars (0x00-0x1F except \t/\n) and the Euro sign
# (0x20AC) appearing inside word tokens. These should be dropped before the
# author regexes try to match, since they create false word boundaries.
_PDF_GLYPH_NOISE_RE = re.compile(r'[\x00-\x08\x0B-\x1F€¸˜]')

# Stand-alone tilde (U+007E) appearing between two letters is also a glyph
# fallback artifact (e.g. "Catalu~na" for "Cataluña"). Only stripped when
# bracketed by letters so we don't disturb intentional tildes elsewhere.
_PDF_INWORD_TILDE_RE = re.compile(r'(?<=[A-Za-zÀ-ſ])~(?=[A-Za-zÀ-ſ])')

# Spacing modifier letters that pymupdf emits as standalone code points when
# a font's diacritic encoding is incomplete — e.g. "Je˙z" for "Jeż" (Polish),
# "Peˇcari´c" for "Pečarić" (Croatian). Strip them only between letters so
# that the word shape survives fuzzy matching.
_PDF_COMBINING_NOISE_RE = re.compile(
    r'(?<=[A-Za-zÀ-ſ])[˙´ˇˆ˚˛¸˜ˉ˃˂](?=[A-Za-zÀ-ſ])'
)

# Heals mid-word hyphenation introduced by PDF line wraps after `_collapse_block`
# joins lines with a space: "distribu- tions" -> "distributions".
# Trade-off: a real compound word that wraps at its hyphen ("Learning-augmented")
# loses the hyphen too, but the joined form still fuzzy-matches the canonical at
# >95% so verification still succeeds.
_WRAP_HYPHEN_RE = re.compile(r'([a-z])-\s+([a-z])')

# Matches the first word-shaped token of a collapsed reference block; used to
# build a meaningful cite key. Allows hyphens, apostrophes, and diacritics.
# Optionally skips a leading initials block ("O.", "J.H.", "J.C.-W.") so that
# Elsevier-style entries ("O. Ellabban, ...") yield the surname "Ellabban"
# rather than the initial "O".
_FIRST_TOKEN_RE = re.compile(
    r"^\s*(?:(?:[A-Z]\.[-\s]?)+\s*)?([A-Za-zÀ-ſ][\w\-']*)"
)

# Split author lists on " and " (case-insensitive) -- ACM style
_AUTHOR_AND_RE = re.compile(r'\s+and\s+', re.IGNORECASE)

# Top-level author boundary for author-year style: " & " or " and "
_AY_BIG_SEP_RE = re.compile(r'\s*(?:&|\sand\s)\s*', re.IGNORECASE)

# Token that is *only* initials, e.g. "A", "F.", "F. Y.", "C M", "L.-L."
# The optional `-[A-Z]\.?` group inside each iteration handles hyphenated
# initial pairs (e.g. "L.-L." for the name "Lars-Lukas Richter").
_INITIALS_TOKEN_RE = re.compile(r'^(?:[A-Z]\.?(?:-[A-Z]\.?)?\s*){1,4}$')

# Token of the form "Surname X" or "Surname X. Y." with trailing initials.
_NAME_WITH_INLINE_INITIALS_RE = re.compile(
    r"^([A-Z][\w\-'À-ſ]+(?:\s+[A-Z][\w\-'À-ſ]+)*?)"
    r"\s+((?:[A-Z]\.?\s*){1,4})$"
)

# Trailing "et al." / ", et al" marker stripped before author parsing.
_TRAILING_ET_AL_RE = re.compile(r',?\s*et\s+al\.?\s*$', re.IGNORECASE)

# A bare capital letter (not followed by another word char or period) -> add "."
_BARE_INITIAL_RE = re.compile(r'\b([A-Z])(?![\w.])')

# --- Nature-style block grammar: "Authors. Title. Venue (Year)." ---------- #
# The year is in TRAILING parens (possibly wrapped by publisher info) — quite
# different from the ACM "Authors. YEAR. Title." anchor. These regexes are
# used as a fallback inside `_parse_numbered_block` when ACM matching fails.

# Any "(...YYYY...)" paren block; `_parse_nature_block` picks the LAST one in
# the block, which is robust against trailing URLs/notes after the paren.
_NATURE_TRAILER_RE = re.compile(
    r'\(([^()]*?\b(?:1[89]\d{2}|20\d{2})\b[^()]*?)\)'
)

# A single author chunk in Nature form: "Last, F." / "Last, F. M." / "Last, F.-M."
# Surname may be multi-word ("Scott Frazier") or hyphenated ("Arribas-Bel").
_NATURE_AUTHOR_CHUNK = (
    r"[A-Z][\w\-']+(?:[-\s][A-Z][\w\-']+)*"   # surname (possibly multi-word / hyphenated)
    r",\s+"
    r"(?:[A-Z]\.[-\s]?)+"                      # block of initials
)

# Author-list terminators (used to split authors from title in the head).
_NATURE_ETAL_END_RE = re.compile(r'\bet\s+al\.')
_NATURE_AMP_END_RE = re.compile(rf'\s+&\s+{_NATURE_AUTHOR_CHUNK}')
_NATURE_SINGLE_RE = re.compile(rf'^{_NATURE_AUTHOR_CHUNK}')


# --- Elsevier-style block grammar -------------------------------------- #
# "F.M. Last1, F.M. Last2, ..., F.M. LastN, Title, Venue Abbr. Vol (YEAR) pp."
# (or "..., Title, Publisher, YEAR." for books; "..., Title, in: Conf., YEAR."
# for proceedings). Distinguished from author-year by initials-first author
# order and comma-separated (rather than period-separated) structure.

# One Elsevier author: initials block + space + surname (possibly multi-word
# or hyphenated). Handles "F. Last", "F.M. Last", "J.C.-W. Lin", "H. Abu Rub".
_ELSEVIER_AUTHOR = (
    r"(?:[A-Z]\.[-\s]?)+"                       # initials block (F. / F.M. / J.-C.)
    r"\s*"
    r"[A-Z][\w\-']+(?:[-\s][A-Z][\w\-']+)*"     # surname (possibly multi-word)
)

# Prefix matcher for the author list — anchored at start of block.
_ELSEVIER_AUTHORS_RE = re.compile(
    rf"^{_ELSEVIER_AUTHOR}(?:,\s+{_ELSEVIER_AUTHOR})*"
)

_ELSEVIER_AND_FINAL_AUTHOR_RE = re.compile(
    rf"^and\s+({_ELSEVIER_AUTHOR})\s*,\s+"
)

# Year in (YYYY) form anywhere after authors; the FIRST such match is the year.
# Issue numbers like "(5)" are excluded by the 4-digit requirement.
_ELSEVIER_YEAR_PAREN_RE = re.compile(r'\(\s*(1[89]\d{2}|20\d{2})\s*\)')

# Year bare near the end of block, after a comma: book and proceedings format.
# Allows trailing conference pages such as ``, 2015, pp. 1-5``.
_ELSEVIER_YEAR_BARE_RE = re.compile(
    r',\s*(1[89]\d{2}|20\d{2})'
    r'(?:\s*,\s*pp?\.?\s*[\dA-Za-z]+(?:\s*[-–]\s*[\dA-Za-z]+)?)?'
    r'\.?\s*$'
)

# Indicators that mark the title->venue boundary (used as a positive lookahead
# after a comma to find where the title ends).
_ELSEVIER_VENUE_START_RE = re.compile(
    r",\s+(?="
    r"[A-Z][a-z]+\.\s+[A-Z]"                # abbreviation chain ("Renew. Sustain...")
    r"|[A-Z][a-z]+\s+[A-Z][a-z]*\."        # "Energy Sci." (full+abbrev)
    r"|[A-Z]{2,}\b"                         # acronym (IEEE, ACM, NASA)
    r"|in:\s"                               # conference "in: Proc..."
    r"|vol\.\s*\d"                          # "vol. 17" book series volume
    r"|(?:Springer|Elsevier|Wiley|Cambridge\b|Oxford\b|MIT\b|"
    r"Cengage|McGraw|Pearson|Routledge|Academic\s+Press|"
    r"Princeton\b|Harvard\b|Yale\b|Stanford\b)"
    r")"
)


# --- Numbered-with-bare-trailing-year block grammar -------------------- #
# "[N] Author1, Author2, ..., and AuthorN. Title. Venue, pages, YEAR."
# Distinguishing features:
#   - Authors are given-name first ("First [Middle] Last" or "FM Last" or "F.
#     Last") and comma-separated, with the last preceded by "and" / "&" or
#     truncated with "et al."
#   - Year is bare (no parens), at the very end of the block, preceded by a
#     comma — proceedings/journal form — or a period for short corporate
#     reports ("World Health Organization. ... covid-19. 2021.").
#   - No (YEAR) parenthesised marker anywhere; that signature is what
#     distinguishes this from the AY-inline and Nature grammars.
# Used by ACM proceedings, IMWUT/Ubicomp, and many engineering venues.

# Year at end (bare), after either ", " or ". ".
_NUMBERED_BARE_YEAR_RE = re.compile(r'(?:,|\.|\s)\s*(1[89]\d{2}|20\d{2})\.?\s*$')

# Trailing metadata patterns stripped before year-end matching.
# USENIX/ACM DL entries append ISBN, ISSN, publisher names, and access
# notices after the year, preventing _NUMBERED_BARE_YEAR_RE from anchoring.
_ISBN_ISSN_STRIP_RE = re.compile(r'\b(?:ISBN|ISSN)\s+[\d\-X]+\.?\s*', re.IGNORECASE)
_ACCESSED_STRIP_RE = re.compile(
    r'\[?[Aa]cces+ed\b[^\]]*\]?\.?\s*$'      # "[Accessed DD-MM-YYYY]."
    r'|,?\s*\d{4}\.\s*\[[^\]]*\]\.\s*$',     # ",2025. [Accessed ...]."
)
_TRAILING_PUBLISHER_RE = re.compile(
    r'(?:\.\s+(?:Association for Computing Machinery|USENIX(?:\s+Association)?|'
    r'IEEE(?:\s+Computer\s+Society|\s+Press)?|ACM(?:\s+Press)?|'
    r'Springer(?:\s+Nature)?|Elsevier|MIT Press|'
    r'Cambridge University Press|Morgan Kaufmann)\.?)+\s*$',
    re.IGNORECASE,
)

# End of author list: "and"/"&" + final author name + ". ".
# Body character class excludes "." so periods terminate tokens, preventing
# greedy overconsumption of title words. Includes À-ÿ for accented surnames
# (Bergés) and allows uppercase body chars for all-caps initials (SIV).
_NUMBERED_AND_AUTHOR_END_RE = re.compile(
    r'(?:,\s+and|\s+and|\s+&)\s+'
    r"(?:(?:[A-Z][A-Za-zÀ-ſ\-']*|de|del|da|dos|du|van|von|der|den|la|le)\s+)*"
    r'[A-Z][A-Za-zÀ-ſ\-\']*'                   # final surname (must start uppercase)
    r'(?:\.[A-Za-zÀ-ſ][A-Za-zÀ-ſ\-\']*)*'     # compound initials: .D .Ab .Séc
    r'\.\s+',                                    # author-list terminator: ". "
)

# Single initialed author ending the author prefix: "C. M. Bishop. Title...".
_NUMBERED_SINGLE_AUTHOR_END_RE = re.compile(
    r'^((?:[A-Z]\.\s*){1,4}'
    r"[A-Z][A-Za-zÀ-ſ\-']+(?:\s+[A-Z][A-Za-zÀ-ſ\-']+)*)\.\s+"
)

# "et al." marker terminating an author list.
_NUMBERED_ET_AL_END_RE = re.compile(r',?\s*et\s+al\.\s+', re.IGNORECASE)


# --- Vancouver-style block grammar ------------------------------------- #
# "Surname I[I][I], Surname I[I][I], ..., et al. Title. Venue YEAR;vol:pp."
# Distinguishing features:
#   - Authors are surname-FIRST, then 1-3 capital initials WITHOUT periods
#     ("Kohansal R", "Buist AS", "Martinez FJ"). This is unique to Vancouver
#     style; other styles use "F. M. Last" or "Last, F.M.".
#   - Authors are comma-separated, optionally terminated by "et al.".
#   - The year is bare (no parens) and followed by a semicolon for journals
#     ("Lancet 2006;367:1216-1219.") or a period for books/reports.

# Lowercase particles that may appear inside a multi-word surname
# ("Montes de Oca", "van der Berg", "Abu al-Hasan"). Limited list — these
# don't introduce false positives because they only act as connectors,
# never as standalone capitalized name tokens.
_NAME_PARTICLE = (
    r"(?:de|del|della|delle|di|da|dos|du|van|von|der|den|el|al|le|la|"
    r"bin|ibn|af|zu|ten|ter)"
)

# One Vancouver author token: "Surname I" or "Multi-word Surname IJK".
# Surnames can include lowercase particles between capitalized parts.
# The trailing `\b(?![a-z])` after the initials is required — otherwise a
# corporate author like "Global Initiative..." would match as surname="Global"
# + initials="I" (with "nitiative" left as title).
_VANCOUVER_AUTHOR = (
    r"[A-Z][\w'À-ſ\-]+"
    rf"(?:\s+(?:[A-Z][\w'À-ſ\-]+|{_NAME_PARTICLE}))*"
    r"\s+[A-Z]{1,5}\b(?![a-z])"
)

# Prefix matcher: authors at start of a Vancouver block.
_VANCOUVER_AUTHORS_RE = re.compile(
    rf"^(?P<authors>{_VANCOUVER_AUTHOR}"
    rf"(?:,\s+(?:{_VANCOUVER_AUTHOR}|et\s+al\.?))*)"
)

# Year in Vancouver venue position. Preferred form: year IMMEDIATELY followed
# by `;` or `:` (the conventional Vancouver "YEAR;vol:pages" separator). This
# disambiguates the publication year from a journal name like "J Appl Physiol
# (1985)" where the parenthesized 1985 is part of the venue identity, not the
# article's year.
_VANCOUVER_YEAR_SEP_RE = re.compile(r'\b(1[89]\d{2}|20\d{2})\s*[;:]')
# Publisher / book form: "...Publisher; YEAR." — the year follows a semicolon
# rather than preceding one. Used as fallback for corporate / report citations.
_VANCOUVER_YEAR_PUB_RE = re.compile(r';\s*(1[89]\d{2}|20\d{2})\b')
_VANCOUVER_YEAR_RE = re.compile(r'\b(1[89]\d{2}|20\d{2})\b')

# Title-end pattern after authors: end-of-sentence punctuation (`.`/`?`/`!`)
# followed by whitespace and the venue. Accepting `?`/`!` matters for titles
# phrased as questions or exclamations ("...friend or foe? Eur Respir J 2018").
_VANCOUVER_TITLE_RE = re.compile(
    r'\s*\.?\s*(?P<title>.+?)[.?!]\s+(?P<rest>.+)$',
    re.DOTALL,
)


# --- NeurIPS/ICLR-style block grammar -------------------------------------- #
# "I. Surname, I. Surname, ..., and I. Surname. Title. Venue, Year."
# Distinguishing features:
#   - Authors are initial-first ("I." or "I. M." or "H.-J.") + surname.
#   - Multiple authors separated by commas; last preceded by "and".
#   - No numbering prefix; entries start directly with the first author's initial.
#   - Year is bare at the end of the citation (no parentheses).

# One initials block: "M.", "H.-J.", "D. F.", "J. C. N.", "Ł." (Unicode).
# Uses \w (1–3 chars) to handle non-ASCII initials like Ł, Š, Ž from
# Central/Eastern European author names.
_NEURIPS_INITIALS = r"(?:\w{1,3}\.(?:-\w{1,3}\.)?(?:\s+)?)+"

# Single NeurIPS author: one or more initials followed by a multi-letter surname.
# The optional second surname word handles two-part surnames like "Si Salem",
# "Montazer Qaem", or "Abu Rub". Backtracking ensures greedy consumption of
# a title word is rejected when no ". " terminator follows.
_NEURIPS_AUTHOR_UNIT = (
    _NEURIPS_INITIALS                      # initials block
    + r"[A-Z][\w\-'À-ÿ]+"               # primary surname
    + r"(?:\s+[A-Z][\w\-'À-ÿ]+)?"       # optional second surname word
)

# Full author list: first author, then zero or more additional authors.
# Additional authors are either comma-separated (3+ authors: "A, B, and C")
# or connected by a bare " and " (exactly 2 authors: "A and B").
# Terminated by ". " (period-space separating authors from title).
_NEURIPS_AUTHORS_RE = re.compile(
    rf"^(?P<authors>"
    rf"{_NEURIPS_AUTHOR_UNIT}"
    rf"(?:"
    rf"(?:,\s+(?:and\s+)?{_NEURIPS_AUTHOR_UNIT})+"  # comma-separated (3+)
    rf"|(?:\s+and\s+{_NEURIPS_AUTHOR_UNIT})"         # bare "and" (2 authors)
    rf")*"
    rf")\.\s+",
)

# Abbreviation-chain detector: "Surname." immediately followed by ". CapWord." —
# signals a journal-abbreviation sequence (e.g., "J. Mach. Learn. Res.") rather
# than a surname. Used to reject false-positive entry-start lines.
_NEURIPS_ABBREV_CHAIN_RE = re.compile(r'^\.\s+[A-Z][a-z]+\.')


# --- ICML/surname-first style block grammar -------------------------------- #
# "Surname, F., Surname, F., ..., and Surname, F. Title. Venue info, Year."
# Distinguishing features:
#   - Authors are surname-first, comma, then one or more initials ("F." or
#     "F. M." or "C.-J."). This is the opposite of NeurIPS "F. Surname".
#   - Authors comma-separated; two authors use bare " and "; three+ use ", and".
#   - Year is bare (no parens) at the end. DOIs and URLs may also be present.

# Year with optional single-letter disambiguation suffix ("2021a", "2021b") at the
# end of an ICML reference block.  The suffix letter is intentionally not
# captured — we extract only the 4-digit year for the BibEntry year field.
_ICML_BARE_YEAR_RE = re.compile(
    r'[,.]\s*(1[89]\d{2}|20\d{2})[a-z]?\.?\s*$'
)

# AAAI/AAAI-proceedings style: year immediately follows the author boundary
# ("Authors. YEAR. Title. Venue.") rather than trailing the venue. When
# _ICML_BARE_YEAR_RE fails to find a trailing year, this pattern checks whether
# the rest string opens with "YEAR. " and strips it so the title is clean.
_ICML_LEADING_YEAR_RE = re.compile(r'^(1[89]\d{2}|20\d{2})[a-z]?\.\s+')

# Entry-start candidate: line begins with a multi-letter surname (≥3 chars) +
# comma + space + capital initial + period. The ≥3-char requirement ensures
# single-letter initials on continuation lines (e.g. "U., Chakkaravarthy")
# don't fire. "\w\-'À-ÿ" allows hyphenated surnames (El-Yaniv) and diacritics.
_ICML_LINE_START_RE = re.compile(r"^[A-Z][a-z][\w\-'À-ÿ]*(?:\s+[A-Z][\w\-'À-ÿ]+)?,\s+[A-Z]\.")

# Individual ICML author chunk: "Surname, F." or "Surname, F. M." or
# "Multi-word Surname, C.-J."  Used with finditer to extract all authors from
# the author string.
_ICML_AUTHOR_CHUNK_RE = re.compile(
    r"[A-Z][\w\-'À-ÿ]+(?:\s+[A-Z][\w\-'À-ÿ]+)*"  # surname (possibly multi-word)
    r",\s+"
    r"(?:[A-Z]\.(?:-[A-Z]\.)?(?:\s+)?)+"            # initials: "F.", "F. M.", "C.-J."
)

# Finds the first ". " that is the author-list/title boundary — i.e., the
# period after the last author's final initial that is NOT followed by:
#   (a) another author surname (≥3 chars + comma + initial),
#   (b) "and" + another author, or
#   (c) another bare initial (handles "Y. T." multi-initial authors).
_ICML_AUTHOR_END_RE = re.compile(
    r'\.\s+(?!'
    r'[A-Z][\w\-\'À-ÿ]{2,}(?:\s+[A-Z][\w\-\'À-ÿ]+)*,\s+[A-Z]\.'  # next author
    r'|and\s+[A-Z][\w\-\'À-ÿ]{2,}(?:\s+[A-Z][\w\-\'À-ÿ]+)*,'  # "and" + next author
    r'|[A-Z]\.(?:-[A-Z]\.)?'                                     # next bare initial
    r')'
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _detect_content_grammar(blocks: list[tuple[int, str]]) -> str:
    """Detect the dominant author/title/year grammar across a sample of blocks.

    Tries each known grammar against the first ≤ 15 blocks and returns the
    name of the grammar with the most matches.  Returns ``'unknown'`` when
    no grammar wins at least 2 matches (triggers the full cascade fallback).

    Return values: ``'acm'``, ``'elsevier'``, ``'ay_inline'``,
    ``'vancouver'``, ``'nature'``, ``'bare_year'``, ``'unknown'``.
    """
    sample = blocks[:15]
    counts: dict[str, int] = {
        'acm': 0, 'ay_inline': 0, 'elsevier': 0,
        'vancouver': 0, 'nature': 0, 'bare_year': 0,
    }
    for _, raw in sample:
        t = _collapse_block(raw)
        if _ACM_YEAR_SPLIT_RE.match(t):
            counts['acm'] += 1
        elif _parse_elsevier_block(t) is not None:
            counts['elsevier'] += 1
        elif _AY_BLOCK_RE.match(t):
            counts['ay_inline'] += 1
        elif _parse_vancouver_block(t) is not None:
            counts['vancouver'] += 1
        elif _parse_nature_block(t) is not None:
            counts['nature'] += 1
        elif _NUMBERED_BARE_YEAR_RE.search(t):
            counts['bare_year'] += 1

    best = max(counts, key=lambda k: counts[k])
    if counts[best] < 2:
        return 'unknown'
    return best


_RECURRING_LINE_THRESHOLD = 3
_RECURRING_LINE_MIN_LEN = 5

# Year-only line: ``2018.`` / ``2024`` on its own. Numbered bibliographies in
# arXiv PDFs commonly wrap the trailing year of an entry to its own line:
#
#     [7] Eunsol Choi, He He, ..., and Luke Zettlemoyer. Quac: Question
#     answering in context. arXiv preprint arXiv:1808.07036,
#     2018.
#     [8] Christopher Clark, ...
#
# The literal ``2018.`` then repeats many times across the document and would
# otherwise be flagged as a recurring header/footer line. Exempt year-only
# lines so the trailing year survives into the entry's collapsed block text.
_YEAR_ONLY_LINE_RE = re.compile(r'^\s*(?:1[89]\d{2}|20\d{2})[a-z]?\.?\s*$')


# "et al." appears in the raw text of a citation when the original author list
# was truncated (e.g. "Smith, J. et al. Title. Venue, 2020."). The author
# parser strips the marker; this re-detects it so downstream fuzzy matching
# can treat the local author list as a known prefix of a longer remote list
# rather than penalising the size difference as a coincidence.
_ET_AL_DETECT_RE = re.compile(r'\bet\s+al\.?', re.IGNORECASE)


def _mark_truncated_author_lists(entries: list[BibEntry]) -> None:
    """Set ``truncated_authors=True`` on entries whose raw text contains an
    ``et al.`` marker. Mutates entries in place."""
    for e in entries:
        raw = e.raw_fields.get('raw_text') if e.raw_fields else None
        if raw and _ET_AL_DETECT_RE.search(raw):
            e.truncated_authors = True


def _strip_recurring_lines(text: str) -> str:
    """Strip running headers/footers — lines that appear identically on many
    pages of the PDF and bleed into entry blocks at page boundaries.

    The motivating case is Nature/Springer PDFs, where every page carries a
    footer like::

        Perspective
        https://doi.org/10.1038/s41467-023-44539-7
        Nature Communications|   (2024) 15:21

    If a bibliography entry crosses such a page break, the splitter pulls
    the footer text into the entry's raw block. Downstream, the block's
    DOI and year extractors then grab the *enclosing paper's* DOI and the
    journal-volume year instead of the cited paper's. The fix is to drop
    these lines before splitting — they are easy to identify because they
    repeat verbatim across pages, while real bibliography entries do not.

    A line is recurring when its trimmed content appears at least
    ``_RECURRING_LINE_THRESHOLD`` times in the document, with a minimum
    length to avoid stripping incidental punctuation lines (those are
    already handled by ``_NOISE_LINE_RE``)."""
    from collections import Counter
    lines = text.split('\n')
    counts = Counter(l.strip() for l in lines if len(l.strip()) >= _RECURRING_LINE_MIN_LEN)
    recurring = {
        l for l, c in counts.items()
        if c >= _RECURRING_LINE_THRESHOLD and not _YEAR_ONLY_LINE_RE.match(l)
    }
    if not recurring:
        return text

    # Any DOI that appears inside a recurring line is, by definition, the
    # enclosing paper's self-DOI (running header/footer). Strip any line
    # that contains it, even if that line itself appears only once — e.g.
    # the "Supplementary information available at <self-DOI>." line near
    # the end of Nature articles, which would otherwise stay and bleed
    # into the last entry's block.
    recurring_dois: set[str] = set()
    for line in recurring:
        for m in _DOI_RE.finditer(line):
            recurring_dois.add(m.group(1))

    log.info(
        "Stripping %d recurring header/footer line(s) and %d self-DOI(s) from PDF text",
        len(recurring), len(recurring_dois),
    )
    keep: list[str] = []
    for l in lines:
        if l.strip() in recurring:
            continue
        if recurring_dois and any(d in l for d in recurring_dois):
            continue
        keep.append(l)
    return '\n'.join(keep)


def parse_pdf_file(path: str) -> list[BibEntry]:
    """Extract and parse the bibliography from a PDF file.

    Returns a list of BibEntry objects with cite keys ref1, ref2, ...

    Raises:
        FileNotFoundError: if the file does not exist.
        ValueError: if no References section is found, no citation format
            can be detected, or detection succeeds but splitting yields
            zero blocks.
    """
    text = _extract_text(path)
    text = _strip_recurring_lines(text)
    refs_text = _find_references_section(text)
    fmt = _detect_format(refs_text)
    blocks = _SPLITTERS[fmt](refs_text)

    if not blocks:
        raise ValueError(
            f"Detected {fmt.value} format but could not split it into entries."
        )

    # For the generic formats that share _parse_block, detect the dominant
    # content grammar and restrict block parsing to that grammar + its fallbacks.
    content_grammar = 'n/a'  # only meaningful for generic formats
    if fmt in _GENERIC_FORMATS:
        content_grammar = _detect_content_grammar(blocks)

    parse_block = _BLOCK_PARSERS[fmt]
    entries: list[BibEntry] = []
    for num, raw in blocks:
        try:
            if fmt in _GENERIC_FORMATS:
                entries.append(_parse_block(num, raw, grammar=content_grammar))
            else:
                entries.append(parse_block(num, raw))
        except Exception as exc:
            log.warning("Failed to parse reference [%d]: %s", num, exc)

    if fmt is CitationFormat.CHICAGO:
        _resolve_chicago_ditto(entries)

    _mark_truncated_author_lists(entries)
    _assign_cite_keys(entries)

    log.info("Parsed %d references from %s (format=%s, grammar=%s)",
             len(entries), path, fmt.value, content_grammar)
    return entries


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def _extract_text(path: str) -> str:
    """Extract full text from a PDF using pymupdf."""
    doc = fitz.open(path)
    pages = [page.get_text() for page in doc]
    doc.close()
    return "\n".join(pages)


# ---------------------------------------------------------------------------
# Section detection
# ---------------------------------------------------------------------------

def _find_references_section(text: str) -> str:
    """Return the bibliography content of the PDF.

    Most papers have a single "References" / "Bibliography" heading at the
    end. Some documents (especially multi-essay MLA collections) contain
    multiple "Works Cited" sections — one per essay. In that case we slice
    each section from its heading to the start of the next, concatenate
    them with section separators, and return the combined block. The block
    splitters then see a single citation stream and produce one entry list.

    Raises ValueError if no heading is found.
    """
    matches = list(_REFS_HEADING_RE.finditer(text))
    if not matches:
        raise ValueError(
            "Could not find a References, Bibliography, or Works Cited "
            "section in the PDF. The heading must appear as a standalone line."
        )
    if len(matches) == 1:
        return text[matches[0].end():]
    # Multiple sections: stitch them together. Each section ends where the
    # next one begins (the body text between sections is included; the per-
    # format splitters will skip lines that don't match the citation grammar).
    parts: list[str] = []
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        parts.append(text[m.end():end])
    log.info("Found %d bibliography sections; concatenating", len(matches))
    return "\n".join(parts)


def _before_appendix(refs_text: str) -> str:
    """Return bibliography text before appendix or supplementary material."""
    app_m = _APPENDIX_HEADING_RE.search(refs_text)
    return refs_text[:app_m.start()] if app_m else refs_text


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

_CHICAGO_PREV_TERMINATORS = ('.', ']', '"', '”', "'", "’")

# Trailing-year signal: ", YEAR." at end of line. Used to gate NAME_YEAR_END
# detection — distinguishes it from ACM_NAME_YEAR (year between authors and
# title) and other first-name-first formats that don't end entries with year.
_NYE_TRAILING_YEAR_RE = re.compile(
    r",\s+(?:1[89]\d{2}|20\d{2})[a-z]?\.\s*$"
)
_NYE_LOOKAHEAD_LINES = 6


def _count_name_year_end_entry_starts(refs_text: str) -> int:
    """Count NAME_YEAR_END entry-start lines.

    Requires three signals together:
    1. Line starts with ``Firstname Lastname`` (``_FMT_NAME_YEAR_END_RE``).
    2. Previous non-blank line ends with a period (entry-close guard) — same
       guard the splitter uses; rejects wrapped author-list continuations.
    3. A line ending with ``, YEAR.`` appears within the next
       ``_NYE_LOOKAHEAD_LINES`` lines. This is the distinguishing signal vs
       ACM_NAME_YEAR (year between authors and title) — without it, ACL/CHI
       proceedings papers' entries would over-count here and steal detection.
    """
    bib = _before_appendix(refs_text)
    lines = bib.split('\n')
    cleaned = ['' if _NOISE_LINE_RE.match(l) else l for l in lines]
    count = 0
    for i, line in enumerate(cleaned):
        if not _FMT_NAME_YEAR_END_RE.match(line):
            continue
        if i > 0:
            prev = ''
            for j in range(i - 1, -1, -1):
                s = cleaned[j].rstrip()
                if s:
                    prev = s
                    break
            if prev and not prev.endswith('.'):
                continue
        # Year-end marker must appear within the next few lines (entry body).
        end = min(i + _NYE_LOOKAHEAD_LINES, len(cleaned))
        if not any(_NYE_TRAILING_YEAR_RE.search(cleaned[j]) for j in range(i, end)):
            continue
        count += 1
    return count


def _count_chicago_entry_starts(refs_text: str) -> int:
    """Count Chicago entry-start lines with the same continuation guard the
    splitter uses. Lifts the splitter's filter out so the detector and the
    splitter agree on what counts as a real entry."""
    raw_lines = refs_text.split('\n')
    lines = [l for l in raw_lines if not _NOISE_LINE_RE.match(l)]
    count = 0
    for i, line in enumerate(lines):
        if not _CHICAGO_ENTRY_START_RE.match(line):
            continue
        if i > 0:
            prev = lines[i - 1].rstrip()
            if prev and not prev.endswith(_CHICAGO_PREV_TERMINATORS):
                continue
        count += 1
    return count


def _detect_format(refs_text: str) -> CitationFormat:
    """Auto-detect which of the supported citation formats this PDF uses.

    Picks the format with the most matches over the configured threshold.
    Ties are broken by priority: bracketed > plain_numbered > author_year
    (most -> least unambiguous).
    """
    # For NEURIPS detection: use the bibliography-only portion (truncate before
    # any appendix/supplementary heading) so that in-text citations in the
    # appendix don't inflate the AUTHOR_YEAR count and crowd out NEURIPS.
    bib_only = _before_appendix(refs_text)

    counts = {
        CitationFormat.ACM_BRACKETED:  len(_FMT_BRACKETED_RE.findall(refs_text)),
        CitationFormat.LATEX_ALPHA:    len(_FMT_ALPHA_RE.findall(refs_text)),
        CitationFormat.AY_BRACKETED:   len(_FMT_AY_BRACKETED_RE.findall(refs_text)),
        CitationFormat.PLAIN_NUMBERED: len(_FMT_PLAIN_NUM_RE.findall(refs_text)),
        CitationFormat.ACM_NAME_YEAR:  len(_FMT_ACM_NAME_YEAR_RE.findall(refs_text)),
        # Chicago count applies the same prev-line-ends-with-period guard the
        # splitter uses; raw _FMT_CHICAGO_RE.findall inflates on wrapped
        # author-list continuation lines like ``Shakeri, Emanuel Taropa, ...``
        # that look like an entry start in isolation.
        CitationFormat.CHICAGO:        _count_chicago_entry_starts(refs_text),
        CitationFormat.AUTHOR_YEAR:    len(_FMT_AUTHOR_YEAR_RE.findall(refs_text)),
        CitationFormat.NEURIPS:        len(_FMT_NEURIPS_RE.findall(bib_only)),
        CitationFormat.ICML:           len(_FMT_ICML_RE.findall(bib_only)),
        # NAME_YEAR_END count also uses a prev-line guard. Placed LAST in the
        # priority list so it only wins when no numbered/initials/surname-first
        # format matches — otherwise its broad "Firstname Lastname" anchor
        # would steal detection from NEURIPS, ACM_BRACKETED-with-year-at-end,
        # and similar.
        CitationFormat.NAME_YEAR_END:  _count_name_year_end_entry_starts(refs_text),
    }
    priority = [
        CitationFormat.ACM_BRACKETED,
        CitationFormat.LATEX_ALPHA,
        CitationFormat.AY_BRACKETED,
        CitationFormat.PLAIN_NUMBERED,
        CitationFormat.ACM_NAME_YEAR,
        CitationFormat.ICML,
        CitationFormat.NEURIPS,
        CitationFormat.CHICAGO,
        CitationFormat.AUTHOR_YEAR,
        CitationFormat.NAME_YEAR_END,
    ]
    best = max(priority, key=lambda f: (counts[f], -priority.index(f)))
    if counts[best] < _DETECT_MIN_MATCHES:
        raise ValueError(
            "Could not identify a citation format in the References section. "
            f"Tried patterns: bracketed [N] ({counts[CitationFormat.ACM_BRACKETED]} matches), "
            f"LaTeX alpha [Key] ({counts[CitationFormat.LATEX_ALPHA]} matches), "
            f"AY bracketed [Name, YEAR] ({counts[CitationFormat.AY_BRACKETED]} matches), "
            f"plain N. ({counts[CitationFormat.PLAIN_NUMBERED]} matches), "
            f"ACM name-year First Last. YEAR ({counts[CitationFormat.ACM_NAME_YEAR]} matches), "
            f"ICML Surname,Initial ({counts[CitationFormat.ICML]} matches), "
            f"NeurIPS/ICLR Initial.Surname ({counts[CitationFormat.NEURIPS]} matches), "
            f"Chicago Last, First ({counts[CitationFormat.CHICAGO]} matches), "
            f"author-year (YYYY) ({counts[CitationFormat.AUTHOR_YEAR]} matches), "
            f"name-year-end First Last…YEAR. ({counts[CitationFormat.NAME_YEAR_END]} matches). "
            f"Need at least {_DETECT_MIN_MATCHES} matches for any format."
        )
    log.info(
        "Detected citation format: %s (counts=%s)",
        best.value,
        {f.value: counts[f] for f in priority},
    )
    return best


# ---------------------------------------------------------------------------
# Splitters
# ---------------------------------------------------------------------------

def _blocks_from_line_starts(
    lines: list[str],
    entry_starts: list[int],
) -> list[tuple[int, str]]:
    """Build numbered raw blocks from line indices identified as entry starts."""
    blocks: list[tuple[int, str]] = []
    for n, start in enumerate(entry_starts, start=1):
        end = entry_starts[n] if n < len(entry_starts) else len(lines)
        blocks.append((n, '\n'.join(lines[start:end])))
    return blocks


def _split_bracketed(refs_text: str) -> list[tuple[int, str]]:
    """Split "[1] ... [2] ..." bracketed references."""
    refs_text = _before_appendix(refs_text)
    matches = list(_REF_START_RE.finditer(refs_text))
    if not matches:
        return []
    blocks = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(refs_text)
        blocks.append((num, refs_text[start:end]))
    return blocks


def _split_plain_numbered(refs_text: str) -> list[tuple[int, str]]:
    """Split "1. Author..." / "1) Author..." plain numbered references."""
    refs_text = _before_appendix(refs_text)
    matches = list(_PLAIN_NUM_START_RE.finditer(refs_text))
    if not matches:
        return []
    blocks = []
    for i, m in enumerate(matches):
        num = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(refs_text)
        blocks.append((num, refs_text[start:end]))
    return blocks


def _heal_fragmented_lines(refs_text: str) -> str:
    """Join runs of 3+ consecutive single-token lines into one logical line.

    Some PDFs (notably APA-style with narrow columns) split organization-author
    headers across many lines, one token per line, e.g.::

        International
        Organization
        for
        Migration
        (IOM).
        (2019).
        International migration law no. 34 - glossary on migration.

    Without this pre-pass, the AY splitter cannot recognize the fragmented
    surname-led start line as a new entry because no contiguous line contains
    both the surname *and* the (YEAR) marker, so the whole entry gets
    silently absorbed into its predecessor.

    A "fragment" is a non-empty line ≤ 25 chars with no internal whitespace.
    Runs of three or more are joined together along with the following
    non-fragment line (which holds the rest of the entry header).
    """
    lines = refs_text.split('\n')
    healed: list[str] = []
    i = 0
    while i < len(lines):
        cur = lines[i].strip()
        if cur and len(cur) <= 25 and not any(c.isspace() for c in cur):
            j = i + 1
            run = [cur]
            while j < len(lines):
                nxt = lines[j].strip()
                if nxt and len(nxt) <= 25 and not any(c.isspace() for c in nxt):
                    run.append(nxt)
                    j += 1
                else:
                    break
            if len(run) >= 3:
                joined = ' '.join(run)
                if j < len(lines):
                    healed.append(joined + ' ' + lines[j])
                    i = j + 1
                else:
                    healed.append(joined)
                    i = j
                continue
        healed.append(lines[i])
        i += 1
    return '\n'.join(healed)


def _split_author_year(refs_text: str) -> list[tuple[int, str]]:
    """Split author-year references by surname-led lines containing (YYYY)."""
    refs_text = _before_appendix(refs_text)
    refs_text = _heal_fragmented_lines(refs_text)
    raw_lines = refs_text.split('\n')
    lines = [l for l in raw_lines if not _NOISE_LINE_RE.match(l)]

    entry_starts: list[int] = []
    for i, line in enumerate(lines):
        if not _AY_NEW_ENTRY_START_RE.match(line):
            continue
        # Continuation guard: if the previous line ends with a comma AND does
        # not already contain a parenthesized year, treat this line as a
        # wrapped author-list continuation, not a new entry.
        if i > 0:
            prev = lines[i - 1].rstrip()
            if prev.endswith(',') and not _AY_YEAR_PAREN_RE.search(prev):
                continue
        # Strong signal: the parenthesized year is on this line.
        if _AY_YEAR_PAREN_RE.search(line):
            entry_starts.append(i)
            continue
        # Weaker signal: line looks like an author list (has a single-letter
        # initial within the first ~50 chars), and a (YYYY) is in the next
        # few lines. This catches entries that wrap before the year.
        # Pure venue continuation lines ("Algorithmica,", "Conference,") fail
        # the initial check and are correctly skipped.
        if _AY_AUTHOR_INITIAL_RE.search(line[:_AY_AUTHOR_LIST_WINDOW]):
            window = ' '.join(lines[i : i + _AY_LOOKAHEAD_LINES + 1])
            if _AY_YEAR_PAREN_RE.search(window):
                entry_starts.append(i)

    if not entry_starts:
        return []

    return _blocks_from_line_starts(lines, entry_starts)


def _split_latex_alpha(refs_text: str) -> list[tuple[int, str]]:
    """Split "[AGS20] ... [BLSSS20] ..." LaTeX alpha-style references.

    Captured alpha key is stashed as the first line of the raw block (prefixed
    with the sentinel ``__ALPHA_KEY__``) so the block parser can extract it
    and assign it as the cite key.
    """
    refs_text = _before_appendix(refs_text)
    matches = list(_REF_ALPHA_START_RE.finditer(refs_text))
    if not matches:
        return []
    blocks = []
    for i, m in enumerate(matches):
        alpha_key = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(refs_text)
        body = refs_text[start:end]
        raw = f"__ALPHA_KEY__{alpha_key}\n{body}"
        blocks.append((i + 1, raw))
    return blocks


def _split_ay_bracketed(refs_text: str) -> list[tuple[int, str]]:
    """Split "[Surname et al., YEAR] ..." author-year-bracketed references.

    The full bracket key (e.g. "Ahle et al., 2020") is stashed with the
    ``__AY_BRACKET_KEY__`` sentinel so the block parser can recover it.
    """
    refs_text = _before_appendix(refs_text)
    matches = list(_REF_AY_BRACKETED_START_RE.finditer(refs_text))
    if not matches:
        return []
    blocks = []
    for i, m in enumerate(matches):
        ay_key = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(refs_text)
        body = refs_text[start:end]
        raw = f"{_AY_BRACKET_KEY_PREFIX}{ay_key}\n{body}"
        blocks.append((i + 1, raw))
    return blocks


# Matches "and X" at end of line where X is a bare capital (truncated name).
_ACM_NAME_YEAR_WRAP_RE = re.compile(r'\band\s+[A-Z]$')
# Number of following lines to scan for the ". YEAR. " anchor.
_ACM_NAME_YEAR_LOOKAHEAD = 2
# First-name-start pattern (capital + lower): used both for entry-start and
# for the "previous line also looks like an author-list line" continuation guard.
_ACM_NAME_YEAR_NAME_START_RE = re.compile(r'^[A-Z][a-z]')


def _acm_name_year_prev(cleaned: list[str], i: int) -> str:
    """Return the last non-empty cleaned line before index i."""
    for j in range(i - 1, -1, -1):
        s = cleaned[j].strip()
        if s:
            return s
    return ''


def _split_acm_name_year(refs_text: str) -> list[tuple[int, str]]:
    """Split ACM name-year references (First Last, ..., and First Last. YEAR. Title.).

    Uses two passes:

    **Primary**: a line is an entry start when ``_FMT_ACM_NAME_YEAR_RE`` matches
    (i.e., the line itself has `. YEAR[a-z]?. ` with at least one space before
    it, ruling out bare-surname wraps like "Yokoo. 2024.").

    **Secondary (lookahead)**: when the author list wraps before the year (e.g.
    "…, and Makoto\\nYokoo. 2024."), the primary fails on the first line.  The
    secondary fires when the line starts with a name-like token, ends WITHOUT a
    trailing period (venue continuations almost always end with `.`), is not a
    known venue/institution start word, and the next 1–2 lines contain the year
    anchor.

    **Continuation guards** on both paths:
    - Previous non-blank line ends with ``,`` or ``and [A-Z]`` → still inside an
      author list; skip.
    - Previous non-blank line also starts with a name-like token AND ends without
      terminal punctuation → the current line is itself a continuation of the
      author list from the previous line; skip.
    """
    refs_text = _before_appendix(refs_text)

    lines = refs_text.split('\n')
    cleaned: list[str] = ['' if _NOISE_LINE_RE.match(l) else l for l in lines]

    _TERMINAL = frozenset('.)]')

    entry_starts: list[int] = []
    for i, line in enumerate(cleaned):
        if not _ACM_NAME_YEAR_NAME_START_RE.match(line):
            continue

        via_primary = bool(_FMT_ACM_NAME_YEAR_RE.match(line))

        if not via_primary:
            # Secondary: line ends without a period, is not a venue word, and
            # the year anchor appears in the next few lines.
            stripped = line.rstrip()
            if stripped.endswith('.'):
                continue
            if _ACM_NAME_YEAR_VENUE_START_RE.match(line):
                continue
            window = ' '.join(cleaned[i + 1 : i + 1 + _ACM_NAME_YEAR_LOOKAHEAD])
            if not _ACM_NAME_YEAR_IN_WINDOW_RE.search(window):
                continue

        prev = _acm_name_year_prev(cleaned, i)

        # Guard 1: still inside author list (comma or bare initial at line end).
        if prev.endswith(',') or _ACM_NAME_YEAR_WRAP_RE.search(prev):
            continue

        # Guard 2: previous non-blank line is itself a name-like author-list
        # line that ended without terminal punctuation — current line is its
        # continuation (handles "…, Masoud\nSaeed Seddighin, and Hadi Yami.").
        prev_stripped = prev.rstrip()
        if (
            _ACM_NAME_YEAR_NAME_START_RE.match(prev)
            and prev_stripped
            and prev_stripped[-1] not in _TERMINAL
            and not prev_stripped[-1].isdigit()
        ):
            continue

        entry_starts.append(i)

    if not entry_starts:
        return []

    return _blocks_from_line_starts(cleaned, entry_starts)


def _split_chicago(refs_text: str) -> list[tuple[int, str]]:
    """Split Chicago-style references at each "Last, First..." line.

    Entry starts are either ``Surname, Firstname`` lines or the em-dash
    ``———`` "ditto authors" marker. Each block is the text between consecutive
    starts (or from a start to end-of-section). Lines between starts are
    continuation wraps and stay attached to the previous block.

    Continuation guard: a candidate start line is a new entry only when the
    previous line ends with ``.`` (closing the previous entry). This rules
    out wraps like "Growth, United States, 1870-1950" that happen to begin
    a continuation line in the middle of an entry's venue clause.
    """
    raw_lines = refs_text.split('\n')
    lines = [l for l in raw_lines if not _NOISE_LINE_RE.match(l)]

    entry_starts: list[int] = []
    for i, line in enumerate(lines):
        if not _CHICAGO_ENTRY_START_RE.match(line):
            continue
        if i > 0:
            prev = lines[i - 1].rstrip()
            # Continuation: previous line doesn't terminate the citation.
            if prev and not prev.endswith(('.', ']', '"', '”', "'", "’")):
                continue
        entry_starts.append(i)

    if not entry_starts:
        return []

    return _blocks_from_line_starts(lines, entry_starts)


_CHICAGO_ENTRY_START_RE = re.compile(
    r"^(?:[A-Z][\w'À-ſ\-]+,\s+[A-Z][a-zÀ-ſ][\w'À-ſ\-]*\b"
    r"(?!\s+[A-Z][a-z]{2,},)"
    r"|———)"
)

# Recurring conference-paper page header (e.g., "Published as a conference
# paper at ICLR 2024") — these appear between entries in some PDF layouts.
_CONF_HEADER_RE = re.compile(
    r'published\s+as\s+a\s+(?:conference\s+)?paper|'
    r'under\s+review\s+at\s+|'
    r'accepted\s+(?:at|to)\s+',
    re.IGNORECASE,
)

# Entry-start candidate: line begins with "Initial. Word" where Initial is a
# single capital + period, and Word starts with an uppercase letter.
_NEURIPS_LINE_START_RE = re.compile(r'^[A-Z]\.\s+[A-Z]')


def _split_neurips(refs_text: str) -> list[tuple[int, str]]:
    """Split NeurIPS/ICLR-style references (Initial. Surname, ... Title. Venue, Year.)

    Entry starts are detected by the ``^[A-Z]. [A-Z]`` pattern, with two
    continuation guards:

    1. **Previous-line guard**: if the previous non-blank, non-noise line ends
       with a lowercase word (conjunction, preposition, partial sentence), the
       current line is a continuation — e.g., "M. Meila and\\n T. Zhang,
       editors," is the editor list inside a venue clause, not a new entry.
    2. **Abbreviation-chain guard**: if the first word after the initial is
       itself an abbreviation (immediately followed by another ". Word." chain,
       as in "J. Mach. Learn. Res."), the line is a venue continuation, not a
       new entry start.
    """
    refs_text = _before_appendix(refs_text)

    lines = refs_text.split('\n')

    # Mark noise/header lines as empty so the prev-line guard sees through them.
    cleaned: list[str] = []
    for l in lines:
        if _NOISE_LINE_RE.match(l) or _CONF_HEADER_RE.search(l):
            cleaned.append('')
        else:
            cleaned.append(l)

    entry_starts: list[int] = []
    for i, line in enumerate(cleaned):
        if not _NEURIPS_LINE_START_RE.match(line):
            continue

        # Guard 1: previous non-blank line must end with terminal punctuation.
        prev = ''
        for j in range(i - 1, -1, -1):
            s = cleaned[j].strip()
            if s:
                prev = s
                break
        if prev:
            last_word = prev.split()[-1]
            # If the line ended with a bare lowercase word (e.g., "and", "of",
            # "the", or a partial word like "Onl") → continuation.
            if last_word and last_word[-1].isalpha() and last_word[-1].islower():
                continue

        # Guard 2: reject abbreviation chains like "J. Mach. Learn. Res."
        # The "surname" slot after the initial would itself be followed
        # immediately by ". CapWord." — flagging this as a venue abbrev chain.
        m = re.match(r'^[A-Z]\.\s+([A-Z][\w\-\'À-ÿ]+)(.*)', line)
        if m:
            after_surname = m.group(2)
            if _NEURIPS_ABBREV_CHAIN_RE.match(after_surname):
                continue

        entry_starts.append(i)

    if not entry_starts:
        return []

    return _blocks_from_line_starts(cleaned, entry_starts)


def _extract_running_headers(lines: list[str]) -> set[str]:
    """Detect paper running headers embedded in the refs section.

    In many ML PDF venues (ICML, NeurIPS, etc.) the paper's own title is
    repeated as a header at the top of each page. After page-number noise
    lines are removed, these headers appear as standalone lines that:
      - follow a page-number line (possibly with intervening blank lines), AND
      - do not themselves look like a reference entry start.

    We collect all such lines and return them so the splitter can treat them
    as transparent noise (like page numbers).
    """
    headers: set[str] = set()
    prev_was_page_num = False
    for line in lines:
        stripped = line.strip()
        if re.match(r'^\d{1,4}$', stripped):
            prev_was_page_num = True
        elif not stripped:
            # Skip blank lines — keep prev_was_page_num state active so the
            # header on the next non-blank line is still identified.
            pass
        elif prev_was_page_num:
            # First non-blank line after a page number: candidate running header
            if not _ICML_LINE_START_RE.match(stripped):
                headers.add(stripped)
            prev_was_page_num = False
        else:
            prev_was_page_num = False
    return headers


def _split_icml(refs_text: str) -> list[tuple[int, str]]:
    """Split ICML-style references (Surname, F., ..., and Surname, F. Title. Venue, Year.)

    Entries start on lines matching ``^Surname, F.`` (≥3-char surname + comma +
    initial). Two guards prevent false positives:

    1. **Terminal-punctuation guard**: the previous non-blank, non-noise line
       must end with a period, closing bracket/paren, or digit — meaning the
       prior entry finished. Lines ending with a comma or bare word are
       continuations of a wrapped author list or venue text.
    2. **Running-header transparency**: the paper's own title often appears as
       a page header between entries. We detect such repeated lines and skip
       them when evaluating the previous-line guard.
    """
    refs_text = _before_appendix(refs_text)

    lines = refs_text.split('\n')

    # Identify running headers (paper title repeated at the top of each page)
    running_headers = _extract_running_headers(lines)

    # Mark noise/header lines as empty so the prev-line guard sees through them.
    cleaned: list[str] = []
    for l in lines:
        stripped = l.strip()
        if _NOISE_LINE_RE.match(l) or stripped in running_headers:
            cleaned.append('')
        else:
            cleaned.append(l)

    _TERMINAL_PUNCT = frozenset('.)]')

    entry_starts: list[int] = []
    for i, line in enumerate(cleaned):
        if not _ICML_LINE_START_RE.match(line):
            continue

        # Guard: the previous non-blank line must end with terminal punctuation.
        # Lines ending with comma, lowercase word, or partial text indicate a
        # wrapped author list or venue continuation — not a new entry.
        prev = ''
        for j in range(i - 1, -1, -1):
            s = cleaned[j].strip()
            if s:
                prev = s
                break
        if prev:
            last_ch = prev[-1]
            if last_ch not in _TERMINAL_PUNCT and not last_ch.isdigit():
                continue

        entry_starts.append(i)

    if not entry_starts:
        return []

    return _blocks_from_line_starts(cleaned, entry_starts)


def _split_name_year_end(refs_text: str) -> list[tuple[int, str]]:
    """Split name-year-end references where each entry begins with a full
    ``Firstname Lastname`` line and ends with ``, YEAR.`` (year at end).

    Entries are unnumbered and the bibliography has no other splittable
    anchor, so the strategy is:

    1. Truncate at the appendix heading (if any) so appendix section titles
       like ``Carbon Footprint`` don't get picked up as entry starts.
    2. Walk lines; a line is an entry start when ``_FMT_NAME_YEAR_END_RE``
       matches AND the previous non-blank line ends with a period (closing
       the previous entry). The same prev-line guard the ICML/NeurIPS
       splitters use, applied here to reject wrapped-author-list
       continuation lines like ``Shakeri, Emanuel Taporta, ...``.
    """
    refs_text = _before_appendix(refs_text)

    lines = refs_text.split('\n')
    cleaned: list[str] = ['' if _NOISE_LINE_RE.match(l) else l for l in lines]

    entry_starts: list[int] = []
    for i, line in enumerate(cleaned):
        if not _FMT_NAME_YEAR_END_RE.match(line):
            continue
        if i > 0:
            prev = ''
            for j in range(i - 1, -1, -1):
                s = cleaned[j].rstrip()
                if s:
                    prev = s
                    break
            if prev and not prev.endswith('.'):
                continue
        entry_starts.append(i)

    if not entry_starts:
        return []

    return _blocks_from_line_starts(cleaned, entry_starts)


def _split_into_blocks(refs_text: str) -> list[tuple[int, str]]:
    """Backward-compatible alias for the bracketed splitter."""
    return _split_bracketed(refs_text)


_SPLITTERS: dict[CitationFormat, Callable[[str], list[tuple[int, str]]]] = {
    CitationFormat.ACM_BRACKETED:  _split_bracketed,
    CitationFormat.PLAIN_NUMBERED: _split_plain_numbered,
    CitationFormat.AUTHOR_YEAR:    _split_author_year,
    CitationFormat.LATEX_ALPHA:    _split_latex_alpha,
    CitationFormat.AY_BRACKETED:   _split_ay_bracketed,
    CitationFormat.ACM_NAME_YEAR:  _split_acm_name_year,
    CitationFormat.CHICAGO:        _split_chicago,
    CitationFormat.NEURIPS:        _split_neurips,
    CitationFormat.ICML:           _split_icml,
    CitationFormat.NAME_YEAR_END:  _split_name_year_end,
}


# Sentinel prefix that `_split_latex_alpha` prepends to each raw block so the
# block parser can recover the alpha cite key.
_ALPHA_KEY_PREFIX = "__ALPHA_KEY__"
_ALPHA_KEY_RE = re.compile(r"^__ALPHA_KEY__([^\n]+)\n?", re.DOTALL)

_AY_BRACKET_KEY_PREFIX = "__AY_BRACKET_KEY__"
_AY_BRACKET_KEY_RE = re.compile(r"^__AY_BRACKET_KEY__([^\n]+)\n?", re.DOTALL)

# Trailing-year grammar: "Authors. Title. Venue, ..., YEAR."
# Used for LaTeX alpha entries where the year is at the END (no parens, no
# leading-year ACM-style anchor). Three lazy groups separated by ". " catch
# the canonical Author/Title/Venue split; year is found by `_LATEX_ALPHA_YEAR_RE`
# below. The `[a-z]` lookbehind on the period boundaries ensures we don't
# break on initials like "L." or "C." in author lists, which end in uppercase.
_LATEX_ALPHA_BLOCK_RE = re.compile(
    r"^(?P<authors>.+?[a-z])\.\s+(?P<title>.+?[a-z])\.\s+(?P<rest>.+)$",
    re.DOTALL,
)

# BibLaTeX style: Authors. "Title". In: Venue Volume.Issue (Year), pp. Pages.
# Handles quoted titles (common in biblatex/Chicago exports) where the
# unquoted parser picks the wrong period as the title/venue boundary.
# Matches both straight quotes (") and curly/smart quotes (“ / ”).
_LATEX_ALPHA_QUOTED_RE = re.compile(
    r'^(?P<authors>.+?)\.\s+[“"](?P<title>[^”"]+?)[”"]\.\s+(?P<rest>.+)$',
    re.DOTALL,
)

# Year in the 1800/1900/2000s only — excludes spurious matches like "1481"
# inside identifiers (e.g. "SOCO-1481"). Matches years preceded by:
#   ,  — comma (books/reports: "Publisher, 2009")
#   (  — open paren (journal volumes: "158 (2018), pp.")
#   .  — period+space (proceedings: "e-Energy. 2022, pp.")
# Negative lookahead prevents matching "2013" in academic-year ranges like "2013-14".
_LATEX_ALPHA_YEAR_RE = re.compile(
    r'(?:[,\(]\s*|\.\s+)(1[89]\d{2}|20\d{2})\b(?!\s*[-–]\d)'
)

# Fallback grammar for LaTeX alpha entries without a "Title. Venue." split —
# common for arXiv preprints, reports, and book-like entries: just
# "Authors. Title, YEAR." (anchored at end so the title doesn't gobble venue).
_LATEX_ALPHA_NOVENUE_RE = re.compile(
    r"^(?P<authors>.+?[a-z])\.\s+(?P<title>.+?),\s+"
    r"(?P<year>1[89]\d{2}|20\d{2})\.?\s*$",
    re.DOTALL,
)


# ---------------------------------------------------------------------------
# Block-level parsing
# ---------------------------------------------------------------------------

def _clean_text_for_grammar(text: str) -> str:
    """Strip trailing DOI/URL/ISBN/publisher noise so year-end regexes can anchor.

    USENIX and full ACM DL entries append metadata after the year:
      "... 2024. Association for Computing Machinery. ISBN 978... doi: X. URL Y."
    Stripping these lets _NUMBERED_BARE_YEAR_RE find the year at '$'.
    Returns the original text unchanged if stripping would leave nothing useful.
    """
    # Strip "doi: 10.xxx/yyy" and "URL https://..." labels+values together
    t = re.sub(r'\b(?:doi|DOI)\s*:\s*\S+', '', text)
    t = re.sub(r'\bURL\s+https?://\S+', '', t)
    t = _URL_RE.sub('', t)       # bare https:// without URL label
    t = _ARXIV_RE.sub('', t)
    t = _ISBN_ISSN_STRIP_RE.sub('', t)
    t = _ACCESSED_STRIP_RE.sub('', t)
    t = _TRAILING_PUBLISHER_RE.sub('', t)
    # Strip trailing bare numeric IDs (5+ digits, e.g. ISSN online IDs or PubMed IDs)
    # that are NOT years — they prevent _NUMBERED_BARE_YEAR_RE from anchoring.
    t = re.sub(r'[,\s]+\d{5,}\.?\s*$', '', t)
    t = re.sub(r'\s+', ' ', t).strip().rstrip('.,;:')
    return t if len(t) > 10 else text


def _parse_block(num: int, raw: str, grammar: str = 'unknown') -> BibEntry:
    """Parse one bibliography block using the detected content grammar.

    When ``grammar`` is ``'unknown'`` (the default), tries every known
    sub-grammar in the original cascade order, preserving backward
    compatibility.  When a specific grammar is supplied (detected by
    ``_detect_content_grammar``), only that grammar and its fallbacks are
    tried, avoiding false positives from unrelated grammars.

    Grammar → priority list of sub-grammars:
      - ``'acm'``:       ``['acm', 'bare_year']``
      - ``'elsevier'``:  ``['elsevier', 'bare_year']``
      - ``'ay_inline'``: ``['ay_inline', 'bare_year']``
      - ``'vancouver'``: ``['vancouver', 'vancouver_corp']``
      - ``'nature'``:    ``['nature', 'bare_year']``
      - ``'bare_year'``: ``['bare_year']``
      - ``'unknown'``:   full cascade (same order as before)
    """
    text = _collapse_block(raw)

    doi = _extract_doi(text)
    eprint = _extract_arxiv(text)
    archiveprefix = "arXiv" if eprint else None
    url = _extract_url(text, doi, eprint)

    # Grammar matchers anchor year to end-of-string; strip trailing metadata
    # (DOI, URL, ISBN, publisher names) so those anchors work even when the
    # citation includes full ACM DL / USENIX-style metadata after the year.
    text_g = _clean_text_for_grammar(text)

    year: Optional[int] = None
    authors: list[str] = []
    title: Optional[str] = None

    sub_grammars = _BLOCK_GRAMMAR_ORDER.get(grammar, _BLOCK_GRAMMAR_ORDER['unknown'])

    matched = False
    for sg in sub_grammars:
        if sg == 'acm':
            m = _ACM_YEAR_SPLIT_RE.match(text_g)
            if m:
                year = int(m.group(2))
                authors = _parse_pdf_authors(m.group(1).strip())
                title = _extract_title(m.group(3).strip())
                matched = True
                break
        elif sg == 'elsevier':
            els = _parse_elsevier_block(text_g)
            if els is not None:
                author_str, year, title = els
                authors = _parse_pdf_authors(author_str)
                matched = True
                break
        elif sg == 'ay_inline':
            ay = _AY_BLOCK_RE.match(text_g)
            if ay:
                year = int(ay.group("year")) if ay.group("year") else None
                author_str = ay.group("authors").strip().rstrip('.,')
                authors = _parse_pdf_authors_ay(author_str) if author_str else []
                title = _extract_title(ay.group("rest").strip())
                matched = True
                break
        elif sg == 'vancouver':
            vc = _parse_vancouver_block(text_g)
            if vc is not None:
                author_str, year, title = vc
                authors = _parse_pdf_authors_vancouver(author_str)
                matched = True
                break
        elif sg == 'nature':
            nat = _parse_nature_block(text_g)
            if nat is not None:
                author_str, year, remainder = nat
                authors = _parse_pdf_authors_ay(author_str) if author_str else []
                title = _extract_title(remainder)
                matched = True
                break
        elif sg == 'vancouver_corp':
            vcc = _parse_vancouver_corporate_block(text_g)
            if vcc is not None:
                author_str, year, title = vcc
                authors = [author_str.rstrip('.,;')] if author_str else []
                matched = True
                break
        elif sg == 'bare_year':
            nb = _parse_numbered_bare_year_block(text_g)
            if nb is not None and (nb[1] is not None or nb[2]):
                author_str, year, title = nb
                authors = _parse_pdf_authors(author_str) if author_str else []
                matched = True
                break

    if not matched:
        ym = _YEAR_RE.search(text)
        if ym:
            year = int(ym.group(1))
        log.debug(
            "Reference [%d] matched no known grammar; partial parse only.",
            num,
        )

    return BibEntry(
        key=f"ref{num}",
        entry_type="article",
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        url=url,
        eprint=eprint,
        archiveprefix=archiveprefix,
        raw_fields={"raw_text": text, "first_token": _first_token(text)},
    )


# Aliases retained for backward compatibility — both block parsers now share
# the same grammar-trying core, since reference grammar is independent of how
# entries are numbered (bracketed, plain, or author-year).
_parse_numbered_block = _parse_block
_parse_author_year_block = _parse_block


def _parse_elsevier_block(text: str) -> Optional[tuple[str, int, str]]:
    """Parse "F.M. Last, ..., Title, Venue Vol (YEAR) pp." form.

    Returns ``(authors_str, year, title)`` or ``None``. Authors must match the
    initials-first comma-separated pattern at the start of the block; the year
    must appear in ``(YYYY)`` parens (journals) or trailing ``, YYYY`` (books
    and proceedings). The title/venue boundary is found by looking for a comma
    followed by a venue-start signal (abbreviation chain, all-caps acronym,
    publisher name, ``vol.``, or ``in:``); if no signal is found, the title is
    cut at the last comma before the year.
    """
    am = _ELSEVIER_AUTHORS_RE.match(text)
    if not am:
        return None
    authors_str = am.group(0)
    after = text[am.end():].lstrip(',').lstrip()
    if not after:
        return None
    and_author = _ELSEVIER_AND_FINAL_AUTHOR_RE.match(after)
    if and_author:
        authors_str = f"{authors_str}, {and_author.group(1)}"
        after = after[and_author.end():]
    if after.startswith('.') or after.lower().startswith('and '):
        return None

    year: Optional[int] = None
    year_pos: Optional[int] = None
    ym = _ELSEVIER_YEAR_PAREN_RE.search(after)
    if ym:
        year = int(ym.group(1))
        year_pos = ym.start()
    else:
        bm = _ELSEVIER_YEAR_BARE_RE.search(after)
        if bm:
            year = int(bm.group(1))
            year_pos = bm.start()
    if year is None or year_pos is None:
        return None

    cm = _ELSEVIER_VENUE_START_RE.search(after, 0, year_pos)
    if cm:
        title = after[:cm.start()].strip().rstrip('.,;')
    else:
        cut = after.rfind(',', 0, year_pos)
        if cut > 0:
            title = after[:cut].strip().rstrip('.,;')
        else:
            title = after[:year_pos].strip().rstrip('.,;')

    return authors_str, year, title


def _parse_vancouver_block(text: str) -> Optional[tuple[str, Optional[int], Optional[str]]]:
    """Parse "Surname I, Surname I, ..., et al. Title. Venue YEAR;vol:pp." form.

    Returns ``(authors_str, year, title)`` or ``None``. The authors prefix
    must match the surname-with-bare-initials pattern; corporate authors
    (e.g. "Global Initiative for Chronic Obstructive Lung Disease") and any
    other shape fall through to ``_parse_vancouver_corporate_block``.
    """
    am = _VANCOUVER_AUTHORS_RE.match(text)
    if not am:
        return None
    authors_str = am.group("authors")
    after = text[am.end():]

    # If the match ends on a bare initial (no period) and is followed by a
    # space + uppercase letter, the entry is in Western "First Initial Surname"
    # order (e.g. "Marianne F Touchie and …"), not Vancouver "Surname Initials"
    # order.  Skip when authors_str ends with "." — that covers "et al." and
    # properly terminated single-author blocks.
    if (
        len(after) >= 2
        and after[0] == ' '
        and after[1].isupper()
        and not authors_str.endswith('.')
    ):
        return None

    tm = _VANCOUVER_TITLE_RE.match(after)
    if not tm:
        return None
    title = tm.group("title").strip().rstrip('.,;:')
    # Reject if the extracted "title" looks like an author-name fragment.
    # This happens when the Vancouver regex grabs only the first name token
    # (e.g. "Jimi B" from "Jimi B. Oke, …") and the ". " separator then
    # makes "Oke, Youssef M" appear to be the title.  A real title almost
    # never starts with a single capitalized word followed immediately by
    # a comma ("LastName,").
    if re.match(r'^[A-Z][a-z]+,\s+[A-Z]', title):
        return None
    rest = tm.group("rest").strip()

    year: Optional[int] = None
    ym = _VANCOUVER_YEAR_SEP_RE.search(rest)
    if ym is None:
        ym = _VANCOUVER_YEAR_RE.search(rest)
    if ym is None:
        # Last-resort fallback: scan the title region too, for "epub ahead of
        # print" variants where the year is embedded in a "[published online
        # ahead of print MONTH DD, YEAR]" annotation appended to the title.
        ym = _VANCOUVER_YEAR_RE.search(title)
    if ym:
        year = int(ym.group(1))

    return authors_str, year, title or None


def _looks_corporate(head: str) -> bool:
    """Return True if a short uninterrupted phrase looks like a corporate-author
    name rather than a sentence-cased title.

    Heuristic: ≤ 6 whitespace-delimited words, all "content" words begin with
    an uppercase letter (small connector words ``of``, ``for``, ``and``,
    ``the`` may be lowercase). Title-cased title fragments like "Central air
    conditioning energy saver" fail because most words start lowercase.
    """
    words = head.strip().split()
    if not words or len(words) > 6:
        return False
    if len(words) == 1 and len(words[0].rstrip(",.;:").lstrip("(").rstrip(")")) == 1:
        return False
    lowers = {"of", "for", "and", "the", "to", "in", "on", "at", "by"}
    for w in words:
        bare = w.rstrip(",.;:").lstrip("(").rstrip(")")
        if not bare:
            return False
        if bare.lower() in lowers:
            continue
        if not bare[0].isupper():
            return False
    return True


def _parse_numbered_bare_year_block(
    text: str,
) -> Optional[tuple[str, Optional[int], Optional[str]]]:
    """Parse "[N] Authors. Title. Venue, pages, YEAR." (numbered with bare
    trailing year — common for ACM proceedings, IMWUT/Ubicomp, engineering
    journals).

    Author-list end is detected by, in order:
      1. ``", et al."`` marker
      2. ``"and X."`` / ``"& X."`` final-author connector
      3. A leading corporate-name phrase (≤ 6 words, all capitalised) before
         the first ``". "`` — covers single-author web/org references.
      4. Otherwise no authors (web/title-only entries).

    Returns ``(authors_str_or_empty, year, title)`` or ``None`` when no bare
    trailing year is found (callers fall through to other grammars).
    """
    ym = _NUMBERED_BARE_YEAR_RE.search(text)
    if ym is None:
        return None
    year = int(ym.group(1))
    pre_year = text[: ym.start()].rstrip(" .,;")

    # Strategy 1: "et al."
    et_al = _NUMBERED_ET_AL_END_RE.search(pre_year)
    # Strategy 2: "and X."
    and_end = _NUMBERED_AND_AUTHOR_END_RE.search(pre_year)

    # Take the LATER of the two if both match — handles entries like
    # "A, B, et al. Title. ..." (et al. wins) vs "A, B, and C. Title. ..." (and wins).
    author_str = ""
    after = pre_year
    if et_al and (not and_end or et_al.start() > and_end.start()):
        author_str = pre_year[: et_al.start()].rstrip(",. ")
        after = pre_year[et_al.end():].strip()
    elif and_end:
        author_str = pre_year[: and_end.end() - 2].strip()  # exclude trailing ". "
        after = pre_year[and_end.end():].strip()
        # The regex stops at the first ". " after "and X" — X may be a middle
        # initial rather than the final surname (e.g. "and A." in "A. H. G. Rinnooy Kan").
        # Detect whether and_end stopped at a bare initial ("and A.") by checking
        # if the tail of the match is just one uppercase letter + ". ".
        and_tail = re.sub(r'^(?:,\s*and|and|&)\s+', '', and_end.group(0).strip())
        stopped_at_initial = bool(re.match(r'^[A-Z]\.\s*$', and_tail))
        if stopped_at_initial:
            # Step 1: consume additional single-letter initials (e.g. "H. G.")
            for _ in range(6):
                m_init = re.match(r'^([A-Z])\.\s+', after)
                if m_init and re.search(r'(?:\s[A-Z]|\.[A-Z])$', author_str):
                    author_str += '. ' + m_init.group(1)
                    after = after[m_init.end():]
                else:
                    break
            # Step 2: consume the surname — single-word ("Frasconi", "Kan") or
            # hyphenated ("Ben-Akiva"). Compound multi-word surnames
            # ("Rinnooy Kan") are handled only when the loop above consumed
            # additional initials, leaving author_str ending in ". [A-Z]";
            # otherwise we restrict to a single name token to avoid eating
            # title words. ``stopped_at_initial`` is a strong enough signal
            # on its own that the immediately-following word is the surname.
            if re.search(r'\. [A-Z]$', author_str):
                # Multi-token surname allowed after consumed-initial chain
                m_ext = re.match(
                    r'^([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-\']+(?:\s+[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-\']+)*)'
                    r'\.\s+',
                    after,
                )
            else:
                # No prior initials consumed: the bare "and X." stop signals
                # that ``after`` begins with X's single-token surname (e.g.
                # "and P. Frasconi." → and_end stopped at "and P.", leaving
                # ``after = 'Frasconi. ...'``). Consume one name token —
                # hyphenated or plain — terminated by ". ".
                m_ext = re.match(
                    r"^([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-']+)\.\s+",
                    after,
                )
            if m_ext:
                author_str += '. ' + m_ext.group(1)
                after = after[m_ext.end():]
        elif re.search(r'(?:^|\s)[A-Z][a-zÀ-ÿ]+\s+(?:[A-Z]\.?){1,4}$', author_str):
            # The generic author-end regex can stop after dotted middle
            # initials in a final author like "Danny H.K. Tsang."; consume
            # the surname before title extraction.
            m_ext = re.match(
                r"^([A-Za-zÀ-ÿ][A-Za-zÀ-ÿ\-']+)\.\s+",
                after,
            )
            if m_ext:
                author_str += '. ' + m_ext.group(1)
                after = after[m_ext.end():]
    else:
        # Strategy 3: single initialed personal author.
        single_author = _NUMBERED_SINGLE_AUTHOR_END_RE.match(pre_year)
        if single_author:
            author_str = single_author.group(1).strip()
            after = pre_year[single_author.end():].strip()
        else:
            # Strategy 4: leading corporate-name phrase.
            pm = re.search(r'\.\s+', pre_year)
            if pm:
                head = pre_year[: pm.start()].strip()
                if _looks_corporate(head):
                    author_str = head
                    after = pre_year[pm.end():].strip()
                else:
                    # Strategy 5: no authors
                    author_str = ""
                    after = pre_year
            else:
                author_str = ""
                after = pre_year

    title = _extract_title(after) if after else None

    # Fallback for URL-only reference entries of the form "Title. URL. Online;
    # accessed ...".  The text before the first ". " looks like a name phrase
    # (may even satisfy _looks_corporate), but `after` is nothing but a URL
    # and an access notice — so the "author" is really the title.
    if not title and author_str and after:
        after_no_url = _URL_RE.sub('', after).strip().lstrip('.,;').strip()
        if not after_no_url or after_no_url.lower().startswith('online'):
            title = author_str.rstrip('.,;')
            author_str = ""

    return author_str, year, title


def _parse_vancouver_corporate_block(
    text: str,
) -> Optional[tuple[str, Optional[int], Optional[str]]]:
    """Fallback for Vancouver entries with corporate/no personal authors.

    Format: ``[Corp Author. ]Title. Venue YEAR;vol:pages.`` (the corporate-
    author prefix is optional). Anchors on the ``YEAR;`` or ``YEAR:`` Vancouver
    separator, then walks back: everything before the LAST "<period> <Capital>"
    is the title (or "Corp Author. Title" when one more period is present).

    Returns ``(authors_str_or_empty, year, title)`` or ``None`` if no Vancouver-
    shaped year separator is found.
    """
    ym = _VANCOUVER_YEAR_SEP_RE.search(text)
    if ym is None:
        ym = _VANCOUVER_YEAR_PUB_RE.search(text)
    if ym is None:
        return None
    year = int(ym.group(1))
    pre_year = text[: ym.start()].rstrip().rstrip(';').rstrip()

    venue_starts = list(re.finditer(r'\.\s+[A-Z]', pre_year))
    if not venue_starts:
        return None
    prefix = pre_year[: venue_starts[-1].start()]

    inner_splits = list(re.finditer(r'\.\s+[A-Z]', prefix))
    if inner_splits:
        cut = inner_splits[-1].start()
        author_str = prefix[:cut].strip()
        title = prefix[cut + 1:].strip().lstrip('.').strip()
    else:
        author_str = ""
        title = prefix.strip()

    return author_str, year, title or None


def _parse_nature_block(text: str) -> Optional[tuple[str, int, str]]:
    """Parse "Authors. Title. Venue (Year)." form.

    Returns ``(authors_str, year, rest_for_title)`` if the trailing
    parenthesized year is found, otherwise ``None``. ``authors_str`` may be
    empty for corporate/anonymous reports (block has no recognizable author
    list); in that case the entire head text is the title.
    """
    matches = list(_NATURE_TRAILER_RE.finditer(text))
    if not matches:
        return None
    # Use the LAST year-bearing paren; entries occasionally have trailing
    # URLs or notes after the (Publisher, Year) marker.
    m = matches[-1]
    trailing_year = _NUMBERED_BARE_YEAR_RE.search(text)
    if trailing_year and trailing_year.start() >= m.end():
        return None
    ym = re.search(r'\b(1[89]\d{2}|20\d{2})\b', m.group(1))
    if not ym:
        return None
    year = int(ym.group(1))
    head = text[: m.start()].strip().rstrip('.')
    authors_str, rest = _split_nature_authors(head)
    return authors_str, year, rest


def _split_nature_authors(head: str) -> tuple[str, str]:
    """Split a Nature-style head into (authors, rest) using author terminators.

    Tries, in order: ``et al.`` marker, ``& Last, F.`` last-author marker,
    or a single ``Last, F.`` prefix. Falls through to ``("", head)`` when
    no author syntax is recognized (corporate/anonymous reports).
    """
    m = _NATURE_ETAL_END_RE.search(head)
    if m:
        return head[: m.end()].rstrip(), head[m.end():].strip()
    m = _NATURE_AMP_END_RE.search(head)
    if m:
        return head[: m.end()].rstrip(), head[m.end():].strip()
    m = _NATURE_SINGLE_RE.match(head)
    if m:
        rest = head[m.end():].strip()
        # Avoid mis-matching: a real title starts with a capital letter.
        if rest and rest[0].isupper():
            return head[: m.end()].rstrip(), rest
    return "", head


# ---- Chicago grammar -------------------------------------------------- #
# Article form: 'Authors. "Title." Venue ...(Date): pages.'
# Book form:    'Authors. Title. City: Publisher, YEAR.'
# Chapter form: 'Authors. "Chapter." In Title, edited by X, pp. Publisher, YEAR.'
#
# Authors block always ends with `.\s+` followed by either a quotation mark
# (article/chapter) or a capitalized title word (book). Mid-author periods —
# initials ("W. Helbock"), abbreviations ("Jr.") — are NOT boundaries.
#
# Year regex: prefers years preceded by `,` or `(` (the conventional Chicago
# anchor "City: Publisher, YYYY." or "(Month YYYY)" or "(YYYYa)"). Takes the
# LAST such match to handle "1969 [1835]" republication-bracket entries.

_CHICAGO_OPEN_QUOTES = '"“‘'
_CHICAGO_CLOSE_QUOTES = '"”’'
# Article-form publication date: year inside parens, possibly with a month
# prefix ("(Jan. 1927)", "(1967a)", "(Winter 1967)"). The most reliable signal
# in Chicago articles; prefer this over any other year.
_CHICAGO_YEAR_PAREN_RE = re.compile(
    r'\(\s*[^)]*?\b(1[789]\d{2}|20\d{2})[a-z]?\s*\)'
)
# Book-form publication date: year after a comma right before final punctuation,
# e.g. "City: Publisher, 1994." or "Working Paper, 2010."
_CHICAGO_YEAR_COMMA_RE = re.compile(
    r',\s+(1[789]\d{2}|20\d{2})[a-z]?\b'
)
# Single em-dash followed by enough of a sequence to constitute the
# "ditto authors" indicator (3 em-dashes or 6+ hyphens).
_CHICAGO_DITTO_RE = re.compile(r'^(?:———|------)\.?\s*')

# Abbreviation tokens that look like initials but aren't (so the authors
# scanner should not treat them as terminating periods).
_CHICAGO_ABBREV_NONTERMINAL = frozenset({
    'no', 'vol', 'ed', 'eds', 'etc', 'jr', 'sr', 'al', 'pp',
    'rev', 'inc', 'co', 'corp', 'op', 'cit',
})


def _find_chicago_authors_end(text: str) -> Optional[int]:
    """Return the index just after the period-space terminating the authors.

    Walks ``\\.\\s+`` boundaries in order. Each candidate is accepted as the
    real boundary only when:

      - the word *immediately* preceding the period is NOT a single capital
        letter (i.e. not a middle initial like ``W.``),
      - the word is NOT a short non-terminal abbreviation (``no``, ``vol``,
        ``ed``, ``Jr``, …),
      - the character starting the next chunk is a quotation mark (article)
        or a capital letter (book title).

    A quotation-mark next char overrides the initial-skipping rule — quoted
    titles unambiguously terminate the author block even after an initial.
    """
    for m in re.finditer(r'\.\s+', text):
        period_pos = m.start()
        end_pos = m.end()
        word_start = period_pos
        while word_start > 0 and text[word_start - 1].isalpha():
            word_start -= 1
        word = text[word_start:period_pos]

        if end_pos >= len(text):
            continue
        next_char = text[end_pos]

        if next_char in _CHICAGO_OPEN_QUOTES:
            return end_pos
        if len(word) == 1 and word.isupper():
            # Single-capital before this period is an initial. It's a
            # middle-initial continuation ("W. Helbock", "R. Haines,") only
            # when the next chunk is a Surname token followed by `.` or `,`
            # (more author content). Otherwise the next chunk is the title
            # ("E. The American Mail:..."), and this period IS the boundary.
            tail = text[end_pos: end_pos + 100]
            if re.match(r"[A-Z][\w\-'À-ſ]+[.,]", tail):
                continue
            return end_pos
        if word.lower() in _CHICAGO_ABBREV_NONTERMINAL:
            continue
        if not next_char.isupper():
            continue
        return end_pos
    return None


def _parse_chicago_block(text: str) -> Optional[tuple[Optional[str], Optional[int], Optional[str], bool]]:
    """Parse a Chicago-style block.

    Returns ``(authors_str, year, title, is_ditto)`` or ``None``. ``is_ditto``
    is ``True`` when the block starts with the em-dash repeated-author marker;
    callers should inherit the author list from the previous entry.
    """
    is_ditto = bool(_CHICAGO_DITTO_RE.match(text))
    if is_ditto:
        text = _CHICAGO_DITTO_RE.sub('', text)

    authors_str: Optional[str] = None
    if not is_ditto:
        end = _find_chicago_authors_end(text)
        if end is None:
            return None
        authors_str = text[: end].rstrip().rstrip('.').rstrip()
        rest = text[end:]
    else:
        rest = text

    title = _extract_chicago_title(rest)
    year = _extract_chicago_year(rest)
    return authors_str, year, title, is_ditto


def _extract_chicago_title(rest: str) -> Optional[str]:
    """Extract the article/book title from the post-author remainder.

    For quoted titles, the *terminating* closing quote is the one followed by
    venue text (whitespace + capital letter, or end of string), not just the
    first closing quote — titles often contain nested quotes (``"'ChatGPT
    seems too good to be true':..."``) or curly apostrophes that look like
    closing quotes (``"Don't..."``).
    """
    rest = rest.lstrip()
    if rest and rest[0] in _CHICAGO_OPEN_QUOTES:
        close_class = '[' + re.escape(_CHICAGO_CLOSE_QUOTES + '"') + ']'
        # Anchor on the publication-year position: the terminating closing
        # quote is the LAST one before the year. This correctly handles
        # titles with embedded quoted phrases (`"X 'inner' Y."`) and stray
        # closing-quote characters used as apostrophes (`"Don't..."`),
        # neither of which can be disambiguated by a structural lookahead.
        year_m = re.search(r'\(\s*(?:1[789]\d{2}|20\d{2})\b', rest)
        search_end = year_m.start() if year_m else len(rest)
        closes = list(re.finditer(close_class, rest[:search_end]))
        # Drop any close-quote at position 0 (defensive — shouldn't happen
        # since rest[0] is an OPEN quote, but apostrophe ambiguity is real).
        closes = [m for m in closes if m.start() > 0]
        if not closes:
            # Fall back to structural lookahead if no year found.
            terminator = re.compile(
                close_class + r'(?=\s*$|\.?\s+["“‘]?[A-Z])'
            )
            m = terminator.search(rest, 1)
            if m is None:
                m = re.search(close_class, rest, flags=0)
            if m is None:
                return None
        else:
            m = closes[-1]
        title = rest[1: m.start()].strip()
        return title.rstrip('.,;:') or None
    # Book / chapter: title is everything up to the first ``. `` boundary,
    # which precedes the venue/publisher chunk.
    m = re.search(r'\.\s+', rest)
    if m:
        title = rest[: m.start()].strip()
        return title.rstrip('.,;:') or None
    return rest.strip().rstrip('.,;:') or None


def _extract_chicago_year(rest: str) -> Optional[int]:
    """Extract publication year from the post-author remainder.

    Order of preference:
      1. Year inside parens (article publication date, e.g. ``(Jan. 1927)``,
         ``(1967a)``) — most reliable signal in Chicago articles.
      2. Year after a comma at the end of the citation (book/report form,
         e.g. ``Publisher, 1994.``) — takes the LAST such match so
         republication brackets like ``Doubleday, 1969 [1835]`` pick 1969.
      3. Last bare year as a final fallback.
    """
    pm = list(_CHICAGO_YEAR_PAREN_RE.finditer(rest))
    if pm:
        return int(pm[-1].group(1))
    cm = list(_CHICAGO_YEAR_COMMA_RE.finditer(rest))
    if cm:
        return int(cm[-1].group(1))
    bare = list(re.finditer(r'\b(1[789]\d{2}|20\d{2})\b', rest))
    return int(bare[-1].group(1)) if bare else None


def _parse_block_chicago(num: int, raw: str) -> BibEntry:
    """Parse one Chicago block. Repeated-author (``———``) entries leave the
    author list empty; ``_resolve_chicago_ditto`` fills them in afterwards.
    """
    text = _collapse_block(raw)
    doi = _extract_doi(text)
    eprint = _extract_arxiv(text)
    archiveprefix = "arXiv" if eprint else None
    url = _extract_url(text, doi, eprint)

    year: Optional[int] = None
    authors: list[str] = []
    title: Optional[str] = None
    is_ditto = False

    parsed = _parse_chicago_block(text)
    if parsed is not None:
        author_str, year, title, is_ditto = parsed
        if author_str:
            authors = _parse_pdf_authors_chicago(author_str)
    else:
        ym = _YEAR_RE.search(text)
        if ym:
            year = int(ym.group(1))

    raw_fields: dict = {"raw_text": text, "first_token": _first_token(text)}
    if is_ditto:
        raw_fields["chicago_ditto"] = True

    return BibEntry(
        key=f"ref{num}",
        entry_type="article",
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        url=url,
        eprint=eprint,
        archiveprefix=archiveprefix,
        raw_fields=raw_fields,
    )


def _resolve_chicago_ditto(entries: list[BibEntry]) -> None:
    """Fill in author lists and surname-tokens for ``———.`` marker entries.

    Walks the entry list in order; each ditto entry inherits the authors AND
    the first-token surname of the most recent non-ditto entry before it.
    Inheriting the first-token matters because `_assign_cite_keys` uses it to
    build the human-readable citation key (e.g. "David1967b" instead of
    "ref8").
    """
    last_authors: list[str] = []
    last_first_token: str = ""
    for e in entries:
        is_ditto = bool(e.raw_fields and e.raw_fields.get("chicago_ditto"))
        if not is_ditto and e.authors:
            last_authors = e.authors
            if e.raw_fields:
                last_first_token = e.raw_fields.get("first_token", "")
        elif is_ditto:
            e.authors = list(last_authors)
            if e.raw_fields and last_first_token:
                e.raw_fields["first_token"] = last_first_token


def _parse_pdf_authors_chicago(raw: str) -> list[str]:
    """Parse a Chicago author string into "First Last" name strings.

    The first author is "Last, First [Middle]" (surname-inverted), and any
    subsequent authors are "First [Middle ]Last". Editor markers like
    ``, (eds.)`` and trailing suffixes like ``Jr.``/``Sr.``/``II``/``III``
    are preserved as part of the last name. Returns names in conventional
    "Forename Surname" order.
    """
    raw = re.sub(r',?\s*\((?:eds?|comp|trans)\.?\)\s*$', '', raw, flags=re.IGNORECASE)
    raw = _TRAILING_ET_AL_RE.sub('', raw)
    raw = raw.strip().rstrip('.,;')

    # Split on " and " — only the LAST occurrence (between penultimate and
    # final author). For 2-author entries that's the only " and ".
    parts = re.split(r',?\s+and\s+', raw, maxsplit=1)

    main = parts[0]
    last_author = parts[1] if len(parts) > 1 else None

    # The MAIN segment is "Last1, First1[, First2 Last2, First3 Last3, ...]".
    # Split on commas, then re-group: token 0 is surname, token 1 is firstname,
    # tokens 2+ are subsequent authors ("First Last") in standard order.
    chunks = [c.strip() for c in re.split(r',\s+', main) if c.strip()]
    names: list[str] = []
    if chunks:
        names.append(f"{chunks[1]} {chunks[0]}" if len(chunks) >= 2 else chunks[0])
        names.extend(chunks[2:])
    if last_author:
        names.append(last_author.strip().rstrip(',.'))
    return names


def _parse_block_alpha(num: int, raw: str) -> BibEntry:
    """Parse a LaTeX alpha-style block.

    Recovers the alpha key from the sentinel prefix prepended by
    ``_split_latex_alpha`` (stashed in ``raw_fields["alpha_key"]`` for the
    cite-key assignment pass). Uses the trailing-year grammar
    ``Authors. Title. Venue, ..., YEAR.`` — falls back to the standard
    multi-grammar block parser if the trailing-year shape doesn't match.
    """
    alpha_key: Optional[str] = None
    km = _ALPHA_KEY_RE.match(raw)
    if km:
        alpha_key = km.group(1).strip()
        raw = raw[km.end():]

    text = _collapse_block(raw)
    doi = _extract_doi(text)
    eprint = _extract_arxiv(text)
    archiveprefix = "arXiv" if eprint else None
    url = _extract_url(text, doi, eprint)

    year: Optional[int] = None
    authors: list[str] = []
    title: Optional[str] = None

    # Try quoted-title grammar first (BibLaTeX/Chicago: Authors. "Title". In: Venue.)
    qm = _LATEX_ALPHA_QUOTED_RE.match(text)
    m = None if qm else _LATEX_ALPHA_BLOCK_RE.match(text)
    if qm or m:
        matched = qm or m
        author_str = matched.group("authors").strip().rstrip(',.')
        title_str = matched.group("title").strip().strip('"')
        rest = matched.group("rest").strip()
        author_str = _TRAILING_ET_AL_RE.sub('', author_str)
        authors = _parse_pdf_authors(author_str) if author_str else []
        title = _extract_title_clean(title_str)
        year_matches = list(_LATEX_ALPHA_YEAR_RE.finditer(rest))
        if year_matches:
            year = int(year_matches[-1].group(1))
        else:
            # Broad fallback: any 4-digit year in 1800-2099 (e.g. "May 2024.")
            broad = re.search(r'\b(1[89]\d{2}|20\d{2})\b', rest)
            if broad:
                year = int(broad.group(1))
    else:
        nv = _LATEX_ALPHA_NOVENUE_RE.match(text)
        if nv:
            author_str = nv.group("authors").strip().rstrip(',.')
            author_str = _TRAILING_ET_AL_RE.sub('', author_str)
            authors = _parse_pdf_authors(author_str) if author_str else []
            title = _extract_title_clean(nv.group("title").strip())
            year = int(nv.group("year"))
        else:
            ym = _YEAR_RE.search(text)
            if ym:
                year = int(ym.group(1))
            log.debug(
                "LaTeX alpha block [%s] matched no grammar; partial parse only.",
                alpha_key or num,
            )

    raw_fields: dict = {"raw_text": text, "first_token": _first_token(text)}
    if alpha_key:
        raw_fields["alpha_key"] = alpha_key

    return BibEntry(
        key=f"ref{num}",
        entry_type="article",
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        url=url,
        eprint=eprint,
        archiveprefix=archiveprefix,
        raw_fields=raw_fields,
    )


def _parse_block_ay_bracketed(num: int, raw: str) -> BibEntry:
    """Parse one author-year-bracketed block "[Author et al., YEAR] content..."

    Strips the bracket-key sentinel, then delegates to ``_parse_block`` with
    ``grammar='ay_inline'`` since the block body uses author-year citation style.
    """
    ay_key: Optional[str] = None
    km = _AY_BRACKET_KEY_RE.match(raw)
    if km:
        ay_key = km.group(1).strip()
        raw = raw[km.end():]

    entry = _parse_block(num, raw, grammar='ay_inline')
    if ay_key:
        entry.raw_fields["ay_bracket_key"] = ay_key
    return entry


def _parse_block_acm_name_year(num: int, raw: str) -> BibEntry:
    """Parse one ACM name-year block (First Last, ..., and First Last. YEAR. Title.)."""
    return _parse_block(num, raw, grammar='acm')


def _extract_title_clean(title: str) -> Optional[str]:
    """Strip identifier noise from a pre-extracted title string."""
    title = _DOI_RE.sub('', title)
    title = _ARXIV_RE.sub('', title)
    title = _URL_RE.sub('', title)
    title = re.sub(r'\s+', ' ', title).strip().strip('"').rstrip('.,;:')
    return title if title else None


def _parse_pdf_authors_neurips(raw: str) -> list[str]:
    """Parse a NeurIPS/ICLR author string into a list of names.

    Input:  ``"M. Al-Roomi, S. Al-Ebrahim, and I. Ahmad"``
    Output: ``["M. Al-Roomi", "S. Al-Ebrahim", "I. Ahmad"]``

    The authors are already in "Initial. Surname" order, which is the
    canonical form used by CrossRef/OpenAlex — no reordering needed.
    """
    raw = _TRAILING_ET_AL_RE.sub('', raw.strip()).rstrip('., ')
    # Normalize `, and ` and ` and ` separators to plain `, `
    raw = re.sub(r',?\s+and\s+', ', ', raw, flags=re.IGNORECASE)
    parts = [p.strip().rstrip('.,') for p in raw.split(',')]
    return [p for p in parts if p]


def _parse_block_name_year_end(num: int, raw: str) -> BibEntry:
    """Parse one NAME_YEAR_END block.

    Format: ``First Last, First Last, ..., and First Last. Title. Venue, YEAR.``

    Strategy:
    1. Strip identifier noise (DOI, arXiv, URL) from the tail so the year
       regex can anchor at end-of-string.
    2. Pull the year off the end with ``_NUMBERED_BARE_YEAR_RE`` (``, YEAR.``).
    3. Split the remaining ``Authors. Title. Venue.`` on the first ``. ``
       whose preceding character is lowercase — period-after-initial like
       ``M. Dai`` is preserved because the lookbehind rejects it.
    4. The author list uses the same comma-and-``and`` separator parser as
       the NeurIPS block parser (the surnames are already in canonical
       ``First Last`` order, so no reordering is needed).
    """
    text = _collapse_block(raw)
    doi = _extract_doi(text)
    eprint = _extract_arxiv(text)
    archiveprefix = "arXiv" if eprint else None
    url = _extract_url(text, doi, eprint)

    # Strip DOI/URL noise first; the year is typically at end of string AFTER
    # any "arXiv preprint arXiv:XXXX.YYYY, YEAR." suffix, so we keep arXiv text
    # intact during year extraction and strip it afterwards.
    cleaned = text
    doi_m = _DOI_RE.search(cleaned)
    if doi_m:
        cleaned = cleaned[:doi_m.start()].rstrip().rstrip('.,;')
    url_m = _URL_RE.search(cleaned)
    if url_m:
        cleaned = cleaned[:url_m.start()].rstrip().rstrip('.,;')

    year: Optional[int] = None
    ym = _NUMBERED_BARE_YEAR_RE.search(cleaned)
    if ym:
        year = int(ym.group(1))
        cleaned = cleaned[:ym.start()].rstrip(' .,;')
    else:
        realistic = list(re.finditer(r'\b(1[89]\d{2}|20\d{2})\b', cleaned))
        if realistic:
            year = int(realistic[-1].group(1))

    arxiv_m = re.search(r'\barXiv\b', cleaned, re.IGNORECASE)
    if arxiv_m:
        cleaned = cleaned[:arxiv_m.start()].rstrip().rstrip('.,;')

    # Find author/title boundary: a ". " that is NOT preceded by a capital
    # letter (which would be an initial like "M. Dai" in the author list).
    boundary = re.search(r'(?<![A-Z])\.\s+(?=[A-Z])', cleaned)
    if boundary:
        author_str = cleaned[:boundary.start()]
        rest = cleaned[boundary.end():]
    else:
        author_str = ""
        rest = cleaned

    authors = _parse_pdf_authors_neurips(author_str) if author_str else []

    # Title goes from start of `rest` to the next ". " (same lookbehind rule).
    title_m = re.search(r'(?<![A-Z])\.\s+', rest)
    if title_m:
        title_raw = rest[:title_m.start()].strip()
    else:
        title_raw = rest.strip()
    title = _extract_title_clean(title_raw) if title_raw else None

    return BibEntry(
        key=f"ref{num}",
        entry_type="article",
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        url=url,
        eprint=eprint,
        archiveprefix=archiveprefix,
        raw_fields={"raw_text": text, "first_token": _first_token(text)},
    )


def _parse_block_neurips(num: int, raw: str) -> BibEntry:
    """Parse one NeurIPS/ICLR-style block.

    Format: ``I. Surname, I. Surname, ..., and I. Surname. Title. Venue, Year.``

    Tries the dedicated NeurIPS author regex first; falls back to the generic
    multi-grammar ``_parse_block`` when the author list can't be identified.
    """
    text = _collapse_block(raw)
    doi = _extract_doi(text)
    eprint = _extract_arxiv(text)
    archiveprefix = "arXiv" if eprint else None
    url = _extract_url(text, doi, eprint)

    year: Optional[int] = None
    authors: list[str] = []
    title: Optional[str] = None

    am = _NEURIPS_AUTHORS_RE.match(text)
    if am:
        author_str = am.group("authors")
        rest = text[am.end():].strip()
        # Strip trailing identifier noise (DOI, arXiv, URL) so that year
        # detection and title extraction see the clean venue/pages string.
        doi_m = _DOI_RE.search(rest)
        if doi_m:
            rest = rest[:doi_m.start()].rstrip().rstrip('.,;')
        # Strip any "arXiv …" suffix: catches both "arXiv:XXXX.YYYY" and
        # non-standard forms like "arXiv math.ST:2002.11457".
        arxiv_pos = re.search(r'\barXiv\b', rest, re.IGNORECASE)
        if arxiv_pos:
            rest = rest[:arxiv_pos.start()].rstrip().rstrip('.,;')
        url_m = _URL_RE.search(rest)
        if url_m:
            rest = rest[:url_m.start()].rstrip().rstrip('.,;')

        # Year is bare at the end: ", 2022." or ". 2019." etc.
        ym = _NUMBERED_BARE_YEAR_RE.search(rest)
        if ym:
            year = int(ym.group(1))
            pre_year = rest[:ym.start()].rstrip(' .,;')
            title = _extract_title(pre_year) if pre_year else None
        else:
            # Fallback: find the last realistic publication year in the text
            # (1800–2099). Taking the LAST match avoids confusing page numbers
            # like "page 1243" with publication years.
            realistic_years = list(re.finditer(r'\b(1[89]\d{2}|20\d{2})\b', rest))
            if realistic_years:
                year = int(realistic_years[-1].group(1))
            title = _extract_title(rest) if rest else None

        authors = _parse_pdf_authors_neurips(author_str)
    else:
        # Fall back to generic multi-grammar parser
        return _parse_block(num, raw)

    return BibEntry(
        key=f"ref{num}",
        entry_type="article",
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        url=url,
        eprint=eprint,
        archiveprefix=archiveprefix,
        raw_fields={"raw_text": text, "first_token": _first_token(text)},
    )


def _parse_pdf_authors_icml(raw: str) -> list[str]:
    """Parse an ICML-style 'Surname, F., Surname, F., ..., and Surname, F.' string.

    Uses ``_ICML_AUTHOR_CHUNK_RE`` to find all 'Surname, F.' units by regex,
    which is more reliable than splitting on commas (where commas appear both
    within and between author units).

    Input:  ``"Acun, B., Lee, B., Kazhamiaka, F., ..., and Wu, C.-J."``
    Output: ``["Acun, B.", "Lee, B.", "Kazhamiaka, F.", ..., "Wu, C.-J."]``
    """
    raw = _TRAILING_ET_AL_RE.sub('', raw.strip()).rstrip(', ')
    chunks = [m.group().strip().rstrip(' ') for m in _ICML_AUTHOR_CHUNK_RE.finditer(raw)]
    if chunks:
        return chunks
    # Fallback: normalize "and" separators and split on comma
    raw = re.sub(r',?\s+and\s+', ', ', raw, flags=re.IGNORECASE)
    parts = [p.strip().rstrip('.,') for p in raw.split(',')]
    return [p for p in parts if p]


def _parse_block_icml(num: int, raw: str) -> BibEntry:
    """Parse one ICML-style block.

    Format: ``Surname, F., Surname, F., ..., and Surname, F. Title. Venue, Year.``

    The author/title boundary is the first ``". "`` that is NOT followed by
    another author or a bare initial (see ``_ICML_AUTHOR_END_RE``). Falls back
    to the generic multi-grammar parser when the author list can't be matched.
    """
    text = _collapse_block(raw)
    doi = _extract_doi(text)
    eprint = _extract_arxiv(text)
    archiveprefix = "arXiv" if eprint else None
    url = _extract_url(text, doi, eprint)

    year: Optional[int] = None
    authors: list[str] = []
    title: Optional[str] = None

    # Find author/title boundary: first ". " not inside the author list.
    bm = _ICML_AUTHOR_END_RE.search(text)
    if bm:
        author_str = text[:bm.start() + 1].strip()   # up to and including "."
        rest = text[bm.end():].strip()

        # Strip trailing identifier noise so year/title extraction is clean.
        doi_m = _DOI_RE.search(rest)
        if doi_m:
            rest = rest[:doi_m.start()].rstrip().rstrip('.,;')
        arxiv_pos = re.search(r'\barXiv\b', rest, re.IGNORECASE)
        if arxiv_pos:
            rest = rest[:arxiv_pos.start()].rstrip().rstrip('.,;')
        # Strip URL (and the optional bare word "URL" that precedes it in
        # some ICML references: "...2021. URL https://arxiv.org/...")
        url_m = re.search(r'(?:\bURL\s+)?https?://', rest, re.IGNORECASE)
        if url_m:
            rest = rest[:url_m.start()].rstrip().rstrip('.,;')

        # Year is bare at the end, with optional letter disambiguation suffix
        # ("2021a", "2021b"). Use the ICML-specific pattern that handles the
        # suffix; fall back to scanning for any realistic year in the text.
        ym = _ICML_BARE_YEAR_RE.search(rest)
        if ym:
            year = int(ym.group(1))
            pre_year = rest[:ym.start()].rstrip(' .,;')
            title = _extract_title(pre_year) if pre_year else None
        else:
            # AAAI-style: year precedes the title ("YEAR. Title. Venue.").
            # Strip the leading year so it doesn't contaminate the title string.
            leading_ym = _ICML_LEADING_YEAR_RE.match(rest)
            if leading_ym:
                year = int(leading_ym.group(1))
                rest = rest[leading_ym.end():]
            else:
                # \b(year)[a-z]? — the trailing [a-z]? matches the suffix, and the
                # word boundary before the year is enough; no trailing \b needed
                # because the suffix letter may be followed by a period (non-word).
                realistic_years = list(re.finditer(r'\b(1[89]\d{2}|20\d{2})[a-z]?', rest))
                if realistic_years:
                    year = int(realistic_years[-1].group(1))
            title = _extract_title(rest) if rest else None

        authors = _parse_pdf_authors_icml(author_str)
    else:
        return _parse_block(num, raw)

    return BibEntry(
        key=f"ref{num}",
        entry_type="article",
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        url=url,
        eprint=eprint,
        archiveprefix=archiveprefix,
        raw_fields={"raw_text": text, "first_token": _first_token(text)},
    )


_BLOCK_PARSERS: dict[CitationFormat, Callable[[int, str], BibEntry]] = {
    CitationFormat.ACM_BRACKETED:  _parse_block,
    CitationFormat.PLAIN_NUMBERED: _parse_block,
    CitationFormat.AUTHOR_YEAR:    _parse_block,
    CitationFormat.LATEX_ALPHA:    _parse_block_alpha,
    CitationFormat.AY_BRACKETED:   _parse_block_ay_bracketed,
    CitationFormat.ACM_NAME_YEAR:  _parse_block_acm_name_year,
    CitationFormat.CHICAGO:        _parse_block_chicago,
    CitationFormat.NEURIPS:        _parse_block_neurips,
    CitationFormat.ICML:           _parse_block_icml,
    CitationFormat.NAME_YEAR_END:  _parse_block_name_year_end,
}


def _collapse_block(raw: str) -> str:
    """Collapse a multi-line block into a single clean string.

    Drops noise lines (page numbers, running heads), heals split URLs,
    repairs mid-word hyphenation from PDF line wraps, and strips stray
    control characters that pymupdf occasionally emits for unmapped glyphs
    (e.g. ``Agust\\x01ı`` for ``Agustí``).
    """
    raw = _SPLIT_URL_RE.sub(r'\1://', raw)
    raw = _PDF_GLYPH_NOISE_RE.sub('', raw)
    raw = _PDF_INWORD_TILDE_RE.sub('', raw)
    raw = _PDF_COMBINING_NOISE_RE.sub('', raw)
    # Strip footnote text that bleeds into the last reference block from the
    # page footer.  Footnote markers are a bare digit immediately followed by
    # uppercase+lowercase (e.g. "2The views expressed…", "1Note that…") —
    # distinct from page numbers ("203"), volume numbers ("27(3)"), and
    # content like "3D printing" where the digit is followed by another
    # uppercase letter only.
    raw = re.sub(r'\n\d[A-Z][a-z].+', '', raw, flags=re.DOTALL)
    lines = raw.split('\n')
    lines = [l for l in lines if not _NOISE_LINE_RE.match(l)]
    text = ' '.join(lines)
    text = re.sub(r'\s+', ' ', text).strip()
    # Heal URL-internal whitespace BEFORE the hyphenation pass — otherwise
    # `_WRAP_HYPHEN_RE` would mangle hyphenated URL paths
    # (``policies-and- guidelines`` → ``policies-andguidelines``).
    prev: Optional[str] = None
    while text != prev:
        prev = text
        text = _URL_REJOIN_RE.sub(r'\1\2', text)
    # Second pass: rejoin DOI/URL suffixes that are purely alphanumeric
    # (e.g. "acs.estlett. 5b00213" after a line break inside a DOI path).
    # Only fires when the URL ends with "." or "-" (clear mid-URL split signal)
    # and the continuation starts with a lowercase letter or digit (so normal
    # sentence continuations starting with uppercase are left untouched).
    text = re.sub(r'(https?://\S+[-.])\s+([a-z0-9]\w*)', r'\1\2', text)
    # Third pass: same healing for bare "doi: 10.xxx/yyy." split across lines
    # (e.g., "doi: 10.14257/ijgdc.2013.\n6.5.09.").
    text = re.sub(
        r'(doi:\s*10\.\d{4,9}/\S+[.-])\s+([a-z0-9]\S*)',
        r'\1\2', text,
    )
    return _WRAP_HYPHEN_RE.sub(r'\1\2', text)


# ---------------------------------------------------------------------------
# Field extractors
# ---------------------------------------------------------------------------

def _extract_doi(text: str) -> Optional[str]:
    m = _DOI_RE.search(text)
    if not m:
        return None
    doi = m.group(1).rstrip('.,;')
    return clean_doi(doi)


def _extract_arxiv(text: str) -> Optional[str]:
    m = _ARXIV_RE.search(text)
    if not m:
        return None
    return clean_arxiv_id(m.group(1))


def _extract_url(text: str, doi: Optional[str], eprint: Optional[str]) -> Optional[str]:
    """Return the first URL in text that is not a DOI or arXiv link."""
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip('.,;)')
        lower = url.lower()
        if 'doi.org' in lower:
            continue
        if 'arxiv.org' in lower:
            continue
        return url
    return None


def _extract_title(remainder: str) -> Optional[str]:
    """Extract the title from the text after the year separator.

    Strategy: compute candidate title-end positions from `_TITLE_END_RE`
    (known venue prefixes / URLs) and `_VENUE_TRAILER_RE` (numeric
    volume/issue/pages or article-number signatures), then cut at the
    earlier of the two. Picking the earliest valid boundary matters when
    the title-end-token marker (e.g. `https://`) appears *after* the venue
    block — without this, the entire "Journal, Vol(Issue), pages" string
    leaks into the title.
    """
    # URL position is a *very strong* title-end signal — and we want to use
    # it *before* stripping URLs, because the URL token regex stops at any
    # whitespace, and PDF-extracted URLs frequently contain whitespace
    # inside long query strings (analytics tracker fragments, `#:~:text=`
    # anchors, etc.). If we strip URLs first, those trailing query
    # fragments survive as inline tokens and leak into the title.
    url_match = _URL_RE.search(remainder)
    if url_match:
        head = remainder[:url_match.start()].rstrip()
        period = head.rfind('.')
        remainder = head[:period].rstrip() if period > 0 else head

    remainder = _DOI_RE.sub('', remainder)
    remainder = _ARXIV_RE.sub('', remainder)
    remainder = _URL_RE.sub('', remainder)
    remainder = re.sub(r'\s+', ' ', remainder).strip().rstrip('.,;')

    candidates: list[int] = []
    m = _TITLE_END_RE.search(remainder)
    if m:
        candidates.append(m.start())
    tm = _VENUE_TRAILER_RE.search(remainder)
    if tm:
        cut = _cut_before_trailer(remainder, tm.start())
        if cut is not None and cut > 0:
            candidates.append(cut)

    if candidates:
        title = remainder[:min(candidates)].strip().rstrip('.,;')
    else:
        title = remainder

    return title if title else None


# Short capitalized words that almost always indicate a journal-abbreviation
# chain ("Renew. Sust. Energy Rev."). A title-ending period would otherwise
# be confused with these; `_cut_before_trailer` walks back past them.
_VENUE_ABBREV_TOKEN_RE = re.compile(r'^[A-Z][a-z]{0,5}$')


def _cut_before_trailer(remainder: str, trailer_start: int) -> Optional[int]:
    """Return the position of the period that most likely ends the title.

    Walks back through the periods that precede the numeric venue trailer.
    Skips cuts whose preceding token is a short capitalized word, which is
    almost always a journal abbreviation ("Rev", "Sust", "Sci") rather than
    the actual end of the title.
    """
    periods = [i for i, c in enumerate(remainder[:trailer_start]) if c == '.']
    if not periods:
        return None
    for cut in reversed(periods):
        candidate = remainder[:cut].rstrip()
        if not candidate:
            continue
        last_token = candidate.rsplit(maxsplit=1)[-1].rstrip('.,;')
        if _VENUE_ABBREV_TOKEN_RE.match(last_token):
            continue
        return cut
    # All periods looked like venue abbreviations. Return the EARLIEST (first)
    # period, not the latest — when J, System, etc. are all skipped as
    # abbreviations, the earliest period in the chain is the better title boundary
    # (e.g., "Task System. J. ACM, 39" → cut at "System." not "J.").
    return periods[0]


# ---------------------------------------------------------------------------
# Author parsing
# ---------------------------------------------------------------------------

def _parse_pdf_authors(raw: str) -> list[str]:
    """Parse an ACM-style author string (assumes "First Last" order).

    ACM references use "First Last" order, comma-separated within a group
    and " and " between the last two authors.
    """
    parts = _AUTHOR_AND_RE.split(raw.strip())
    names: list[str] = []
    for part in parts:
        # Split on ", " only when the next token looks like a personal name
        # (capital letter + lowercase or period) -- avoids splitting "Gurobi, LLC".
        sub_parts = re.split(r',\s+(?=[A-Z][a-z.])', part)
        for sub in sub_parts:
            name = sub.strip().rstrip(',')
            if name:
                names.append(name)
    return names


def _parse_pdf_authors_vancouver(raw: str) -> list[str]:
    """Parse a Vancouver author string into a list of "F. Surname" names.

    Input shape: ``"Kohansal R, Martinez-Camblor P, Buist AS, ..., et al."``.
    Output shape: ``["R. Kohansal", "P. Martinez-Camblor", "A. S. Buist", ...]``
    — i.e., initials with periods, expanded ahead of the surname.
    Trailing ``et al.`` is stripped.
    """
    raw = _TRAILING_ET_AL_RE.sub('', raw.strip()).rstrip(',.')
    if not raw:
        return []
    out: list[str] = []
    for chunk in re.split(r',\s+', raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        # Last whitespace-separated token is the initials block (1-3 caps);
        # everything before is the surname (possibly multi-word).
        parts = chunk.rsplit(maxsplit=1)
        if len(parts) != 2:
            out.append(chunk)
            continue
        surname, initials = parts
        # Insert "." after each initial letter: "AS" -> "A. S.".
        spaced = '. '.join(initials) + '.'
        out.append(f"{spaced} {surname}")
    return out


def _parse_pdf_authors_ay(raw: str) -> list[str]:
    """Parse an author-year style author string into a list of names.

    Handles three sub-styles:
      - "Last, F. M., Last2, F."   ->  ["F. M. Last", "F. Last2"]
      - "Last F., Last2 F. M."     ->  ["F. Last", "F. M. Last2"]   (inline initials)
      - "Wang Wei, Lan Yingjie"    ->  ["Wang Wei", "Lan Yingjie"]   (no initials)

    Strips trailing "et al." and normalizes bare initials by adding a period.
    """
    raw = _TRAILING_ET_AL_RE.sub('', raw.strip())
    if not raw:
        return []

    big_parts = _AY_BIG_SEP_RE.split(raw)
    names: list[str] = []
    for part in big_parts:
        part = part.strip().rstrip(',.')
        if not part:
            continue
        tokens = [t.strip() for t in part.split(',') if t.strip()]
        if not tokens:
            continue

        if any(_INITIALS_TOKEN_RE.match(t) for t in tokens):
            # Comma-separated Last, F. M., Last2, F. ... grouping.
            names.extend(_group_lastname_initials(tokens))
        elif any(_NAME_WITH_INLINE_INITIALS_RE.match(t) for t in tokens):
            # Each chunk has inline initials like "Lechowicz A".
            for t in tokens:
                m = _NAME_WITH_INLINE_INITIALS_RE.match(t)
                if m:
                    surname = m.group(1).strip()
                    initials = m.group(2).strip()
                    names.append(f"{initials} {surname}")
                else:
                    names.append(t)
        else:
            # No initials anywhere -> treat each chunk as a full name.
            names.extend(tokens)

    return [_normalize_initials(n) for n in names]


def _group_lastname_initials(tokens: list[str]) -> list[str]:
    """Group [Last, F. M., Last2, F.] tokens into ["F. M. Last", "F. Last2"]."""
    out: list[str] = []
    i = 0
    while i < len(tokens):
        head = tokens[i]
        if i + 1 < len(tokens) and _INITIALS_TOKEN_RE.match(tokens[i + 1]):
            out.append(f"{tokens[i + 1].strip()} {head}".strip())
            i += 2
        else:
            m = _NAME_WITH_INLINE_INITIALS_RE.match(head)
            if m:
                surname = m.group(1).strip()
                initials = m.group(2).strip()
                out.append(f"{initials} {surname}")
            else:
                out.append(head)
            i += 1
    return out


def _normalize_initials(name: str) -> str:
    """Ensure bare capital-letter initials carry a trailing period."""
    return _BARE_INITIAL_RE.sub(r'\1.', name)


# ---------------------------------------------------------------------------
# Cite-key generation
# ---------------------------------------------------------------------------

def _first_token(text: str) -> str:
    """Return the first word-shaped token of a collapsed block, or ''."""
    m = _FIRST_TOKEN_RE.match(text)
    return m.group(1) if m else ""


def _assign_cite_keys(entries: list[BibEntry]) -> None:
    """Replace positional `ref{N}` keys with `{FirstToken}{Year}` keys.

    For author-year style PDFs the first token IS the first author's surname,
    so this yields readable keys like "Barberis2003". Collisions (same surname,
    same year) get a lowercase letter suffix on the 2nd+ occurrences:
    "Mohr2014", "Mohr2014a", "Mohr2014b". Entries missing a first token or
    authors fall back to `ref{N}` (positional).
    """
    counts: dict[str, int] = {}
    for i, e in enumerate(entries):
        alpha_key = e.raw_fields.get("alpha_key", "") if e.raw_fields else ""
        if alpha_key:
            # LaTeX alpha PDFs already have meaningful keys ("AGS20", "DBB+14")
            # from the original .bib file — use them verbatim.
            e.key = alpha_key
            continue
        token = e.raw_fields.get("first_token", "") if e.raw_fields else ""
        if not token:
            base = f"ref{i + 1}"
        elif e.year:
            base = f"{token}{e.year}"
        else:
            base = token
        n = counts.get(base, 0)
        counts[base] = n + 1
        e.key = base if n == 0 else f"{base}{chr(ord('a') + n - 1)}"
