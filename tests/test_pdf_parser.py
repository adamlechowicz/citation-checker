"""Tests for the PDF bibliography parser."""

from __future__ import annotations

import pytest
from pathlib import Path

from citation_checker.models import BibEntry
from citation_checker.pdf_parser import (
    CitationFormat,
    _assign_cite_keys,
    _collapse_block,
    _detect_format,
    _extract_doi,
    _extract_arxiv,
    _extract_url,
    _extract_title,
    _find_references_section,
    _first_token,
    _looks_corporate,
    _parse_author_year_block,
    _parse_block_alpha,
    _parse_block_icml,
    _parse_elsevier_block,
    _parse_nature_block,
    _parse_numbered_block,
    _parse_numbered_bare_year_block,
    _parse_pdf_authors,
    _parse_pdf_authors_ay,
    _parse_pdf_authors_chicago,
    _parse_pdf_authors_icml,
    _parse_pdf_authors_vancouver,
    _parse_block_chicago,
    _parse_chicago_block,
    _parse_vancouver_block,
    _parse_vancouver_corporate_block,
    _resolve_chicago_ditto,
    _split_chicago,
    _split_icml,
    _find_chicago_authors_end,
    _split_author_year,
    _split_into_blocks,
    _split_latex_alpha,
    _split_nature_authors,
    _split_plain_numbered,
    parse_pdf_file,
    _ACM_YEAR_SPLIT_RE,
)

# Path to the real PDF used for integration tests
PDF_PATH = Path(__file__).parent.parent / "2511.18554v2.pdf"


# ---------------------------------------------------------------------------
# Unit tests — no PDF file required
# ---------------------------------------------------------------------------

class TestExtractDoi:
    def test_doi_prefix(self):
        assert _extract_doi("doi:10.1145/3469379.3469382") == "10.1145/3469379.3469382"

    def test_doi_url(self):
        assert _extract_doi("https://doi.org/10.1016/j.apenergy.2017.07.034") == "10.1016/j.apenergy.2017.07.034"

    def test_dx_doi_url(self):
        assert _extract_doi("http://dx.doi.org/10.1109/tsg.2014.2320261") == "10.1109/tsg.2014.2320261"

    def test_trailing_punctuation_stripped(self):
        doi = _extract_doi("doi:10.1145/146585.146588.")
        assert doi is not None
        assert not doi.endswith(".")

    def test_no_doi(self):
        assert _extract_doi("No identifier here, just text.") is None


class TestExtractArxiv:
    def test_new_format(self):
        assert _extract_arxiv("arXiv:2206.13606 [cs.DS]") == "2206.13606"

    def test_old_format(self):
        assert _extract_arxiv("arXiv:math/0501328 [math.CA]") == "math/0501328"

    def test_version_suffix_stripped(self):
        assert _extract_arxiv("arXiv:1912.01703v1") == "1912.01703"

    def test_case_insensitive(self):
        assert _extract_arxiv("arxiv:2206.13606") == "2206.13606"

    def test_pdf_duplicated_arxiv_prefix(self):
        assert _extract_arxiv("arXiv preprint arXiv: Arxiv- 2301.07608") == "2301.07608"

    def test_no_arxiv(self):
        assert _extract_arxiv("No identifier here.") is None


class TestExtractUrl:
    def test_plain_url(self):
        url = _extract_url("See https://example.com/paper for details.", None, None)
        assert url == "https://example.com/paper"

    def test_doi_url_skipped(self):
        url = _extract_url("https://doi.org/10.1145/123", "10.1145/123", None)
        assert url is None

    def test_arxiv_url_skipped(self):
        url = _extract_url("https://arxiv.org/abs/2206.13606", None, "2206.13606")
        assert url is None

    def test_trailing_punctuation_stripped(self):
        url = _extract_url("Visit https://example.com.", None, None)
        assert url is not None
        assert not url.endswith(".")


class TestCollapseBlock:
    def test_joins_lines(self):
        result = _collapse_block("line one\nline two\nline three")
        assert result == "line one line two line three"

    def test_removes_page_numbers(self):
        result = _collapse_block("before\n23\nafter")
        assert "23" not in result.split()  # "23" should not appear as standalone token

    def test_heals_split_url(self):
        raw = "See https:\n//arxiv.org/abs/1706.03762 for details."
        result = _collapse_block(raw)
        assert "https://arxiv.org/abs/1706.03762" in result

    def test_removes_et_al_running_head(self):
        result = _collapse_block("reference text\nSmith et al.\nmore text")
        assert "Smith et al." not in result


class TestAcmYearSplit:
    def test_basic_split(self):
        text = "Abdul Afram and Farrokh Janabi-Sharifi. 2014. Theory and applications of HVAC. Building."
        m = _ACM_YEAR_SPLIT_RE.match(text)
        assert m is not None
        assert m.group(2) == "2014"
        assert "Afram" in m.group(1)
        assert "Theory" in m.group(3)

    def test_year_in_range(self):
        text = "Author Name. 1998. A Classic Paper. Some Journal."
        m = _ACM_YEAR_SPLIT_RE.match(text)
        assert m is not None
        assert m.group(2) == "1998"

    def test_no_match_without_period_year_period(self):
        text = "No year in this text."
        m = _ACM_YEAR_SPLIT_RE.match(text)
        assert m is None


class TestParsePdfAuthors:
    def test_two_authors_and(self):
        result = _parse_pdf_authors("Abdul Afram and Farrokh Janabi-Sharifi")
        assert len(result) == 2
        assert "Abdul Afram" in result
        assert "Farrokh Janabi-Sharifi" in result

    def test_comma_separated_with_and(self):
        result = _parse_pdf_authors("A. Mamun, I. Narayanan, D. Wang, and H.K. Fathy")
        assert len(result) == 4

    def test_single_author(self):
        result = _parse_pdf_authors("Isi Mitrani")
        assert result == ["Isi Mitrani"]

    def test_corporate_author_not_split(self):
        # "LLC" after comma should not be treated as a new author name
        result = _parse_pdf_authors("Gurobi Optimization, LLC")
        # Should be one name, not two
        assert len(result) == 1


class TestFindReferencesSection:
    def test_finds_references_heading(self):
        text = "Introduction\n...\nReferences\n[1] Some paper.\n"
        result = _find_references_section(text)
        assert "[1]" in result

    def test_finds_bibliography_heading(self):
        text = "Body text.\nBibliography\n[1] A paper."
        result = _find_references_section(text)
        assert "[1]" in result

    def test_all_caps_heading(self):
        text = "Conclusion.\nREFERENCES\n[1] A paper."
        result = _find_references_section(text)
        assert "[1]" in result

    def test_raises_if_not_found(self):
        with pytest.raises(ValueError, match="References"):
            _find_references_section("No bibliography here at all.")


class TestSplitIntoBlocks:
    def test_splits_three_blocks(self):
        text = "[1] First reference.\n[2] Second reference.\n[3] Third."
        blocks = _split_into_blocks(text)
        assert len(blocks) == 3
        assert blocks[0][0] == 1
        assert blocks[1][0] == 2
        assert blocks[2][0] == 3

    def test_block_content(self):
        text = "[1] First paper. 2020. Title.\n[2] Second."
        blocks = _split_into_blocks(text)
        assert "First paper" in blocks[0][1]

    def test_empty_text(self):
        assert _split_into_blocks("No numbered references here.") == []

    def test_truncates_before_appendix(self):
        text = (
            "[1] First reference. 2020. Title.\n"
            "[2] Second reference. 2021. Title.\n"
            "Appendix\n"
            "[3] This is appendix text, not a bibliography entry.\n"
        )
        blocks = _split_into_blocks(text)
        assert len(blocks) == 2
        assert "appendix" not in blocks[-1][1].lower()

    def test_truncates_neurips_letter_appendix_heading(self):
        text = (
            "[1] First reference. 2020. Title.\n"
            "[2] Second reference. 2021. Title.\n"
            "17\n\n"
            "A\n"
            "Model Architecture\n"
            "Appendix text with bracketed citations [61].\n"
        )
        blocks = _split_into_blocks(text)
        assert len(blocks) == 2
        assert "Model Architecture" not in blocks[-1][1]


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

class TestDetectFormat:
    def test_detects_bracketed(self):
        text = (
            "[1] Author A. 2020. Title one. Journal.\n"
            "[2] Author B. 2021. Title two. Journal.\n"
            "[3] Author C. 2022. Title three. Journal.\n"
            "[4] Author D. 2023. Title four. Journal.\n"
        )
        assert _detect_format(text) == CitationFormat.ACM_BRACKETED

    def test_detects_plain_numbered_dot(self):
        text = (
            "1. Author A. 2020. Title one. Journal.\n"
            "2. Author B. 2021. Title two. Journal.\n"
            "3. Author C. 2022. Title three. Journal.\n"
            "4. Author D. 2023. Title four. Journal.\n"
        )
        assert _detect_format(text) == CitationFormat.PLAIN_NUMBERED

    def test_detects_plain_numbered_paren(self):
        text = (
            "1) Author A. 2020. Title one. Journal.\n"
            "2) Author B. 2021. Title two. Journal.\n"
            "3) Author C. 2022. Title three. Journal.\n"
            "4) Author D. 2023. Title four. Journal.\n"
        )
        assert _detect_format(text) == CitationFormat.PLAIN_NUMBERED

    def test_detects_author_year(self):
        text = (
            "Adams J. (2020). Title one. Journal, 1, 1-10.\n"
            "Brown K. (2021). Title two. Journal, 2, 11-20.\n"
            "Chen L. (2022). Title three. Journal, 3, 21-30.\n"
            "Davis M. (2023). Title four. Journal, 4, 31-40.\n"
        )
        assert _detect_format(text) == CitationFormat.AUTHOR_YEAR

    def test_tie_breaks_to_bracketed(self):
        # 3 bracketed AND 3 author-year hits -> bracketed wins by priority
        text = (
            "[1] Adams J. 2020. T. J.\n"
            "[2] Brown K. 2021. T. J.\n"
            "[3] Chen L. 2022. T. J.\n"
            "Davis M. (2023). T. J.\n"
            "Evans N. (2024). T. J.\n"
            "Foster O. (2025). T. J.\n"
        )
        assert _detect_format(text) == CitationFormat.ACM_BRACKETED

    def test_raises_when_below_threshold(self):
        text = "[1] One reference only. 2020. Title. Journal.\n"
        with pytest.raises(ValueError) as exc_info:
            _detect_format(text)
        msg = str(exc_info.value)
        assert "bracketed" in msg
        assert "plain" in msg
        assert "author-year" in msg
        assert "LaTeX alpha" in msg

    def test_detects_latex_alpha(self):
        text = (
            "[AGS20]\nAdams J. Title one. Journal, 2020.\n"
            "[BKL18]\nBrown K. Title two. Journal, 2018.\n"
            "[CR20]\nChen L and Davis M. Title three. Journal, 2020.\n"
            "[DeG74]\nDavis M. Title four. Journal, 1974.\n"
        )
        assert _detect_format(text) == CitationFormat.LATEX_ALPHA

    def test_detects_latex_alpha_with_plus_keys(self):
        # The et-al "+" variant: [DBB+14], [NHH+14], [SCP+20]
        text = (
            "[DBB+14] Author A. Title one. Journal, 2014.\n"
            "[NHH+14] Author B. Title two. Journal, 2014.\n"
            "[SCP+20] Author C. Title three. Journal, 2020.\n"
        )
        assert _detect_format(text) == CitationFormat.LATEX_ALPHA


# ---------------------------------------------------------------------------
# Plain numbered splitter
# ---------------------------------------------------------------------------

class TestSplitPlainNumbered:
    def test_splits_dot_separator(self):
        text = (
            "1. Author A. 2020. First. Journal.\n"
            "2. Author B. 2021. Second. Journal.\n"
            "3. Author C. 2022. Third. Journal.\n"
        )
        blocks = _split_plain_numbered(text)
        assert [b[0] for b in blocks] == [1, 2, 3]
        assert "First" in blocks[0][1]

    def test_splits_paren_separator(self):
        text = "1) Author A. 2020. Foo. J.\n2) Author B. 2021. Bar. J.\n"
        blocks = _split_plain_numbered(text)
        assert len(blocks) == 2

    def test_ignores_numbered_list_in_title(self):
        # A "1." inside body text without a leading capital after it
        # should not be detected as a reference start.
        text = "1. Real Reference. 2020. Title. J.\n2. another item\n"
        blocks = _split_plain_numbered(text)
        # The second line lacks a capital after "2.", so only one block.
        assert len(blocks) == 1


# ---------------------------------------------------------------------------
# Author-year splitter
# ---------------------------------------------------------------------------

class TestSplitAuthorYear:
    def test_splits_three_entries(self):
        text = (
            "Adams J. (2020). Title one. Journal, 1, 1-10.\n"
            "Brown K. (2021). Title two. Journal, 2, 11-20.\n"
            "Chen L. (2022). Title three. Journal, 3, 21-30.\n"
        )
        blocks = _split_author_year(text)
        assert len(blocks) == 3
        assert [b[0] for b in blocks] == [1, 2, 3]
        assert "Adams" in blocks[0][1]
        assert "Brown" in blocks[1][1]

    def test_multiline_wrapped_entry(self):
        # First entry wraps onto a continuation line BEFORE the year appears.
        text = (
            "Smith J, Doe A, Garcia B,\n"
            "Peterson R\n"
            "(2024). Wrapped title. Journal, 1, 1-2.\n"
            "Williams Q. (2025). Another. Journal, 2, 3-4.\n"
            "Xu Z. (2023). Third. Journal, 3, 5-6.\n"
        )
        blocks = _split_author_year(text)
        # 3 entries (Smith, Williams, Xu); wrap lines stay inside Smith's block.
        assert len(blocks) == 3
        assert "Peterson" in blocks[0][1]
        assert "Williams" in blocks[1][1]

    def test_no_space_before_paren(self):
        text = (
            "Boehmer E, Jones C M, et al.(2021) Tracking. JF, 76, 22-30.\n"
            "Cooper D. (2022). Other. J, 1, 1.\n"
            "Daniels F. (2023). Third. J, 2, 2.\n"
        )
        blocks = _split_author_year(text)
        assert len(blocks) == 3

    def test_chinese_style_authors(self):
        text = (
            "Wang Wei, Lan Yingjie. (2021). Online one-way trading. J, 41, 24-87.\n"
            "Xu Y, Zhang W, Zheng F. (2011). Optimal algorithms. TCS, 412, 192.\n"
            "Yang L, Hajiesmaili M H. (2020). Online linear. POMACS, 4, 1-29.\n"
        )
        blocks = _split_author_year(text)
        assert len(blocks) == 3

    def test_ignores_noise_lines(self):
        text = (
            "Adams J. (2020). One. J, 1, 1.\n"
            "20\n"  # bare page number
            "Brown K. (2021). Two. J, 2, 2.\n"
            "Chen L. (2022). Three. J, 3, 3.\n"
        )
        blocks = _split_author_year(text)
        assert len(blocks) == 3


# ---------------------------------------------------------------------------
# Author-year block parser
# ---------------------------------------------------------------------------

class TestAuthorYearBlock:
    def test_basic_parse(self):
        raw = (
            "Lechowicz A, Christianson N, et al. (2024). "
            "Online Conversion with Switching Costs: Robust and Learning-"
            "augmented Algorithms. ACM SIGMETRICS Performance Evaluation Review, "
            "52(1), 45-46."
        )
        e = _parse_author_year_block(1, raw)
        assert e.year == 2024
        assert any("Lechowicz" in a for a in e.authors)
        assert any(a.startswith("A.") for a in e.authors)
        assert e.title is not None
        assert "Online Conversion" in e.title

    def test_comma_heavy_initials(self):
        raw = (
            "Chin, F. Y., Fu, B., Guo, J., Hu, J., Jiang, M., Zhou, D. (2015). "
            "Competitive algorithms for unbounded one-way trading. "
            "Theoretical Computer Science, 607, 35-48."
        )
        e = _parse_author_year_block(2, raw)
        assert e.year == 2015
        assert len(e.authors) >= 5
        assert any("Chin" in a for a in e.authors)
        assert "F. Y. Chin" in e.authors
        assert e.title is not None
        assert e.title.startswith("Competitive")

    def test_ampersand_separator(self):
        raw = (
            "Lee, R., Sun, B., Hajiesmaili, M., & Lui, J. C. (2024). "
            "Online Search with Predictions. "
            "In Proceedings of the 15th ACM Conference, 386-407."
        )
        e = _parse_author_year_block(3, raw)
        assert e.year == 2024
        assert len(e.authors) == 4
        assert e.authors[0] == "R. Lee"

    def test_no_space_around_year(self):
        raw = (
            "Boehmer E, Jones C M, Zhang X Y, et al.(2021) Tracking retail. "
            "The Journal of Finance, 76(5), 2249-2305."
        )
        e = _parse_author_year_block(4, raw)
        assert e.year == 2021
        assert any("Boehmer" in a for a in e.authors)

    def test_bare_year_fallback(self):
        # Author-year file but this entry uses ACM-style "Authors. YEAR. Title."
        raw = "Authors X, Y, Z. 2020. Some Title. Some Journal, 1, 1-2."
        e = _parse_author_year_block(5, raw)
        assert e.year == 2020
        assert e.title is not None
        assert "Some Title" in e.title

    def test_year_wins_over_page_range(self):
        raw = (
            "Smith J. (2024). Wonderful Paper. "
            "The Journal of Whatever, 76(5), 2249-2305."
        )
        e = _parse_author_year_block(6, raw)
        assert e.year == 2024


# ---------------------------------------------------------------------------
# Author-year author string parser
# ---------------------------------------------------------------------------

class TestParsePdfAuthorsAY:
    def test_last_comma_initials(self):
        result = _parse_pdf_authors_ay("Chin, F. Y., Fu, B., Guo, J.")
        assert result == ["F. Y. Chin", "B. Fu", "J. Guo"]

    def test_inline_initials_inversion(self):
        result = _parse_pdf_authors_ay("Lechowicz A, Christianson N")
        assert result == ["A. Lechowicz", "N. Christianson"]

    def test_ampersand_separator(self):
        result = _parse_pdf_authors_ay("Lee, R., Sun, B., & Lui, J. C.")
        assert result == ["R. Lee", "B. Sun", "J. C. Lui"]

    def test_strips_trailing_et_al(self):
        result = _parse_pdf_authors_ay("Boehmer E, Jones C M, et al.")
        assert result == ["E. Boehmer", "C. M. Jones"]

    def test_chinese_style_no_initials(self):
        result = _parse_pdf_authors_ay("Wang Wei, Lan Yingjie")
        assert result == ["Wang Wei", "Lan Yingjie"]

    def test_diacritics_preserved(self):
        result = _parse_pdf_authors_ay("Wøhlk, S., Müller, A.")
        assert "S. Wøhlk" in result
        assert "A. Müller" in result

    def test_surname_prefix(self):
        # "Da Gama Batista J" -> "J. Da Gama Batista" (prefix preserved)
        result = _parse_pdf_authors_ay("Da Gama Batista J, Massaro D")
        assert "J. Da Gama Batista" in result
        assert "D. Massaro" in result


# ---------------------------------------------------------------------------
# Wrap-hyphen healing in _collapse_block
# ---------------------------------------------------------------------------

class TestWrapHyphenHealing:
    def test_heals_mid_word_hyphenation(self):
        assert "distributions" in _collapse_block("distribu-\ntions with monotone")

    def test_heals_multiple_breaks(self):
        text = _collapse_block("Algo-\nrithm and oppor-\ntunity cost")
        assert "Algorithm" in text
        assert "opportunity" in text

    def test_does_not_remove_hyphen_between_normal_words(self):
        # In-word hyphens between two lowercase letters with NO line break
        # remain untouched: "time-varying", "one-way", "k-search".
        text = _collapse_block("time-varying price bounds and one-way trading")
        assert "time-varying" in text
        assert "one-way" in text

    def test_preserves_hyphen_before_capital(self):
        # "Pareto-optimal" with no break should stay intact.
        text = _collapse_block("Pareto-optimal Algorithm")
        assert "Pareto-optimal" in text

    def test_rejoins_doi_suffix_split_at_period(self):
        # DOI path broken at a period: "acs.estlett.\n5b00213"
        raw = "Title. Journal, 1(1), 1-2. https://doi.org/10.1021/acs.estlett.\n5b00213"
        text = _collapse_block(raw)
        assert "10.1021/acs.estlett.5b00213" in text

    def test_doi_rejoin_does_not_eat_uppercase_next_sentence(self):
        # A URL ending with "." followed by an uppercase word (new sentence)
        # must NOT be merged.
        raw = "See https://example.com/page. The next sentence."
        text = _collapse_block(raw)
        assert "page. The" in text


# ---------------------------------------------------------------------------
# Title extraction — new venue-trailer fallback
# ---------------------------------------------------------------------------

class TestVenueTrailerExtraction:
    def test_algorithmica_with_volume_pages(self):
        rest = "Online search with time-varying price bounds. Algorithmica, 55(4), 619-642."
        assert _extract_title(rest) == "Online search with time-varying price bounds"

    def test_european_journal_with_volume_pages(self):
        rest = ("Competitive analysis of the online inventory problem. "
                "European Journal of Operational Research, 207(2), 685-696.")
        assert _extract_title(rest) == "Competitive analysis of the online inventory problem"

    def test_theoretical_cs_year_volume(self):
        # "Theoretical Computer Science, 2015, 593: 139-145" — colon trailer.
        rest = ("Online (J, K)-search problem and its competitive analysis. "
                "Theoretical Computer Science, 2015, 593: 139-145.")
        assert _extract_title(rest) == "Online (J, K)-search problem and its competitive analysis"

    def test_mathematics_of_no_pages(self):
        # No volume/pages -> the new venue prefix must catch it.
        rest = ("Robust online selection with uncertain offer acceptance. "
                "Mathematics of Operations Research.")
        assert _extract_title(rest) == "Robust online selection with uncertain offer acceptance"

    def test_strips_leading_ocr_fragment(self):
        # "T Theoretical Computer Science" — stray "T " gets discarded
        # along with the rest of the venue prefix.
        rest = ("Competitive algorithms for unbounded one-way trading. "
                "T Theoretical Computer Science, 607, 35-48.")
        assert _extract_title(rest) == "Competitive algorithms for unbounded one-way trading"

    def test_article_id_trailer(self):
        rest = ("Robust one-way trading with limited number of transactions. "
                "International Journal of Production Economics, 247, Article 108437.")
        assert _extract_title(rest) == "Robust one-way trading with limited number of transactions"

    def test_arxiv_preprint_suffix(self):
        rest = (
            "Human-timescale adaptation in an open-ended task space. "
            "arXiv preprint arXiv: Arxiv-2301.07608"
        )
        assert _extract_title(rest) == "Human-timescale adaptation in an open-ended task space"


# ---------------------------------------------------------------------------
# Nature-style block parsing (numbered format, year in trailing parens)
# ---------------------------------------------------------------------------

class TestNatureBlock:
    def test_multi_author_ampersand(self):
        raw = ("Fu, R., Feldman, D., Margolis, R., Woodhouse, M. & Ardani, K. "
               "US Solar Photovoltaic System Cost Benchmark: Q1 2017 Technical Report "
               "(National Renewable Energy Laboratory, 2017).")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2017
        assert e.authors == [
            "R. Fu", "D. Feldman", "R. Margolis", "M. Woodhouse", "K. Ardani"
        ]
        assert e.title is not None
        assert e.title.startswith("US Solar Photovoltaic")

    def test_et_al_authors(self):
        raw = ("Haegel, N. M. et al. Terawatt-scale photovoltaics: trajectories "
               "and challenges. Science 356, 141–143 (2017).")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2017
        assert e.authors == ["N. M. Haegel"]
        assert e.title == "Terawatt-scale photovoltaics: trajectories and challenges"

    def test_single_author_hyphenated_initial(self):
        raw = ("Richter, L.-L. Social Effects in the Diffusion of Solar "
               "Photovoltaic Technology in the UK Working Paper "
               "(Univ. Cambridge, 2013).")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2013
        assert e.authors == ["L.-L. Richter"]
        assert e.title is not None
        assert e.title.startswith("Social Effects")

    def test_corporate_no_authors(self):
        raw = ("Solar Market Insight Report 2017 Q3 Technical Report "
               "(Solar Energy Industries Association, 2017).")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2017
        assert e.authors == []
        assert e.title is not None
        assert "Solar Market Insight" in e.title

    def test_trailing_url_after_year_paren(self):
        # Entries with URLs after the year paren must still extract correctly.
        raw = ("CA SB-338 Integrated Resource Plan: Peak Demand "
               "(LegiScan, 2017); https://legiscan.com/CA/bill/SB338/2017")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2017
        assert "Integrated Resource Plan" in e.title

    def test_multi_word_surname_in_ampersand(self):
        raw = ("Ghaith, A. F., Epplin, F. M. & Scott Frazier, R. "
               "Economics of grid-tied household solar panel systems versus "
               "grid-only electricity. Renew. Sust. Energy Rev. 76, 407–424 (2017).")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2017
        assert "R. Scott Frazier" in e.authors
        assert e.title == ("Economics of grid-tied household solar panel systems "
                           "versus grid-only electricity")

    def test_journal_abbreviation_chain_stripped(self):
        # Title-end walk-back must skip past venue abbreviations like
        # "Renew. Sust. Energy Rev." and land at the actual title period.
        raw = ("Bollinger, B. & Gillingham, K. Peer effects in the diffusion "
               "of solar photovoltaic panels. Market. Sci. 31, 900–912 (2012).")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2012
        assert e.title == "Peer effects in the diffusion of solar photovoltaic panels"


class TestElsevierBlock:
    """Elsevier-style content: 'F.M. Last1, F.M. Last2, ..., Title, Venue Vol (YEAR) pp.'

    The initials-first authors are comma-separated (not period-separated),
    and (YYYY) is preceded by a digit (volume number) rather than a period.
    """

    def test_standard_journal_with_vol_pages(self):
        raw = ("O. Ellabban, H. Abu Rub, F. Blaabjerg, Renewable energy "
               "resources: Current status, future prospects and their "
               "enabling technology, Renew. Sustain. Energy Rev. 39 (2014) "
               "748-764.")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2014
        assert e.authors == ["O. Ellabban", "H. Abu Rub", "F. Blaabjerg"]
        assert e.title == ("Renewable energy resources: Current status, "
                           "future prospects and their enabling technology")

    def test_journal_with_issue_paren(self):
        # "Vol (Issue) (YEAR)" form — (5) is the issue, (2018) is the year.
        raw = ("E.F. Moran, M.C. Lopez, N. Moore, Sustainable hydropower in "
               "the 21st century, Proc. Natl. Acad. Sci. 115 (47) (2018) "
               "11891-11898.")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2018
        assert "E.F. Moran" in e.authors
        assert e.title == "Sustainable hydropower in the 21st century"

    def test_journal_no_volume_pages(self):
        # "Venue. (YYYY)." with no volume/pages.
        raw = ("J.H. Syu, M.E. Wu, G. Srivastava, An IoT-based hedge system "
               "for solar power generation, IEEE Internet Things J. (2021).")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2021
        assert e.authors == ["J.H. Syu", "M.E. Wu", "G. Srivastava"]
        assert e.title == ("An IoT-based hedge system for solar power "
                           "generation")

    def test_book_bare_year(self):
        raw = ("T.J. Kazmierski, S. Beeby, Energy Harvesting Systems, "
               "Springer, 2014.")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2014
        assert e.authors == ["T.J. Kazmierski", "S. Beeby"]
        assert e.title == "Energy Harvesting Systems"

    def test_conference_with_in_marker(self):
        raw = ("N. Zhang, Y. Yan, S. Xu, W. Su, Game-theory-based electricity "
               "market clearing mechanisms for an open and transactive "
               "distribution grid, in: 2015 IEEE Power & Energy Society "
               "General Meeting, IEEE, 2015")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2015
        assert e.title == ("Game-theory-based electricity market clearing "
                           "mechanisms for an open and transactive "
                           "distribution grid")

    def test_conference_with_trailing_pages(self):
        raw = ("N. Zhang, Y. Yan, S. Xu, W. Su, Game-theory-based electricity "
               "market clearing mechanisms for an open and transactive "
               "distribution grid, in: 2015 IEEE Power & Energy Society "
               "General Meeting, IEEE, 2015, pp. 1-5.")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2015
        assert e.authors == ["N. Zhang", "Y. Yan", "S. Xu", "W. Su"]
        assert e.title == ("Game-theory-based electricity market clearing "
                           "mechanisms for an open and transactive "
                           "distribution grid")

    def test_hyphenated_initials(self):
        # "J.C.-W. Lin" — initials separated by a hyphen.
        raw = ("J.C.-W. Lin, X. Wang, A novel approach, IEEE Access 7 (2019) "
               "1-10.")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2019
        assert "J.C.-W. Lin" in e.authors

    def test_multi_word_surname(self):
        # "H. Abu Rub" — surname has a space.
        raw = ("H. Abu Rub, F. Blaabjerg, Power systems, Energy 50 (2013) "
               "100-110.")
        e = _parse_numbered_block(1, raw)
        assert "H. Abu Rub" in e.authors
        assert "F. Blaabjerg" in e.authors

    def test_first_token_skips_initials(self):
        # The cite key should be the SURNAME, not the leading initial.
        raw = ("O. Ellabban, H. Abu Rub, F. Blaabjerg, X, Y 1 (2014) 1-2.")
        e = _parse_numbered_block(1, raw)
        assert e.raw_fields["first_token"] == "Ellabban"

    def test_rejects_acm_and_connector_author_list(self):
        raw = (
            "Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies "
            "with gradient descent is difficult. IEEE Transactions on Neural Networks, "
            "5(2):157-166, 1994."
        )
        assert _parse_elsevier_block(raw) is None


class TestIEEEAuthorYearInsideBracketed:
    """IEEE-style PDFs use bracketed numbering [N] but the *content* of each
    entry is author-year shaped: ``Authors. (YEAR). Title. Venue, vol, pp.``
    """

    def test_ieee_author_year_inline(self):
        raw = ("Nadeem, L., Azam, M. A., Amin, Y., Al-Ghamdi, M. A., "
               "Chai, K. K., Khan, M. F. N., & Khan, M. A. (2021). "
               "Integration of D2D, network slicing, and MEC in 5G cellular "
               "networks: Survey and challenges. IEEE Access, 9, 37590-37612.")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2021
        assert "L. Nadeem" in e.authors
        assert "M. A. Khan" in e.authors
        assert e.title == (
            "Integration of D2D, network slicing, and MEC in 5G cellular "
            "networks: Survey and challenges"
        )

    def test_title_can_start_with_digit(self):
        # "5G-V2X: …" — title begins with a digit, regex must allow \w not just [A-Z].
        raw = ("Abdel Hakeem, S. A., Hady, A. A., & Kim, H. (2020). "
               "5G-V2X: Standardization, architecture, use cases, "
               "network-slicing, and edgecomputing. Wireless Networks, "
               "26, 6015-6041.")
        e = _parse_numbered_block(1, raw)
        assert e.year == 2020
        assert e.title is not None
        assert e.title.startswith("5G-V2X:")
        assert any("Abdel" in a for a in e.authors)


class TestSplitLatexAlpha:
    """LaTeX alpha keys: [AGS20], [DBB+14], [DeG74], [Nic98], etc."""

    def test_splits_keys_on_own_line(self):
        text = (
            "[AGS20]\nAdams J. Title one. Journal, 2020.\n"
            "[BG08]\nBrown K. Title two. Journal, 2008.\n"
            "[CR20]\nChen L. Title three. Journal, 2020.\n"
        )
        blocks = _split_latex_alpha(text)
        assert len(blocks) == 3
        # Block 1's body should start with the entry text (alpha key stashed
        # in the sentinel prefix that the block parser consumes).
        assert "__ALPHA_KEY__AGS20" in blocks[0][1]
        assert "Adams J." in blocks[0][1]

    def test_splits_keys_inline_with_text(self):
        # Some PDFs put the key on the same line as the first author.
        text = (
            "[GMJCK13] P.H. Guerra, W. Meira. Title. Venue, 2013.\n"
            "[HMRU21] S. Haddadan. Title two. Venue, 2021.\n"
            "[Nic98] R. Nickerson. Title three. Venue, 1998.\n"
        )
        blocks = _split_latex_alpha(text)
        assert len(blocks) == 3
        assert "__ALPHA_KEY__GMJCK13" in blocks[0][1]
        assert "__ALPHA_KEY__Nic98" in blocks[2][1]

    def test_accepts_plus_in_key(self):
        text = (
            "[DBB+14]\nA. De. Title. Venue, 2014.\n"
            "[NHH+14]\nT. Nguyen. Title. Venue, 2014.\n"
            "[SCP+20]\nK. Sasahara. Title. Venue, 2020.\n"
        )
        blocks = _split_latex_alpha(text)
        assert len(blocks) == 3


class TestLatexAlphaBlock:
    """Block-level parsing of LaTeX alpha entries.

    Format: ``[Key]\\nAuthors. Title. Venue, ..., YEAR.`` (or trailing
    publisher / identifier noise after the year).
    """

    def _parse(self, alpha_key: str, body: str):
        raw = f"__ALPHA_KEY__{alpha_key}\n{body}"
        return _parse_block_alpha(1, raw)

    def test_standard_journal(self):
        e = self._parse(
            "BG08",
            "Delia Baldassarri and Andrew Gelman. Partisans without "
            "constraint: Political polarization and trends in american "
            "public opinion. American Journal of Sociology, 114(2):408-446, "
            "2008.",
        )
        assert e.key == "ref1"  # alpha key applied later by _assign_cite_keys
        assert e.raw_fields["alpha_key"] == "BG08"
        assert e.year == 2008
        assert e.authors == ["Delia Baldassarri", "Andrew Gelman"]
        assert e.title == (
            "Partisans without constraint: Political polarization and "
            "trends in american public opinion"
        )

    def test_three_authors_with_oxford_and(self):
        e = self._parse(
            "AGS20",
            "Guy Aridor, Duarte Goncalves, and Shan Sikdar. Deconstructing "
            "the filter bubble: User decision-making and recommender "
            "systems. Fourteenth ACM Conference on Recommender Systems, "
            "2020.",
        )
        assert e.year == 2020
        assert e.authors == ["Guy Aridor", "Duarte Goncalves", "Shan Sikdar"]
        assert e.title.startswith("Deconstructing the filter bubble")

    def test_leading_initial_not_mistaken_for_author(self):
        # "L. Elisa Celis" — the initial "L." would split the author block at
        # the wrong period if the lookbehind didn't require lowercase before
        # the boundary period.
        e = self._parse(
            "CKSV19",
            "L. Elisa Celis, Sayash Kapoor, Farnood Salehi, and Nisheeth "
            "Vishnoi. Controlling polarization in personalization. "
            "Proceedings of the Conference on Fairness, Accountability, "
            "and Transparency, 2019.",
        )
        assert e.year == 2019
        assert len(e.authors) == 4
        assert "L. Elisa Celis" in e.authors
        assert e.title == "Controlling polarization in personalization"

    def test_year_inside_identifier_not_picked_up(self):
        # Trailing identifier "SOCO-1481" contains a 4-digit number, but
        # the comma-anchored year regex rules it out (no leading ", ").
        e = self._parse(
            "TJ19",
            "Edson C. Tandoc Jr. The facts of fake news: A research "
            "review. Sociology Compass, 13(9):e12724, 2019. e12724 "
            "SOCO-1481.R1.",
        )
        assert e.year == 2019

    def test_year_in_body_text_excluded(self):
        # When the bibliography is followed by a body section, the block
        # picks up stray text. The comma-anchored year regex prefers the
        # citation year over incidental years in body text.
        e = self._parse(
            "XBZ21",
            "Wanyue Xu, Qi Bao, and Zhongzhi Zhang. Fast evaluation for "
            "relevant quantities of opinion dynamics. In Proceedings of "
            "the 30th International World Wide Web Conference (WWW), "
            "2021. A Deferred Proofs We now prove Lemma 6. The year 2013 "
            "appears here but not after a comma.",
        )
        assert e.year == 2021

    def test_novenue_fallback_authors_title_year(self):
        # arXiv preprints and reports: "Authors. Title, YEAR." (no venue).
        e = self._parse(
            "CR20",
            "Mayee Chen and Miklos Racz. Network disruption: maximizing "
            "disagreement and polarization in social networks, 2020.",
        )
        assert e.year == 2020
        assert e.authors == ["Mayee Chen", "Miklos Racz"]
        assert e.title == (
            "Network disruption: maximizing disagreement and "
            "polarization in social networks"
        )

    def test_single_author_book(self):
        e = self._parse(
            "Par12",
            "Eli Pariser. The filter bubble: what the Internet is hiding "
            "from you. Viking, 2012.",
        )
        assert e.year == 2012
        assert e.authors == ["Eli Pariser"]
        assert e.title == "The filter bubble: what the Internet is hiding from you"

    def test_alpha_key_assigned_as_cite_key(self):
        # The post-pass `_assign_cite_keys` should use the alpha key verbatim.
        e = self._parse(
            "DeG74",
            "Morris H. DeGroot. Reaching a consensus. Journal of the "
            "American Statistical Association, 69:118-121, 1974.",
        )
        _assign_cite_keys([e])
        assert e.key == "DeG74"


class TestChicagoAuthorsEnd:
    """Boundary detection: where do the Chicago authors end?

    The trickiest cases involve middle initials (must be consumed as part of
    the author name, not treated as a boundary period) and titles that begin
    with a single capital letter ("A New Economic View..." or "The American").
    """

    def test_simple_book(self):
        text = "Boorstin, Daniel. The Americans: ... 1965."
        end = _find_chicago_authors_end(text)
        assert text[:end].rstrip().rstrip('.') == "Boorstin, Daniel"

    def test_quoted_article_title_after_initial(self):
        text = 'David, Paul A. "New Light on a Statistical Dark Age." Venue.'
        end = _find_chicago_authors_end(text)
        # Quote unambiguously terminates the authors even after an initial.
        assert text[:end].rstrip() == 'David, Paul A.'

    def test_middle_initial_continuation(self):
        text = 'Blevins, Cameron, and Richard W. Helbock. "Title." Venue.'
        end = _find_chicago_authors_end(text)
        # "W." is a middle initial of Helbock — must be consumed, not break.
        assert text[:end].rstrip().rstrip('.') == "Blevins, Cameron, and Richard W. Helbock"

    def test_title_starting_with_article_after_initial(self):
        text = "Fuller, Wayne E. The American Mail: ... 1972."
        end = _find_chicago_authors_end(text)
        # "E." is an initial AND the title starts with "The" — the boundary
        # is after "E." because "The American..." is not a name continuation.
        assert text[:end].rstrip().rstrip('.') == "Fuller, Wayne E"

    def test_title_starting_with_single_letter_article(self):
        text = "Atack, Jeremy, and Peter Passell. A New Economic View."
        end = _find_chicago_authors_end(text)
        assert text[:end].rstrip().rstrip('.') == "Atack, Jeremy, and Peter Passell"

    def test_long_author_list_with_initials(self):
        text = ("Carter, Susan B., Scott Sigmund Gartner, Michael R. Haines, "
                "Alan L. Olmstead, Richard Sutch, and Gavin Wright, (eds.). "
                "Historical Statistics of the United States.")
        end = _find_chicago_authors_end(text)
        # Every "X. Surname" pair must stay in authors; the boundary is the
        # period after "(eds.)".
        result = text[:end]
        assert "Gavin Wright" in result
        assert "(eds.)" in result
        assert "Historical" not in result

    def test_suffix_jr_preserved(self):
        text = 'John, Richard R., Jr. "Private Mail Delivery." Venue.'
        end = _find_chicago_authors_end(text)
        assert "Jr" in text[:end]


class TestChicagoSplitter:
    def test_basic_split(self):
        text = (
            'Atack, Jeremy, and Peter Passell. A New Economic View. NY: Norton, 1994.\n'
            'Boorstin, Daniel. The Americans. NY: Random House, 1965.\n'
            'Coyle, Diane. GDP: A Brief History. Princeton, 2014.\n'
        )
        blocks = _split_chicago(text)
        assert len(blocks) == 3

    def test_ditto_marker(self):
        text = (
            'David, Paul A. "First Article." Venue 1 (1967a): 1-2.\n'
            '———. "Second Article." Venue 2 (1967b): 3-4.\n'
            '———. "Third Article." Venue 3 (1970): 5-6.\n'
        )
        blocks = _split_chicago(text)
        assert len(blocks) == 3

    def test_city_state_in_publisher_not_split(self):
        # "Washington, DC" inside the publisher clause must NOT trigger
        # a phantom new entry — it's part of the previous citation.
        text = (
            'North, Simon D. History of the Press. Washington, DC: GPO, 1884.\n'
            'Smith, John. Some Other Book. New York: Random House, 2000.\n'
        )
        blocks = _split_chicago(text)
        assert len(blocks) == 2
        assert "Washington, DC" in blocks[0][1]
        assert "1884" in blocks[0][1]

    def test_continuation_lines_not_split(self):
        # Wrap-continuation lines beginning with a "Word, Word" sequence in
        # the middle of an entry's venue clause must NOT trigger a phantom
        # new entry start.
        text = (
            'Easterlin, Richard. "State Income Estimates." In Population\n'
            'Growth, United States, 1870-1950, Vol. I, ... 1957.\n'
            'Engerman, Stanley. "The Effects of Slavery." Venue 1967.\n'
        )
        blocks = _split_chicago(text)
        assert len(blocks) == 2


class TestChicagoBlockParse:
    def test_article_with_quoted_title(self):
        raw = (
            'Cole, Arthur. "Cyclical and Sectional Variations in the Sale '
            'of Public Lands, 1816-60." Review of Economics and Statistics '
            '9, no. 1 (Jan. 1927): 41-53.'
        )
        e = _parse_block_chicago(1, raw)
        assert e.authors == ["Arthur Cole"]
        assert e.year == 1927
        assert e.title == (
            "Cyclical and Sectional Variations in the Sale of Public "
            "Lands, 1816-60"
        )

    def test_book_with_title(self):
        raw = (
            "Atack, Jeremy, and Peter Passell. A New Economic View of "
            "American History: from Colonial Times to 1940. Second "
            "Edition. New York: W. W. Norton, 1994."
        )
        e = _parse_block_chicago(1, raw)
        assert e.year == 1994
        assert e.authors == ["Jeremy Atack", "Peter Passell"]
        assert e.title.startswith("A New Economic View")

    def test_republication_brackets_picks_repub_year(self):
        # "Doubleday, 1969 [1835]." — 1969 is the republication year, 1835
        # is the bracketed original. Chicago year-extractor should prefer
        # the comma-anchored 1969.
        raw = "DeTocqueville, Alexis. Democracy in America. Garden City, NY: Doubleday, 1969 [1835]."
        e = _parse_block_chicago(1, raw)
        assert e.year == 1969
        assert e.title == "Democracy in America"

    def test_book_with_initial_then_title_starting_with_The(self):
        raw = "Fuller, Wayne E. The American Mail: Enlarger of the Common Life. Chicago: University of Chicago Press, 1972."
        e = _parse_block_chicago(1, raw)
        assert e.year == 1972
        assert e.authors == ["Wayne E Fuller"]
        assert e.title == "The American Mail: Enlarger of the Common Life"

    def test_ditto_marker_no_authors(self):
        raw = '———. "An Improved Annual Chronology." Journal 66 (2006): 103-21.'
        e = _parse_block_chicago(1, raw)
        # Ditto entry has no authors of its own — they're filled in later by
        # _resolve_chicago_ditto using the previous non-ditto entry.
        assert e.authors == []
        assert e.year == 2006
        assert e.title == "An Improved Annual Chronology"
        assert e.raw_fields["chicago_ditto"] is True


class TestMlaWorksCited:
    """MLA "Works Cited" sections — same grammar as Chicago, but the
    heading is different and a single PDF may contain multiple sections
    (e.g. essay collections with one bibliography per chapter).
    """

    def test_works_cited_heading_detected(self):
        from citation_checker.pdf_parser import _find_references_section
        text = (
            "Some body text here.\n"
            "\nWorks Cited\n"
            "Smith, John. A Book. NY: Norton, 2020.\n"
            "Doe, Jane. Another Book. NY: Random House, 2021.\n"
        )
        refs = _find_references_section(text)
        assert "Smith, John" in refs

    def test_multiple_works_cited_sections_concatenated(self):
        from citation_checker.pdf_parser import _find_references_section
        text = (
            "Body of essay 1.\n"
            "\nWorks Cited\n"
            "Smith, John. Book One. NY: Norton, 2020.\n"
            "Body of essay 2.\n"
            "\nWorks Cited\n"
            "Doe, Jane. Book Two. NY: Random House, 2021.\n"
        )
        refs = _find_references_section(text)
        # Both bibliographies must be in the returned block.
        assert "Smith, John" in refs
        assert "Doe, Jane" in refs

    def test_nested_quotes_in_title(self):
        # MLA titles often quote a phrase inside the article title.
        # Terminator should be the closing quote BEFORE the (year), not the
        # first inner closing quote.
        raw = (
            'Gallagher, Chris W. "“This Weird Thing I\'m Discovering” '
            'Toward a Critical Pedagogical Approach to Ghostwriting." '
            'Pedagogy 24.2 (2024): 195-213.'
        )
        e = _parse_block_chicago(1, raw)
        assert e.year == 2024
        assert e.authors == ["Chris W Gallagher"]
        assert "Toward a Critical Pedagogical Approach to Ghostwriting" in e.title

    def test_curly_apostrophe_in_title_not_a_closing_quote(self):
        # "Don't" uses curly apostrophe ’ which is in the close-quote class
        # but mustn't be treated as title-end.
        raw = (
            'Johnson, Gavin P. "Don’t act like you forgot: '
            'Approaching another literacy “crisis” by '
            'reconsidering what we know about teaching writing." '
            'Journal of X 9 (2023): 11-22.'
        )
        e = _parse_block_chicago(1, raw)
        assert e.year == 2023
        assert "Don’t act like you forgot" in e.title
        assert "teaching writing" in e.title

    def test_et_al_stripped_from_authors(self):
        # MLA uses "Lastname, First, et al." for 3+ authors. The "et al."
        # marker should NOT end up as a fake author named "et al".
        raw = (
            'Adisa, Kofi, et al. "Building a Culture for Generative AI '
            'Literacy." Some Workshop, 2023.'
        )
        e = _parse_block_chicago(1, raw)
        assert e.year == 2023
        assert e.authors == ["Kofi Adisa"]


class TestApaParsing:
    """APA-specific cases: `(n.d.)` no-date entries, heavily column-wrapped
    organization headers, article-number-only venue trailers, and URL
    line-break healing."""

    def test_n_d_marker_recognized_as_entry_start(self):
        from citation_checker.pdf_parser import _split_author_year
        text = (
            "Smith, J. (2020). First title. Journal A, 1(1), 1-10.\n"
            "Org. (n.d.). Second title with no date. https://org.example/foo\n"
            "Jones, K. (2021). Third title. Journal B, 2(1), 11-20.\n"
        )
        blocks = _split_author_year(text)
        assert len(blocks) == 3
        assert "Org. (n.d.)" in blocks[1][1]

    def test_n_d_block_parses_year_as_none(self):
        from citation_checker.pdf_parser import _parse_block
        e = _parse_block(1, "Org. (n.d.). Some inclusive language guide. https://org.example/x")
        assert e.year is None
        assert e.title is not None
        assert "inclusive language" in e.title.lower()

    def test_fragmented_org_header_recovered(self):
        """A run of 3+ consecutive single-token lines (PDF column wrap) is
        joined so the resulting entry can be detected by the splitter."""
        from citation_checker.pdf_parser import _split_author_year
        text = (
            "Previous, P. (2018). Previous entry. https://x.example/a\n"
            "International\n"
            "Organization\n"
            "for\n"
            "Migration\n"
            "(IOM).\n"
            "(2019).\n"
            "International migration law no. 34 - glossary on migration.\n"
            "https://publications.iom.int/books/glossary\n"
            "Next, N. (2020). Next entry. https://x.example/b\n"
        )
        blocks = _split_author_year(text)
        joined = " | ".join(b[1] for b in blocks)
        assert "International Organization for Migration (IOM). (2019)" in joined
        assert len(blocks) == 3

    def test_article_number_trailer_cuts_title(self):
        """Asian J Psychiatry 48, 101909 — single article number, no page range."""
        from citation_checker.pdf_parser import _extract_title
        t = _extract_title(
            "Labels used for persons with severe mental illness and their "
            "stigma experience in North India. Asian Journal of Psychiatry, "
            "48, 101909."
        )
        assert t == ("Labels used for persons with severe mental illness and "
                     "their stigma experience in North India")

    def test_e_prefixed_article_number_trailer(self):
        """PLoS One 16(4), e0249751 — e-prefixed article number with paren issue."""
        from citation_checker.pdf_parser import _extract_title
        t = _extract_title(
            "Establishing how social capital is studied in relation to "
            "cardiovascular disease. PLoS One, 16(4), e0249751."
        )
        assert t == ("Establishing how social capital is studied in relation "
                     "to cardiovascular disease")

    def test_paren_issue_range_trailer(self):
        """Schizophrenia Research 168(1–2), 9–15 — combined-issue range."""
        from citation_checker.pdf_parser import _extract_title
        t = _extract_title(
            "Stigma related to labels and symptoms in individuals at "
            "clinical high-risk for psychosis. Schizophrenia Research, "
            "168(1–2), 9–15."
        )
        assert t == ("Stigma related to labels and symptoms in individuals "
                     "at clinical high-risk for psychosis")

    def test_title_end_prefers_earlier_cut_over_url(self):
        """When both a venue trailer (JAMA, 325(11), 1049–1052) and an https
        URL appear, the title must end before the venue, not before the URL."""
        from citation_checker.pdf_parser import _extract_title
        t = _extract_title(
            "The reporting of race and ethnicity in medical and science "
            "journals: Comments invited. JAMA, 325(11), 1049–1052. "
            "https://doi.org/10.1001/jama.2021.2104"
        )
        assert t == ("The reporting of race and ethnicity in medical and "
                     "science journals: Comments invited")

    def test_url_line_break_healing(self):
        """A URL wrapped mid-path in the PDF should be glued back together
        so the URL regex can match the whole thing and the URL fragments
        don't leak into the title."""
        from citation_checker.pdf_parser import _collapse_block
        raw = (
            "Org. (2024). Diversity guide. "
            "https://www.elsevier.com/\n"
            "researcher/author/policies-and-\n"
            "guidelines/edi"
        )
        out = _collapse_block(raw)
        assert "https://www.elsevier.com/researcher/author/policies-and-guidelines/edi" in out

    def test_url_rejoin_does_not_absorb_plain_words(self):
        """Words after a URL that have no URL-shape chars must stay separate."""
        from citation_checker.pdf_parser import _collapse_block
        raw = "Title. https://example.com/path Then Some Sentence."
        out = _collapse_block(raw)
        # The URL stays intact and "Then Some Sentence" is NOT absorbed.
        assert "https://example.com/path" in out
        assert "Then Some Sentence" in out


class TestChicagoDittoResolution:
    def _make_entry(self, key, authors, year, title, ditto=False):
        from citation_checker.models import BibEntry
        raw_fields = {
            "raw_text": "",
            "first_token": authors[0].split()[-1] if authors else "",
        }
        if ditto:
            raw_fields["chicago_ditto"] = True
        return BibEntry(
            key=key, entry_type="article", title=title, authors=list(authors),
            year=year, doi=None, url=None, eprint=None, archiveprefix=None,
            raw_fields=raw_fields,
        )

    def test_inherits_authors_from_previous_entry(self):
        e1 = self._make_entry("a", ["Paul A David"], 1967, "First")
        e2 = self._make_entry("b", [], 1967, "Second", ditto=True)
        e3 = self._make_entry("c", [], 1970, "Third", ditto=True)
        _resolve_chicago_ditto([e1, e2, e3])
        assert e2.authors == ["Paul A David"]
        assert e3.authors == ["Paul A David"]
        # And first_token should be carried over so cite keys are sensible.
        assert e2.raw_fields["first_token"] == "David"
        assert e3.raw_fields["first_token"] == "David"

    def test_chain_breaks_at_next_real_author(self):
        e1 = self._make_entry("a", ["David"], 1967, "First")
        e2 = self._make_entry("b", [], 1967, "Second", ditto=True)
        e3 = self._make_entry("c", ["Davis"], 2004, "Third")
        e4 = self._make_entry("d", [], 2006, "Fourth", ditto=True)
        _resolve_chicago_ditto([e1, e2, e3, e4])
        assert e4.authors == ["Davis"]


class TestParsePdfAuthorsChicago:
    def test_inverts_first_author(self):
        # "Last, First, and First Last" -> ["First Last", "First Last"]
        assert _parse_pdf_authors_chicago(
            "Atack, Jeremy, and Peter Passell"
        ) == ["Jeremy Atack", "Peter Passell"]

    def test_three_or_more_authors(self):
        # First is inverted; rest are already "First Last" in source text.
        names = _parse_pdf_authors_chicago(
            "Carter, Susan B., Scott Sigmund Gartner, and Gavin Wright"
        )
        assert names[0] == "Susan B. Carter"
        assert "Scott Sigmund Gartner" in names
        assert "Gavin Wright" in names

    def test_strips_editor_suffix(self):
        names = _parse_pdf_authors_chicago(
            "Carter, Susan B., and Gavin Wright, (eds.)"
        )
        assert "(eds.)" not in " ".join(names)


class TestVancouverBlock:
    """Vancouver-style content: 'Surname I1, Surname I2, ..., et al. Title. Venue YEAR;vol:pp.'

    Key features distinguishing this from other formats:
      - Authors are surname-first with bare (no period) capital initials.
      - Year follows the venue directly, separated by ';' (journal) or '.' (book).
      - No parenthesized year anywhere in a well-formed entry.
    """

    def test_standard_journal(self):
        raw = (
            "Kohansal R, Martinez-Camblor P, Agusti A, Buist AS, Mannino DM, "
            "Soriano JB. The natural history of chronic airflow obstruction "
            "revisited: an analysis of the Framingham offspring cohort. Am J "
            "Respir Crit Care Med 2009;180:3-10."
        )
        e = _parse_numbered_block(1, raw)
        assert e.year == 2009
        assert "R. Kohansal" in e.authors
        assert "J. B. Soriano" in e.authors
        assert e.title == (
            "The natural history of chronic airflow obstruction revisited: "
            "an analysis of the Framingham offspring cohort"
        )

    def test_et_al_terminator(self):
        raw = (
            "Celli B, Fabbri L, Criner G, Martinez FJ, Mannino D, "
            "Vogelmeier C, et al. Definition and nomenclature of chronic "
            "obstructive pulmonary disease: time for its revision. Am J "
            "Respir Crit Care Med 2022;206:1317-1325."
        )
        e = _parse_numbered_block(1, raw)
        assert e.year == 2022
        assert len(e.authors) == 6
        assert "B. Celli" in e.authors
        assert "F. J. Martinez" in e.authors  # multi-letter initials expanded
        assert e.title.startswith("Definition and nomenclature")

    def test_question_mark_title(self):
        # Title ending with "?" — must be accepted as a sentence terminator.
        raw = (
            "Agusti A, Fabbri LM, Singh D, Vestbo J, Celli B, Franssen FME, "
            "et al. Inhaled corticosteroids in COPD: friend or foe? "
            "Eur Respir J 2018;52:1801219."
        )
        e = _parse_numbered_block(1, raw)
        assert e.year == 2018
        assert e.title == "Inhaled corticosteroids in COPD: friend or foe"
        assert len(e.authors) == 6

    def test_journal_paren_in_name_excluded_from_year(self):
        # "J Appl Physiol (1985)" is the journal name — the actual publication
        # year is 1988, picked up because it's followed by ";". Without the
        # YEAR_SEP regex this would mis-read the year as 1985.
        raw = (
            "Martin TR, Feldman HA, Fredberg JJ, Castile RG, Mead J, "
            "Wohl ME. Relationship between maximal expiratory flows and "
            "lung volumes in growing humans. J Appl Physiol (1985) "
            "1988;65:822-828."
        )
        e = _parse_numbered_block(1, raw)
        assert e.year == 1988
        assert e.title.startswith("Relationship between maximal expiratory")
        assert len(e.authors) == 6

    def test_lowercase_particle_in_surname(self):
        # "Montes de Oca M" — multi-word surname with lowercase particle.
        raw = (
            "Montes de Oca M. Smoking cessation/vaccinations. "
            "Clin Chest Med 2020;41:495-512."
        )
        e = _parse_numbered_block(1, raw)
        assert e.year == 2020
        assert e.authors == ["M. Montes de Oca"]
        assert e.title == "Smoking cessation/vaccinations"

    def test_hyphenated_surname(self):
        raw = (
            "Houchen-Wolloff L, Steiner MC. Pulmonary rehabilitation at "
            "a time of social distancing: prime time for tele-rehabilitation? "
            "Thorax 2020;75:446-447."
        )
        e = _parse_numbered_block(1, raw)
        assert e.year == 2020
        assert "L. Houchen-Wolloff" in e.authors
        assert e.title.startswith("Pulmonary rehabilitation at a time")

    def test_extended_initials_up_to_five(self):
        # "FAHM" — 4-letter initials block.
        raw = (
            "Cleutjens FAHM, Spruit MA, Ponds RWHM, Vanfleteren LEGW, "
            "Franssen FME, Gijsen C, et al. Cognitive impairment and "
            "clinical characteristics in patients with chronic obstructive "
            "pulmonary disease. Chron Respir Dis 2018;15:91-102."
        )
        e = _parse_numbered_block(1, raw)
        assert e.year == 2018
        assert len(e.authors) == 6
        assert any("Cleutjens" in a for a in e.authors)

    def test_corporate_author_fallback(self):
        # Authors regex doesn't match (no Surname+Initials pattern); the
        # corporate-Vancouver fallback should still extract author+title+year.
        raw = (
            "Nocturnal Oxygen Therapy Trial Group. Continuous or nocturnal "
            "oxygen therapy in hypoxemic chronic obstructive lung disease: "
            "a clinical trial. Ann Intern Med 1980;93:391-398."
        )
        e = _parse_numbered_block(1, raw)
        assert e.year == 1980
        assert e.authors == ["Nocturnal Oxygen Therapy Trial Group"]
        assert e.title.startswith("Continuous or nocturnal oxygen therapy")

    def test_corporate_author_publisher_year_form(self):
        # Book/report citation: "Corp. Title. Publisher; YEAR." — year follows
        # the semicolon rather than preceding one.
        raw = (
            "Global Initiative for Chronic Obstructive Lung Disease. "
            "Global strategy for prevention, diagnosis and management of "
            "COPD: 2023 report. Global Initiative for Chronic Obstructive "
            "Lung Disease; 2023. Available from: https://goldcopd.org/"
        )
        e = _parse_numbered_block(1, raw)
        assert e.year == 2023
        assert e.authors == ["Global Initiative for Chronic Obstructive Lung Disease"]
        assert "Global strategy for prevention" in e.title

    def test_no_personal_author_no_corporate_prefix(self):
        # Entry begins directly with the title (no author at all).
        raw = (
            "Long term domiciliary oxygen therapy in chronic hypoxic cor "
            "pulmonale complicating chronic bronchitis and emphysema: "
            "report of the Medical Research Council Working Party. Lancet "
            "1981;1:681-686."
        )
        e = _parse_numbered_block(1, raw)
        assert e.year == 1981
        assert e.authors == []
        assert e.title.startswith("Long term domiciliary oxygen therapy")

    def test_glyph_noise_stripped_from_surname(self):
        # PDF extraction can emit stray control chars or Euro signs for
        # diacritics — "G€unen H" for "Günen H", "Agust\x01ı A" for "Agustí A".
        raw = (
            "G€unen H, Tarraf H, Nemati A, Al Ghobain M, Al Mutairi S, "
            "Aoun Bacha Z. Waterpipe tobacco smoking. Tuberk Toraks "
            "2016;64:94-96."
        )
        e = _parse_numbered_block(1, raw)
        assert e.year == 2016
        assert e.title == "Waterpipe tobacco smoking"
        assert any("Gunen" in a for a in e.authors)

    def test_corporate_author_no_match_returns_none(self):
        # Directly: no year-with-semicolon, no year-publisher form, no
        # Vancouver author signature → fallback returns None.
        assert _parse_vancouver_corporate_block("Nothing here. Just text.") is None


class TestParsePdfAuthorsVancouver:
    def test_expands_initials_with_periods(self):
        assert _parse_pdf_authors_vancouver("Kohansal R, Buist AS") == [
            "R. Kohansal",
            "A. S. Buist",
        ]

    def test_strips_trailing_et_al(self):
        names = _parse_pdf_authors_vancouver("Smith A, Jones B, et al.")
        assert names == ["A. Smith", "B. Jones"]

    def test_multi_word_surname(self):
        assert _parse_pdf_authors_vancouver("Montes de Oca M") == ["M. Montes de Oca"]


class TestSplitNatureAuthors:
    def test_et_al_terminator(self):
        head = "Haegel, N. M. et al. Terawatt-scale photovoltaics"
        authors, rest = _split_nature_authors(head)
        assert authors == "Haegel, N. M. et al."
        assert rest == "Terawatt-scale photovoltaics"

    def test_ampersand_terminator(self):
        head = "Feldman, D. & Bolinger, M. On the Path to SunShot"
        authors, rest = _split_nature_authors(head)
        assert authors == "Feldman, D. & Bolinger, M."
        assert rest == "On the Path to SunShot"

    def test_single_author(self):
        head = "Richter, L.-L. Social Effects"
        authors, rest = _split_nature_authors(head)
        assert authors == "Richter, L.-L."
        assert rest == "Social Effects"

    def test_no_author_corporate(self):
        head = "Solar Market Insight Report 2017 Q3 Technical Report"
        authors, rest = _split_nature_authors(head)
        assert authors == ""
        assert rest == head


# ---------------------------------------------------------------------------
# Cite-key generation
# ---------------------------------------------------------------------------

def _make_entry(title="t", authors=None, year=None, first_token="") -> BibEntry:
    return BibEntry(
        key="placeholder",
        entry_type="article",
        title=title,
        authors=authors or [],
        year=year,
        doi=None,
        url=None,
        eprint=None,
        archiveprefix=None,
        raw_fields={"raw_text": "", "first_token": first_token},
    )


class TestFirstToken:
    def test_extracts_first_word(self):
        assert _first_token("Barberis N, Thaler R. (2003). Survey.") == "Barberis"

    def test_handles_diacritics(self):
        assert _first_token("Wøhlk, S. (2010). Title.") == "Wøhlk"

    def test_handles_hyphenated_surname(self):
        assert _first_token("El-Yaniv R. (1989). Solutions.") == "El-Yaniv"

    def test_empty_input(self):
        assert _first_token("") == ""

    def test_leading_whitespace(self):
        assert _first_token("   Lechowicz A.") == "Lechowicz"

    def test_skips_leading_initial(self):
        # Elsevier-style "F. Last, ..." — initial must be skipped.
        assert _first_token("O. Ellabban, H. Abu Rub") == "Ellabban"

    def test_skips_multi_initial_block(self):
        # "J.H." and "J.C.-W." styles.
        assert _first_token("J.H. Syu, X. Wang") == "Syu"
        assert _first_token("J.C.-W. Lin, A. B") == "Lin"


class TestAssignCiteKeys:
    def test_single_entry_year_present(self):
        entries = [_make_entry(year=2003, first_token="Barberis")]
        _assign_cite_keys(entries)
        assert entries[0].key == "Barberis2003"

    def test_collision_letter_suffix(self):
        entries = [
            _make_entry(year=2014, first_token="Mohr"),
            _make_entry(year=2014, first_token="Mohr"),
            _make_entry(year=2014, first_token="Mohr"),
        ]
        _assign_cite_keys(entries)
        assert [e.key for e in entries] == ["Mohr2014", "Mohr2014a", "Mohr2014b"]

    def test_different_years_no_collision(self):
        entries = [
            _make_entry(year=2015, first_token="Chin"),
            _make_entry(year=2017, first_token="Chin"),
        ]
        _assign_cite_keys(entries)
        assert [e.key for e in entries] == ["Chin2015", "Chin2017"]

    def test_no_first_token_falls_back_to_ref(self):
        entries = [
            _make_entry(year=2020, first_token="Adams"),
            _make_entry(year=2021, first_token=""),
            _make_entry(year=2022, first_token="Brown"),
        ]
        _assign_cite_keys(entries)
        assert entries[0].key == "Adams2020"
        assert entries[1].key == "ref2"
        assert entries[2].key == "Brown2022"

    def test_no_year_uses_bare_token(self):
        entries = [_make_entry(year=None, first_token="Smith")]
        _assign_cite_keys(entries)
        assert entries[0].key == "Smith"


class TestLooksCorporate:
    def test_org_name_all_caps_words(self):
        assert _looks_corporate("World Health Organization") is True

    def test_org_with_connector_lowercase(self):
        assert _looks_corporate("International Organization for Migration") is True

    def test_too_many_words(self):
        # 7 words — exceeds the ≤6 threshold
        assert _looks_corporate("The International Organization for Very Big Stuff") is False

    def test_sentence_cased_title_fails(self):
        # Title-cased strings with lowercase content words are not corporate names
        assert _looks_corporate("Central air conditioning energy") is False

    def test_single_word(self):
        assert _looks_corporate("NASA") is True

    def test_empty(self):
        assert _looks_corporate("") is False


class TestIcmlFormat:
    """ICML citation style: Surname, F., ..., and Surname, F. Title. Venue, Year."""

    # ---- format detection ----

    def test_detects_icml(self):
        text = (
            "Acun, B., Lee, B., Kazhamiaka, F., and Wu, C.-J. "
            "Understanding training efficiency of deep learning recommendation "
            "models at scale. ISCA, 2021.\n"
            "Agrawal, S., and Devanur, N. R. Fast algorithms for online stochastic "
            "convex programming. SODA, 2015.\n"
            "Azar, Y., Buchbinder, N., and Devanur, N. R. Online convex optimization "
            "against slow adversaries. ICALP, 2016.\n"
        )
        assert _detect_format(text) == CitationFormat.ICML

    def test_icml_not_confused_with_author_year(self):
        # ICML has no parenthesized year — format detector must pick ICML
        text = (
            "Bubeck, S., and Sellke, M. A universal law of robustness via isoperimetry. "
            "NeurIPS, 2021.\n"
            "Chen, X., Zhou, D., and Gu, Q. Nearly minimax optimal reinforcement "
            "learning for linear mixture Markov decision processes. COLT, 2021.\n"
            "Cutkosky, A., and Orabona, F. Momentum-based variance reduction in "
            "non-convex SGD. NeurIPS, 2019.\n"
        )
        assert _detect_format(text) == CitationFormat.ICML

    # ---- splitter ----

    def test_splits_three_entries(self):
        text = (
            "Agrawal, S., and Devanur, N. R. Fast algorithms for online stochastic "
            "convex programming. SODA, 2015.\n"
            "Azar, Y., Buchbinder, N., and Devanur, N. R. Online convex optimization "
            "against slow adversaries. ICALP, 2016.\n"
            "Bubeck, S., and Sellke, M. A universal law of robustness. NeurIPS, 2021.\n"
        )
        blocks = _split_icml(text)
        assert len(blocks) == 3

    def test_splitter_does_not_split_on_wrapped_author_comma(self):
        # The continuation of a multi-author list wraps onto the next line and
        # must NOT be treated as a new entry start.
        text = (
            "Agrawal, S., and Devanur, N. R. Fast algorithms. SODA, 2015.\n"
            "Kazhamiaka, F., Iyer, A. P., Netravali, R., Bhattacharya, M.,\n"
            "and Wu, C.-J. Meridian: a system for geo-distributed machine learning. "
            "OSDI, 2021.\n"
            "Chen, X., and Zhou, D. Minimax optimality. COLT, 2021.\n"
        )
        blocks = _split_icml(text)
        # The Kazhamiaka entry wraps: the second line starts with "and Wu" which
        # is not ICML line-start pattern, so it's continuation. Total = 3 entries.
        assert len(blocks) == 3

    def test_splitter_strips_appendix(self):
        # Text after the Appendix heading must be truncated before splitting.
        text = (
            "Acun, B., and Lee, B. Scalable training. ISCA, 2021.\n"
            "Bubeck, S. Convex optimization. FnT, 2015.\n"
            "\nAppendix\n"
            "Chen, X. This is appendix content that looks like a ref. 2020.\n"
        )
        blocks = _split_icml(text)
        assert len(blocks) == 2

    # ---- author parser ----

    def test_parse_authors_two(self):
        result = _parse_pdf_authors_icml("Bubeck, S., and Sellke, M.")
        assert result == ["Bubeck, S.", "Sellke, M."]

    def test_parse_authors_many(self):
        raw = "Acun, B., Lee, B., Kazhamiaka, F., Iyer, A. P., and Wu, C.-J."
        result = _parse_pdf_authors_icml(raw)
        assert len(result) == 5
        assert "Acun, B." in result
        assert "Wu, C.-J." in result

    def test_parse_authors_hyphenated_initial(self):
        result = _parse_pdf_authors_icml("Wu, C.-J., and Chen, X.")
        assert "Wu, C.-J." in result
        assert "Chen, X." in result

    def test_parse_authors_single(self):
        result = _parse_pdf_authors_icml("Shalev-Shwartz, S.")
        assert len(result) == 1
        assert result[0] == "Shalev-Shwartz, S."

    # ---- block parser ----

    def test_parses_two_author_entry(self):
        raw = "Bubeck, S., and Sellke, M. A universal law of robustness via isoperimetry. NeurIPS, 2021."
        e = _parse_block_icml(1, raw)
        assert e.year == 2021
        assert "Bubeck, S." in e.authors
        assert "Sellke, M." in e.authors
        assert e.title == "A universal law of robustness via isoperimetry"

    def test_parses_many_author_entry(self):
        raw = (
            "Acun, B., Lee, B., Kazhamiaka, F., Iyer, A. P., Nitu, V., "
            "Bhattacharya, M., and Wu, C.-J. Understanding training efficiency "
            "of deep learning recommendation models at scale. In Proceedings of "
            "the 48th Annual International Symposium on Computer Architecture, "
            "ISCA 2021, pp. 914-926. IEEE, 2021."
        )
        e = _parse_block_icml(1, raw)
        assert e.year == 2021
        assert len(e.authors) == 7
        assert "Acun, B." in e.authors
        assert e.title is not None
        assert "training efficiency" in e.title.lower()

    def test_parses_year_with_letter_suffix(self):
        raw = "Sun, Y., Zhou, D., and Gu, Q. Sparse recovery. ICML, 2021a."
        e = _parse_block_icml(1, raw)
        assert e.year == 2021

    def test_parses_doi(self):
        raw = (
            "Azar, Y., Buchbinder, N., and Devanur, N. R. "
            "Online convex optimization. ICALP, 2016. "
            "doi:10.4230/LIPIcs.ICALP.2016.1"
        )
        e = _parse_block_icml(1, raw)
        assert e.doi is not None
        assert e.doi.startswith("10.4230")

    def test_parses_arxiv_eprint(self):
        raw = (
            "Chen, X., Zhou, D., and Gu, Q. Nearly minimax optimal RL. "
            "COLT, 2021. arXiv:2102.06132."
        )
        e = _parse_block_icml(1, raw)
        assert e.eprint == "2102.06132"

    def test_url_word_stripped_before_https(self):
        # Some ICML PDFs emit "URL https://..." — the "URL" word must not
        # bleed into the title.
        raw = (
            "Zhang, A., and Guo, M. Online allocation with fairness. "
            "ICML, 2021. URL https://arxiv.org/abs/2103.00000."
        )
        e = _parse_block_icml(1, raw)
        assert e.title is not None
        assert "URL" not in e.title
        assert e.title == "Online allocation with fairness"

    def test_first_token_is_surname(self):
        raw = "Radovanovic, M., and Nanni, M. Reverse nearest neighbors. VLDB, 2010."
        e = _parse_block_icml(1, raw)
        assert e.raw_fields["first_token"] == "Radovanovic"


class TestNumberedBareYearBlock:
    def test_acm_initialed_author_list_with_and_connector(self):
        text = (
            "Y. Bengio, P. Simard, and P. Frasconi. Learning long-term dependencies "
            "with gradient descent is difficult. IEEE Transactions on Neural Networks, "
            "5(2):157-166, 1994."
        )
        result = _parse_numbered_bare_year_block(text)
        assert result is not None
        author_str, year, title = result
        assert author_str == "Y. Bengio, P. Simard, and P. Frasconi"
        assert year == 1994
        assert title == "Learning long-term dependencies with gradient descent is difficult"

    def test_final_author_with_lowercase_particle(self):
        text = (
            "Jan Wohlke, Felix Schmitt, and Herke van Hoof. "
            "A performance-based start state curriculum framework for reinforcement learning. "
            "In Proceedings of AAMAS, pages 1503-1511, 2020."
        )
        result = _parse_numbered_bare_year_block(text)
        assert result is not None
        author_str, year, title = result
        assert author_str == "Jan Wohlke, Felix Schmitt, and Herke van Hoof"
        assert year == 2020
        assert title == "A performance-based start state curriculum framework for reinforcement learning"

    def test_single_initialed_author(self):
        text = "C. M. Bishop. Neural networks for pattern recognition. Oxford university press, 1995."
        result = _parse_numbered_bare_year_block(text)
        assert result is not None
        author_str, year, title = result
        assert author_str == "C. M. Bishop"
        assert year == 1995
        assert title == "Neural networks for pattern recognition. Oxford university press"

    def test_multi_author_and_connector(self):
        text = "Smith J, Jones K, and Brown L. Deep learning for graphs. NeurIPS, 2019."
        result = _parse_numbered_bare_year_block(text)
        assert result is not None
        author_str, year, title = result
        assert year == 2019
        assert "Smith" in author_str
        assert "Brown" in author_str
        assert title is not None
        assert "Deep learning" in title

    def test_final_author_with_dotted_middle_initials(self):
        text = (
            "Bo Sun, Russell Lee, Mohammad Hajiesmaili, Adam Wierman, and "
            "Danny H.K. Tsang. Pareto-Optimal Learning-Augmented Algorithms "
            "for Online Conversion Problems. In Advances in Neural Information "
            "Processing Systems 34 (NeurIPS 2021), page 55a988dfb00a914717b3000a3374694c, 2021."
        )
        result = _parse_numbered_bare_year_block(text)
        assert result is not None
        author_str, year, title = result
        assert year == 2021
        assert author_str.endswith("Danny H.K. Tsang")
        assert title == "Pareto-Optimal Learning-Augmented Algorithms for Online Conversion Problems"

    def test_neurips_parenthetical_venue_falls_through_to_bare_year(self):
        from citation_checker.pdf_parser import _parse_block

        raw = (
            "Antonios Antoniadis, Christian Coester, Marek Eliáš, Adam Polak, "
            "and Bertrand Simon. Learningaugmented dynamic power management "
            "with multiple states via new ski rental bounds. In Advances in "
            "Neural Information Processing Systems (NeurIPS 2021), 2021. "
            "https://proceedings.neurips.cc/paper/2021/hash/"
            "8b8388180314a337c9aa3c5aa8e2f37a-Abstract.html."
        )
        e = _parse_block(1, raw, grammar="nature")
        assert e.year == 2021
        assert e.authors == [
            "Antonios Antoniadis",
            "Christian Coester",
            "Marek Eliáš",
            "Adam Polak",
            "Bertrand Simon",
        ]
        assert e.title == (
            "Learningaugmented dynamic power management with multiple states "
            "via new ski rental bounds"
        )

    def test_et_al_truncated_list(self):
        text = "Wang X, Li Y, et al. Attention is all you need. Advances in NIPS, 2017."
        result = _parse_numbered_bare_year_block(text)
        assert result is not None
        author_str, year, title = result
        assert year == 2017
        assert "Wang" in author_str
        assert "Attention" in title

    def test_single_author_with_initials(self):
        text = "Knuth D. The art of computer programming. Addison-Wesley, 1997."
        result = _parse_numbered_bare_year_block(text)
        assert result is not None
        author_str, year, title = result
        assert year == 1997
        assert title is not None
        assert "art of computer programming" in title.lower()

    def test_corporate_author(self):
        text = "World Health Organization. Global health report. WHO Press, 2020."
        result = _parse_numbered_bare_year_block(text)
        assert result is not None
        author_str, year, title = result
        assert year == 2020
        assert "World Health Organization" in author_str
        assert title is not None
        assert "Global health" in title

    def test_no_match_returns_none(self):
        # No trailing year — should not match
        text = "Smith J. Some title without a year at the end."
        assert _parse_numbered_bare_year_block(text) is None

    def test_year_in_parentheses_not_matched(self):
        # APA-style "(2019)" is not a bare trailing year
        text = "Smith J. (2019). Some title. Journal of Things."
        assert _parse_numbered_bare_year_block(text) is None

    def test_ampersand_author_connector(self):
        text = "Garcia A & Martinez B. Smart homes and IoT. Sensors, 2021."
        result = _parse_numbered_bare_year_block(text)
        assert result is not None
        author_str, year, title = result
        assert year == 2021
        assert title is not None
        assert "Smart homes" in title

    def test_title_extracted_without_venue(self):
        # Venue text should not appear in the extracted title
        text = "Lee C, Park D, and Kim E. Federated learning survey. ACM Computing Surveys, 54(1), 1-35, 2022."
        result = _parse_numbered_bare_year_block(text)
        assert result is not None
        author_str, year, title = result
        assert year == 2022
        assert title is not None
        assert "ACM" not in title
        assert "Federated learning" in title


# ---------------------------------------------------------------------------
# Integration tests against the real PDF
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not PDF_PATH.exists(), reason="PDF fixture not present")
class TestRealPdf:
    @pytest.fixture(scope="class")
    def entries(self):
        return parse_pdf_file(str(PDF_PATH))

    @pytest.fixture(scope="class")
    def by_key(self, entries):
        return {e.key: e for e in entries}

    def test_by_key_fixture_built(self, by_key):
        assert len(by_key) > 0

    def test_parses_nonzero_entries(self, entries):
        assert len(entries) > 50

    def test_keys_are_unique_and_nonempty(self, entries):
        keys = [e.key for e in entries]
        assert all(k for k in keys)
        assert len(set(keys)) == len(keys), "cite keys must be unique"

    def test_all_entry_types_are_article(self, entries):
        for e in entries:
            assert e.entry_type == "article"

    def test_first_entry_has_afram(self, entries):
        # ACM-style PDFs use "First Last" so the first token is the first name.
        # Just spot-check that the first entry's fields look right.
        e = entries[0]
        assert e.year == 2014
        assert any("Afram" in a for a in e.authors)
        assert e.title is not None
        assert "HVAC" in e.title or "control" in e.title.lower()

    def test_arxiv_entry_has_eprint(self, entries):
        arxiv_entries = [e for e in entries if e.eprint]
        assert len(arxiv_entries) > 0

    def test_doi_entry_has_doi(self, entries):
        doi_entries = [e for e in entries if e.doi]
        assert len(doi_entries) > 0

    def test_no_entry_is_completely_empty(self, entries):
        for e in entries:
            has_something = any([e.title, e.authors, e.year, e.doi, e.eprint])
            assert has_something, f"{e.key} has no parsed fields at all"


class TestTruncatedAuthorsDetection:
    """Bibliography entries whose original citation listed authors as
    ``First Author et al.`` carry only a single parsed author. Downstream
    fuzzy matching needs to know this so it doesn't treat the size disparity
    against a 6-author remote record as a coincidence."""

    def test_et_al_entry_is_flagged(self):
        from citation_checker.pdf_parser import _parse_block
        e = _parse_block(
            1,
            "Albillos, A. et al. The exocytotic event in chromaffin cells revealed by "
            "patch amperometry. Nature 389, 509-512 (1997).",
            grammar='nature',
        )
        from citation_checker.pdf_parser import _mark_truncated_author_lists
        _mark_truncated_author_lists([e])
        assert e.truncated_authors is True
        assert len(e.authors) == 1

    def test_full_author_list_is_not_flagged(self):
        from citation_checker.pdf_parser import _parse_block, _mark_truncated_author_lists
        e = _parse_block(
            1,
            "Brunger, A. T., Choi, U. B., Lai, Y., Leitz, J. & Zhou, Q. Molecular "
            "mechanisms of fast neurotransmitter release. Annu Rev Biophys 47, "
            "469-497 (2018).",
            grammar='nature',
        )
        _mark_truncated_author_lists([e])
        assert e.truncated_authors is False

    def test_fuzzy_lax_floor_when_truncated(self):
        """When local was truncated by 'et al.', the strict author-floor
        rule for highly-skewed sizes (remote/local >= 3) must NOT fire.
        Otherwise initial-vs-fullname mismatch on the single listed author
        flips a real VERIFY into a MISMATCH."""
        from citation_checker.fuzzy import compare_records
        from citation_checker.models import BibEntry, RemoteRecord, VerificationStatus
        local = BibEntry(
            key="x", entry_type="article", title="The exocytotic event",
            authors=["A. Albillos"], year=1997, doi=None, url=None,
            eprint=None, archiveprefix=None, raw_fields={},
            truncated_authors=True,
        )
        remote = RemoteRecord(
            title="The exocytotic event",
            authors=["Almudena Albillos", "Guillermo Dernick",
                     "Heiko Horstmann", "Walter Schroeder",
                     "Manfred Lindau", "Erwin Neher"],
            year=1997, source="crossref", raw_response={},
        )
        _, status, _ = compare_records(local, remote)
        assert status is VerificationStatus.VERIFIED

    def test_fuzzy_strict_floor_when_not_truncated(self):
        """Without the truncation marker, the same size disparity still
        invokes the strict floor — protecting against shared-surname
        coincidences."""
        from citation_checker.fuzzy import compare_records
        from citation_checker.models import BibEntry, RemoteRecord, VerificationStatus
        local = BibEntry(
            key="x", entry_type="article", title="The exocytotic event",
            authors=["A. Albillos"], year=1997, doi=None, url=None,
            eprint=None, archiveprefix=None, raw_fields={},
            truncated_authors=False,
        )
        remote = RemoteRecord(
            title="The exocytotic event",
            authors=["Almudena Albillos", "Guillermo Dernick",
                     "Heiko Horstmann", "Walter Schroeder",
                     "Manfred Lindau", "Erwin Neher"],
            year=1997, source="crossref", raw_response={},
        )
        _, status, _ = compare_records(local, remote)
        assert status is VerificationStatus.MISMATCH


class TestStripRecurringLines:
    """Page header/footer lines (paper title, journal name, self-DOI) repeat
    on every page of a PDF and pymupdf interleaves them with body text. If
    they leak into a bibliography entry's block, downstream DOI/year
    extraction picks up the enclosing paper's metadata instead of the cited
    paper's — the original symptom was the Heuser2024 entry in
    ``nature_comms_quantum.pdf`` getting tagged with the host paper's DOI."""

    def test_strips_lines_repeated_above_threshold(self):
        from citation_checker.pdf_parser import _strip_recurring_lines
        # Three identical footer lines (above threshold=3) + unique content.
        text = (
            "Real bibliography entry 1.\n"
            "Nature Communications| (2024) 15:21\n"
            "Real bibliography entry 2.\n"
            "Nature Communications| (2024) 15:21\n"
            "Real bibliography entry 3.\n"
            "Nature Communications| (2024) 15:21\n"
        )
        cleaned = _strip_recurring_lines(text)
        assert "Nature Communications" not in cleaned
        assert "Real bibliography entry 1." in cleaned
        assert "Real bibliography entry 3." in cleaned

    def test_keeps_lines_repeated_below_threshold(self):
        from citation_checker.pdf_parser import _strip_recurring_lines
        # A unique journal title in a single entry must survive.
        text = (
            "Smith, J. A study of X. J. Cell Biol. 88, 564–580 (1981).\n"
            "Doe, J. A study of Y. J. Cell Biol. 99, 100–120 (1990).\n"
        )
        assert _strip_recurring_lines(text) == text

    def test_strips_lines_containing_recurring_doi_even_if_unique(self):
        """The self-DOI may also appear in a one-off 'Supplementary information
        available at <self-DOI>.' line that occurs once but references the
        same DOI as the recurring footer. That line must also be stripped so
        the last bibliography entry doesn't pick up the host paper's DOI."""
        from citation_checker.pdf_parser import _strip_recurring_lines
        recurring_footer = "https://doi.org/10.1038/s41467-023-44539-7"
        once_off = "Supplementary information available at https://doi.org/10.1038/s41467-023-44539-7."
        text = (
            "Entry one.\n"
            f"{recurring_footer}\n"
            "Entry two.\n"
            f"{recurring_footer}\n"
            "Entry three.\n"
            f"{recurring_footer}\n"
            f"{once_off}\n"
            "Acknowledgements section.\n"
        )
        cleaned = _strip_recurring_lines(text)
        assert "10.1038/s41467-023-44539-7" not in cleaned
        assert "Entry one." in cleaned and "Acknowledgements" in cleaned

    def test_year_only_lines_are_preserved(self):
        """In numbered arXiv bibliographies the trailing year often wraps to
        its own line (``2018.``) and recurs across many entries. Stripping it
        as a recurring footer would lose the year for those entries — see the
        Eunsol2018 regression in arxiv_2310.06825_mistral.pdf where the year
        was being eaten and the arXiv ID prefix `1808` became the year."""
        from citation_checker.pdf_parser import _strip_recurring_lines
        text = (
            "[1] Smith. Title one, arXiv:1808.00001,\n2018.\n"
            "[2] Doe. Title two, arXiv:1808.00002,\n2018.\n"
            "[3] Roe. Title three, arXiv:1808.00003,\n2018.\n"
            "[4] Adam. Title four, arXiv:1808.00004,\n2018.\n"
        )
        cleaned = _strip_recurring_lines(text)
        assert cleaned.count("2018.") == 4

    def test_short_lines_are_not_treated_as_recurring(self):
        """Single-token recurring lines (page numbers, single letters) are
        handled by _NOISE_LINE_RE downstream; this helper only targets
        longer header/footer text to avoid stripping incidental punctuation."""
        from citation_checker.pdf_parser import _strip_recurring_lines
        text = "Foo.\n.\n.\n.\nBar.\n"
        # The single "." lines repeat 3 times but are below MIN_LEN.
        cleaned = _strip_recurring_lines(text)
        assert cleaned.count(".") >= 3


class TestChicagoWrappedAuthorListRejection:
    """The Chicago detector and splitter must not mistake a wrapped
    author-list continuation line for an entry start.

    Regression test for the Llama 2 PDF: bibliographies with very long
    author lists wrap mid-name, so a line like
    ``Shakeri, Emanuel Taropa, Paige Bailey, ...`` appears at column 0
    inside an entry. ``Shakeri, Emanuel`` matches the surface ``Last, First``
    pattern but is *not* a real entry start; the discriminator is the
    full-length surname-comma right after the first name."""

    @pytest.mark.parametrize("line", [
        "Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, Eric Chu",
        "Bradbury, Siddhartha Brahma, James Bradbury, Jonathan Heek",
        "Garcia, Sebastian Gehrmann, Lucas Gonzalez, Guy Gur-Ari",
    ])
    def test_wrapped_author_list_is_not_chicago_entry_start(self, line):
        from citation_checker.pdf_parser import _FMT_CHICAGO_RE, _CHICAGO_ENTRY_START_RE
        assert _FMT_CHICAGO_RE.match(line) is None
        assert _CHICAGO_ENTRY_START_RE.match(line) is None

    @pytest.mark.parametrize("line", [
        "Amari, Shun-Ichi. Natural gradient works efficiently in learning.",
        "Deng, Li, Li, Jinyu, Huang, Jui-Ting, Yao, Kaisheng, Yu, Dong,",
        "Hinton, Geoffrey E, Srivastava, Nitish, Krizhevsky, Alex.",
        "Maas, Andrew L, Daly, Raymond E, Pham, Peter T.",
        "———. Some other work by the same author.",
    ])
    def test_legitimate_chicago_lines_still_match(self, line):
        from citation_checker.pdf_parser import _FMT_CHICAGO_RE
        assert _FMT_CHICAGO_RE.match(line) is not None

    def test_count_helper_uses_prev_line_guard(self):
        """A first-name + last-name line whose previous line ends with `,`
        (still inside an author list) must not be counted as a new entry."""
        from citation_checker.pdf_parser import _count_chicago_entry_starts
        # Two real entries; the second's first line is preceded by `,` (the
        # first entry's author list continues), so the bare-regex matcher
        # would over-count without the prev-line guard.
        text = (
            "Smith, John. Title one. Venue, 2020.\n"
            "Doe, Jane, John Smith, Adam Lechowicz,\n"  # wrapped author list
            "Roe, Richard. Title two. Venue, 2021."
        )
        assert _count_chicago_entry_starts(text) == 2


class TestNameYearEnd:
    """``First Last, ..., and First Last. Title. Venue, YEAR.`` style.

    Distinguished from ACM_NAME_YEAR (which is ``Authors. YEAR. Title.``,
    year between authors and title) by where the year sits."""

    def test_detected_when_year_at_end(self):
        from citation_checker.pdf_parser import _detect_format, CitationFormat
        refs = (
            "Daron Acemoglu and Pascual Restrepo. Artificial intelligence, "
            "automation, and work. In The economics of artificial intelligence, "
            "pages 197-236. University of Chicago Press, 2018.\n"
            "Joshua Ainslie and Sumit Sanghai. Generalized multi-query "
            "transformer models from multi-head checkpoints, 2023.\n"
            "Yuntao Bai and Andy Jones. Constitutional AI: Harmlessness from "
            "AI feedback. arXiv preprint arXiv:2212.08073, 2022.\n"
        )
        assert _detect_format(refs) is CitationFormat.NAME_YEAR_END

    def test_not_detected_when_year_is_early(self):
        """ACL-style ``Authors. YEAR. Title.`` (year between authors and title)
        must still detect as ACM_NAME_YEAR, not NAME_YEAR_END — the splitter
        gate requires ``, YEAR.`` at end-of-line."""
        from citation_checker.pdf_parser import _detect_format, CitationFormat
        refs = (
            "Mario Barrantes, Benedikt Herudek, and Richard Wang. 2020. "
            "Adversarial nli for factual correctness in text summarisation. "
            "arXiv preprint arXiv:2005.11739.\n"
            "Jacob Devlin, Ming-Wei Chang, and Kenton Lee. 2019. BERT: "
            "Pre-training of deep bidirectional transformers. In NAACL.\n"
            "Yonatan Belinkov and Nadir Durrani. 2017. What do neural machine "
            "translation models learn about morphology. In ACL.\n"
        )
        assert _detect_format(refs) is CitationFormat.ACM_NAME_YEAR

    def test_block_parser_extracts_authors_title_year(self):
        from citation_checker.pdf_parser import _parse_block_name_year_end
        raw = (
            "Joshua Ainslie, James Lee-Thorp, Michiel de Jong, Yury "
            "Zemlyanskiy, Federico Lebron, and Sumit Sanghai. Gqa: "
            "Training generalized multi-query transformer models from "
            "multi-head checkpoints, 2023."
        )
        e = _parse_block_name_year_end(1, raw)
        assert e.year == 2023
        assert e.title is not None and "multi-query transformer" in e.title.lower()
        assert e.authors[0] == "Joshua Ainslie"
        assert "Sumit Sanghai" in e.authors
        # Middle-initial author "Andrew M. Dai" — initials shouldn't break the split
        assert all(a.count(".") <= 2 for a in e.authors), e.authors

    def test_block_parser_handles_arxiv_suffix_with_year_at_end(self):
        """Year must survive the arXiv-suffix stripping path."""
        from citation_checker.pdf_parser import _parse_block_name_year_end
        raw = (
            "Yuntao Bai and Andy Jones. Constitutional AI: Harmlessness from "
            "AI feedback. arXiv preprint arXiv:2212.08073, 2022."
        )
        e = _parse_block_name_year_end(1, raw)
        assert e.year == 2022
        assert e.eprint == "2212.08073"
        assert e.title is not None and "Constitutional AI" in e.title

    def test_splitter_skips_wrapped_author_list_continuations(self):
        """Same false-positive pattern as the Chicago wrapped-list test, but
        for the NAME_YEAR_END splitter."""
        from citation_checker.pdf_parser import _split_name_year_end
        text = (
            "John Smith and Adam Lechowicz. First paper title, 2020.\n"
            "Jane Doe, Adam Lechowicz, John Smith, Bob Roe,\n"  # wrapped author list — should NOT start a new entry
            "Carol Lin, and Dave Chen. Second paper title, 2021.\n"
        )
        blocks = _split_name_year_end(text)
        assert len(blocks) == 2, [b[1][:60] for b in blocks]
