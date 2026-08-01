import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from PIL import ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import generate_flashcard_bmps as bmp
from enrich_flashcards import (
    enrich_word,
    extract_definition_and_translations,
    parse_examples_and_synonyms_response,
)
from epub_to_flashcards import filter_rows_by_source_text, parse_csv_response


class BilingualCsvTests(unittest.TestCase):
    def test_rejects_commentary_without_requested_header(self):
        response = """Now,I will select the 30 best items that meet strict C1-C2 criteria:
Note: I've selected items that are idiomatic expressions,"phrasal verbs, advanced vocabulary, and strong collocations."
"""

        self.assertEqual(parse_csv_response(response, "English", "Spanish"), [])

    def test_parses_only_rows_after_requested_header(self):
        response = """I selected the strongest candidates.
```csv
English,Spanish
bear in mind,tener en cuenta
"by and large","en general, por lo general"
```
"""

        self.assertEqual(
            parse_csv_response(response, "English", "Spanish"),
            [
                ("bear in mind", "tener en cuenta"),
                ("by and large", "en general, por lo general"),
            ],
        )

    def test_source_filter_requires_the_complete_phrase(self):
        rows = [
            ("selected items that are idiomatic", "invalid"),
            ("bear in mind", "tener en cuenta"),
        ]
        chapter = "Bear in mind that selected examples need careful review."

        self.assertEqual(
            filter_rows_by_source_text(rows, chapter),
            [("bear in mind", "tener en cuenta")],
        )


class StarDictParsingTests(unittest.TestCase):
    def test_separates_flat_translation_from_definition(self):
        entry = '<div><div><font class="grammar" color="green">noun</font></div>process in which code is refactored<div>refactorización</div></div>'

        self.assertEqual(
            extract_definition_and_translations(entry),
            ("noun. process in which code is refactored", ["refactorización"], ""),
        )

    def test_preserves_unicode_pronunciation(self):
        entry = '<div>/<font color="gray">ˈaː.kɪˌtek.t͡ʃə</font>/, /<font color="gray">ˈɑɹ.kɪˌtɛk.t͡ʃɚ</font>/<br><div><font class="grammar" color="green">noun</font></div>art and science of designing buildings<div>arquitectura</div></div>'

        self.assertEqual(
            extract_definition_and_translations(entry),
            (
                "noun. art and science of designing buildings",
                ["arquitectura"],
                "ˈaː.kɪˌtek.t͡ʃə / ˈɑɹ.kɪˌtɛk.t͡ʃɚ",
            ),
        )

    def test_keeps_nested_definitions_and_translations(self):
        entry = '<div><div><font class="grammar" color="green">noun</font></div><ol><li>inconsistency</li><li>discrepant state</li></ol><div>discrepancia</div></div>'

        definition, translations, pronunciation = extract_definition_and_translations(entry)

        self.assertEqual(definition, "noun. inconsistency discrepant state")
        self.assertEqual(translations, ["discrepancia"])
        self.assertEqual(pronunciation, "")


    def test_grammar_without_definition_is_empty(self):
        entry = '<div><div><font class="grammar" color="green">adjective</font></div><div>robusto</div></div>'

        self.assertEqual(
            extract_definition_and_translations(entry),
            ("", ["robusto"], ""),
        )

    def test_parser_collects_examples_and_source_language_synonyms(self):
        response = """word,example,synonyms
refactoring,We improved the design through careful refactoring.,restructuring / cleanup
robust,The solution is robust and reliable.,sturdy / resilient
mystery,,empty translation
"""

        examples, synonyms = parse_examples_and_synonyms_response(
            response, ["refactoring", "robust", "mystery"]
        )

        self.assertEqual(
            examples["refactoring"],
            "We improved the design through careful refactoring.",
        )
        self.assertEqual(synonyms["refactoring"], "restructuring, cleanup")
        self.assertEqual(synonyms["robust"], "sturdy, resilient")
        self.assertEqual(synonyms.get("mystery"), "empty translation")
        self.assertEqual(
            examples,
            {
                "refactoring": "We improved the design through careful refactoring.",
                "robust": "The solution is robust and reliable.",
            },
        )
        self.assertEqual(
            synonyms,
            {
                "refactoring": "restructuring, cleanup",
                "robust": "sturdy, resilient",
                "mystery": "empty translation",
            },
        )


    def test_enrichment_excludes_target_language_translations(self):
        class Dictionary:
            def lookup(self, word):
                return '<div><div><font class="grammar" color="green">noun</font></div>process in which code is refactored<div>refactorización</div></div>'

            def synonyms(self, word):
                return ["restructuring"]

        self.assertEqual(
            enrich_word(Dictionary(), "refactoring"),
            (
                "noun. process in which code is refactored",
                ["restructuring"],
                "",
            ),
        )


class BmpRenderingTests(unittest.TestCase):
    def test_formats_at_most_two_pronunciation_variants(self):
        self.assertEqual(
            bmp.format_pronunciation("tɛkˈnɑ.lə.dʒi / tɛkˈnɒl.ə.dʒi / third"),
            "/tɛkˈnɑ.lə.dʒi/  /tɛkˈnɒl.ə.dʒi/  …",
        )

    def test_section_titles_for_returns_expected_language(self):
        self.assertEqual(bmp.section_titles_for("en")["definition"], "Definition")
        self.assertEqual(bmp.section_titles_for("es")["definition"], "Definición")
        # Unknown language falls back to English.
        self.assertEqual(bmp.section_titles_for("xx")["definition"], "Definition")
        self.assertEqual(bmp.section_titles_for(None)["definition"], "Definition")

    def test_render_card_uses_localized_section_titles(self):
        headings = []
        original = bmp.draw_section_title

        def record_heading(draw, text, font, y, x_left, x_right, fill=0):
            headings.append(text)
            return original(draw, text, font, y, x_left, x_right, fill)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "card.bmp"
            with patch.object(bmp, "find_font", return_value=ImageFont.load_default()), patch.object(
                bmp, "draw_section_title", side_effect=record_heading
            ):
                bmp.render_card(
                    word="refactorización",
                    pronunciation="ri-fak-to-ri-za-SION",
                    definition="sustantivo. proceso en el que se refactoriza código",
                    example="Mejoramos el diseño mediante una refactorización cuidadosa.",
                    synonyms="reestructuración / limpieza",
                    book_name="El Libro del Caballero",
                    page_no=1,
                    total_pages=1,
                    output_path=output,
                    section_titles=bmp.section_titles_for("es"),
                )

        self.assertEqual(
            headings,
            ["Pronunciación", "Definición", "Uso", "Sinónimos"],
        )

    def test_font_path_overrides_only_the_word_role(self):
        """When font_path is set, only the word role uses it. The italic /
        sans / body roles keep the curated candidates so the card retains
        IPA coverage and visual hierarchy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fonts_dir = Path(tmpdir)
            ttf_path = fonts_dir / "Custom.ttf"
            ttf_path.write_bytes(b"\x00\x01\x00\x00")

            output = fonts_dir / "card.bmp"
            with patch.object(
                bmp.ImageFont, "truetype", return_value=bmp.ImageFont.load_default()
            ) as truetype:
                bmp.render_card(
                    word="refactoring",
                    pronunciation="",
                    definition="noun. process in which code is refactored",
                    example="",
                    synonyms="",
                    book_name="Book",
                    output_path=output,
                    font_path=str(ttf_path),
                )

        called_paths = [str(p) for p in (call.args[0] for call in truetype.call_args_list)]
        # Exactly one call should use the user-supplied font (the word role).
        word_overrides = [p for p in called_paths if p == str(ttf_path)]
        self.assertEqual(
            len(word_overrides), 1,
            f"expected exactly one ImageFont.truetype call to use the override, "
            f"got {len(word_overrides)}: {word_overrides}",
        )
        # The remaining calls go through find_font and should not match the
        # override path (they fall back to FONT_CANDIDATES).
        curated_calls = [p for p in called_paths if p != str(ttf_path)]
        self.assertGreaterEqual(
            len(curated_calls), 6,
            "expected the italic / sans / body roles to fall back to find_font",
        )

    def test_font_path_does_not_affect_ipa_shrink_loop(self):
        """The IPA shrink-to-fit fallback uses the curated italic regardless
        of the user's font_path choice (so phonetic glyphs stay covered)."""
        fonts_dir = Path(tempfile.mkdtemp())
        try:
            ttf_path = fonts_dir / "Custom.ttf"
            ttf_path.write_bytes(b"\x00\x01\x00\x00")
            output = fonts_dir / "card.bmp"

            with patch.object(
                bmp, "find_font",
                return_value=bmp.ImageFont.load_default(),
            ):
                bmp.render_card(
                    word="refactoring",
                    pronunciation="a" * 200,  # forces the shrink loop
                    definition="noun. process in which code is refactored",
                    example="",
                    synonyms="",
                    book_name="Book",
                    output_path=output,
                    font_path=str(ttf_path),
                )
        finally:
            import shutil
            shutil.rmtree(fonts_dir, ignore_errors=True)

    def test_section_heading_does_not_draw_a_divider(self):
        class DrawRecorder:
            def __init__(self):
                self.text_calls = []

            def text(self, position, text, font, fill, **kwargs):
                self.text_calls.append((position, text, fill))

        draw = DrawRecorder()
        font = ImageFont.load_default()

        next_y = bmp.draw_section_title(draw, "Definition", font, 10, 20, 200)

        self.assertEqual(draw.text_calls[0][1], "DEFINITION")
        self.assertGreater(next_y, 10)

    def test_omits_empty_usage_and_synonym_sections(self):
        headings = []
        original = bmp.draw_section_title

        def record_heading(draw, text, font, y, x_left, x_right, fill=0):
            headings.append(text)
            return original(draw, text, font, y, x_left, x_right, fill)

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "card.bmp"
            with patch.object(bmp, "find_font", return_value=ImageFont.load_default()), patch.object(
                bmp, "draw_section_title", side_effect=record_heading
            ):
                bmp.render_card(
                    word="refactoring",
                    pronunciation="ri-fak-tor-ing",
                    definition="noun. process in which code is refactored",
                    example="",
                    synonyms="",
                    book_name="The Gentle-Man Book",
                    page_no=1,
                    total_pages=1,
                    output_path=output,
                )

        self.assertEqual(headings, ["Pronunciation", "Definition"])

    def test_rejects_card_without_definition(self):
        with self.assertRaisesRegex(ValueError, "definition is required"):
            bmp.render_card(
                word="robust",
                pronunciation="",
                definition="  ",
                example="",
                synonyms="",
                book_name="The Gentle-Man Book",
            )

    def test_missing_scalable_fonts_raise_instead_of_silently_falling_back(self):
        candidates = {key: ["/missing/font.ttf"] for key in bmp.FONT_CANDIDATES}
        with patch.object(bmp, "FONT_CANDIDATES", candidates), patch.object(
            bmp, "_EXTRA_FONT_ROOTS", ("/tmp/no-fonts-here-12345",)
        ):
            with self.assertRaisesRegex(RuntimeError, "install DejaVu fonts"):
                bmp.find_font("serif", 56)

    def test_finds_font_via_extra_roots_when_curated_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fonts_dir = Path(tmpdir)
            ttf_path = fonts_dir / "CustomSans.ttf"
            ttf_path.write_bytes(b"\x00\x01\x00\x00")

            with patch.dict(bmp.FONT_CANDIDATES, {"serif": ["/missing/font.ttf"], "sans": ["/missing/font.ttf"], "sans_bold": ["/missing/font.ttf"], "serif_italic": ["/missing/font.ttf"], "sans_italic": ["/missing/font.ttf"], "mono": ["/missing/font.ttf"]}), patch.object(
                bmp, "_EXTRA_FONT_ROOTS", (str(fonts_dir),)
            ), patch.object(
                bmp.ImageFont, "truetype", return_value=bmp.ImageFont.load_default()
            ) as truetype:
                font = bmp.find_font("serif", 56)

        self.assertIsInstance(font, bmp.ImageFont.FreeTypeFont)
        truetype.assert_called()
        called_paths = [call.args[0] for call in truetype.call_args_list]
        self.assertTrue(any("CustomSans.ttf" in str(path) for path in called_paths))

    def test_draw_centered_text_shrinks_via_mock(self):
        """Direct: shrink-to-fit shrinks the size and stops when it fits."""
        big_font = MagicMock(spec=ImageFont.FreeTypeFont)
        big_font.size = 60

        sizes_used: list[int] = []

        def fake_find(family, size, **kwargs):
            sizes_used.append(size)
            f = MagicMock(spec=ImageFont.FreeTypeFont)
            f.size = size
            return f

        def fake_measure(text, font):
            return (font.size * 10, font.size // 2)

        with patch.object(bmp, "find_font", side_effect=fake_find), \
             patch.object(bmp, "measure", side_effect=fake_measure):
            bmp.draw_centered_text(
                MagicMock(), "Xteink Flashcards", big_font, 0, width=240,
            )

        self.assertTrue(sizes_used, "find_font was not called")
        self.assertLess(min(sizes_used), 60)

    def test_draw_centered_text_skips_shrink_when_text_already_fits(self):
        font = MagicMock(spec=ImageFont.FreeTypeFont)
        font.size = 20

        sizes_used: list[int] = []

        def fake_find(family, size, **kwargs):
            sizes_used.append(size)
            f = MagicMock(spec=ImageFont.FreeTypeFont)
            f.size = size
            return f

        with patch.object(bmp, "find_font", side_effect=fake_find), \
             patch.object(bmp, "measure", return_value=(100, 20)):
            bmp.draw_centered_text(MagicMock(), "refactoring", font, 0, width=240)

        self.assertEqual(sizes_used, [])

    def test_truncate_lines_shrinks_when_token_is_wider_than_max_width(self):
        """Long URLs that can't be split still fit by shrinking the font."""
        big_font = MagicMock(spec=ImageFont.FreeTypeFont)
        big_font.size = 30

        sizes_used: list[int] = []

        def fake_find(family, size, **kwargs):
            sizes_used.append(size)
            f = MagicMock(spec=ImageFont.FreeTypeFont)
            f.size = size
            return f

        def fake_measure(text, font):
            return (font.size * 100, font.size // 2)

        with patch.object(bmp, "find_font", side_effect=fake_find), \
             patch.object(bmp, "measure", side_effect=fake_measure):
            lines = bmp.truncate_lines(
                "xtctool.com/flashcard-generator",
                big_font, max_width=200, max_lines=4,
            )

        # The single long token still fits because the font was shrunk.
        self.assertEqual(len(lines), 1)
        self.assertIn("xtctool", lines[0])
        self.assertTrue(sizes_used, "find_font was not called")
        self.assertLess(min(sizes_used), 30)


if __name__ == "__main__":
    unittest.main()
