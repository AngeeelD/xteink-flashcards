import json
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "webapp"))

import app as webapp


class BookNameTests(unittest.TestCase):
    def test_uses_uploaded_filename_instead_of_job_id(self):
        self.assertEqual(
            webapp._book_name_from_upload("The Gentle-Man Book -- AngeeelD.epub"),
            "The Gentle-Man Book",
        )

    def test_strips_client_path_components(self):
        self.assertEqual(
            webapp._book_name_from_upload("C:\\uploads\\Architecture.epub"),
            "Architecture",
        )


class WebBilingualPipelineTests(unittest.TestCase):
    def test_retries_invalid_model_csv_once(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            epub_path = root / "job-id.epub"
            with zipfile.ZipFile(epub_path, "w") as archive:
                archive.writestr(
                    "chapter_001.xhtml",
                    "<p>Bear in mind " + "advanced context " * 30 + "</p>",
                )
            invalid = "Now,I will select the best items"
            valid = "English,Spanish\nbear in mind,tener en cuenta"

            with patch.object(
                webapp, "ollama_generate_bi", side_effect=[invalid, valid]
            ) as generate:
                result = webapp._generate_bilingual_per_chapter(
                    epub_path=epub_path,
                    chapters=[("chapter_001", "chapter_001.xhtml")],
                    source="en",
                    target="es",
                    items=1,
                    csv_outdir=root / "csv",
                    update_status=lambda **kwargs: None,
                )

            output = (root / "csv" / "chapter_001.csv").read_text()

        self.assertEqual(result, (1, 1))
        self.assertEqual(generate.call_count, 2)
        self.assertEqual(output, "English,Spanish\nbear in mind,tener en cuenta\n")


class WebBmpPipelineTests(unittest.TestCase):
    def test_dark_reuses_light_cards_without_calling_ollama_twice(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            epub_path = root / "job-id.epub"
            with zipfile.ZipFile(epub_path, "w") as archive:
                archive.writestr(
                    "chapter_001.xhtml",
                    "<p>refactoring " + "context " * 40 + "</p>",
                )

            def write_bmp(**kwargs):
                kwargs["output_path"].write_bytes(b"bmp")

            with patch.object(webapp, "OLLAMA_HOST", "http://ollama-host:11434"), patch.object(
                webapp, "ef_ollama_generate", return_value="word\nrefactoring"
            ) as generate, patch.object(
                webapp,
                "enrich_word",
                return_value=("noun. process in which code is refactored", [], ""),
            ),             patch.object(
                webapp,
                "ollama_batch_examples_and_synonyms",
                return_value=(
                    {"refactoring": "We improved the design through careful refactoring."},
                    {"refactoring": "restructuring"},
                ),
            ) as examples, patch.object(
                webapp, "render_card", side_effect=write_bmp
            ) as render_card:
                light_result = webapp._generate_bmps_per_chapter(
                    epub_path=epub_path,
                    book_name="The Gentle-Man Book",
                    chapters=[("chapter_001", "chapter_001.xhtml")],
                    source="en",
                    items=1,
                    with_examples=True,
                    bmp_outdir=root / "bmp",
                    enriched_csv_outdir=root / "csv",
                    sd=object(),
                    update_status=lambda **kwargs: None,
                )
                dark_result = webapp._generate_bmps_per_chapter(
                    epub_path=epub_path,
                    book_name="The Gentle-Man Book",
                    chapters=[("chapter_001", "chapter_001.xhtml")],
                    source="en",
                    items=1,
                    with_examples=True,
                    bmp_outdir=root / "bmp_dark",
                    enriched_csv_outdir=root / "csv",
                    sd=object(),
                    update_status=lambda **kwargs: None,
                    darkmode=True,
                )
                light_archive_count = webapp._zip_dir(root / "bmp", root / "light.zip")
                dark_archive_count = webapp._zip_dir(
                    root / "bmp_dark", root / "dark.zip"
                )

        self.assertEqual(light_result, (1, 1))
        self.assertEqual(dark_result, (1, 1))
        self.assertEqual(light_archive_count, dark_archive_count)
        generate.assert_called_once()
        examples.assert_called_once_with(
            ["refactoring"],
            "English",
            webapp.OLLAMA_MODEL,
            host="http://ollama-host:11434",
        )
        self.assertEqual(render_card.call_count, 2)
        self.assertEqual(
            [call.kwargs["darkmode"] for call in render_card.call_args_list],
            [False, True],
        )
        self.assertEqual(render_card.call_args.kwargs["book_name"], "The Gentle-Man Book")
        self.assertEqual(
            render_card.call_args.kwargs["example"],
            "We improved the design through careful refactoring.",
        )
        self.assertEqual(render_card.call_args.kwargs["synonyms"], "restructuring")


    def test_discards_card_without_definition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            epub_path = root / "job-id.epub"
            with zipfile.ZipFile(epub_path, "w") as archive:
                archive.writestr(
                    "chapter_001.xhtml",
                    "<p>robust " + "context " * 40 + "</p>",
                )

            with patch.object(
                webapp, "ef_ollama_generate", return_value="word\nrobust"
            ), patch.object(
                webapp, "enrich_word", return_value=("", ["robusto"], "")
            ), patch.object(webapp, "render_card") as render_card:
                result = webapp._generate_bmps_per_chapter(
                    epub_path=epub_path,
                    book_name="The Gentle-Man Book",
                    chapters=[("chapter_001", "chapter_001.xhtml")],
                    source="en",
                    items=1,
                    with_examples=False,
                    bmp_outdir=root / "bmp",
                    enriched_csv_outdir=root / "csv",
                    sd=object(),
                    update_status=lambda **kwargs: None,
                )

        self.assertEqual(result, (0, 1))
        render_card.assert_not_called()


class DownloadTests(unittest.TestCase):
    def test_does_not_create_empty_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "empty"
            source.mkdir()
            output = root / "flashcards.zip"

            count = webapp._zip_dir(source, output)

            self.assertEqual(count, 0)
            self.assertFalse(output.exists())

    def test_downloads_dark_screensaver_archive(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            job_id = "fbe6bb10cae1"
            job_dir = jobs_dir / job_id
            job_dir.mkdir()
            (job_dir / "meta.json").write_text(json.dumps({"status": "done"}))
            (job_dir / "screensaver_dark.zip").write_bytes(b"dark archive")

            with patch.object(webapp, "JOBS_DIR", jobs_dir):
                response = webapp.app.test_client().get(
                    f"/download/{job_id}/screensaver_dark"
                )
                response_data = response.get_data()
                content_disposition = response.headers["Content-Disposition"]
                response.close()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response_data, b"dark archive")
        self.assertIn(
            f"{job_id}_screensaver_dark.zip",
            content_disposition,
        )


if __name__ == "__main__":
    unittest.main()
