import io
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


class FormConfigTests(unittest.TestCase):
    def test_legacy_form_keys_map_to_per_output_defaults(self):
        from werkzeug.datastructures import ImmutableMultiDict

        form = ImmutableMultiDict({
            "source": "en",
            "target": "es",
            "items": "30",
            "device": "x3",
            "generate_csv": "on",
            "generate_bmp": "on",
            "darkmode": "on",
            "with_examples": "on",
        })
        config = webapp._resolve_form_config(form)

        self.assertTrue(config["csv_enabled"])
        self.assertTrue(config["bmp_enabled"])
        self.assertTrue(config["bmp_dark"])
        self.assertEqual(config["csv_source"], "en")
        self.assertEqual(config["bmp_source"], "en")
        self.assertEqual(config["bmp_items"], 30)

    def test_per_output_form_keys_override_legacy(self):
        from werkzeug.datastructures import ImmutableMultiDict

        form = ImmutableMultiDict({
            # legacy defaults — should be ignored when per-output keys present
            "source": "en",
            "target": "es",
            "items": "30",
            "device": "x3",
            "generate_csv": "on",
            "generate_bmp": "on",
            "darkmode": "on",
            "with_examples": "on",
            # per-output overrides
            "csv_source": "es",
            "csv_target": "en",
            "csv_items": "20",
            "csv_enabled": "on",
            "bmp_source": "en",
            "bmp_items": "12",
            "bmp_device": "x4",
            "bmp_dark": "",  # explicitly off
            "bmp_enabled": "on",
            "bmp_with_examples": "on",
        })
        config = webapp._resolve_form_config(form)

        self.assertEqual(config["csv_source"], "es")
        self.assertEqual(config["csv_target"], "en")
        self.assertEqual(config["csv_items"], 20)
        self.assertEqual(config["bmp_source"], "en")
        self.assertEqual(config["bmp_items"], 12)
        self.assertEqual(config["bmp_device"], "x4")
        self.assertFalse(config["bmp_dark"])
        self.assertTrue(config["bmp_with_examples"])


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
    def _make_epub(self, root: Path, chapter_count: int = 1) -> Path:
        epub_path = root / "job-id.epub"
        with zipfile.ZipFile(epub_path, "w") as archive:
            for n in range(1, chapter_count + 1):
                archive.writestr(
                    f"chapter_{n:03d}.xhtml",
                    "<p>refactoring " + "context " * 40 + "</p>",
                )
        return epub_path

    def test_enrichment_runs_once_for_light_and_dark_renders(self):
        """The new architecture runs Ollama once via _enrich_per_chapter
        and reuses the resulting CSVs for both light and dark renders."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            epub_path = self._make_epub(root)
            enriched_dir = root / "enriched"
            sd = object()

            with patch.object(webapp, "OLLAMA_HOST", "http://ollama-host:11434"), patch.object(
                webapp, "ef_ollama_generate", return_value="word\nrefactoring"
            ) as generate, patch.object(
                webapp,
                "enrich_word",
                return_value=("noun. process in which code is refactored", [], ""),
            ), patch.object(
                webapp,
                "ollama_batch_examples_and_synonyms",
                return_value=(
                    {"refactoring": "We improved the design through careful refactoring."},
                    {"refactoring": "restructuring"},
                ),
            ) as examples, patch.object(
                webapp, "render_card", side_effect=lambda **kwargs: kwargs["output_path"].write_bytes(b"bmp")
            ) as render_card:
                enrich_result = webapp._enrich_per_chapter(
                    epub_path=epub_path,
                    book_name="The Gentle-Man Book",
                    chapters=[("chapter_001", "chapter_001.xhtml")],
                    source="en",
                    items=1,
                    with_examples=True,
                    enriched_csv_outdir=enriched_dir,
                    sd=sd,
                    update_status=lambda **kwargs: None,
                )
                light_result = webapp._render_bmps_per_chapter(
                    enriched_csv_outdir=enriched_dir,
                    bmp_outdir=root / "bmp",
                    book_name="The Gentle-Man Book",
                    darkmode=False,
                    width=webapp.WIDTH_DEFAULT,
                    height=webapp.HEIGHT_DEFAULT,
                    update_status=lambda **kwargs: None,
                )
                dark_result = webapp._render_bmps_per_chapter(
                    enriched_csv_outdir=enriched_dir,
                    bmp_outdir=root / "bmp_dark",
                    book_name="The Gentle-Man Book",
                    darkmode=True,
                    width=webapp.WIDTH_DEFAULT,
                    height=webapp.HEIGHT_DEFAULT,
                    update_status=lambda **kwargs: None,
                )
                light_archive_count = webapp._zip_dir(root / "bmp", root / "light.zip")
                dark_archive_count = webapp._zip_dir(root / "bmp_dark", root / "dark.zip")

        self.assertEqual(enrich_result, (1, 1))
        self.assertEqual(light_result, (1, 1))
        self.assertEqual(dark_result, (1, 1))
        self.assertEqual(light_archive_count, dark_archive_count)
        # Ollama (vocab extraction + examples) only fires once, in enrichment.
        generate.assert_called_once()
        examples.assert_called_once_with(
            ["refactoring"],
            "English",
            webapp.OLLAMA_MODEL,
            host="http://ollama-host:11434",
        )
        # render_card fires twice: once for light, once for dark, both reading
        # the same enriched CSV.
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

    def test_dark_render_without_enrichment_produces_no_bmps(self):
        """Rendering alone (without prior enrichment) must not invent data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            empty_enriched = root / "enriched"
            empty_enriched.mkdir()

            result = webapp._render_bmps_per_chapter(
                enriched_csv_outdir=empty_enriched,
                bmp_outdir=root / "bmp",
                book_name="Book",
                darkmode=True,
                width=webapp.WIDTH_DEFAULT,
                height=webapp.HEIGHT_DEFAULT,
                update_status=lambda **kwargs: None,
            )

        self.assertEqual(result, (0, 0))

    def test_enrichment_discards_card_without_definition(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            epub_path = self._make_epub(root)
            enriched_dir = root / "enriched"

            with patch.object(
                webapp, "ef_ollama_generate", return_value="word\nrobust"
            ), patch.object(
                webapp, "enrich_word", return_value=("", ["robusto"], "")
            ):
                result = webapp._enrich_per_chapter(
                    epub_path=epub_path,
                    book_name="Book",
                    chapters=[("chapter_001", "chapter_001.xhtml")],
                    source="en",
                    items=1,
                    with_examples=False,
                    enriched_csv_outdir=enriched_dir,
                    sd=object(),
                    update_status=lambda **kwargs: None,
                )

        self.assertEqual(result, (0, 1))
        self.assertFalse(any(enriched_dir.glob("chapter_*.csv")))

    def test_dedupe_words_drops_case_insensitive_duplicates(self):
        self.assertEqual(
            webapp._dedupe_words(["Bear", "bear", "BEAR", "cat"]),
            ["Bear", "cat"],
        )

    def test_dedupe_words_preserves_first_seen_order(self):
        self.assertEqual(
            webapp._dedupe_words(["zebra", "Apple", "apple", "banana"]),
            ["zebra", "Apple", "banana"],
        )

    def test_dedupe_words_handles_empty_list(self):
        self.assertEqual(webapp._dedupe_words([]), [])

    def test_enrichment_dedupes_repeated_words_from_ollama(self):
        """If Ollama returns the same word twice (or different cases),
        the enriched CSV contains it only once — otherwise we'd render
        duplicate BMPs for the same word."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            # Custom EPUB that mentions the words we're going to feed.
            epub_path = root / "job-id.epub"
            with zipfile.ZipFile(epub_path, "w") as archive:
                archive.writestr(
                    "chapter_001.xhtml",
                    "<p>refactoring " + "context " * 40 + "</p>",
                )
            enriched_dir = root / "enriched"

            with patch.object(
                webapp, "ef_ollama_generate",
                # Same word twice (different case) plus another word.
                return_value="word\nrefactoring\nREFACTORING",
            ), patch.object(
                webapp, "enrich_word",
                return_value=("noun. process in which code is refactored", [], ""),
            ):
                webapp._enrich_per_chapter(
                    epub_path=epub_path,
                    book_name="Book",
                    chapters=[("chapter_001", "chapter_001.xhtml")],
                    source="en",
                    items=5,
                    with_examples=False,
                    enriched_csv_outdir=enriched_dir,
                    sd=object(),
                    update_status=lambda **kwargs: None,
                )

            csv_text = (enriched_dir / "chapter_001.csv").read_text()
            # The second occurrence should not appear; only the first-seen
            # spelling remains.
            self.assertEqual(csv_text.count("refactoring"), 1)
            self.assertNotIn("REFACTORING", csv_text)


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

    def test_dark_only_mode_skips_light_archive(self):
        """End-to-end: darkmode True produces screensaver_dark.zip only."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            upload_dir = Path(tmpdir) / "uploads"
            upload_dir.mkdir()
            with patch.object(webapp, "JOBS_DIR", jobs_dir), \
                 patch.object(webapp, "UPLOADS_DIR", upload_dir), \
                 patch.object(
                     webapp, "ef_ollama_generate", return_value="word\nrefactoring"
                 ), patch.object(
                     webapp, "enrich_word",
                     return_value=("noun. process in which code is refactored", [], ""),
                 ), patch.object(
                     webapp, "render_card",
                     side_effect=lambda **kwargs: kwargs["output_path"].write_bytes(b"bmp"),
                 ):
                job_id = "deadbeef"
                epub_path = upload_dir / f"{job_id}.epub"
                with zipfile.ZipFile(epub_path, "w") as archive:
                    archive.writestr(
                        "chapter_001.xhtml",
                        "<p>refactoring " + "context " * 40 + "</p>",
                    )

                config = {
                    "csv_enabled": False,
                    "bmp_enabled": True,
                    "bmp_source": "en",
                    "bmp_items": 1,
                    "bmp_device": "x3",
                    "bmp_dark": True,
                    "bmp_with_examples": False,
                    "device": "x3",
                    "original_filename": "book.epub",
                }
                webapp._run_job(job_id, dict(config))

                meta = webapp._read_meta(job_id)

        self.assertEqual(meta["status"], "done")
        self.assertIsNone(meta.get("screensaver_zip"))
        self.assertIsNotNone(meta.get("screensaver_dark_zip"))
        self.assertIsNone(meta.get("flashcards_zip"))
        self.assertGreater(meta["bmp_dark_count"], 0)
        self.assertIsNone(meta.get("bmp_light_count"))

    def test_independent_chapter_ranges_per_output(self):
        """Different csv/bmp start-end produce separate output bundles."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            upload_dir = Path(tmpdir) / "uploads"
            upload_dir.mkdir()
            with patch.object(webapp, "JOBS_DIR", jobs_dir), \
                 patch.object(webapp, "UPLOADS_DIR", upload_dir):
                job_id = "feedface"
                epub_path = upload_dir / f"{job_id}.epub"
                with zipfile.ZipFile(epub_path, "w") as archive:
                    for n in (1, 2, 3):
                        archive.writestr(
                            f"chapter_{n:03d}.xhtml",
                            "<p>refactoring " + "context " * 40 + "</p>",
                        )

                config = {
                    "csv_enabled": True,
                    "csv_source": "en",
                    "csv_target": "es",
                    "csv_items": 1,
                    "csv_start": "1",
                    "csv_end": "1",
                    "bmp_enabled": True,
                    "bmp_source": "en",
                    "bmp_items": 1,
                    "bmp_start": "2",
                    "bmp_end": "3",
                    "bmp_device": "x3",
                    "bmp_dark": False,
                    "bmp_with_examples": False,
                    "device": "x3",
                    "original_filename": "book.epub",
                }
                with patch.object(
                    webapp, "ollama_generate_bi",
                    return_value="English,Spanish\nrefactoring,refactorización",
                ), patch.object(
                    webapp, "ef_ollama_generate", return_value="word\nrefactoring"
                ), patch.object(
                    webapp, "enrich_word",
                    return_value=("noun. process in which code is refactored", [], ""),
                ), patch.object(
                    webapp, "render_card",
                    side_effect=lambda **kwargs: kwargs["output_path"].write_bytes(b"bmp"),
                ):
                    webapp._run_job(job_id, dict(config))

                meta = webapp._read_meta(job_id)

        self.assertEqual(meta["status"], "done")
        # CSV is filtered to chapter 001 only.
        self.assertEqual(meta["flashcards_count"], 1)
        # BMPs run on chapter 002-003.
        self.assertGreaterEqual(meta["bmp_light_count"], 2)

    def test_done_state_fills_chapters_done_counter(self):
        """When a job finishes, chapters_done is snapped to chapters_total
        so the progress bar on the job page ends at 100%."""
        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            upload_dir = Path(tmpdir) / "uploads"
            upload_dir.mkdir()
            with patch.object(webapp, "JOBS_DIR", jobs_dir), \
                 patch.object(webapp, "UPLOADS_DIR", upload_dir), \
                 patch.object(
                     webapp, "ef_ollama_generate", return_value="word\nrefactoring"
                 ), patch.object(
                     webapp, "enrich_word",
                     return_value=("noun. process in which code is refactored", [], ""),
                 ), patch.object(
                     webapp, "render_card",
                     side_effect=lambda **kwargs: kwargs["output_path"].write_bytes(b"bmp"),
                 ):
                job_id = "abcdef0123"
                epub_path = upload_dir / f"{job_id}.epub"
                with zipfile.ZipFile(epub_path, "w") as archive:
                    for n in (1, 2, 3):
                        archive.writestr(
                            f"chapter_{n:03d}.xhtml",
                            "<p>refactoring " + "context " * 40 + "</p>",
                        )

                config = {
                    "csv_enabled": False,
                    "bmp_enabled": True,
                    "bmp_source": "en",
                    "bmp_items": 1,
                    "bmp_device": "x3",
                    "bmp_dark": False,
                    "bmp_with_examples": False,
                    "device": "x3",
                    "original_filename": "book.epub",
                }
                webapp._run_job(job_id, dict(config))
                meta = webapp._read_meta(job_id)

        self.assertEqual(meta["status"], "done")
        self.assertEqual(meta["chapters_total"], 3)
        self.assertEqual(meta["chapters_done"], meta["chapters_total"])

    def test_render_uses_book_level_page_counter(self):
        """When a job has 3 chapters with 1 card each, the per-card footer
        counter reflects position in the whole book (1/3, 2/3, 3/3) rather
        than per-chapter (1/1, 1/1, 1/1)."""
        captured: list[dict] = []

        def record_render(**kwargs):
            captured.append({
                "page_no": kwargs.get("page_no"),
                "total_pages": kwargs.get("total_pages"),
            })
            kwargs["output_path"].write_bytes(b"bmp")

        with tempfile.TemporaryDirectory() as tmpdir:
            jobs_dir = Path(tmpdir)
            upload_dir = Path(tmpdir) / "uploads"
            upload_dir.mkdir()
            with patch.object(webapp, "JOBS_DIR", jobs_dir), \
                 patch.object(webapp, "UPLOADS_DIR", upload_dir), \
                 patch.object(
                     webapp, "ef_ollama_generate", return_value="word\nrefactoring"
                 ), patch.object(
                     webapp, "enrich_word",
                     return_value=("noun. process in which code is refactored", [], ""),
                 ), patch.object(
                     webapp, "render_card", side_effect=record_render
                 ):
                job_id = "bookcount1"
                epub_path = upload_dir / f"{job_id}.epub"
                with zipfile.ZipFile(epub_path, "w") as archive:
                    for n in (1, 2, 3):
                        archive.writestr(
                            f"chapter_{n:03d}.xhtml",
                            "<p>refactoring " + "context " * 40 + "</p>",
                        )

                config = {
                    "csv_enabled": False,
                    "bmp_enabled": True,
                    "bmp_source": "en",
                    "bmp_items": 1,
                    "bmp_device": "x3",
                    "bmp_dark": False,
                    "bmp_with_examples": False,
                    "device": "x3",
                    "original_filename": "book.epub",
                }
                webapp._run_job(job_id, dict(config))

        self.assertEqual(len(captured), 3)
        # Three chapters, one card each → counters should be 1/3, 2/3, 3/3.
        self.assertEqual([c["page_no"] for c in captured], [1, 2, 3])
        self.assertTrue(all(c["total_pages"] == 3 for c in captured))


def _stub_bmp_render(**kwargs):
    """Replace render_card with a function that writes a tiny placeholder BMP."""
    from PIL import Image as _PILImage
    output_path = kwargs["output_path"]
    width = kwargs.get("width", 528)
    height = kwargs.get("height", 792)
    bg = 255 if not kwargs.get("darkmode") else 0
    img = _PILImage.new("1", (width, height), bg)
    img.save(output_path, format="BMP")


class DefaultFontTests(unittest.TestCase):
    def test_resolve_default_font_returns_a_path(self):
        """The resolved default font path is non-empty on hosts with at
        least one curated candidate installed."""
        path = webapp._resolve_default_font()
        self.assertTrue(path, "expected a font path on the test host")
        self.assertIsInstance(path, str)

    def test_resolve_default_font_handles_missing_fonts(self):
        """When find_font raises, the helper returns None instead of
        propagating the RuntimeError to the index route."""
        with patch.object(webapp, "find_font",
                          side_effect=RuntimeError("no fonts")):
            self.assertIsNone(webapp._resolve_default_font())

    def test_index_route_exposes_default_font_to_the_template(self):
        """The rendered index page contains a JS constant with the path of
        the font the curated default will use."""
        with patch.object(webapp, "_resolve_default_font",
                          return_value="/System/Library/Fonts/Palatino.ttc"), \
             patch.object(webapp, "list_available_fonts", return_value=[]), \
             patch.object(webapp, "_active_job_id", None):
            response = webapp.app.test_client().get("/")
            html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("/System/Library/Fonts/Palatino.ttc", html)


class CustomFontTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._fonts_dir = Path(self._tmpdir.name) / "fonts"
        self._fonts_dir.mkdir()
        self._patch = patch.object(webapp, "CUSTOM_FONTS_DIR", self._fonts_dir)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmpdir.cleanup()

    def test_upload_font_accepts_valid_file(self):
        client = webapp.app.test_client()
        # Minimal TTF header so the size check passes; we don't validate
        # that Pillow can actually load it (we're testing the route, not
        # the renderer).
        data = {
            "font": (io.BytesIO(b"\x00\x01\x00\x00" + b"x" * 100), "CustomSans.ttf"),
        }
        response = client.post("/upload-font", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertTrue(payload["path"].endswith("CustomSans.ttf"))
        self.assertEqual(payload["family"], "CustomSans")
        self.assertTrue((self._fonts_dir / "CustomSans.ttf").exists())

    def test_upload_font_rejects_unsupported_extension(self):
        client = webapp.app.test_client()
        data = {
            "font": (io.BytesIO(b"\x00\x01"), "evil.exe"),
        }
        response = client.post("/upload-font", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertFalse((self._fonts_dir / "evil.exe").exists())

    def test_upload_font_rejects_oversized_file(self):
        client = webapp.app.test_client()
        # MAX_FONT_BYTES + 1 bytes
        oversize = b"\x00" * (webapp.MAX_FONT_BYTES + 1)
        data = {
            "font": (io.BytesIO(oversize), "huge.ttf"),
        }
        response = client.post("/upload-font", data=data, content_type="multipart/form-data")
        self.assertEqual(response.status_code, 400)
        self.assertFalse((self._fonts_dir / "huge.ttf").exists())

    def test_upload_font_sanitizes_filename(self):
        client = webapp.app.test_client()
        data = {
            "font": (io.BytesIO(b"\x00\x01\x00\x00"), "../../etc/passwd.ttf"),
        }
        response = client.post("/upload-font", data=data, content_type="multipart/form-data")
        # Should sanitize the filename rather than let it traverse.
        self.assertEqual(response.status_code, 200)
        # The saved file should be inside the fonts dir, not at /etc/passwd.
        files_in_dir = list(self._fonts_dir.iterdir())
        self.assertEqual(len(files_in_dir), 1)
        self.assertTrue(files_in_dir[0].name.endswith(".ttf"))
        self.assertNotIn("..", files_in_dir[0].name)

    def test_fonts_for_picker_includes_uploaded_fonts(self):
        # Drop a font file into the custom fonts dir directly.
        self._fonts_dir.joinpath("MyCustom.otf").write_bytes(b"x")
        entries = webapp._fonts_for_picker()
        custom = [e for e in entries if "MyCustom" in e["label"]]
        self.assertEqual(len(custom), 1)
        self.assertTrue(custom[0]["label"].endswith("(custom)"))

    def test_index_route_exposes_uploaded_fonts_in_dropdown(self):
        self._fonts_dir.joinpath("UserPick.otf").write_bytes(b"x")
        with patch.object(webapp, "_active_job_id", None), \
             patch.object(webapp, "_resolve_default_font", return_value=None):
            response = webapp.app.test_client().get("/")
            html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("UserPick.otf (custom)", html)

    def test_uploaded_font_persists_across_multiple_requests(self):
        """Once a font is uploaded it must remain in the picker across
        subsequent requests — this is the in-container persistence case
        that the user expects ('upload, come back weeks later, font is
        still there' while the container is up)."""
        client = webapp.app.test_client()
        data = {
            "font": (io.BytesIO(b"\x00\x01\x00\x00" + b"x" * 100), "PersistSans.ttf"),
        }
        response = client.post("/upload-font", data=data,
                               content_type="multipart/form-data")
        response.close()
        self.assertEqual(response.status_code, 200)

        # Subsequent requests (simulating different sessions / reloads)
        # should still list the uploaded font.
        for _ in range(3):
            with patch.object(webapp, "_active_job_id", None), \
                 patch.object(webapp, "_resolve_default_font", return_value=None):
                resp = client.get("/")
                html = resp.get_data(as_text=True)
                resp.close()
            self.assertEqual(resp.status_code, 200)
            self.assertIn("PersistSans.ttf (custom)", html)

        # And the file is still on disk.
        self.assertTrue((self._fonts_dir / "PersistSans.ttf").exists())


class RecentJobsTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._jobs_dir = Path(self._tmpdir.name) / "jobs"
        self._jobs_dir.mkdir()
        self._patch = patch.object(webapp, "JOBS_DIR", self._jobs_dir)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmpdir.cleanup()

    def _make_job(self, job_id, *, finished_at, status="done",
                  bmp_subdir="bmp", flash_count=3, bmp_count=3,
                  original_filename=None):
        from PIL import Image as _PILImage
        job_dir = self._jobs_dir / job_id
        job_dir.mkdir()
        meta = {
            "id": job_id,
            "status": status,
            "finished_at": finished_at,
            "original_filename": original_filename or f"{job_id}.epub",
            "flashcards_count": flash_count,
            "bmp_count": bmp_count,
            "bmp_dark_count": None,
            "bmp_light_count": bmp_count if bmp_subdir == "bmp" else None,
        }
        (job_dir / "meta.json").write_text(json.dumps(meta))
        if bmp_subdir:
            bmp_dir = job_dir / bmp_subdir
            bmp_dir.mkdir(exist_ok=True)
            # Write a real (tiny) BMP so Pillow can identify it.
            img = _PILImage.new("1", (8, 8), 255)
            img.save(bmp_dir / "chapter_001_000_first.bmp", format="BMP")
        return job_dir

    def test_recent_jobs_returns_done_jobs_sorted_descending(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        self._make_job("older", finished_at=(now - timedelta(days=5)).isoformat())
        self._make_job("newer", finished_at=(now - timedelta(days=1)).isoformat())
        self._make_job("middle", finished_at=(now - timedelta(days=3)).isoformat())

        jobs = webapp._recent_jobs(limit=10)

        self.assertEqual([j["id"] for j in jobs], ["newer", "middle", "older"])

    def test_recent_jobs_skips_running_and_errored_jobs(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        self._make_job("done-job",
                       finished_at=(now - timedelta(hours=2)).isoformat(),
                       status="done")
        self._make_job("running-job",
                       finished_at="",
                       status="running")
        self._make_job("errored-job",
                       finished_at=(now - timedelta(hours=1)).isoformat(),
                       status="error")

        jobs = webapp._recent_jobs(limit=10)

        self.assertEqual([j["id"] for j in jobs], ["done-job"])

    def test_recent_jobs_finds_thumbnail_in_bmp_dir(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        self._make_job("with-thumb",
                       finished_at=(now - timedelta(hours=1)).isoformat(),
                       bmp_subdir="bmp")

        jobs = webapp._recent_jobs(limit=10)

        self.assertEqual(jobs[0]["thumbnail"].split("/")[-1], "chapter_001_000_first.bmp")

    def test_recent_jobs_falls_back_to_dark_bmp_when_light_missing(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        self._make_job("dark-only",
                       finished_at=(now - timedelta(hours=1)).isoformat(),
                       bmp_subdir="bmp_dark")

        jobs = webapp._recent_jobs(limit=10)

        self.assertIn("bmp_dark", jobs[0]["thumbnail"])

    def test_relative_time_filter(self):
        from datetime import datetime, timezone, timedelta
        app = webapp.app
        # Verify the Jinja filter is registered.
        self.assertIn("relative_time", app.jinja_env.filters)
        filter_fn = app.jinja_env.filters["relative_time"]
        now = datetime.now(timezone.utc)
        self.assertEqual(
            filter_fn((now - timedelta(days=3)).isoformat()), "3d ago",
        )
        self.assertEqual(
            filter_fn((now - timedelta(hours=4)).isoformat()), "4h ago",
        )
        self.assertEqual(
            filter_fn((now - timedelta(minutes=12)).isoformat()), "12m ago",
        )
        self.assertEqual(filter_fn(now.isoformat()), "just now")
        self.assertEqual(filter_fn(""), "")

    def test_thumbnail_endpoint_serves_png(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        self._make_job("with-bmps",
                       finished_at=(now - timedelta(hours=1)).isoformat())

        with webapp.app.test_client() as client:
            response = client.get("/job/with-bmps/thumbnail.png")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.mimetype, "image/png")
        self.assertTrue(response.get_data().startswith(b"\x89PNG"))

    def test_thumbnail_endpoint_returns_404_when_no_bmps(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        # Make a job dir with meta.json but no BMPs.
        job_dir = self._jobs_dir / "no-bmps"
        job_dir.mkdir()
        (job_dir / "meta.json").write_text(json.dumps({
            "id": "no-bmps",
            "status": "done",
            "finished_at": now.isoformat(),
        }))

        with webapp.app.test_client() as client:
            response = client.get("/job/no-bmps/thumbnail.png")

        self.assertEqual(response.status_code, 404)

    def test_thumbnail_endpoint_returns_404_for_missing_job(self):
        with webapp.app.test_client() as client:
            response = client.get("/job/does-not-exist/thumbnail.png")
        self.assertEqual(response.status_code, 404)

    def test_index_route_renders_recent_jobs_carousel(self):
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone.utc)
        self._make_job("alpha",
                       finished_at=(now - timedelta(hours=1)).isoformat(),
                       original_filename="alpha-book.epub")

        with patch.object(webapp, "_active_job_id", None), \
             patch.object(webapp, "_resolve_default_font", return_value=None), \
             patch.object(webapp, "list_available_fonts", return_value=[]):
            response = webapp.app.test_client().get("/")
            html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn("Recent jobs", html)
        self.assertIn("alpha-book.epub", html)
        self.assertIn("/job/alpha/thumbnail.png", html)

    def test_index_route_makes_active_job_clickable(self):
        with patch.object(webapp, "_active_job_id", "abc123"), \
             patch.object(webapp, "_read_meta",
                          return_value={"id": "abc123", "status": "running",
                                        "phase_label": "Detecting chapters...",
                                        "original_filename": "x.epub"}), \
             patch.object(webapp, "_recent_jobs", return_value=[]), \
             patch.object(webapp, "_resolve_default_font", return_value=None), \
             patch.object(webapp, "list_available_fonts", return_value=[]):
            response = webapp.app.test_client().get("/")
            html = response.get_data(as_text=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn('href="/job/abc123"', html)
        self.assertIn("class=\"job-link\"", html)


class PreviewEndpointTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self._preview_dir = Path(self._tmpdir.name)
        self._patch = patch.object(webapp, "PREVIEWS_DIR", self._preview_dir)
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._tmpdir.cleanup()

    def test_preview_returns_png_and_caches_result(self):
        client = webapp.app.test_client()

        payload = {"darkmode": False, "with_examples": True, "device": "x3"}
        first = client.post("/preview", json=payload)
        first_bytes = first.get_data()
        first.close()
        self.assertEqual(first.status_code, 200)
        self.assertEqual(first.mimetype, "image/png")
        self.assertTrue(first_bytes.startswith(b"\x89PNG"))

        # Second request with the same fingerprint returns the cached PNG.
        cache_files = list(webapp.PREVIEWS_DIR.glob("*.png"))
        self.assertEqual(len(cache_files), 1)
        second = client.post("/preview", json=payload)
        second_bytes = second.get_data()
        second.close()
        self.assertEqual(second_bytes, first_bytes)

    def test_preview_returns_dark_variant(self):
        client = webapp.app.test_client()
        light = client.post("/preview", json={"darkmode": False, "device": "x3"})
        light_bytes = light.get_data()
        light.close()
        dark = client.post("/preview", json={"darkmode": True, "device": "x3"})
        dark_bytes = dark.get_data()
        dark.close()
        self.assertEqual(light.status_code, 200)
        self.assertEqual(dark.status_code, 200)
        self.assertNotEqual(light_bytes, dark_bytes)
        self.assertEqual(len(list(webapp.PREVIEWS_DIR.glob("*.png"))), 2)

    def test_preview_source_lang_drives_section_titles(self):
        """Different source_lang payloads produce different fingerprints."""
        client = webapp.app.test_client()
        en = client.post(
            "/preview", json={"source_lang": "en", "darkmode": False, "device": "x3"}
        )
        en_bytes = en.get_data()
        en.close()
        es = client.post(
            "/preview", json={"source_lang": "es", "darkmode": False, "device": "x3"}
        )
        es_bytes = es.get_data()
        es.close()
        self.assertEqual(en.status_code, 200)
        self.assertEqual(es.status_code, 200)
        self.assertNotEqual(en_bytes, es_bytes)
        self.assertEqual(len(list(webapp.PREVIEWS_DIR.glob("*.png"))), 2)

    def test_preview_records_app_version_in_footer(self):
        """render_card receives the APP_VERSION in book_name for the footer."""
        client = webapp.app.test_client()
        with patch.object(
            webapp, "render_card", side_effect=_stub_bmp_render
        ) as render_card:
            response = client.post(
                "/preview", json={"darkmode": False, "device": "x3"}
            )
            response.close()
        self.assertEqual(response.status_code, 200)
        render_card.assert_called_once()
        book_name = render_card.call_args.kwargs["book_name"]
        self.assertIn(webapp.APP_VERSION, book_name)

    def test_preview_passes_font_path_through(self):
        client = webapp.app.test_client()
        with patch.object(
            webapp, "render_card", side_effect=_stub_bmp_render
        ) as render_card:
            response = client.post(
                "/preview",
                json={"darkmode": False, "device": "x3", "font": "/tmp/foo.ttf"},
            )
            response.close()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            render_card.call_args.kwargs["font_path"], "/tmp/foo.ttf"
        )


if __name__ == "__main__":
    unittest.main()
