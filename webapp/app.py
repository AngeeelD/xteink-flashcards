"""
Flashcard web service.

Upload an epub → get back either or both:
  - flashcards.zip:    per-chapter bilingual CSVs (English|Spanish or vice versa)
  - screensaver.zip:   BMPs ready as 528x792 e-ink lock screens (light variant)
  - screensaver_dark.zip: same BMPs inverted (dark variant)

Each output has its own chapter range, source language and item count. The
screensaver output is rendered from a single enrichment pass so light and
dark share the exact same cards (no duplicate Ollama calls).

Single job at a time (Ollama is a shared resource).
"""

import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image
from flask import (
    Flask,
    abort,
    jsonify,
    render_template,
    request,
    send_file,
    url_for,
)

# Make existing scripts importable
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from epub_to_flashcards import (  # noqa: E402
    ollama_generate as ollama_generate_bi,
    parse_csv_response as parse_bi_csv,
    filter_rows_by_source_text as filter_bi_rows,
    build_prompt as build_bi_prompt,
    html_to_text,
    list_chapter_files,
)
from enrich_flashcards import (  # noqa: E402
    StarDict,
    enrich_word,
    extract_vocabulary_for_chapter,
    ollama_batch_examples_and_synonyms,
    write_enriched_csv,
    LANG_NAMES,
    PROMPT_TEMPLATE as ENRICH_PROMPT,
    parse_vocab_single_column,
    filter_words_by_source_text,
    ollama_generate as ef_ollama_generate,
)
from generate_flashcard_bmps import (  # noqa: E402
    render_card,
    WIDTH_DEFAULT, HEIGHT_DEFAULT,
    WIDTH_X4, HEIGHT_X4,
    find_font,
    list_available_fonts,
    section_titles_for,
)

# ----------------------------------------------------------------------------

APP_VERSION = "0.5.0"

app = Flask(__name__)

WEBAPP_DIR = Path(__file__).resolve().parent
JOBS_DIR = WEBAPP_DIR / "jobs"
UPLOADS_DIR = WEBAPP_DIR / "uploads"
PREVIEWS_DIR = WEBAPP_DIR / "previews"
CUSTOM_FONTS_DIR = WEBAPP_DIR / "fonts"
WIKTDICT_EN_ES = WEBAPP_DIR.parent / "wikdict-en-es"
WIKTDICT_ES_EN = WEBAPP_DIR.parent / "wikdict-es-en"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)
CUSTOM_FONTS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_FONT_EXTS = {".ttf", ".otf", ".ttc"}
MAX_FONT_BYTES = 10 * 1024 * 1024  # 10 MB cap per font file

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")

# Single-job lock (Ollama + CPU constraints on the Mac mini)
_job_lock = threading.Lock()
_active_job_id: str | None = None


# ----------------------------------------------------------------------------
# Status helpers
# ----------------------------------------------------------------------------

def _meta_path(job_id: str) -> Path:
    return JOBS_DIR / job_id / "meta.json"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _book_name_from_upload(filename: str | None) -> str:
    normalized = (filename or "").replace("\\", "/")
    basename = Path(normalized).name
    book_name = Path(basename).stem.split(" -- ")[0].strip()
    return book_name or "book"


def _read_meta(job_id: str) -> dict:
    p = _meta_path(job_id)
    if not p.exists():
        return {"status": "not_found"}
    return json.loads(p.read_text())


def _write_meta(job_id: str, data: dict) -> None:
    p = _meta_path(job_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2))


def _filter_chapters(
    chapters: list[tuple[str, str]],
    start: int | None,
    end: int | None,
) -> list[tuple[str, str]]:
    """Filter chapters by inclusive numeric range."""
    if not start and not end:
        return list(chapters)
    filtered: list[tuple[str, str]] = []
    for stem, internal in chapters:
        m = re.search(r"(\d+)", stem)
        if not m:
            continue
        n = int(m.group(1))
        if start and n < int(start):
            continue
        if end and n > int(end):
            continue
        filtered.append((stem, internal))
    return filtered


def _coerce_bool(value: object) -> bool:
    """Coerce form values to bool. HTML checkboxes send 'on' when checked."""
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).lower() in {"on", "true", "1", "yes"}


def _resolve_form_config(form) -> dict:
    """Build the job config from the form, supporting legacy and per-output keys."""
    # Legacy fields stay as the default; csv_*/bmp_* override per output.
    legacy_source = form.get("source", "en")
    legacy_target = form.get("target", "es")
    legacy_items = int(form.get("items", "30") or 30)
    legacy_start = form.get("start") or None
    legacy_end = form.get("end") or None
    legacy_device = form.get("device", "x3")
    legacy_generate_csv = _coerce_bool(form.get("generate_csv", "on"))
    legacy_generate_bmp = _coerce_bool(form.get("generate_bmp", "on"))
    legacy_dark = _coerce_bool(form.get("darkmode"))
    legacy_with_examples = _coerce_bool(form.get("with_examples"))

    csv_present = any(form.get(k) is not None for k in (
        "csv_enabled", "csv_source", "csv_target", "csv_items",
        "csv_start", "csv_end",
    ))
    bmp_present = any(form.get(k) is not None for k in (
        "bmp_enabled", "bmp_source", "bmp_items",
        "bmp_start", "bmp_end", "bmp_device", "bmp_font",
        "bmp_dark", "bmp_with_examples",
    ))

    csv_enabled = (
        _coerce_bool(form.get("csv_enabled", "on")) if csv_present
        else legacy_generate_csv
    )
    csv_source = form.get("csv_source") or legacy_source
    csv_target = form.get("csv_target") or legacy_target
    csv_items = int(form.get("csv_items") or legacy_items)
    csv_start = form.get("csv_start") or legacy_start
    csv_end = form.get("csv_end") or legacy_end

    bmp_enabled = (
        _coerce_bool(form.get("bmp_enabled", "on")) if bmp_present
        else legacy_generate_bmp
    )
    bmp_source = form.get("bmp_source") or legacy_source
    bmp_items = int(form.get("bmp_items") or legacy_items)
    bmp_start = form.get("bmp_start") or legacy_start
    bmp_end = form.get("bmp_end") or legacy_end
    bmp_device = form.get("bmp_device") or legacy_device
    bmp_dark = (
        _coerce_bool(form.get("bmp_dark")) if bmp_present
        else legacy_dark
    )
    bmp_with_examples = (
        _coerce_bool(form.get("bmp_with_examples")) if bmp_present
        else legacy_with_examples
    )
    bmp_font = (form.get("bmp_font") or "").strip() or None

    return {
        "csv_enabled": csv_enabled,
        "csv_source": csv_source,
        "csv_target": csv_target,
        "csv_items": csv_items,
        "csv_start": csv_start,
        "csv_end": csv_end,
        "bmp_enabled": bmp_enabled,
        "bmp_source": bmp_source,
        "bmp_items": bmp_items,
        "bmp_start": bmp_start,
        "bmp_end": bmp_end,
        "bmp_device": bmp_device,
        "bmp_dark": bmp_dark,
        "bmp_with_examples": bmp_with_examples,
        "bmp_font": bmp_font,
        # Backward-compat aliases used by the legacy test surface.
        "source": legacy_source,
        "target": legacy_target,
        "items": legacy_items,
        "start": legacy_start,
        "end": legacy_end,
        "device": legacy_device,
        "generate_csv": legacy_generate_csv,
        "generate_bmp": legacy_generate_bmp,
        "darkmode": legacy_dark,
        "with_examples": legacy_with_examples,
    }


# ----------------------------------------------------------------------------
# Worker
# ----------------------------------------------------------------------------

def _generate_bilingual_per_chapter(
    epub_path: Path,
    chapters: list[tuple[str, str]],
    source: str,
    target: str,
    items: int,
    csv_outdir: Path,
    update_status,
) -> tuple[int, int]:
    """Per-chapter bilingual flashcard CSVs (English|Spanish or vice versa)."""
    csv_outdir.mkdir(parents=True, exist_ok=True)
    total = len(chapters)
    written = 0
    for i, (stem, internal) in enumerate(chapters, start=1):
        update_status(
            phase="bilingual_csv",
            phase_label=f"Generating bilingual CSV ({i}/{total}: {stem})",
            chapters_done=i - 1,
            chapters_total=total,
        )
        with open(epub_path, "rb") as f:
            import zipfile
            with zipfile.ZipFile(f) as zf:
                html = zf.read(internal).decode("utf-8", errors="ignore")
        text = html_to_text(html)
        if len(text) < 200:
            continue
        source_label = LANG_NAMES.get(source, source.capitalize())
        target_label = LANG_NAMES.get(target, target.capitalize())
        prompt = build_bi_prompt(
            source_lang=LANG_NAMES.get(source, source),
            target_lang=LANG_NAMES.get(target, target),
            source_label=source_label,
            target_label=target_label,
            chapter_text=text,
            num_items=items,
        )
        rows: list[tuple[str, str]] = []
        for attempt in range(2):
            current_prompt = prompt
            if attempt:
                current_prompt += (
                    "\n\nYour previous response was invalid. Start with exactly "
                    f"{source_label},{target_label} and output CSV rows only."
                )
            try:
                response = ollama_generate_bi(
                    current_prompt, OLLAMA_MODEL, host=OLLAMA_HOST
                )
            except Exception as e:
                print(f"[job] ollama failed for {stem}: {e}")
                continue
            rows = parse_bi_csv(
                response,
                source_label=source_label,
                target_label=target_label,
            )
            rows = filter_bi_rows(rows, text)[:items]
            if rows:
                break
        if not rows:
            continue
        num = re.search(r"(\d+)", stem)
        num_s = num.group(1).zfill(3) if num else f"{i:03d}"
        out = csv_outdir / f"chapter_{num_s}.csv"
        with out.open("w", encoding="utf-8") as f:
            f.write(f"{source_label},{target_label}\n")
            for src, tgt in rows:
                def esc(s):
                    if "," in s or "\n" in s or '"' in s:
                        return '"' + s.replace('"', '""') + '"'
                    return s
                f.write(f"{esc(src)},{esc(tgt)}\n")
        written += 1
    return written, total


def _enrich_per_chapter(
    epub_path: Path,
    book_name: str,
    chapters: list[tuple[str, str]],
    source: str,
    items: int,
    with_examples: bool,
    enriched_csv_outdir: Path,
    sd: StarDict,
    update_status,
) -> tuple[int, int]:
    """Extract + enrich vocabulary per chapter and persist enriched CSVs.

    Single Ollama + StarDict pass per chapter. The enriched CSVs are then
    rendered into either the light or dark variant by `_render_bmps_per_chapter`.
    """
    enriched_csv_outdir.mkdir(parents=True, exist_ok=True)
    source_lang = LANG_NAMES.get(source, source)
    total = len(chapters)
    written = 0

    for i, (stem, internal) in enumerate(chapters, start=1):
        update_status(
            phase="enrich",
            phase_label=f"Enriching {stem} ({i}/{total})",
            chapters_done=i - 1,
            chapters_total=total,
        )
        with open(epub_path, "rb") as epub_file:
            import zipfile
            with zipfile.ZipFile(epub_file) as archive:
                html = archive.read(internal).decode("utf-8", errors="ignore")
        text = html_to_text(html)
        if len(text) < 200:
            continue

        pool = int(items * 2.5)
        prompt_enr = ENRICH_PROMPT.format(
            source_lang=source_lang,
            source_lang_cap=source_lang.capitalize(),
            chapter_text=text[:12000],
            num_items=items,
            pool_size=pool,
        )
        try:
            response_enr = ef_ollama_generate(
                prompt_enr, OLLAMA_MODEL, host=OLLAMA_HOST
            )
        except Exception as e:
            print(f"[job] enrich ollama failed for {stem}: {e}")
            continue
        words = parse_vocab_single_column(response_enr)
        words = filter_words_by_source_text(words, text)[:items]

        enriched: list[dict] = []
        for word in words:
            definition, suggestions, pronunciation = enrich_word(sd, word)
            if not definition.strip():
                continue
            seen: set[str] = set()
            clean_syns: list[str] = []
            for suggestion in suggestions:
                key = suggestion.casefold()
                if key in seen or key == word.casefold():
                    continue
                seen.add(key)
                clean_syns.append(suggestion)
            enriched.append({
                "word": word,
                "definition": definition,
                "pronunciation": pronunciation,
                "example": "",
                "synonyms": ", ".join(clean_syns[:6]),
                "chapter": stem,
                "book": book_name,
            })

        if with_examples and enriched:
            update_status(phase_label=f"Examples for {stem} ({i}/{total})")
            try:
                examples, generated_synonyms = ollama_batch_examples_and_synonyms(
                    [row["word"] for row in enriched],
                    source_lang,
                    OLLAMA_MODEL,
                    host=OLLAMA_HOST,
                )
                for row in enriched:
                    key = row["word"].casefold()
                    row["example"] = examples.get(key, "")
                    if generated_synonyms.get(key):
                        row["synonyms"] = generated_synonyms[key]
            except Exception as e:
                print(f"[job] examples failed for {stem}: {e}")

        if not enriched:
            continue

        num = re.search(r"(\d+)", stem)
        num_s = num.group(1).zfill(3) if num else f"{i:03d}"
        enriched_csv_path = enriched_csv_outdir / f"chapter_{num_s}.csv"
        write_enriched_csv(enriched_csv_path, enriched, append=False)
        written += 1

    return written, total


def _render_bmps_per_chapter(
    enriched_csv_outdir: Path,
    bmp_outdir: Path,
    book_name: str,
    darkmode: bool,
    width: int,
    height: int,
    update_status,
    font_path: str | None = None,
    source_lang: str | None = None,
) -> tuple[int, int]:
    """Render light or dark BMPs from previously-enriched CSVs.

    Pure read-side: no Ollama, no StarDict. The caller is responsible for
    having populated `enriched_csv_outdir` via `_enrich_per_chapter`.
    """
    bmp_outdir.mkdir(parents=True, exist_ok=True)
    enriched_paths = sorted(enriched_csv_outdir.glob("chapter_*.csv"))
    if not enriched_paths:
        return 0, 0
    total = len(enriched_paths)
    written = 0
    titles = section_titles_for(source_lang)

    for i, enriched_csv_path in enumerate(enriched_paths, start=1):
        update_status(
            phase="bmp_dark" if darkmode else "bmp",
            phase_label=(
                f"{'Dark' if darkmode else 'Light'} BMPs "
                f"({i}/{total}: {enriched_csv_path.stem})"
            ),
            chapters_done=i - 1,
            chapters_total=total,
        )
        with enriched_csv_path.open(encoding="utf-8", newline="") as csv_file:
            enriched = [
                row
                for row in csv.DictReader(csv_file)
                if (row.get("word") or "").strip()
                and (row.get("definition") or "").strip()
            ]
        if not enriched:
            continue

        stem = enriched_csv_path.stem.replace("chapter_", "")
        for j, row in enumerate(enriched):
            slug = re.sub(r"[^a-z0-9]+", "_", row["word"].lower()).strip("_")[:30]
            bmp_path = bmp_outdir / f"chapter_{stem}_{j:03d}_{slug}.bmp"
            render_card(
                word=row["word"],
                pronunciation=row.get("pronunciation", ""),
                definition=row["definition"],
                example=row.get("example", ""),
                synonyms=row.get("synonyms", ""),
                book_name=row.get("book", book_name),
                page_no=j + 1,
                total_pages=len(enriched),
                output_path=bmp_path,
                darkmode=darkmode,
                width=width,
                height=height,
                font_path=font_path,
                section_titles=titles,
            )
        written += len(enriched)

    return written, total


def _zip_dir(dirpath: Path, outpath: Path) -> int:
    """Zip every file under dirpath (flat). Returns file count."""
    outpath.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in dirpath.rglob("*") if p.is_file())
    if not files:
        if outpath.exists():
            outpath.unlink()
        return 0
    with zipfile.ZipFile(outpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            zf.write(f, arcname=f.relative_to(dirpath))
    return len(files)


def _run_job(job_id: str, config: dict) -> None:
    global _active_job_id
    meta = _read_meta(job_id)
    meta["status"] = "running"
    meta["started_at"] = _now_iso()
    _write_meta(job_id, meta)

    job_dir = JOBS_DIR / job_id
    csv_dir = job_dir / "csv"
    bmp_dir = job_dir / "bmp"
    dark_bmp_dir = job_dir / "bmp_dark"
    enriched_csv_dir = job_dir / "enriched"
    epub_path = UPLOADS_DIR / f"{job_id}.epub"
    book_name = _book_name_from_upload(config.get("original_filename"))

    def update(**kw):
        m = _read_meta(job_id)
        m.update(kw)
        _write_meta(job_id, m)

    try:
        bmp_source = config.get("bmp_source") or config.get("source") or "en"
        if bmp_source == "en":
            dict_dir = WIKTDICT_EN_ES
        elif bmp_source == "es":
            dict_dir = WIKTDICT_ES_EN
        else:
            raise ValueError(f"Unsupported source language: {bmp_source}")
        if not dict_dir.exists():
            raise FileNotFoundError(f"Dictionary not found: {dict_dir}")

        # Detect chapters once; per-output filters are applied below.
        update(phase="detecting", phase_label="Detecting chapters…")
        all_chapters = list_chapter_files(epub_path)
        if not all_chapters:
            raise RuntimeError("No chapters found in this epub")

        csv_chapters = _filter_chapters(
            all_chapters, config.get("csv_start"), config.get("csv_end")
        )
        bmp_chapters = _filter_chapters(
            all_chapters, config.get("bmp_start"), config.get("bmp_end")
        )
        update(chapters_total=len(all_chapters))

        # ---------- Phase A: bilingual CSVs ----------
        if config.get("csv_enabled"):
            csv_written, _ = _generate_bilingual_per_chapter(
                epub_path, csv_chapters,
                source=config.get("csv_source") or config.get("source") or "en",
                target=config.get("csv_target") or config.get("target") or "es",
                items=config.get("csv_items") or config.get("items") or 30,
                csv_outdir=csv_dir,
                update_status=update,
            )
            update(phase="zipping", phase_label=f"Zipping {csv_written} CSV files…")
            zip_path = job_dir / "flashcards.zip"
            n = _zip_dir(csv_dir, zip_path)
            update(
                flashcards_zip=(f"jobs/{job_id}/flashcards.zip" if n else None),
                flashcards_count=n,
            )

        # ---------- Phase B: enriched + BMPs ----------
        if config.get("bmp_enabled"):
            update(phase="loading_dict", phase_label=f"Loading StarDict ({dict_dir.name})…")
            sd = StarDict(dict_dir)
            update(stardict_words=sd.wordcount)

            # Resolution
            device = config.get("bmp_device") or config.get("device") or "x3"
            if device == "x4":
                bmp_w, bmp_h = WIDTH_X4, HEIGHT_X4
            else:
                bmp_w, bmp_h = WIDTH_DEFAULT, HEIGHT_DEFAULT

            # Phase B1: single enrichment pass (regardless of light/dark).
            enriched_written, _ = _enrich_per_chapter(
                epub_path, book_name, bmp_chapters,
                source=bmp_source,
                items=config.get("bmp_items") or config.get("items") or 30,
                with_examples=config.get("bmp_with_examples", False),
                enriched_csv_outdir=enriched_csv_dir,
                sd=sd,
                update_status=update,
            )

            # Phase B2: render exactly one variant.
            darkmode = bool(config.get("bmp_dark") or config.get("darkmode"))
            if darkmode:
                out_dir = dark_bmp_dir
                archive_name = "screensaver_dark.zip"
            else:
                out_dir = bmp_dir
                archive_name = "screensaver.zip"

            render_written, _ = _render_bmps_per_chapter(
                enriched_csv_outdir=enriched_csv_dir,
                bmp_outdir=out_dir,
                book_name=book_name,
                darkmode=darkmode,
                width=bmp_w,
                height=bmp_h,
                update_status=update,
                font_path=config.get("bmp_font"),
                source_lang=bmp_source,
            )
            update(
                phase="zipping",
                phase_label=(
                    f"Zipping {render_written} "
                    f"{'dark' if darkmode else 'light'} BMPs…"
                ),
            )
            zip_path = job_dir / archive_name
            n = _zip_dir(out_dir, zip_path)
            update(
                bmp_count=n,
                bmp_dark_count=n if darkmode else None,
                bmp_light_count=n if not darkmode else None,
                device=device,
                bmp_resolution=f"{bmp_w}x{bmp_h}",
                enriched_chapters=enriched_written,
            )
            meta_field = (
                "screensaver_dark_zip" if darkmode else "screensaver_zip"
            )
            update(**{
                meta_field: (
                    f"jobs/{job_id}/{archive_name}" if n else None
                ),
            })

        # ---------- Done ----------
        meta = _read_meta(job_id)
        meta["status"] = "done"
        meta["finished_at"] = _now_iso()
        meta["phase"] = "done"
        meta["phase_label"] = "Done."
        # Fill the progress bar: the per-chapter updates leave chapters_done
        # at i-1 of the last chapter, so the bar shows 0/N. Snap it to total.
        meta["chapters_done"] = meta.get("chapters_total", meta.get("chapters_done", 0))
        _write_meta(job_id, meta)

    except Exception as e:
        meta = _read_meta(job_id)
        meta["status"] = "error"
        meta["error"] = str(e)
        meta["finished_at"] = _now_iso()
        meta["phase"] = "error"
        meta["phase_label"] = f"Error: {e}"
        _write_meta(job_id, meta)

    finally:
        with _job_lock:
            if _active_job_id == job_id:
                _active_job_id = None


# ----------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------

@app.route("/")
def index():
    active = _read_meta(_active_job_id) if _active_job_id else None
    fonts = _fonts_for_picker()
    default_font = _resolve_default_font()
    return render_template(
        "index.html",
        active_job=active,
        fonts=fonts,
        default_font=default_font,
    )


def _resolve_default_font() -> str | None:
    """Return the filesystem path of the font render_card would use when the
    user picks the '(default)' option. Resolved server-side so the browser
    can log it on page load — useful for deciding which files to bundle in
    the Docker image."""
    try:
        font = find_font("serif", 56, bold=False)
    except RuntimeError:
        return None
    return getattr(font, "path", None) or str(font)


def _fonts_for_picker() -> list[dict[str, str]]:
    """Return the dropdown data: system fonts from generate_flashcard_bmps
    plus any user-uploaded fonts in CUSTOM_FONTS_DIR."""
    entries = list(list_available_fonts())
    seen = {entry["path"] for entry in entries}
    if CUSTOM_FONTS_DIR.exists():
        for pattern in ("*.ttf", "*.otf", "*.ttc"):
            for match in sorted(CUSTOM_FONTS_DIR.glob(pattern)):
                path_str = str(match)
                if path_str in seen:
                    continue
                seen.add(path_str)
                entries.append({
                    "path": path_str,
                    "label": f"{match.name} (custom)",
                    "family": match.stem,
                })
    return entries


def _safe_font_filename(filename: str) -> str:
    """Reduce a user-supplied filename to a safe on-disk form: only ascii
    letters, digits, dots, dashes and underscores are kept; everything
    else collapses to underscores. Prevents path traversal."""
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9._-]", "_", name) or "custom.ttf"


@app.route("/upload", methods=["POST"])
def upload():
    global _active_job_id
    with _job_lock:
        if _active_job_id:
            other = _read_meta(_active_job_id)
            return jsonify({
                "error": "another_job_running",
                "message": f"Job {other.get('id')} is still running. Wait for it to finish.",
                "job_id": _active_job_id,
            }), 409
        if "epub" not in request.files:
            abort(400, "No epub file uploaded")
        epub = request.files["epub"]
        if not epub.filename or not epub.filename.lower().endswith(".epub"):
            abort(400, "Please upload a .epub file")

        job_id = uuid.uuid4().hex[:12]
        _active_job_id = job_id

        # Save upload
        epub.save(UPLOADS_DIR / f"{job_id}.epub")

        config = _resolve_form_config(request.form)
        config["id"] = job_id
        config["original_filename"] = epub.filename
        config["uploaded_at"] = _now_iso()
        config["status"] = "pending"
        config["phase"] = "queued"
        config["phase_label"] = "Queued for processing…"
        _write_meta(job_id, config)

        # Spawn worker
        t = threading.Thread(
            target=_run_job, args=(job_id, dict(config)), daemon=True,
            name=f"job-{job_id}",
        )
        t.start()

        if request.accept_mimetypes.best == "application/json" or \
           request.args.get("format") == "json":
            return jsonify({"job_id": job_id, "redirect": url_for("job", job_id=job_id)})
        return _redirect(url_for("job", job_id=job_id))


@app.route("/job/<job_id>")
def job(job_id):
    meta = _read_meta(job_id)
    if meta.get("status") == "not_found":
        abort(404)
    return render_template("job.html", meta=meta, job_id=job_id)


@app.route("/job/<job_id>/status.json")
def job_status(job_id):
    meta = _read_meta(job_id)
    if meta.get("status") == "not_found":
        return jsonify({"status": "not_found"}), 404
    return jsonify(meta)


@app.route("/download/<job_id>/<kind>")
def download(job_id, kind):
    meta = _read_meta(job_id)
    if meta.get("status") != "done":
        abort(409, "Job not finished yet")
    archives = {
        "flashcards": "flashcards.zip",
        "screensaver": "screensaver.zip",
        "screensaver_dark": "screensaver_dark.zip",
    }
    archive_name = archives.get(kind)
    if archive_name is None:
        return abort(404)
    path = JOBS_DIR / job_id / archive_name
    filename = f"{job_id}_{archive_name}"
    if not path.exists():
        abort(404, "Output not generated (you didn't request that output)")
    return send_file(
        path,
        mimetype="application/zip",
        as_attachment=True,
        download_name=filename,
    )


@app.route("/preview", methods=["POST"])
def preview():
    """Render a sample card and return a cached PNG preview.

    The preview always uses the existing `render_card` so what the user sees
    matches what they will get for any real chapter card.
    """
    payload = request.get_json(silent=True) or {}
    font = (payload.get("font") or "").strip() or None
    darkmode = bool(payload.get("darkmode"))
    with_examples = bool(payload.get("with_examples"))
    source_lang = (payload.get("source_lang") or "").strip() or None
    device = payload.get("device") or "x3"
    if device == "x4":
        width, height = WIDTH_X4, HEIGHT_X4
    else:
        width, height = WIDTH_DEFAULT, HEIGHT_DEFAULT

    word = "Xteink Flashcards"
    pronunciation = "/ˈziː.tɪŋk/"
    definition = (
        "Generate bilingual flashcards + e-ink screensaver BMPs from any EPUB."
    )
    example = (
        "Upload an epub, choose flashcards and/or screensaver, "
        "download zipped BMPs ready for cpr-vcodex."
    )
    synonyms = "flashcard-generator (xtctool.com/flashcard-generator)"

    fingerprint_src = repr(sorted({
        "font": font or "",
        "darkmode": darkmode,
        "with_examples": with_examples,
        "source_lang": source_lang or "",
        "width": width,
        "height": height,
    }.items()))
    fingerprint = hashlib.sha256(fingerprint_src.encode()).hexdigest()[:16]
    cache_path = PREVIEWS_DIR / f"{fingerprint}.png"
    if not cache_path.exists():
        with tempfile.TemporaryDirectory() as tmp:
            bmp_path = Path(tmp) / "card.bmp"
            render_card(
                word=word,
                pronunciation=pronunciation,
                definition=definition,
                example=example if with_examples else "",
                synonyms=synonyms,
                book_name=f"Xteink Flashcards · v{APP_VERSION}",
                page_no=1,
                total_pages=1,
                output_path=bmp_path,
                darkmode=darkmode,
                width=width,
                height=height,
                font_path=font,
                section_titles=section_titles_for(source_lang),
            )
            with Image.open(bmp_path) as img:
                img.save(cache_path, format="PNG")

    return send_file(
        cache_path,
        mimetype="image/png",
        max_age=3600,
    )


@app.route("/upload-font", methods=["POST"])
def upload_font():
    """Accept a user-supplied font file and store it in CUSTOM_FONTS_DIR.

    The uploaded file becomes available in the screensaver font dropdown
    on the next page render. Used when the container or host doesn't ship
    the font the user wants to use for their cards.
    """
    if "font" not in request.files:
        abort(400, "No font file uploaded")
    file = request.files["font"]
    if not file.filename:
        abort(400, "Empty filename")

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_FONT_EXTS:
        abort(400, f"Unsupported font format: {ext}. Use .ttf, .otf or .ttc.")

    # Validate size before saving.
    file.stream.seek(0, os.SEEK_END)
    size = file.stream.tell()
    file.stream.seek(0)
    if size > MAX_FONT_BYTES:
        abort(400, f"Font file is too large ({size} bytes; max {MAX_FONT_BYTES}).")

    safe_name = _safe_font_filename(file.filename)
    target = CUSTOM_FONTS_DIR / safe_name
    file.save(target)

    return jsonify({
        "path": str(target),
        "label": f"{target.name} (custom)",
        "family": target.stem,
    })


@app.route("/health")
def health():
    return jsonify({"status": "ok", "active_job": _active_job_id})


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def _redirect(url):
    from flask import redirect
    return redirect(url)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("WEBAPP_PORT", "5000")), debug=False)
