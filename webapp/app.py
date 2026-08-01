"""
Flashcard web service.

Upload an epub → get back either or both:
  - flashcards.zip:    per-chapter bilingual CSVs (English|Spanish or vice versa)
  - screensaver.zip:   BMPs ready as 528x792 e-ink lock screens

Single job at a time (Ollama is a shared resource).
"""

import csv
import io
import json
import os
import re
import sys
import threading
import uuid
import zipfile
from datetime import datetime, timezone
from pathlib import Path

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
)

# ----------------------------------------------------------------------------

app = Flask(__name__)

WEBAPP_DIR = Path(__file__).resolve().parent
JOBS_DIR = WEBAPP_DIR / "jobs"
UPLOADS_DIR = WEBAPP_DIR / "uploads"
WIKTDICT_EN_ES = WEBAPP_DIR.parent / "wikdict-en-es"
WIKTDICT_ES_EN = WEBAPP_DIR.parent / "wikdict-es-en"

JOBS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

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


def _generate_bmps_per_chapter(
    epub_path: Path,
    book_name: str,
    chapters: list[tuple[str, str]],
    source: str,
    items: int,
    with_examples: bool,
    bmp_outdir: Path,
    enriched_csv_outdir: Path,
    sd: StarDict,
    update_status,
    darkmode: bool = False,
    width: int = WIDTH_DEFAULT,
    height: int = HEIGHT_DEFAULT,
) -> tuple[int, int]:
    """Generate one card set and render it consistently in light or dark mode."""
    bmp_outdir.mkdir(parents=True, exist_ok=True)
    enriched_csv_outdir.mkdir(parents=True, exist_ok=True)
    source_lang = LANG_NAMES.get(source, source)
    total = len(chapters)
    written = 0

    for i, (stem, internal) in enumerate(chapters, start=1):
        update_status(
            phase="bmp_dark" if darkmode else "bmp",
            phase_label=(
                f"{'Dark mode' if darkmode else 'Light mode'} BMPs ({i}/{total}: {stem})"
            ),
            chapters_done=i - 1,
            chapters_total=total,
        )
        num = re.search(r"(\d+)", stem)
        num_s = num.group(1).zfill(3) if num else f"{i:03d}"
        enriched_csv_path = enriched_csv_outdir / f"chapter_{num_s}.csv"

        if darkmode:
            if not enriched_csv_path.exists():
                continue
            with enriched_csv_path.open(encoding="utf-8", newline="") as csv_file:
                enriched = [
                    row
                    for row in csv.DictReader(csv_file)
                    if (row.get("word") or "").strip()
                    and (row.get("definition") or "").strip()
                ]
        else:
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

        for j, row in enumerate(enriched):
            slug = re.sub(r"[^a-z0-9]+", "_", row["word"].lower()).strip("_")[:30]
            bmp_path = bmp_outdir / f"chapter_{num_s}_{j:03d}_{slug}.bmp"
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
            )

        if not darkmode:
            write_enriched_csv(enriched_csv_path, enriched, append=False)
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
    enriched_csv_dir = job_dir / "enriched"
    epub_path = UPLOADS_DIR / f"{job_id}.epub"
    book_name = _book_name_from_upload(config.get("original_filename"))

    def update(**kw):
        m = _read_meta(job_id)
        m.update(kw)
        _write_meta(job_id, m)

    try:
        # Pick dictionary
        source = config["source"]
        if source == "en":
            dict_dir = WIKTDICT_EN_ES
        elif source == "es":
            dict_dir = WIKTDICT_ES_EN
        else:
            raise ValueError(f"Unsupported source language: {source}")
        if not dict_dir.exists():
            raise FileNotFoundError(f"Dictionary not found: {dict_dir}")

        # Detect chapters
        update(phase="detecting", phase_label="Detecting chapters…")
        chapters = list_chapter_files(epub_path)
        if config.get("start") or config.get("end"):
            filtered = []
            for stem, internal in chapters:
                m = re.search(r"(\d+)", stem)
                if not m:
                    continue
                n = int(m.group(1))
                if config.get("start") and n < int(config["start"]):
                    continue
                if config.get("end") and n > int(config["end"]):
                    continue
                filtered.append((stem, internal))
            chapters = filtered
        update(chapters_total=len(chapters))

        if not chapters:
            raise RuntimeError("No chapters found in this epub")

        # ---------- Phase A: bilingual CSVs ----------
        if config.get("generate_csv"):
            csv_written, _ = _generate_bilingual_per_chapter(
                epub_path, chapters,
                source=source,
                target=config["target"],
                items=config["items"],
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
        if config.get("generate_bmp"):
            update(phase="loading_dict", phase_label=f"Loading StarDict ({dict_dir.name})…")
            sd = StarDict(dict_dir)
            update(stardict_words=sd.wordcount)

            # Resolve output resolution for the chosen device
            device = config.get("device", "x3")
            if device == "x4":
                bmp_w, bmp_h = WIDTH_X4, HEIGHT_X4
            else:
                bmp_w, bmp_h = WIDTH_DEFAULT, HEIGHT_DEFAULT

            # Pass B1: light-mode BMPs (always)
            bmp_written, _ = _generate_bmps_per_chapter(
                epub_path, book_name, chapters,
                source=source,
                items=config["items"],
                with_examples=config.get("with_examples", False),
                bmp_outdir=bmp_dir,
                enriched_csv_outdir=enriched_csv_dir,
                sd=sd,
                update_status=update,
                darkmode=False,
                width=bmp_w,
                height=bmp_h,
            )
            update(phase="zipping", phase_label=f"Zipping {bmp_written} light BMPs…")
            zip_path = job_dir / "screensaver.zip"
            n = _zip_dir(bmp_dir, zip_path)
            update(
                screensaver_zip=(f"jobs/{job_id}/screensaver.zip" if n else None),
                bmp_count=n,
                device=device,
                bmp_resolution=f"{bmp_w}x{bmp_h}",
            )

            # Pass B2: dark-mode BMPs (only if requested).
            if config.get("darkmode"):
                dark_bmp_dir = job_dir / "bmp_dark"
                bmp_dark_written, _ = _generate_bmps_per_chapter(
                    epub_path, book_name, chapters,
                    source=source,
                    items=config["items"],
                    with_examples=config.get("with_examples", False),
                    bmp_outdir=dark_bmp_dir,
                    enriched_csv_outdir=enriched_csv_dir,
                    sd=sd,
                    update_status=update,
                    darkmode=True,
                    width=bmp_w,
                    height=bmp_h,
                )
                update(phase="zipping", phase_label=f"Zipping {bmp_dark_written} dark BMPs…")
                dark_zip_path = job_dir / "screensaver_dark.zip"
                nd = _zip_dir(dark_bmp_dir, dark_zip_path)
                update(
                    screensaver_dark_zip=(
                        f"jobs/{job_id}/screensaver_dark.zip" if nd else None
                    ),
                    bmp_dark_count=nd,
                )

        # ---------- Done ----------
        meta = _read_meta(job_id)
        meta["status"] = "done"
        meta["finished_at"] = _now_iso()
        meta["phase"] = "done"
        meta["phase_label"] = "Done."
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
    return render_template("index.html", active_job=active)


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

        # Save config from form
        config = {
            "id": job_id,
            "source": request.form.get("source", "en"),
            "target": request.form.get("target", "es"),
            "items": int(request.form.get("items", "30")),
            "start": request.form.get("start") or None,
            "end": request.form.get("end") or None,
            "device": request.form.get("device", "x3"),
            "generate_csv": request.form.get("generate_csv") == "on",
            "generate_bmp": request.form.get("generate_bmp") == "on",
            "darkmode": request.form.get("darkmode") == "on",
            "with_examples": request.form.get("with_examples") == "on",
            "original_filename": epub.filename,
            "uploaded_at": _now_iso(),
            "status": "pending",
            "phase": "queued",
            "phase_label": "Queued for processing…",
        }
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
