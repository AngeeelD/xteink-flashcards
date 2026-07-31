# Xteink Flashcards — Epub → Bilingual CSVs + E-ink Screen Saver BMPs

A local web service (Flask + Ollama + StarDict) that converts an epub into:

1. **`flashcards.zip`** — per-chapter bilingual CSV files (`English,Spanish` or `Spanish,English`) ready to drop on a Xteink X3 e-reader for vocabulary study.
2. **`screensaver.zip`** — per-chapter BMPs sized `528×792` portrait, 1-bit monochrome, styled as dictionary entries with word, pronunciation (IPA), definition, usage, synonyms. Designed as lock-screen / sleep images for the e-reader.

Built to run on a Mac mini (or any Linux/macOS host) with Ollama installed locally and the StarDict dictionaries cloned into this directory.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│  Browser (http://macmini.local:5000)                    │
└──────────────────┬───────────────────────────────────────┘
                   │ multipart upload
┌──────────────────▼───────────────────────────────────────┐
│  Docker container (Flask + gunicorn)                     │
│  - webapp/app.py                                         │
│  - imports scripts/epub_to_flashcards.py                │
│  - imports scripts/enrich_flashcards.py                  │
│  - imports scripts/generate_flashcard_bmps.py            │
└──────┬─────────────────────────────┬─────────────────────┘
       │ localhost:11434             │ /wikdict-en-es
       ▼                             ▼
┌──────────────────┐      ┌──────────────────────────────┐
│  Ollama (host)   │      │  StarDict files (.ifo/.idx    │
│  llama3.1:8b     │      │  /.dict/.syn) baked into     │
└──────────────────┘      │  image via COPY              │
                         └──────────────────────────────┘
```

---

## Prerequisites

- Docker (Desktop or OrbStack)
- Ollama installed and running on the host (`ollama serve`)
- A pulled Ollama model: `ollama pull llama3.1:8b`
- The two StarDict dictionaries in this directory:
  - `wikdict-en-es/` (English → Spanish)
  - `wikdict-es-en/` (Spanish → English)

---

## Run locally (development, no Docker)

```bash
# 1. install python deps
pip install -r requirements.txt

# 2. start
cd webapp
python app.py

# browse http://localhost:5000
```

The app reads Ollama from `localhost:11434` by default. Override with:

```bash
OLLAMA_HOST=http://localhost:11434 OLLAMA_MODEL=llama3.1 python app.py
```

---

## Run via Docker (recommended for the Mac mini)

```bash
# build image (StarDict files are COPYed into it)
docker compose build

# start the service in the background, exposed on host port 5000
docker compose up -d

# follow logs
docker compose logs -f webapp

# stop
docker compose down
```

Then browse to:

```
http://<macmini-name-or-ip>:5000
```

### Two outputs, your choice

The form lets you check one, both, or neither of:

- `flashcards.zip` — bilingual flashcards (English|Spanish)
- `screensaver.zip` — BMPs ready to drop on the Xteink as images

Both can be generated in the same job if you check both boxes. You can also
restrict to a chapter range (`Start chapter` / `End chapter`) for quick tests.

---

## Project layout

```
.
├── scripts/                    # helper scripts (also importable as modules)
│   ├── epub_to_flashcards.py   # bilingual extraction pipeline
│   ├── enrich_flashcards.py    # StarDict + Ollama enrichment
│   └── generate_flashcard_bmps.py  # 528×792 BMP renderer
├── webapp/
│   ├── app.py                  # Flask app
│   ├── templates/              # base.html, index.html, job.html
│   ├── static/style.css        # a few CSS tweaks on top of PicoCSS
│   ├── jobs/                   # runtime: per-job output + meta.json (gitignored)
│   └── uploads/                # runtime: original epubs (gitignored)
├── wikdict-en-es/              # StarDict dictionary (English → Spanish)
├── wikdict-es-en/              # StarDict dictionary (Spanish → English)
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── requirements.txt
├── .gitignore
├── README.md
└── AGENTS.md                   # workspace notes
```

---

## How long does a job take?

Roughly 30–90 seconds per chapter for the bilingual CSVs (one Ollama call per
chapter). For BMP generation, +30 seconds per chapter if you tick "include
examples". A 24-chapter book with both outputs and examples ≈ 30 minutes.

The web UI polls `/job/<id>/status.json` every 2.5 seconds and shows current
phase + per-chapter progress.

---

## Notes

- One job at a time. If you upload while another is running, the second upload
  gets a 409 with the active job id — wait for it (or kill the container).
- All processing runs inside the container; nothing is uploaded to the cloud.
- The StarDict files are small (~5 MB each) so we COPY them into the image
  instead of mounting — no need for bind mounts that drift out of sync.
