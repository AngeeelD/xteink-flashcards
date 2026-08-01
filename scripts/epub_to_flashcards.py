#!/usr/bin/env python3
"""
epub_to_flashcards.py

Pipeline to convert an epub book into per-chapter CSV flashcard files
for language learning (C1-C2 level).

Uses a local LLM via Ollama (default: llama3.1:8b) to extract vocabulary,
so no API tokens are spent.

Usage:
    python epub_to_flashcards.py --epub book.epub --source en --target es
    python epub_to_flashcards.py --epub book.epub --source es --target en --model qwen2.5:7b
    python epub_to_flashcards.py --epub book.epub --source en --target es --start 18 --end 18
"""

import argparse
import re
import sys
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote

import urllib.request
import json


# ----------------------------- HTML extraction -----------------------------

class TextExtractor(HTMLParser):
    """Strip HTML, keep only visible text."""

    SKIP_TAGS = {"script", "style", "head", "meta", "link"}

    def __init__(self):
        super().__init__()
        self.parts = []
        self.skip = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP_TAGS:
            self.skip = True

    def handle_endtag(self, tag):
        if tag in self.SKIP_TAGS:
            self.skip = False
        if tag in {"p", "br", "li", "h1", "h2", "h3", "h4", "blockquote", "div"}:
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.skip:
            text = data.strip()
            if text:
                self.parts.append(text + " ")

    def get_text(self) -> str:
        text = "".join(self.parts)
        # collapse multiple blank lines
        return re.sub(r"\n{3,}", "\n\n", text).strip()


def html_to_text(html: str) -> str:
    parser = TextExtractor()
    parser.feed(html)
    return parser.get_text()


# ----------------------------- Ollama client -----------------------------

def ollama_generate(prompt: str, model: str, host: str = "http://localhost:11434") -> str:
    """Send a single prompt to a local Ollama model and return its response."""
    url = f"{host}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.4,  # some creativity, but mostly deterministic
            "num_ctx": 8192,
        },
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "").strip()


# ----------------------------- Prompt template ----------------------------

PROMPT_TEMPLATE = """You are an expert linguist creating C1-C2 level flashcards for a {source_lang} speaker learning {target_lang}.

Your task: from the text below, generate a POOL of {pool_size} candidate vocabulary items, then SELECT the {num_items} BEST ones that meet strict C1-C2 criteria. Always output exactly {num_items} lines.

============================================================
STRICT C1-C2 CRITERIA (you will be graded on this)
============================================================

GOOD items (extract these, but ONLY if they appear in the text below):
- Phrasal verbs / multi-word verb phrases: "make up for", "figure out", "carry out", "put up with"
- Idiomatic expressions (cannot be understood literally): "by far", "at stake", "bear in mind", "by and large"
- Advanced vocabulary NOT in basic word lists: "leverage" (as verb), "mitigate", "drawback", "compelling", "blatantly"
- Strong collocations: "deeply rooted", "heavily rely on", "blatantly obvious", "in the long run"
- Subjunctive patterns / advanced grammar worth noting: "had it not been for", "no sooner...than", "were it not for"

BAD items (DO NOT include these):
- Basic nouns like: team, code, documentation, feedback, iteration, software, repository, plan, system, method, function
- Basic verbs like: use, make, do, have, work, run, get, set, put, take, give, go, come
- Basic adjectives: good, bad, big, small, new, old, important, useful, easy
- Single technical jargon that's just a noun (e.g., "WSL", "HomeBrew", "LazyVim" — these are proper nouns or product names, NOT vocabulary)
- Words shorter than 4 letters unless they are critical idioms (e.g., "at stake" is fine, but "run" alone is not)
- Literal translations of common phrases (e.g., "commit code" -> "cometer código" is wrong; the real phrase is "hacer commit de código")

============================================================
EXAMPLES OF GOOD OUTPUT FORMAT (these are illustrative; do NOT extract these unless they appear in the text below)
============================================================

{source_lang_cap} source, {target_lang_cap} target:
- "make up for" -> "compensar"
- "figure out" -> "averiguar / resolver"
- "bear in mind" -> "tener en cuenta"
- "deeply rooted" -> "profundamente arraigado"
- "by and large" -> "en general / por lo general"
- "drawback" -> "desventaja / inconveniente"
- "leverage" -> "aprovechar / sacar partido a"

CRITICAL: The examples above show FORMAT only. Extract items that ACTUALLY APPEAR in the text below. Do not invent items just because they look like the examples.

============================================================
OUTPUT FORMAT (strict)
============================================================

Output ONLY this CSV block, nothing else. No commentary, no markdown fences, no explanations, no greetings, no preamble.

{source_label},{target_label}
item in {source_lang},translation in {target_lang}
item in {source_lang},translation in {target_lang}
...

Rules:
- EXACTLY {num_items} data lines (no more, no less)
- Each line has exactly ONE comma separating source from translation
- Use " / " to separate alternative translations
- Source language must be 100% in {source_lang}; target 100% in {target_lang}
- Multi-word phrases are preferred over single words
- If a phrase contains a comma, wrap the whole field in double quotes

============================================================
TEXT TO ANALYZE
============================================================

\"\"\"
{chapter_text}
\"\"\"

Now output the {num_items} CSV lines:"""


def build_prompt(source_lang: str, target_lang: str, source_label: str,
                 target_label: str, chapter_text: str, num_items: int) -> str:
    # Ask for 2.5x more than we want, so the model has a larger pool to choose from
    pool_size = int(num_items * 2.5)
    return PROMPT_TEMPLATE.format(
        source_lang=source_lang,
        target_lang=target_lang,
        source_lang_cap=source_lang.capitalize(),
        target_lang_cap=target_lang.capitalize(),
        source_label=source_label,
        target_label=target_label,
        chapter_text=chapter_text[:12000],  # larger context window for richer analysis
        num_items=num_items,
        pool_size=pool_size,
    )


# ----------------------------- CSV parsing -----------------------------

def parse_csv_response(
    response: str,
    source_label: str,
    target_label: str | None = None,
) -> list[tuple[str, str]]:
    """Parse only a model response containing the requested CSV header."""
    import csv
    import io

    lines = response.splitlines()

    def parse_line(raw: str) -> list[str] | None:
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("```"):
            return None
        line = re.sub(r"^[-*\d.\)\]]+\s*", "", line)
        try:
            fields = next(csv.reader(io.StringIO(line)))
        except (csv.Error, StopIteration):
            return None
        return [field.strip() for field in fields]

    header_index = None
    for index, raw in enumerate(lines):
        fields = parse_line(raw)
        if not fields or len(fields) != 2:
            continue
        source_matches = fields[0].casefold() == source_label.casefold()
        target_matches = target_label is None or fields[1].casefold() == target_label.casefold()
        if source_matches and target_matches:
            header_index = index
            break

    if header_index is None:
        return []

    rows: list[tuple[str, str]] = []
    seen: set[str] = set()
    for raw in lines[header_index + 1:]:
        if raw.strip().startswith("```"):
            if rows:
                break
            continue
        fields = parse_line(raw)
        if not fields or len(fields) != 2:
            continue
        src, tgt = fields
        if not src or not tgt or len(src) > 160 or len(tgt) > 500:
            continue
        key = src.casefold()
        if key in seen:
            continue
        seen.add(key)
        rows.append((src, tgt))
    return rows


def filter_rows_by_source_text(
    rows: list[tuple[str, str]], chapter_text: str
) -> list[tuple[str, str]]:
    """Keep rows whose complete source phrase appears in the chapter text."""
    if not chapter_text:
        return []

    def normalize(value: str) -> str:
        value = re.sub(r"[^\w\s]", " ", value.casefold())
        return re.sub(r"\s+", " ", value).strip()

    normalized_text = normalize(chapter_text)
    leading_articles = (
        "the ", "a ", "an ",
        "el ", "la ", "los ", "las ",
        "un ", "una ", "unos ", "unas ",
        "to ",
    )

    def appears(candidate: str) -> bool:
        normalized = normalize(candidate)
        if not normalized:
            return False
        pattern = rf"(?<!\w){re.escape(normalized)}(?!\w)"
        return re.search(pattern, normalized_text) is not None

    kept: list[tuple[str, str]] = []
    for src, tgt in rows:
        source = src.strip()
        if appears(source):
            kept.append((source, tgt))
            continue
        normalized_source = normalize(source)
        for article in leading_articles:
            if normalized_source.startswith(article) and appears(normalized_source[len(article):]):
                kept.append((source, tgt))
                break
    return kept


# ----------------------------- Epub chapter extraction -----------------------------

def list_chapter_files(epub_path: Path, pattern: str | None = None) -> list[tuple[str, str]]:
    """Return list of (chapter_id, content_path) sorted by chapter number."""
    with zipfile.ZipFile(epub_path) as zf:
        names = zf.namelist()
        # look for xhtml/html files inside OEBPS
        candidates = []
        for n in names:
            if not n.endswith((".xhtml", ".html", ".htm")):
                continue
            base = Path(n).name
            if base.startswith("cover") or base.startswith("toc") or base.startswith("title") or base.startswith("info"):
                continue
            if pattern:
                if not re.search(pattern, base):
                    continue
            candidates.append(n)

    def chapter_key(path: str) -> tuple[int, str]:
        base = Path(path).name
        m = re.search(r"(\d+)", base)
        if m:
            return (int(m.group(1)), base)
        return (9999, base)

    candidates.sort(key=chapter_key)
    return [(Path(p).stem, p) for p in candidates]


def read_chapter(zf: zipfile.ZipFile, internal_path: str) -> str:
    html = zf.read(internal_path).decode("utf-8", errors="ignore")
    return html_to_text(html)


# ----------------------------- Main pipeline -----------------------------

def slugify(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower())
    s = s.strip("_")
    return s[:max_len] or "chapter"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate C1-C2 flashcards from an epub using a local Ollama model.")
    ap.add_argument("--epub", required=True, help="Path to the epub file")
    ap.add_argument("--source", required=True, help="Source language code (en, es, fr, ...)")
    ap.add_argument("--target", required=True, help="Target language code (en, es, fr, ...)")
    ap.add_argument("--source-label", default=None, help="Column 1 label (defaults to language name)")
    ap.add_argument("--target-label", default=None, help="Column 2 label (defaults to language name)")
    ap.add_argument("--model", default="llama3.1", help="Ollama model name (default: llama3.1)")
    ap.add_argument("--items", type=int, default=40, help="Items per chapter (default: 40)")
    ap.add_argument("--output-dir", default=".", help="Where to write CSVs (default: current dir)")
    ap.add_argument("--start", type=int, default=None, help="Start chapter number (inclusive)")
    ap.add_argument("--end", type=int, default=None, help="End chapter number (inclusive)")
    ap.add_argument("--pattern", default=None, help="Regex pattern for chapter filenames")
    ap.add_argument("--book-prefix", default=None, help="Filename prefix for CSVs (default: book stem)")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be done, don't call model")
    args = ap.parse_args()

    LANG_NAMES = {
        "en": "English", "es": "Spanish", "fr": "French",
        "de": "German", "it": "Italian", "pt": "Portuguese",
    }
    source_label = args.source_label or LANG_NAMES.get(args.source, args.source.capitalize())
    target_label = args.target_label or LANG_NAMES.get(args.target, args.target.capitalize())

    epub_path = Path(args.epub).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    book_prefix = args.book_prefix or slugify(epub_path.stem.split(" -- ")[0], max_len=20)

    print(f"[+] Opening epub: {epub_path}")
    chapters = list_chapter_files(epub_path, args.pattern)
    print(f"[+] Found {len(chapters)} candidate chapter files")

    if not chapters:
        print("[-] No chapters found. Try a different --pattern.", file=sys.stderr)
        return 1

    # filter by chapter number range
    if args.start is not None or args.end is not None:
        filtered = []
        for stem, internal in chapters:
            m = re.search(r"(\d+)", stem)
            if not m:
                continue
            num = int(m.group(1))
            if args.start is not None and num < args.start:
                continue
            if args.end is not None and num > args.end:
                continue
            filtered.append((stem, internal))
        chapters = filtered
        print(f"[+] Filtered to chapters {args.start}-{args.end}: {len(chapters)} files")

    if args.dry_run:
        for stem, internal in chapters:
            print(f"  would process {stem} ({internal})")
        return 0

    with zipfile.ZipFile(epub_path) as zf:
        for stem, internal in chapters:
            print(f"\n[+] Processing: {stem}")
            text = read_chapter(zf, internal)
            if len(text) < 200:
                print(f"    skipping (too short: {len(text)} chars)")
                continue
            prompt = build_prompt(
                source_lang=LANG_NAMES.get(args.source, args.source),
                target_lang=LANG_NAMES.get(args.target, args.target),
                source_label=source_label,
                target_label=target_label,
                chapter_text=text,
                num_items=args.items,
            )
            print(f"    calling ollama ({args.model})...")
            try:
                response = ollama_generate(prompt, args.model)
            except Exception as e:
                print(f"    ERROR calling ollama: {e}", file=sys.stderr)
                continue
            rows = parse_csv_response(response, source_label, target_label)
            if not rows:
                print(f"    no rows parsed; raw response was:")
                print(response[:500])
                continue

            # post-filter: drop rows whose source phrase isn't actually in the chapter
            original_count = len(rows)
            rows = filter_rows_by_source_text(rows, text)
            dropped = original_count - len(rows)
            if dropped:
                print(f"    post-filter dropped {dropped} hallucinated items")

            # filename
            num_match = re.search(r"(\d+)", stem)
            num = num_match.group(1) if num_match else stem
            out_path = output_dir / f"{book_prefix}_cap{num.zfill(3)}.csv"

            with out_path.open("w", encoding="utf-8") as f:
                f.write(f"{source_label},{target_label}\n")
                for src, tgt in rows[: args.items]:
                    # csv-safe: quote if contains comma or newline
                    def esc(s):
                        if "," in s or "\n" in s or '"' in s:
                            return '"' + s.replace('"', '""') + '"'
                        return s
                    f.write(f"{esc(src)},{esc(tgt)}\n")

            written = min(len(rows), args.items)
            print(f"    wrote {out_path} ({written} items)")

    print("\n[+] Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())