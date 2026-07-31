#!/usr/bin/env python3
"""
enrich_flashcards.py

End-to-end pipeline: epub -> per-chapter enriched CSV (word, definition, example, synonyms).

Can also work on existing flashcard CSVs (legacy mode).

Definitions + synonyms come from a local StarDict dictionary (free, instant).
Examples come from a local Ollama model (free, slower) — optional.

Output columns:
    word,definition,example,synonyms

Modes of operation
------------------

1) End-to-end (epub in, enriched CSVs out):
   python enrich_flashcards.py --epub book.epub --source en --target es --with-examples

2) Legacy batch (existing CSVs in, enriched CSVs out):
   python enrich_flashcards.py --dict-dir wikdict-en-es --csv-glob "chapter_*.csv"

3) Single file (either format):
   python enrich_flashcards.py --dict-dir wikdict-en-es --csv in.csv
   python enrich_flashcards.py --epub book.epub --source en --target es --pattern "cap01" --start 1 --end 1
"""

import argparse
import csv
import gzip
import json
import re
import struct
import sys
import urllib.request
import zipfile
from html.parser import HTMLParser
from pathlib import Path


# ============================================================================
# EPUB EXTRACTION
# ============================================================================

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
        return re.sub(r"\n{3,}", "\n\n", "".join(self.parts)).strip()


def html_to_text(html: str) -> str:
    p = TextExtractor()
    p.feed(html)
    return p.get_text()


def list_chapter_files(epub_path: Path, pattern: str | None = None) -> list[tuple[str, str]]:
    """Return (chapter_id, internal_path) sorted by chapter number."""
    with zipfile.ZipFile(epub_path) as zf:
        names = zf.namelist()
        candidates = []
        for n in names:
            if not n.endswith((".xhtml", ".html", ".htm")):
                continue
            base = Path(n).name
            if base.startswith(("cover", "toc", "title", "info", "nota_", "parte",
                                "portadilla", "sinopsis", "titulo", "autor", "cubierta", "dedicatoria")):
                continue
            if pattern and not re.search(pattern, base):
                continue
            candidates.append(n)

    def key(p):
        m = re.search(r"(\d+)", Path(p).name)
        return (int(m.group(1)) if m else 9999, Path(p).name)

    candidates.sort(key=key)
    return [(Path(p).stem, p) for p in candidates]


# ============================================================================
# OLLAMA — vocabulary extraction from chapter text
# ============================================================================

PROMPT_TEMPLATE = """You are an expert linguist identifying C1-C2 level vocabulary worth memorizing in {source_lang}.

Your task: from the text below, generate a POOL of {pool_size} candidate vocabulary items, then SELECT the {num_items} BEST ones that meet strict C1-C2 criteria. Always output exactly {num_items} lines.

All output must be in {source_lang} only — no translations, no other languages.

============================================================
STRICT C1-C2 CRITERIA (you will be graded on this)
============================================================

GOOD items (extract these, but ONLY if they appear in the text below):
- **Single-word advanced vocabulary** (one word per item, no phrases): "leverage", "mitigate", "drawback", "compelling", "blatantly", "robust", "undermine"
- **Single-word abstract nouns**: "constraint", "abstraction", "discrepancy", "nuance", "tenet"
- **Single-word advanced adjectives**: "concise", "verbose", "arbitrary", "incoherent", "sporadic"

BAD items (DO NOT include these):
- Multi-word phrases: "make up for", "figure out", "bear in mind", "deeply rooted" — the dictionary cannot define phrases well. SINGLE WORDS ONLY.
- Basic nouns like: team, code, documentation, feedback, iteration, software, repository, plan, system, method, function
- Basic verbs like: use, make, do, have, work, run, get, set, put, take, give, go, come
- Basic adjectives: good, bad, big, small, new, old, important, useful, easy
- Proper nouns and product names: "Go", "TDD", "LazyVim", "WSL", "TypeScript", "Angular"

============================================================
EXAMPLES OF GOOD OUTPUT FORMAT (these are illustrative; do NOT extract these unless they appear in the text below)
============================================================

{source_lang_cap} vocabulary items (SINGLE WORDS only, no phrases):
- "leverage"
- "mitigate"
- "drawback"
- "robust"
- "abstraction"
- "constraint"

CRITICAL: The examples above show FORMAT only. Extract items that ACTUALLY APPEAR in the text below. Do not invent items just because they look like the examples.

============================================================
OUTPUT FORMAT (strict)
============================================================

Output ONLY this CSV block, nothing else. No commentary, no markdown fences, no explanations, no greetings, no preamble.

word
item in {source_lang}
item in {source_lang}
item in {source_lang}
...

Rules:
- EXACTLY {num_items} data lines (no more, no less)
- Single column only (just the word/phrase in {source_lang})
- NO translations, NO definitions, NO commas inside cells
- Multi-word phrases are preferred over single words
- If a phrase contains a comma, wrap it in double quotes

============================================================
TEXT TO ANALYZE
============================================================

\"\"\"
{chapter_text}
\"\"\"

Now output the {num_items} words/phrases:"""


def ollama_generate(prompt: str, model: str, host: str = "http://localhost:11434") -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4, "num_ctx": 8192},
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8")).get("response", "").strip()


def parse_vocab_single_column(response: str) -> list[str]:
    """Parse a single-column vocabulary response (word/phrase per line)."""
    lines = response.splitlines()
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # strip list bullets / numbering
        line = re.sub(r"^[-*\d.\)\]]+\s*", "", line)
        line = line.strip("`").strip()
        # skip header
        if line.lower() in {"word", "words", "vocabulary", "item", "phrase"}:
            continue
        # try CSV parse (handles quoted phrases with commas)
        try:
            import io
            parsed = list(csv.reader(io.StringIO(line)))
            if parsed and parsed[0]:
                # single column only — take the first field
                word = parsed[0][0].strip()
            else:
                word = line.split(",")[0].strip()
        except Exception:
            word = line.split(",")[0].strip()
        word = word.strip('"').strip("'").strip()
        if not word or len(word) < 2:
            continue
        if word.lower() in seen:
            continue
        seen.add(word.lower())
        out.append(word)
    return out


def filter_words_by_source_text(words: list[str], chapter_text: str) -> list[str]:
    """Drop words/phrases that don't appear in the chapter, and enforce single-word only."""
    if not chapter_text:
        return []
    text_lower = chapter_text.lower()
    text_stripped = re.sub(r"[^\w\s]", " ", text_lower)
    text_stripped = re.sub(r"\s+", " ", text_stripped)
    LEADING_ARTICLES = (
        "the ", "a ", "an ", "el ", "la ", "los ", "las ",
        "un ", "una ", "unos ", "unas ", "to ", "a ",
    )
    kept = []
    for word in words:
        w = word.strip()
        # Enforce single word: drop anything with spaces or hyphens
        if " " in w or "-" in w:
            continue
        w_lower = w.lower()
        if not w_lower or len(w_lower) < 3:
            continue
        if w_lower in text_lower or w_lower in text_stripped:
            kept.append(w)
            continue
        matched = False
        for art in LEADING_ARTICLES:
            if w_lower.startswith(art):
                trimmed = w_lower[len(art):]
                if trimmed in text_lower or trimmed in text_stripped:
                    kept.append(w)
                    matched = True
                    break
        if matched:
            continue
        subwords = re.findall(r"\b\w{4,}\b", w_lower)
        if any(w in text_stripped for w in subwords):
            kept.append(w)
    return kept


def extract_vocabulary_for_chapter(
    chapter_text: str,
    source_lang: str,
    num_items: int,
    model: str,
) -> list[str]:
    """Use Ollama to extract vocabulary, then filter by source-text presence."""
    pool = int(num_items * 2.5)
    prompt = PROMPT_TEMPLATE.format(
        source_lang=source_lang,
        source_lang_cap=source_lang.capitalize(),
        chapter_text=chapter_text[:12000],
        num_items=num_items,
        pool_size=pool,
    )
    response = ollama_generate(prompt, model)
    words = parse_vocab_single_column(response)
    words = filter_words_by_source_text(words, chapter_text)
    return words[:num_items]


# ============================================================================
# STARDICT — definition + synonyms
# ============================================================================

class StarDict:
    def __init__(self, dict_dir: Path):
        self.dict_dir = dict_dir
        self.ifo = self._parse_ifo()
        self.wordcount = int(self.ifo.get("wordcount", 0))
        self.synwordcount = int(self.ifo.get("synwordcount", 0))

        idx_path = self._resolve("idx")
        self._index: list[tuple[bytes, int, int]] = []
        with self._open(idx_path, "rb") as f:
            data = f.read()
        i = 0
        seen = 0
        while i < len(data) and seen < self.wordcount:
            nul = data.index(b"\x00", i)
            word = data[i:nul]
            offset, size = struct.unpack(">II", data[nul + 1 : nul + 9])
            self._index.append((word, offset, size))
            i = nul + 9
            seen += 1

        self._lower_index = {w.lower(): (off, sz) for w, off, sz in self._index}

        self._syn: dict[bytes, list[int]] = {}
        syn_path = self.dict_dir / "stardict.syn"
        if syn_path.exists():
            with syn_path.open("rb") as f:
                data = f.read()
            i = 0
            while i < len(data):
                nul = data.index(b"\x00", i)
                syn = data[i:nul]
                idx = struct.unpack(">I", data[nul + 1 : nul + 5])[0]
                self._syn.setdefault(syn.lower(), []).append(idx)
                i = nul + 5

        self._dict_file = None
        self._dict_path = self._resolve("dict")

    def _resolve(self, ext: str) -> Path:
        for cand in (f"stardict.{ext}",):
            p = self.dict_dir / cand
            if p.exists():
                return p
        if ext == "dict":
            for cand in ("stardict.dz",):
                p = self.dict_dir / cand
                if p.exists():
                    return p
        raise FileNotFoundError(f"stardict.{ext} not found in {self.dict_dir}")

    def _open(self, path: Path, mode: str):
        if path.suffix == ".dz":
            return gzip.open(path, mode)
        return path.open(mode)

    def _parse_ifo(self) -> dict[str, str]:
        out: dict[str, str] = {}
        p = self.dict_dir / "stardict.ifo"
        if not p.exists():
            return out
        for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line and not line.startswith("StarDict"):
                k, _, v = line.partition("=")
                out[k.strip()] = v.strip()
        return out

    def lookup(self, word: str) -> str | None:
        entry = self._lower_index.get(word.lower().encode("utf-8"))
        if not entry:
            return None
        offset, size = entry
        if self._dict_file is None:
            self._dict_file = self._open(self._dict_path, "rb")
        self._dict_file.seek(offset)
        return self._dict_file.read(size).decode("utf-8", errors="ignore")

    def synonyms(self, word: str) -> list[str]:
        out: list[str] = []
        for idx in self._syn.get(word.lower().encode("utf-8"), []):
            if 0 <= idx < len(self._index):
                out.append(self._index[idx][0].decode("utf-8", errors="ignore"))
        return out


def extract_definition_and_translations(html: str) -> tuple[str, list[str], str]:
    """Pull (definition, translations, pronunciation) out of StarDict HTML.

    Two structures appear in wikdict.com StarDict entries:

    Flat structure (most common): definition followed by sibling <div> translations:
        <div>/<font color="gray">IPA1</font>/, .../<br>
        <div><font class="grammar">noun</font></div>english definition
        <div>spanish 1</div>
        <div>spanish 2</div>
        </div>

    Nested structure (used for words with multiple senses):
        <ol>
          <li><div>sense 1 english</div><div>spanish1</div></li>
          <li><div>sense 2 english</div><div>spanish2</div></li>
        </ol>

    Returns:
        (definition, translations, pronunciation_ipa)
    """
    # 1) Extract pronunciation first — capture IPA from <font color="gray">
    # The pronunciation is a single short word like "tekˈnäləjē"
    pronunciations: list[str] = []
    for m in re.finditer(r'<font\s+color="gray"[^>]*>([^<]+)</font>', html):
        ipa = m.group(1).strip()
        if ipa and len(ipa) < 60:
            pronunciations.append(ipa)
    # Some entries have multiple pronunciation variants separated by ", /"
    # Keep all of them — they reflect regional variation
    pronunciation = " / ".join(pronunciations) if pronunciations else ""

    # Wipe pronunciation blocks from the HTML
    cleaned = re.sub(r'<font\s+color="gray"[^>]*>.*?</font>', " ", html, flags=re.DOTALL)
    cleaned = re.sub(r"/[^/<>]{1,80}/", " ", cleaned)
    cleaned = re.sub(r"\s*/\s*", " ", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)

    # 2) Extract grammar tag
    grammar_match = re.search(r'<font\s+class="grammar"[^>]*>([^<]+)</font>', cleaned)
    grammar = grammar_match.group(1).strip() if grammar_match else ""
    if grammar_match:
        cleaned = cleaned.replace(grammar_match.group(0), " ")

    # 3) Find where English definition ends and translations begin.
    # In flat structure: it's the first <div> with no grammar-style contents
    # after the grammar tag. In nested: it's the first <ol>.
    #
    # Simpler heuristic: the english_part is the text between the grammar tag and
    # either the <ol> OR the first <div> whose contents are pure Spanish (heuristic).

    translations: list[str] = []

    # Detect <ol> first
    ol_match = re.search(r"<ol[^>]*>", cleaned, flags=re.IGNORECASE)
    if ol_match:
        english_part = cleaned[: ol_match.start()]
        translations_part = cleaned[ol_match.start():]
        # extract from <li><div>X</div></li>
        li_blocks = re.findall(r"<li[^>]*>(.*?)</li>",
                                translations_part, flags=re.IGNORECASE | re.DOTALL)
        seen: set[str] = set()
        for li in li_blocks:
            divs = re.findall(r"<div>([^<]+)</div>", li)
            for d in divs:
                d = d.strip()
                if d and d not in seen:
                    seen.add(d)
                    translations.append(d)
    else:
        # Flat structure: the english definition and translations are siblings
        # inside the outer <div>. We need to walk the string from the end backwards,
        # extracting "<div>X</div>" pairs right before each closing </div>.
        #
        # Pattern at the end: ...english_def</div><div>es1</div></div>
        # So we look for "<div>X</div></div>" runs from the end, taking X as
        # a Spanish translation (short, no further nesting).
        english_part = cleaned

        # Walk backwards: every time we see "</div>" at the cursor, check
        # if it closes a "<div>X</div>" with X being short text.
        # The trailing siblings are translations.
        leaf_pattern = re.compile(r"<div>([^<>]{1,60}?)</div>", flags=re.IGNORECASE)

        # Identify the trailing translation run by scanning from the end.
        # A "leaf" at the end of the string is: <div>X</div> immediately followed
        # only by </div>s and end-of-string.
        end_pos = len(cleaned.rstrip())
        translations_trailing: list[str] = []
        cursor = end_pos

        # Quick helper: is content between prev_div_end and current match just whitespace+</div>?
        while cursor > 0:
            # find the last "<div>X</div>" before cursor
            region = cleaned[:cursor]
            last = None
            for m in leaf_pattern.finditer(region):
                last = m
            if not last:
                break
            # Check what's between last.end() and cursor
            between = cleaned[last.end():cursor].strip()
            if between and between != "</div>":
                # not contiguous — stop
                break
            # Need the *last* matched leaf that's adjacent — re-find properly:
            # we took the very last leaf in region, that's wrong.
            break

        # Simpler approach: greedy match "<div>([^<>]+)</div>(?:</div>)?" from end
        # We walk backwards collecting the last N leaves where each is followed by </div>
        # or end-of-string.
        translations_trailing = []

        # Find all leaves
        leaves = list(leaf_pattern.finditer(cleaned))
        if leaves:
            # Walk from the end; collect contiguous trailing leaves
            idx = len(leaves) - 1
            tail: list = []
            while idx >= 0:
                leaf = leaves[idx]
                if not tail:
                    # last leaf: must end near end-of-string (after optional </div>)
                    rest = cleaned[leaf.end():].strip()
                    if rest in ("", "</div>"):
                        tail.append(leaf)
                else:
                    # this leaf's end should be at previous tail's start (preceded by </div>)
                    prev = tail[-1]
                    between = cleaned[leaf.end():prev.start()].strip()
                    if between in ("", "</div>"):
                        tail.append(leaf)
                    else:
                        break
                idx -= 1
            tail.reverse()
            seen: set[str] = set()
            for m in tail:
                d = m.group(1).strip()
                if d and d not in seen:
                    seen.add(d)
                    translations_trailing.append(d)

        # If we found a contiguous trailing block, those are translations.
        # Otherwise, fall back to "all leaves after the grammar tag that aren't
        # the english_def leaf".
        if translations_trailing:
            translations = translations_trailing
        elif len(leaves) >= 2:
            # Heuristic: first leaf is usually grammar+ipa, second is english_def,
            # remaining are translations.
            for m in leaves[2:]:
                d = m.group(1).strip()
                if d and d not in translations:
                    translations.append(d)

    # 4) Convert english_part to plain text
    text = re.sub(r"<[^>]+>", " ", english_part)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s+", " ", text).strip()

    # 5) Trim trailing punctuation/whitespace, drop leading grammar-related commas
    text = re.sub(r"^[\s,;:\.\|]+", "", text)
    text = re.sub(r"[\s,;:\.\|]+$", "", text)

    # 6) Use the first sentence / clause as the clean definition
    definition = text
    m = re.search(r"[.;]\s+[A-Z]", text)
    if m:
        first = text[: m.end() - 1].strip()
        if len(first) > 5:
            definition = first

    # If translations are still embedded in the english text (flat structure),
    # try to split on the first occurrence of any translation token
    if translations:
        # nothing to do — translations are already extracted
        pass
    else:
        # Maybe translations are mixed into the text. Keep the whole text as def.
        pass

    translations = [t for t in translations if t]

    if grammar and definition:
        definition = f"{grammar}. {definition}"
    elif grammar and not definition:
        definition = grammar

    return definition, translations, pronunciation


# ============================================================================
# OLLAMA — examples (optional)
# ============================================================================

EXAMPLES_PROMPT = """You are a lexicographer. Generate exactly ONE natural example sentence in {source_lang} for each of these {source_lang} words/phrases.

Each example should:
- Be 8-18 words long
- Use the word naturally in context (the way a native speaker would)
- NOT include translations or explanations
- NOT include quotation marks

Format STRICTLY as a CSV block with no commentary:

word,example
{word1},example sentence 1
{word2},example sentence 2
...

Words/phrases to cover:
{word_list}

CSV output:"""


def ollama_batch_examples(
    words: list[str], source_lang: str, model: str,
    host: str = "http://localhost:11434",
) -> dict[str, str]:
    if not words:
        return {}
    word_list = "\n".join(f"- {w}" for w in words)
    prompt = EXAMPLES_PROMPT.format(
        source_lang=source_lang,
        word_list=word_list,
        word1=words[0],
        word2=words[1] if len(words) > 1 else "",
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.5, "num_ctx": 4096},
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        response = json.loads(resp.read().decode("utf-8")).get("response", "").strip()
    out: dict[str, str] = {}
    for line in response.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("word,"):
            continue
        line = re.sub(r"^[-*\d.\)\]]+\s*", "", line)
        if "," not in line:
            continue
        src, ex = line.split(",", 1)
        src = src.strip().strip('"').strip("'")
        ex = ex.strip().strip('"').strip("'")
        if src and ex:
            out[src.lower()] = ex
    return out


# ============================================================================
# CSV WRITING HELPERS
# ============================================================================

def write_enriched_csv(path: Path, rows: list[dict], append: bool = False) -> None:
    """Write or append enriched rows to a CSV. If append=True and file exists, skip header."""
    mode = "a" if append and path.exists() else "w"
    write_header = mode == "w"
    with path.open(mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        if write_header:
            writer.writerow([
                "word", "pronunciation", "definition", "example", "synonyms", "chapter", "book",
            ])
        for r in rows:
            writer.writerow([
                r["word"],
                r.get("pronunciation", ""),
                r["definition"],
                r["example"],
                r["synonyms"],
                r.get("chapter", ""),
                r.get("book", ""),
            ])


def enrich_word(
    sd: StarDict, word: str, source_word: bool = True
) -> tuple[str, list[str], str]:
    """Return (definition, list_of_synonym/translation candidates, pronunciation_ipa)."""
    html = sd.lookup(word)
    if not html:
        return "", [], ""
    definition, translations, pronunciation = extract_definition_and_translations(html)
    syns = sd.synonyms(word) if source_word else []
    return definition, translations + syns, pronunciation


# ============================================================================
# MAIN PIPELINE
# ============================================================================

LANG_NAMES = {
    "en": "English", "es": "Spanish", "fr": "French",
    "de": "German", "it": "Italian", "pt": "Portuguese",
}


def process_epub(
    epub_path: Path,
    dict_dir: Path,
    source: str,
    items: int, model: str,
    pattern: str | None,
    start: int | None, end: int | None,
    with_examples: bool,
    example_batch_size: int,
    output_dir: Path,
    book_prefix: str | None,
    single_csv: bool,
) -> int:
    source_lang = LANG_NAMES.get(source, source)
    book_prefix = book_prefix or slugify(epub_path.stem.split(" -- ")[0], max_len=20)
    book_name = epub_path.stem.split(" -- ")[0]

    print(f"[+] Opening epub: {epub_path.name}")
    chapters = list_chapter_files(epub_path, pattern)
    print(f"[+] Found {len(chapters)} candidate chapter files")

    if start is not None or end is not None:
        filtered = []
        for stem, internal in chapters:
            m = re.search(r"(\d+)", stem)
            if not m:
                continue
            num = int(m.group(1))
            if start is not None and num < start:
                continue
            if end is not None and num > end:
                continue
            filtered.append((stem, internal))
        chapters = filtered
        print(f"[+] Filtered to chapters {start}-{end}: {len(chapters)} files")

    if not chapters:
        print("[-] No chapters to process.", file=sys.stderr)
        return 1

    sd = StarDict(dict_dir)
    print(f"[+] Loaded StarDict: {sd.wordcount} words, {sd.synwordcount} synonym entries\n")

    # Determine the consolidated CSV path (if single_csv mode)
    consolidated_path: Path | None = None
    if single_csv:
        consolidated_path = output_dir / f"{book_prefix}_all.csv"
        # Wipe previous consolidated CSV
        if consolidated_path.exists():
            consolidated_path.unlink()
        print(f"[+] Writing single consolidated CSV: {consolidated_path}\n")

    with zipfile.ZipFile(epub_path) as zf:
        for stem, internal in chapters:
            print(f"[+] Processing: {stem}")
            text = html_to_text(zf.read(internal).decode("utf-8", errors="ignore"))
            if len(text) < 200:
                print(f"    skipping (too short: {len(text)} chars)")
                continue

            # Extract chapter number for tracking
            num_match = re.search(r"(\d+)", stem)
            num = num_match.group(1) if num_match else stem
            chapter_label = f"cap{num.zfill(3)}"

            # 1. Extract vocabulary via Ollama (single column)
            print(f"    extracting vocabulary via Ollama ({model})...")
            try:
                words = extract_vocabulary_for_chapter(
                    chapter_text=text,
                    source_lang=source_lang,
                    num_items=items,
                    model=model,
                )
            except Exception as e:
                print(f"    ERROR: {e}", file=sys.stderr)
                continue
            if not words:
                print(f"    no vocabulary extracted, skipping")
                continue
            print(f"    extracted {len(words)} candidate words")

            # 2. Enrich with StarDict (definition + synonyms + pronunciation)
            enriched: list[dict] = []
            for word in words:
                definition, suggestions, pronunciation = enrich_word(sd, word)
                seen = set()
                clean_syns = []
                for s in suggestions:
                    s_l = s.lower()
                    if s_l in seen or s_l == word.lower():
                        continue
                    seen.add(s_l)
                    clean_syns.append(s)
                enriched.append({
                    "word": word,
                    "definition": definition,
                    "pronunciation": pronunciation,
                    "example": "",
                    "synonyms": ", ".join(clean_syns[:6]),
                    "chapter": chapter_label,
                    "book": book_name,
                })

            # 3. Optionally generate examples via Ollama (batched)
            if with_examples and enriched:
                print(f"    generating examples via Ollama ({len(enriched)} words)...")
                examples: dict[str, str] = {}
                for i in range(0, len(enriched), example_batch_size):
                    batch = [r["word"] for r in enriched[i:i + example_batch_size]]
                    try:
                        examples.update(ollama_batch_examples(
                            batch, source_lang, model,
                        ))
                    except Exception as e:
                        print(f"    batch {i//example_batch_size + 1} failed: {e}",
                              file=sys.stderr)
                for r in enriched:
                    r["example"] = examples.get(r["word"].lower(), "")

            # 4. Write output
            if single_csv:
                # Append to consolidated CSV (no header on append)
                write_enriched_csv(consolidated_path, enriched,
                                   append=consolidated_path.exists() and consolidated_path.stat().st_size > 0)
                print(f"    appended {len(enriched)} items to {consolidated_path}\n")
            else:
                out_path = output_dir / f"{book_prefix}_cap{num.zfill(3)}.csv"
                write_enriched_csv(out_path, enriched, append=False)
                print(f"    wrote {out_path} ({len(enriched)} items)\n")

    print("[+] Done.")
    return 0


def process_csvs(
    inputs: list[Path],
    dict_dir: Path,
    output_dir: Path,
    source_col: int,
    with_examples: bool, model: str,
    source_lang: str,
    example_batch_size: int,
) -> int:
    sd = StarDict(dict_dir)
    print(f"[+] Loaded StarDict: {sd.wordcount} words, {sd.synwordcount} synonym entries")

    for csv_in in inputs:
        csv_out = output_dir / (csv_in.stem + "_enriched.csv") if output_dir != Path(".") \
                   else csv_in.with_name(csv_in.stem + "_enriched.csv")

        rows_data: list[dict] = []
        with csv_in.open(encoding="utf-8", newline="") as f_in:
            reader = csv.reader(f_in)
            try:
                next(reader)
            except StopIteration:
                continue
            for row in reader:
                if not row or len(row) <= source_col:
                    continue
                src = row[source_col].strip()
                if not src:
                    continue
                definition, suggestions, pronunciation = enrich_word(sd, src)
                seen = set()
                clean_syns = []
                for s in suggestions:
                    s_l = s.lower()
                    if s_l in seen or s_l == src.lower():
                        continue
                    seen.add(s_l)
                    clean_syns.append(s)
                rows_data.append({
                    "word": src,
                    "definition": definition,
                    "pronunciation": pronunciation,
                    "example": "",
                    "synonyms": ", ".join(clean_syns[:6]),
                })

        if with_examples and rows_data:
            examples: dict[str, str] = {}
            for i in range(0, len(rows_data), example_batch_size):
                batch = [r["word"] for r in rows_data[i:i + example_batch_size]]
                try:
                    examples.update(ollama_batch_examples(
                        batch, source_lang, model,
                    ))
                except Exception as e:
                    print(f"    batch failed: {e}", file=sys.stderr)
            for r in rows_data:
                r["example"] = examples.get(r["word"].lower(), "")

        write_enriched_csv(csv_out, rows_data)
        print(f"[+] {csv_in.name} -> {csv_out.name} ({len(rows_data)} items)")

    print("\n[+] Done.")
    return 0


def slugify(s: str, max_len: int = 40) -> str:
    s = re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")
    return s[:max_len] or "chapter"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Enrich vocab with definition, example, synonyms (single-language).",
        epilog="All output is in --source language only — no translations, no other languages.",
    )
    ap.add_argument("--dict-dir", required=True, help="Path to a StarDict directory (e.g. wikdict-en-es)")
    ap.add_argument("--with-examples", action="store_true", help="Generate example sentences using Ollama")
    ap.add_argument("--ollama-model", default="llama3.1")
    ap.add_argument("--example-batch-size", type=int, default=15)
    ap.add_argument("--output-dir", default=".", help="Where to write enriched CSVs")
    ap.add_argument("--source-col", type=int, default=0, help="Column index in legacy CSV with the word")

    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--csv", help="Single input CSV (legacy mode)")
    mode.add_argument("--csv-glob", help="Glob pattern (legacy mode)")
    mode.add_argument("--epub", help="Epub file path (end-to-end mode)")

    # End-to-end (epub) flags
    ap.add_argument("--source", help="Source language code (e.g. en, es) — required with --epub")
    ap.add_argument("--items", type=int, default=30, help="Items per chapter (default 30)")
    ap.add_argument("--pattern", help="Regex to filter chapter filenames")
    ap.add_argument("--start", type=int, help="Start chapter number")
    ap.add_argument("--end", type=int, help="End chapter number")
    ap.add_argument("--book-prefix", help="Filename prefix for output CSVs")
    ap.add_argument("--single-csv", action="store_true",
                    help="Consolidate all chapters into ONE CSV (named <prefix>_all.csv) "
                         "with chapter and book columns added")

    args = ap.parse_args()

    dict_dir = Path(args.dict_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not (dict_dir / "stardict.ifo").exists():
        print(f"[-] {dict_dir} doesn't look like a StarDict directory", file=sys.stderr)
        return 1

    source_lang = LANG_NAMES.get(args.source or "", args.source or "English")

    if args.epub:
        if not args.source:
            print("[-] --epub requires --source", file=sys.stderr)
            return 1
        return process_epub(
            epub_path=Path(args.epub).expanduser().resolve(),
            dict_dir=dict_dir,
            source=args.source,
            items=args.items, model=args.ollama_model,
            pattern=args.pattern, start=args.start, end=args.end,
            with_examples=args.with_examples,
            example_batch_size=args.example_batch_size,
            output_dir=output_dir,
            book_prefix=args.book_prefix,
            single_csv=args.single_csv,
        )

    # Legacy: existing CSVs
    if args.csv:
        inputs = [Path(args.csv).expanduser().resolve()]
    else:
        inputs = sorted(Path(".").glob(args.csv_glob))
        if not inputs:
            print(f"[-] No files matched {args.csv_glob}", file=sys.stderr)
            return 1

    return process_csvs(
        inputs=inputs, dict_dir=dict_dir, output_dir=output_dir,
        source_col=args.source_col,
        with_examples=args.with_examples, model=args.ollama_model,
        source_lang=source_lang,
        example_batch_size=args.example_batch_size,
    )


if __name__ == "__main__":
    sys.exit(main())