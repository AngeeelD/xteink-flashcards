#!/usr/bin/env python3
"""
generate_flashcard_bmps.py

Read an enriched flashcard CSV and produce one stylized 528x792 BMP per row.
Designed as a lockscreen / screensaver for a Xteink X3 e-ink reader.

Layout style: classic dictionary entry.
- Word: large serif, with subtle underline + drop shadow
- Section titles: small uppercase sans
- Body text: regular sans
- Synonyms: italic (dictionary-style)
- Rounded inner border (card-like)
- Footer: source attribution + "n / total" counter

Output: 528 x 792 portrait, 1-bit monochrome (e-ink optimized), BMP format.
"""

import argparse
import csv
import re
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


# ----------------------------- Config -----------------------------

# Device presets. Override via --width/--height in the CLI.
WIDTH_DEFAULT = 528   # Xteink X3
HEIGHT_DEFAULT = 792  # Xteink X3
WIDTH_X4 = 480        # Xteink X4
HEIGHT_X4 = 800       # Xteink X4

BG_LIGHT = 255   # white background (light mode)
BG_DARK = 0      # black background (dark mode)
FG_LIGHT = 0     # black text (light mode)
FG_DARK = 255    # white text (dark mode)

OUTER_MARGIN = 18
CARD_PADDING = 24


# ----------------------------- Font helpers -----------------------------

FONT_CANDIDATES = {
    "serif": [
        "/System/Library/Fonts/Palatino.ttc",
        "/System/Library/Fonts/Times.ttc",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Palatino.ttc",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ],
    "serif_italic": [
        "/System/Library/Fonts/Supplemental/Times New Roman Italic.ttf",
        "/System/Library/Fonts/Palatino.ttc",  # has italic faces
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
    ],
    "sans": [
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ],
    "sans_bold": [
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/System/Library/Fonts/SFNSDisplay.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ],
    "sans_italic": [
        "/System/Library/Fonts/SFNSItalic.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    ],
    "mono": [
        "/System/Library/Fonts/SFNSMono.ttf",
        "/Library/Fonts/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ],
}

# Extra scan directories used only when the curated candidates above fail.
# Useful when the container has not been rebuilt with `fonts-dejavu-core`
# or when running on the Mac mini host where additional TTFs are available.
_EXTRA_FONT_ROOTS = (
    "/System/Library/Fonts/Supplemental",
    "/System/Library/Fonts",
    "/Library/Fonts",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "/host-fonts",
)


def discover_system_fonts() -> list[Path]:
    """Return every TTF/TTC font found in well-known system locations."""
    seen: set[Path] = set()
    fonts: list[Path] = []
    for root in _EXTRA_FONT_ROOTS:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for pattern in ("*.ttf", "*.ttc", "*.otf"):
            for match in root_path.rglob(pattern):
                if match in seen:
                    continue
                seen.add(match)
                fonts.append(match)
    return fonts


def find_font(
    family: str,
    size: int,
    bold: bool = False,
    italic: bool = False,
    extra_roots: tuple[str, ...] | None = None,
) -> ImageFont.FreeTypeFont:
    """Find a scalable font for the requested family and style.

    Falls back to a runtime scan of system fonts when the curated
    candidates list is empty (e.g. the container has not been
    rebuilt with DejaVu fonts).
    """
    if bold and italic:
        key = "sans_bold"
    elif italic:
        key = "serif_italic" if family == "serif" else "sans_italic"
    elif bold:
        key = "sans_bold" if family == "sans" else "serif"
    else:
        key = family

    if family == "serif" and not bold and not italic:
        candidates = FONT_CANDIDATES["serif"]
    else:
        candidates = FONT_CANDIDATES.get(key, FONT_CANDIDATES[family])

    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue

    extra_seen: set[str] = set()
    for candidate in candidates:
        extra_seen.add(candidate)
    roots = tuple(extra_roots) if extra_roots is not None else _EXTRA_FONT_ROOTS
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for match in root_path.rglob("*.ttf"):
            if str(match) in extra_seen:
                continue
            try:
                return ImageFont.truetype(match, size)
            except Exception:
                continue

    raise RuntimeError(
        f"No scalable font found for {family}; install DejaVu fonts (apt-get install fonts-dejavu-core)"
    )


def list_available_fonts() -> list[dict[str, str]]:
    """Return a deduplicated list of fonts available for the picker."""
    seen: set[str] = set()
    entries: list[dict[str, str]] = []
    for candidate in discover_system_fonts():
        path_str = str(candidate)
        if path_str in seen:
            continue
        seen.add(path_str)
        entries.append({
            "path": path_str,
            "label": candidate.name,
            "family": candidate.stem,
        })
    return entries


# ----------------------------- Drawing primitives -----------------------------

def measure(text: str, font: ImageFont.FreeTypeFont) -> tuple[int, int]:
    """Return (width, height) of a text line."""
    bbox = font.getbbox(text)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        trial = (current + " " + w).strip()
        w_width, _ = measure(trial, font)
        if w_width <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def truncate_lines(text: str, font: ImageFont.FreeTypeFont,
                   max_width: int, max_lines: int,
                   family: str = "sans", bold: bool = False,
                   italic: bool = False) -> list[str]:
    """Wrap text into lines that each fit within `max_width`.

    Safe-area behaviour: if any wrapped line is still wider than
    `max_width` (the usual culprit is a long URL or compound word that
    cannot be split on a space), the font is shrunk in 3-pt steps until
    every line fits or the size floor (10-pt) is reached. Only after the
    shrink succeeds do we cap at `max_lines` and add the ellipsis.
    """
    if not text:
        return [""]
    current_font = font
    for _ in range(6):
        lines = wrap_text(text, current_font, max_width)
        widest = max((measure(line, current_font)[0] for line in lines), default=0)
        if widest <= max_width:
            break
        new_size = max(10, getattr(current_font, "size", 24) - 3)
        if new_size >= getattr(current_font, "size", 24):
            break
        try:
            current_font = find_font(family, new_size, bold=bold, italic=italic)
        except RuntimeError:
            break

    lines = wrap_text(text, current_font, max_width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while True:
            test = last.rstrip(",.;:- ") + "…"
            if measure(test, current_font)[0] <= max_width or len(last) <= 4:
                lines[-1] = test
                break
            last = last[:-1]
    return lines


def draw_centered_text(draw, text: str, font, y: int, width: int = WIDTH_DEFAULT,
                       fill: int = 0, shadow: bool = False, dy: int = 1,
                       family: str = "serif", bold: bool = False) -> int:
    """Draw text horizontally centered on `width`.

    Safe-area behaviour: if `text` would overflow the card's content area
    (`width` minus the outer margin and card padding on each side), the
    font is shrunk in 4-pt steps until the text fits or hits a 14-pt floor.
    Without this, long words or wide user-selected fonts can poke past
    the inner border and land outside the bezel.
    """
    text = text.strip()
    if not text:
        return y
    safe_width = width - 2 * OUTER_MARGIN - 2 * CARD_PADDING
    current_font = font
    w, h = measure(text, current_font)
    while w > safe_width and getattr(current_font, "size", 0) > 14:
        new_size = max(14, getattr(current_font, "size", 14) - 4)
        if new_size == getattr(current_font, "size", 0):
            break
        try:
            current_font = find_font(family, new_size, bold=bold)
        except RuntimeError:
            break
        w, h = measure(text, current_font)
    x = (width - w) // 2
    if shadow:
        # subtle 1-pixel shadow (in 1-bit, this just outlines slightly)
        draw.text((x + dy, y + dy), text, font=current_font, fill=fill)
    draw.text((x, y), text, font=current_font, fill=fill)
    return y + h


def draw_section_title(draw, text: str, font, y: int,
                       x_left: int, x_right: int, fill: int = 0) -> int:
    """Draw a bold uppercase section heading."""
    text = text.upper()
    w, h = measure(text, font)
    x = x_left + ((x_right - x_left) - w) // 2
    draw.text((x, y), text, font=font, fill=fill)
    return y + h + 14


def draw_body_lines(draw, lines: list[str], font, x: int, y: int,
                    max_width: int, line_spacing: int = 6, fill: int = 0,
                    italic: bool = False) -> int:
    """Draw a list of pre-wrapped lines. Returns new y."""
    if not lines:
        return y
    current_y = y
    for line in lines:
        # remove extra spaces
        line = re.sub(r"\s+", " ", line).strip()
        draw.text((x, current_y), line, font=font, fill=fill)
        # advance by line height + spacing
        _, h = measure(line, font)
        current_y += h + line_spacing
    return current_y - line_spacing


# ----------------------------- Card rendering -----------------------------

def format_pronunciation(pronunciation: str, max_variants: int = 3) -> str:
    """Render IPA pronunciations one per line.

    Each variant is wrapped in slashes (e.g. ``/kæt/``) and joined with
    newlines so render_card can draw them on separate rows. Empty parts
    are dropped. Capped at ``max_variants`` (default 3) so a word with
    seven StarDict pronunciations doesn't push the rest of the card into
    the footer — the remaining variants are silently dropped.
    """
    variants = [part.strip().strip("/") for part in pronunciation.split(" / ")]
    variants = [part for part in variants if part]
    return "\n".join(f"/{part}/" for part in variants[:max_variants])


def ensure_terminator(text: str, terminator: str = ".") -> str:
    """Ensure the text ends with a sentence terminator.

    Strips trailing whitespace first; leaves existing terminators
    (``.``, ``!``, ``?``) and the truncation marker (``…``) alone.
    """
    text = text.rstrip()
    if not text or text[-1] in ".!?…":
        return text
    return text + terminator


# Section titles rendered on each card. The keys match the canonical English
# labels; `render_card` looks up the localized value for the source language.
SECTION_TITLES: dict[str, dict[str, str]] = {
    "en": {
        "pronunciation": "Pronunciation",
        "definition": "Definition",
        "usage": "Usage",
        "synonyms": "Synonyms",
    },
    "es": {
        "pronunciation": "Pronunciación",
        "definition": "Definición",
        "usage": "Uso",
        "synonyms": "Sinónimos",
    },
}


def section_titles_for(source_lang: str | None) -> dict[str, str]:
    """Return the section-title dict for a source language code (e.g. 'en')."""
    if not source_lang:
        return SECTION_TITLES["en"]
    return SECTION_TITLES.get(source_lang, SECTION_TITLES["en"])


def render_card(
    word: str,
    pronunciation: str,
    definition: str,
    example: str,
    synonyms: str,
    book_name: str,
    page_no: int = 0,
    total_pages: int = 0,
    output_path: Path = None,
    darkmode: bool = False,
    width: int = WIDTH_DEFAULT,
    height: int = HEIGHT_DEFAULT,
    font_path: str | None = None,
    section_titles: dict[str, str] | None = None,
) -> None:
    if not definition.strip():
        raise ValueError("definition is required")

    # Local colors (avoids mutating module globals).
    bg = BG_DARK if darkmode else BG_LIGHT
    fg = FG_DARK if darkmode else FG_LIGHT

    # Compute a font-size scale relative to the X3 baseline (528×792).
    # We scale by width primarily, since text wrapping depends on horizontal space;
    # but we cap with height so fonts don't overshoot a small canvas.
    scale = min(width / WIDTH_DEFAULT, height / HEIGHT_DEFAULT)
    def s(base):
        return max(8, int(round(base * scale)))

    titles = section_titles or SECTION_TITLES["en"]

    img = Image.new("1", (width, height), bg)
    draw = ImageDraw.Draw(img)

    # Fonts (scaled to canvas size). When the caller pins a specific font
    # path, it overrides every role EXCEPT the IPA section. The IPA section
    # always resolves through find_font("serif", italic=True) because
    # phonetic glyphs (/ʃ/, /θ/, /ŋ/, etc.) require a font with full IPA
    # coverage — Times New Roman Italic on macOS, DejaVu Serif Italic on
    # Linux. The user's pick is often a TrueType collection
    # (Palatino.ttc, Times.ttc) whose first face is the Roman cut and
    # lacks those glyphs; using the TTC verbatim breaks the pronunciation
    # line. Keeping the IPA on the curated italic makes the card readable
    # regardless of what the user picks.
    if font_path and Path(font_path).exists():
        try:
            word_font = ImageFont.truetype(font_path, s(56))
            title_font = ImageFont.truetype(font_path, s(20))
            body_font = ImageFont.truetype(font_path, s(26))
            body_italic_font = ImageFont.truetype(font_path, s(26))
            small_font = ImageFont.truetype(font_path, s(14))
            number_font = ImageFont.truetype(font_path, s(22))
        except Exception:
            word_font = find_font("serif", s(56), bold=False)
            title_font = find_font("sans", s(20), bold=True)
            body_font = find_font("sans", s(26))
            body_italic_font = find_font("sans", s(26), italic=True)
            small_font = find_font("sans", s(14))
            number_font = find_font("serif", s(22), italic=True)
    else:
        word_font = find_font("serif", s(56), bold=False)
        title_font = find_font("sans", s(20), bold=True)
        body_font = find_font("sans", s(26))
        body_italic_font = find_font("sans", s(26), italic=True)
        small_font = find_font("sans", s(14))
        number_font = find_font("serif", s(22), italic=True)

    # IPA always resolves through the curated italic regardless of the
    # user's font pick — see the comment block above.
    ipa_font = find_font("serif", s(22), italic=True)

    # ----------- Outer card border (rounded rectangle) -----------
    card_rect = [OUTER_MARGIN, OUTER_MARGIN, width - OUTER_MARGIN - 1, height - OUTER_MARGIN - 1]
    draw.rounded_rectangle(card_rect, radius=max(8, s(14)), outline=fg, width=2)

    inner_x_left = card_rect[0] + CARD_PADDING
    inner_x_right = card_rect[2] - CARD_PADDING
    content_w = inner_x_right - inner_x_left

    # ----------- Word at top (serif, centered) -----------
    y = OUTER_MARGIN + CARD_PADDING + 14
    y = draw_centered_text(draw, word, word_font, y, width=width, fill=fg)
    # Underline (thin, decorative, classical dictionary style)
    underline_y = y + 16
    line_w_min, _ = measure(word, word_font)
    line_w = max(line_w_min + 40, 80)
    cx = width // 2
    draw.line([(cx - line_w // 2, underline_y), (cx + line_w // 2, underline_y)],
              fill=fg, width=1)
    y = underline_y + 22

    # ----------- Pronunciation -----------
    ipa_text = format_pronunciation(pronunciation)
    if ipa_text:
        y = draw_section_title(
            draw, titles["pronunciation"], title_font, y, inner_x_left, inner_x_right, fill=fg
        )
        # Each pronunciation goes on its own line, drawn individually
        # because Pillow's draw.text does not honour embedded newlines.
        for line in ipa_text.split("\n"):
            current_font = ipa_font
            if measure(line, current_font)[0] > content_w:
                for size in [s(24), s(22), s(20)]:
                    candidate = find_font("serif", size, italic=True)
                    if measure(line, candidate)[0] <= content_w:
                        current_font = candidate
                        break
            draw.text(
                ((width - measure(line, current_font)[0]) // 2, y),
                line, font=current_font, fill=fg,
            )
            y += measure(line, current_font)[1] + s(4)
        y += s(24)  # gap before next section

    # ----------- Definition -----------
    y = draw_section_title(
        draw, titles["definition"], title_font, y, inner_x_left, inner_x_right, fill=fg
    )
    def_lines = truncate_lines(
        ensure_terminator(definition), body_font, content_w, max_lines=4
    )
    y = draw_body_lines(draw, def_lines, body_font, inner_x_left, y, content_w,
                        line_spacing=8, fill=fg)
    y += s(28)  # gap before next section

    # ----------- Usage / Example -----------
    if example.strip():
        y = draw_section_title(
            draw, titles["usage"], title_font, y, inner_x_left, inner_x_right, fill=fg
        )
        ex_lines = truncate_lines(
            ensure_terminator(example), body_font, content_w, max_lines=4
        )
        y = draw_body_lines(draw, ex_lines, body_font, inner_x_left, y, content_w,
                            line_spacing=8, fill=fg)
        y += s(28)

    # ----------- Synonyms (italic, dictionary style) -----------
    if synonyms.strip():
        y = draw_section_title(
            draw, titles["synonyms"], title_font, y, inner_x_left, inner_x_right, fill=fg
        )
        syn_lines = truncate_lines(
            ensure_terminator(synonyms), body_italic_font, content_w,
            max_lines=4, italic=True,
        )
        y = draw_body_lines(draw, syn_lines, body_italic_font, inner_x_left, y, content_w,
                            line_spacing=8, fill=fg)

    # ----------- Footer: source on left, page number on right -----------
    footer_y = height - OUTER_MARGIN - CARD_PADDING - 8
    # Tiny separator line above
    sep_y = footer_y - 22
    draw.line([(inner_x_left, sep_y), (inner_x_right, sep_y)], fill=fg, width=1)

    if book_name:
        # truncate book name if needed
        max_chars = 38
        if len(book_name) > max_chars:
            book_name = book_name[: max_chars - 1] + "…"
        draw.text((inner_x_left, footer_y), book_name, font=small_font, fill=fg)

    if total_pages > 0:
        counter_text = f"{page_no} / {total_pages}"
        cw, _ = measure(counter_text, number_font)
        draw.text((inner_x_right - cw, footer_y - 2), counter_text,
                  font=number_font, fill=fg)

    img.save(output_path, format="BMP")


# ----------------------------- Main -----------------------------

def slug(s: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")
    return s[:max_len] or "card"


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Generate stylized e-ink BMP flashcards (Xteink X3 or X4).",
    )
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output-dir", default="bmp")
    ap.add_argument("--max-items", type=int, default=0,
                    help="If > 0, limit to this many rows (0 = all)")
    ap.add_argument("--darkmode", action="store_true",
                    help="Invert colors: black background, white text.")
    ap.add_argument("--device", choices=["x3", "x4"], default="x3",
                    help="E-reader model (drives output resolution).")
    ap.add_argument("--width", type=int, default=None,
                    help="Override output width (overrides --device).")
    ap.add_argument("--height", type=int, default=None,
                    help="Override output height (overrides --device).")
    args = ap.parse_args()

    # Resolve resolution from --device or explicit --width/--height
    if args.width and args.height:
        width, height = args.width, args.height
    elif args.device == "x4":
        width, height = WIDTH_X4, HEIGHT_X4
    else:
        width, height = WIDTH_DEFAULT, HEIGHT_DEFAULT

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    if args.darkmode:
        out_dir = out_dir.parent / f"{out_dir.name}_dark"
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[-] CSV not found: {csv_path}", file=sys.stderr)
        return 1

    # First pass: count total rows that have a word
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("word") or "").strip() and (row.get("definition") or "").strip():
                rows.append(row)
    total = len(rows)

    # Apply max-items cap
    if args.max_items and args.max_items > 0:
        rows = rows[: args.max_items]

    count = 0
    for i, row in enumerate(rows):
        word = (row.get("word") or "").strip()
        pronunciation = (row.get("pronunciation") or "").strip()
        definition = (row.get("definition") or "").strip()
        example = (row.get("example") or "").strip()
        synonyms = (row.get("synonyms") or "").strip()
        book = (row.get("book") or "").strip()

        out_name = f"{slug(word)}_{i:04d}.bmp"
        out_path = out_dir / out_name
        try:
            render_card(
                word, pronunciation, definition, example, synonyms,
                book_name=book,
                page_no=i + 1, total_pages=total,
                output_path=out_path,
                darkmode=args.darkmode,
                width=width,
                height=height,
            )
        except Exception as e:
            print(f"    failed on '{word}': {e}", file=sys.stderr)
            continue
        count += 1
        if count % 10 == 0:
            print(f"[+] {count} cards rendered...")

    print(f"\n[+] Done. {count} BMPs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())