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

WIDTH = 528
HEIGHT = 792
BG = 255   # white
FG = 0     # black

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


def find_font(family: str, size: int, bold: bool = False, italic: bool = False) -> ImageFont.FreeTypeFont:
    """Find a usable font for the given family + style."""
    if bold and italic:
        key = "sans_bold"  # fallback; most files don't have explicit bold-italic
    elif italic:
        key = "serif_italic" if family == "serif" else "sans_italic"
    elif bold:
        key = "sans_bold" if family == "sans" else "serif"
    else:
        key = family

    # If family=serif and not bold/italic, prefer the serif list directly
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
    return ImageFont.load_default()


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
                   max_width: int, max_lines: int) -> list[str]:
    lines = wrap_text(text, font, max_width)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        last = lines[-1]
        while True:
            test = last.rstrip(",.;:- ") + "…"
            if measure(test, font)[0] <= max_width or len(last) <= 4:
                lines[-1] = test
                break
            last = last[:-1]
    return lines


def draw_centered_text(draw, text: str, font, y: int, width: int = WIDTH,
                       fill: int = FG, shadow: bool = False, dy: int = 1) -> int:
    text = text.strip()
    if not text:
        return y
    w, h = measure(text, font)
    x = (width - w) // 2
    if shadow:
        # subtle 1-pixel shadow (in 1-bit, this just outlines slightly)
        draw.text((x + dy, y + dy), text, font=font, fill=FG)
    draw.text((x, y), text, font=font, fill=FG)
    return y + h


def draw_section_title(draw, text: str, font, y: int,
                       x_left: int, x_right: int, fill: int = FG) -> int:
    """Section title in caps, centered between x_left and x_right, with thin rule below."""
    text = text.upper()
    w, h = measure(text, font)
    x = x_left + ((x_right - x_left) - w) // 2
    draw.text((x, y), text, font=font, fill=fill)
    rule_y = y + h + 8
    # Centered rule, narrower than the body width
    rule_in = 40
    draw.line([(x_left + rule_in, rule_y), (x_right - rule_in, rule_y)],
              fill=fill, width=1)
    return rule_y + 14  # body starts here


def draw_body_lines(draw, lines: list[str], font, x: int, y: int,
                    max_width: int, line_spacing: int = 6, fill: int = FG,
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
) -> None:
    img = Image.new("1", (WIDTH, HEIGHT), BG)
    draw = ImageDraw.Draw(img)

    # Fonts
    word_font = find_font("serif", 56, bold=False)         # slightly smaller word
    ipa_font = find_font("serif", 22, italic=True)         # italic small for IPA
    title_font = find_font("sans", 20, bold=True)
    body_font = find_font("sans", 26)
    body_italic_font = find_font("sans", 26, italic=True)
    small_font = find_font("sans", 14)
    number_font = find_font("serif", 22, italic=True)

    # ----------- Outer card border (rounded rectangle) -----------
    card_rect = [OUTER_MARGIN, OUTER_MARGIN, WIDTH - OUTER_MARGIN - 1, HEIGHT - OUTER_MARGIN - 1]
    draw.rounded_rectangle(card_rect, radius=14, outline=FG, width=2)

    inner_x_left = card_rect[0] + CARD_PADDING
    inner_x_right = card_rect[2] - CARD_PADDING
    content_w = inner_x_right - inner_x_left

    # ----------- Word at top (serif, centered) -----------
    y = OUTER_MARGIN + CARD_PADDING + 14
    y = draw_centered_text(draw, word, word_font, y, width=WIDTH)
    # Underline (thin, decorative, classical dictionary style)
    underline_y = y + 16
    line_w_min, _ = measure(word, word_font)
    line_w = max(line_w_min + 40, 80)
    cx = WIDTH // 2
    draw.line([(cx - line_w // 2, underline_y), (cx + line_w // 2, underline_y)],
              fill=FG, width=1)
    y = underline_y + 22

    # ----------- Pronunciation (italic small, centered) -----------
    # Show at most 2 variants to keep the layout clean
    if pronunciation:
        ipa_to_show = pronunciation
        # If multiple variants separated by " / ", keep first two
        variants = [v.strip() for v in ipa_to_show.split(" / ") if v.strip()]
        if len(variants) > 2:
            ipa_to_show = " / ".join(variants[:2]) + "  …"
        elif len(variants) == 0:
            ipa_to_show = ""
        else:
            ipa_to_show = " / ".join(variants)

        ipa_text = f"/{ipa_to_show}/"
        ipa_w, ipa_h = measure(ipa_text, ipa_font)
        if ipa_w > content_w:
            # shrink font
            for size in [24, 22, 20]:
                candidate = find_font("serif", size, italic=True)
                if measure(ipa_text, candidate)[0] <= content_w:
                    ipa_font = candidate
                    break
        draw.text(((WIDTH - measure(ipa_text, ipa_font)[0]) // 2, y), ipa_text,
                  font=ipa_font, fill=FG)
        y += measure(ipa_text, ipa_font)[1] + 8

    # ----------- Pronunciation label / helper -----------
    # little "ipa" tag
    if pronunciation:
        tag = "pronunciation"
        tag_font = small_font
        tag_w, tag_h = measure(tag, tag_font)
        draw.text(((WIDTH - tag_w) // 2, y), tag, font=tag_font, fill=FG)
        y += tag_h + 20

    # ----------- Definition -----------
    y = draw_section_title(draw, "Definition", title_font, y, inner_x_left, inner_x_right)
    def_lines = truncate_lines(definition, body_font, content_w, max_lines=4)
    y = draw_body_lines(draw, def_lines, body_font, inner_x_left, y, content_w,
                        line_spacing=8)
    y += 28  # gap before next section

    # ----------- Usage / Example -----------
    y = draw_section_title(draw, "Usage", title_font, y, inner_x_left, inner_x_right)
    ex_lines = truncate_lines(example, body_font, content_w, max_lines=4)
    y = draw_body_lines(draw, ex_lines, body_font, inner_x_left, y, content_w,
                        line_spacing=8)
    y += 28

    # ----------- Synonyms (italic, dictionary style) -----------
    y = draw_section_title(draw, "Synonyms", title_font, y, inner_x_left, inner_x_right)
    syn_lines = truncate_lines(synonyms, body_italic_font, content_w, max_lines=4)
    y = draw_body_lines(draw, syn_lines, body_italic_font, inner_x_left, y, content_w,
                        line_spacing=8)

    # ----------- Footer: source on left, page number on right -----------
    footer_y = HEIGHT - OUTER_MARGIN - CARD_PADDING - 8
    # Tiny separator line above
    sep_y = footer_y - 22
    draw.line([(inner_x_left, sep_y), (inner_x_right, sep_y)], fill=FG, width=1)

    if book_name:
        # truncate book name if needed
        max_chars = 38
        if len(book_name) > max_chars:
            book_name = book_name[: max_chars - 1] + "…"
        draw.text((inner_x_left, footer_y), book_name, font=small_font, fill=FG)

    if total_pages > 0:
        counter_text = f"{page_no} / {total_pages}"
        cw, _ = measure(counter_text, number_font)
        draw.text((inner_x_right - cw, footer_y - 2), counter_text,
                  font=number_font, fill=FG)

    img.save(output_path, format="BMP")


# ----------------------------- Main -----------------------------

def slug(s: str, max_len: int = 60) -> str:
    s = re.sub(r"[^a-z0-9_]+", "_", s.lower()).strip("_")
    return s[:max_len] or "card"


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate stylized 528x792 BMP flashcards.")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--output-dir", default="bmp")
    ap.add_argument("--max-items", type=int, default=0,
                    help="If > 0, limit to this many rows (0 = all)")
    args = ap.parse_args()

    csv_path = Path(args.csv).expanduser().resolve()
    out_dir = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[-] CSV not found: {csv_path}", file=sys.stderr)
        return 1

    # First pass: count total rows that have a word
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("word") or "").strip():
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