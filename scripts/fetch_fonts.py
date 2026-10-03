"""
scripts/fetch_fonts.py
Fetch fonts, build fonts_manifest.json, and generate contact_sheet.png.

Tasks:
1. Download 28 handwriting/informal Arabic fonts from Google Fonts into data/fonts/.
2. Compute SHA-256 for each font.
3. Save data/fonts_manifest.json (family, url, license, sha256).
4. Generate data/font_samples/contact_sheet.png:
   - Displays all 28 fonts rendering the exact same 2-line Arabic text.
   - Labeled with the font name.
   - Uniform layout without category grouping.
   - Rendered using HarfBuzz OpenType shaping.
"""

from __future__ import annotations
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from typing import List, Dict

import freetype
from PIL import Image, ImageDraw, ImageFont
import uharfbuzz as hb

FONTS_DIR = Path("data/fonts")
SAMPLES_DIR = Path("data/font_samples")
MANIFEST_PATH = Path("data/fonts_manifest.json")

FONTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# 28 candidate fonts on Google Fonts
FONT_FAMILIES = [
    "Alkalami",
    "Amiri",
    "Aref Ruqaa",
    "Aref Ruqaa Ink",
    "Badeen Display",
    "Baloo Bhaijaan 2",
    "Blaka",
    "Blaka Ink",
    "Cairo Play",
    "El Messiri",
    "Gulzar",
    "Harmattan",
    "Jomhuria",
    "Katibeh",
    "Lalezar",
    "Lateef",
    "Lemonada",
    "Marhey",
    "Mirza",
    "Playpen Sans Arabic",
    "Qahiri",
    "Rakkas",
    "Reem Kufi Fun",
    "Reem Kufi Ink",
    "Ruwudu",
    "Scheherazade New",
    "Vibes",
    "Zain",
]

SAMPLE_LINE_1 = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"
SAMPLE_LINE_2 = "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"


def fetch_ttf_url(family: str) -> str:
    url_name = family.replace(" ", "+")
    css_url = f"https://fonts.googleapis.com/css2?family={url_name}&display=swap"
    req = urllib.request.Request(css_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        css = resp.read().decode("utf-8")
    urls = re.findall(r"src:\s*url\((https://[^\)]+\.ttf)\)", css)
    if not urls:
        urls = re.findall(r"src:\s*url\((https://[^\)]+)\)", css)
    if not urls:
        raise ValueError(f"Could not find download URL for {family}")
    return urls[0]


def download_and_hash_fonts() -> List[Dict[str, str]]:
    manifest = []
    print("=== Downloading & Hashing 28 Fonts ===")

    for fam in FONT_FAMILIES:
        slug = re.sub(r"[^a-zA-Z0-9_]", "_", fam)
        ttf_path = FONTS_DIR / f"{slug}.ttf"

        url = fetch_ttf_url(fam)
        if not ttf_path.exists() or ttf_path.stat().st_size < 1000:
            print(f"Downloading {fam}...")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req) as resp:
                content = resp.read()
            ttf_path.write_bytes(content)

        data = ttf_path.read_bytes()
        sha256 = hashlib.sha256(data).hexdigest()

        manifest.append({
            "family": fam,
            "filename": ttf_path.name,
            "url": url,
            "license": "SIL Open Font License 1.1",
            "sha256": sha256,
        })
        print(f"  {fam:<22} -> sha256: {sha256[:16]}... ({len(data)} bytes)")

    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"\nSaved font manifest to {MANIFEST_PATH}")
    return manifest


def render_harfbuzz_line(font_path: Path, text: str, font_size: int = 26) -> Image.Image:
    """Renders Arabic text with native HarfBuzz OpenType shaping and FreeType rasterization."""
    blob = hb.Blob.from_file_path(str(font_path))
    face = hb.Face(blob)
    hb_font = hb.Font(face)
    scale = int(font_size * 64)
    hb_font.scale = (scale, scale)

    buf = hb.Buffer()
    buf.add_str(text)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)

    infos = buf.glyph_infos
    positions = buf.glyph_positions

    ft_face = freetype.Face(str(font_path))
    ft_face.set_char_size(scale, scale)

    total_advance = sum(pos.x_advance for pos in positions) >> 6
    width = max(total_advance + 20, 50)
    height = int(font_size * 2.2)
    baseline = int(font_size * 1.5)

    img = Image.new("L", (width, height), color=255)

    pen_x = 10
    for info, pos in zip(infos, positions):
        glyph_id = info.codepoint
        ft_face.load_glyph(glyph_id, freetype.FT_LOAD_RENDER)
        slot = ft_face.glyph
        bitmap = slot.bitmap

        bx = (pen_x + (pos.x_offset >> 6)) + slot.bitmap_left
        by = (baseline - (pos.y_offset >> 6)) - slot.bitmap_top

        if bitmap.width > 0 and bitmap.rows > 0:
            import numpy as np
            bm_arr = np.array(bitmap.buffer, dtype=np.uint8).reshape((bitmap.rows, bitmap.width))
            mask = Image.fromarray(bm_arr)
            img.paste(Image.new("L", (bitmap.width, bitmap.rows), color=0), (bx, by), mask=mask)

        pen_x += (pos.x_advance >> 6)

    return img


def generate_contact_sheet(manifest: List[Dict[str, str]]):
    """Generates a uniform contact sheet displaying all 28 fonts with the same 2-line text."""
    print("\n=== Generating Contact Sheet (data/font_samples/contact_sheet.png) ===")
    
    # 2 columns x 14 rows grid
    cols = 2
    rows = (len(manifest) + cols - 1) // cols
    cell_w = 650
    cell_h = 130
    sheet_w = cols * cell_w + 40
    sheet_h = rows * cell_h + 80

    sheet = Image.new("RGB", (sheet_w, sheet_h), color=(250, 250, 252))
    draw = ImageDraw.Draw(sheet)

    # Title header
    header_font = ImageFont.load_default()
    draw.text((25, 20), "ARABIC FONT SURVEY CONTACT SHEET — 28 GOOGLE FONTS (SAME 2-LINE TEXT)", fill=(30, 30, 30))
    draw.line([(25, 45), (sheet_w - 25, 45)], fill=(200, 200, 200), width=1)

    for idx, item in enumerate(manifest):
        fam = item["family"]
        r = idx // cols
        c = idx % cols
        x0 = 25 + c * cell_w
        y0 = 60 + r * cell_h

        # Cell border card
        draw.rectangle([x0, y0, x0 + cell_w - 20, y0 + cell_h - 15], outline=(220, 220, 225), fill=(255, 255, 255), width=1)

        # Label: Font number and family name
        label_text = f"{idx + 1:02d}. {fam}"
        draw.text((x0 + 15, y0 + 10), label_text, fill=(80, 80, 80))

        # Render 2 lines with this font
        font_path = FONTS_DIR / item["filename"]
        try:
            line1_img = render_harfbuzz_line(font_path, SAMPLE_LINE_1, font_size=20)
            line2_img = render_harfbuzz_line(font_path, SAMPLE_LINE_2, font_size=20)

            # Paste line 1 and line 2 (align to right within cell for RTL aesthetic)
            l1_x = max(x0 + 15, x0 + cell_w - 35 - line1_img.width)
            l2_x = max(x0 + 15, x0 + cell_w - 35 - line2_img.width)

            sheet.paste(line1_img, (l1_x, y0 + 32))
            sheet.paste(line2_img, (l2_x, y0 + 72))
        except Exception as e:
            draw.text((x0 + 20, y0 + 50), f"Error rendering: {e}", fill=(200, 0, 0))

    contact_path = SAMPLES_DIR / "contact_sheet.png"
    sheet.save(contact_path)
    print(f"Contact sheet saved to {contact_path} ({sheet_w}x{sheet_h})")


def main():
    manifest = download_and_hash_fonts()
    generate_contact_sheet(manifest)


if __name__ == "__main__":
    main()
