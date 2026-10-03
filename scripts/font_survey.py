"""
scripts/font_survey.py
Survey of >=25 Arabic handwriting/informal fonts under free licenses (Google Fonts).

Tasks:
1. Download TTF files for candidate handwriting/informal Arabic fonts.
2. Inspect license (OFL / Apache), diacritic support (tashkeel in CMAP),
   and OpenType cursive shaping support (GSUB table + HarfBuzz/Raqm).
3. Render one sample Ayah per font into data/font_samples/{font_name}.png.
4. Report a structured summary table and save full details to markdown.
"""

from __future__ import annotations
import re
import sys
import urllib.request
from pathlib import Path
from typing import Dict, List

from PIL import Image, ImageDraw, ImageFont
import arabic_reshaper
from bidi.algorithm import get_display
from fontTools.ttLib import TTFont
import uharfbuzz as hb

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

FONTS_DIR = Path("data/fonts")
SAMPLES_DIR = Path("data/font_samples")
FONTS_DIR.mkdir(parents=True, exist_ok=True)
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

# 28 candidate fonts on Google Fonts with handwriting, informal, calligraphic, or display feel
CANDIDATES = [
    "Aref Ruqaa",
    "Aref Ruqaa Ink",
    "Playpen Sans Arabic",
    "Marhey",
    "Vibes",
    "Lateef",
    "Mirza",
    "Scheherazade New",
    "Lemonada",
    "Katibeh",
    "Rakkas",
    "Alkalami",
    "Gulzar",
    "Blaka",
    "Blaka Ink",
    "Cairo Play",
    "Reem Kufi Fun",
    "Reem Kufi Ink",
    "Baloo Bhaijaan 2",
    "Lalezar",
    "Ruwudu",
    "Jomhuria",
    "Qahiri",
    "Badeen Display",
    "El Messiri",
    "Harmattan",
    "Zain",
    "Amiri",
]

# Standard core Tashkeel codepoints to verify diacritic support
TASHKEEL_CODEPOINTS = {
    0x064B: "Fathatan",
    0x064C: "Dammatan",
    0x064D: "Kasratan",
    0x064E: "Fatha",
    0x064F: "Damma",
    0x0650: "Kasra",
    0x0651: "Shadda",
    0x0652: "Sukun",
}

SAMPLE_AYAH = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ"


def fetch_ttf_url(family: str) -> str:
    """Queries Google Fonts CSS v2 endpoint to retrieve direct TTF URL."""
    url_name = family.replace(" ", "+")
    css_url = f"https://fonts.googleapis.com/css2?family={url_name}&display=swap"
    req = urllib.request.Request(css_url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        css = resp.read().decode("utf-8")
    urls = re.findall(r"src:\s*url\((https://[^\)]+\.ttf)\)", css)
    if not urls:
        # Fallback to any url format
        urls = re.findall(r"src:\s*url\((https://[^\)]+)\)", css)
    if not urls:
        raise ValueError(f"Could not find download URL for font family {family}")
    return urls[0]


def download_font(family: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", family)
    ttf_path = FONTS_DIR / f"{slug}.ttf"
    if ttf_path.exists() and ttf_path.stat().st_size > 1000:
        return ttf_path

    url = fetch_ttf_url(family)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp:
        content = resp.read()
    ttf_path.write_bytes(content)
    return ttf_path


def inspect_font(ttf_path: Path) -> dict:
    tt = TTFont(ttf_path)

    # 1. License string from name table
    license_str = "OFL (Open Font License)"
    name_table = tt.get("name")
    if name_table:
        for record in name_table.names:
            if record.nameID in (13, 14):  # License Description / License URL
                try:
                    text = record.toUnicode()
                    if "apache" in text.lower():
                        license_str = "Apache 2.0"
                    elif "ofl" in text.lower() or "open font license" in text.lower():
                        license_str = "SIL OFL 1.1"
                except Exception:
                    pass

    # 2. Diacritic / Tashkeel support
    cmap = tt.getBestCmap() or {}
    tashkeel_found = sum(1 for cp in TASHKEEL_CODEPOINTS if cp in cmap)
    tashkeel_pct = (tashkeel_found / len(TASHKEEL_CODEPOINTS)) * 100
    diacritic_support = "Full (8/8)" if tashkeel_found == 8 else f"Partial ({tashkeel_found}/8)"

    # 3. OpenType Shaping support (GSUB table presence)
    has_gsub = "GSUB" in tt

    # 4. Shape verification using HarfBuzz
    blob = hb.Blob.from_file_path(str(ttf_path))
    face = hb.Face(blob)
    font = hb.Font(face)
    buf = hb.Buffer()
    buf.add_str(SAMPLE_AYAH)
    buf.guess_segment_properties()
    hb.shape(font, buf)
    harfbuzz_shaped = len(buf.glyph_infos) > 0 and has_gsub

    return {
        "license": license_str,
        "diacritics": diacritic_support,
        "tashkeel_pct": tashkeel_pct,
        "has_gsub": has_gsub,
        "shaping_ok": harfbuzz_shaped,
        "glyph_count": len(cmap),
    }


def render_sample(ttf_path: Path, family: str) -> Path:
    slug = re.sub(r"[^a-zA-Z0-9_]", "_", family)
    out_img = SAMPLES_DIR / f"{slug}.png"

    # Reshape Arabic text for Pillow rendering
    reshaped = arabic_reshaper.reshape(SAMPLE_AYAH)
    bidi_text = get_display(reshaped)

    font_size = 32
    try:
        font = ImageFont.truetype(str(ttf_path), size=font_size)
    except Exception:
        font = ImageFont.load_default()

    # Image canvas
    width, height = 700, 100
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Draw label (font family name) in small gray
    draw.text((15, 8), f"{family}", fill=(120, 120, 120))

    # Draw Arabic text in black
    draw.text((30, 38), bidi_text, font=font, fill=(0, 0, 0))

    img.save(out_img)
    return out_img


def main():
    print("=" * 80)
    print("        ARABIC HANDWRITING & INFORMAL FONT SURVEY (>=25 FONTS)")
    print("=" * 80)

    results = []
    print(f"Surveying {len(CANDIDATES)} fonts from Google Fonts...\n")

    for i, fam in enumerate(CANDIDATES, 1):
        print(f"[{i:02d}/{len(CANDIDATES)}] Processing {fam}...")
        try:
            ttf_path = download_font(fam)
            info = inspect_font(ttf_path)
            sample_img = render_sample(ttf_path, fam)
            results.append({
                "family": fam,
                "file": ttf_path.name,
                "license": info["license"],
                "diacritics": info["diacritics"],
                "has_gsub": info["has_gsub"],
                "shaping_ok": "Yes" if info["shaping_ok"] else "No",
                "sample_img": sample_img.name,
                "glyphs": info["glyph_count"],
            })
            print(f"   -> License: {info['license']} | Diacritics: {info['diacritics']} | Shapes: {'Yes' if info['shaping_ok'] else 'No'}")
        except Exception as e:
            print(f"   -> Failed {fam}: {e}")

    print("\n" + "=" * 80)
    print("                      FONT SURVEY SUMMARY TABLE")
    print("=" * 80)
    print(f"{'#':<3} | {'Font Family':<22} | {'License':<13} | {'Diacritics':<14} | {'Shaping':<8} | {'Sample PNG'}")
    print("-" * 80)
    for i, r in enumerate(results, 1):
        print(f"{i:<3} | {r['family']:<22} | {r['license']:<13} | {r['diacritics']:<14} | {r['shaping_ok']:<8} | {r['sample_img']}")

    # Write Markdown Report
    report_file = SAMPLES_DIR / "font_survey_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# Arabic Handwriting & Informal Font Survey (Google Fonts)\n\n")
        f.write(f"Total surveyed fonts: **{len(results)}** (Target: >=25)\n\n")
        f.write("| # | Font Family | License | Diacritic Support | OpenType Shaping | Sample File |\n")
        f.write("| :-: | :--- | :---: | :---: | :---: | :--- |\n")
        for i, r in enumerate(results, 1):
            f.write(f"| {i} | **{r['family']}** | {r['license']} | {r['diacritics']} | {r['shaping_ok']} | `{r['sample_img']}` |\n")

    print(f"\nWrote font survey report to {report_file}")
    print(f"Rendered {len(results)} sample images to {SAMPLES_DIR}/")


if __name__ == "__main__":
    main()
