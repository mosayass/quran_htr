"""
scripts/make_v3_montage.py
Task 2.13 Item 2:
Creates data/synth_preview_v3/montage.png:
24 labeled samples from Corpus v3 train distribution:
- 8 one-word samples
- 8 full-diacritics samples
- 8 light-diacritics samples
Displayed at both actual 64px scale and 3x zoom (192px height) with font & text labels.
"""

from __future__ import annotations
import random
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from render_lines import render_arabic_text, FONTS_DIR
from generate_corpus_v3 import (
    AlignedQuranWordStream, make_light_diacritics, augment_line_relative_thickness,
    check_line_metrics_fast, check_stroke_width_only
)

CONFIG_PATH = ROOT / "configs" / "synth.yaml"
OUT_MONTAGE = ROOT / "data" / "synth_preview_v3" / "montage.png"


def main():
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    sw_bounds = cfg["calibration_bounds"]["stroke_width"]
    fr_bounds = cfg["calibration_bounds"]["ink_fraction"]
    sw_min, sw_max = sw_bounds["min"], sw_bounds["max"]
    fr_min, fr_max = fr_bounds["min"], 0.160

    stream = AlignedQuranWordStream(
        ROOT / "data" / "tanzil" / "quran-simple-clean.txt",
        ROOT / "data" / "tanzil" / "quran-simple.txt",
        ROOT / "data" / "derived" / "splits.json"
    )

    all_fonts = sorted(list(FONTS_DIR.glob("*.ttf")))
    held_out = set(cfg["fonts"].get("held_out", []))
    weights_map = cfg["fonts"]["weights"]
    train_fonts = [f for f in all_fonts if weights_map.get(f.stem, 0.0) > 0 and f.stem not in held_out]

    rng_py = random.Random(42)
    rng_np = np.random.RandomState(42)

    categories = [
        ("1-Word Samples", "one_word", 8),
        ("Full Diacritics", "full", 8),
        ("Light Diacritics", "light", 8),
    ]

    samples = []

    for cat_title, mode, count in categories:
        cat_samples = []
        font_idx = 0
        attempts = 0
        while len(cat_samples) < count and attempts < 200:
            attempts += 1
            fp = train_fonts[font_idx % len(train_fonts)]
            font_idx += 1

            if mode == "one_word":
                n_words = 1
                diac_choice = rng_py.choice(["none", "light", "full"])
            elif mode == "full":
                n_words = rng_py.randint(3, 7)
                diac_choice = "full"
            else: # light
                n_words = rng_py.randint(3, 7)
                diac_choice = "light"

            clean_text, vowelled_text, sura = stream.sample_window(
                split="train", n_words=n_words, rng=rng_py
            )

            if diac_choice == "none":
                render_text = clean_text
            elif diac_choice == "light":
                render_text = make_light_diacritics(vowelled_text, rng_py)
            else:
                render_text = vowelled_text

            f_size = rng_py.randint(32, 40)
            spacing = rng_py.uniform(0.8, 1.3)

            raw = render_arabic_text(fp, render_text, font_size=f_size, word_spacing_factor=spacing)
            ink_tight = rng_np.rand() < 0.20
            aug = augment_line_relative_thickness(raw, rng=rng_np, apply_noise=(rng_np.rand() < 0.60), ink_tight=ink_tight)

            aug_arr = np.array(aug)
            if n_words in (1, 2):
                if not check_stroke_width_only(aug_arr, sw_min, sw_max):
                    continue
            else:
                if not check_line_metrics_fast(aug_arr, sw_min, sw_max, fr_min, fr_max):
                    continue

            cat_samples.append({
                "category": cat_title,
                "mode": diac_choice,
                "font": fp.stem,
                "words": n_words,
                "clean": clean_text,
                "img": aug,
            })
        samples.extend(cat_samples)

    print(f"Generated {len(samples)} montage samples.")

    # Compose Montage Canvas
    # Each item has:
    # 1. Actual 64px scale
    # 2. 3x Zoomed crop (192px height)
    # 3. Label text
    col_w = 1100
    row_h = 240
    header_h = 80
    total_h = header_h + len(samples) * row_h + 40

    canvas = Image.new("RGB", (col_w, total_h), color=(250, 252, 255))
    draw = ImageDraw.Draw(canvas)

    # Title
    draw.rectangle([(0, 0), (col_w, header_h)], fill=(15, 23, 42))
    draw.text((30, 18), "CORPUS V3 SYNTHETIC PREVIEW MONTAGE (24 SAMPLES)", fill=(56, 189, 248))
    draw.text((30, 45), "Top: 1:1 Actual Scale (64px)  |  Bottom: 3x Nearest-Neighbor Zoom (192px) with Diacritic Placement", fill=(148, 163, 184))

    y_cur = header_h + 20
    for idx, s in enumerate(samples, 1):
        # Section divider
        draw.rectangle([(20, y_cur), (col_w - 20, y_cur + row_h - 15)], fill=(255, 255, 255), outline=(226, 232, 240))

        # Metadata Label
        meta_str = f"#{idx:02d} [{s['category']}] Font: {s['font']} | Words: {s['words']} | Diac: {s['mode']}\nText: {s['clean']}"
        draw.text((35, y_cur + 10), meta_str, fill=(30, 41, 59))

        # Actual 64px image
        img_64 = s["img"].convert("RGB")
        canvas.paste(img_64, (35, y_cur + 55))

        # 3x Zoom crop (take central or right-aligned snippet of width 240px, zoomed to 720x192)
        crop_w = min(img_64.width, 260)
        # For RTL, take rightmost part of ink
        x_start = max(0, img_64.width - crop_w)
        snippet = img_64.crop((x_start, 0, x_start + crop_w, 64))
        zoomed = snippet.resize((crop_w * 3, 192), Image.Resampling.NEAREST)

        # Scale down zoom if exceeds canvas width
        if zoomed.width > (col_w - 70):
            zoomed = zoomed.crop((0, 0, col_w - 70, 192))

        # Paste zoom below actual
        # To keep row_h compact, draw zoom on right side or below:
        # Layout: Actual 64px on left, 3x Zoom on right!
        # Actual at (35, y_cur + 60), Zoom at (450, y_cur + 20)
        # Let's adjust layout:
        # Actually row_h = 240 fits zoom at (400, y_cur + 20)
        y_cur += row_h

    # Re-draw with refined side-by-side layout
    canvas = Image.new("RGB", (1280, header_h + len(samples) * 230 + 30), color=(248, 250, 252))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([(0, 0), (1280, header_h)], fill=(15, 23, 42))
    draw.text((30, 16), "CORPUS V3 SYNTHETIC PREVIEW MONTAGE (24 SAMPLES)", fill=(56, 189, 248))
    draw.text((30, 44), "Left: Actual 64px Scale + Info  |  Right: 3x Zoom (192px Height) for Stroke & Diacritic Inspection", fill=(148, 163, 184))

    y = header_h + 15
    for idx, s in enumerate(samples, 1):
        draw.rectangle([(20, y), (1260, y + 215)], fill=(255, 255, 255), outline=(226, 232, 240))

        # Left Info
        draw.text((35, y + 15), f"#{idx:02d} - {s['category'].upper()}", fill=(14, 165, 233))
        draw.text((35, y + 35), f"Font: {s['font']}", fill=(15, 23, 42))
        draw.text((35, y + 55), f"Words: {s['words']} | Diac: {s['mode']}", fill=(100, 116, 139))
        draw.text((35, y + 75), f"Rasm: {s['clean'][:45]}", fill=(71, 85, 105))

        # Left Actual 64px
        img_64 = s["img"].convert("RGB")
        disp_w = min(img_64.width, 420)
        disp_img = img_64.crop((0, 0, disp_w, 64))
        canvas.paste(disp_img, (35, y + 115))
        draw.rectangle([(35, y + 115), (35 + disp_w, y + 179)], outline=(203, 213, 225))

        # Right 3x Zoom
        crop_w = min(img_64.width, 240)
        snippet = img_64.crop((0, 0, crop_w, 64))
        zoomed = snippet.resize((crop_w * 3, 192), Image.Resampling.NEAREST)
        max_zoom_w = 750
        if zoomed.width > max_zoom_w:
            zoomed = zoomed.crop((0, 0, max_zoom_w, 192))
        canvas.paste(zoomed, (490, y + 12))
        draw.rectangle([(490, y + 12), (490 + zoomed.width, y + 204)], outline=(203, 213, 225))

        y += 230

    OUT_MONTAGE.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_MONTAGE)
    print(f"Saved verified 24-sample montage to {OUT_MONTAGE} ({OUT_MONTAGE.stat().st_size / (1024*1024):.2f} MB)")


if __name__ == "__main__":
    main()
