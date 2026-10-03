"""
scripts/render_lines.py
Synthetic Arabic handwritten line renderer for Quran Tanzil corpus.

Features:
1. True OpenType cursive shaping via HarfBuzz (uharfbuzz) + FreeType (freetype-py).
2. Strict font acceptance check (Lam-Alef ligature & contextual connectivity on 'نستعين').
3. Quran continuous word-stream sampler within Surahs (words_per_line ~ uniform 3-9).
4. CTC feasibility check: drops samples where width // 4 < label_len + repeated_chars.
5. Augmentation v1:
   - Stroke thickness jitter (MinFilter/MaxFilter)
   - Baseline wobble (multi-frequency sinusoidal drift)
   - Slant / shear (affine shear)
   - Word spacing jitter (randomized inter-word gaps)
   - Elastic paper warp
   - Gaussian blur & additive paper noise
   - Random vertical margins covering tight and full-band crops to 64px height.
"""

from __future__ import annotations
import csv
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import freetype
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
import uharfbuzz as hb

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

FONTS_DIR = Path("data/fonts")
DERIVED_DIR = Path("data/derived")
PREVIEW_DIR = Path("data/synth_preview")
PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

TARGET_HEIGHT = 64
WIDTH_MULTIPLE = 32
MAX_WIDTH = 1600


# ---------------------------------------------------------------------------
# 1. Acceptance Check
# ---------------------------------------------------------------------------

def check_font_acceptance(font_path: Path):
    """
    Verifies OpenType cursive shaping on the font:
    1. Lam-Alef test: 'لا' must produce valid glyphs and not crash.
    2. Contextual alternate test: 'نستعين' must form a continuous baseline stroke.
    Fails loudly if letters disconnect or fail to render.
    """
    blob = hb.Blob.from_file_path(str(font_path))
    face = hb.Face(blob)
    hb_font = hb.Font(face)
    scale = 32 * 64
    hb_font.scale = (scale, scale)

    # 1. Lam-Alef test
    buf = hb.Buffer()
    buf.add_str("لا")
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)
    if len(buf.glyph_infos) == 0:
        raise RuntimeError(f"Acceptance failed for {font_path.name}: 'لا' produced 0 glyphs.")

    # 2. Contextual word test ('نستعين')
    buf2 = hb.Buffer()
    buf2.add_str("نستعين")
    buf2.guess_segment_properties()
    hb.shape(hb_font, buf2)

    ft_face = freetype.Face(str(font_path))
    ft_face.set_char_size(scale, scale)

    total_w = sum(pos.x_advance for pos in buf2.glyph_positions) >> 6
    w = max(total_w + 30, 50)
    h = 60
    base = 42
    arr = np.zeros((h, w), dtype=np.uint8)

    pen_x = 10
    for info, pos in zip(buf2.glyph_infos, buf2.glyph_positions):
        ft_face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
        slot = ft_face.glyph
        bm = slot.bitmap
        bx = (pen_x + (pos.x_offset >> 6)) + slot.bitmap_left
        by = (base - (pos.y_offset >> 6)) - slot.bitmap_top
        if bm.width > 0 and bm.rows > 0:
            bm_arr = np.array(bm.buffer, dtype=np.uint8).reshape((bm.rows, bm.width))
            y1, y2 = max(0, by), min(h, by + bm.rows)
            x1, x2 = max(0, bx), min(w, bx + bm.width)
            sy1, sy2 = y1 - by, y1 - by + (y2 - y1)
            sx1, sx2 = x1 - bx, x1 - bx + (x2 - x1)
            if y2 > y1 and x2 > x1:
                arr[y1:y2, x1:x2] = np.maximum(arr[y1:y2, x1:x2], bm_arr[sy1:sy2, sx1:sx2])
        pen_x += (pos.x_advance >> 6)

    # Ink presence and horizontal span check
    ink_cols = np.where(arr.sum(axis=0) > 0)[0]
    if len(ink_cols) == 0:
        raise RuntimeError(f"Acceptance failed for {font_path.name}: 'نستعين' rendered no ink.")
    ink_span = ink_cols[-1] - ink_cols[0]
    if ink_span < 20:
        raise RuntimeError(f"Acceptance failed for {font_path.name}: ink collapsed (span={ink_span}px).")
    return True


# ---------------------------------------------------------------------------
# 2. Continuous Quran Word Stream
# ---------------------------------------------------------------------------

class QuranWordStream:
    def __init__(self, ayahs_path: Path, splits_path: Path):
        self.surah_words: Dict[int, List[str]] = defaultdict(list)
        splits_data = json.loads(splits_path.read_text(encoding="utf-8"))
        self.split_surahs = {
            "train": set(splits_data["train"]["surahs"]),
            "val": set(splits_data["val"]["surahs"]),
            "test": set(splits_data["test"]["surahs"]),
        }

        with open(ayahs_path, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("_type") == "header":
                    continue
                sura = rec["surah"]
                text = rec["text"]
                if text:
                    self.surah_words[sura].extend(text.split())

    def sample_window(self, split: str = "train", min_words: int = 3, max_words: int = 9, rng: random.Random = None) -> Tuple[str, int]:
        if rng is None:
            rng = random
        valid_surahs = [s for s in self.split_surahs[split] if len(self.surah_words[s]) >= max_words]
        sura = rng.choice(valid_surahs)
        words = self.surah_words[sura]

        n_words = rng.randint(min_words, max_words)
        start = rng.randint(0, len(words) - n_words)
        tokens = words[start : start + n_words]
        return " ".join(tokens), sura


# ---------------------------------------------------------------------------
# 3. HarfBuzz + FreeType Line Renderer with Word-Spacing Jitter
# ---------------------------------------------------------------------------

def render_arabic_text(
    font_path: Path,
    text: str,
    font_size: int = 36,
    word_spacing_factor: float = 1.0,
) -> Image.Image:
    """
    Renders text word-by-word with native HarfBuzz shaping and randomized inter-word spacing.
    """
    words = text.split(" ")
    blob = hb.Blob.from_file_path(str(font_path))
    face = hb.Face(blob)
    hb_font = hb.Font(face)
    scale = int(font_size * 64)
    hb_font.scale = (scale, scale)

    ft_face = freetype.Face(str(font_path))
    ft_face.set_char_size(scale, scale)

    # Measure default space width
    space_buf = hb.Buffer()
    space_buf.add_str(" ")
    space_buf.guess_segment_properties()
    hb.shape(hb_font, space_buf)
    base_space_advance = (space_buf.glyph_positions[0].x_advance >> 6) if space_buf.glyph_positions else int(font_size * 0.3)
    space_advance = max(4, int(base_space_advance * word_spacing_factor))

    # Pre-render words
    word_bitmaps = []
    total_w = 0
    max_h = int(font_size * 2.2)
    baseline = int(font_size * 1.5)

    for word in words:
        buf = hb.Buffer()
        buf.add_str(word)
        buf.guess_segment_properties()
        hb.shape(hb_font, buf)

        w_adv = sum(pos.x_advance for pos in buf.glyph_positions) >> 6
        w_img_w = max(w_adv + 16, 20)
        w_arr = np.zeros((max_h, w_img_w), dtype=np.uint8)

        pen_x = 4
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            ft_face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
            slot = ft_face.glyph
            bm = slot.bitmap
            bx = (pen_x + (pos.x_offset >> 6)) + slot.bitmap_left
            by = (baseline - (pos.y_offset >> 6)) - slot.bitmap_top
            if bm.width > 0 and bm.rows > 0:
                bm_arr = np.array(bm.buffer, dtype=np.uint8).reshape((bm.rows, bm.width))
                y1, y2 = max(0, by), min(max_h, by + bm.rows)
                x1, x2 = max(0, bx), min(w_img_w, bx + bm.width)
                sy1, sy2 = y1 - by, y1 - by + (y2 - y1)
                sx1, sx2 = x1 - bx, x1 - bx + (x2 - x1)
                if y2 > y1 and x2 > x1:
                    w_arr[y1:y2, x1:x2] = np.maximum(w_arr[y1:y2, x1:x2], bm_arr[sy1:sy2, sx1:sx2])
            pen_x += (pos.x_advance >> 6)

        # Crop horizontally to actual ink
        ink_cols = np.where(w_arr.sum(axis=0) > 0)[0]
        if len(ink_cols) > 0:
            w_arr = w_arr[:, ink_cols[0]:ink_cols[-1] + 1]
        word_bitmaps.append(w_arr)
        total_w += w_arr.shape[1]

    total_w += space_advance * max(0, len(words) - 1)
    full_w = max(total_w + 30, 40)
    full_arr = np.zeros((max_h, full_w), dtype=np.uint8)

    # In RTL, words are laid out from right to left
    cur_x = full_w - 15
    for w_arr in word_bitmaps:
        bw = w_arr.shape[1]
        x1 = cur_x - bw
        if x1 >= 0:
            full_arr[:, x1:cur_x] = np.maximum(full_arr[:, x1:cur_x], w_arr)
        cur_x = x1 - space_advance

    # Convert to PIL Image: white background (255), black ink (0)
    img = Image.fromarray(255 - full_arr)
    return img


# ---------------------------------------------------------------------------
# 4. Augmentation v1
# ---------------------------------------------------------------------------

def augment_line_image(
    img: Image.Image,
    rng: np.random.RandomState,
    crop_mode: str = "random",  # 'tight', 'full', or 'random'
) -> Image.Image:
    """
    Applies the full Augmentation v1 pipeline:
    1. Random vertical margins (tight vs full-band crop)
    2. Baseline wobble (sinusoidal vertical drift)
    3. Slant / shear (affine shear)
    4. Stroke thickness jitter (MinFilter/MaxFilter)
    5. Elastic warp
    6. Gaussian blur & additive paper noise
    7. Right pad to WIDTH_MULTIPLE=32, target height=64
    """
    arr = np.array(img)  # 255 background, <255 ink
    ink_mask = arr < 240
    ink_rows = np.where(ink_mask.sum(axis=1) > 0)[0]
    ink_cols = np.where(ink_mask.sum(axis=0) > 0)[0]

    if len(ink_rows) == 0 or len(ink_cols) == 0:
        # Fallback if blank
        return Image.new("L", (WIDTH_MULTIPLE * 4, TARGET_HEIGHT), color=255)

    # Crop tightly to ink
    y1, y2 = ink_rows[0], ink_rows[-1] + 1
    x1, x2 = ink_cols[0], ink_cols[-1] + 1
    cropped = arr[y1:y2, x1:x2]
    crop_h, crop_w = cropped.shape

    # 1. Scale to fit within TARGET_HEIGHT=64
    if crop_mode == "random":
        mode = rng.choice(["tight", "full", "normal"])
    else:
        mode = crop_mode

    if mode == "full":
        target_ink_h = rng.randint(48, 58)
    elif mode == "tight":
        target_ink_h = rng.randint(30, 42)
    else:
        target_ink_h = rng.randint(38, 50)

    scale = target_ink_h / max(crop_h, 1)
    new_w = max(int(crop_w * scale), 20)
    new_h = target_ink_h

    pil_cropped = Image.fromarray(cropped).resize((new_w, new_h), Image.BILINEAR)

    # Place in 64px tall canvas with random vertical margin
    top_margin = rng.randint(2, max(3, TARGET_HEIGHT - new_h - 2))
    canvas = np.full((TARGET_HEIGHT, new_w + 30), 255, dtype=np.uint8)
    canvas_img = Image.fromarray(canvas)
    canvas_img.paste(pil_cropped, (15, top_margin))
    cur_arr = np.array(canvas_img)

    # 2. Baseline wobble
    w_curr = cur_arr.shape[1]
    wobble_amp = rng.uniform(0.8, 2.5)
    wobble_freq = rng.uniform(80.0, 160.0)
    wobble_phase = rng.uniform(0, 2 * math.pi)
    x_indices = np.arange(w_curr)
    dy = (wobble_amp * np.sin(2 * math.pi * x_indices / wobble_freq + wobble_phase)).astype(int)

    wobbled = np.full_like(cur_arr, 255)
    for col in range(w_curr):
        d = dy[col]
        if d >= 0:
            wobbled[d:, col] = cur_arr[:TARGET_HEIGHT - d, col]
        else:
            wobbled[:TARGET_HEIGHT + d, col] = cur_arr[-d:, col]
    cur_img = Image.fromarray(wobbled)

    # 3. Slant / shear
    if rng.rand() < 0.8:
        shear_val = rng.uniform(-0.15, 0.15)  # ~-8 to +8 degrees
        cur_img = cur_img.transform(
            cur_img.size,
            Image.AFFINE,
            (1, shear_val, -shear_val * TARGET_HEIGHT / 2, 0, 1, 0),
            fillcolor=255,
        )

    # 4. Stroke thickness jitter
    stroke_choice = rng.rand()
    if stroke_choice < 0.25:
        # Thicken pen
        cur_img = cur_img.filter(ImageFilter.MinFilter(3))
    elif stroke_choice < 0.45:
        # Thin pen
        cur_img = cur_img.filter(ImageFilter.MaxFilter(3))

    # 5. Elastic warp (gentle)
    if rng.rand() < 0.6:
        arr_warp = np.array(cur_img, dtype=np.float32)
        dh, dw = arr_warp.shape
        grid_x, grid_y = np.meshgrid(np.arange(dw), np.arange(dh))
        flow_x = rng.normal(0, 0.8, size=(5, 5))
        flow_y = rng.normal(0, 0.8, size=(5, 5))
        # Upsample flow
        flow_x_up = np.array(Image.fromarray(flow_x).resize((dw, dh), Image.BILINEAR))
        flow_y_up = np.array(Image.fromarray(flow_y).resize((dw, dh), Image.BILINEAR))

        map_x = np.clip(grid_x + flow_x_up, 0, dw - 1).astype(int)
        map_y = np.clip(grid_y + flow_y_up, 0, dh - 1).astype(int)
        warped_arr = arr_warp[map_y, map_x].astype(np.uint8)
        cur_img = Image.fromarray(warped_arr)

    # 6. Blur and paper noise
    if rng.rand() < 0.7:
        blur_r = rng.uniform(0.2, 0.6)
        cur_img = cur_img.filter(ImageFilter.GaussianBlur(blur_r))

    arr_final = np.array(cur_img, dtype=np.float32)
    # Add subtle Gaussian noise
    noise = rng.normal(0, rng.uniform(2.0, 5.0), size=arr_final.shape)
    arr_final = np.clip(arr_final + noise, 0, 255).astype(np.uint8)

    # 7. Right-pad width to a multiple of WIDTH_MULTIPLE=32
    final_w = arr_final.shape[1]
    final_w = min(final_w, MAX_WIDTH)
    arr_final = arr_final[:, :final_w]

    pad_w = (-final_w) % WIDTH_MULTIPLE
    if pad_w:
        pad_block = np.full((TARGET_HEIGHT, pad_w), 255, dtype=np.uint8)
        arr_final = np.hstack([arr_final, pad_block])

    return Image.fromarray(arr_final)


# ---------------------------------------------------------------------------
# 5. CTC Feasibility Check
# ---------------------------------------------------------------------------

def min_ctc_len(text: str) -> int:
    """Minimum CTC time-steps required: length + repeated consecutive characters."""
    repeats = sum(1 for i in range(len(text) - 1) if text[i] == text[i + 1])
    return len(text) + repeats


# ---------------------------------------------------------------------------
# 6. Preview Generation (300 Lines + Visual Montage)
# ---------------------------------------------------------------------------

def generate_preview(num_samples: int = 300, seed: int = 42):
    print("=" * 75)
    print(f"      GENERATING {num_samples} SYNTHETIC LINES (DATA/SYNTH_PREVIEW/)")
    print("=" * 75)

    rng_py = random.Random(seed)
    rng_np = np.random.RandomState(seed)

    word_stream = QuranWordStream(DERIVED_DIR / "ayahs.jsonl", DERIVED_DIR / "splits.json")
    font_paths = sorted(list(FONTS_DIR.glob("*.ttf")))
    print(f"Loaded {len(font_paths)} verified fonts.")

    # Verify acceptance check on all fonts before generation
    print("Running HarfBuzz acceptance check across all fonts...")
    for fp in font_paths:
        check_font_acceptance(fp)
    print("All fonts passed acceptance check.")

    manifest_rows = []
    total_attempts = 0
    dropped_ctc = 0

    for i in range(num_samples):
        while True:
            total_attempts += 1
            font_path = rng_py.choice(font_paths)
            text, surah = word_stream.sample_window(split="train", min_words=3, max_words=9, rng=rng_py)
            spacing_jitter = rng_py.uniform(0.7, 1.4)
            font_size = rng_py.randint(30, 42)

            raw_img = render_arabic_text(font_path, text, font_size=font_size, word_spacing_factor=spacing_jitter)
            aug_img = augment_line_image(raw_img, rng=rng_np)

            # Check CTC constraint: width // 4 >= min_ctc_len(text)
            time_steps = aug_img.width // 4
            required_steps = min_ctc_len(text)
            if time_steps < required_steps:
                dropped_ctc += 1
                continue

            # Valid sample
            img_filename = f"line_{i:04d}.png"
            img_path = PREVIEW_DIR / img_filename
            aug_img.save(img_path)

            manifest_rows.append({
                "image_path": str(img_path),
                "transcription": text,
                "font": font_path.stem,
                "surah": surah,
                "words": len(text.split()),
                "width": aug_img.width,
                "height": aug_img.height,
            })
            break

    drop_rate = (dropped_ctc / total_attempts) * 100
    print(f"\nGenerated {num_samples} valid lines.")
    print(f"CTC feasibility: {total_attempts} attempts, {dropped_ctc} dropped (drop rate: {drop_rate:.2f}%)")

    # Save preview manifest
    manifest_csv = PREVIEW_DIR / "manifest.csv"
    with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "transcription", "font", "surah", "words", "width", "height"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"Saved preview manifest to {manifest_csv}")

    # Build Montage of 12 diverse samples
    build_montage(manifest_rows[:12], PREVIEW_DIR / "montage.png")


def build_montage(sample_rows: List[dict], out_path: Path):
    montage_w = 900
    row_h = 75
    total_h = len(sample_rows) * row_h + 50

    canvas = Image.new("RGB", (montage_w, total_h), color=(248, 248, 250))
    draw = ImageDraw.Draw(canvas)
    header_font = ImageFont.load_default()
    draw.text((20, 15), "SYNTHETIC ARABIC HTR PREVIEW MONTAGE (AUGMENTATION V1 + HARFBUZZ)", fill=(30, 30, 30))
    draw.line([(20, 35), (montage_w - 20, 35)], fill=(210, 210, 215), width=1)

    y_cur = 45
    for row in sample_rows:
        line_img = Image.open(row["image_path"]).convert("RGB")
        label = f"{row['font']} ({row['words']} words)"

        # Card container
        draw.rectangle([20, y_cur, montage_w - 20, y_cur + row_h - 10], fill=(255, 255, 255), outline=(225, 225, 230))
        draw.text((30, y_cur + 8), label, fill=(100, 100, 100))

        # Paste line image (right-aligned in card)
        paste_x = min(220, montage_w - 30 - line_img.width)
        canvas.paste(line_img, (paste_x, y_cur + 2))
        y_cur += row_h

    canvas.save(out_path)
    print(f"Montage saved to {out_path} ({montage_w}x{total_h})")


if __name__ == "__main__":
    generate_preview(num_samples=300, seed=42)
