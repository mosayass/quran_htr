"""
scripts/generate_diac_test_sets.py
Task 2.11 Step 2:
1. Verify word counts match between Tanzil simple-clean and simple (pause marks stripped).
2. Generate synth_test_diac (2,000 lines) and synth_test_unseen_fonts_diac (2,000 lines)
   from the test surahs:
   - Rendered text: vowelled (simple variant with pause marks stripped).
   - Labels: simple-clean rasm-only.
   - Shards saved to:
     data/shards/synth/synth_test_diac_000.pt
     data/shards/synth/synth_test_unseen_fonts_diac_000.pt
"""

from __future__ import annotations
import json
import math
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt, maximum_filter
import torch
import uharfbuzz as hb
import freetype
import yaml

ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = ROOT / "data" / "fonts"
DERIVED_DIR = ROOT / "data" / "derived"
CONFIG_PATH = ROOT / "configs" / "synth.yaml"
SHARDS_OUT_DIR = ROOT / "data" / "shards" / "synth"
CONTACT_SHEET_DIR = ROOT / "data" / "font_samples"

PAUSE_MARKS_RE = re.compile(r"[\u0615-\u061a\u06d6-\u06ed\u06e9]")


def min_ctc_len(text: str) -> int:
    n = len(text)
    dups = sum(1 for i in range(1, len(text)) if text[i] == text[i - 1])
    return n + dups


def render_arabic_text(
    font_path: Path,
    text: str,
    font_size: int = 34,
    word_spacing_factor: float = 1.0,
) -> Image.Image:
    words = text.split(" ")
    blob = hb.Blob.from_file_path(str(font_path))
    face = hb.Face(blob)
    hb_font = hb.Font(face)
    scale = int(font_size * 64)
    hb_font.scale = (scale, scale)

    ft_face = freetype.Face(str(font_path))
    ft_face.set_char_size(scale, scale)

    space_buf = hb.Buffer()
    space_buf.add_str(" ")
    space_buf.guess_segment_properties()
    hb.shape(hb_font, space_buf)
    base_space_advance = (space_buf.glyph_positions[0].x_advance >> 6) if space_buf.glyph_positions else int(font_size * 0.3)
    space_advance = max(4, int(base_space_advance * word_spacing_factor))

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

        ink_cols = np.where(w_arr.sum(axis=0) > 0)[0]
        if len(ink_cols) > 0:
            w_arr = w_arr[:, ink_cols[0]:ink_cols[-1] + 1]
        word_bitmaps.append(w_arr)
        total_w += w_arr.shape[1]

    total_w += space_advance * max(0, len(words) - 1)
    full_w = max(total_w + 30, 40)
    full_arr = np.zeros((max_h, full_w), dtype=np.uint8)

    cur_x = full_w - 15
    for w_arr in word_bitmaps:
        bw = w_arr.shape[1]
        x1 = cur_x - bw
        if x1 >= 0:
            full_arr[:, x1:cur_x] = np.maximum(full_arr[:, x1:cur_x], w_arr)
        cur_x = x1 - space_advance

    img = Image.fromarray(255 - full_arr)
    return img


def check_line_metrics_fast(
    arr: np.ndarray,
    sw_min: float,
    sw_max: float,
    fr_min: float,
    fr_max: float,
) -> bool:
    ink = arr < 128
    cols = np.where(ink.sum(axis=0) > 0)[0]
    if len(cols) == 0:
        return False
    span_w = cols[-1] - cols[0] + 1
    ink_frac = float(ink.sum() / (64 * span_w))
    if ink_frac < fr_min or ink_frac > fr_max:
        return False

    mid = (cols[0] + cols[-1]) // 2
    crop_w = 160
    c_start = max(cols[0], mid - crop_w // 2)
    c_end = min(cols[-1] + 1, c_start + crop_w)
    crop = ink[:, c_start:c_end]
    if not crop.any():
        return False

    dist = distance_transform_edt(crop)
    loc_max = (dist == maximum_filter(dist, size=3)) & (dist > 0.5)
    if not loc_max.any():
        return False

    sw = float(np.median(2.0 * dist[loc_max]))
    return (sw >= sw_min) and (sw <= sw_max)


def augment_line_relative_thickness(
    img: Image.Image,
    rng: np.random.RandomState,
    apply_noise: bool = True,
) -> Image.Image:
    arr = np.array(img)
    ink_mask = arr < 240
    ink_rows = np.where(ink_mask.sum(axis=1) > 0)[0]
    ink_cols = np.where(ink_mask.sum(axis=0) > 0)[0]

    if len(ink_rows) > 0 and len(ink_cols) > 0:
        top, bot = ink_rows[0], ink_rows[-1]
        left, right = ink_cols[0], ink_cols[-1]
        ink_h = bot - top + 1
        ink_w = right - left + 1

        pad_top = rng.randint(4, 16)
        pad_bot = rng.randint(4, 16)
        y1 = max(0, top - pad_top)
        y2 = min(arr.shape[0], bot + pad_bot + 1)
        x1 = max(0, left - 12)
        x2 = min(arr.shape[1], right + 12 + 1)
        crop = img.crop((x1, y1, x2, y2))
    else:
        crop = img

    target_h = 64
    w, h = crop.size
    scaled_w = max(32, int(round(w * (target_h / float(h)))))
    resized = crop.resize((scaled_w, target_h), Image.Resampling.BILINEAR)

    padded_w = math.ceil(scaled_w / 32) * 32
    canvas = Image.new("L", (padded_w, target_h), color=255)
    canvas.paste(resized, (0, 0))

    if not apply_noise:
        return canvas

    cur_arr = np.array(canvas, dtype=np.float32)
    # Subtle thickness jitter
    r = rng.rand()
    if r < 0.20:
        filt = maximum_filter(cur_arr, size=(2, 2))
        cur_arr = np.minimum(cur_arr, filt)
    elif r < 0.40:
        filt = distance_transform_edt(cur_arr < 128)
        cur_arr = np.clip(cur_arr + (rng.rand() * 15 - 5), 0, 255)

    # Background subtle gradient/grain
    noise = rng.normal(0, 3.5, cur_arr.shape)
    cur_arr = np.clip(cur_arr + noise, 0, 255).astype(np.uint8)

    cur_img = Image.fromarray(cur_arr)
    if rng.rand() < 0.25:
        cur_img = cur_img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.6)))
    return cur_img


def compute_font_quotas(font_paths, weights_map, total_lines):
    tot_weight = sum(weights_map.get(fp.stem, 0.0) for fp in font_paths)
    quotas = {}
    for fp in font_paths:
        w = weights_map.get(fp.stem, 0.0)
        quotas[fp] = round(total_lines * (w / tot_weight))
    diff = total_lines - sum(quotas.values())
    quotas[list(quotas.keys())[0]] += diff
    return quotas


class AlignedQuranWordStream:
    def __init__(self, clean_path: Path, simple_path: Path, splits_path: Path):
        self.surah_clean: Dict[int, List[str]] = defaultdict(list)
        self.surah_vowelled: Dict[int, List[str]] = defaultdict(list)

        with open(clean_path, encoding="utf-8") as f:
            clean_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
        with open(simple_path, encoding="utf-8") as f:
            simple_lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

        surah_map = []
        with open(DERIVED_DIR / "ayahs.jsonl", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("_type") == "header":
                    continue
                surah_map.append(rec["surah"])

        mismatches = 0
        for s_num, c_ayah, s_ayah in zip(surah_map, clean_lines, simple_lines):
            s_clean_marks = PAUSE_MARKS_RE.sub("", s_ayah)
            c_words = c_ayah.split()
            v_words = s_clean_marks.split()
            if len(c_words) != len(v_words):
                mismatches += 1
                continue
            self.surah_clean[s_num].extend(c_words)
            self.surah_vowelled[s_num].extend(v_words)

        self.mismatches = mismatches
        splits_data = json.loads(splits_path.read_text(encoding="utf-8"))
        self.split_surahs = {
            "train": set(splits_data["train"]["surahs"]),
            "val": set(splits_data["val"]["surahs"]),
            "test": set(splits_data["test"]["surahs"]),
        }

    def sample_window(self, split: str = "test", min_words: int = 3, max_words: int = 9, rng: random.Random = None) -> Tuple[str, str, int]:
        if rng is None:
            rng = random
        valid_surahs = [s for s in self.split_surahs[split] if len(self.surah_clean[s]) >= max_words]
        sura = rng.choice(valid_surahs)
        c_words = self.surah_clean[sura]
        v_words = self.surah_vowelled[sura]

        n_words = rng.randint(min_words, max_words)
        start = rng.randint(0, len(c_words) - n_words)
        clean_text = " ".join(c_words[start : start + n_words])
        vowelled_text = " ".join(v_words[start : start + n_words])
        return clean_text, vowelled_text, sura


_STREAM: AlignedQuranWordStream | None = None

def _get_stream():
    global _STREAM
    if _STREAM is None:
        _STREAM = AlignedQuranWordStream(
            ROOT / "data" / "tanzil" / "quran-simple-clean.txt",
            ROOT / "data" / "tanzil" / "quran-simple.txt",
            DERIVED_DIR / "splits.json"
        )
    return _STREAM


def generate_diac_split(
    split_name: str,
    target_count: int,
    font_quotas: Dict[Path, int],
    cfg: dict,
    out_shard_path: Path,
    seed: int = 42,
):
    print(f"\n{'='*75}\nGenerating diacritic set '{split_name}': {target_count} lines...")
    sw_bounds = cfg["calibration_bounds"]["stroke_width"]
    fr_bounds = cfg["calibration_bounds"]["ink_fraction"]
    sw_min, sw_max = sw_bounds["min"], sw_bounds["max"]
    fr_min, fr_max = fr_bounds["min"], 0.160  # Relaxed upper ink fraction to accommodate diacritic marks
    max_w = cfg["constraints"]["max_width_px"]

    stream = _get_stream()
    rng_py = random.Random(seed)
    rng_np = np.random.RandomState(seed)

    all_images = []
    all_texts = []
    all_fonts = []
    all_surahs = []
    total_attempts = 0
    total_rejects = 0

    t0 = time.time()
    for fp, quota in font_quotas.items():
        if quota <= 0:
            continue
        count_font = 0
        attempts_this_font = 0
        max_attempts_font = max(quota * 15, 100)
        while count_font < quota and attempts_this_font < max_attempts_font:
            total_attempts += 1
            attempts_this_font += 1
            n_words = rng_py.randint(3, 9)
            clean_text, vowelled_text, sura = stream.sample_window(
                split="test", min_words=n_words, max_words=n_words, rng=rng_py
            )

            spacing = rng_py.uniform(0.7, 1.4)
            f_size = rng_py.randint(30, 42)

            raw = render_arabic_text(fp, vowelled_text, font_size=f_size, word_spacing_factor=spacing)

            apply_noise = rng_np.rand() < 0.60
            aug = augment_line_relative_thickness(raw, rng=rng_np, apply_noise=apply_noise)

            if aug.width >= max_w or (aug.width // 4 < min_ctc_len(clean_text)):
                total_rejects += 1
                continue

            aug_arr = np.array(aug)
            if not check_line_metrics_fast(aug_arr, sw_min, sw_max, fr_min, fr_max):
                total_rejects += 1
                continue

            all_images.append(torch.from_numpy(aug_arr))
            all_texts.append(clean_text)  # Strict simple-clean rasm-only label
            all_fonts.append(fp.stem)
            all_surahs.append(sura)
            count_font += 1

        print(f"  [Progress] Font {fp.stem:22s}: {count_font}/{quota} lines generated")

    combined = list(zip(all_images, all_texts, all_fonts, all_surahs))
    random.Random(seed).shuffle(combined)
    all_images, all_texts, all_fonts, all_surahs = zip(*combined)

    out_shard_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "images": list(all_images),
        "texts": list(all_texts),
        "fonts": list(all_fonts),
        "surahs": list(all_surahs),
        "count": len(all_images),
    }, out_shard_path)

    elapsed = max(time.time() - t0, 1e-4)
    throughput = len(all_images) / elapsed
    rej_pct = (total_rejects / total_attempts) * 100 if total_attempts else 0.0
    sz_mb = out_shard_path.stat().st_size / (1024 * 1024)
    print(f"Split '{split_name}' saved to {out_shard_path.name}: {len(all_images)} lines ({sz_mb:.1f} MB) in {elapsed:.1f}s ({throughput:.1f} l/s) | Rejects: {rej_pct:.1f}%")


def main():
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    weights_map = cfg["fonts"]["weights"]
    held_out = set(cfg["fonts"]["held_out"])

    all_fonts = sorted(list(FONTS_DIR.glob("*.ttf")))
    train_fonts = [f for f in all_fonts if weights_map.get(f.stem, 0.0) > 0 and f.stem not in held_out]
    held_out_fonts = [f for f in all_fonts if f.stem in held_out]

    # Verify alignment
    stream = _get_stream()
    print(f"[Alignment Check] Word mismatches across Quran: {stream.mismatches}")

    # 1. synth_test_diac (2,000 lines from test surahs, in-corpus fonts)
    test_quotas = compute_font_quotas(train_fonts, weights_map, 2000)
    out_synth_diac = SHARDS_OUT_DIR / "synth_test_diac_000.pt"
    generate_diac_split("synth_test_diac", 2000, test_quotas, cfg, out_synth_diac, seed=500)

    # 2. synth_test_unseen_fonts_diac (2,000 lines: 1,000 Zain, 1,000 Mirza)
    unseen_quotas = {fp: 1000 for fp in held_out_fonts}
    out_unseen_diac = SHARDS_OUT_DIR / "synth_test_unseen_fonts_diac_000.pt"
    generate_diac_split("synth_test_unseen_fonts_diac", 2000, unseen_quotas, cfg, out_unseen_diac, seed=600)

    print("\n" + "=" * 75)
    print("TASK 2.11 STEP 2 COMPLETE: DIACRITIC TEST SETS GENERATED")
    print(f"  - synth_test_diac:              {out_synth_diac} ({out_synth_diac.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  - synth_test_unseen_fonts_diac: {out_unseen_diac} ({out_unseen_diac.stat().st_size / (1024*1024):.1f} MB)")
    print("=" * 75)


if __name__ == "__main__":
    main()
