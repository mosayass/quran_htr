"""
scripts/render_lines.py
Synthetic Arabic handwritten line renderer for Quran Tanzil corpus.

Features:
1. True OpenType cursive shaping via HarfBuzz (uharfbuzz) + FreeType (freetype-py).
2. Connected-component font acceptance check (dots removed for 'نستعين' and ligature check for 'لا').
3. Config-driven generation (configs/synth.yaml): per-font weights, held-out list,
   words_per_line range, augmentation strengths.
4. Refuse/drop lines whose width after 64px scaling is >= 1500px; log drop rate.
5. CTC feasibility check: drops samples where width // 4 < label_len + repeated_chars.
6. Manifest CSV output in dataset.py format (image_path, transcription), split by
   data/derived/splits.json.
7. Measures and reports generation throughput in lines/sec.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Any

import freetype
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont
from scipy.ndimage import label as nd_label
import uharfbuzz as hb
import yaml

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

FONTS_DIR = Path("data/fonts")
DERIVED_DIR = Path("data/derived")
CONFIG_PATH = Path("configs/synth.yaml")
SYNTH_DIR = Path("data/synth")
PREVIEW_DIR = Path("data/synth_preview")

TARGET_HEIGHT = 64
WIDTH_MULTIPLE = 32
DEFAULT_MAX_WIDTH = 1500


# ---------------------------------------------------------------------------
# 1. Connected-Component Acceptance Check (Task 2.3 Item 1)
# ---------------------------------------------------------------------------

def _render_word_mask(font_path: Path, word: str, font_size: int = 36) -> np.ndarray:
    """Helper to render a single word to a uint8 binary ink mask using HarfBuzz + FreeType."""
    blob = hb.Blob.from_file_path(str(font_path))
    face = hb.Face(blob)
    hb_font = hb.Font(face)
    scale = int(font_size * 64)
    hb_font.scale = (scale, scale)

    ft_face = freetype.Face(str(font_path))
    ft_face.set_char_size(scale, scale)

    buf = hb.Buffer()
    buf.add_str(word)
    buf.guess_segment_properties()
    hb.shape(hb_font, buf)

    total_w = sum(pos.x_advance for pos in buf.glyph_positions) >> 6
    w = max(total_w + 40, 50)
    h = 80
    base = 55
    arr = np.zeros((h, w), dtype=np.uint8)

    pen_x = 15
    for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
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
    return arr


def check_font_acceptance(font_path: Path) -> Tuple[bool, Dict[str, Any]]:
    """
    Connected-component test (dots removed) for 'نستعين' and 'لا':
    - 'لا': Must form 1 connected component (no disjoint strokes).
    - 'نستعين': Baseline stroke must form 1 connected component when dots are filtered out.
    Returns (passed, details).
    """
    details: Dict[str, Any] = {"font": font_path.stem}

    # 1. 'لا' ligature test
    arr_la = _render_word_mask(font_path, "لا") > 30
    lbl_la, num_la = nd_label(arr_la, structure=np.ones((3, 3)))
    details["la_components"] = num_la

    # 2. 'نستعين' contextual connectivity test (dots removed)
    arr_nas = _render_word_mask(font_path, "نستعين") > 30
    lbl_nas, num_nas = nd_label(arr_nas, structure=np.ones((3, 3)))
    details["nastaeen_total_components"] = num_nas

    sizes = [int(np.sum(lbl_nas == i)) for i in range(1, num_nas + 1)]
    if not sizes:
        details["nastaeen_base"] = 0
        details["passed"] = False
        details["reason"] = "No ink rendered for 'نستعين'"
        return False, details

    max_size = max(sizes)
    # Dots are small components (< 0.20 * max_size)
    base_components = [s for s in sizes if s >= 0.20 * max_size]
    num_base = len(base_components)
    details["nastaeen_base"] = num_base
    details["component_sizes"] = sizes

    reasons = []
    if num_la != 1:
        reasons.append(f"'لا' produced {num_la} components (expected 1 continuous ligature)")
    if num_base != 1:
        reasons.append(f"'نستعين' base stroke has {num_base} components with dots removed (expected 1)")

    passed = len(reasons) == 0
    details["passed"] = passed
    details["reason"] = "; ".join(reasons) if reasons else "Passed all connected-component checks"
    return passed, details


def audit_all_fonts(font_paths: List[Path]) -> List[Dict[str, Any]]:
    """Runs acceptance check across all fonts and reports results without dropping any."""
    print("=" * 75)
    print("       FONT ACCEPTANCE AUDIT: CONNECTED COMPONENTS (DOTS REMOVED)")
    print("=" * 75)
    results = []
    for fp in font_paths:
        passed, details = check_font_acceptance(fp)
        results.append(details)
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {fp.stem:28s} | 'لا': {details['la_components']} comp | "
              f"'نستعين' base: {details['nastaeen_base']} comp (total {details['nastaeen_total_components']})")
        if not passed:
            print(f"       -> Details: {details['reason']}")

    fails = [r for r in results if not r["passed"]]
    print("-" * 75)
    print(f"Audit Summary: {len(results) - len(fails)}/28 fonts passed. {len(fails)} font(s) failed.")
    for f in fails:
        print(f"  * Reported failure: {f['font']} -- {f['reason']}")
    print("Notice: As required by Task 2.3 Item 1, failing fonts are reported but NOT dropped.")
    print("=" * 75)
    return results


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


# ---------------------------------------------------------------------------
# 4. Augmentation v1 with Config Parameters
# ---------------------------------------------------------------------------

def augment_line_image(
    img: Image.Image,
    rng: np.random.RandomState,
    crop_mode: str = "random",
    aug_cfg: dict | None = None,
) -> Image.Image:
    """Applies Augmentation v1 parameterized by configs/synth.yaml."""
    if aug_cfg is None:
        aug_cfg = {}

    wobble_cfg = aug_cfg.get("baseline_wobble", {})
    shear_cfg = aug_cfg.get("shear", {})
    thickness_cfg = aug_cfg.get("stroke_thickness", {})
    warp_cfg = aug_cfg.get("elastic_warp", {})
    blur_cfg = aug_cfg.get("blur", {})
    noise_cfg = aug_cfg.get("noise", {})

    arr = np.array(img)
    ink_mask = arr < 240
    ink_rows = np.where(ink_mask.sum(axis=1) > 0)[0]
    ink_cols = np.where(ink_mask.sum(axis=0) > 0)[0]

    if len(ink_rows) == 0 or len(ink_cols) == 0:
        return Image.new("L", (WIDTH_MULTIPLE * 4, TARGET_HEIGHT), color=255)

    y1, y2 = ink_rows[0], ink_rows[-1] + 1
    x1, x2 = ink_cols[0], ink_cols[-1] + 1
    cropped = arr[y1:y2, x1:x2]
    crop_h, crop_w = cropped.shape

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

    top_margin = rng.randint(2, max(3, TARGET_HEIGHT - new_h - 2))
    canvas = np.full((TARGET_HEIGHT, new_w + 30), 255, dtype=np.uint8)
    canvas_img = Image.fromarray(canvas)
    canvas_img.paste(pil_cropped, (15, top_margin))
    cur_arr = np.array(canvas_img)

    # 1. Baseline wobble
    if wobble_cfg.get("enabled", True):
        w_curr = cur_arr.shape[1]
        amp_min = wobble_cfg.get("amp_min", 0.8)
        amp_max = wobble_cfg.get("amp_max", 2.5)
        freq_min = wobble_cfg.get("freq_min", 80.0)
        freq_max = wobble_cfg.get("freq_max", 160.0)
        wobble_amp = rng.uniform(amp_min, amp_max)
        wobble_freq = rng.uniform(freq_min, freq_max)
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
    else:
        cur_img = canvas_img

    # 2. Slant / shear
    if shear_cfg.get("enabled", True) and rng.rand() < shear_cfg.get("prob", 0.8):
        max_sh = shear_cfg.get("max_shear", 0.15)
        shear_val = rng.uniform(-max_sh, max_sh)
        cur_img = cur_img.transform(
            cur_img.size,
            Image.AFFINE,
            (1, shear_val, -shear_val * TARGET_HEIGHT / 2, 0, 1, 0),
            fillcolor=255,
        )

    # 3. Stroke thickness jitter
    if thickness_cfg.get("enabled", True):
        stroke_choice = rng.rand()
        thicken_p = thickness_cfg.get("thicken_prob", 0.25)
        thin_p = thickness_cfg.get("thin_prob", 0.20)
        if stroke_choice < thicken_p:
            cur_img = cur_img.filter(ImageFilter.MinFilter(3))
        elif stroke_choice < (thicken_p + thin_p):
            cur_img = cur_img.filter(ImageFilter.MaxFilter(3))

    # 4. Elastic warp
    if warp_cfg.get("enabled", True) and rng.rand() < warp_cfg.get("prob", 0.6):
        arr_warp = np.array(cur_img, dtype=np.float32)
        dh, dw = arr_warp.shape
        grid_x, grid_y = np.meshgrid(np.arange(dw), np.arange(dh))
        sigma = warp_cfg.get("sigma", 0.8)
        flow_x = rng.normal(0, sigma, size=(5, 5))
        flow_y = rng.normal(0, sigma, size=(5, 5))
        flow_x_up = np.array(Image.fromarray(flow_x).resize((dw, dh), Image.BILINEAR))
        flow_y_up = np.array(Image.fromarray(flow_y).resize((dw, dh), Image.BILINEAR))

        map_x = np.clip(grid_x + flow_x_up, 0, dw - 1).astype(int)
        map_y = np.clip(grid_y + flow_y_up, 0, dh - 1).astype(int)
        warped_arr = arr_warp[map_y, map_x].astype(np.uint8)
        cur_img = Image.fromarray(warped_arr)

    # 5. Blur and paper noise
    if blur_cfg.get("enabled", True) and rng.rand() < blur_cfg.get("prob", 0.7):
        rmin = blur_cfg.get("radius_min", 0.2)
        rmax = blur_cfg.get("radius_max", 0.6)
        blur_r = rng.uniform(rmin, rmax)
        cur_img = cur_img.filter(ImageFilter.GaussianBlur(blur_r))

    arr_final = np.array(cur_img, dtype=np.float32)
    if noise_cfg.get("enabled", True) and rng.rand() < noise_cfg.get("prob", 1.0):
        smin = noise_cfg.get("std_min", 2.0)
        smax = noise_cfg.get("std_max", 5.0)
        noise = rng.normal(0, rng.uniform(smin, smax), size=arr_final.shape)
        arr_final = np.clip(arr_final + noise, 0, 255).astype(np.uint8)
    else:
        arr_final = arr_final.astype(np.uint8)

    # 6. Right-pad width to multiple of WIDTH_MULTIPLE=32
    final_w = arr_final.shape[1]
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
# 6. Config-Driven Line Generator (Task 2.3 Items 2 & 3)
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_dataset_splits(
    config_path: Path = CONFIG_PATH,
    out_dir: Path = SYNTH_DIR,
    counts: Dict[str, int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generates synthetic Quranic lines split by data/derived/splits.json.
    Enforces:
    - Width after 64px scaling < max_width_px (refuses/drops >= 1500px)
    - CTC feasibility check
    - Manifest CSV in dataset.py format (image_path, transcription)
    - Per-font weights & held-out font tracking
    """
    if counts is None:
        counts = {"train": 100, "val": 20, "test": 20}

    cfg = load_config(config_path)
    words_min = cfg.get("words_per_line", {}).get("min", 3)
    words_max = cfg.get("words_per_line", {}).get("max", 9)
    max_w_limit = cfg.get("constraints", {}).get("max_width_px", DEFAULT_MAX_WIDTH)
    aug_cfg = cfg.get("augmentation", {})

    fonts_cfg = cfg.get("fonts", {})
    weights_map = fonts_cfg.get("weights", {})
    default_wt = fonts_cfg.get("default_weight", 1.0)
    held_out = set(fonts_cfg.get("held_out", []))

    all_fonts = sorted(list(FONTS_DIR.glob("*.ttf")))
    font_weights = [weights_map.get(fp.stem, default_wt) for fp in all_fonts]

    word_stream = QuranWordStream(DERIVED_DIR / "ayahs.jsonl", DERIVED_DIR / "splits.json")
    rng_py = random.Random(seed)
    rng_np = np.random.RandomState(seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 75)
    print("           SYNTHETIC QURAN HTR LINE GENERATOR (CONFIG-DRIVEN)")
    print(f"Target splits: {counts} | Max width limit: {max_w_limit}px | Seed: {seed}")
    print(f"Config fonts: {len(all_fonts)} total, {len(held_out)} held-out listed")
    print("=" * 75)

    stats = {
        "total_attempts": 0,
        "dropped_width": 0,
        "dropped_ctc": 0,
        "generated": {k: 0 for k in counts},
        "elapsed_sec": 0.0,
        "throughput_lines_sec": 0.0,
    }

    t0 = time.time()

    for split, target_n in counts.items():
        split_img_dir = out_dir / split
        split_img_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = []

        print(f"\nGenerating split: '{split}' ({target_n} lines)...")
        for i in range(target_n):
            while True:
                stats["total_attempts"] += 1
                font_path = rng_py.choices(all_fonts, weights=font_weights, k=1)[0]
                text, surah = word_stream.sample_window(
                    split=split, min_words=words_min, max_words=words_max, rng=rng_py
                )

                spacing_jitter = rng_py.uniform(
                    aug_cfg.get("word_spacing_factor", {}).get("min", 0.7),
                    aug_cfg.get("word_spacing_factor", {}).get("max", 1.4),
                )
                font_size = rng_py.randint(
                    aug_cfg.get("font_size", {}).get("min", 30),
                    aug_cfg.get("font_size", {}).get("max", 42),
                )

                raw_img = render_arabic_text(
                    font_path, text, font_size=font_size, word_spacing_factor=spacing_jitter
                )
                aug_img = augment_line_image(raw_img, rng=rng_np, aug_cfg=aug_cfg)

                # Constraint 1: refuse or drop any line whose width after 64px scaling is >= max_width_px
                if aug_img.width >= max_w_limit:
                    stats["dropped_width"] += 1
                    continue

                # Constraint 2: CTC feasibility check: width // 4 >= min_ctc_len(text)
                time_steps = aug_img.width // 4
                required_steps = min_ctc_len(text)
                if time_steps < required_steps:
                    stats["dropped_ctc"] += 1
                    continue

                # Valid line
                img_name = f"line_{i:05d}.png"
                img_path = split_img_dir / img_name
                aug_img.save(img_path)

                # Output format for dataset.py: (image_path, transcription)
                manifest_rows.append({
                    "image_path": str(img_path),
                    "transcription": text,
                })
                stats["generated"][split] += 1
                break

        # Save manifest CSV
        manifest_csv = out_dir / f"{split}.csv"
        with open(manifest_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["image_path", "transcription"])
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(f"Wrote {len(manifest_rows)} rows to {manifest_csv}")

    t1 = time.time()
    elapsed = max(t1 - t0, 1e-4)
    total_gen = sum(stats["generated"].values())
    stats["elapsed_sec"] = elapsed
    stats["throughput_lines_sec"] = total_gen / elapsed

    tot_attempts = stats["total_attempts"]
    d_width = stats["dropped_width"]
    d_ctc = stats["dropped_ctc"]
    rate_w = (d_width / tot_attempts) * 100 if tot_attempts else 0
    rate_ctc = (d_ctc / tot_attempts) * 100 if tot_attempts else 0
    tot_rate = ((d_width + d_ctc) / tot_attempts) * 100 if tot_attempts else 0

    print("\n" + "=" * 75)
    print("                     GENERATION & BENCHMARK REPORT")
    print("=" * 75)
    print(f"Total lines generated:    {total_gen} lines in {elapsed:.2f}s")
    print(f"Generator Throughput:     {stats['throughput_lines_sec']:.2f} lines/sec")
    print(f"Total candidate attempts: {tot_attempts}")
    print(f"Dropped width >= 1500px:  {d_width} ({rate_w:.2f}%)")
    print(f"Dropped CTC infeasible:   {d_ctc} ({rate_ctc:.2f}%)")
    print(f"Overall Drop Rate:        {tot_rate:.2f}%")
    print("=" * 75)
    return stats


# ---------------------------------------------------------------------------
# 7. Main Entry Point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Quran synthetic HTR line renderer")
    p.add_argument("--config", default=str(CONFIG_PATH), help="Path to synth.yaml")
    p.add_argument("--out_dir", default=str(SYNTH_DIR), help="Output directory for manifests & images")
    p.add_argument("--audit_fonts", action="store_true", help="Run font acceptance audit and report")
    p.add_argument("--num_train", type=int, default=100, help="Train samples to generate")
    p.add_argument("--num_val", type=int, default=20, help="Val samples to generate")
    p.add_argument("--num_test", type=int, default=20, help="Test samples to generate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()

    font_paths = sorted(list(FONTS_DIR.glob("*.ttf")))

    if args.audit_fonts:
        audit_all_fonts(font_paths)
        return

    # Always perform acceptance audit before generation to report failing fonts
    audit_all_fonts(font_paths)

    counts = {
        "train": args.num_train,
        "val": args.num_val,
        "test": args.num_test,
    }
    generate_dataset_splits(
        config_path=Path(args.config),
        out_dir=Path(args.out_dir),
        counts=counts,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
