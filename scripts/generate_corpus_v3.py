"""
scripts/generate_corpus_v3.py
Task 2.11 Step 4:
Corpus v3 Synthetic Shard Generator:
1. Exact word-count quotas enforced:
   - 1 word:  25%
   - 2 words: 10%
   - 3-9 words: 65% uniform across lengths 3..9
2. Relaxation on short samples:
   - NO KHATT ink-fraction rejection on 1-2-word samples (retains stroke-width bounds).
3. Diacritics rendered at random levels:
   - 40% none (rasm-only)
   - 30% light (partially vowelled: random vowel dropout, preserving shaddah/sukun)
   - 30% full (fully vowelled simple variant)
   - Shard labels are STRICTLY rasm-only (simple-clean).
4. Volume & Splits:
   - 60,000 train (train surahs, 26 calibrated fonts)
   - 3,000 val (val surahs, 26 calibrated fonts)
   - 3,000 test (test surahs, 26 calibrated fonts)
   - Saved to data/shards/corpus_v3/ (existing shards untouched).
5. Reports exact histogram and diacritic-level split.
"""

from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import gc
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Dict, List, Tuple, Any

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
        sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
    except Exception:
        pass

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
CORPUS_V3_DIR = ROOT / "data" / "shards" / "corpus_v3"

PAUSE_MARKS_RE = re.compile(r"[\u0615-\u061a\u06d6-\u06ed\u06e9]")
SHORT_VOWELS = "\u064E\u064F\u0650"  # fatha, damma, kasra


def make_light_diacritics(vowelled_text: str, rng: random.Random) -> str:
    """Produces lightly-vowelled text by randomly dropping ~70% of short vowels while keeping shaddah/sukun/tanween."""
    chars = []
    for c in vowelled_text:
        if c in SHORT_VOWELS and rng.random() < 0.70:
            continue  # drop short vowel
        chars.append(c)
    return "".join(chars)


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


def check_stroke_width_only(arr: np.ndarray, sw_min: float, sw_max: float) -> bool:
    """Stroke-width validation for 1-2 word crops (exempt from ink-fraction rejection)."""
    ink = arr < 128
    cols = np.where(ink.sum(axis=0) > 0)[0]
    if len(cols) == 0:
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
    ink_tight: bool = False,
) -> Image.Image:
    arr = np.array(img)
    ink_mask = arr < 240
    ink_rows = np.where(ink_mask.sum(axis=1) > 0)[0]
    ink_cols = np.where(ink_mask.sum(axis=0) > 0)[0]

    if len(ink_rows) > 0 and len(ink_cols) > 0:
        top, bot = ink_rows[0], ink_rows[-1]
        left, right = ink_cols[0], ink_cols[-1]

        if ink_tight:
            pad_top = rng.randint(2, 6)
            pad_bot = rng.randint(2, 6)
            pad_x = rng.randint(2, 8)
        else:
            pad_top = rng.randint(4, 16)
            pad_bot = rng.randint(4, 16)
            pad_x = rng.randint(6, 24)

        y1 = max(0, top - pad_top)
        y2 = min(arr.shape[0], bot + pad_bot + 1)
        x1 = max(0, left - pad_x)
        x2 = min(arr.shape[1], right + pad_x + 1)
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
    r = rng.rand()
    if r < 0.20:
        filt = maximum_filter(cur_arr, size=(2, 2))
        cur_arr = np.minimum(cur_arr, filt)
    elif r < 0.40:
        filt = distance_transform_edt(cur_arr < 128)
        cur_arr = np.clip(cur_arr + (rng.rand() * 15 - 5), 0, 255)

    noise = rng.normal(0, 3.5, cur_arr.shape)
    cur_arr = np.clip(cur_arr + noise, 0, 255).astype(np.uint8)

    cur_img = Image.fromarray(cur_arr)
    if rng.rand() < 0.25:
        cur_img = cur_img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.6)))
    return cur_img


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

        for s_num, c_ayah, s_ayah in zip(surah_map, clean_lines, simple_lines):
            s_clean_marks = PAUSE_MARKS_RE.sub("", s_ayah)
            c_words = c_ayah.split()
            v_words = s_clean_marks.split()
            if len(c_words) == len(v_words):
                self.surah_clean[s_num].extend(c_words)
                self.surah_vowelled[s_num].extend(v_words)

        splits_data = json.loads(splits_path.read_text(encoding="utf-8"))
        self.split_surahs = {
            "train": set(splits_data["train"]["surahs"]),
            "val": set(splits_data["val"]["surahs"]),
            "test": set(splits_data["test"]["surahs"]),
        }

    def sample_window(self, split: str, n_words: int, rng: random.Random) -> Tuple[str, str, int]:
        valid_surahs = [s for s in self.split_surahs[split] if len(self.surah_clean[s]) >= n_words]
        sura = rng.choice(valid_surahs)
        c_words = self.surah_clean[sura]
        v_words = self.surah_vowelled[sura]

        start = rng.randint(0, len(c_words) - n_words)
        clean_text = " ".join(c_words[start : start + n_words])
        vowelled_text = " ".join(v_words[start : start + n_words])
        return clean_text, vowelled_text, sura


def sample_word_count(rng: random.Random) -> int:
    """Enforces: 1w 25%, 2w 10%, 3-9w 65% uniform (9.2857% each)."""
    r = rng.random()
    if r < 0.25:
        return 1
    elif r < 0.35:
        return 2
    else:
        return rng.randint(3, 9)


def sample_diacritic_level(rng: random.Random) -> str:
    """Enforces: 40% none, 30% light, 30% full."""
    r = rng.random()
    if r < 0.40:
        return "none"
    elif r < 0.70:
        return "light"
    else:
        return "full"


def _worker_generate_corpus_v3_quota(args_tuple):
    (
        font_path, target_quota, split, sw_min, sw_max, fr_min, fr_max, max_w, seed
    ) = args_tuple

    rng_py = random.Random(seed)
    rng_np = np.random.RandomState(seed)
    stream = AlignedQuranWordStream(
        ROOT / "data" / "tanzil" / "quran-simple-clean.txt",
        ROOT / "data" / "tanzil" / "quran-simple.txt",
        DERIVED_DIR / "splits.json"
    )

    images = []
    texts = []
    fonts = []
    surahs = []
    word_counts = []
    diac_levels = []
    attempts = 0
    rejects = 0

    while len(images) < target_quota:
        attempts += 1
        n_words = sample_word_count(rng_py)
        diac_lvl = sample_diacritic_level(rng_py)

        clean_text, vowelled_text, sura = stream.sample_window(split, n_words, rng_py)

        if diac_lvl == "none":
            render_text = clean_text
        elif diac_lvl == "light":
            render_text = make_light_diacritics(vowelled_text, rng_py)
        else:
            render_text = vowelled_text

        spacing = rng_py.uniform(0.7, 1.4)
        f_size = rng_py.randint(30, 42)

        raw = render_arabic_text(font_path, render_text, font_size=f_size, word_spacing_factor=spacing)

        apply_noise = rng_np.rand() < 0.60
        ink_tight = rng_np.rand() < 0.20
        aug = augment_line_relative_thickness(raw, rng=rng_np, apply_noise=apply_noise, ink_tight=ink_tight)

        if aug.width >= max_w or (aug.width // 4 < min_ctc_len(clean_text)):
            rejects += 1
            continue

        aug_arr = np.array(aug)

        # Rejection rule: 1-2 words are EXEMPT from ink-fraction rejection
        if n_words <= 2:
            if not check_stroke_width_only(aug_arr, sw_min, sw_max):
                rejects += 1
                continue
        else:
            if not check_line_metrics_fast(aug_arr, sw_min, sw_max, fr_min, fr_max):
                rejects += 1
                continue

        images.append(torch.from_numpy(aug_arr))
        texts.append(clean_text)  # Shard labels strictly rasm-only
        fonts.append(font_path.stem)
        surahs.append(sura)
        word_counts.append(n_words)
        diac_levels.append(diac_lvl)

    return images, texts, fonts, surahs, word_counts, diac_levels, attempts, rejects


def generate_corpus_v3_split(
    split_name: str,
    target_count: int,
    split_key: str,
    font_quotas: Dict[Path, int],
    cfg: dict,
    out_dir: Path,
    num_workers: int = 6,
    max_per_shard: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    print(f"\n{'='*75}\nGenerating Corpus v3 '{split_name}': {target_count} lines...")
    sw_bounds = cfg["calibration_bounds"]["stroke_width"]
    fr_bounds = cfg["calibration_bounds"]["ink_fraction"]
    sw_min, sw_max = sw_bounds["min"], sw_bounds["max"]
    fr_min, fr_max = fr_bounds["min"], fr_bounds["max"]
    max_w = cfg["constraints"]["max_width_px"]

    tasks = []
    idx = 0
    for fp, quota in font_quotas.items():
        if quota > 0:
            tasks.append((
                fp, quota, split_key, sw_min, sw_max, fr_min, fr_max, max_w, seed + idx * 888
            ))
            idx += 1

    t0 = time.time()
    with mp.Pool(processes=min(num_workers, len(tasks))) as pool:
        results = pool.map(_worker_generate_corpus_v3_quota, tasks)

    all_images = []
    all_texts = []
    all_fonts = []
    all_surahs = []
    all_words = []
    all_diacs = []
    total_attempts = 0
    total_rejects = 0

    for imgs, txts, f_names, s_list, w_list, d_list, att, rej in results:
        all_images.extend(imgs)
        all_texts.extend(txts)
        all_fonts.extend(f_names)
        all_surahs.extend(s_list)
        all_words.extend(w_list)
        all_diacs.extend(d_list)
        total_attempts += att
        total_rejects += rej

    combined = list(zip(all_images, all_texts, all_fonts, all_surahs, all_words, all_diacs))
    random.Random(seed).shuffle(combined)
    all_images, all_texts, all_fonts, all_surahs, all_words, all_diacs = zip(*combined)

    out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = math.ceil(len(all_images) / max_per_shard)
    for s_idx in range(num_shards):
        s_imgs = list(all_images[s_idx * max_per_shard : (s_idx + 1) * max_per_shard])
        s_txts = list(all_texts[s_idx * max_per_shard : (s_idx + 1) * max_per_shard])
        s_fonts = list(all_fonts[s_idx * max_per_shard : (s_idx + 1) * max_per_shard])
        s_surahs = list(all_surahs[s_idx * max_per_shard : (s_idx + 1) * max_per_shard])
        shard_path = out_dir / f"{split_name}_{s_idx:03d}.pt"
        torch.save({
            "images": s_imgs,
            "texts": s_txts,
            "fonts": s_fonts,
            "surahs": s_surahs,
            "count": len(s_imgs),
        }, shard_path)
        sz_mb = shard_path.stat().st_size / (1024 * 1024)
        print(f"  Saved shard {shard_path.name}: {len(s_imgs)} lines ({sz_mb:.1f} MB)")

    elapsed = max(time.time() - t0, 1e-4)
    throughput = len(all_images) / elapsed
    rej_pct = (total_rejects / total_attempts) * 100 if total_attempts else 0.0

    # Calculate histograms
    word_hist = Counter(all_words)
    diac_hist = Counter(all_diacs)

    print(f"\nSplit '{split_name}' complete: {len(all_images)} lines in {elapsed:.1f}s ({throughput:.1f} l/s) | Rejects: {rej_pct:.1f}%")
    print(f"  Word Count Distribution:")
    for k in sorted(word_hist.keys()):
        pct = (word_hist[k] / len(all_images)) * 100
        print(f"    {k} words: {word_hist[k]:5d} ({pct:5.2f}%)")
    print(f"  Diacritic Levels:")
    for k in ["none", "light", "full"]:
        pct = (diac_hist[k] / len(all_images)) * 100
        print(f"    {k:6s}: {diac_hist[k]:5d} ({pct:5.2f}%)")

    return {
        "split": split_name,
        "count": len(all_images),
        "word_hist": dict(word_hist),
        "diac_hist": dict(diac_hist),
        "elapsed": elapsed,
        "throughput": throughput,
    }


def main():
    p = argparse.ArgumentParser(description="Corpus v3 generator")
    p.add_argument("--num_train", type=int, default=60000)
    p.add_argument("--num_val", type=int, default=3000)
    p.add_argument("--num_test", type=int, default=3000)
    p.add_argument("--num_workers", type=int, default=min(6, os.cpu_count() or 4))
    args = p.parse_args()

    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    weights_map = cfg["fonts"]["weights"]
    held_out = set(cfg["fonts"]["held_out"])

    all_fonts = sorted(list(FONTS_DIR.glob("*.ttf")))
    train_fonts = [f for f in all_fonts if weights_map.get(f.stem, 0.0) > 0 and f.stem not in held_out]

    # Quotas for Corpus v3
    def compute_font_quotas(font_paths, weights_map, total_lines):
        tot_weight = sum(weights_map.get(fp.stem, 0.0) for fp in font_paths)
        quotas = {}
        for fp in font_paths:
            w = weights_map.get(fp.stem, 0.0)
            quotas[fp] = round(total_lines * (w / tot_weight))
        diff = total_lines - sum(quotas.values())
        quotas[list(quotas.keys())[0]] += diff
        return quotas

    train_quotas = compute_font_quotas(train_fonts, weights_map, args.num_train)
    val_quotas = compute_font_quotas(train_fonts, weights_map, args.num_val)
    test_quotas = compute_font_quotas(train_fonts, weights_map, args.num_test)

    # 1. corpus_v3_train
    res_train = generate_corpus_v3_split(
        "corpus_v3_train", args.num_train, "train", train_quotas, cfg, CORPUS_V3_DIR, num_workers=args.num_workers, seed=101
    )

    # 2. corpus_v3_val
    res_val = generate_corpus_v3_split(
        "corpus_v3_val", args.num_val, "val", val_quotas, cfg, CORPUS_V3_DIR, num_workers=args.num_workers, seed=202
    )

    # 3. corpus_v3_test
    res_test = generate_corpus_v3_split(
        "corpus_v3_test", args.num_test, "test", test_quotas, cfg, CORPUS_V3_DIR, num_workers=args.num_workers, seed=303
    )

    print("\n" + "=" * 75)
    print("                  CORPUS V3 GENERATION SUMMARY")
    print("=" * 75)
    for r in [res_train, res_val, res_test]:
        print(f"  {r['split']:20s}: {r['count']:5d} lines in {r['elapsed']:.1f}s ({r['throughput']:.1f} l/s)")
    print("=" * 75)


if __name__ == "__main__":
    main()
