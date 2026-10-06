"""
scripts/generate_wordmix_shards.py
Task 2.10 Item 5:
Generate word-mix corpus with unit_mix:
- 1 word: 25%, 2 words: 10%, 3-9 words: 65%
- Ruled-band crop + random horizontal margins 4-24px, 20% ink-tight crops
- Same noise/augmentation rules and per-font quotas as synth.yaml
- Splits: 60k train, 3k val, 3k test
- Save to data/shards/wordmix/
- Zip to dist/shards_wordmix.zip
- Compute and report size and word-count histogram
"""

from __future__ import annotations
import math
import multiprocessing as mp
import os
import random
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from PIL import Image, ImageFilter
from scipy.ndimage import distance_transform_edt, maximum_filter
import torch
import yaml

from render_lines import (
    FONTS_DIR, DERIVED_DIR, QuranWordStream,
    render_arabic_text, min_ctc_len, TARGET_HEIGHT, WIDTH_MULTIPLE
)
from generate_synth_shards import check_line_metrics_fast, compute_font_quotas

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "configs" / "synth.yaml"
WORDMIX_SHARDS_DIR = ROOT / "data" / "shards" / "wordmix"
DIST_DIR = ROOT / "dist"


def augment_wordmix_line(
    img: Image.Image,
    rng: np.random.RandomState,
    crop_mode: str = "random",
    margin_x: int = 15,
    apply_noise: bool = True,
) -> Image.Image:
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

    if crop_mode == "tight":
        target_ink_h = rng.randint(30, 42)
    elif crop_mode == "full":
        target_ink_h = rng.randint(48, 58)
    else:
        target_ink_h = rng.randint(38, 50)

    scale = target_ink_h / max(crop_h, 1)
    new_w = max(int(crop_w * scale), 20)
    new_h = target_ink_h

    pil_cropped = Image.fromarray(cropped).resize((new_w, new_h), Image.BILINEAR)

    top_margin = rng.randint(2, max(3, TARGET_HEIGHT - new_h - 2))
    # Ruled-band canvas with random horizontal margins 4-24px
    canvas_w = new_w + 2 * margin_x
    canvas = np.full((TARGET_HEIGHT, canvas_w), 255, dtype=np.uint8)
    canvas_img = Image.fromarray(canvas)
    canvas_img.paste(pil_cropped, (margin_x, top_margin))
    cur_arr = np.array(canvas_img)

    # 1. Baseline wobble
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

    # 2. Slant / shear
    if rng.rand() < 0.8:
        shear_val = rng.uniform(-0.15, 0.15)
        cur_img = cur_img.transform(
            cur_img.size,
            Image.AFFINE,
            (1, shear_val, -shear_val * TARGET_HEIGHT / 2, 0, 1, 0),
            fillcolor=255,
        )

    # 3. Relative thickness jitter (max +-1px)
    stroke_choice = rng.rand()
    if stroke_choice < 0.25:
        thickened = cur_img.filter(ImageFilter.MinFilter(3))
        cur_img = Image.blend(cur_img, thickened, alpha=0.5)
    elif stroke_choice < 0.45:
        thinned = cur_img.filter(ImageFilter.MaxFilter(3))
        cur_img = Image.blend(cur_img, thinned, alpha=0.5)

    # 4. Elastic warp
    if rng.rand() < 0.6:
        arr_warp = np.array(cur_img, dtype=np.float32)
        dh, dw = arr_warp.shape
        grid_x, grid_y = np.meshgrid(np.arange(dw, dtype=np.float32), np.arange(dh, dtype=np.float32))
        flow_x = rng.normal(0, 0.8, size=(5, 5)).astype(np.float32)
        flow_y = rng.normal(0, 0.8, size=(5, 5)).astype(np.float32)
        flow_x_up = np.array(Image.fromarray(flow_x).resize((dw, dh), Image.BILINEAR), dtype=np.float32)
        flow_y_up = np.array(Image.fromarray(flow_y).resize((dw, dh), Image.BILINEAR), dtype=np.float32)
        map_x = np.clip(grid_x + flow_x_up, 0, dw - 1).astype(np.int32)
        map_y = np.clip(grid_y + flow_y_up, 0, dh - 1).astype(np.int32)
        cur_img = Image.fromarray(arr_warp[map_y, map_x].astype(np.uint8))

    # Pad width to multiple of WIDTH_MULTIPLE=32
    final_w = cur_img.width
    pad_w = (-final_w) % WIDTH_MULTIPLE
    if pad_w:
        padded_img = Image.new("L", (final_w + pad_w, TARGET_HEIGHT), color=255)
        padded_img.paste(cur_img, (0, 0))
        cur_img = padded_img

    # Noise & Blur: 60% noise, 40% clean
    if apply_noise:
        if rng.rand() < 0.7:
            blur_r = rng.uniform(0.2, 0.5)
            cur_img = cur_img.filter(ImageFilter.GaussianBlur(blur_r))
        arr_final = np.array(cur_img, dtype=np.float32)
        noise = rng.normal(0, rng.uniform(2.0, 3.5), size=arr_final.shape)
        arr_final = np.clip(arr_final + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr_final)
    else:
        return cur_img


def _worker_generate_wordmix_quota(args_tuple) -> Tuple[List[torch.Tensor], List[str], List[str], List[int], int, int]:
    (
        font_path, target_quota, split, sw_min, sw_max, fr_min, fr_max, max_w, seed
    ) = args_tuple

    rng_py = random.Random(seed)
    rng_np = np.random.RandomState(seed)
    word_stream = QuranWordStream(DERIVED_DIR / "ayahs.jsonl", DERIVED_DIR / "splits.json")

    images = []
    texts = []
    fonts = []
    surahs = []
    attempts = 0
    rejects = 0

    while len(images) < target_quota:
        attempts += 1

        # unit_mix: 1 word 25%, 2 words 10%, 3-9 words 65%
        r_unit = rng_py.random()
        if r_unit < 0.25:
            n_words = 1
        elif r_unit < 0.35:
            n_words = 2
        else:
            n_words = rng_py.randint(3, 9)

        text, sura = word_stream.sample_window(split=split, min_words=n_words, max_words=n_words, rng=rng_py)
        spacing = rng_py.uniform(0.7, 1.4)
        f_size = rng_py.randint(30, 42)

        raw = render_arabic_text(font_path, text, font_size=f_size, word_spacing_factor=spacing)

        # 20% ink-tight crops, 80% normal/full
        crop_mode = "tight" if rng_np.rand() < 0.20 else rng_np.choice(["normal", "full"])
        # Random horizontal margins 4-24px
        margin_x = rng_np.randint(4, 25)
        # 60% noise / 40% clean
        apply_noise = rng_np.rand() < 0.60

        aug = augment_wordmix_line(raw, rng=rng_np, crop_mode=crop_mode, margin_x=margin_x, apply_noise=apply_noise)

        if aug.width >= max_w:
            rejects += 1
            continue

        if aug.width // 4 < min_ctc_len(text):
            rejects += 1
            continue

        aug_arr = np.array(aug)
        if not check_line_metrics_fast(aug_arr, sw_min, sw_max, fr_min, fr_max):
            rejects += 1
            continue

        images.append(torch.from_numpy(aug_arr))
        texts.append(text)
        fonts.append(font_path.stem)
        surahs.append(sura)

    return images, texts, fonts, surahs, attempts, rejects


def generate_wordmix_split(
    split_name: str,
    split_key: str,
    font_quotas: Dict[Path, int],
    cfg: dict,
    out_dir: Path,
    seed: int,
    num_workers: int,
    max_per_shard: int = 10000,
) -> Tuple[int, Counter]:
    sw_bounds = cfg["calibration_bounds"]["stroke_width"]
    fr_bounds = cfg["calibration_bounds"]["ink_fraction"]
    sw_min, sw_max = sw_bounds["min"], sw_bounds["max"]
    fr_min, fr_max = fr_bounds["min"], fr_bounds["max"]
    max_w = cfg["constraints"]["max_width_px"]

    tasks = []
    idx = 0
    for fp, quota in font_quotas.items():
        if quota > 0:
            tasks.append((fp, quota, split_key, sw_min, sw_max, fr_min, fr_max, max_w, seed + idx * 777))
            idx += 1

    t0 = time.time()
    with mp.Pool(processes=min(num_workers, len(tasks))) as pool:
        results = pool.map(_worker_generate_wordmix_quota, tasks)

    all_images, all_texts, all_fonts, all_surahs = [], [], [], []
    for imgs, txts, f_names, s_list, _, _ in results:
        all_images.extend(imgs)
        all_texts.extend(txts)
        all_fonts.extend(f_names)
        all_surahs.extend(s_list)

    combined = list(zip(all_images, all_texts, all_fonts, all_surahs))
    random.Random(seed).shuffle(combined)
    all_images, all_texts, all_fonts, all_surahs = zip(*combined)

    # Compute word-count histogram
    word_hist = Counter(len(t.split()) for t in all_texts)

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
        print(f"  Saved {shard_path.name}: {len(s_imgs)} lines ({shard_path.stat().st_size / (1024*1024):.1f} MB)", flush=True)

    elapsed = max(time.time() - t0, 1e-4)
    print(f"Split '{split_name}' complete: {len(all_images)} lines in {elapsed:.1f}s ({len(all_images)/elapsed:.1f} l/s)", flush=True)
    return len(all_images), word_hist


def main():
    print("=" * 65)
    print("TASK 2.10.5: GENERATING WORD-MIX CORPUS (60k Train, 3k Val, 3k Test)")
    print("=" * 65)

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    fonts_manifest_path = ROOT / "data" / "fonts_manifest.json"
    with open(fonts_manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    weights_map = cfg["fonts"]["weights"]
    held_out = set(cfg["fonts"].get("held_out", []))

    all_fonts = sorted(list(FONTS_DIR.glob("*.ttf")))
    train_font_paths = [f for f in all_fonts if weights_map.get(f.stem, 0.0) > 0 and f.stem not in held_out]

    quotas_train = compute_font_quotas(train_font_paths, weights_map, 60000)
    quotas_val = compute_font_quotas(train_font_paths, weights_map, 3000)
    quotas_test = compute_font_quotas(train_font_paths, weights_map, 3000)

    num_workers = min(3, os.cpu_count() or 2)
    total_hist = Counter()

    print(f"\n[1/3] Generating wordmix_train (60,000 lines)...", flush=True)
    _, h_train = generate_wordmix_split("wordmix_train", "train", quotas_train, cfg, WORDMIX_SHARDS_DIR, seed=101, num_workers=num_workers)
    total_hist += h_train

    print(f"\n[2/3] Generating wordmix_val (3,000 lines)...", flush=True)
    _, h_val = generate_wordmix_split("wordmix_val", "val", quotas_val, cfg, WORDMIX_SHARDS_DIR, seed=202, num_workers=num_workers)
    total_hist += h_val

    print(f"\n[3/3] Generating wordmix_test (3,000 lines)...", flush=True)
    _, h_test = generate_wordmix_split("wordmix_test", "test", quotas_test, cfg, WORDMIX_SHARDS_DIR, seed=303, num_workers=num_workers)
    total_hist += h_test

    # Print Histogram
    total_lines = sum(total_hist.values())
    print("\n" + "=" * 50)
    print(f"WORD-COUNT HISTOGRAM (Total {total_lines:,} lines)")
    print("=" * 50)
    for n_w in sorted(total_hist.keys()):
        cnt = total_hist[n_w]
        pct = cnt / total_lines * 100.0
        bar = "#" * int(pct / 2)
        print(f"  {n_w} word{'s' if n_w > 1 else ' '} : {cnt:6d} ({pct:5.1f}%) | {bar}")

    # Zip to dist/shards_wordmix.zip
    DIST_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DIST_DIR / "shards_wordmix.zip"
    print(f"\nZipping {WORDMIX_SHARDS_DIR} -> {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for pt_file in sorted(WORDMIX_SHARDS_DIR.glob("*.pt")):
            zf.write(pt_file, arcname=pt_file.name)
            print(f"  Added {pt_file.name} to zip")

    zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"\nFinished: {zip_path.name} created successfully ({zip_size_mb:.1f} MB)")


if __name__ == "__main__":
    import json
    main()
