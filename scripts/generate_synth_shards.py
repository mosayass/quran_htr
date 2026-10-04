"""
scripts/generate_synth_shards.py
High-speed multiprocessing synthetic line generator with calibrated font weights,
p5-p95 bounds rejection, relative thickness jitter, and direct uint8 shard output.
"""

from __future__ import annotations
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from render_lines import (
    FONTS_DIR, DERIVED_DIR, QuranWordStream,
    render_arabic_text, min_ctc_len, TARGET_HEIGHT, WIDTH_MULTIPLE
)
from shards import write_shard_file

CONFIG_PATH = ROOT / "configs" / "synth.yaml"
SHARDS_OUT_DIR = ROOT / "data" / "shards" / "synth"


def check_line_metrics_fast(
    arr: np.ndarray,
    sw_min: float,
    sw_max: float,
    fr_min: float,
    fr_max: float,
) -> bool:
    """Fast calibration check: ink fraction check first, then representative crop distance transform."""
    ink = arr < 128
    cols = np.where(ink.sum(axis=0) > 0)[0]
    if len(cols) == 0:
        return False
    span_w = cols[-1] - cols[0] + 1
    ink_frac = float(ink.sum() / (64 * span_w))
    if ink_frac < fr_min or ink_frac > fr_max:
        return False

    # Measure stroke width on representative center crop
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
    crop_mode: str = "random",
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
        grid_x, grid_y = np.meshgrid(np.arange(dw), np.arange(dh))
        flow_x = rng.normal(0, 0.8, size=(5, 5))
        flow_y = rng.normal(0, 0.8, size=(5, 5))
        flow_x_up = np.array(Image.fromarray(flow_x).resize((dw, dh), Image.BILINEAR))
        flow_y_up = np.array(Image.fromarray(flow_y).resize((dw, dh), Image.BILINEAR))
        map_x = np.clip(grid_x + flow_x_up, 0, dw - 1).astype(int)
        map_y = np.clip(grid_y + flow_y_up, 0, dh - 1).astype(int)
        cur_img = Image.fromarray(arr_warp[map_y, map_x].astype(np.uint8))

    # 5. Gaussian blur
    if rng.rand() < 0.7:
        blur_r = rng.uniform(0.2, 0.6)
        cur_img = cur_img.filter(ImageFilter.GaussianBlur(blur_r))

    # 6. Add subtle paper noise
    arr_final = np.array(cur_img, dtype=np.float32)
    noise = rng.normal(0, rng.uniform(2.0, 4.0), size=arr_final.shape)
    arr_final = np.clip(arr_final + noise, 0, 255).astype(np.uint8)

    return Image.fromarray(arr_final)


def _worker_generate_chunk(args_tuple) -> Tuple[List[torch.Tensor], List[str], int, int, int]:
    (
        target_count, split, font_paths, font_weights,
        sw_min, sw_max, fr_min, fr_max, max_w_limit, seed
    ) = args_tuple

    rng_py = random.Random(seed)
    rng_np = np.random.RandomState(seed)
    word_stream = QuranWordStream(DERIVED_DIR / "ayahs.jsonl", DERIVED_DIR / "splits.json")

    images = []
    texts = []
    attempts = 0
    rejects_bounds = 0
    rejects_ctc_width = 0

    while len(images) < target_count:
        attempts += 1
        font_path = rng_py.choices(font_paths, weights=font_weights)[0]
        text, surah = word_stream.sample_window(split=split, min_words=3, max_words=9, rng=rng_py)

        spacing = rng_py.uniform(0.7, 1.4)
        f_size = rng_py.randint(30, 42)

        raw = render_arabic_text(font_path, text, font_size=f_size, word_spacing_factor=spacing)
        aug = augment_line_relative_thickness(raw, rng=rng_np)

        if aug.width >= max_w_limit:
            rejects_ctc_width += 1
            continue

        time_steps = aug.width // 4
        if time_steps < min_ctc_len(text):
            rejects_ctc_width += 1
            continue

        aug_arr = np.array(aug)
        if not check_line_metrics_fast(aug_arr, sw_min, sw_max, fr_min, fr_max):
            rejects_bounds += 1
            continue

        t = torch.from_numpy(aug_arr).unsqueeze(0)
        images.append(t)
        texts.append(text)

    return images, texts, attempts, rejects_bounds, rejects_ctc_width


def generate_split_shards(
    split_name: str,
    target_count: int,
    split_surah_key: str,
    font_paths: List[Path],
    font_weights: List[float],
    cfg: dict,
    out_dir: Path,
    num_workers: int = 8,
    max_per_shard: int = 10000,
    seed: int = 42,
) -> Dict[str, Any]:
    print(f"\n{'='*75}\nGenerating split '{split_name}': {target_count} lines across {num_workers} workers...")
    sw_bounds = cfg["calibration_bounds"]["stroke_width"]
    fr_bounds = cfg["calibration_bounds"]["ink_fraction"]
    sw_min, sw_max = sw_bounds["min"], sw_bounds["max"]
    fr_min, fr_max = fr_bounds["min"], fr_bounds["max"]
    max_w = cfg["constraints"]["max_width_px"]

    chunk_size = math.ceil(target_count / num_workers)
    tasks = []
    for w in range(num_workers):
        n_w = min(chunk_size, target_count - w * chunk_size)
        if n_w <= 0:
            break
        tasks.append((
            n_w, split_surah_key, font_paths, font_weights,
            sw_min, sw_max, fr_min, fr_max, max_w, seed + w * 1000
        ))

    t0 = time.time()
    with mp.Pool(processes=len(tasks)) as pool:
        results = pool.map(_worker_generate_chunk, tasks)

    total_images = []
    total_texts = []
    total_attempts = 0
    total_rejects_bounds = 0
    total_rejects_ctc_width = 0

    for imgs, txts, att, rej_b, rej_c in results:
        total_images.extend(imgs)
        total_texts.extend(txts)
        total_attempts += att
        total_rejects_bounds += rej_b
        total_rejects_ctc_width += rej_c

    out_dir.mkdir(parents=True, exist_ok=True)
    num_shards = math.ceil(len(total_images) / max_per_shard)
    for s_idx in range(num_shards):
        s_imgs = total_images[s_idx * max_per_shard : (s_idx + 1) * max_per_shard]
        s_txts = total_texts[s_idx * max_per_shard : (s_idx + 1) * max_per_shard]
        shard_path = out_dir / f"{split_name}_{s_idx:03d}.pt"
        write_shard_file(s_imgs, s_txts, shard_path)
        sz_mb = shard_path.stat().st_size / (1024 * 1024)
        print(f"  Saved shard {shard_path.name}: {len(s_imgs)} lines ({sz_mb:.1f} MB)")

    elapsed = max(time.time() - t0, 1e-4)
    throughput = len(total_images) / elapsed
    rej_bounds_pct = (total_rejects_bounds / total_attempts) * 100 if total_attempts else 0
    total_rej_pct = ((total_rejects_bounds + total_rejects_ctc_width) / total_attempts) * 100 if total_attempts else 0

    print(f"Split '{split_name}' complete:")
    print(f"  Generated:       {len(total_images)} lines in {elapsed:.2f}s ({throughput:.1f} lines/sec)")
    print(f"  Attempts:        {total_attempts}")
    print(f"  Rejects p5-p95:  {total_rejects_bounds} ({rej_bounds_pct:.2f}%)")
    print(f"  Total reject %:  {total_rej_pct:.2f}%")

    return {
        "split": split_name,
        "count": len(total_images),
        "elapsed_sec": elapsed,
        "throughput": throughput,
        "attempts": total_attempts,
        "rejects_bounds": total_rejects_bounds,
        "rejects_bounds_pct": rej_bounds_pct,
        "total_reject_pct": total_rej_pct,
    }


def main():
    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    weights_map = cfg["fonts"]["weights"]
    held_out = set(cfg["fonts"]["held_out"])

    all_fonts = sorted(list(FONTS_DIR.glob("*.ttf")))
    train_fonts = [f for f in all_fonts if weights_map.get(f.stem, 0.0) > 0 and f.stem not in held_out]
    train_weights = [weights_map[f.stem] for f in train_fonts]

    held_out_fonts = [f for f in all_fonts if f.stem in held_out]
    held_out_weights = [1.0] * len(held_out_fonts)

    print(f"Configured {len(train_fonts)} calibrated fonts for train/val/test, {len(held_out_fonts)} held-out fonts.")
    num_cores = min(8, os.cpu_count() or 4)

    # 1. synth_train: 80,000 lines
    res_train = generate_split_shards(
        "synth_train", 80000, "train", train_fonts, train_weights,
        cfg, SHARDS_OUT_DIR, num_workers=num_cores, max_per_shard=10000, seed=100
    )

    # 2. synth_val: 4,000 lines
    res_val = generate_split_shards(
        "synth_val", 4000, "val", train_fonts, train_weights,
        cfg, SHARDS_OUT_DIR, num_workers=num_cores, max_per_shard=10000, seed=200
    )

    # 3. synth_test: 4,000 lines
    res_test = generate_split_shards(
        "synth_test", 4000, "test", train_fonts, train_weights,
        cfg, SHARDS_OUT_DIR, num_workers=num_cores, max_per_shard=10000, seed=300
    )

    # 4. synth_test_unseen_fonts: 2,000 lines
    res_unseen = generate_split_shards(
        "synth_test_unseen_fonts", 2000, "test", held_out_fonts, held_out_weights,
        cfg, SHARDS_OUT_DIR, num_workers=num_cores, max_per_shard=10000, seed=400
    )

    print("\n" + "=" * 75)
    print("           SYNTHETIC SHARDS GENERATION REPORT")
    print("=" * 75)
    for r in [res_train, res_val, res_test, res_unseen]:
        print(f"  {r['split']:25s}: {r['count']:5d} lines | Rej: {r['rejects_bounds_pct']:.1f}% | {r['throughput']:.1f} l/s")
    print("=" * 75)


if __name__ == "__main__":
    main()
