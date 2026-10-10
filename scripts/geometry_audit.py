"""
scripts/geometry_audit.py
Task 2.14 Item 2:
Geometry table at 64px height:
- Vertical ink extent: min/max/extent of ink rows (extent = y_max - y_min + 1)
- Baseline row: center of mass / peak of horizontal projection profile
- Body-height proxy: 10th-90th percentile vertical spread of ink mass
- Stroke width: 2 * mean distance from ink pixels to background (via distance transform on skeleton)
- Ink fraction: sum(ink pixels) / (64 * W)
- Connected component count: number of 8-connected foreground ink components
Compares:
1. Capture Set (Tablet Stylus, 100 samples)
2. KHATT Test (Scanned Human Handwriting, held-out)
3. Synth v3 Clean (Synthesized Unvowelled Rasm)
"""

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import scipy.ndimage as ndi

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from vocab import Vocabulary
from shards import ShardDataset


def analyze_image_geometry(arr_gray: np.ndarray) -> dict:
    """
    arr_gray: uint8 2D array (H=64, W) where white background is ~255 and ink is dark (< 180).
    """
    h, w = arr_gray.shape
    ink_mask = arr_gray < 180
    ink_count = np.sum(ink_mask)

    if ink_count == 0:
        return {
            "ink_extent": 0,
            "baseline_row": h / 2,
            "body_height": 0,
            "stroke_width": 0,
            "ink_fraction": 0,
            "num_components": 0,
            "width": w,
        }

    y_indices, x_indices = np.where(ink_mask)
    y_min, y_max = y_indices.min(), y_indices.max()
    ink_extent = y_max - y_min + 1

    # Horizontal projection profile (sum ink across X for each row Y)
    row_ink = np.sum(ink_mask, axis=1).astype(float)
    total_ink = np.sum(row_ink)
    if total_ink > 0:
        baseline_row = np.sum(np.arange(h) * row_ink) / total_ink
        # Body-height proxy: span between 15th and 85th percentile of ink mass along Y
        cum_ink = np.cumsum(row_ink) / total_ink
        y_15 = np.searchsorted(cum_ink, 0.15)
        y_85 = np.searchsorted(cum_ink, 0.85)
        body_height = max(1, y_85 - y_15)
    else:
        baseline_row = h / 2
        body_height = ink_extent

    # Stroke width via distance transform
    dist = ndi.distance_transform_edt(ink_mask)
    # Average thickness is roughly 2 * mean distance along stroke interior
    stroke_width = 2.0 * np.mean(dist[dist > 0.5]) if np.any(dist > 0.5) else 0.0

    # Ink fraction
    ink_fraction = ink_count / (h * w)

    # Connected components
    labeled, num_components = ndi.label(ink_mask)

    return {
        "ink_extent": ink_extent,
        "baseline_row": baseline_row,
        "body_height": body_height,
        "stroke_width": stroke_width,
        "ink_fraction": ink_fraction,
        "num_components": num_components,
        "width": w,
    }


def compute_dataset_stats(samples: list[np.ndarray]) -> dict:
    stats_list = [analyze_image_geometry(s) for s in samples]
    return {
        "ink_extent": (np.mean([s["ink_extent"] for s in stats_list]), np.median([s["ink_extent"] for s in stats_list])),
        "baseline_row": (np.mean([s["baseline_row"] for s in stats_list]), np.median([s["baseline_row"] for s in stats_list])),
        "body_height": (np.mean([s["body_height"] for s in stats_list]), np.median([s["body_height"] for s in stats_list])),
        "stroke_width": (np.mean([s["stroke_width"] for s in stats_list]), np.median([s["stroke_width"] for s in stats_list])),
        "ink_fraction": (np.mean([s["ink_fraction"] for s in stats_list]), np.median([s["ink_fraction"] for s in stats_list])),
        "num_components": (np.mean([s["num_components"] for s in stats_list]), np.median([s["num_components"] for s in stats_list])),
        "width": (np.mean([s["width"] for s in stats_list]), np.median([s["width"] for s in stats_list])),
    }


def main():
    print("=" * 80)
    print("TASK 2.14 ITEM 2: GEOMETRY AUDIT AT 64px HEIGHT")
    print("=" * 80)

    # 1. Capture Set images
    capture_dir = ROOT / "data" / "capture"
    capture_images = []
    for png_path in sorted(capture_dir.glob("*.png")):
        img = Image.open(png_path).convert("L")
        if img.height != 64:
            new_w = max(1, round(img.width * (64 / img.height)))
            img = img.resize((new_w, 64), Image.BILINEAR)
        capture_images.append(np.array(img))
    print(f"Loaded {len(capture_images)} Capture Set images.")

    # 2. KHATT Test images
    khatt_shard_path = ROOT / "data" / "shards" / "khatt" / "khatt_test_000.pt"
    vocab = Vocabulary.load("vocab.json")
    khatt_images = []
    if khatt_shard_path.exists():
        ds_khatt = ShardDataset(khatt_shard_path, vocab, augment=False)
        for i in range(min(len(ds_khatt), 500)):
            # Tensor is [-1, 1], shape (1, 64, W), background is +1.0
            t = ds_khatt[i].image[0].numpy()
            uint8_img = np.clip((t * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            khatt_images.append(uint8_img)
        print(f"Loaded {len(khatt_images)} KHATT Test sample images.")
    else:
        print(f"Warning: {khatt_shard_path} not found.")

    # 3. Synth v3 Clean images
    synth_shard_path = ROOT / "data" / "shards" / "synth" / "synth_test_000.pt"
    if not synth_shard_path.exists():
        synth_shard_path = ROOT / "data" / "shards" / "wordmix" / "wordmix_test_000.pt"
    synth_images = []
    if synth_shard_path.exists():
        ds_synth = ShardDataset(synth_shard_path, vocab, augment=False)
        for i in range(min(len(ds_synth), 500)):
            t = ds_synth[i].image[0].numpy()
            uint8_img = np.clip((t * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)
            synth_images.append(uint8_img)
        print(f"Loaded {len(synth_images)} Synth v3 (Clean) sample images from {synth_shard_path.name}.")
    else:
        print(f"Warning: Synth test shard not found.")

    # Compute statistics
    cap_stats = compute_dataset_stats(capture_images)
    khatt_stats = compute_dataset_stats(khatt_images) if khatt_images else None
    synth_stats = compute_dataset_stats(synth_images) if synth_images else None

    # Print Geometry Comparison Table
    print("\n" + "=" * 90)
    print("GEOMETRY COMPARISON TABLE AT 64px HEIGHT (Mean [Median])")
    print("=" * 90)
    metrics = [
        ("Vertical Ink Extent (px)", "ink_extent", "{:.1f} [{:.1f}]"),
        ("Baseline Row (from top, px)", "baseline_row", "{:.1f} [{:.1f}]"),
        ("Body Height Proxy (px)", "body_height", "{:.1f} [{:.1f}]"),
        ("Stroke Width (px)", "stroke_width", "{:.2f} [{:.2f}]"),
        ("Ink Fraction (%)", "ink_fraction", "{:.2f}% [{:.2f}%]"),
        ("Component Count (per line)", "num_components", "{:.1f} [{:.1f}]"),
        ("Mean Line Width (px)", "width", "{:.1f} [{:.1f}]"),
    ]

    header = f"{'Metric':<32} | {'Capture Set (100)':<18} | {'KHATT Test':<18} | {'Synth v3 Clean':<18}"
    print(header)
    print("-" * 90)

    for label, key, fmt in metrics:
        cap_val = cap_stats[key]
        cap_str = fmt.format(cap_val[0] if key != "ink_fraction" else cap_val[0]*100,
                             cap_val[1] if key != "ink_fraction" else cap_val[1]*100)

        if khatt_stats:
            k_val = khatt_stats[key]
            k_str = fmt.format(k_val[0] if key != "ink_fraction" else k_val[0]*100,
                               k_val[1] if key != "ink_fraction" else k_val[1]*100)
        else:
            k_str = "N/A"

        if synth_stats:
            s_val = synth_stats[key]
            s_str = fmt.format(s_val[0] if key != "ink_fraction" else s_val[0]*100,
                               s_val[1] if key != "ink_fraction" else s_val[1]*100)
        else:
            s_str = "N/A"

        print(f"{label:<32} | {cap_str:<18} | {k_str:<18} | {s_str:<18}")

    print("=" * 90)


if __name__ == "__main__":
    main()
