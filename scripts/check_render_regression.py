"""
scripts/check_render_regression.py
Task 2.12 Step 1:
Render regression test:
Compares 500 lines from the v2 pipeline output (data/shards/synth/synth_val_000.pt)
against re-rendering with the current renderer using the generator's calibrated pipeline
(same fonts and texts, f_size=30..42, spacing=0.7..1.4, bounds check).
Compares:
- Median stroke width
- Median ink fraction
- Median line width
Reports drift and flags if >10%.
"""

from __future__ import annotations
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, maximum_filter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from render_lines import render_arabic_text
from generate_synth_shards import augment_line_relative_thickness, check_line_metrics_fast

FONTS_DIR = ROOT / "data" / "fonts"
V2_SHARD_PATH = ROOT / "data" / "shards" / "synth" / "synth_val_000.pt"


def compute_metrics(arr: np.ndarray) -> tuple[float, float, int]:
    ink = arr < 128
    w = arr.shape[1]
    if not ink.any():
        return 0.0, 0.0, w
    cols = np.where(ink.sum(axis=0) > 0)[0]
    span_w = cols[-1] - cols[0] + 1
    ink_frac = float(ink.sum() / (64 * span_w))

    mid = (cols[0] + cols[-1]) // 2
    crop_w = 160
    c_start = max(cols[0], mid - crop_w // 2)
    c_end = min(cols[-1] + 1, c_start + crop_w)
    crop = ink[:, c_start:c_end]
    if not crop.any():
        return 0.0, ink_frac, w

    dist = distance_transform_edt(crop)
    loc_max = (dist == maximum_filter(dist, size=3)) & (dist > 0.5)
    if not loc_max.any():
        sw = 0.0
    else:
        sw = float(np.median(2.0 * dist[loc_max]))
    return sw, ink_frac, w


def main():
    print(f"Loading v2 reference shard from {V2_SHARD_PATH}...", flush=True)
    v2_data = torch.load(V2_SHARD_PATH, map_location="cpu", weights_only=False)
    n_samples = min(500, len(v2_data["images"]))
    print(f"Loaded {n_samples} samples for regression comparison.", flush=True)

    v2_sws, v2_fracs, v2_widths = [], [], []
    new_sws, new_fracs, new_widths = [], [], []

    rng = np.random.RandomState(42)

    for i in range(n_samples):
        v2_arr = v2_data["images"][i].numpy()
        sw_v2, fr_v2, w_v2 = compute_metrics(v2_arr)
        if sw_v2 > 0:
            v2_sws.append(sw_v2)
            v2_fracs.append(fr_v2)
            v2_widths.append(w_v2)

        text = v2_data["texts"][i]
        font_name = v2_data["fonts"][i]
        font_path = FONTS_DIR / f"{font_name}.ttf"
        if not font_path.exists():
            continue

        # Render with generator calibrated pipeline & rejection filter
        passed = False
        new_arr = None
        for _ in range(10):
            f_size = rng.randint(30, 42)
            spacing = rng.uniform(0.7, 1.4)
            raw = render_arabic_text(font_path, text, font_size=f_size, word_spacing_factor=spacing)
            apply_noise = rng.rand() < 0.60
            aug = augment_line_relative_thickness(raw, rng=rng, apply_noise=apply_noise)
            arr = np.array(aug)
            if check_line_metrics_fast(arr, 2.0, 4.0, 0.043, 0.110):
                new_arr = arr
                passed = True
                break

        if not passed:
            new_arr = arr

        sw_new, fr_new, w_new = compute_metrics(new_arr)
        if sw_new > 0:
            new_sws.append(sw_new)
            new_fracs.append(fr_new)
            new_widths.append(w_new)

    med_sw_v2 = float(np.median(v2_sws))
    med_fr_v2 = float(np.median(v2_fracs))
    med_w_v2 = float(np.median(v2_widths))

    med_sw_new = float(np.median(new_sws))
    med_fr_new = float(np.median(new_fracs))
    med_w_new = float(np.median(new_widths))

    drift_sw = abs(med_sw_new - med_sw_v2) / med_sw_v2 * 100
    drift_fr = abs(med_fr_new - med_fr_v2) / med_fr_v2 * 100
    drift_w = abs(med_w_new - med_w_v2) / med_w_v2 * 100

    print("\n" + "=" * 70)
    print("RENDER REGRESSION REPORT (500 SAMPLES: V2 vs CURRENT)")
    print("=" * 70)
    print(f"Metric              | v2 Reference | Current Renderer | Drift (%)")
    print("-" * 70)
    print(f"Median Stroke Width | {med_sw_v2:12.3f} | {med_sw_new:16.3f} | {drift_sw:8.2f}%")
    print(f"Median Ink Fraction | {med_fr_v2:12.4f} | {med_fr_new:16.4f} | {drift_fr:8.2f}%")
    print(f"Median Line Width   | {med_w_v2:12.1f} | {med_w_new:16.1f} | {drift_w:8.2f}%")
    print("=" * 70)

    max_drift = max(drift_sw, drift_fr, drift_w)
    if max_drift > 10.0:
        print(f"DRIFT DETECTED: Maximum drift {max_drift:.2f}% exceeds 10% threshold.")
    else:
        print(f"PASSED: Maximum drift {max_drift:.2f}% is within <=10% threshold.")


if __name__ == "__main__":
    main()
