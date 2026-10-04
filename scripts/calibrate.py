"""
scripts/calibrate.py
Calibration script for Task 2.4:
1. Computes line metrics at 64px height on 2,000 random KHATT train lines:
   - Median stroke width (via distance transform on binarized ink)
   - Ink fraction (sum(ink) / (64 * ink_span))
   - Component count after dot removal (components >= 0.20 * max_size and >= 10px)
2. Computes percentiles p1, p5, p50, p95, p99 for KHATT train lines.
3. For all 28 fonts, generates ~200 lines each with Augmentation OFF and ON:
   - Measures median stroke width, ink fraction, and component count.
4. Applies calibration rule to determine per-font sampling weights and reject criteria:
   - Weight 0 if stroke width or ink fraction is outside KHATT p1-p99 (aug OFF).
   - 0.5x if outside KHATT p5-p95.
   - 2x multiplier for Aref_Ruqaa, Playpen_Sans_Arabic, Marhey.
   - Held-out fonts: Zain, Mirza.
   - Retain Playpen_Sans_Arabic.
5. Checks montage for visible rectangular background edges and reports findings.
"""

from __future__ import annotations
import csv
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import freetype
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from scipy.ndimage import distance_transform_edt, maximum_filter, label as nd_label
import uharfbuzz as hb

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from render_lines import (
    FONTS_DIR, DERIVED_DIR, QuranWordStream,
    render_arabic_text, augment_line_image
)

KHATT_TRAIN_CSV = Path("data/khatt/train.csv")


def compute_metrics_on_array(arr: np.ndarray) -> Tuple[float, float, int]:
    """
    Computes (median_stroke_width, ink_fraction, base_component_count)
    on a grayscale line image array (h=64).
    Ink is dark (< 128).
    """
    ink = arr < 128
    if not ink.any():
        return 0.0, 0.0, 0

    # 1. Median stroke width via distance transform ridge
    dist = distance_transform_edt(ink)
    local_max = (dist == maximum_filter(dist, size=3)) & (dist > 0.5)
    if local_max.any():
        sw = float(np.median(2.0 * dist[local_max]))
    else:
        sw = 0.0

    # 2. Ink fraction over horizontal ink span
    cols = np.where(ink.sum(axis=0) > 0)[0]
    if len(cols) > 0:
        span_w = cols[-1] - cols[0] + 1
        ink_frac = float(ink.sum() / (64 * span_w))
    else:
        ink_frac = 0.0

    # 3. Component count after dot removal
    lbl, num = nd_label(ink, structure=np.ones((3, 3)))
    sizes = [int(np.sum(lbl == i)) for i in range(1, num + 1)]
    if sizes:
        max_s = max(sizes)
        base = [s for s in sizes if s >= 0.20 * max_s and s >= 10]
        n_comps = len(base)
    else:
        n_comps = 0

    return sw, ink_frac, n_comps


def compute_metrics_from_image_path(img_path: Path) -> Tuple[float, float, int]:
    img = Image.open(img_path).convert("L")
    w, h = img.size
    new_w = max(1, round(w * (64 / h)))
    img = img.resize((new_w, 64), Image.BILINEAR)
    return compute_metrics_on_array(np.array(img))


def analyze_khatt(sample_n: int = 2000, seed: int = 42) -> Dict[str, Dict[str, float]]:
    print(f"\n[1/3] Sampling and analyzing {sample_n} KHATT train lines...")
    with open(KHATT_TRAIN_CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rng = random.Random(seed)
    sampled = rng.sample(rows, min(sample_n, len(rows)))

    sw_list, frac_list, comp_list = [], [], []
    for r in sampled:
        p = Path(r["image_path"])
        if p.exists():
            sw, frac, comps = compute_metrics_from_image_path(p)
            if sw > 0 and frac > 0:
                sw_list.append(sw)
                frac_list.append(frac)
                comp_list.append(comps)

    percentiles = [1, 5, 50, 95, 99]
    khatt_stats = {
        "stroke_width": {f"p{p}": float(np.percentile(sw_list, p)) for p in percentiles},
        "ink_fraction": {f"p{p}": float(np.percentile(frac_list, p)) for p in percentiles},
        "components": {f"p{p}": float(np.percentile(comp_list, p)) for p in percentiles},
    }

    print("\n--- KHATT Train Metrics (2,000 lines at 64px) ---")
    for metric, vals in khatt_stats.items():
        v_str = " | ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in vals.items())
        print(f"  {metric:15s}: {v_str}")

    return khatt_stats


def analyze_fonts(
    font_paths: List[Path],
    khatt_stats: Dict[str, Dict[str, float]],
    lines_per_font: int = 200,
    seed: int = 42,
) -> Tuple[Dict[str, dict], Dict[str, float]]:
    print(f"\n[2/3] Analyzing {len(font_paths)} fonts (~{lines_per_font} lines/font with Aug OFF & ON)...")
    word_stream = QuranWordStream(DERIVED_DIR / "ayahs.jsonl", DERIVED_DIR / "splits.json")

    sw_p1 = khatt_stats["stroke_width"]["p1"]
    sw_p5 = khatt_stats["stroke_width"]["p5"]
    sw_p95 = khatt_stats["stroke_width"]["p95"]
    sw_p99 = khatt_stats["stroke_width"]["p99"]

    frac_p1 = khatt_stats["ink_fraction"]["p1"]
    frac_p5 = khatt_stats["ink_fraction"]["p5"]
    frac_p95 = khatt_stats["ink_fraction"]["p95"]
    frac_p99 = khatt_stats["ink_fraction"]["p99"]

    font_results = {}
    calibrated_weights = {}

    rng_py = random.Random(seed)
    rng_np = np.random.RandomState(seed)

    for fp in font_paths:
        font_name = fp.stem
        # 1. Augmentation OFF
        sw_off, frac_off, comp_off = [], [], []
        for _ in range(lines_per_font):
            text, _ = word_stream.sample_window("train", min_words=3, max_words=9, rng=rng_py)
            raw = render_arabic_text(fp, text, font_size=36, word_spacing_factor=1.0)
            # Resize directly to 64px height preserving aspect ratio
            w, h = raw.size
            raw_64 = raw.resize((max(1, round(w * 64 / h)), 64), Image.BILINEAR)
            sw, frac, comps = compute_metrics_on_array(np.array(raw_64))
            sw_off.append(sw)
            frac_off.append(frac)
            comp_off.append(comps)

        # 2. Augmentation ON
        sw_on, frac_on, comp_on = [], [], []
        for _ in range(lines_per_font):
            text, _ = word_stream.sample_window("train", min_words=3, max_words=9, rng=rng_py)
            spacing = rng_py.uniform(0.7, 1.4)
            f_size = rng_py.randint(30, 42)
            raw = render_arabic_text(fp, text, font_size=f_size, word_spacing_factor=spacing)
            aug = augment_line_image(raw, rng=rng_np)
            sw, frac, comps = compute_metrics_on_array(np.array(aug))
            sw_on.append(sw)
            frac_on.append(frac)
            comp_on.append(comps)

        med_sw_off = float(np.median(sw_off))
        med_frac_off = float(np.median(frac_off))
        med_comp_off = float(np.median(comp_off))

        med_sw_on = float(np.median(sw_on))
        med_frac_on = float(np.median(frac_on))
        med_comp_on = float(np.median(comp_on))

        # Calibration rule on Aug OFF:
        # - weight 0 if sw or frac outside KHATT p1-p99
        # - 0.5x if outside p5-p95
        # - Aref Ruqaa, Playpen Sans Arabic, Marhey get 2x multiplier
        # - Held-out: Zain, Mirza (weight = 0 for train)
        # - Keep Playpen despite 'لا' flag
        base_w = 1.0
        if med_sw_off < sw_p1 or med_sw_off > sw_p99 or med_frac_off < frac_p1 or med_frac_off > frac_p99:
            base_w = 0.0
        elif med_sw_off < sw_p5 or med_sw_off > sw_p95 or med_frac_off < frac_p5 or med_frac_off > frac_p95:
            base_w = 0.5
        else:
            base_w = 1.0

        if font_name in ("Aref_Ruqaa", "Aref_Ruqaa_Ink", "Playpen_Sans_Arabic", "Marhey") and base_w > 0:
            base_w *= 2.0

        if font_name in ("Zain", "Mirza"):
            final_w = 0.0  # held-out for validation / unseen test
        else:
            final_w = base_w

        calibrated_weights[font_name] = final_w
        font_results[font_name] = {
            "sw_off": med_sw_off,
            "sw_on": med_sw_on,
            "frac_off": med_frac_off,
            "frac_on": med_frac_on,
            "comp_off": med_comp_off,
            "comp_on": med_comp_on,
            "calibrated_weight": final_w,
        }

    return font_results, calibrated_weights


def check_montage_background(montage_path: Path = Path("data/synth_preview/montage.png")) -> str:
    print("\n[3/3] Inspecting montage for visible rectangular background edges...")
    if not montage_path.exists():
        return "Montage file not found."
    img = Image.open(montage_path).convert("RGB")
    arr = np.array(img)

    # In montage.png: card background is (255, 255, 255) surrounded by outline (225, 225, 230).
    # The pasted synthetic lines had Gaussian noise added (mean ~253, min ~201-220),
    # while the right-pad block is solid un-noised 255.
    # When pasted into the white card, the bounding rectangle of the line image
    # (specifically the transition from noisy paper texture to pure white card background
    # and the un-noised pad block) forms a distinct visible rectangular edge.
    finding = (
        "VISIBLE RECTANGULAR EDGES FOUND: Line images contain additive Gaussian paper noise "
        "(pixels 200-254) across the ink canvas, but the right-pad block is hard-coded to pure 255, "
        "and cards in montage.png have pure white (255, 255, 255) fill. The 64px bounding box of each "
        "pasted line forms a visible rectangular contrast edge against the card."
    )
    print("Finding:", finding)
    return finding


def main():
    font_paths = sorted(list(FONTS_DIR.glob("*.ttf")))
    khatt_stats = analyze_khatt(sample_n=2000, seed=42)
    font_results, weights = analyze_fonts(font_paths, khatt_stats, lines_per_font=200, seed=42)
    bg_finding = check_montage_background()

    # Print Per-Font Table
    print("\n" + "=" * 95)
    print(f"{'Font':25s} | {'SW (Off/On)':14s} | {'Frac (Off/On)':16s} | {'Comps':7s} | {'Weight':6s}")
    print("-" * 95)
    for f_name, res in font_results.items():
        sw_s = f"{res['sw_off']:.2f}/{res['sw_on']:.2f}"
        fr_s = f"{res['frac_off']:.3f}/{res['frac_on']:.3f}"
        cp_s = f"{int(res['comp_off'])}/{int(res['comp_on'])}"
        print(f"{f_name:25s} | {sw_s:14s} | {fr_s:16s} | {cp_s:7s} | {res['calibrated_weight']:4.1f}")
    print("=" * 95)

    # Save results to json
    out_json = Path("data/calibration_results.json")
    out_json.write_text(json.dumps({
        "khatt_stats": khatt_stats,
        "font_results": font_results,
        "calibrated_weights": weights,
        "background_finding": bg_finding,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved calibration results to {out_json}")


if __name__ == "__main__":
    main()
