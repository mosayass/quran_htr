"""
scripts/render_strokes.py
High-quality 4x supersampled stroke renderer for tablet ink.

Parameters:
- zoom: Magnification factor centered on baseline (default 1.4x)
- width_after_zoom: Stroke thickness in pixels in the target 64px image frame (default 1.8px)
- rotation_deg: Affine rotation in degrees ([-3, 3])
- shear_x: Horizontal shear factor ([-0.15, 0.15])
- wobble_amp: Amplitude of vertical baseline wobble (px)
- wobble_freq: Frequency of baseline wobble
- jitter_sigma: Gaussian noise added to point coordinates (px)
- seed: Random seed for deterministic reproducibility

Adheres strictly to dataset.py conventions:
- TARGET_HEIGHT = 64
- Grayscale mode 'L'
- White background (255), black ink (0)
"""

from __future__ import annotations
import math
import random
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from PIL import Image, ImageDraw

GUIDE_TOP_Y = 50.0
GUIDE_BASE_Y = 160.0
GUIDE_BOT_Y = 200.0
BAND_HEIGHT = GUIDE_BOT_Y - GUIDE_TOP_Y  # 150.0 px
TARGET_HEIGHT = 64


def render_strokes(
    sample_strokes: List[List[Dict[str, Any]]],
    zoom: float = 1.4,
    width_after_zoom: float = 1.8,
    rotation_deg: float = 0.0,
    shear_x: float = 0.0,
    wobble_amp: float = 0.0,
    wobble_freq: float = 0.05,
    jitter_sigma: float = 0.0,
    target_height: int = TARGET_HEIGHT,
    center_y: float = GUIDE_BASE_Y,
    antialiasing_factor: int = 4,
    pad_x: float = 16.0,
    seed: Optional[int] = None,
) -> Image.Image:
    """
    Renders stroke sequences into a 64px-height PIL Image using 4x supersampling.
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    all_pts = [p for s in sample_strokes for p in s]
    if not all_pts:
        return Image.new("L", (128, target_height), color=255)

    # 1. Stroke-level point jitter and wobble (in canvas coordinates)
    processed_strokes = []
    for stroke in sample_strokes:
        if not stroke:
            continue
        new_stroke = []
        for p in stroke:
            px = float(p["x"])
            py = float(p["y"])
            if jitter_sigma > 0.0:
                px += float(rng.normal(0.0, jitter_sigma))
                py += float(rng.normal(0.0, jitter_sigma))
            if wobble_amp > 0.0:
                py += wobble_amp * math.sin(px * wobble_freq)
            new_stroke.append({"x": px, "y": py})
        processed_strokes.append(new_stroke)

    proc_pts = [p for s in processed_strokes for p in s]
    xs = [p["x"] for p in proc_pts]
    min_x, max_x = min(xs), max(xs)

    # 2. Geometry & Crop bounds centered at center_y (baseline)
    window_h = BAND_HEIGHT / max(0.1, zoom)
    base_ratio = (center_y - GUIDE_TOP_Y) / BAND_HEIGHT
    crop_y1 = center_y - base_ratio * window_h
    crop_x1 = max(0.0, min_x - pad_x)
    crop_x2 = max_x + pad_x
    crop_w = max(32.0, crop_x2 - crop_x1)

    # 3. High-resolution canvas
    scale = (target_height * antialiasing_factor) / window_h
    canvas_w = max(32, int(round(crop_w * scale)))
    canvas_h = int(round(window_h * scale))

    w_draw = max(1.0, width_after_zoom * antialiasing_factor)
    r = w_draw / 2.0

    canvas = Image.new("L", (canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(canvas)

    for stroke in processed_strokes:
        if len(stroke) < 2:
            if stroke:
                p = stroke[0]
                px = (p["x"] - crop_x1) * scale
                py = (p["y"] - crop_y1) * scale
                draw.ellipse([px - r, py - r, px + r, py + r], fill=0)
            continue

        for i in range(len(stroke) - 1):
            p1 = stroke[i]
            p2 = stroke[i + 1]
            x1 = (p1["x"] - crop_x1) * scale
            y1 = (p1["y"] - crop_y1) * scale
            x2 = (p2["x"] - crop_x1) * scale
            y2 = (p2["y"] - crop_y1) * scale

            draw.line([(x1, y1), (x2, y2)], fill=0, width=int(round(w_draw)))
            draw.ellipse([x1 - r, y1 - r, x1 + r, y1 + r], fill=0)
            draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=0)

    # 4. Affine transformations: rotation and shear (if requested)
    if abs(rotation_deg) > 1e-3 or abs(shear_x) > 1e-3:
        # Affine transform: PIL Image.transform with AFFINE
        # x' = a*x + b*y + c, y' = d*x + e*y + f
        # Rotation around center:
        rad = math.radians(rotation_deg)
        cos_t, sin_t = math.cos(rad), math.sin(rad)
        cx, cy = canvas_w / 2.0, canvas_h / 2.0

        # Combine rotation and horizontal shear
        # forward: [x, y]^T -> [x - shear*y, y] then rotated
        # PIL expects inverse mapping matrix:
        a = cos_t
        b = -sin_t + shear_x * cos_t
        c = cx - cx * a - cy * b
        d = sin_t
        e = cos_t + shear_x * sin_t
        f = cy - cx * d - cy * e

        canvas = canvas.transform(
            (canvas_w, canvas_h),
            Image.Transform.AFFINE,
            (a, b, c, d, e, f),
            resample=Image.Resampling.BILINEAR,
            fillcolor=255,
        )

    # 5. Downsample to target_height=64
    final_w = max(32, int(round(canvas_w * (target_height / canvas_h))))
    return canvas.resize((final_w, target_height), Image.Resampling.BILINEAR)


def run_unit_test() -> bool:
    """
    Unit test: at capture-export settings (zoom=1.0, width=2.55px post-zoom),
    asserts pixel IoU >= 0.95 against clean exported PNGs in data/capture/.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    strokes_file = root / "data" / "capture" / "strokes.json"
    capture_dir = root / "data" / "capture"

    if not strokes_file.exists():
        print(f"[Unit Test Skipped] strokes.json not found at {strokes_file}")
        return True

    with open(strokes_file, encoding="utf-8") as f:
        strokes_data = json.load(f)

    strokes_by_id = {}
    for item in strokes_data:
        if isinstance(item, dict) and "id" in item:
            strokes_by_id[item["id"]] = item

    # Test on a representative clean exported sample
    test_id = "sample_line_020"
    if test_id not in strokes_by_id or not (capture_dir / f"{test_id}.png").exists():
        # Fallback to first available sample
        test_id = list(strokes_by_id.keys())[0]

    orig_png_path = capture_dir / f"{test_id}.png"
    orig_img = Image.open(orig_png_path).convert("L")
    orig_bin = np.array(orig_img) < 200

    item = strokes_by_id[test_id]
    rendered = render_strokes(
        item["strokes"],
        zoom=1.0,
        width_after_zoom=2.55,
        target_height=64,
        center_y=GUIDE_BASE_Y,
        antialiasing_factor=4,
    )

    # Resize to exact width of original for pixelwise comparison
    res_img = rendered.resize(orig_img.size, Image.Resampling.BILINEAR)
    res_bin = np.array(res_img) < 200

    intersection = np.logical_and(orig_bin, res_bin).sum()
    union = np.logical_or(orig_bin, res_bin).sum()
    iou = float(intersection) / float(union) if union > 0 else 0.0

    print(f"[Unit Test] {test_id} Pixel IoU at Capture-Export Settings: {iou:.4f} (Target: >= 0.95)")
    assert iou >= 0.95, f"Unit test failed: IoU {iou:.4f} < 0.95"
    return True


if __name__ == "__main__":
    run_unit_test()
