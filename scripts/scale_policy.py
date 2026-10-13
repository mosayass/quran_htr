"""
scripts/scale_policy.py
Label-free, writer-level global-target scale normalization (docs/capture_and_preprocessing_spec.md,
"Scale Normalization Formula (Global Target Fill)"), with an exact never-clip bound.

For a writer's unlabeled samples (strokes only, no transcripts):

  H_ink   = median over samples of total vertical ink extent (CSS px)
  H_inkmx = max over samples of total vertical ink extent
  px_per_css(z) = z * 64 / H_band
  S_target  = 48 / (H_ink   * 64 / H_band)          # one GLOBAL target: ink fills 48 px of the 64 px tensor
  S_formula = 58 / (H_inkmx * 64 / H_band)          # spec's cap for the largest sample
  S_clip    = min over samples of the exact baseline-anchored bound (render_strokes zooms about the
              baseline): z <= (base_ratio - m) * H_band / A  and  z <= (1 - base_ratio - m) * H_band / D,
              A/D = ink extent above/below baseline, m = stroke half-width margin (1.5 px / 64)
  S_w = min(S_target, S_formula, S_clip)
"""

from __future__ import annotations
from typing import Dict, List, Tuple

import numpy as np

MARGIN_PX = 2.5
TARGET_PX = 48.0
CAP_PX = 58.0


def sample_band(s: dict) -> Tuple[float, float, float]:
    """(band_height, baseline_y, band_top_y) in the renderer's coordinates (matches get_canonical_zoom_and_render)."""
    geom = s.get("geometry") or {}
    if geom:
        bt = float(geom.get("band_top_css", 0.0))
        bb = float(geom.get("band_base_css", 147.0))
        bot = float(geom.get("band_bot_css", 180.0))
        bh = float(geom.get("band_height_css", bot - bt))
        return bh, bb - bt, 0.0
    return 150.0, 160.0, 50.0


def canonical_zoom_of(s: dict) -> float:
    return 1.0 if (s.get("geometry") or {}) else 150.0 / 108.0


def ink_extents(s: dict) -> Tuple[float, float, float]:
    """(above_baseline, below_baseline, total_extent) of the ink in CSS px."""
    bh, cy, _ = sample_band(s)
    ys = [float(p["y"]) for st in s["strokes"] for p in st]
    if not ys:
        return 0.0, 0.0, 0.0
    return max(0.0, cy - min(ys)), max(0.0, max(ys) - cy), max(ys) - min(ys)


def clip_bound(s: dict) -> float:
    """Largest zoom at which this sample's ink (plus stroke margin) is not clipped vertically."""
    bh, cy, bt = sample_band(s)
    base_ratio = (cy - bt) / bh
    a, d, _ = ink_extents(s)
    m = MARGIN_PX / 64.0
    bounds = []
    if a > 0:
        bounds.append((base_ratio - m) * bh / a)
    if d > 0:
        bounds.append((1.0 - base_ratio - m) * bh / d)
    return min(bounds) if bounds else float("inf")


def writer_scale(samples: List[dict]) -> Dict[str, float]:
    bh = float(np.median([sample_band(s)[0] for s in samples]))
    totals = [ink_extents(s)[2] for s in samples]
    h_med, h_max = float(np.median(totals)), float(np.max(totals))
    s_target = TARGET_PX / (h_med * 64.0 / bh)
    s_formula = CAP_PX / (h_max * 64.0 / bh)
    s_clip = min(clip_bound(s) for s in samples)
    return {"band_height": bh, "h_ink_median": h_med, "h_ink_max": h_max, "S_target": s_target,
            "S_formula58": s_formula, "S_clip_exact": s_clip, "S_w": min(s_target, s_formula, s_clip)}


def is_clipped(img_arr: np.ndarray, thresh: int = 128) -> bool:
    """Empirical check on a rendered (64, W) uint8 image: ink touching the top or bottom row."""
    return bool(img_arr[0, :].min() < thresh or img_arr[-1, :].min() < thresh)
