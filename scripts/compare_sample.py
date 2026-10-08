import sys
from pathlib import Path
import torch
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from render_lines import render_arabic_text
from generate_synth_shards import augment_line_relative_thickness
from scipy.ndimage import distance_transform_edt, maximum_filter

def get_metrics(arr):
    ink = arr < 128
    if not ink.any():
        return 0.0, 0.0, arr.shape[1]
    cols = np.where(ink.sum(axis=0) > 0)[0]
    span_w = cols[-1] - cols[0] + 1
    ink_frac = float(ink.sum() / (64 * span_w))
    dist = distance_transform_edt(ink)
    loc_max = (dist == maximum_filter(dist, size=3)) & (dist > 0.5)
    sw = float(np.median(2.0 * dist[loc_max])) if loc_max.any() else 0.0
    return sw, ink_frac, arr.shape[1]

v2 = torch.load("data/shards/synth/synth_val_000.pt", map_location="cpu", weights_only=False)
rng = np.random.RandomState(42)

for i in range(5):
    txt = v2["texts"][i]
    font = v2["fonts"][i]
    fp = Path("data/fonts") / f"{font}.ttf"
    raw = render_arabic_text(fp, txt, font_size=36)
    aug = augment_line_relative_thickness(raw, rng=rng, apply_noise=True)
    v2_sw, v2_fr, v2_w = get_metrics(v2["images"][i].numpy())
    new_sw, new_fr, new_w = get_metrics(np.array(aug))
    print(f"Sample {i} ({font}): v2 sw={v2_sw:.2f}, new sw={new_sw:.2f} | v2 fr={v2_fr:.3f}, new fr={new_fr:.3f} | v2 w={v2_w}, new w={new_w}")
