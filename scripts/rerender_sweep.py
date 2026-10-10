"""
scripts/rerender_sweep.py
Task 2.14 Item 3:
Re-render sweep from strokes.json across 7 geometry/stroke policies:
  (a) current ruled-band crop (Y=50..200, stroke width 2.8)
  (b) tight vertical crop to ink extent + 8% margin
  (c) band crop zoomed 1.3x and 1.6x about the baseline (Y=160)
  (d) stroke width 2.0 / 2.8 / 3.6 (ruled-band)
  (e) pressure-varying width using pt['p'] (1.5 to 5.0 px)

Renders all 100 samples for each variant into data/capture/sweep/<variant>/
and scores best.pt on all 100 per variant.
"""

from __future__ import annotations
import sys
import json
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vocab import Vocabulary
from model import CRNN
from dataset import LineImageDataset, collate_fn
from evaluate import greedy_decode
from metrics import corpus_cer_wer

# Guidelines from capture/index.html (nominal unscaled canvas coordinates)
GUIDE_TOP_Y = 50.0
GUIDE_BASE_Y = 160.0
GUIDE_BOT_Y = 200.0
BAND_HEIGHT = GUIDE_BOT_Y - GUIDE_TOP_Y  # 150.0 px


def render_sample(
    sample_strokes: list[list[dict]],
    variant: str,
    target_height: int = 64,
) -> Image.Image:
    """
    Renders strokes according to variant policy and returns a PIL grayscale image of height 64.
    """
    # 1. Compute ink bounding box
    all_pts = [p for stroke in sample_strokes for p in stroke]
    if not all_pts:
        return Image.new("L", (128, target_height), color=255)

    xs = [p["x"] for p in all_pts]
    ys = [p["y"] for p in all_pts]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 2. Determine crop window and stroke parameters based on variant
    pad_x = 16.0
    crop_x1 = max(0.0, min_x - pad_x)
    crop_x2 = max_x + pad_x
    crop_w = max(32.0, crop_x2 - crop_x1)

    stroke_width_nominal = 2.8 * (BAND_HEIGHT / target_height)  # ~6.56 nominal canvas px
    use_pressure = False

    if variant == "current_ruled_band":
        crop_y1 = GUIDE_TOP_Y
        crop_y2 = GUIDE_BOT_Y
        stroke_width_nominal = 2.8 * (BAND_HEIGHT / target_height)

    elif variant == "tight_crop_8pct":
        # Tight to ink vertical extent + 8% margin
        ink_h = max(10.0, max_y - min_y)
        margin = ink_h * 0.08
        crop_y1 = min_y - margin
        crop_y2 = max_y + margin
        crop_h = crop_y2 - crop_y1
        # Stroke width scaled so it becomes 2.8 at target_height
        stroke_width_nominal = 2.8 * (crop_h / target_height)

    elif variant == "band_zoom_1.3x":
        # Zoom 1.3x about baseline
        zoom = 1.3
        new_h = BAND_HEIGHT / zoom
        # baseline sits at 110 px from top of band (160 - 50 = 110)
        base_offset_ratio = (GUIDE_BASE_Y - GUIDE_TOP_Y) / BAND_HEIGHT
        crop_y1 = GUIDE_BASE_Y - base_offset_ratio * new_h
        crop_y2 = crop_y1 + new_h
        stroke_width_nominal = 2.8 * (new_h / target_height)

    elif variant == "band_zoom_1.6x":
        zoom = 1.6
        new_h = BAND_HEIGHT / zoom
        base_offset_ratio = (GUIDE_BASE_Y - GUIDE_TOP_Y) / BAND_HEIGHT
        crop_y1 = GUIDE_BASE_Y - base_offset_ratio * new_h
        crop_y2 = crop_y1 + new_h
        stroke_width_nominal = 2.8 * (new_h / target_height)

    elif variant == "stroke_w2.0":
        crop_y1 = GUIDE_TOP_Y
        crop_y2 = GUIDE_BOT_Y
        stroke_width_nominal = 2.0 * (BAND_HEIGHT / target_height)

    elif variant == "stroke_w3.6":
        crop_y1 = GUIDE_TOP_Y
        crop_y2 = GUIDE_BOT_Y
        stroke_width_nominal = 3.6 * (BAND_HEIGHT / target_height)

    elif variant == "pressure_varying":
        crop_y1 = GUIDE_TOP_Y
        crop_y2 = GUIDE_BOT_Y
        use_pressure = True

    else:
        raise ValueError(f"Unknown variant: {variant}")

    crop_h = max(10.0, crop_y2 - crop_y1)

    # 3. Create unscaled canvas for crisp rendering
    # Render at 2x resolution, then downscale smoothly with Lanczos/Bilinear
    render_scale = (target_height * 2.0) / crop_h
    out_w = max(32, int(round(crop_w * render_scale)))
    out_h = int(round(crop_h * render_scale))

    canvas = Image.new("L", (out_w, out_h), color=255)
    draw = ImageDraw.Draw(canvas)

    for stroke in sample_strokes:
        if len(stroke) < 2:
            if stroke:
                p = stroke[0]
                px = (p["x"] - crop_x1) * render_scale
                py = (p["y"] - crop_y1) * render_scale
                r = (stroke_width_nominal * render_scale) / 2.0
                draw.ellipse([px - r, py - r, px + r, py + r], fill=0)
            continue

        for i in range(len(stroke) - 1):
            p1 = stroke[i]
            p2 = stroke[i + 1]
            x1 = (p1["x"] - crop_x1) * render_scale
            y1 = (p1["y"] - crop_y1) * render_scale
            x2 = (p2["x"] - crop_x1) * render_scale
            y2 = (p2["y"] - crop_y1) * render_scale

            if use_pressure:
                p_val = p2.get("p", 0.5)
                # Map pressure [0.1..1.0] to [1.5..4.5] target pixels
                pw = (1.5 + p_val * 3.0) * (BAND_HEIGHT / target_height)
                w_px = max(2.0, pw * render_scale)
            else:
                w_px = max(2.0, stroke_width_nominal * render_scale)

            draw.line([(x1, y1), (x2, y2)], fill=0, width=int(round(w_px)))
            # Round joints
            r = w_px / 2.0
            draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=0)

    # 4. Resize to target_height=64
    final_w = max(32, round(out_w * (target_height / out_h)))
    final_img = canvas.resize((final_w, target_height), Image.BILINEAR)
    return final_img


def generate_and_evaluate_sweep(
    checkpoint_path: Path,
    vocab_path: Path = Path("vocab.json"),
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    strokes_file = ROOT / "data" / "capture" / "strokes.json"
    with open(strokes_file, encoding="utf-8") as f:
        raw_strokes = json.load(f)

    # Deduplicate strokes by keeping last occurrence of each id
    strokes_by_id = {}
    for item in raw_strokes:
        if isinstance(item, dict) and "id" in item:
            strokes_by_id[item["id"]] = item

    sample_ids = sorted(strokes_by_id.keys())
    print(f"Loaded and deduplicated {len(sample_ids)} stroke samples.")

    # Load model and vocab
    device = torch.device(device_str)
    vocab = Vocabulary.load(vocab_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"Loaded model {checkpoint_path.name} on {device} (order={order}).")

    variants = [
        ("current_ruled_band", "Ruled Band Crop (Current, 2.8px)"),
        ("tight_crop_8pct",    "Tight Vertical Crop (+8% Margin)"),
        ("band_zoom_1.3x",     "Band Crop Zoomed 1.3x (Baseline Centered)"),
        ("band_zoom_1.6x",     "Band Crop Zoomed 1.6x (Baseline Centered)"),
        ("stroke_w2.0",        "Stroke Width 2.0px (Ruled Band)"),
        ("stroke_w3.6",        "Stroke Width 3.6px (Ruled Band)"),
        ("pressure_varying",   "Pressure-Varying Width (1.5-4.5px)"),
    ]

    sweep_dir = ROOT / "data" / "capture" / "sweep"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    for var_key, var_title in variants:
        v_dir = sweep_dir / var_key
        v_dir.mkdir(parents=True, exist_ok=True)
        manifest_rows = [["sample_id", "image_path", "transcription"]]

        # Render all 100
        for sid in sample_ids:
            item = strokes_by_id[sid]
            img = render_sample(item["strokes"], var_key, target_height=64)
            img_path = v_dir / f"{sid}.png"
            img.save(img_path)
            manifest_rows.append([sid, str(img_path), item["text"]])

        v_manifest = v_dir / "manifest.csv"
        with open(v_manifest, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(manifest_rows)

        # Evaluate
        ds = LineImageDataset(v_manifest, vocab, augment=False, label_order=order)
        loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

        preds, targets = [], []
        with torch.no_grad():
            for batch in loader:
                imgs = batch["images"].to(device)
                log_probs = model(imgs)
                batch_preds = greedy_decode(log_probs, vocab, label_order=order)
                preds.extend(batch_preds)
                targets.extend(batch["texts"])

        cer, wer = corpus_cer_wer(preds, targets)
        results[var_key] = (cer * 100, wer * 100)
        print(f"  [{var_title:<45}] CER: {cer*100:5.2f}% | WER: {wer*100:5.2f}%")

    print("\n" + "=" * 80)
    print("TASK 2.14 ITEM 3: RE-RENDER SWEEP RESULTS (best.pt on 100 Tablet Samples)")
    print("=" * 80)
    print(f"{'Variant / Policy':<48} | {'CER (%)':<10} | {'WER (%)':<10}")
    print("-" * 80)
    for var_key, var_title in variants:
        c, w = results[var_key]
        print(f"{var_title:<48} | {c:5.2f}%    | {w:5.2f}%")
    print("=" * 80)
    return results


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/step2b_v3/best.pt",
                   help="Path to best.pt checkpoint")
    p.add_argument("--vocab_path", default="vocab.json")
    args = p.parse_args()

    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.exists():
        # Fallback search
        for alt in [
            Path("checkpoints/best.pt"),
            Path("/content/drive/MyDrive/quran_htr_writer_checkpoints/step2b_v3/best.pt"),
        ]:
            if alt.exists():
                ckpt_p = alt
                break

    if not ckpt_p.exists():
        print(f"Error: Checkpoint {args.checkpoint} not found. Please specify --checkpoint PATH")
        sys.exit(1)

    generate_and_evaluate_sweep(ckpt_p, Path(args.vocab_path))


if __name__ == "__main__":
    main()
