"""
scripts/task_2_15_suite.py
Task 2.15 Implementation Suite:
1. Joint Grid on Capture Strokes:
   - Zoom in {1.0, 1.2, 1.4, 1.6, 1.8} x Stroke Width in {1.4, 1.8, 2.2, 2.6} px (width applied post-zoom)
   - Paired bootstrap 95% CIs vs current baseline policy (B=1000)
   - Pixelwise reconciliation analysis
2. estimate_body_height(strokes):
   - Baseline and body-height estimation from stroke points alone
   - Clamp zoom to [1.0, 2.2]
   - Evaluate target body height in {20, 22, 24} px vs fixed zoom (reporting words and lines separately)
3. Closed-set verification on capture set with best preprocessing:
   - (a) 60 words: rank of true word among all ~14.8k unique Quran words by CTC log-likelihood; top-1/5/10 & median rank
   - (b) 40 lines: 1 corrupted variant per line; accuracy (logP(true) > logP(corrupt)), plus FAR/FRR at 3 thresholds
"""

from __future__ import annotations
import sys
import json
import csv
import random
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vocab import Vocabulary
from model import CRNN
from dataset import collate_fn, Sample, LineImageDataset
from evaluate import greedy_decode
from metrics import corpus_cer_wer, cer as compute_single_cer

# Guideline constants
GUIDE_TOP_Y = 50.0
GUIDE_BASE_Y = 160.0
GUIDE_BOT_Y = 200.0
BAND_HEIGHT = GUIDE_BOT_Y - GUIDE_TOP_Y  # 150.0 px


# =========================================================================
# 1. RENDERING UTILITIES
# =========================================================================

def render_strokes_custom(
    sample_strokes: list[list[dict]],
    zoom: float,
    stroke_width_px: float,
    target_height: int = 64,
    center_y: float = GUIDE_BASE_Y,
    antialiasing_factor: int = 4,
) -> Image.Image:
    """
    Renders strokes with decoupled zoom and post-zoom stroke width.
    zoom: magnification factor about center_y (baseline).
    stroke_width_px: stroke width directly in the final 64px image coordinate frame.
    """
    all_pts = [p for stroke in sample_strokes for p in stroke]
    if not all_pts:
        return Image.new("L", (128, target_height), color=255)

    xs = [p["x"] for p in all_pts]
    min_x, max_x = min(xs), max(xs)

    # Window in unscaled canvas coordinates
    window_h = BAND_HEIGHT / zoom
    base_ratio = (center_y - GUIDE_TOP_Y) / BAND_HEIGHT
    crop_y1 = center_y - base_ratio * window_h
    crop_y2 = crop_y1 + window_h

    pad_x = 16.0
    crop_x1 = max(0.0, min_x - pad_x)
    crop_x2 = max_x + pad_x
    crop_w = max(32.0, crop_x2 - crop_x1)

    # High-resolution supersampling for smooth antialiased strokes
    scale = (target_height * antialiasing_factor) / window_h
    canvas_w = max(32, int(round(crop_w * scale)))
    canvas_h = int(round(window_h * scale))

    # Width applied strictly post-zoom (scaled to supersampled canvas)
    w_draw = max(1.0, stroke_width_px * antialiasing_factor)

    canvas = Image.new("L", (canvas_w, canvas_h), color=255)
    draw = ImageDraw.Draw(canvas)

    for stroke in sample_strokes:
        if len(stroke) < 2:
            if stroke:
                p = stroke[0]
                px = (p["x"] - crop_x1) * scale
                py = (p["y"] - crop_y1) * scale
                r = w_draw / 2.0
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
            r = w_draw / 2.0
            draw.ellipse([x2 - r, y2 - r, x2 + r, y2 + r], fill=0)

    # Resize down to target_height=64 using high-quality Lanczos / Bilinear
    final_w = max(32, int(round(canvas_w * (target_height / canvas_h))))
    return canvas.resize((final_w, target_height), Image.Resampling.LANCZOS)


# =========================================================================
# 2. BODY HEIGHT & BASELINE ESTIMATOR
# =========================================================================

def estimate_body_height(strokes: list[list[dict]]) -> tuple[float, float, float]:
    """
    Estimates baseline Y and core body height from stroke points alone.
    Returns (estimated_baseline_y, estimated_body_height_px, suggested_zoom).
    """
    all_pts = [p for stroke in strokes for p in stroke]
    if not all_pts:
        return GUIDE_BASE_Y, 20.0, 1.0

    ys = np.array([p["y"] for p in all_pts], dtype=float)
    y_min, y_max = ys.min(), ys.max()

    # Kernel density / histogram of Y coordinates
    bins = np.linspace(y_min, y_max, num=min(50, max(10, int(y_max - y_min + 1))))
    hist, edges = np.histogram(ys, bins=bins)
    centers = (edges[:-1] + edges[1:]) / 2.0

    # Baseline is typically near the dense lower peak of strokes (above descenders)
    # 75th percentile of stroke points is a robust baseline proxy
    baseline_y = float(np.percentile(ys, 75))

    # Body height is distance between main upper connection line (~25th pct) and baseline (~75th pct)
    p25 = float(np.percentile(ys, 20))
    p80 = float(np.percentile(ys, 80))
    body_h = max(8.0, p80 - p25)

    # Desired on-screen body height ~22px out of 64px
    # In nominal canvas: BAND_HEIGHT = 150px. At zoom=1, body_h * (64 / 150) = body_h * 0.426.
    # To achieve target body height H_tgt: zoom = H_tgt / (body_h * (64 / 150))
    target_body = 22.0
    nominal_scaled = body_h * (64.0 / BAND_HEIGHT)
    suggested_zoom = float(np.clip(target_body / max(1.0, nominal_scaled), 1.0, 2.2))

    return baseline_y, body_h, suggested_zoom


# =========================================================================
# 3. EVALUATION HELPER (In-Memory Batch Evaluation)
# =========================================================================

def evaluate_in_memory_images(
    images_and_texts: list[tuple[Image.Image, str]],
    model: nn.Module,
    vocab: Vocabulary,
    device: torch.device,
    order: str = "visual",
    batch_size: int = 32,
) -> tuple[float, float, list[float], list[str], list[str]]:
    """
    Evaluates in-memory PIL images and texts.
    Returns (overall_cer, overall_wer, list_of_sample_cers, preds, targets).
    """
    model.eval()
    samples = []
    for img, text in images_and_texts:
        w, h = img.size
        # Normalize to [-1, 1]
        tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        # Pad width to multiple of 32
        pad_w = (-w) % 32
        if pad_w:
            tensor = nn.functional.pad(tensor, (0, pad_w), value=1.0)

        label_text = text[::-1] if order == "visual" else text
        target = torch.tensor(vocab.encode(label_text), dtype=torch.long)
        samples.append(Sample(image=tensor, target=target, text=text))

    all_preds, all_targets = [], []
    sample_cers = []

    for i in range(0, len(samples), batch_size):
        batch_samples = samples[i:i + batch_size]
        batch = collate_fn(batch_samples)
        with torch.no_grad():
            imgs = batch["images"].to(device)
            log_probs = model(imgs)
            preds = greedy_decode(log_probs, vocab, label_order=order)
            all_preds.extend(preds)
            all_targets.extend(batch["texts"])

    for p, t in zip(all_preds, all_targets):
        sample_cers.append(compute_single_cer(p, t))

    cer, wer = corpus_cer_wer(all_preds, all_targets)
    return cer * 100.0, wer * 100.0, sample_cers, all_preds, all_targets


def paired_bootstrap_ci(
    base_cers: list[float],
    cand_cers: list[float],
    b_iters: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """
    Computes paired bootstrap 95% confidence interval for delta CER = cand_cer - base_cer.
    Returns (mean_delta, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    n = len(base_cers)
    deltas = []
    for _ in range(b_iters):
        idx = rng.integers(0, n, size=n)
        d_mean = np.mean(np.array(cand_cers)[idx] - np.array(base_cers)[idx])
        deltas.append(d_mean * 100.0)

    deltas = np.sort(deltas)
    lo = float(np.percentile(deltas, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(deltas, 100.0 * (1.0 - alpha / 2.0)))
    mean_d = float(np.mean(deltas))
    return mean_d, lo, hi


# =========================================================================
# 4. MAIN TASK 2.15 RUNNER
# =========================================================================

def run_task_2_15(
    checkpoint_path: Path,
    vocab_path: Path = Path("vocab.json"),
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    vocab = Vocabulary.load(vocab_path)
    print(f"Device: {device} | Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load 100 deduplicated samples from strokes.json and manifest
    strokes_file = ROOT / "data" / "capture" / "strokes.json"
    with open(strokes_file, encoding="utf-8") as f:
        raw_strokes = json.load(f)

    strokes_by_id = {}
    for item in raw_strokes:
        if isinstance(item, dict) and "id" in item:
            strokes_by_id[item["id"]] = item
    sample_ids = sorted(strokes_by_id.keys())
    print(f"Loaded {len(sample_ids)} deduplicated capture samples.")

    # Load original exported images for baseline comparison
    orig_pairs = []
    manifest_csv = ROOT / "data" / "capture" / "manifest.csv"
    manifest_rows = {}
    with open(manifest_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            manifest_rows[r["sample_id"]] = r

    for sid in sample_ids:
        r = manifest_rows[sid]
        img_p = ROOT / "data" / "capture" / r["image_file"]
        if not img_p.exists():
            img_p = Path(r["image_file"])
        orig_pairs.append((Image.open(img_p).convert("L"), r["transcription"]))

    base_cer, base_wer, base_cers, _, _ = evaluate_in_memory_images(
        orig_pairs, model, vocab, device, order=order
    )
    print(f"\n[Baseline Exported PNGs (Current Policy)] CER: {base_cer:.2f}% | WER: {base_wer:.2f}%")

    # ---------------------------------------------------------------------
    # ITEM 1: JOINT GRID (5 Zooms x 4 Stroke Widths)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ITEM 1: JOINT GRID SWEEP (Zoom x Stroke Width Post-Zoom)")
    print("=" * 90)

    zooms = [1.0, 1.2, 1.4, 1.6, 1.8]
    stroke_widths = [1.4, 1.8, 2.2, 2.6]

    grid_results = {}
    best_grid_cer = float("inf")
    best_grid_setting = None

    print(f"{'Zoom':<6} | {'Width (px)':<12} | {'CER (%)':<10} | {'WER (%)':<10} | {'ΔCER vs Base':<15} | {'95% CI':<18}")
    print("-" * 90)

    for z in zooms:
        for sw in stroke_widths:
            cand_pairs = []
            for sid in sample_ids:
                item = strokes_by_id[sid]
                img = render_strokes_custom(item["strokes"], zoom=z, stroke_width_px=sw)
                cand_pairs.append((img, item["text"]))

            c_cer, c_wer, c_cers, _, _ = evaluate_in_memory_images(
                cand_pairs, model, vocab, device, order=order
            )
            mean_d, ci_lo, ci_hi = paired_bootstrap_ci(base_cers, c_cers)
            grid_results[(z, sw)] = (c_cer, c_wer, mean_d, ci_lo, ci_hi)

            if c_cer < best_grid_cer:
                best_grid_cer = c_cer
                best_grid_setting = (z, sw)

            ci_str = f"[{ci_lo:+.2f}%, {ci_hi:+.2f}%]"
            d_str = f"{mean_d:+.2f}%"
            print(f"{z:<6.1f} | {sw:<12.1f} | {c_cer:<10.2f} | {c_wer:<10.2f} | {d_str:<15} | {ci_str:<18}")

    print(f"\n--> Best Grid Configuration: Zoom {best_grid_setting[0]:.1f}x, Width {best_grid_setting[1]:.1f}px -> CER {best_grid_cer:.2f}%")

    # ---------------------------------------------------------------------
    # ITEM 2: BODY-HEIGHT ADAPTIVE NORMALIZATION
    # ---------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ITEM 2: BODY-HEIGHT ADAPTIVE ESTIMATION vs FIXED ZOOM")
    print("=" * 90)

    target_body_heights = [20.0, 22.0, 24.0]
    best_sw = best_grid_setting[1]  # Use best stroke width from grid

    print(f"{'Policy / Target':<28} | {'Overall CER':<14} | {'Words CER (60)':<16} | {'Lines CER (40)':<16}")
    print("-" * 90)

    # Fixed best zoom comparison
    fixed_z = best_grid_setting[0]
    cand_pairs_fixed = []
    is_word_mask = ["sample_word_" in sid for sid in sample_ids]

    for sid in sample_ids:
        item = strokes_by_id[sid]
        img = render_strokes_custom(item["strokes"], zoom=fixed_z, stroke_width_px=best_sw)
        cand_pairs_fixed.append((img, item["text"]))

    c_cer, _, c_cers, _, _ = evaluate_in_memory_images(cand_pairs_fixed, model, vocab, device, order=order)
    w_cer = np.mean([c for c, m in zip(c_cers, is_word_mask) if m]) * 100.0
    l_cer = np.mean([c for c, m in zip(c_cers, is_word_mask) if not m]) * 100.0
    print(f"Fixed Zoom {fixed_z:.1f}x (Best)       | {c_cer:<14.2f}% | {w_cer:<16.2f}% | {l_cer:<16.2f}%")

    adaptive_results = {}
    for tgt_h in target_body_heights:
        cand_pairs_adapt = []
        for sid in sample_ids:
            item = strokes_by_id[sid]
            base_y, body_h, _ = estimate_body_height(item["strokes"])
            nominal_scaled = body_h * (64.0 / BAND_HEIGHT)
            z_adapt = float(np.clip(tgt_h / max(1.0, nominal_scaled), 1.0, 2.2))
            img = render_strokes_custom(item["strokes"], zoom=z_adapt, stroke_width_px=best_sw, center_y=base_y)
            cand_pairs_adapt.append((img, item["text"]))

        a_cer, _, a_cers, _, _ = evaluate_in_memory_images(cand_pairs_adapt, model, vocab, device, order=order)
        aw_cer = np.mean([c for c, m in zip(a_cers, is_word_mask) if m]) * 100.0
        al_cer = np.mean([c for c, m in zip(a_cers, is_word_mask) if not m]) * 100.0
        adaptive_results[tgt_h] = (a_cer, aw_cer, al_cer)
        print(f"Adaptive Body {tgt_h:.0f}px (clamp 1-2.2)| {a_cer:<14.2f}% | {aw_cer:<16.2f}% | {al_cer:<16.2f}%")

    # ---------------------------------------------------------------------
    # ITEM 3: CLOSED-SET VERIFICATION (Words Ranking & Lines Verification)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ITEM 3: CLOSED-SET VERIFICATION ON CAPTURE SET")
    print("=" * 90)

    # Best preprocessing policy:
    best_prep_z = best_grid_setting[0]
    best_prep_sw = best_grid_setting[1]
    print(f"Using Best Preprocessing: Zoom={best_prep_z:.1f}x, Width={best_prep_sw:.1f}px")

    best_prep_images = {}
    for sid in sample_ids:
        item = strokes_by_id[sid]
        best_prep_images[sid] = render_strokes_custom(item["strokes"], zoom=best_prep_z, stroke_width_px=best_prep_sw)

    # (a) 60 WORDS RANKING AMONG ALL UNIQUE QURAN WORDS
    print("\n--- (a) 60 Words CTC Log-Likelihood Ranking ---")
    tanzil_clean = ROOT / "data" / "tanzil" / "quran-simple-clean.txt"
    all_quran_words = set()
    with open(tanzil_clean, encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                line = line.split("|", 2)[-1]
            for w in line.strip().split():
                if all(c in vocab.char2idx for c in w):
                    all_quran_words.add(w)

    dict_words = sorted(list(all_quran_words))
    print(f"Built closed Quran dictionary: {len(dict_words)} unique words.")

    word_ids = [sid for sid in sample_ids if "sample_word_" in sid]
    word_ranks = []
    top1, top5, top10 = 0, 0, 0

    ctc_loss_fn = nn.CTCLoss(reduction="none", zero_infinity=True)

    # Pre-encode all candidate words
    cand_encs = [vocab.encode(w[::-1] if order == "visual" else w) for w in dict_words]
    cand_lens = [len(e) for e in cand_encs]
    word_to_idx = {w: i for i, w in enumerate(dict_words)}

    for wid in word_ids:
        item = strokes_by_id[wid]
        true_w = item["text"]
        img = best_prep_images[wid]

        # Convert to tensor
        tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        pad_w = (-img.width) % 32
        if pad_w:
            tensor = nn.functional.pad(tensor, (0, pad_w), value=1.0)
        img_tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            log_probs = model(img_tensor)  # (T, 1, V)
            T_len = log_probs.shape[0]

            # Batch evaluate CTC loss for all dictionary words against this image
            # Split dictionary into batches of 1024
            scores = []
            chunk_size = 1024
            for b_start in range(0, len(dict_words), chunk_size):
                b_words = dict_words[b_start:b_start + chunk_size]
                b_encs = cand_encs[b_start:b_start + chunk_size]
                b_lens = torch.tensor(cand_lens[b_start:b_start + chunk_size], dtype=torch.long)
                flat_targets = torch.tensor([idx for enc in b_encs for idx in enc], dtype=torch.long)

                expanded_log_probs = log_probs.expand(T_len, len(b_words), -1)
                input_lengths = torch.full((len(b_words),), T_len, dtype=torch.long)

                # CTCLoss returns -log P(target | X)
                loss = ctc_loss_fn(expanded_log_probs, flat_targets, input_lengths, b_lens)
                scores.extend(loss.cpu().tolist())

        scores = np.array(scores)
        # Sort candidates by ascending loss (highest likelihood)
        sorted_indices = np.argsort(scores)
        true_dict_idx = word_to_idx.get(true_w, -1)

        if true_dict_idx != -1:
            rank = int(np.where(sorted_indices == true_dict_idx)[0][0]) + 1
        else:
            rank = len(dict_words)

        word_ranks.append(rank)
        if rank == 1:
            top1 += 1
        if rank <= 5:
            top5 += 1
        if rank <= 10:
            top10 += 1

    n_words = len(word_ids)
    med_rank = float(np.median(word_ranks))
    print(f"Top-1 Accuracy : {top1}/{n_words} ({top1/n_words*100:.1f}%)")
    print(f"Top-5 Accuracy : {top5}/{n_words} ({top5/n_words*100:.1f}%)")
    print(f"Top-10 Accuracy: {top10}/{n_words} ({top10/n_words*100:.1f}%)")
    print(f"Median True Word Rank: {med_rank:.1f} (out of {len(dict_words)})")

    # (b) 40 LINES CORRUPTED VARIANT VERIFICATION
    print("\n--- (b) 40 Lines Verification (True vs Corrupted) ---")
    line_ids = [sid for sid in sample_ids if "sample_line_" in sid]

    # Create 1 realistic corrupted variant per line
    # Dot confusions for corruption
    dot_pairs = [("ب", "ت"), ("ت", "ث"), ("ن", "ي"), ("ج", "ح"), ("د", "ذ"), ("ر", "ز"), ("س", "ش"), ("ص", "ض")]

    line_corruptions = {}
    random.seed(42)

    for lid in line_ids:
        raw_text = strokes_by_id[lid]["text"]
        words = raw_text.split()
        if not words:
            continue

        corrupted_words = list(words)
        target_w_idx = random.randint(0, len(words) - 1)
        orig_w = words[target_w_idx]

        # Try single letter homoglyph mutation first
        mutated = False
        chars = list(orig_w)
        for i_c, c in enumerate(chars):
            for a, b in dot_pairs:
                if c == a:
                    chars[i_c] = b
                    mutated = True
                    break
                elif c == b:
                    chars[i_c] = a
                    mutated = True
                    break
            if mutated:
                break

        if mutated:
            corrupted_words[target_w_idx] = "".join(chars)
        else:
            # Swap with random dictionary word of similar length
            candidates = [w for w in dict_words if abs(len(w) - len(orig_w)) <= 1 and w != orig_w]
            corrupted_words[target_w_idx] = random.choice(candidates) if candidates else "الله"

        line_corruptions[lid] = " ".join(corrupted_words)

    # Compute CTC loss of true vs corrupt
    correct_verifications = 0
    true_word_norm_losses = []
    corr_word_norm_losses = []

    for lid in line_ids:
        true_t = strokes_by_id[lid]["text"]
        corr_t = line_corruptions[lid]
        img = best_prep_images[lid]

        tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        pad_w = (-img.width) % 32
        if pad_w:
            tensor = nn.functional.pad(tensor, (0, pad_w), value=1.0)
        img_tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            log_probs = model(img_tensor)  # (T, 1, V)
            T_len = log_probs.shape[0]

            t_enc = vocab.encode(true_t[::-1] if order == "visual" else true_t)
            c_enc = vocab.encode(corr_t[::-1] if order == "visual" else corr_t)

            targets = torch.tensor(t_enc + c_enc, dtype=torch.long)
            target_lens = torch.tensor([len(t_enc), len(c_enc)], dtype=torch.long)
            in_lens = torch.full((2,), T_len, dtype=torch.long)
            expanded_log_probs = log_probs.expand(T_len, 2, -1)

            losses = ctc_loss_fn(expanded_log_probs, targets, in_lens, target_lens).cpu().numpy()
            true_loss = losses[0]
            corr_loss = losses[1]

            # True > Corrupt in probability means true_loss < corr_loss
            if true_loss < corr_loss:
                correct_verifications += 1

            # Per-word normalized log-likelihood: -loss / num_words
            n_w = len(true_t.split())
            true_word_norm_losses.append(-true_loss / n_w)
            corr_word_norm_losses.append(-corr_loss / n_w)

    line_verif_acc = (correct_verifications / len(line_ids)) * 100.0
    print(f"Line Verification Accuracy (True > Corrupted): {correct_verifications}/{len(line_ids)} ({line_verif_acc:.1f}%)")

    # False Accept Rate (FAR) and False Reject Rate (FRR) at 3 thresholds
    print("\nFAR & FRR across 3 Per-Word Log-Likelihood Thresholds (τ):")
    # True scores vs Corrupt scores
    true_scores = np.array(true_word_norm_losses)
    corr_scores = np.array(corr_word_norm_losses)

    thresholds = [-1.5, -2.0, -2.5]
    print(f"{'Threshold (τ)':<16} | {'False Reject Rate (FRR)':<25} | {'False Accept Rate (FAR)':<25}")
    print("-" * 75)
    for tau in thresholds:
        # FRR: True score < tau
        frr = np.mean(true_scores < tau) * 100.0
        # FAR: Corrupted score >= tau
        far = np.mean(corr_scores >= tau) * 100.0
        print(f"{tau:<16.1f} | {frr:<25.1f}% | {far:<25.1f}%")

    print("\n" + "=" * 90)
    print("TASK 2.15 SUITE COMPLETED SUCCESSFULLY")
    print("=" * 90)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/content/drive/MyDrive/quran_htr_writer_checkpoints/step2b_v3/best.pt")
    args = p.parse_args()

    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.exists():
        for alt in [
            Path("checkpoints/step2b_v3/best.pt"),
            Path("checkpoints/best.pt"),
        ]:
            if alt.exists():
                ckpt_p = alt
                break

    run_task_2_15(ckpt_p)
