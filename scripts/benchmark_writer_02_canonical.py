"""
scripts/benchmark_writer_02_canonical.py
Task 2.21 Item 2:
1. Writer_02 baseline at canonical scale (zoom = 1.0, width = 2.2px):
   - CER (words / lines / overall)
   - Closed-set ranking (full 14,872 dictionary + 200-distractor protocol, labeled)
   - 3-state outcomes for true and corrupted claims (thresholds frozen from writer_01)
2. Zoom grid diagnostics: {0.7, 0.85, 1.0, 1.15, 1.3}
3. Per-writer geometry table (ink-extent ratio, body-height, stroke width, sampling rate, device, stylus)
4. Montage of 20 worst and 10 best writer_02 samples saved to reports/montage_writer_02_worst_best.png
"""

from __future__ import annotations
import csv
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from model import CRNN
from decode import greedy_decode
from metrics import cer, corpus_cer_wer
from render_strokes import render_strokes
from verify_ref import verify_reference
from task_2_17_suite import apply_corruption, extract_mutashabih_pairs


def get_sample_tensor(s: dict, zoom: float = 1.0, width: float = 2.2) -> tuple[torch.Tensor, Image.Image]:
    geom = s.get("geometry") or {}
    bt = float(geom.get("band_top_css", 0.0))
    bb = float(geom.get("band_base_css", 147.0))
    bot = float(geom.get("band_bot_css", 180.0))
    bh = float(geom.get("band_height_css", bot - bt))
    img = render_strokes(
        s["strokes"],
        zoom=zoom,
        width_after_zoom=width,
        center_y=(bb - bt),
        band_top_y=0.0,
        band_height=bh,
        pad_x=16.0,
    )
    arr = (np.array(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    pad_w = (-arr.shape[1]) % 32
    if pad_w > 0:
        arr = np.pad(arr, ((0, 0), (0, pad_w)), mode="constant", constant_values=1.0)
    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    return tensor, img


def main():
    device = torch.device("cpu")
    vocab = Vocabulary.load(ROOT / "vocab.json")
    ckpt = torch.load(ROOT / "checkpoints" / "best.pt", map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    # Load 14,872 encodable Quran words
    with open(ROOT / "data" / "tanzil" / "quran-simple-clean.txt", encoding="utf-8") as f:
        all_raw = set(f.read().split())
    encodable_dict = sorted([w for w in all_raw if all(c in vocab.char2idx for c in w)])

    # Load writer_02 samples
    with open(ROOT / "data" / "capture_v2" / "writer_02" / "strokes.json", encoding="utf-8") as f:
        raw_strokes = json.load(f)
    strokes_by_id = {item["id"]: item for item in raw_strokes if isinstance(item, dict) and "id" in item}
    samples = list(strokes_by_id.values())

    line_samples = [s for s in samples if s.get("mode") == "line" or "line" in s.get("id", "")]
    word_samples = [s for s in samples if s.get("mode") == "word" or "word" in s.get("id", "")]

    print(f"Loaded {len(samples)} writer_02 samples ({len(line_samples)} lines, {len(word_samples)} words).")

    # 1. Zoom Grid Diagnostics {0.7, 0.85, 1.0, 1.15, 1.3}
    print("\n--- Zoom Grid Diagnostics (writer_02) ---")
    zoom_grid = [0.7, 0.85, 1.0, 1.15, 1.3]
    grid_results = {}
    for z in zoom_grid:
        preds_all, targets_all = [], []
        preds_l, targets_l = [], []
        preds_w, targets_w = [], []
        for s in samples:
            tensor, _ = get_sample_tensor(s, zoom=z, width=2.2)
            with torch.no_grad():
                lp = model(tensor)
            pred = greedy_decode(lp, vocab, label_order=order)[0]
            gt = s["text"]
            preds_all.append(pred)
            targets_all.append(gt)
            if s.get("mode") == "line" or "line" in s.get("id", ""):
                preds_l.append(pred)
                targets_l.append(gt)
            else:
                preds_w.append(pred)
                targets_w.append(gt)
        c_all, w_all = corpus_cer_wer(preds_all, targets_all)
        c_l, _ = corpus_cer_wer(preds_l, targets_l)
        c_w, _ = corpus_cer_wer(preds_w, targets_w)
        grid_results[z] = {"cer_all": c_all * 100, "cer_line": c_l * 100, "cer_word": c_w * 100, "wer_all": w_all * 100}
        print(f"Zoom {z:4.2f}x | Overall CER: {c_all*100:5.2f}% | Lines: {c_l*100:5.2f}% | Words: {c_w*100:5.2f}% | WER: {w_all*100:5.2f}%")

    # 2. Canonical Scale (1.0x, 2.2px) Detailed Benchmark
    print("\n--- Canonical Scale (1.0x, 2.2px) Evaluation ---")
    sample_records = []
    for s in samples:
        tensor, img = get_sample_tensor(s, zoom=1.0, width=2.2)
        with torch.no_grad():
            lp = model(tensor)
        pred = greedy_decode(lp, vocab, label_order=order)[0]
        gt = s["text"]
        scer = cer(pred, gt)
        sample_records.append({
            "sample": s,
            "pred": pred,
            "gt": gt,
            "cer": scer,
            "lp": lp,
            "img": img,
            "is_line": (s.get("mode") == "line" or "line" in s.get("id", "")),
        })

    # Closed-set ranking on word crops
    dict_encoded = [(w, [vocab.char2idx[c] for c in (w[::-1] if order == "visual" else w) if c in vocab.char2idx]) for w in encodable_dict]
    cand_encs = [torch.tensor(e, dtype=torch.long) for _, e in dict_encoded]
    cand_lens = torch.tensor([len(e) for e in cand_encs], dtype=torch.long)
    word_to_idx = {w: i for i, (w, _) in enumerate(dict_encoded)}

    chunk_size = 1024
    chunks = []
    for b_start in range(0, len(dict_encoded), chunk_size):
        b_end = min(b_start + chunk_size, len(dict_encoded))
        b_lens = cand_lens[b_start:b_end]
        b_flat = torch.cat([cand_encs[i] for i in range(b_start, b_end)])
        chunks.append((b_start, b_end, b_lens, b_flat))

    ctc_loss_fn = nn.CTCLoss(reduction="none", zero_infinity=True)
    ranks_full = []
    ranks_200 = []
    rng = random.Random(42)

    word_records = [r for r in sample_records if not r["is_line"]]
    for r in word_records:
        lp = r["lp"]
        gt = r["gt"]
        T_len = lp.shape[0]
        scores = []
        for b_start, b_end, b_lens, b_flat in chunks:
            n_cand = b_end - b_start
            exp_lp = lp.expand(T_len, n_cand, -1)
            in_lens = torch.full((n_cand,), T_len, dtype=torch.long)
            loss = ctc_loss_fn(exp_lp, b_flat, in_lens, b_lens)
            scores.extend(loss.cpu().tolist())
        scores = np.array(scores)
        true_idx = word_to_idx.get(gt, -1)
        if true_idx != -1:
            true_score = scores[true_idx]
            rank_f = int(np.sum(scores < true_score)) + 1
        else:
            rank_f = len(dict_encoded)
        ranks_full.append(rank_f)

        pool_idx = rng.sample(range(len(dict_encoded)), 200)
        sub_scores = scores[pool_idx]
        if true_idx != -1:
            dist_higher = int(np.sum(sub_scores < true_score))
            scaled_r = 1 + int(round(dist_higher * (len(dict_encoded) / 200.0)))
        else:
            scaled_r = len(dict_encoded)
        ranks_200.append(scaled_r)

    # 3-State Verification (Frozen from writer_01: t_accept = -0.60, t_wrong = -0.95)
    T_ACCEPT = -0.60
    T_WRONG = -0.95
    true_states = {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0}
    corr_states = defaultdict(lambda: {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0})

    mut_pairs = extract_mutashabih_pairs()
    mut_dict = defaultdict(list)
    for p in mut_pairs:
        mut_dict[p["prefix"]].append(p["word_b"])
    corruption_types = ["dot_swap", "del_ins", "word_swap", "adj_swap", "skip_word", "mutashabih"]

    for r in sample_records:
        lp_np = r["lp"].squeeze(1).numpy()
        text = r["gt"]
        v_true = verify_reference(lp_np, text, vocab.char2idx, vocab.idx2char, t_accept=T_ACCEPT, t_wrong=T_WRONG, order=order, norm="char")
        for res in v_true:
            true_states[res["state"]] += 1

        for ctype in corruption_types:
            corr = apply_corruption(text, ctype, encodable_dict, mut_dict, rng)
            if corr is None:
                continue
            corr_text, c_idx = corr
            v_corr = verify_reference(lp_np, corr_text, vocab.char2idx, vocab.idx2char, t_accept=T_ACCEPT, t_wrong=T_WRONG, order=order, norm="char")
            if v_corr:
                target_r = v_corr[c_idx] if c_idx < len(v_corr) else v_corr[-1]
                corr_states[ctype][target_r["state"]] += 1

    # Montage Generation: 20 Worst and 10 Best Samples
    sorted_records = sorted(sample_records, key=lambda x: x["cer"])
    best_10 = sorted_records[:10]
    worst_20 = sorted_records[-20:][::-1]

    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    montage_path = reports_dir / "montage_writer_02_worst_best.png"

    # Create composite image (30 items total: 10 best, then 20 worst)
    row_h = 75
    img_w = 1000
    total_h = row_h * (10 + 20 + 2) + 60
    montage = Image.new("RGB", (img_w, total_h), color=(255, 255, 255))
    draw = ImageDraw.Draw(montage)

    y_off = 10
    draw.text((20, y_off), "WRITER_02: 10 BEST SAMPLES (LOWEST CER)", fill=(0, 100, 0))
    y_off += 30

    def draw_row(item, y_pos, label_color):
        s_id = item["sample"]["id"]
        img = item["img"]
        # Thumbnail max 280w x 55h
        thumb = img.copy()
        thumb.thumbnail((280, 55))
        montage.paste(thumb, (20, y_pos))
        txt_meta = f"ID: {s_id} | CER: {item['cer']*100:.1f}%"
        txt_gt = f"GT:   {item['gt']}"
        txt_pred = f"Pred: {item['pred']}"
        draw.text((320, y_pos), txt_meta, fill=label_color)
        draw.text((320, y_pos + 18), txt_gt, fill=(30, 30, 30))
        draw.text((320, y_pos + 36), txt_pred, fill=(80, 80, 80))
        draw.line([(10, y_pos + 65), (img_w - 10, y_pos + 65)], fill=(220, 220, 220), width=1)

    for b in best_10:
        draw_row(b, y_off, (0, 120, 0))
        y_off += row_h

    y_off += 20
    draw.text((20, y_off), "WRITER_02: 20 WORST SAMPLES (HIGHEST CER)", fill=(180, 0, 0))
    y_off += 30

    for w in worst_20:
        draw_row(w, y_off, (180, 0, 0))
        y_off += row_h

    montage.save(montage_path)
    print(f"\nSaved montage to {montage_path}")

    # Output Full Report Summary
    print("\n" + "=" * 90)
    print("TASK 2.21 ITEM 2: BASELINE AT CANONICAL SCALE (1.0x, 2.2px)")
    print("=" * 90)
    c_1_all = grid_results[1.0]["cer_all"]
    c_1_l = grid_results[1.0]["cer_line"]
    c_1_w = grid_results[1.0]["cer_word"]
    w_1_all = grid_results[1.0]["wer_all"]
    print(f"1. Accuracy Metrics (Canonical 1.0x):")
    print(f"   * Lines CER: {c_1_l:5.2f}% | Words CER: {c_1_w:5.2f}% | Overall CER: {c_1_all:5.2f}% | WER: {w_1_all:5.2f}%")

    top1_f = np.mean([1 if r == 1 else 0 for r in ranks_full]) * 100.0
    top5_f = np.mean([1 if r <= 5 else 0 for r in ranks_full]) * 100.0
    med_f = float(np.median(ranks_full))
    top1_200 = np.mean([1 if r == 1 else 0 for r in ranks_200]) * 100.0
    med_200 = float(np.median(ranks_200))

    print(f"\n2. Closed-Set Word Rank (125 words):")
    print(f"   * Full Dict (14,872 words): Top-1 {top1_f:.2f}%, Top-5 {top5_f:.2f}%, Median Rank {med_f:.1f}")
    print(f"   * 200-Distractor Protocol:  Top-1 {top1_200:.2f}%, Median Rank {med_200:.1f}")

    tot_true = sum(true_states.values())
    print(f"\n3. 3-State Verification (Frozen writer_01 thresholds: t_acc={T_ACCEPT}, t_wrg={T_WRONG}):")
    print(f"   * TRUE Claims ({tot_true} words): ACCEPT {true_states['ACCEPT']/tot_true*100:.1f}% | UNCERTAIN {true_states['UNCERTAIN']/tot_true*100:.1f}% | WRONG {true_states['WRONG']/tot_true*100:.1f}%")
    print("   * CORRUPTED Claims by Type:")
    for ct in corruption_types:
        st = corr_states[ct]
        tot_c = max(1, sum(st.values()))
        print(f"     - {ct:<12}: ACCEPT {st['ACCEPT']/tot_c*100:4.1f}% | UNCERTAIN {st['UNCERTAIN']/tot_c*100:4.1f}% | WRONG {st['WRONG']/tot_c*100:4.1f}%")


if __name__ == "__main__":
    main()
