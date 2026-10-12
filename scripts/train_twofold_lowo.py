"""
scripts/train_twofold_lowo.py
Task 2.20 Item 2:
Two-fold cross-writer fine-tune pipeline:
- Fold A: train writer_01 (80% train, 20% calib) -> test writer_02 (145 samples)
- Fold B: train writer_02 (80% train, 20% calib) -> test writer_01 (100 samples)
- Mix: 0.5 strokes / 0.3 synth v3 / 0.2 KHATT
- Schedule: 6 epochs x 6k samples, cosine lr 5e-5 to 5e-6, fixed schedule
- Recalibrate thresholds on 20% slice of training writer
- Evaluate baseline vs fine-tuned metrics on held-out test writer
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from model import CRNN, compute_output_seq_len
from dataset import StrokeDataset, collate_fn
from shards import ShardDataset
from decode import greedy_decode
from metrics import cer, corpus_cer_wer
from render_strokes import render_strokes
from verify_ref import ctc_viterbi_alignment, verify_reference
from task_2_17_suite import apply_corruption, extract_mutashabih_pairs


def run_recalibration(model, calib_samples, vocab, order="visual", device="cpu"):
    """Recalibrate t_accept (95% true acceptance) and t_wrong (90% corrupted reject) on 20% slice."""
    model.eval()
    true_llrs = []
    corr_llrs = []
    mut_pairs = extract_mutashabih_pairs()
    mut_dict = defaultdict(list)
    for p in mut_pairs:
        mut_dict[p["prefix"]].append(p["word_b"])

    with open(ROOT / "data" / "tanzil" / "quran-simple-clean.txt", encoding="utf-8") as f:
        words_pool = sorted([w for w in set(f.read().split()) if all(c in vocab.char2idx for c in w)])

    rng = random.Random(42)
    corruption_types = ["dot_swap", "del_ins", "word_swap", "adj_swap", "skip_word", "mutashabih"]

    for s in calib_samples:
        geom = s.get("geometry") or {}
        bt = float(geom.get("band_top_css", 0.0))
        bb = float(geom.get("band_base_css", 147.0))
        bot = float(geom.get("band_bot_css", 180.0))
        bh = float(geom.get("band_height_css", bot - bt))
        img = render_strokes(s["strokes"], zoom=1.4, width_after_zoom=1.8, center_y=(bb - bt), band_top_y=0.0, band_height=bh, pad_x=16.0)
        arr = (np.array(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        pad_w = (-arr.shape[1]) % 32
        if pad_w > 0:
            arr = np.pad(arr, ((0, 0), (0, pad_w)), mode="constant", constant_values=1.0)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            lp = model(tensor).squeeze(1).cpu().numpy()
        gt = s["text"]

        # True claims
        v_true = verify_reference(lp, gt, vocab.char2idx, vocab.idx2char, t_accept=-0.6, t_wrong=-0.95, order=order, norm="char")
        for r in v_true:
            true_llrs.append(r["llr_char"])

        # Corrupted claims
        for ctype in corruption_types:
            corr = apply_corruption(gt, ctype, words_pool, mut_dict, rng)
            if corr is None:
                continue
            corr_text, c_idx = corr
            v_corr = verify_reference(lp, corr_text, vocab.char2idx, vocab.idx2char, t_accept=-0.6, t_wrong=-0.95, order=order, norm="char")
            if v_corr:
                target_r = v_corr[c_idx] if c_idx < len(v_corr) else v_corr[-1]
                corr_llrs.append(target_r["llr_char"])

    # 95% true-accept -> 5th percentile of true LLR
    t_accept = float(np.percentile(true_llrs, 5.0)) if true_llrs else -0.60
    # 90% corrupted wrong -> 90th percentile of corrupted LLR
    t_wrong = float(np.percentile(corr_llrs, 90.0)) if corr_llrs else -0.95
    return t_accept, t_wrong


def evaluate_test_writer(model, test_samples, vocab, t_accept, t_wrong, order="visual", device="cpu"):
    """Evaluates test writer on CER (lines/words), closed-set rank, and 3-state outcomes."""
    model.eval()
    preds_line, targets_line = [], []
    preds_word, targets_word = [], []
    all_preds, all_targets = [], []
    word_samples = [s for s in test_samples if s.get("mode") == "word" or "word" in s.get("id", "")]

    with open(ROOT / "data" / "tanzil" / "quran-simple-clean.txt", encoding="utf-8") as f:
        encodable_dict = sorted([w for w in set(f.read().split()) if all(c in vocab.char2idx for c in w)])

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

    for s in test_samples:
        geom = s.get("geometry") or {}
        bt = float(geom.get("band_top_css", 0.0))
        bb = float(geom.get("band_base_css", 147.0))
        bot = float(geom.get("band_bot_css", 180.0))
        bh = float(geom.get("band_height_css", bot - bt))
        img = render_strokes(s["strokes"], zoom=1.4, width_after_zoom=1.8, center_y=(bb - bt), band_top_y=0.0, band_height=bh, pad_x=16.0)
        arr = (np.array(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        pad_w = (-arr.shape[1]) % 32
        if pad_w > 0:
            arr = np.pad(arr, ((0, 0), (0, pad_w)), mode="constant", constant_values=1.0)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            lp = model(tensor)
        pred = greedy_decode(lp, vocab, label_order=order)[0]
        gt = s["text"]
        all_preds.append(pred)
        all_targets.append(gt)
        if s.get("mode") == "line" or "line" in s.get("id", ""):
            preds_line.append(pred)
            targets_line.append(gt)
        else:
            preds_word.append(pred)
            targets_word.append(gt)
            # Ranking
            T_len = lp.shape[0]
            scores = []
            for b_start, b_end, b_lens, b_flat in chunks:
                n_cand = b_end - b_start
                exp_lp = lp.expand(T_len, n_cand, -1)
                in_lens = torch.full((n_cand,), T_len, dtype=torch.long, device=device)
                loss = ctc_loss_fn(exp_lp, b_flat.to(device), in_lens, b_lens.to(device))
                scores.extend(loss.cpu().tolist())
            scores = np.array(scores)
            true_idx = word_to_idx.get(gt, -1)
            if true_idx != -1:
                rank_full = int(np.sum(scores < scores[true_idx])) + 1
            else:
                rank_full = len(dict_encoded)
            ranks_full.append(rank_full)

            pool_idx = rng.sample(range(len(dict_encoded)), 200)
            if true_idx != -1:
                dist_higher = int(np.sum(scores[pool_idx] < scores[true_idx]))
                scaled_r = 1 + int(round(dist_higher * (len(dict_encoded) / 200.0)))
            else:
                scaled_r = len(dict_encoded)
            ranks_200.append(scaled_r)

    c_all, w_all = corpus_cer_wer(all_preds, all_targets)
    c_line, _ = corpus_cer_wer(preds_line, targets_line) if preds_line else (0.0, 0.0)
    c_word, _ = corpus_cer_wer(preds_word, targets_word) if preds_word else (0.0, 0.0)

    # 3-State verification
    mut_pairs = extract_mutashabih_pairs()
    mut_dict = defaultdict(list)
    for p in mut_pairs:
        mut_dict[p["prefix"]].append(p["word_b"])

    true_states = {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0}
    corr_states = defaultdict(lambda: {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0})
    corruption_types = ["dot_swap", "del_ins", "word_swap", "adj_swap", "skip_word", "mutashabih"]

    for s in test_samples:
        geom = s.get("geometry") or {}
        bt = float(geom.get("band_top_css", 0.0))
        bb = float(geom.get("band_base_css", 147.0))
        bot = float(geom.get("band_bot_css", 180.0))
        bh = float(geom.get("band_height_css", bot - bt))
        img = render_strokes(s["strokes"], zoom=1.4, width_after_zoom=1.8, center_y=(bb - bt), band_top_y=0.0, band_height=bh, pad_x=16.0)
        arr = (np.array(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        pad_w = (-arr.shape[1]) % 32
        if pad_w > 0:
            arr = np.pad(arr, ((0, 0), (0, pad_w)), mode="constant", constant_values=1.0)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            lp = model(tensor).squeeze(1).cpu().numpy()
        text = s["text"]

        v_true = verify_reference(lp, text, vocab.char2idx, vocab.idx2char, t_accept=t_accept, t_wrong=t_wrong, order=order, norm="char")
        for r in v_true:
            true_states[r["state"]] += 1

        for ctype in corruption_types:
            corr = apply_corruption(text, ctype, encodable_dict, mut_dict, rng)
            if corr is None:
                continue
            corr_text, c_idx = corr
            v_corr = verify_reference(lp, corr_text, vocab.char2idx, vocab.idx2char, t_accept=t_accept, t_wrong=t_wrong, order=order, norm="char")
            if v_corr:
                target_r = v_corr[c_idx] if c_idx < len(v_corr) else v_corr[-1]
                corr_states[ctype][target_r["state"]] += 1

    return {
        "cer_all": c_all * 100.0,
        "cer_line": c_line * 100.0,
        "cer_word": c_word * 100.0,
        "wer_all": w_all * 100.0,
        "top1_full": float(np.mean([1 if r == 1 else 0 for r in ranks_full]) * 100.0),
        "median_rank_full": float(np.median(ranks_full)),
        "top1_200": float(np.mean([1 if r == 1 else 0 for r in ranks_200]) * 100.0),
        "true_accept_pct": true_states["ACCEPT"] / max(sum(true_states.values()), 1) * 100.0,
        "corr_wrong_pct": sum(corr_states[c]["WRONG"] for c in corruption_types) / max(sum(sum(corr_states[c].values()) for c in corruption_types), 1) * 100.0,
        "corr_states": {c: dict(corr_states[c]) for c in corruption_types},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", choices=["A", "B", "both"], default="both")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--samples_per_epoch", type=int, default=6000)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--min_lr", type=float, default=5e-6)
    args = p.parse_args()

    vocab = Vocabulary.load(ROOT / "vocab.json")
    print(f"[Two-Fold LOWO] Starting evaluation/fine-tuning pipeline on {args.device}...")


if __name__ == "__main__":
    main()
