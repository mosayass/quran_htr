"""
scripts/benchmark_writer_02_baseline.py
Task 2.20 Item 1:
Comprehensive Baseline Benchmark on writer_02 (held-out, no training):
- Thresholds frozen from writer_01 only: t_accept = -0.60, t_wrong = -0.95 (per-char norm).
- Compares Preprocessing 1.0x (production export) vs 1.4x / 1.8px (supersampled).
- Evaluates:
  1. CER / WER for lines (20) and words (125) separately, with paired bootstrap 95% CIs.
  2. Closed-set ranking on word crops:
     (a) Full encodable dictionary (14,872 unique words)
     (b) 200-distractor protocol (labeled)
  3. 3-state outcomes (ACCEPT / UNCERTAIN / WRONG) for TRUE and CORRUPTED claims across 6 types under per-char norm.
  4. Comparative Table: writer_01 vs writer_02.
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
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from model import CRNN
from decode import greedy_decode
from metrics import cer, corpus_cer_wer
from render_strokes import render_strokes
from verify_ref import ctc_viterbi_alignment, verify_reference
from task_2_17_suite import apply_corruption, extract_mutashabih_pairs


def paired_bootstrap_ci(baseline_losses: list[float], candidate_losses: list[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 42) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    b_arr = np.array(baseline_losses)
    c_arr = np.array(candidate_losses)
    diff = c_arr - b_arr
    n = len(diff)
    boot_means = [np.mean(diff[rng.integers(0, n, n)]) * 100.0 for _ in range(n_boot)]
    boot_means = np.sort(boot_means)
    lo = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return float(np.mean(diff) * 100.0), lo, hi


def run_benchmark():
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
    print(f"Loaded dictionary: {len(all_raw)} raw unique words -> {len(encodable_dict)} encodable words (72 classes).")

    # Load writer_02 samples
    strokes_p = ROOT / "data" / "capture_v2" / "writer_02" / "strokes.json"
    manifest_p = ROOT / "data" / "capture_v2" / "writer_02" / "manifest.csv"
    with open(strokes_p, encoding="utf-8") as f:
        raw_strokes = json.load(f)

    # Deduplicate: latest save wins
    strokes_by_id = {}
    for item in raw_strokes:
        if isinstance(item, dict) and "id" in item:
            strokes_by_id[item["id"]] = item

    samples = list(strokes_by_id.values())
    line_samples = [s for s in samples if s.get("mode") == "line" or "line" in s.get("id", "")]
    word_samples = [s for s in samples if s.get("mode") == "word" or "word" in s.get("id", "")]
    print(f"Writer 02 Samples: {len(samples)} total ({len(line_samples)} lines, {len(word_samples)} words).")

    # Preprocessing evaluation: Policy 1.0x (nominal) vs Policy 1.4x (1.8px)
    def preprocess_sample(s: dict, zoom: float, stroke_w: float) -> torch.Tensor:
        geom = s.get("geometry") or {}
        band_top = float(geom.get("band_top_css", 0.0))
        band_base = float(geom.get("band_base_css", 147.0))
        band_bot = float(geom.get("band_bot_css", 180.0))
        band_h = float(geom.get("band_height_css", band_bot - band_top))
        img = render_strokes(
            s["strokes"],
            zoom=zoom,
            width_after_zoom=stroke_w,
            center_y=(band_base - band_top),
            band_top_y=0.0,
            band_height=band_h,
            pad_x=16.0,
        )
        w, _ = img.size
        arr = (np.array(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        pad_w = (-w) % 32
        if pad_w > 0:
            arr = np.pad(arr, ((0, 0), (0, pad_w)), mode="constant", constant_values=1.0)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    # 1. CER / WER for both preprocessing policies
    results_policy = {}
    for pol_name, (zm, sw) in [("1.0x_nominal", (1.0, 2.55)), ("1.4x_optimal", (1.4, 1.8))]:
        preds, targets = [], []
        sample_cers = []
        preds_line, targets_line = [], []
        preds_word, targets_word = [], []
        for s in samples:
            tensor = preprocess_sample(s, zm, sw)
            with torch.no_grad():
                lp = model(tensor)
            pred = greedy_decode(lp, vocab, label_order=order)[0]
            gt = s["text"]
            preds.append(pred)
            targets.append(gt)
            sample_cers.append(cer(pred, gt))
            if s.get("mode") == "line" or "line" in s.get("id", ""):
                preds_line.append(pred)
                targets_line.append(gt)
            else:
                preds_word.append(pred)
                targets_word.append(gt)

        c_all, w_all = corpus_cer_wer(preds, targets)
        c_line, w_line = corpus_cer_wer(preds_line, targets_line)
        c_word, w_word = corpus_cer_wer(preds_word, targets_word)
        results_policy[pol_name] = {
            "cer_all": c_all * 100,
            "wer_all": w_all * 100,
            "cer_line": c_line * 100,
            "wer_line": w_line * 100,
            "cer_word": c_word * 100,
            "wer_word": w_word * 100,
            "sample_cers": sample_cers,
        }

    diff_mean, ci_lo, ci_hi = paired_bootstrap_ci(
        results_policy["1.0x_nominal"]["sample_cers"],
        results_policy["1.4x_optimal"]["sample_cers"],
    )

    # 2. Closed-Set Word Ranking on 125 words under Policy 1.4x
    # Pre-encode all dictionary words once
    dict_encoded = []
    for cand in encodable_dict:
        cand_lbl = cand[::-1] if order == "visual" else cand
        enc = [vocab.char2idx[c] for c in cand_lbl if c in vocab.char2idx]
        dict_encoded.append((cand, enc))

    # Pre-encode all dictionary words once and chunk for PyTorch CTCLoss
    import torch.nn as nn
    ctc_loss_fn = nn.CTCLoss(reduction="none", zero_infinity=True)

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

    ranks_full = []
    ranks_200 = []
    rng = random.Random(42)

    print(f"\nEvaluating closed-set word ranking across {len(word_samples)} word crops...", flush=True)
    for idx, s in enumerate(word_samples):
        gt = s["text"]
        tensor = preprocess_sample(s, 1.4, 1.8)
        with torch.no_grad():
            lp = model(tensor)  # (T, 1, V)
        T_len = lp.shape[0]

        scores = []
        for b_start, b_end, b_lens, b_flat in chunks:
            n_cand = b_end - b_start
            expanded_lp = lp.expand(T_len, n_cand, -1)
            in_lens = torch.full((n_cand,), T_len, dtype=torch.long)
            loss = ctc_loss_fn(expanded_lp, b_flat, in_lens, b_lens)
            scores.extend(loss.cpu().tolist())

        scores = np.array(scores)
        true_idx = word_to_idx.get(gt, -1)
        if true_idx != -1:
            true_score = scores[true_idx]
            rank_full = int(np.sum(scores < true_score)) + 1
        else:
            rank_full = len(dict_encoded)
        ranks_full.append(rank_full)

        # 200-distractor sampled ranking protocol
        pool_indices = rng.sample(range(len(dict_encoded)), 200)
        sub_scores = scores[pool_indices]
        if true_idx != -1:
            dist_higher = int(np.sum(sub_scores < true_score))
            scaled_r = 1 + int(round(dist_higher * (len(dict_encoded) / 200.0)))
        else:
            scaled_r = len(dict_encoded)
        ranks_200.append(scaled_r)

        if (idx + 1) % 25 == 0:
            print(f"  Processed {idx + 1}/{len(word_samples)} words ranking...", flush=True)

    # 3. 3-State Verification Outcomes on TRUE and CORRUPTED claims (per-char norm)
    # Frozen thresholds from writer_01:
    T_ACCEPT = -0.60
    T_WRONG = -0.95

    true_states = {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0}
    corr_states = defaultdict(lambda: {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0})

    mut_pairs = extract_mutashabih_pairs()
    mut_dict = defaultdict(list)
    for p in mut_pairs:
        mut_dict[p["prefix"]].append(p["word_b"])

    corruption_types = ["dot_swap", "del_ins", "word_swap", "adj_swap", "skip_word", "mutashabih"]

    print("\nEvaluating 3-state verification under frozen writer_01 thresholds (per-char norm)...")
    for s in samples:
        tensor = preprocess_sample(s, 1.4, 1.8)
        with torch.no_grad():
            lp = model(tensor).squeeze(1).numpy()
        text = s["text"]

        # True claims
        v_true = verify_reference(
            lp, text, vocab.char2idx, vocab.idx2char,
            t_accept=T_ACCEPT, t_wrong=T_WRONG, order=order, norm="char"
        )
        for r in v_true:
            true_states[r["state"]] += 1

        # Corrupted claims per type
        for ctype in corruption_types:
            corr = apply_corruption(text, ctype, encodable_dict, mut_dict, rng)
            if corr is None:
                continue
            corr_text, c_idx = corr
            v_corr = verify_reference(
                lp, corr_text, vocab.char2idx, vocab.idx2char,
                t_accept=T_ACCEPT, t_wrong=T_WRONG, order=order, norm="char"
            )
            if v_corr:
                target_r = v_corr[c_idx] if c_idx < len(v_corr) else v_corr[-1]
                corr_states[ctype][target_r["state"]] += 1

    # Print Report Output
    print("\n" + "=" * 95)
    print("TASK 2.20 ITEM 1: BASELINE BENCHMARK ON WRITER_02 (HELD-OUT, FROZEN THRESHOLDS)")
    print("=" * 95)
    print("Thresholds (Frozen from writer_01 only):")
    print(f"  * t_accept = {T_ACCEPT:.2f}  [Set at 95% True-Accept rate on writer_01 DEV true claims]")
    print(f"  * t_wrong  = {T_WRONG:.2f}  [Set at >=90% WRONG rate on writer_01 DEV corrupted claims]")
    print(f"  * Normalization: Per-character (LLR = Delta / num_chars)")

    print("\n1. Preprocessing Policy Comparison (writer_02, 145 samples):")
    print(f"{'Preprocessing Policy':<24} | {'Overall CER':<12} | {'Lines CER (20)':<15} | {'Words CER (125)':<16} | {'Overall WER':<12}")
    print("-" * 88)
    p1 = results_policy["1.0x_nominal"]
    p2 = results_policy["1.4x_optimal"]
    print(f"{'1.0x / Production':<24} | {p1['cer_all']:>10.2f}% | {p1['cer_line']:>13.2f}% | {p1['cer_word']:>14.2f}% | {p1['wer_all']:>10.2f}%")
    print(f"{'1.4x / 1.8px (Optimal)':<24} | {p2['cer_all']:>10.2f}% | {p2['cer_line']:>13.2f}% | {p2['cer_word']:>14.2f}% | {p2['wer_all']:>10.2f}%")
    print(f"ΔCER (1.4x vs 1.0x): {diff_mean:+.2f}% | 95% CI: [{ci_lo:+.2f}%, {ci_hi:+.2f}%]")

    print("\n2. Closed-Set Word Ranking Protocol (writer_02, 125 word crops):")
    print(f"{'Protocol / Dictionary':<32} | {'Top-1 (%)':<12} | {'Top-5 (%)':<12} | {'Top-10 (%)':<12} | {'Median Rank':<12}")
    print("-" * 88)
    top1_f = np.mean([r <= 1 for r in ranks_full]) * 100
    top5_f = np.mean([r <= 5 for r in ranks_full]) * 100
    top10_f = np.mean([r <= 10 for r in ranks_full]) * 100
    med_f = float(np.median(ranks_full))
    print(f"{'Full Dictionary (14,872 words)':<32} | {top1_f:>10.2f}% | {top5_f:>10.2f}% | {top10_f:>10.2f}% | {med_f:>10.1f}")

    top1_s = np.mean([r <= 1 for r in ranks_200]) * 100
    top5_s = np.mean([r <= 5 for r in ranks_200]) * 100
    top10_s = np.mean([r <= 10 for r in ranks_200]) * 100
    med_s = float(np.median(ranks_200))
    print(f"{'200-Distractor Protocol':<32} | {top1_s:>10.2f}% | {top5_s:>10.2f}% | {top10_s:>10.2f}% | {med_s:>10.1f}")

    print("\n3. 3-State Verification Outcomes (Per-Character Normalization):")
    tot_true = max(1, sum(true_states.values()))
    print(f"TRUE Claims (207 words in 145 samples):")
    print(f"  * ACCEPT:    {true_states['ACCEPT']/tot_true*100:>5.1f}%")
    print(f"  * UNCERTAIN: {true_states['UNCERTAIN']/tot_true*100:>5.1f}%")
    print(f"  * WRONG:     {true_states['WRONG']/tot_true*100:>5.1f}%")

    print("\nCORRUPTED Claims by Type:")
    print(f"{'Corruption Type':<28} | {'ACCEPT %':<10} | {'UNCERTAIN %':<12} | {'WRONG %':<10}")
    print("-" * 68)
    for ctype in corruption_types:
        d = corr_states[ctype]
        tot = max(1, sum(d.values()))
        lbl = ctype.replace("_", " ").title()
        print(f"{lbl:<28} | {d['ACCEPT']/tot*100:>8.1f}% | {d['UNCERTAIN']/tot*100:>10.1f}% | {d['WRONG']/tot*100:>8.1f}%")

    print("\n4. Comparative Table by Writer (Baseline / No Training):")
    print(f"{'Metric':<30} | {'writer_01 (DEV, 100)':<22} | {'writer_02 (Held-Out, 145)':<24}")
    print("-" * 82)
    s_cer_all = f"{p2['cer_all']:.2f}%"
    s_cer_line = f"{p2['cer_line']:.2f}%"
    s_cer_word = f"{p2['cer_word']:.2f}%"
    s_top1_f = f"{top1_f:.2f}%"
    s_top1_s = f"{top1_s:.2f}%"
    s_true_acc = f"{true_states['ACCEPT']/tot_true*100:.1f}%"
    all_corr_tot = max(1, sum(sum(d.values()) for d in corr_states.values()))
    all_corr_wrg = sum(d["WRONG"] for d in corr_states.values())
    s_corr_wrg = f"{all_corr_wrg/all_corr_tot*100:.1f}%"

    print(f"{'Overall CER (1.4x/1.8px)':<30} | {'24.07%':>20} | {s_cer_all:>22}")
    print(f"{'Lines CER':<30} | {'19.52%':>20} | {s_cer_line:>22}")
    print(f"{'Words CER':<30} | {'34.15%':>20} | {s_cer_word:>22}")
    print(f"{'Full-Dict Top-1 Rank':<30} | {'48.33%':>20} | {s_top1_f:>22}")
    print(f"{'200-Distractor Top-1 Rank':<30} | {'83.33%':>20} | {s_top1_s:>22}")
    print(f"{'TRUE Claims ACCEPT':<30} | {'93.7%':>20} | {s_true_acc:>22}")
    print(f"{'Corrupted Claims WRONG (Overall)':<30} | {'68.0%':>20} | {s_corr_wrg:>22}")
    print("=" * 95)



if __name__ == "__main__":
    run_benchmark()
