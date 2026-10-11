"""
scripts/eval_char_norm.py
Task 2.19 Item 2:
Evaluate per-character normalization on synthetic test shards:
- Compares per-character LLR vs legacy per-frame LLR.
- Reports 3-state outcomes (ACCEPT / UNCERTAIN / WRONG) separately for:
  1. True claims (clean synthetic text)
  2. Dot-family letter swap
  3. Letter deletion / insertion
"""

from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
import random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from model import CRNN
from shards import ShardDataset
from verify import ctc_viterbi_alignment, greedy_decode_segment
from task_2_17_suite import apply_corruption, extract_mutashabih_pairs


def evaluate_norm(
    checkpoint_path: Path,
    vocab_path: Path,
    shards_path: Path,
    n_samples: int = 500,
    seed: int = 42,
):
    device = torch.device("cpu")
    vocab = Vocabulary.load(vocab_path)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    shard_files = [shards_path] if shards_path.is_file() else sorted(list(shards_path.glob("synth_test_000.pt")))
    if not shard_files:
        shard_files = sorted(list(shards_path.glob("*test*.pt")))
    ds = ShardDataset(shard_files, vocab=vocab, augment=False, label_order=order)
    print(f"Loaded {len(ds)} samples from {[p.name for p in shard_files]}. Evaluating {n_samples}...")

    rng = random.Random(seed)
    indices = rng.sample(range(len(ds)), min(n_samples, len(ds)))

    # Track scores: mode ("char", "frame") -> type -> list of LLRs
    results = {
        "char": defaultdict(list),
        "frame": defaultdict(list),
    }

    mut_pairs = extract_mutashabih_pairs()
    mut_dict = defaultdict(list)
    for p in mut_pairs:
        mut_dict[p["prefix"]].append(p["word_b"])

    with open(ROOT / "data" / "tanzil" / "quran-simple-clean.txt", encoding="utf-8") as f:
        quran_words = list(set(f.read().split()))

    for count, idx in enumerate(indices):
        item = ds[idx]
        img_tensor = item.image.unsqueeze(0).to(device)
        clean_text = item.text

        with torch.no_grad():
            lp = model(img_tensor).squeeze(1).numpy()  # (T, V)

        T, V = lp.shape
        max_free_per_frame = np.max(lp, axis=1)

        # Evaluate target word on TRUE claim and CORRUPTED claims
        words = clean_text.split()
        if not words:
            continue
        c_idx = len(words) // 2
        target_w = words[c_idx]

        # Forced alignment for clean text
        label_text = clean_text[::-1] if order == "visual" else clean_text
        enc_target = [vocab.char2idx[c] for c in label_text if c in vocab.char2idx]
        path, _ = ctc_viterbi_alignment(lp, enc_target, blank_idx=vocab.blank_id)

        # Word span frames
        vis_words = [w[::-1] for w in words[::-1]] if order == "visual" else words
        char_pos = 0
        w_spans = []
        for w in vis_words:
            w_enc = [vocab.char2idx[c] for c in w if c in vocab.char2idx]
            w_len = len(w_enc)
            w_spans.append((w, char_pos, char_pos + w_len, w_enc))
            char_pos += w_len + 1

        v_idx = len(words) - 1 - c_idx if order == "visual" else c_idx
        _, c_start, c_end, w_enc = w_spans[v_idx]
        active_states = set(range(2 * c_start, 2 * c_end + 1))
        frames = [t for t in range(T) if path[t] in active_states]
        t1 = min(frames) if frames else 0
        t2 = max(frames) if frames else T - 1
        seg_len = max(1, t2 - t1 + 1)
        char_len = max(1, len(w_enc))

        seg_lp = lp[t1 : t2 + 1]
        _, s_exp_true = ctc_viterbi_alignment(seg_lp, w_enc, blank_idx=vocab.blank_id)
        s_free = float(np.sum(max_free_per_frame[t1 : t2 + 1]))

        llr_true_frame = (s_exp_true - s_free) / seg_len
        llr_true_char = (s_exp_true - s_free) / char_len

        results["frame"]["true"].append(llr_true_frame)
        results["char"]["true"].append(llr_true_char)

        # Corruptions: Dot-swap and Del/Ins
        for ctype in ["dot_swap", "del_ins"]:
            corr = apply_corruption(clean_text, ctype, quran_words, mut_dict, rng)
            if corr is None:
                continue
            corr_text, corr_idx = corr
            corr_words = corr_text.split()
            if corr_idx >= len(corr_words):
                continue
            corr_w = corr_words[corr_idx]

            # Score corrupted word on its segment
            corr_w_label = corr_w[::-1] if order == "visual" else corr_w
            corr_w_enc = [vocab.char2idx[c] for c in corr_w_label if c in vocab.char2idx]
            corr_char_len = max(1, len(corr_w_enc))

            # Segment frames from full forced alignment or target word
            _, s_exp_corr = ctc_viterbi_alignment(seg_lp, corr_w_enc, blank_idx=vocab.blank_id)
            llr_corr_frame = (s_exp_corr - s_free) / seg_len
            llr_corr_char = (s_exp_corr - s_free) / corr_char_len

            results["frame"][ctype].append(llr_corr_frame)
            results["char"][ctype].append(llr_corr_char)

        if (count + 1) % 20 == 0:
            print(f"  Processed {count + 1}/{len(indices)} samples...", flush=True)

    return results


def print_comparison(results: dict, t_acc_frame=-0.60, t_wrg_frame=-1.10, t_acc_char=-0.60, t_wrg_char=-0.95):
    print("\n" + "=" * 90, flush=True)
    print("PER-CHARACTER vs PER-FRAME NORMALIZATION: 3-STATE OUTCOMES (SYNTHETIC)", flush=True)
    print("=" * 90, flush=True)

    for mode, t_acc, t_wrg in [("frame", t_acc_frame, t_wrg_frame), ("char", t_acc_char, t_wrg_char)]:
        print(f"\n--- NORMALIZATION: {mode.upper()} (t_accept={t_acc:.2f}, t_wrong={t_wrg:.2f}) ---", flush=True)
        print(f"{'Condition':<28} | {'ACCEPT %':<12} | {'UNCERTAIN %':<12} | {'WRONG %':<12} | {'Mean LLR':<10}", flush=True)
        print("-" * 80, flush=True)
        for ctype, label in [("true", "TRUE Claims (Clean)"), ("dot_swap", "Dot-family Letter Swap"), ("del_ins", "Letter Deletion / Insertion")]:
            scores = np.array(results[mode][ctype], dtype=np.float32)
            n = len(scores)
            acc = float(np.mean(scores >= t_acc) * 100)
            wrg = float(np.mean(scores <= t_wrg) * 100)
            unc = 100.0 - acc - wrg
            mean_llr = float(np.mean(scores))
            print(f"{label:<28} | {acc:>10.1f}% | {unc:>10.1f}% | {wrg:>10.1f}% | {mean_llr:>8.3f}", flush=True)
    print("=" * 90, flush=True)


if __name__ == "__main__":
    res = evaluate_norm(
        checkpoint_path=ROOT / "checkpoints" / "best.pt",
        vocab_path=ROOT / "vocab.json",
        shards_path=ROOT / "data" / "shards" / "synth",
        n_samples=100,
    )
    print_comparison(res)

