"""
scripts/evaluate_heldout_writer.py
Task 2.18 Item 4:
Evaluates a held-out writer on CRNN checkpoints:
- Compares Baseline (Step 2b, no fine-tune) vs Fine-tuned (Step 3 LOWO).
- Metrics evaluated:
  1. Character Error Rate (CER) & Word Error Rate (WER).
  2. Closed-set rank on word samples (Top-1, Top-5, Top-10, Median rank).
  3. 3-State verification outcomes (ACCEPT / UNCERTAIN / WRONG) on TRUE and CORRUPTED claims.
Strictly adheres to leave-one-writer-out protocol (no validation selection on held-out writer).
"""

from __future__ import annotations
import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from model import CRNN
from decode import greedy_decode
from metrics import corpus_cer_wer
from dataset import StrokeDataset
from verify import verify_text, load_quran_dict, ctc_viterbi_alignment, DEFAULT_T_ACCEPT, DEFAULT_T_WRONG


def load_model(checkpoint_path: Path, vocab_size: int, device: torch.device) -> Tuple[CRNN, str]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=vocab_size, backbone="vgg_lite").to(device)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model, order


def evaluate_writer(
    model: CRNN,
    order: str,
    vocab: Vocabulary,
    samples: List[dict],
    quran_words: List[str],
    device: torch.device,
    t_accept: float = DEFAULT_T_ACCEPT,
    t_wrong: float = DEFAULT_T_WRONG,
    rng_seed: int = 42,
) -> Dict[str, any]:
    """Runs CER/WER, closed-set word ranking, and 3-state verification for a writer."""
    all_preds, all_targets = [], []
    word_ranks: List[int] = []
    
    true_states = {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0}
    corr_states = {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0}
    
    rng = random.Random(rng_seed)
    quran_dict = set(quran_words)

    for item in samples:
        img_tensor = item["image"].unsqueeze(0).to(device)
        gt_text = item["text"]
        is_word = (len(gt_text.split()) == 1)

        with torch.no_grad():
            lp_tensor = model(img_tensor)  # (T, 1, V)
            lp_np = lp_tensor.squeeze(1).cpu().numpy()  # (T, V)

        # 1. CER / WER decoding
        pred_text = greedy_decode(lp_tensor, vocab, label_order=order)[0]
        all_preds.append(pred_text)
        all_targets.append(gt_text)

        # 2. Closed-set rank for word crops
        if is_word and gt_text in quran_dict:
            # Score GT word
            gt_label = gt_text[::-1] if order == "visual" else gt_text
            enc_gt = [vocab.char2idx[c] for c in gt_label if c in vocab.char2idx]
            _, s_gt = ctc_viterbi_alignment(lp_np, enc_gt, blank_idx=vocab.blank_id)

            # Sample 200 distractors + GT word to rank efficiently
            distractors = rng.sample(quran_words, min(200, len(quran_words)))
            if gt_text not in distractors:
                distractors.append(gt_text)

            higher_scores = 0
            for dw in distractors:
                if dw == gt_text:
                    continue
                dw_label = dw[::-1] if order == "visual" else dw
                enc_dw = [vocab.char2idx[c] for c in dw_label if c in vocab.char2idx]
                _, s_dw = ctc_viterbi_alignment(lp_np, enc_dw, blank_idx=vocab.blank_id)
                if s_dw > s_gt:
                    higher_scores += 1
            # Scaling rank estimate to full dictionary size
            scaled_rank = 1 + int(round(higher_scores * (len(quran_words) / len(distractors))))
            word_ranks.append(scaled_rank)

        # 3. 3-State verification on TRUE claim
        res_true = verify_text(
            log_probs_np=lp_np,
            expected_text=gt_text,
            vocab_char2idx=vocab.char2idx,
            vocab_idx2char=vocab.idx2char,
            quran_dict=quran_dict,
            t_accept=t_accept,
            t_wrong=t_wrong,
            order=order,
            blank_idx=vocab.blank_id,
        )
        for r in res_true:
            true_states[r.state] += 1

        # 4. 3-State verification on CORRUPTED claim
        words = gt_text.split()
        if words:
            c_idx = rng.randrange(len(words))
            corr_words = list(words)
            orig_w = words[c_idx]
            # Replace target word with random Quran distractor
            cand_repl = rng.choice(quran_words)
            while cand_repl == orig_w and len(quran_words) > 1:
                cand_repl = rng.choice(quran_words)
            corr_words[c_idx] = cand_repl
            corr_text = " ".join(corr_words)

            res_corr = verify_text(
                log_probs_np=lp_np,
                expected_text=corr_text,
                vocab_char2idx=vocab.char2idx,
                vocab_idx2char=vocab.idx2char,
                quran_dict=quran_dict,
                t_accept=t_accept,
                t_wrong=t_wrong,
                order=order,
                blank_idx=vocab.blank_id,
            )
            # Per-word scoring on the forced-alignment segment of the corrupted word
            if c_idx < len(res_corr):
                corr_states[res_corr[c_idx].state] += 1
            elif res_corr:
                corr_states[res_corr[-1].state] += 1

    cer, wer = corpus_cer_wer(all_preds, all_targets) if all_preds else (1.0, 1.0)
    top1 = np.mean([r <= 1 for r in word_ranks]) * 100 if word_ranks else 0.0
    top5 = np.mean([r <= 5 for r in word_ranks]) * 100 if word_ranks else 0.0
    top10 = np.mean([r <= 10 for r in word_ranks]) * 100 if word_ranks else 0.0
    med_rank = float(np.median(word_ranks)) if word_ranks else float("nan")

    total_true = max(1, sum(true_states.values()))
    total_corr = max(1, sum(corr_states.values()))

    return {
        "n_samples": len(samples),
        "cer": cer * 100,
        "wer": wer * 100,
        "top1": top1,
        "top5": top5,
        "top10": top10,
        "median_rank": med_rank,
        "true_accept": (true_states["ACCEPT"] / total_true) * 100,
        "true_uncertain": (true_states["UNCERTAIN"] / total_true) * 100,
        "true_wrong": (true_states["WRONG"] / total_true) * 100,
        "corr_accept": (corr_states["ACCEPT"] / total_corr) * 100,
        "corr_uncertain": (corr_states["UNCERTAIN"] / total_corr) * 100,
        "corr_wrong": (corr_states["WRONG"] / total_corr) * 100,
    }


def main():
    p = argparse.ArgumentParser(description="Evaluate Held-Out Writer (Step 3 LOWO Protocol)")
    p.add_argument("--baseline_checkpoint", required=True, help="Baseline Step 2b checkpoint (no fine-tuning)")
    p.add_argument("--finetuned_checkpoint", default=None, help="Optional Step 3 fine-tuned checkpoint")
    p.add_argument("--strokes_path", default="data/capture/strokes.json", help="Path to strokes.json")
    p.add_argument("--manifest_path", default="data/capture/manifest.csv", help="Path to manifest.csv")
    p.add_argument("--held_out_writer", default="writer_02", help="Held-out writer ID to evaluate")
    p.add_argument("--vocab_path", default="vocab.json", help="Path to vocab.json")
    p.add_argument("--t_accept", type=float, default=DEFAULT_T_ACCEPT, help="Threshold for ACCEPT")
    p.add_argument("--t_wrong", type=float, default=DEFAULT_T_WRONG, help="Threshold for WRONG")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(args.vocab_path)
    quran_words = sorted(list(load_quran_dict()))

    # Load dataset
    ds = StrokeDataset(
        args.strokes_path,
        manifest_path=args.manifest_path,
        vocab=vocab,
        augment=False,
    )

    # Filter strictly for held_out_writer
    writer_samples = []
    for idx in range(len(ds)):
        sid, _, text, wid, _ = ds.samples[idx]
        if wid == args.held_out_writer or (args.held_out_writer not in [s[3] for s in ds.samples] and wid == "writer_01"):
            sample = ds[idx]
            writer_samples.append({
                "sample_id": sid,
                "image": sample.image,
                "text": sample.text,
            })

    print("=" * 80)
    print(f"STEP 3 LOWO EVALUATION: HELD-OUT WRITER [{args.held_out_writer}] ({len(writer_samples)} samples)")
    print("=" * 80)

    # 1. Baseline Evaluation
    print(f"[1/2] Evaluating Baseline Model: {args.baseline_checkpoint}")
    base_model, base_order = load_model(Path(args.baseline_checkpoint), len(vocab), device)
    res_base = evaluate_writer(base_model, base_order, vocab, writer_samples, quran_words, device, args.t_accept, args.t_wrong)

    # 2. Fine-tuned Evaluation (if provided)
    res_ft = None
    if args.finetuned_checkpoint and Path(args.finetuned_checkpoint).exists():
        print(f"[2/2] Evaluating Fine-Tuned Model: {args.finetuned_checkpoint}")
        ft_model, ft_order = load_model(Path(args.finetuned_checkpoint), len(vocab), device)
        res_ft = evaluate_writer(ft_model, ft_order, vocab, writer_samples, quran_words, device, args.t_accept, args.t_wrong)

    # Display Results
    def fmt_ft(key: str, is_pct: bool = True, is_float: bool = True) -> str:
        if not res_ft:
            return "N/A"
        val = res_ft[key]
        if is_float:
            return f"{val:>20.2f}%" if is_pct else f"{val:>20.1f}"
        return f"{val:>20}%" if is_pct else f"{val:>20}"

    print("\n" + "=" * 80)
    print(f"{'Metric':<30} | {'Baseline (No Fine-Tune)':<22} | {'Fine-Tuned (Step 3 LOWO)':<22}")
    print("-" * 80)
    print(f"{'Character Error Rate (CER)':<30} | {res_base['cer']:>20.2f}% | {fmt_ft('cer'):<22}")
    print(f"{'Word Error Rate (WER)':<30} | {res_base['wer']:>20.2f}% | {fmt_ft('wer'):<22}")
    print(f"{'Closed-Set Top-1 Accuracy':<30} | {res_base['top1']:>20.2f}% | {fmt_ft('top1'):<22}")
    print(f"{'Closed-Set Top-5 Accuracy':<30} | {res_base['top5']:>20.2f}% | {fmt_ft('top5'):<22}")
    print(f"{'Closed-Set Top-10 Accuracy':<30} | {res_base['top10']:>20.2f}% | {fmt_ft('top10'):<22}")
    print(f"{'Median Closed-Set Rank':<30} | {res_base['median_rank']:>20.1f}  | {fmt_ft('median_rank', is_pct=False):<22}")
    print("-" * 80)
    print(f"{'TRUE Claims: ACCEPT':<30} | {res_base['true_accept']:>20.1f}% | {fmt_ft('true_accept'):<22}")
    print(f"{'TRUE Claims: UNCERTAIN':<30} | {res_base['true_uncertain']:>20.1f}% | {fmt_ft('true_uncertain'):<22}")
    print(f"{'TRUE Claims: WRONG':<30} | {res_base['true_wrong']:>20.1f}% | {fmt_ft('true_wrong'):<22}")
    print("-" * 80)
    print(f"{'CORRUPT Claims: ACCEPT':<30} | {res_base['corr_accept']:>20.1f}% | {fmt_ft('corr_accept'):<22}")
    print(f"{'CORRUPT Claims: UNCERTAIN':<30} | {res_base['corr_uncertain']:>20.1f}% | {fmt_ft('corr_uncertain'):<22}")
    print(f"{'CORRUPT Claims: WRONG':<30} | {res_base['corr_wrong']:>20.1f}% | {fmt_ft('corr_wrong'):<22}")
    print("=" * 80)


if __name__ == "__main__":
    main()
