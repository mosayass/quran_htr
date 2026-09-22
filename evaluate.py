"""
evaluate.py
Evaluate a trained CRNN checkpoint on a held-out test manifest.
Calculates corpus-level CER and WER using greedy and optional beam search decode,
and displays sample predictions side-by-side with ground truth.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from vocab import Vocabulary
from dataset import LineImageDataset, collate_fn
from model import CRNN
from decode import greedy_decode, beam_search_decode
from metrics import corpus_cer_wer


def evaluate(
    model: CRNN,
    loader: DataLoader,
    vocab: Vocabulary,
    device: torch.device,
    use_beam_search: bool = False,
    beam_width: int = 10,
    max_samples_display: int = 10,
):
    model.eval()
    all_preds, all_targets = [], []
    sample_pairs = []

    print(f"Evaluating {len(loader.dataset)} test samples on {device}...")
    with torch.no_grad():
        for step, batch in enumerate(loader):
            images = batch["images"].to(device)
            log_probs = model(images)  # (T, B, V)

            if use_beam_search:
                preds = beam_search_decode(log_probs, vocab, beam_width=beam_width)
            else:
                preds = greedy_decode(log_probs, vocab)

            all_preds.extend(preds)
            all_targets.extend(batch["texts"])

            if len(sample_pairs) < max_samples_display:
                for p, t in zip(preds, batch["texts"]):
                    if len(sample_pairs) < max_samples_display:
                        sample_pairs.append((p, t))

    test_cer, test_wer = corpus_cer_wer(all_preds, all_targets) if all_preds else (1.0, 1.0)
    return test_cer, test_wer, sample_pairs


def main():
    p = argparse.ArgumentParser(description="Evaluate Quran HTR CRNN on test split")
    p.add_argument("--test_manifest", required=True, help="Path to test.csv")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    p.add_argument("--vocab_path", default="vocab.json", help="Path to vocab.json")
    p.add_argument("--backbone", default="vgg_lite", choices=["vgg_lite", "mobilenetv3_small"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--beam_search", action="store_true", help="Use beam search decoding instead of greedy")
    p.add_argument("--beam_width", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=0)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    vocab = Vocabulary.load(args.vocab_path)
    test_ds = LineImageDataset(args.test_manifest, vocab)
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn
    )

    model = CRNN(vocab_size=len(vocab), backbone=args.backbone).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    epoch = ckpt.get("epoch", "unknown")
    best_cer = ckpt.get("best_cer", None)
    best_cer_str = f"{best_cer:.4f}" if best_cer is not None else "N/A"
    print(f"Loaded checkpoint: {args.checkpoint} (Epoch {epoch}, Val Best CER: {best_cer_str})")

    cer, wer, samples = evaluate(
        model, test_loader, vocab, device,
        use_beam_search=args.beam_search, beam_width=args.beam_width
    )

    decode_mode = f"Beam Search (width={args.beam_width})" if args.beam_search else "Greedy Decode"
    print("\n" + "=" * 50)
    print(f"  FINAL TEST RESULTS ({decode_mode})")
    print("=" * 50)
    print(f"  Corpus Character Error Rate (CER): {cer * 100:.2f}%")
    print(f"  Corpus Word Error Rate (WER):      {wer * 100:.2f}%")
    print("=" * 50 + "\n")

    print("--- Sample Predictions vs Ground Truth ---")
    for i, (pred, target) in enumerate(samples, 1):
        print(f"[{i}]")
        print(f"  Target: {target}")
        print(f"  Pred:   {pred}")
        print()


if __name__ == "__main__":
    main()
