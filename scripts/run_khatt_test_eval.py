"""
scripts/run_khatt_test_eval.py
Task 2.14 Item 5:
Evaluates Step 2b checkpoints (best.pt and best_real.pt) on KHATT TEST (khatt_test_000.pt),
reporting exact CER, WER, and dot confusions on held-out authentic human handwriting.
"""

from __future__ import annotations
import sys
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vocab import Vocabulary
from model import CRNN
from dataset import collate_fn
from shards import ShardDataset
from evaluate import greedy_decode
from metrics import corpus_cer_wer
from scripts.evaluate_extended import count_dot_letter_confusions


def evaluate_khatt_test(
    ckpt_paths: list[Path],
    shard_path: Path,
    vocab_path: Path = Path("vocab.json"),
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    vocab = Vocabulary.load(vocab_path)

    if not shard_path.exists():
        print(f"Error: KHATT test shard {shard_path} not found.")
        sys.exit(1)

    print("=" * 80)
    print(f"TASK 2.14 ITEM 5: KHATT TEST EVALUATION ({shard_path.name})")
    print("=" * 80)

    results = {}

    for cp in ckpt_paths:
        if not cp.exists():
            print(f"Skipping {cp}: not found.")
            continue

        ckpt = torch.load(cp, map_location=device, weights_only=False)
        order = ckpt.get("label_order", "visual")
        model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        ds = ShardDataset(shard_path, vocab, augment=False, label_order=order)
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
        dot_errs, _ = count_dot_letter_confusions(preds, targets)
        results[cp.name] = {
            "cer": cer * 100,
            "wer": wer * 100,
            "dot_errs": dot_errs,
            "samples": len(targets),
        }
        print(f"[{cp.name:<16}] CER: {cer*100:5.2f}% | WER: {wer*100:5.2f}% | [بتثني] Confusions: {dot_errs} (N={len(targets)})")

    print("\n" + "=" * 80)
    print("SUMMARY: STEP 2b CHECKPOINTS ON KHATT TEST")
    print("=" * 80)
    print(f"{'Checkpoint':<20} | {'CER (%)':<10} | {'WER (%)':<10} | {'[بتثني] Dot Confusions':<22}")
    print("-" * 80)
    for c_name, r in results.items():
        print(f"{c_name:<20} | {r['cer']:5.2f}%    | {r['wer']:5.2f}%    | {r['dot_errs']:<22}")
    print("=" * 80)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="+", default=[
        "checkpoints/step2b_v3/best.pt",
        "checkpoints/step2b_v3/best_real.pt",
    ])
    p.add_argument("--shard_path", default="data/shards/khatt/khatt_test_000.pt")
    args = p.parse_args()

    ckpt_paths = [Path(c) for c in args.checkpoints]
    evaluate_khatt_test(ckpt_paths, Path(args.shard_path))


if __name__ == "__main__":
    main()
