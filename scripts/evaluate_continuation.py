"""
scripts/evaluate_continuation.py
Task 2.11 Step 3:
Evaluates best.pt and best_real.pt across:
- KHATT test
- Synthetic test
- Word test set (wordmix_test)
- synth_test_diac
- synth_test_unseen_fonts_diac
Produces formatted Markdown baseline comparison table.
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
        sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vocab import Vocabulary
from model import CRNN
from dataset import collate_fn
from shards import ShardDataset
from evaluate import evaluate


def eval_checkpoint_on_shards(
    ckpt_path: Path,
    shard_dict: dict[str, Path],
    vocab: Vocabulary,
    device: torch.device,
    batch_size: int = 32,
) -> dict[str, tuple[float, float]]:
    print(f"\nLoading checkpoint {ckpt_path.name}...")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    label_order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    results = {}
    for name, shard_path in shard_dict.items():
        if not shard_path.exists():
            print(f"  [Skip] {name}: {shard_path} not found")
            results[name] = (float("nan"), float("nan"))
            continue

        ds = ShardDataset(shard_path, vocab, augment=False, label_order=label_order)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        cer, wer, _ = evaluate(model, loader, vocab, device, label_order=label_order, max_samples_display=0)
        results[name] = (cer * 100, wer * 100)
        print(f"  [{name:28s}] CER: {cer*100:5.2f}% | WER: {wer*100:5.2f}%")

    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best_ckpt", default="checkpoints/cont_best.pt")
    p.add_argument("--best_real_ckpt", default="checkpoints/cont_best_real.pt")
    p.add_argument("--vocab_path", default="vocab.json")
    p.add_argument("--batch_size", type=int, default=32)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(args.vocab_path)

    test_shards = {
        "KHATT Test (Real)": ROOT / "data" / "shards" / "khatt" / "khatt_test_000.pt",
        "Synth Test (Clean Rasm)": ROOT / "data" / "shards" / "synth" / "synth_test_000.pt",
        "Synth Test Diac (Marks On)": ROOT / "data" / "shards" / "synth" / "synth_test_diac_000.pt",
        "Unseen Fonts Diac": ROOT / "data" / "shards" / "synth" / "synth_test_unseen_fonts_diac_000.pt",
        "Wordmix Test Set": ROOT / "data" / "shards" / "wordmix" / "wordmix_test_000.pt",
    }

    res_best = eval_checkpoint_on_shards(Path(args.best_ckpt), test_shards, vocab, device, args.batch_size)
    res_best_real = eval_checkpoint_on_shards(Path(args.best_real_ckpt), test_shards, vocab, device, args.batch_size)

    print("\n" + "=" * 80)
    print("BASELINE EVALUATION TABLE (TASK 2.11 STEP 3)")
    print("=" * 80)
    print(f"{'Evaluation Set':<30} | {'best.pt (Synth-Best)':<22} | {'best_real.pt (Real-Best)':<22}")
    print(f"{'':<30} | {'CER (%)':<10} {'WER (%)':<10} | {'CER (%)':<10} {'WER (%)':<10}")
    print("-" * 80)
    for name in test_shards.keys():
        b_cer, b_wer = res_best.get(name, (float("nan"), float("nan")))
        r_cer, r_wer = res_best_real.get(name, (float("nan"), float("nan")))
        print(f"{name:<30} | {b_cer:5.2f}%    {b_wer:5.2f}%    | {r_cer:5.2f}%    {r_wer:5.2f}%")
    print("=" * 80)


if __name__ == "__main__":
    main()
