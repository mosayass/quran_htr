"""
scripts/evaluate_extended.py
Task 2.13 Item 4:
Extended Evaluation Tool:
1. Supports multiple checkpoints (e.g. cont_best.pt, cont_best_real.pt, step2b checkpoints).
2. Computes [ب ت ث ن ي] dot-letter confusion matrix & count on KHATT test and clean synth test.
3. Groups CER/WER into per-word-length buckets (1 word, 2 words, 3-5 words, 6-9 words).
4. Evaluates across:
   - KHATT Test (Real)
   - Synth Test (Clean Rasm)
   - Synth Test Diac (Marks On)
   - Unseen Fonts Diac
   - Wordmix Test Set
   - Corpus v3 Test Set
   - Capture Set (if present in data/capture/manifest.csv)
"""

from __future__ import annotations
import argparse
import difflib
from collections import Counter, defaultdict
from pathlib import Path
import sys
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
    sys.stderr.reconfigure(line_buffering=True, encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vocab import Vocabulary
from model import CRNN
from dataset import collate_fn, LineImageDataset
from shards import ShardDataset
from evaluate import evaluate, greedy_decode
from metrics import corpus_cer_wer

DOT_LETTERS = set("بتثني")


def count_dot_letter_confusions(preds: list[str], targets: list[str]) -> tuple[int, Counter]:
    """
    Counts substitutions among [ب ت ث ن ي] using character-level sequence matching.
    Returns (total_confusions, Counter of (target_char, pred_char) -> count).
    """
    total_confusions = 0
    confusion_pairs = Counter()

    for pred, target in zip(preds, targets):
        matcher = difflib.SequenceMatcher(None, target, pred)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                t_sub = target[i1:i2]
                p_sub = pred[j1:j2]
                # Compare pairwise characters
                min_len = min(len(t_sub), len(p_sub))
                for k in range(min_len):
                    t_ch = t_sub[k]
                    p_ch = p_sub[k]
                    if t_ch in DOT_LETTERS and p_ch in DOT_LETTERS and t_ch != p_ch:
                        confusion_pairs[(t_ch, p_ch)] += 1
                        total_confusions += 1
    return total_confusions, confusion_pairs


def evaluate_bucketed(
    model: torch.nn.Module,
    loader: DataLoader,
    vocab: Vocabulary,
    device: torch.device,
    label_order: str = "visual",
) -> dict:
    model.eval()
    all_preds = []
    all_targets = []
    buckets = {
        "1w": {"preds": [], "targets": []},
        "2w": {"preds": [], "targets": []},
        "3-5w": {"preds": [], "targets": []},
        "6-9w": {"preds": [], "targets": []},
    }

    with torch.no_grad():
        for batch in loader:
            images = batch["images"].to(device)
            log_probs = model(images)
            preds = greedy_decode(log_probs, vocab, label_order=label_order)
            all_preds.extend(preds)
            all_targets.extend(batch["texts"])

            for p, t in zip(preds, batch["texts"]):
                w_count = len(t.split())
                if w_count == 1:
                    b_key = "1w"
                elif w_count == 2:
                    b_key = "2w"
                elif 3 <= w_count <= 5:
                    b_key = "3-5w"
                else:
                    b_key = "6-9w"
                buckets[b_key]["preds"].append(p)
                buckets[b_key]["targets"].append(t)

    if all_preds:
        overall_cer, overall_wer = corpus_cer_wer(all_preds, all_targets)
    else:
        overall_cer, overall_wer = float("nan"), float("nan")

    dot_confusions, dot_pairs = count_dot_letter_confusions(all_preds, all_targets)

    bucket_stats = {}
    for b_key, b_data in buckets.items():
        if b_data["preds"]:
            b_cer, b_wer = corpus_cer_wer(b_data["preds"], b_data["targets"])
            bucket_stats[b_key] = (b_cer * 100, b_wer * 100, len(b_data["preds"]))
        else:
            bucket_stats[b_key] = (float("nan"), float("nan"), 0)

    return {
        "cer": overall_cer * 100,
        "wer": overall_wer * 100,
        "dot_confusions": dot_confusions,
        "dot_pairs": dot_pairs,
        "buckets": bucket_stats,
    }


def main():
    p = argparse.ArgumentParser(description="Extended Evaluation Tool for Quran HTR")
    p.add_argument("--checkpoints", nargs="+", default=["checkpoints/cont_best.pt", "checkpoints/cont_best_real.pt"],
                   help="List of checkpoint paths to evaluate")
    p.add_argument("--vocab_path", default="vocab.json")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    device = torch.device(args.device)
    vocab = Vocabulary.load(args.vocab_path)
    print(f"Device: {device} | Checkpoints to evaluate: {args.checkpoints}")

    # Look for KHATT test and val shards
    khatt_test_path = ROOT / "data" / "shards" / "khatt" / "khatt_test_000.pt"
    if not khatt_test_path.exists():
        for candidate in [
            Path("/content/data/shards/khatt/khatt_test_000.pt"),
            Path("/content/drive/MyDrive/quran_htr_data/khatt_test_000.pt"),
            Path("/content/drive/MyDrive/quran_htr_data/shards/khatt/khatt_test_000.pt"),
            Path("/content/drive/MyDrive/shards_khatt/khatt_test_000.pt"),
            Path("/content/drive/MyDrive/khatt_test_000.pt"),
        ]:
            if candidate.exists():
                khatt_test_path = candidate
                break

    khatt_val_path = ROOT / "data" / "shards" / "khatt" / "khatt_val_000.pt"
    if not khatt_val_path.exists():
        for candidate in [
            Path("/content/data/shards/khatt/khatt_val_000.pt"),
            Path("/content/drive/MyDrive/quran_htr_data/khatt_val_000.pt"),
            Path("/content/drive/MyDrive/quran_htr_data/shards/khatt/khatt_val_000.pt"),
            Path("/content/drive/MyDrive/shards_khatt/khatt_val_000.pt"),
        ]:
            if candidate.exists():
                khatt_val_path = candidate
                break

    test_sets = {}
    if khatt_test_path.exists():
        test_sets["KHATT Test (Real)"] = khatt_test_path
    if khatt_val_path.exists():
        test_sets["KHATT Val (Real)"] = khatt_val_path

    test_sets.update({
        "Synth Test (Clean Rasm)": ROOT / "data" / "shards" / "synth" / "synth_test_000.pt",
        "Synth Test Diac (Marks On)": ROOT / "data" / "shards" / "synth" / "synth_test_diac_000.pt",
        "Unseen Fonts Diac": ROOT / "data" / "shards" / "synth" / "synth_test_unseen_fonts_diac_000.pt",
        "Wordmix Test Set": ROOT / "data" / "shards" / "wordmix" / "wordmix_test_000.pt",
        "Corpus v3 Test Set": ROOT / "data" / "shards" / "corpus_v3" / "corpus_v3_test_000.pt",
    })

    # Include capture set if present
    capture_csv = ROOT / "data" / "capture" / "manifest.csv"
    if capture_csv.exists():
        test_sets["Capture Set (Tablet)"] = capture_csv

    all_ckpt_results = {}

    for ckpt_str in args.checkpoints:
        ckpt_path = Path(ckpt_str)
        if not ckpt_path.exists():
            print(f"Skipping {ckpt_path}: not found.")
            continue

        print(f"\n{'='*75}\nLoading checkpoint: {ckpt_path.name}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        label_order = ckpt.get("label_order", "visual")
        model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        ckpt_results = {}
        for set_name, p_set in test_sets.items():
            if not p_set.exists():
                print(f"  [Skip] {set_name}: {p_set.name} not found")
                continue

            if p_set.suffix == ".csv":
                ds = LineImageDataset(p_set, vocab, augment=False, label_order=label_order)
            else:
                ds = ShardDataset(p_set, vocab, augment=False, label_order=label_order)

            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
            res = evaluate_bucketed(model, loader, vocab, device, label_order=label_order)
            ckpt_results[set_name] = res

            print(f"  [{set_name:28s}] CER: {res['cer']:5.2f}% | WER: {res['wer']:5.2f}% | [بتثني] Confusions: {res['dot_confusions']}")
            # Print word-length buckets
            b_str = " | ".join(f"{k}: {v[0]:.1f}% ({v[2]})" for k, v in res["buckets"].items() if v[2] > 0)
            if b_str:
                print(f"     Buckets (CER): {b_str}")

        all_ckpt_results[ckpt_path.name] = ckpt_results

    # Summary Table
    print("\n" + "=" * 85)
    print("EXTENDED BASELINE EVALUATION & [بتثني] CONFUSION REPORT")
    print("=" * 85)
    header = f"{'Evaluation Set':<28}"
    for c_name in all_ckpt_results.keys():
        header += f" | {c_name[:16]:<16} (Dot Err)"
    print(header)
    print("-" * 85)

    for set_name in test_sets.keys():
        row = f"{set_name:<28}"
        has_any = False
        for c_name, c_res in all_ckpt_results.items():
            if set_name in c_res:
                has_any = True
                cer = c_res[set_name]["cer"]
                dot_c = c_res[set_name]["dot_confusions"]
                row += f" | {cer:5.2f}% ({dot_c:3d})"
            else:
                row += f" | {'nan%':<16}"
        if has_any:
            print(row)
    print("=" * 85)


if __name__ == "__main__":
    main()
