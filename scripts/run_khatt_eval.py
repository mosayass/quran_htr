"""
scripts/run_khatt_eval.py
Task 2.9 Steps 1 & 3:
1. Evaluate best.pt and best_real.pt on KHATT val and test separately.
3. KHATT test error analysis: top-15 substitutions/deletions, dot-pair confusions,
   CER by line length (1-3, 4-6, 7-9 words).
"""

from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from torch.utils.data import DataLoader
import Levenshtein

from vocab import Vocabulary
from model import CRNN
from shards import ShardDataset
from dataset import collate_fn
from decode import greedy_decode
from metrics import corpus_cer_wer


DOT_GROUPS = [
    ("ب ت ث ن ي", set("بتثني")),
    ("ج ح خ", set("جحخ")),
    ("د ذ", set("دذ")),
    ("ر ز", set("رز")),
    ("س ش", set("سش")),
    ("ص ض", set("صض")),
    ("ط ظ", set("طظ")),
    ("ع غ", set("عغ")),
    ("ف ق", set("فق")),
]


def load_model_from_ckpt(ckpt_path: Path, vocab_size: int, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=vocab_size, backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch = ckpt.get("epoch", "unknown")
    return model, order, epoch


def evaluate_shard(model: CRNN, shard_path: Path, vocab: Vocabulary, order: str, device: torch.device, batch_size: int = 32):
    ds = ShardDataset(shard_path, vocab, augment=False, label_order=order)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)
    
    preds, targets = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            log_probs = model(imgs)
            batch_preds = greedy_decode(log_probs, vocab, label_order=order)
            preds.extend(batch_preds)
            targets.extend(batch["texts"])
            
    cer, wer = corpus_cer_wer(preds, targets)
    return cer, wer, preds, targets


def run_error_analysis(preds: list[str], targets: list[str]):
    substitutions = Counter()
    deletions = Counter()
    insertions = Counter()
    
    # Dot confusions: group_name -> Counter((target, pred))
    dot_confusions = defaultdict(Counter)

    # Line length buckets: (1-3, 4-6, 7-9, 10+)
    length_buckets = {
        "1-3 words": {"preds": [], "targets": []},
        "4-6 words": {"preds": [], "targets": []},
        "7-9 words": {"preds": [], "targets": []},
        "10+ words": {"preds": [], "targets": []},
    }

    for p, t in zip(preds, targets):
        num_words = len(t.split())
        if num_words <= 3:
            b_key = "1-3 words"
        elif num_words <= 6:
            b_key = "4-6 words"
        elif num_words <= 9:
            b_key = "7-9 words"
        else:
            b_key = "10+ words"
        length_buckets[b_key]["preds"].append(p)
        length_buckets[b_key]["targets"].append(t)

        ops = Levenshtein.opcodes(t, p)
        for tag, i1, i2, j1, j2 in ops:
            if tag == "replace":
                sub_t = t[i1:i2]
                sub_p = p[j1:j2]
                min_l = min(len(sub_t), len(sub_p))
                for k in range(min_l):
                    char_t, char_p = sub_t[k], sub_p[k]
                    substitutions[(char_t, char_p)] += 1
                    # Check dot-pair confusion groups
                    for grp_name, grp_set in DOT_GROUPS:
                        if char_t in grp_set and char_p in grp_set and char_t != char_p:
                            dot_confusions[grp_name][(char_t, char_p)] += 1

                # Remaining unaligned
                if len(sub_t) > min_l:
                    for k in range(min_l, len(sub_t)):
                        deletions[sub_t[k]] += 1
                elif len(sub_p) > min_l:
                    for k in range(min_l, len(sub_p)):
                        insertions[sub_p[k]] += 1

            elif tag == "delete":
                for char in t[i1:i2]:
                    deletions[char] += 1
            elif tag == "insert":
                for char in p[j1:j2]:
                    insertions[char] += 1

    return substitutions, deletions, dot_confusions, length_buckets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running KHATT Evaluation on {device}...")

    vocab = Vocabulary.load("vocab.json")
    val_shard = Path("data/shards/khatt/khatt_val_000.pt")
    test_shard = Path("data/shards/khatt/khatt_test_000.pt")

    ckpt_best = Path("checkpoints/best.pt")
    ckpt_best_real = Path("checkpoints/best_real.pt")

    if not ckpt_best.exists() or not ckpt_best_real.exists():
        print(f"Error: Missing checkpoints in checkpoints/ folder.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("STEP 1: KHATT EVALUATION MATRIX (val & test)")
    print("=" * 60)

    results = {}
    test_preds_best_real, test_targets_best_real = None, None

    for name, path in [("best.pt", ckpt_best), ("best_real.pt", ckpt_best_real)]:
        model, order, ep = load_model_from_ckpt(path, len(vocab), device)
        print(f"\nEvaluating {name} (Epoch {ep}, order={order})...")

        # KHATT val
        val_cer, val_wer, _, _ = evaluate_shard(model, val_shard, vocab, order, device)
        # KHATT test
        test_cer, test_wer, preds, targets = evaluate_shard(model, test_shard, vocab, order, device)

        results[name] = {
            "val_cer": val_cer, "val_wer": val_wer,
            "test_cer": test_cer, "test_wer": test_wer,
        }
        print(f"  [{name} on KHATT val ] CER: {val_cer*100:.2f}% | WER: {val_wer*100:.2f}%")
        print(f"  [{name} on KHATT test] CER: {test_cer*100:.2f}% | WER: {test_wer*100:.2f}%")

        if name == "best_real.pt":
            test_preds_best_real = preds
            test_targets_best_real = targets

    print("\n--- Summary Matrix ---")
    print(f"{'Checkpoint':<15} | {'Val CER':<10} | {'Val WER':<10} | {'Test CER':<10} | {'Test WER':<10}")
    print("-" * 65)
    for name, m in results.items():
        print(f"{name:<15} | {m['val_cer']*100:6.2f}%   | {m['val_wer']*100:6.2f}%   | {m['test_cer']*100:6.2f}%   | {m['test_wer']*100:6.2f}%")

    print("\n" + "=" * 60)
    print("STEP 3: KHATT TEST ERROR ANALYSIS (best_real.pt)")
    print("=" * 60)

    subs, dels, dot_confs, length_buckets = run_error_analysis(test_preds_best_real, test_targets_best_real)

    print("\n[Top 15 Character Substitutions: target -> predicted]")
    for i, ((t_c, p_c), count) in enumerate(subs.most_common(15), 1):
        print(f"  {i:2d}. '{t_c}' -> '{p_c}': {count} occurrences")

    print("\n[Top 15 Character Deletions: target -> omitted]")
    for i, (t_c, count) in enumerate(dels.most_common(15), 1):
        print(f"  {i:2d}. '{t_c}': {count} occurrences")

    print("\n[Dot-Pair Confusions within Homoglyph Groups]")
    for grp_name, _ in DOT_GROUPS:
        confs = dot_confs[grp_name]
        total_in_grp = sum(confs.values())
        top_pairs = ", ".join([f"{t}->{p} ({cnt})" for (t, p), cnt in confs.most_common(5)]) or "None"
        print(f"  Group [{grp_name:<11}]: {total_in_grp:3d} confusions | Top: {top_pairs}")

    print("\n[CER by Line Length (Word Buckets)]")
    for b_key, data in length_buckets.items():
        if data["targets"]:
            b_cer, b_wer = corpus_cer_wer(data["preds"], data["targets"])
            print(f"  {b_key:<12}: {len(data['targets']):4d} lines | CER: {b_cer*100:6.2f}% | WER: {b_wer*100:6.2f}%")
        else:
            print(f"  {b_key:<12}:    0 lines | N/A")


if __name__ == "__main__":
    main()
