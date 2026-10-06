"""
scripts/run_word_eval.py
Task 2.9 Step 2:
Word-level eval on the new best.pt for Task 2.7 sets (synth_test_000.pt and synth_test_unseen_fonts_000.pt):
- CER and WER
- Exact-word accuracy
- Buckets by word length (1-2, 3-4, 5+ characters)
- Top-20 character confusions
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


def evaluate_dataset(model: CRNN, shard_path: Path, vocab: Vocabulary, order: str, device: torch.device, batch_size: int = 32):
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

    return preds, targets


def analyze_words_and_confusions(preds: list[str], targets: list[str]):
    # Word length buckets: total target words and correctly predicted words
    buckets = {
        "1-2 chars": {"total": 0, "correct": 0},
        "3-4 chars": {"total": 0, "correct": 0},
        "5+ chars":  {"total": 0, "correct": 0},
    }
    total_words = 0
    total_correct = 0

    char_confusions = Counter()

    for p, t in zip(preds, targets):
        t_words = t.split()
        p_words = p.split()

        # Word-level alignment via Levenshtein opcodes
        ops = Levenshtein.opcodes(t_words, p_words)
        for tag, i1, i2, j1, j2 in ops:
            if tag == "equal":
                for w in t_words[i1:i2]:
                    l = len(w)
                    b_key = "1-2 chars" if l <= 2 else ("3-4 chars" if l <= 4 else "5+ chars")
                    buckets[b_key]["total"] += 1
                    buckets[b_key]["correct"] += 1
                    total_words += 1
                    total_correct += 1
            elif tag in ("replace", "delete"):
                for w in t_words[i1:i2]:
                    l = len(w)
                    b_key = "1-2 chars" if l <= 2 else ("3-4 chars" if l <= 4 else "5+ chars")
                    buckets[b_key]["total"] += 1
                    total_words += 1

        # Character-level confusions via Levenshtein
        c_ops = Levenshtein.opcodes(t, p)
        for tag, i1, i2, j1, j2 in c_ops:
            if tag == "replace":
                sub_t = t[i1:i2]
                sub_p = p[j1:j2]
                min_l = min(len(sub_t), len(sub_p))
                for k in range(min_l):
                    char_confusions[(sub_t[k], sub_p[k])] += 1

    return buckets, total_words, total_correct, char_confusions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running Word-Level Evaluation on {device}...")

    vocab = Vocabulary.load("vocab.json")
    ckpt_path = Path("checkpoints/best.pt")
    if not ckpt_path.exists():
        print(f"Error: Missing {ckpt_path}")
        sys.exit(1)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    epoch = ckpt.get("epoch", "unknown")
    print(f"Loaded checkpoint: {ckpt_path} (Epoch {epoch}, label_order={order})")

    test_sets = [
        ("Held-out Test Surahs", Path("data/shards/synth/synth_test_000.pt")),
        ("Held-out Unseen Fonts", Path("data/shards/synth/synth_test_unseen_fonts_000.pt")),
    ]

    for set_name, shard_p in test_sets:
        print("\n" + "=" * 60)
        print(f"EVALUATION: {set_name} ({shard_p.name})")
        print("=" * 60)

        preds, targets = evaluate_dataset(model, shard_p, vocab, order, device)
        cer, wer = corpus_cer_wer(preds, targets)
        buckets, total_w, correct_w, char_confs = analyze_words_and_confusions(preds, targets)

        exact_acc = (correct_w / total_w * 100) if total_w > 0 else 0.0

        print(f"  Character Error Rate (CER): {cer * 100:.2f}%")
        print(f"  Word Error Rate (WER):      {wer * 100:.2f}%")
        print(f"  Exact-Word Accuracy:        {exact_acc:.2f}% ({correct_w}/{total_w} words)")

        print("\n  [Accuracy by Word Length Buckets]")
        for b_name, b_stat in buckets.items():
            tot = b_stat["total"]
            corr = b_stat["correct"]
            acc = (corr / tot * 100) if tot > 0 else 0.0
            print(f"    {b_name:<10}: {acc:6.2f}% ({corr:5d} / {tot:5d} words)")

        print("\n  [Top 20 Character Confusions (target -> predicted)]")
        if not char_confs:
            print("    No substitutions observed!")
        else:
            for i, ((t_c, p_c), count) in enumerate(char_confs.most_common(20), 1):
                print(f"    {i:2d}. '{t_c}' -> '{p_c}': {count} occurrences")


if __name__ == "__main__":
    main()
