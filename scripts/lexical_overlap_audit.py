"""
scripts/lexical_overlap_audit.py
Task 2.25 item 3: lexical-overlap audit (dev writers only).

(a) Share of TEST-prompt words (tokens and unique types) that also occur in the TRAIN prompts, per fold /
    test writer, plus the share of test samples whose words are ALL seen.
(b) Word-attributed CER stratified by seen vs unseen words for the UNTRAINED base checkpoint
    (fine-tuned stratification is logged by train_twofold_lowo.py into every run record).
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from train_twofold_lowo import (find_base_checkpoint, generate_all_group_splits, load_base_model, load_dev_splits,
                                predict_records, stratify_cer, words_of, assert_no_locked_test)


def overlap(train, test) -> dict:
    seen = words_of(train)
    tok = [w for s in test for w in s["text"].split()]
    types = {w for w in tok}
    all_seen = sum(all(w in seen for w in s["text"].split()) for s in test)
    return {
        "n_train": len(train), "n_test": len(test),
        "test_tokens": len(tok), "tokens_seen": sum(w in seen for w in tok),
        "token_share_seen": round(100.0 * sum(w in seen for w in tok) / max(1, len(tok)), 2),
        "test_types": len(types), "types_seen": sum(w in seen for w in types),
        "type_share_seen": round(100.0 * sum(w in seen for w in types) / max(1, len(types)), 2),
        "samples_fully_seen_pct": round(100.0 * all_seen / max(1, len(test)), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/lexical_overlap_audit.json")
    args = ap.parse_args()
    assert_no_locked_test(args.out)

    generate_all_group_splits()
    vocab = Vocabulary.load(ROOT / "vocab.json")
    model, _ = load_base_model(find_base_checkpoint(args.base_checkpoint), vocab, args.device)
    sp = load_dev_splits()
    folds = {
        "A (train w01 -> test w02)": (sp["w1_tr"], sp["w2_all"], "writer_02"),
        "B (train w02 -> test w01)": (sp["w2_tr"], sp["w1_all"], "writer_01"),
        "C (w02 50% groups -> w02 other 50%)": (sp["fc_pure_tr"], sp["fc_te"], "writer_02"),
        "CLC (w02 14-group pool -> w02 held-out groups)": (sp["w2_tr"] + sp["w2_cal"], sp["w2_held"], "writer_02"),
    }
    out = {}
    for name, (tr, te, writer) in folds.items():
        ov = overlap(tr, te)
        recs = predict_records(model, te, vocab, args.device)
        strat = stratify_cer(recs, words_of(tr))
        out[name] = {"test_writer": writer, "overlap": ov, "base_model_cer_by_lexicon": strat}
        print(f"\n{name}  [test writer {writer}]")
        print(f"  test tokens seen in train: {ov['tokens_seen']}/{ov['test_tokens']} ({ov['token_share_seen']}%) | "
              f"types {ov['types_seen']}/{ov['test_types']} ({ov['type_share_seen']}%) | fully-seen samples {ov['samples_fully_seen_pct']}%")
        for k in ("seen", "unseen"):
            v = strat[k]
            print(f"  base-model CER on {k:6s} words: {v['cer'] if v['cer'] is None else round(v['cer'], 2)}%  ({v['words']} words, {v['chars']} chars)")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
