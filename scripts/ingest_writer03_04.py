"""
scripts/ingest_writer03_04.py
Task 2.28: Ingest writer_03 and writer_04, generate Fold C splits,
and evaluate zero-shot base model on CPU.
"""

import json
import re
from collections import defaultdict
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from metrics import corpus_cer_wer
from render_strokes import render_strokes
from scale_policy import sample_band
from vocab import Vocabulary
import train_twofold_lowo as T
prompts_p = ROOT / "data" / "capture_prompts.json"
prompts_meta = json.load(open(prompts_p, encoding="utf-8")) if prompts_p.exists() else {}
w_to_line = {w["rasm"]: w["parent_line_id"] for w in prompts_meta.get("words", []) if "parent_line_id" in w}


def process_writer(wname: str):
    dir_p = ROOT / "data" / "capture_v2" / wname
    strokes_p = dir_p / "strokes.json"
    data = json.load(open(strokes_p, encoding="utf-8"))
    by_id = {x["id"]: x for x in data}
    groups = defaultdict(list)
    for sid, s in by_id.items():
        m_line = re.search(r"line_(\d+)", sid)
        if "line" in sid and m_line:
            groups[f"group_line_{int(m_line.group(1)):02d}"].append(sid)
        else:
            p_line = w_to_line.get(s["text"])
            if p_line:
                m_parent = re.search(r"(\d+)", p_line)
                p_num = int(m_parent.group(1)) if m_parent else 0
                groups[f"group_line_{p_num:02d}"].append(sid)
            else:
                groups["group_extra"].append(sid)

    line_gids = sorted([g for g in groups if g != "group_extra"])
    extra_sids = sorted(groups["group_extra"])
    n_extra_half = len(extra_sids) // 2

    tr_sids = [sid for g in line_gids[:10] for sid in groups[g]] + extra_sids[:n_extra_half]
    te_sids = [sid for g in line_gids[10:] for sid in groups[g]] + extra_sids[n_extra_half:]
    cal_c = max(5, int(0.2 * len(tr_sids)))
    pure_tr_sids = tr_sids[:-cal_c]
    cal_sids = tr_sids[-cal_c:]

    assert len(tr_sids) + len(te_sids) == len(by_id), "Sample count mismatch!"
    assert not (set(tr_sids) & set(te_sids)), "Train/test leakage!"

    with open(dir_p / "fold_c_train.json", "w", encoding="utf-8") as f:
        json.dump([by_id[s] for s in tr_sids], f, ensure_ascii=False, indent=2)
    with open(dir_p / "fold_c_test.json", "w", encoding="utf-8") as f:
        json.dump([by_id[s] for s in te_sids], f, ensure_ascii=False, indent=2)

    return {
        "total": len(by_id),
        "train_pool": len(tr_sids),
        "pure_train": len(pure_tr_sids),
        "calib": len(cal_sids),
        "held_out_test": len(te_sids),
        "all_samples": list(by_id.values()),
        "test_samples": [by_id[s] for s in te_sids],
        "train_samples": [by_id[s] for s in tr_sids],
        "pure_train_samples": [by_id[s] for s in pure_tr_sids],
    }


def eval_cer_wer(model, samples, vocab, device="cpu"):
    recs = T.predict_records(model, samples, vocab, device)
    preds = [r["pred"] for r in recs]
    targets = [r["text"] for r in recs]
    cer_v, wer_v = corpus_cer_wer(preds, targets)
    return round(cer_v * 100.0, 2), round(wer_v * 100.0, 2)


def main():
    import argparse
    import torch
    ap = argparse.ArgumentParser(description="Ingest and evaluate zero-shot base model on writer_03 and writer_04")
    ap.add_argument("--device", default="auto", help="auto | cuda | cpu")
    ap.add_argument("--checkpoint", default=None, help="Path to base checkpoint")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Device: {device}")

    vocab = Vocabulary.load(ROOT / "vocab.json")
    ckpt_p = T.find_base_checkpoint(args.checkpoint)
    assert ckpt_p is not None, "Base checkpoint not found!"
    print(f"Loading base checkpoint: {ckpt_p}")
    model, msg = T.load_base_model(ckpt_p, vocab, device)
    print(f"Base model loaded: {msg}")

    results = {}
    for wname in ("writer_03", "writer_04"):
        info = process_writer(wname)
        cer_all, wer_all = eval_cer_wer(model, info["all_samples"], vocab, device)
        cer_test, wer_test = eval_cer_wer(model, info["test_samples"], vocab, device)
        results[wname] = {
            "counts": {
                "total": info["total"],
                "train_pool": info["train_pool"],
                "pure_train": info["pure_train"],
                "calib": info["calib"],
                "held_out_test": info["held_out_test"],
            },
            f"zero_shot_base_{device}": {
                "full_cer": cer_all,
                "full_wer": wer_all,
                "test_cer": cer_test,
                "test_wer": wer_test,
            },
        }
        print(f"\n[{wname}] (Total: {info['total']}, Train: {info['pure_train']} tr + {info['calib']} cal, Test: {info['held_out_test']}):")
        print(f"  Full Set: CER={cer_all}% | WER={wer_all}%")
        print(f"  Held-out Test: CER={cer_test}% | WER={wer_test}%")

    out_p = ROOT / "reports" / "task28_ingest_and_zeroshot.json"
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved report -> {out_p}")


if __name__ == "__main__":
    main()

