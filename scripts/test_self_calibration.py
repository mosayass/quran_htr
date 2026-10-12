"""
scripts/test_self_calibration.py
Task 2.21 Item 3:
Self-calibration test: thresholds from the first 20 samples of the test writer,
no fine-tune, evaluate the rest (125 samples).
"""

from __future__ import annotations
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from model import CRNN
from render_strokes import render_strokes
from verify_ref import verify_reference
from task_2_17_suite import apply_corruption, extract_mutashabih_pairs


def main():
    vocab = Vocabulary.load(ROOT / "vocab.json")
    ckpt = torch.load(ROOT / "checkpoints" / "best.pt", map_location="cpu", weights_only=False)
    order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite")
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    with open(ROOT / "data" / "capture_v2" / "writer_02" / "strokes.json", encoding="utf-8") as f:
        raw = json.load(f)
    by_id = {x["id"]: x for x in raw if isinstance(x, dict) and "id" in x}
    samples = list(by_id.values())

    calib_20 = samples[:20]
    eval_rest = samples[20:]

    def prep(s):
        geom = s.get("geometry") or {}
        bt = float(geom.get("band_top_css", 0.0))
        bb = float(geom.get("band_base_css", 147.0))
        bot = float(geom.get("band_bot_css", 180.0))
        bh = float(geom.get("band_height_css", bot - bt))
        img = render_strokes(s["strokes"], zoom=1.0, width_after_zoom=2.2, center_y=(bb - bt), band_top_y=0.0, band_height=bh, pad_x=16.0)
        arr = (np.array(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
        pad_w = (-arr.shape[1]) % 32
        if pad_w > 0:
            arr = np.pad(arr, ((0, 0), (0, pad_w)), mode="constant", constant_values=1.0)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    with open(ROOT / "data" / "tanzil" / "quran-simple-clean.txt", encoding="utf-8") as f:
        words_pool = sorted([w for w in set(f.read().split()) if all(c in vocab.char2idx for c in w)])
    mut_pairs = extract_mutashabih_pairs()
    mut_dict = defaultdict(list)
    for p in mut_pairs:
        mut_dict[p["prefix"]].append(p["word_b"])
    rng = random.Random(42)
    ctypes = ["dot_swap", "del_ins", "word_swap", "adj_swap", "skip_word", "mutashabih"]

    true_llrs, corr_llrs = [], []
    for s in calib_20:
        t = prep(s)
        with torch.no_grad():
            lp = model(t).squeeze(1).numpy()
        gt = s["text"]
        v_true = verify_reference(lp, gt, vocab.char2idx, vocab.idx2char, t_accept=-0.6, t_wrong=-0.95, order=order, norm="char")
        for r in v_true:
            true_llrs.append(r["llr"])
        for ctype in ctypes:
            corr = apply_corruption(gt, ctype, words_pool, mut_dict, rng)
            if corr is None:
                continue
            ctext, cidx = corr
            v_corr = verify_reference(lp, ctext, vocab.char2idx, vocab.idx2char, t_accept=-0.6, t_wrong=-0.95, order=order, norm="char")
            if v_corr:
                target_r = v_corr[cidx] if cidx < len(v_corr) else v_corr[-1]
                corr_llrs.append(target_r["llr"])

    t_acc_self = float(np.percentile(true_llrs, 5.0))
    t_wrg_self = float(np.percentile(corr_llrs, 90.0))
    print(f"Self-Calibrated Thresholds (from first 20 samples of writer_02):")
    print(f"  * t_accept = {t_acc_self:.4f} (5th percentile of true claims)")
    print(f"  * t_wrong  = {t_wrg_self:.4f} (90th percentile of corrupted claims)")

    t_states = {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0}
    c_states = defaultdict(lambda: {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0})
    for s in eval_rest:
        t = prep(s)
        with torch.no_grad():
            lp = model(t).squeeze(1).numpy()
        gt = s["text"]
        v_true = verify_reference(lp, gt, vocab.char2idx, vocab.idx2char, t_accept=t_acc_self, t_wrong=t_wrg_self, order=order, norm="char")
        for r in v_true:
            t_states[r["state"]] += 1
        for ctype in ctypes:
            corr = apply_corruption(gt, ctype, words_pool, mut_dict, rng)
            if corr is None:
                continue
            ctext, cidx = corr
            v_corr = verify_reference(lp, ctext, vocab.char2idx, vocab.idx2char, t_accept=t_acc_self, t_wrong=t_wrg_self, order=order, norm="char")
            if v_corr:
                target_r = v_corr[cidx] if cidx < len(v_corr) else v_corr[-1]
                c_states[ctype][target_r["state"]] += 1

    tot_t = sum(t_states.values())
    acc_pct = t_states["ACCEPT"] / tot_t * 100
    unc_pct = t_states["UNCERTAIN"] / tot_t * 100
    wrg_pct = t_states["WRONG"] / tot_t * 100

    tot_c = sum(sum(c_states[c].values()) for c in ctypes)
    tot_c_wrg = sum(c_states[c]["WRONG"] for c in ctypes)
    c_wrg_pct = tot_c_wrg / tot_c * 100

    print(f"\nEvaluated on Remaining {len(eval_rest)} Samples (No Fine-Tuning):")
    print(f"  * TRUE Claims ({tot_t} words): ACCEPT {acc_pct:.1f}% | UNCERTAIN {unc_pct:.1f}% | WRONG {wrg_pct:.1f}%")
    print(f"  * CORRUPTED Claims ({tot_c} words): WRONG {c_wrg_pct:.1f}%")
    for ct in ctypes:
        sub_tot = sum(c_states[ct].values())
        print(f"    - {ct:<12}: WRONG {c_states[ct]['WRONG']/sub_tot*100:.1f}%")


if __name__ == "__main__":
    main()
