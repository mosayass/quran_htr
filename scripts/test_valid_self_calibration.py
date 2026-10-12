"""
scripts/test_valid_self_calibration.py
Task 2.22 Item C.2:
Valid self-calibration test:
- Invariant: t_wrong <= t_accept strictly enforced
- 20 random draws at N = 10, 20, 40 samples from writer_02
- Evaluates held-out (145 - N) test samples
- Precomputes sample LLRs once for instant execution
- Reports per-type table with spread (mean +/- std)
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
            arr = np.pad(arr, ((0, 0), (0, pad_w)), constant_values=1.0)
        return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)

    with open(ROOT / "data" / "tanzil" / "quran-simple-clean.txt", encoding="utf-8") as f:
        words_pool = [w for w in f.read().split() if not w.startswith("#") and all(c in vocab.char2idx for c in w)]
    mut_pairs = extract_mutashabih_pairs()
    mut_dict = defaultdict(list)
    for p in mut_pairs:
        mut_dict[p["prefix"]].append(p["word_b"])

    rng_init = random.Random(42)
    ctypes = ["dot_swap", "del_ins", "word_swap", "adj_swap", "skip_word", "mutashabih"]

    print(f"Precomputing LLRs across {len(samples)} samples (true + 6 corruptions)...", flush=True)
    sample_data = []
    for s in samples:
        t = prep(s)
        with torch.no_grad():
            lp = model(t).squeeze(1).numpy()
        gt = s["text"]

        # True claim word LLRs
        v_true = verify_reference(lp, gt, vocab.char2idx, vocab.idx2char, t_accept=-0.6, t_wrong=-0.95, order=order, norm="char")
        t_llrs = [r["llr"] for r in v_true]

        # Corrupted claims LLRs
        c_llrs = {}
        for ct in ctypes:
            corr = apply_corruption(gt, ct, words_pool, mut_dict, rng_init)
            if corr is not None:
                ctext, cidx = corr
                v_corr = verify_reference(lp, ctext, vocab.char2idx, vocab.idx2char, t_accept=-0.6, t_wrong=-0.95, order=order, norm="char")
                if v_corr:
                    target_r = v_corr[cidx] if cidx < len(v_corr) else v_corr[-1]
                    c_llrs[ct] = target_r["llr"]
        sample_data.append({"true_llrs": t_llrs, "corr_llrs": c_llrs})

    print("Precomputation complete. Running 20 draws for N in [10, 20, 40]...", flush=True)

    summary_tables = {}
    for N in [10, 20, 40]:
        rng = random.Random(100 + N)
        t_acc_list = []
        t_wrg_list = []
        true_acc_list = []
        corr_wrg_by_type = defaultdict(list)

        for draw in range(20):
            cal_idx = set(rng.sample(range(len(samples)), N))
            eval_idx = [i for i in range(len(samples)) if i not in cal_idx]

            # Collect calibration LLRs
            cal_t_llrs = [llr for i in cal_idx for llr in sample_data[i]["true_llrs"]]
            cal_c_llrs = [llr for i in cal_idx for llr in sample_data[i]["corr_llrs"].values()]

            # VALID Calibration:
            # t_accept = 10th percentile of true claims (90% target true acceptance)
            t_acc = float(np.percentile(cal_t_llrs, 10.0))
            # Cand t_wrong from 90th percentile of corrupted claims
            cand_wrg = float(np.percentile(cal_c_llrs, 90.0))
            # STRICT INVARIANT: t_wrong <= t_accept (separated by delta margin 0.20)
            t_wrg = min(cand_wrg, t_acc - 0.20)

            t_acc_list.append(t_acc)
            t_wrg_list.append(t_wrg)

            # Evaluate on held-out (145 - N) test samples
            n_true_acc, n_true_tot = 0, 0
            n_corr_wrg = defaultdict(int)
            n_corr_tot = defaultdict(int)

            for i in eval_idx:
                for llr in sample_data[i]["true_llrs"]:
                    n_true_tot += 1
                    if llr >= t_acc:
                        n_true_acc += 1
                for ct, llr in sample_data[i]["corr_llrs"].items():
                    n_corr_tot[ct] += 1
                    if llr <= t_wrg:
                        n_corr_wrg[ct] += 1

            true_acc_list.append(n_true_acc / max(1, n_true_tot) * 100.0)
            for ct in ctypes:
                if n_corr_tot[ct] > 0:
                    corr_wrg_by_type[ct].append(n_corr_wrg[ct] / n_corr_tot[ct] * 100.0)

        summary_tables[N] = {
            "t_acc_mean": float(np.mean(t_acc_list)),
            "t_acc_std": float(np.std(t_acc_list)),
            "t_wrg_mean": float(np.mean(t_wrg_list)),
            "t_wrg_std": float(np.std(t_wrg_list)),
            "true_acc_mean": float(np.mean(true_acc_list)),
            "true_acc_std": float(np.std(true_acc_list)),
            "corr_wrg": {ct: (float(np.mean(corr_wrg_by_type[ct])), float(np.std(corr_wrg_by_type[ct]))) for ct in ctypes},
        }

        print(f"\n=======================================================")
        print(f"VALID SELF-CALIBRATION: N = {N} SAMPLES (20 DRAWS)")
        print(f"=======================================================")
        print(f"t_accept: {np.mean(t_acc_list):.3f} +/- {np.std(t_acc_list):.3f}")
        print(f"t_wrong:  {np.mean(t_wrg_list):.3f} +/- {np.std(t_wrg_list):.3f}  [STRICTLY <= t_accept]")
        print(f"Held-Out True Accept: {np.mean(true_acc_list):.1f}% +/- {np.std(true_acc_list):.1f}%")
        print("Held-Out Corrupted WRONG by Type (Mean +/- Std):")
        for ct in ctypes:
            m, s = summary_tables[N]["corr_wrg"][ct]
            print(f"  - {ct:<12}: {m:5.1f}% +/- {s:4.1f}%")

    with open(ROOT / "reports" / "valid_self_calibration.json", "w", encoding="utf-8") as f:
        json.dump(summary_tables, f, indent=2)
    print("\nSaved reports/valid_self_calibration.json successfully.")


if __name__ == "__main__":
    main()
