"""
scripts/audit_calibration_outliers.py
Task 2.31: Small-N Calibration Outlier Rule Audit across Folds C, C3, C4.
Evaluates deterministic calibration rules across saved mix_0.85 checkpoints (NO training).
"""

from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from model import CRNN
import train_twofold_lowo as T
from verify_ref import verify_reference
from task_2_17_suite import apply_corruption


def get_checkpoint_path(fold: str, seed: int, sweep27_dir: Path, sweep28_dir: Path) -> Path:
    """Exact known locations from Task 2.27 and Task 2.28 runs on Google Drive."""
    target_dir = sweep27_dir if fold == "C" else sweep28_dir
    p = target_dir / f"{fold}_mix_0.85_s{seed}.pt"
    if p.exists():
        return p
    # Fallback to local copy if running locally
    local_p = ROOT / "checkpoints" / target_dir.name / f"{fold}_mix_0.85_s{seed}.pt"
    if local_p.exists():
        return local_p
    raise FileNotFoundError(
        f"Checkpoint not found at exact path: {p} (or local fallback: {local_p}).\n"
        f"Please verify the exact checkpoint directory."
    )



def extract_calib_llrs(model: nn.Module, calib_samples: List[dict], vocab: Vocabulary,
                       device: str = "cpu") -> Tuple[List[float], List[float], List[Tuple[str, str, float]]]:
    q_dict = T._quran_dict(5000)
    mut_dict = T._mut_dict()
    true_llrs, corr_llrs = [], []
    detailed = []
    rng = random.Random(123)
    model.eval()
    with torch.no_grad():
        for s in calib_samples:
            inp = T.get_canonical_zoom_and_render(s, augment=False, device=device)
            out = model(inp).squeeze(1).cpu().numpy()
            txt = s["text"]
            for w in verify_reference(out, txt, vocab.char2idx, vocab.idx2char, t_accept=-0.6, t_wrong=-0.95, order="visual", norm="char"):
                true_llrs.append(w["llr"])
                detailed.append((s["id"], txt, round(w["llr"], 4)))
            c_info = apply_corruption(txt, "dot_swap", q_dict, mut_dict, rng)
            if c_info:
                for w in verify_reference(out, c_info[0], vocab.char2idx, vocab.idx2char, t_accept=-0.6, t_wrong=-0.95, order="visual", norm="char"):
                    corr_llrs.append(w["llr"])
    return true_llrs, corr_llrs, detailed


def evaluate_test_verification(model: nn.Module, test_samples: List[dict], vocab: Vocabulary,
                               t_acc: float, t_wrg: float, device: str = "cpu") -> dict:
    q_dict = T._quran_dict(14872)
    mut_dict = T._mut_dict()
    true_states = {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0}
    ctypes = ["dot_swap", "del_ins", "word_swap", "adj_swap", "skip_word", "mutashabih"]
    corr_wrg = defaultdict(lambda: {"WRONG": 0, "TOTAL": 0})
    rng = random.Random(42)

    model.eval()
    with torch.no_grad():
        for s in test_samples:
            inp = T.get_canonical_zoom_and_render(s, augment=False, device=device)
            out = model(inp).squeeze(1).cpu().numpy()
            txt = s["text"]
            for w in verify_reference(out, txt, vocab.char2idx, vocab.idx2char, t_accept=t_acc, t_wrong=t_wrg, order="visual", norm="char"):
                true_states[w["state"]] += 1
            for ct in ctypes:
                c_info = apply_corruption(txt, ct, q_dict, mut_dict, rng)
                if c_info:
                    cv = verify_reference(out, c_info[0], vocab.char2idx, vocab.idx2char, t_accept=t_acc, t_wrong=t_wrg, order="visual", norm="char")
                    corr_wrg[ct]["TOTAL"] += 1
                    if any(w["state"] == "WRONG" for w in cv):
                        corr_wrg[ct]["WRONG"] += 1

    tot = sum(true_states.values()) or 1
    return {
        "true_accept_rate": round(100.0 * true_states["ACCEPT"] / tot, 2),
        "true_unc_rate": round(100.0 * true_states["UNCERTAIN"] / tot, 2),
        "true_wrg_rate": round(100.0 * true_states["WRONG"] / tot, 2),
        "corr_rejection": {ct: (round(100.0 * corr_wrg[ct]["WRONG"] / corr_wrg[ct]["TOTAL"], 2)
                                if corr_wrg[ct]["TOTAL"] else None) for ct in ctypes},
    }


def compute_thresholds(true_llrs: List[float], corr_llrs: List[float], rule_name: str) -> Tuple[float, float]:
    """
    Candidate rules:
    - 'old_p5': np.percentile(true_llrs, 5) with min clamp t_acc - 0.20
    - 'trimmed_outlier_p5': drop min 1 outlier if len < 20, then p5
    - 'p15_unclamped': t_acc = percentile(true, 15), t_wrg = percentile(corr, 90) (no artificial clamp if monotonic)
    - 'trimmed_p10': drop min 1 outlier if len < 20, t_acc = percentile(true_clean, 10)
    - 'iqr_clipped': median - 1.0 * IQR
    """
    t_wrg_raw = float(np.percentile(corr_llrs, 90)) if corr_llrs else -0.95

    if rule_name == "old_p5":
        t_acc = float(np.percentile(true_llrs, 5))
        t_wrg = min(t_acc - 0.20, t_wrg_raw)
    elif rule_name == "trimmed_outlier_p5":
        clean = sorted(true_llrs)[1:] if len(true_llrs) < 20 else true_llrs
        t_acc = float(np.percentile(clean, 5))
        t_wrg = min(t_acc - 0.20, t_wrg_raw)
    elif rule_name == "trimmed_p10":
        clean = sorted(true_llrs)[1:] if len(true_llrs) < 20 else true_llrs
        t_acc = float(np.percentile(clean, 10))
        t_wrg = min(t_acc - 0.20, t_wrg_raw)
    elif rule_name == "p15_safe":
        # p15 targets rank ~2 for N=13-14, safe against single outlier; clamp maintains monotonicity
        t_acc = float(np.percentile(true_llrs, 15))
        t_wrg = min(t_acc - 0.20, t_wrg_raw)
    elif rule_name == "iqr_clipped":
        med = float(np.median(true_llrs))
        iqr = float(np.percentile(true_llrs, 75) - np.percentile(true_llrs, 25))
        t_acc = med - 1.0 * iqr
        t_wrg = min(t_acc - 0.20, t_wrg_raw)
    else:
        raise ValueError(f"Unknown rule: {rule_name}")

    return round(t_acc, 4), round(t_wrg, 4)


def main():
    ap = argparse.ArgumentParser(description="Audit outlier-safe calibration rules across Folds C, C3, C4")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--sweep27_dir", default="/content/drive/MyDrive/quran_htr_writer_checkpoints/sweep27",
                    help="Directory containing C_mix_0.85_s{0,1}.pt")
    ap.add_argument("--sweep28_dir", default="/content/drive/MyDrive/quran_htr_writer_checkpoints/sweep28",
                    help="Directory containing C3_mix_0.85_s*.pt and C4_mix_0.85_s*.pt")
    args = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Device: {device}")

    vocab = Vocabulary.load(ROOT / "vocab.json")
    sp = T.load_dev_splits()

    sweep27_dir = Path(args.sweep27_dir) if Path(args.sweep27_dir).is_absolute() else ROOT / args.sweep27_dir
    sweep28_dir = Path(args.sweep28_dir) if Path(args.sweep28_dir).is_absolute() else ROOT / args.sweep28_dir

    folds_data = [
        ("C", sp["fc_cal"], sp["fc_te"], "writer_02"),
        ("C3", sp["w3_fc_cal"], sp["w3_fc_te"], "writer_03"),
        ("C4", sp["w4_fc_cal"], sp["w4_fc_te"], "writer_04"),
    ]

    rules_to_test = ["old_p5", "trimmed_outlier_p5", "trimmed_p10", "p15_safe"]

    print("\n" + "=" * 80)
    print("STEP 1: SORTED CALIBRATION LLR SCORES ACROSS FOLDS & SEEDS")
    print("=" * 80)

    calib_scores_summary = {}

    for fold, cal_samples, te_samples, wname in folds_data:
        calib_scores_summary[fold] = {}
        for seed in (0, 1):
            try:
                ckpt_p = get_checkpoint_path(fold, seed, sweep27_dir, sweep28_dir)
            except FileNotFoundError as e:
                print(f"[Warning] {e}")
                continue

            ckpt = torch.load(ckpt_p, map_location=device)
            state = ckpt["model_state"] if "model_state" in ckpt else ckpt
            model = CRNN(num_chars=len(vocab)).to(device)
            model.load_state_dict(state)

            true_llrs, corr_llrs, detailed = extract_calib_llrs(model, cal_samples, vocab, device=device)
            sorted_detailed = sorted(detailed, key=lambda x: x[2])
            calib_scores_summary[fold][seed] = {
                "true_llrs": true_llrs,
                "corr_llrs": corr_llrs,
                "sorted": sorted_detailed,
                "ckpt": ckpt_p,
            }

            print(f"\n[{fold} / {wname} | seed={seed} | n_calib={len(cal_samples)}] Checkpoint: {ckpt_p.name}")
            print(f"  Sorted True LLRs ({len(true_llrs)} words):")
            for rank, (sid, txt, val) in enumerate(sorted_detailed):
                print(f"    rank {rank:2d}: LLR = {val:7.4f} | {sid:30} | '{txt}'")
            print(f"  Summary stats: min={sorted_detailed[0][2]:.4f}, p5={np.percentile(true_llrs, 5):.4f}, "
                  f"p15={np.percentile(true_llrs, 15):.4f}, median={np.median(true_llrs):.4f}, max={sorted_detailed[-1][2]:.4f}")
            print(f"  Corr 90th percentile: {np.percentile(corr_llrs, 90):.4f}")

    print("\n" + "=" * 80)
    print("STEP 2: HELD-OUT TEST VERIFICATION METRICS UNDER CANDIDATE RULES")
    print("=" * 80)

    results_table = defaultdict(lambda: defaultdict(dict))

    for rule in rules_to_test:
        print(f"\n>>> TESTING RULE: {rule} <<<")
        for fold, cal_samples, te_samples, wname in folds_data:
            seed_metrics = []
            for seed in (0, 1):
                if fold not in calib_scores_summary or seed not in calib_scores_summary[fold]:
                    continue
                info = calib_scores_summary[fold][seed]
                true_llrs = info["true_llrs"]
                corr_llrs = info["corr_llrs"]
                t_acc, t_wrg = compute_thresholds(true_llrs, corr_llrs, rule)

                # Re-load model for test eval
                ckpt = torch.load(info["ckpt"], map_location=device)
                state = ckpt["model_state"] if "model_state" in ckpt else ckpt
                model = CRNN(num_chars=len(vocab)).to(device)
                model.load_state_dict(state)

                res = evaluate_test_verification(model, te_samples, vocab, t_acc, t_wrg, device=device)
                res["thresholds"] = [t_acc, t_wrg]
                seed_metrics.append(res)

            if len(seed_metrics) == 2:
                m0, m1 = seed_metrics[0], seed_metrics[1]
                t_acc_mean = round((m0["thresholds"][0] + m1["thresholds"][0]) / 2, 4)
                t_wrg_mean = round((m0["thresholds"][1] + m1["thresholds"][1]) / 2, 4)
                acc_mean = round((m0["true_accept_rate"] + m1["true_accept_rate"]) / 2, 2)
                unc_mean = round((m0["true_unc_rate"] + m1["true_unc_rate"]) / 2, 2)
                wrg_mean = round((m0["true_wrg_rate"] + m1["true_wrg_rate"]) / 2, 2)
                dot_mean = round((m0["corr_rejection"]["dot_swap"] + m1["corr_rejection"]["dot_swap"]) / 2, 2)
                del_mean = round((m0["corr_rejection"]["del_ins"] + m1["corr_rejection"]["del_ins"]) / 2, 2)

                results_table[rule][fold] = {
                    "t_acc": t_acc_mean, "t_wrg": t_wrg_mean,
                    "accept": acc_mean, "uncertain": unc_mean, "wrong": wrg_mean,
                    "dot_swap": dot_mean, "del_ins": del_mean,
                }
                print(f"  {fold:4} ({wname:10}): Thresh=[{t_acc_mean:7.4f}, {t_wrg_mean:7.4f}] | "
                      f"Acc={acc_mean:5.2f}% Unc={unc_mean:5.2f}% Wrg={wrg_mean:5.2f}% | "
                      f"DotRej={dot_mean:5.2f}% DelRej={del_mean:5.2f}%")

    out_p = ROOT / "reports" / "task31_calibration_audit.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(results_table, f, indent=2)
    print(f"\nSaved audit report -> {out_p}")


if __name__ == "__main__":
    main()
