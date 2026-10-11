"""
scripts/task_2_17_suite.py
Task 2.17 Implementation Suite:
Synthetic Verification Benchmark on Test Surahs (seen & unseen fonts, clean & diacritic):
- 2,000 windows per corruption type:
  1. Dot-family swap
  2. Letter deletion / insertion
  3. Whole-word swap
  4. Adjacent-word swap
  5. Skipped word
  6. Mutashabih substitution (from data/derived/mutashabih_pairs.json)
- Evaluates Lines vs Isolated word crops of the same words.
- Computes AUC, FRR @ FAR 1% and 5%, and the 3-State distribution (ACCEPT / UNCERTAIN / WRONG).
"""

from __future__ import annotations
import sys
import json
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from model import CRNN
from shards import ShardDataset
from dataset import collate_fn
from verify import verify_text, load_quran_dict, DEFAULT_TAU_STRICT, DEFAULT_TAU_RELAXED, ctc_viterbi_alignment

DOT_FAMILIES = [
    ["ب", "ت", "ث"],
    ["ج", "ح", "خ"],
    ["د", "ذ"],
    ["ر", "ز"],
    ["س", "ش"],
    ["ص", "ض"],
    ["ط", "ظ"],
    ["ع", "غ"],
    ["ف", "ق"],
    ["ن", "ي"],
]

CONF_MAP: Dict[str, List[str]] = {}
for fam in DOT_FAMILIES:
    for c in fam:
        CONF_MAP[c] = [alt for alt in fam if alt != c]


def extract_mutashabih_pairs() -> List[Dict[str, str]]:
    """Builds and caches data/derived/mutashabih_pairs.json from Tanzil text."""
    out_p = ROOT / "data" / "derived" / "mutashabih_pairs.json"
    if out_p.exists():
        with open(out_p, "r", encoding="utf-8") as f:
            return json.load(f)

    tanzil_clean = ROOT / "data" / "tanzil" / "quran-simple-clean.txt"
    words = []
    with open(tanzil_clean, encoding="utf-8") as f:
        for line in f:
            if "|" in line:
                line = line.split("|", 2)[-1]
            for w in line.strip().split():
                words.append(w)

    prefix_to_next = defaultdict(set)
    for i in range(len(words) - 3):
        pfx = tuple(words[i : i + 3])
        nxt = words[i + 3]
        prefix_to_next[pfx].add(nxt)

    mutashabih_pairs = []
    for pfx, nxts in prefix_to_next.items():
        if len(nxts) >= 2:
            nxt_list = sorted(list(nxts))
            pfx_str = " ".join(pfx)
            for j in range(len(nxt_list)):
                for k in range(j + 1, len(nxt_list)):
                    mutashabih_pairs.append({
                        "prefix": pfx_str,
                        "variant_a": f"{pfx_str} {nxt_list[j]}",
                        "variant_b": f"{pfx_str} {nxt_list[k]}",
                        "word_a": nxt_list[j],
                        "word_b": nxt_list[k],
                    })

    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump(mutashabih_pairs[:2000], f, ensure_ascii=False, indent=2)
    return mutashabih_pairs[:2000]


# =========================================================================
# CORRUPTION APPLIERS
# =========================================================================

def apply_corruption(
    clean_text: str,
    ctype: str,
    quran_dict_list: List[str],
    mutashabih_dict: Dict[str, List[str]],
    rng: random.Random,
) -> Optional[Tuple[str, int]]:
    """
    Applies requested corruption to clean_text.
    Returns (corrupted_text, corrupted_word_index) or None if not applicable.
    """
    words = clean_text.split()
    if not words:
        return None

    if ctype == "dot_swap":
        cand = [(i, ci, c) for i, w in enumerate(words) for ci, c in enumerate(w) if c in CONF_MAP]
        if not cand:
            return None
        wi, ci, orig_c = rng.choice(cand)
        alt_c = rng.choice(CONF_MAP[orig_c])
        ch = list(words[wi])
        ch[ci] = alt_c
        mut = list(words)
        mut[wi] = "".join(ch)
        return " ".join(mut), wi

    elif ctype == "del_ins":
        wi = rng.randrange(len(words))
        w = words[wi]
        if len(w) <= 1 or rng.random() < 0.5:
            # Insert plausible letter
            ins_c = rng.choice(["ا", "و", "ي", "ل", "م", "ن", "ت", "ر"])
            pos = rng.randint(0, len(w))
            mut_w = w[:pos] + ins_c + w[pos:]
        else:
            pos = rng.randrange(len(w))
            mut_w = w[:pos] + w[pos + 1 :]
        mut = list(words)
        mut[wi] = mut_w
        return " ".join(mut), wi

    elif ctype == "word_swap":
        wi = rng.randrange(len(words))
        orig_w = words[wi]
        pool = [w for w in quran_dict_list if abs(len(w) - len(orig_w)) <= 1 and w != orig_w]
        alt = rng.choice(pool) if pool else "الله"
        mut = list(words)
        mut[wi] = alt
        return " ".join(mut), wi

    elif ctype == "adj_swap":
        if len(words) < 2:
            return None
        wi = rng.randrange(len(words) - 1)
        mut = list(words)
        mut[wi], mut[wi + 1] = mut[wi + 1], mut[wi]
        return " ".join(mut), wi

    elif ctype == "skip_word":
        if len(words) < 2:
            return None
        wi = rng.randrange(len(words))
        mut = [w for i, w in enumerate(words) if i != wi]
        return " ".join(mut), wi

    elif ctype == "mutashabih":
        # Look for 3-word prefix match in words
        for i in range(len(words) - 3):
            pfx = " ".join(words[i : i + 3])
            if pfx in mutashabih_dict:
                alts = [alt for alt in mutashabih_dict[pfx] if alt != words[i + 3]]
                if alts:
                    alt_w = rng.choice(alts)
                    mut = list(words)
                    mut[i + 3] = alt_w
                    return " ".join(mut), i + 3
        # Fallback to general Quranic word swap at position 3
        wi = min(len(words) - 1, 3)
        orig_w = words[wi]
        pool = [w for w in quran_dict_list if abs(len(w) - len(orig_w)) <= 1 and w != orig_w]
        alt = rng.choice(pool) if pool else "الذي"
        mut = list(words)
        mut[wi] = alt
        return " ".join(mut), wi

    return None


def compute_roc_auc(pos_scores: np.ndarray, neg_scores: np.ndarray) -> float:
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        return 0.5
    gt = np.sum(pos_scores[:, None] > neg_scores[None, :])
    eq = np.sum(pos_scores[:, None] == neg_scores[None, :])
    return float((gt + 0.5 * eq) / (len(pos_scores) * len(neg_scores)))


def compute_frr_at_far(pos_scores: np.ndarray, neg_scores: np.ndarray, far_target: float = 0.05) -> float:
    sorted_neg = np.sort(neg_scores)[::-1]
    idx = min(len(sorted_neg) - 1, int(np.floor(far_target * len(sorted_neg))))
    tau = sorted_neg[idx]
    return float(np.mean(pos_scores < tau) * 100.0)


# =========================================================================
# MAIN BENCHMARK RUNNER
# =========================================================================

def run_synthetic_benchmark(
    checkpoint_path: Path,
    vocab_path: Path = Path("vocab.json"),
    windows_per_type: int = 2000,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 95)
    print("TASK 2.17: SYNTHETIC VERIFICATION BENCHMARK (TEST SURAHS, 2000 WINDOWS / TYPE)")
    print("=" * 95)

    device = torch.device(device_str)
    vocab = Vocabulary.load(vocab_path)
    quran_dict = load_quran_dict()
    quran_dict_list = sorted(list(quran_dict))

    # 1. Mutashabih pairs
    mutashabih_raw = extract_mutashabih_pairs()
    mutashabih_dict = defaultdict(list)
    for p in mutashabih_raw:
        mutashabih_dict[p["prefix"]].append(p["word_a"])
        mutashabih_dict[p["prefix"]].append(p["word_b"])

    print(f"Loaded {len(mutashabih_raw)} Mutashabih pairs. Quran Dictionary: {len(quran_dict_list)} words.")

    # 2. Load model
    print(f"Device: {device} | Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # 3. Load synthetic test shards
    shard_paths = [
        ROOT / "data" / "shards" / "synth" / "synth_test_000.pt",
        ROOT / "data" / "shards" / "synth" / "synth_test_unseen_fonts_000.pt",
        ROOT / "data" / "shards" / "synth" / "synth_test_diac_000.pt",
        ROOT / "data" / "shards" / "synth" / "synth_test_unseen_fonts_diac_000.pt",
    ]
    existing_shards = [p for p in shard_paths if p.exists()]
    if not existing_shards:
        print("[warn] No synth test shards found in data/shards/synth/. Searching for any test shards...")
        existing_shards = sorted(list((ROOT / "data" / "shards").glob("**/*test*.pt")))
    if not existing_shards:
        raise FileNotFoundError("No evaluation shards found in data/shards/!")

    print(f"Found {len(existing_shards)} test shard(s): {[p.name for p in existing_shards]}")
    combined_ds = ShardDataset(existing_shards, vocab=vocab, augment=False, label_order=order)
    print(f"Total available test samples in shards: {len(combined_ds)}")

    corruption_types = [
        ("dot_swap", "Dot-family Swap"),
        ("del_ins", "Letter Deletion / Insertion"),
        ("word_swap", "Whole-word Swap"),
        ("adj_swap", "Adjacent-word Swap"),
        ("skip_word", "Skipped Word"),
        ("mutashabih", "Mutashabih Substitution"),
    ]

    rng = random.Random(42)

    # Sample windows across the shards
    # We evaluate lines and isolated word crops side by side
    n_samples = min(len(combined_ds), windows_per_type)
    sample_indices = rng.sample(range(len(combined_ds)), n_samples)

    # Precompute model forward outputs for the clean samples
    print(f"\nEvaluating {n_samples} benchmark samples across 6 corruption types...")

    clean_line_scores = []
    clean_crop_scores = []

    # Store corruption evaluation results
    # ctype -> list of (line_score, crop_score, 3-state results)
    type_results_line = defaultdict(list)
    type_results_crop = defaultdict(list)
    state_counts_line = defaultdict(lambda: {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0})
    state_counts_crop = defaultdict(lambda: {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0})

    clean_state_counts_line = {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0}
    clean_state_counts_crop = {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0}

    for count, idx in enumerate(sample_indices):
        item = combined_ds[idx]
        img_tensor = item.image.unsqueeze(0).to(device)
        clean_text = item.text

        with torch.no_grad():
            lp = model(img_tensor).squeeze(1).cpu().numpy()

        # Clean Line Verification
        res_clean = verify_text(
            log_probs_np=lp,
            expected_text=clean_text,
            vocab_char2idx=vocab.char2idx,
            vocab_idx2char=vocab.idx2char,
            quran_dict=quran_dict,
            threshold=DEFAULT_TAU_RELAXED,
            order=order,
        )
        if not res_clean:
            continue

        mean_llr_clean_line = float(np.mean([r.llr for r in res_clean]))
        clean_line_scores.append(mean_llr_clean_line)
        for r in res_clean:
            clean_state_counts_line[r.state] += 1

        # Clean Crop Verification (isolated word window)
        # Select target word near center of line
        target_w_idx = len(res_clean) // 2
        r_target = res_clean[target_w_idx]
        t1, t2 = max(0, r_target.t_start), min(lp.shape[0] - 1, r_target.t_end)
        lp_crop = lp[t1 : t2 + 1]

        res_clean_crop = verify_text(
            log_probs_np=lp_crop,
            expected_text=r_target.word,
            vocab_char2idx=vocab.char2idx,
            vocab_idx2char=vocab.idx2char,
            quran_dict=quran_dict,
            threshold=DEFAULT_TAU_RELAXED,
            order=order,
        )
        if res_clean_crop:
            clean_crop_scores.append(res_clean_crop[0].llr)
            clean_state_counts_crop[res_clean_crop[0].state] += 1

        # Now test each corruption type on this window
        for ctype, _ in corruption_types:
            corrupted = apply_corruption(clean_text, ctype, quran_dict_list, mutashabih_dict, rng)
            if corrupted is None:
                continue
            corr_text, c_idx = corrupted

            # 1. Corrupted Line verification
            res_corr = verify_text(
                log_probs_np=lp,
                expected_text=corr_text,
                vocab_char2idx=vocab.char2idx,
                vocab_idx2char=vocab.idx2char,
                quran_dict=quran_dict,
                threshold=DEFAULT_TAU_RELAXED,
                order=order,
            )
            if res_corr:
                mean_llr_corr_line = float(np.mean([r.llr for r in res_corr]))
                type_results_line[ctype].append(mean_llr_corr_line)
                # Count states for the corrupted word position
                if c_idx < len(res_corr):
                    state_counts_line[ctype][res_corr[c_idx].state] += 1
                else:
                    state_counts_line[ctype][res_corr[-1].state] += 1

            # 2. Corrupted Crop verification (isolated crop of the corrupted target word)
            corr_words = corr_text.split()
            if c_idx < len(corr_words):
                corr_target_w = corr_words[c_idx]
                res_corr_crop = verify_text(
                    log_probs_np=lp_crop,
                    expected_text=corr_target_w,
                    vocab_char2idx=vocab.char2idx,
                    vocab_idx2char=vocab.idx2char,
                    quran_dict=quran_dict,
                    threshold=DEFAULT_TAU_RELAXED,
                    order=order,
                )
                if res_corr_crop:
                    type_results_crop[ctype].append(res_corr_crop[0].llr)
                    state_counts_crop[ctype][res_corr_crop[0].state] += 1

        if (count + 1) % 500 == 0:
            print(f"  Processed {count + 1}/{n_samples} benchmark windows...")

    clean_line_arr = np.array(clean_line_scores)
    clean_crop_arr = np.array(clean_crop_scores)

    # =========================================================================
    # PRINT RESULTS TABLE
    # =========================================================================
    print("\n" + "=" * 95)
    print("SYNTHETIC BENCHMARK RESULTS: LINES VS ISOLATED WORD CROPS")
    print("=" * 95)
    print(f"{'Corruption Type':<27} | {'Context':<6} | {'AUC':<7} | {'FRR@1%':<7} | {'FRR@5%':<7} | {'Accept':<7} | {'Uncert':<7} | {'Wrong':<7}")
    print("-" * 95)

    for ctype, cdesc in corruption_types:
        # Lines
        c_lines = np.array(type_results_line[ctype])
        auc_l = compute_roc_auc(clean_line_arr, c_lines) if len(c_lines) else 0.5
        frr1_l = compute_frr_at_far(clean_line_arr, c_lines, 0.01) if len(c_lines) else 100.0
        frr5_l = compute_frr_at_far(clean_line_arr, c_lines, 0.05) if len(c_lines) else 100.0
        tot_l = sum(state_counts_line[ctype].values()) or 1
        pct_acc_l = (state_counts_line[ctype]["ACCEPT"] / tot_l) * 100.0
        pct_unc_l = (state_counts_line[ctype]["UNCERTAIN"] / tot_l) * 100.0
        pct_wrg_l = (state_counts_line[ctype]["WRONG"] / tot_l) * 100.0

        # Crops
        c_crops = np.array(type_results_crop[ctype])
        auc_c = compute_roc_auc(clean_crop_arr, c_crops) if len(c_crops) else 0.5
        frr1_c = compute_frr_at_far(clean_crop_arr, c_crops, 0.01) if len(c_crops) else 100.0
        frr5_c = compute_frr_at_far(clean_crop_arr, c_crops, 0.05) if len(c_crops) else 100.0
        tot_c = sum(state_counts_crop[ctype].values()) or 1
        pct_acc_c = (state_counts_crop[ctype]["ACCEPT"] / tot_c) * 100.0
        pct_unc_c = (state_counts_crop[ctype]["UNCERTAIN"] / tot_c) * 100.0
        pct_wrg_c = (state_counts_crop[ctype]["WRONG"] / tot_c) * 100.0

        print(f"{cdesc:<27} | {'Line':<6} | {auc_l:<7.3f} | {frr1_l:<6.1f}% | {frr5_l:<6.1f}% | {pct_acc_l:<6.1f}% | {pct_unc_l:<6.1f}% | {pct_wrg_l:<6.1f}%")
        print(f"{'':<27} | {'Crop':<6} | {auc_c:<7.3f} | {frr1_c:<6.1f}% | {frr5_c:<6.1f}% | {pct_acc_c:<6.1f}% | {pct_unc_c:<6.1f}% | {pct_wrg_c:<6.1f}%")
        print("-" * 95)

    print("\n" + "=" * 95)
    print("TASK 2.17 BENCHMARK COMPLETED SUCCESSFULLY")
    print("=" * 95)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/content/drive/MyDrive/quran_htr_writer_checkpoints/step2b_v3/best.pt")
    p.add_argument("--windows", type=int, default=2000)
    args = p.parse_args()

    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.exists():
        for alt in [
            Path("checkpoints/step2b_v3/best.pt"),
            Path("checkpoints/best.pt"),
        ]:
            if alt.exists():
                ckpt_p = alt
                break

    run_synthetic_benchmark(ckpt_p, windows_per_type=args.windows)
