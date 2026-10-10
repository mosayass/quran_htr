"""
scripts/task_2_16_suite.py
Task 2.16 Implementation Suite:
Verification v2 (label all results DEV):
1. CTC forced alignment to obtain per-word frame spans [t_start, t_end].
2. Per-word log-likelihood ratio (LLR):
   - Expected-text score minus best free-path score over the same frames.
   - Also vs best confusable alternative over the same frames.
3. 4 Corruption types evaluated separately:
   (a) Dot-family letter swap (ب/ت/ث, ج/ح/خ, د/ذ, ر/ز, س/ش, ص/ض, ط/ظ, ع/غ, ف/ق)
   (b) Letter deletion / insertion
   (c) Whole-word swap (Quran dictionary word of similar length)
   (d) Missing / extra word
4. Evaluates both Lines (40 samples) and Word crops (60 samples).
5. Reports ROC AUC per type, and FRR at FAR 1% and FAR 5%.
All results strictly labeled DEV.
"""

from __future__ import annotations
import sys
import json
import csv
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from model import CRNN
from render_strokes import render_strokes

# Dot-family letter equivalence groups
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

CONFUSABLE_MAP: Dict[str, List[str]] = {}
for fam in DOT_FAMILIES:
    for c in fam:
        CONFUSABLE_MAP[c] = [alt for alt in fam if alt != c]


# =========================================================================
# 1. CTC FORCED ALIGNMENT & SCORING UTILITIES
# =========================================================================

def ctc_viterbi_alignment(
    log_probs: torch.Tensor,  # (T, V)
    target_indices: List[int],
    blank_idx: int = 0,
) -> Tuple[List[int], float]:
    """
    Computes Viterbi forced alignment path and total score for target sequence under CTC.
    Returns (state_path_per_frame, total_path_score).
    State path length is T, with states in 0 .. 2*U.
    State 2*k + 1 corresponds to target token k.
    """
    T, V = log_probs.shape
    U = len(target_indices)
    if U == 0:
        # Sequence of blanks
        path = [0] * T
        score = log_probs[:, blank_idx].sum().item()
        return path, score

    K = 2 * U + 1
    # Extended target sequence: blank, y0, blank, y1, ..., blank
    ext_target = [blank_idx] * K
    for i, tok in enumerate(target_indices):
        ext_target[2 * i + 1] = tok

    # Viterbi DP trellis: shape (T, K)
    neg_inf = -1e9
    trellis = np.full((T, K), neg_inf, dtype=np.float32)
    backpointers = np.zeros((T, K), dtype=np.int32)

    lp = log_probs.cpu().numpy()

    # t = 0 initialization
    trellis[0, 0] = lp[0, ext_target[0]]
    trellis[0, 1] = lp[0, ext_target[1]]

    # Forward DP
    for t in range(1, T):
        for k in range(K):
            tok = ext_target[k]
            emit_lp = lp[t, tok]

            # Predecessors: self-loop k, or from k-1
            best_prev = trellis[t - 1, k]
            best_p_idx = k

            if k > 0 and trellis[t - 1, k - 1] > best_prev:
                best_prev = trellis[t - 1, k - 1]
                best_p_idx = k - 1

            # Skip blank: from k-2 if k >= 2 and non-blank and different char
            if k >= 2 and (k % 2 == 1) and (ext_target[k] != ext_target[k - 2]):
                if trellis[t - 1, k - 2] > best_prev:
                    best_prev = trellis[t - 1, k - 2]
                    best_p_idx = k - 2

            trellis[t, k] = best_prev + emit_lp
            backpointers[t, k] = best_p_idx

    # Termination: choose best of final two states (K-2 or K-1)
    cand_end = [K - 2, K - 1] if K >= 2 else [0]
    best_final_k = cand_end[0]
    best_final_score = trellis[T - 1, cand_end[0]]
    if len(cand_end) > 1 and trellis[T - 1, cand_end[1]] > best_final_score:
        best_final_k = cand_end[1]
        best_final_score = trellis[T - 1, cand_end[1]]

    # Backtracking
    path = [0] * T
    curr = best_final_k
    for t in range(T - 1, -1, -1):
        path[t] = int(curr)
        curr = backpointers[t, curr]

    return path, float(best_final_score)


def score_word_ctc_segment(
    log_probs_segment: torch.Tensor,  # (T_seg, V)
    word_indices: List[int],
    blank_idx: int = 0,
) -> float:
    """Computes CTC Viterbi score of a single word over an assigned frame segment."""
    if log_probs_segment.shape[0] == 0:
        return -1e9
    _, score = ctc_viterbi_alignment(log_probs_segment, word_indices, blank_idx)
    return score


def extract_word_frame_spans(
    path: List[int],
    text: str,
    vocab: Vocabulary,
    order: str = "visual",
    blank_idx: int = 0,
) -> List[Tuple[str, int, int, List[int]]]:
    """
    Given alignment path and text transcription, maps each word to its active frame span [t_start, t_end].
    Returns list of (word, t_start, t_end, word_token_indices).
    """
    T = len(path)
    words = text.split()
    if not words:
        return []

    # Map characters in encoded string to their corresponding words
    if order == "visual":
        # In visual order, words appear right-to-left (reversed)
        vis_text = text[::-1]
        vis_words = [w[::-1] for w in text.split()[::-1]]
        encoded = vocab.encode(vis_text)
    else:
        vis_text = text
        vis_words = words
        encoded = vocab.encode(vis_text)

    # Find character index ranges for each word in encoded sequence
    # Tokens are chars; space ' ' is index for space
    word_info = []
    char_pos = 0
    for w in vis_words:
        w_enc = vocab.encode(w)
        w_len = len(w_enc)
        word_info.append((w, char_pos, char_pos + w_len, w_enc))
        char_pos += w_len + 1  # +1 for space

    # Map each frame to character index (state 2*k + 1 -> character k)
    spans = []
    for w, c_start, c_end, w_enc in word_info:
        # States for this word are 2*c_start .. 2*(c_end-1)+2
        active_states = set(range(2 * c_start, 2 * c_end + 1))
        frames = [t for t in range(T) if path[t] in active_states]
        if frames:
            t_min, t_max = min(frames), max(frames)
        else:
            t_min, t_max = 0, T - 1
        # Recover original logical word string
        orig_w = w[::-1] if order == "visual" else w
        spans.append((orig_w, t_min, t_max, w_enc))

    return spans


# =========================================================================
# 2. CORRUPTION GENERATORS
# =========================================================================

def mutate_dot_swap(text: str, rng: random.Random) -> Optional[str]:
    """Corruption Type A: Dot-family letter swap (e.g. ب <-> ت, ج <-> ح)."""
    words = text.split()
    if not words:
        return None
    candidates = []
    for wi, w in enumerate(words):
        for ci, c in enumerate(w):
            if c in CONFUSABLE_MAP and CONFUSABLE_MAP[c]:
                candidates.append((wi, ci, c))
    if not candidates:
        return None
    wi, ci, orig_c = rng.choice(candidates)
    alt_c = rng.choice(CONFUSABLE_MAP[orig_c])
    chars = list(words[wi])
    chars[ci] = alt_c
    mutated_words = list(words)
    mutated_words[wi] = "".join(chars)
    return " ".join(mutated_words)


def mutate_delete_insert(text: str, vocab: Vocabulary, rng: random.Random) -> Optional[str]:
    """Corruption Type B: Letter deletion or insertion."""
    words = text.split()
    if not words:
        return None
    wi = rng.randrange(len(words))
    w = words[wi]
    if len(w) <= 1:
        mode = "insert"
    else:
        mode = rng.choice(["delete", "insert"])

    if mode == "delete":
        # Delete internal or edge letter
        del_pos = rng.randrange(len(w))
        mutated_w = w[:del_pos] + w[del_pos + 1:]
    else:
        # Insert a plausible Arabic letter
        ins_candidates = ["ا", "و", "ي", "ل", "م", "ن", "ت", "ر"]
        ins_c = rng.choice(ins_candidates)
        ins_pos = rng.randint(0, len(w))
        mutated_w = w[:ins_pos] + ins_c + w[ins_pos:]

    mutated_words = list(words)
    mutated_words[wi] = mutated_w
    return " ".join(mutated_words)


def mutate_word_swap(text: str, quran_dict: List[str], rng: random.Random) -> Optional[str]:
    """Corruption Type C: Whole-word swap for a Quran word of similar length."""
    words = text.split()
    if not words:
        return None
    wi = rng.randrange(len(words))
    orig_w = words[wi]
    pool = [w for w in quran_dict if abs(len(w) - len(orig_w)) <= 1 and w != orig_w]
    if not pool:
        pool = ["الله", "الذي", "قال", "كان"]
    mutated_w = rng.choice(pool)
    mutated_words = list(words)
    mutated_words[wi] = mutated_w
    return " ".join(mutated_words)


def mutate_missing_extra(text: str, quran_dict: List[str], rng: random.Random) -> Optional[str]:
    """Corruption Type D: Missing or extra word."""
    words = text.split()
    if len(words) <= 1:
        # Single word: append an extra word
        extra_w = rng.choice(quran_dict[:500])
        return f"{text} {extra_w}"
    mode = rng.choice(["missing", "extra"])
    if mode == "missing":
        wi = rng.randrange(len(words))
        mutated_words = [w for i, w in enumerate(words) if i != wi]
        return " ".join(mutated_words)
    else:
        extra_w = rng.choice(quran_dict[:500])
        ins_pos = rng.randint(0, len(words))
        mutated_words = list(words)
        mutated_words.insert(ins_pos, extra_w)
        return " ".join(mutated_words)


# =========================================================================
# 3. VERIFICATION METRICS (AUC, FAR, FRR)
# =========================================================================

def compute_roc_auc(true_scores: np.ndarray, corr_scores: np.ndarray) -> float:
    """Computes exact ROC AUC using Mann-Whitney U rank statistic."""
    n_true = len(true_scores)
    n_corr = len(corr_scores)
    if n_true == 0 or n_corr == 0:
        return 0.5
    # Count pairs where true > corr
    gt = np.sum(true_scores[:, None] > corr_scores[None, :])
    eq = np.sum(true_scores[:, None] == corr_scores[None, :])
    return float((gt + 0.5 * eq) / (n_true * n_corr))


def compute_frr_at_far(
    true_scores: np.ndarray,
    corr_scores: np.ndarray,
    target_far: float = 0.05,
) -> float:
    """
    Computes FRR (%) at a specific FAR operating threshold.
    FAR = fraction of corrupted samples >= tau.
    FRR = fraction of true samples < tau.
    """
    sorted_corr = np.sort(corr_scores)[::-1]  # descending
    n_corr = len(sorted_corr)
    # Target FAR index
    idx = int(np.floor(target_far * n_corr))
    idx = min(idx, n_corr - 1)
    tau = sorted_corr[idx]
    frr = np.mean(true_scores < tau) * 100.0
    return float(frr)


# =========================================================================
# 4. MAIN TASK 2.16 VERIFICATION SUITE
# =========================================================================

def run_task_2_16_verification(
    checkpoint_path: Path,
    vocab_path: Path = Path("vocab.json"),
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print("=" * 90)
    print("TASK 2.16: VERIFICATION V2 (DEV EVALUATION)")
    print("=" * 90)

    device = torch.device(device_str)
    vocab = Vocabulary.load(vocab_path)
    print(f"Device: {device} | Loading checkpoint: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Load 100 deduplicated capture samples
    strokes_file = ROOT / "data" / "capture" / "strokes.json"
    with open(strokes_file, encoding="utf-8") as f:
        raw_strokes = json.load(f)

    strokes_by_id = {}
    for item in raw_strokes:
        if isinstance(item, dict) and "id" in item:
            strokes_by_id[item["id"]] = item
    sample_ids = sorted(strokes_by_id.keys())
    print(f"Loaded {len(sample_ids)} deduplicated capture samples (DEV).")

    # Load Tanzil clean dictionary for whole-word corruption
    tanzil_clean = ROOT / "data" / "tanzil" / "quran-simple-clean.txt"
    all_quran_words = set()
    if tanzil_clean.exists():
        with open(tanzil_clean, encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    line = line.split("|", 2)[-1]
                for w in line.strip().split():
                    if all(c in vocab.char2idx for c in w):
                        all_quran_words.add(w)
    quran_dict = sorted(list(all_quran_words)) if all_quran_words else ["الله", "محمد", "رسول", "الحق"]

    # Render all 100 samples with optimal preprocessing: Zoom 1.4x, Width 1.8px
    rendered_images = {}
    for sid in sample_ids:
        rendered_images[sid] = render_strokes(
            strokes_by_id[sid]["strokes"],
            zoom=1.4,
            width_after_zoom=1.8,
            antialiasing_factor=4,
        )

    # Split into Lines (40) and Words (60)
    line_ids = [sid for sid in sample_ids if "sample_line_" in sid]
    word_ids = [sid for sid in sample_ids if "sample_word_" in sid]

    # Generate the 4 corruption variants for each sample
    rng = random.Random(42)
    corruption_types = [
        ("dot_swap", "Dot-family Letter Swap"),
        ("del_ins", "Letter Deletion / Insertion"),
        ("word_swap", "Whole-word Swap"),
        ("missing_extra", "Missing / Extra Word"),
    ]

    def generate_variants(sids: List[str]) -> Dict[str, Dict[str, str]]:
        variants: Dict[str, Dict[str, str]] = {ctype: {} for ctype, _ in corruption_types}
        for sid in sids:
            true_t = strokes_by_id[sid]["text"]
            # 1. Dot swap
            c_dot = mutate_dot_swap(true_t, rng) or true_t
            # 2. Deletion / Insertion
            c_del = mutate_delete_insert(true_t, vocab, rng) or true_t
            # 3. Whole-word swap
            c_word = mutate_word_swap(true_t, quran_dict, rng) or true_t
            # 4. Missing / extra
            c_miss = mutate_missing_extra(true_t, quran_dict, rng) or true_t

            variants["dot_swap"][sid] = c_dot
            variants["del_ins"][sid] = c_del
            variants["word_swap"][sid] = c_word
            variants["missing_extra"][sid] = c_miss
        return variants

    line_corruptions = generate_variants(line_ids)
    word_corruptions = generate_variants(word_ids)

    # Evaluation function for a given text candidate against an image
    def evaluate_sample_llr(
        img: Image.Image,
        eval_text: str,
    ) -> Tuple[float, float]:
        """
        Computes per-word LLR:
        (1) expected - best free-path score over the same frames
        (2) expected - best confusable alternative score over the same frames
        """
        w, h = img.size
        tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        pad_w = (-w) % 32
        if pad_w:
            tensor = nn.functional.pad(tensor, (0, pad_w), value=1.0)
        img_tensor = tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            log_probs = model(img_tensor).squeeze(1)  # (T, V)

        T, V = log_probs.shape
        label_text = eval_text[::-1] if order == "visual" else eval_text
        enc_target = vocab.encode(label_text)

        # 1. Forced alignment path
        path, _ = ctc_viterbi_alignment(log_probs, enc_target, blank_idx=0)

        # 2. Extract word segments
        spans = extract_word_frame_spans(path, eval_text, vocab, order=order, blank_idx=0)
        if not spans:
            return -10.0, -10.0

        llr_free_list = []
        llr_conf_list = []

        lp_np = log_probs.cpu().numpy()
        max_free_per_frame = np.max(lp_np, axis=1)

        for w_str, t1, t2, w_enc in spans:
            t1_cl = max(0, min(T - 1, t1))
            t2_cl = max(t1_cl, min(T - 1, t2))
            seg_len = t2_cl - t1_cl + 1

            seg_lp = log_probs[t1_cl : t2_cl + 1]
            s_exp = score_word_ctc_segment(seg_lp, w_enc, blank_idx=0)
            s_free = float(np.sum(max_free_per_frame[t1_cl : t2_cl + 1]))

            # Normalized per-frame / per-word
            llr_f = (s_exp - s_free) / max(1, seg_len)
            llr_free_list.append(llr_f)

            # Confusable alternative
            # Mutate one dot-family char in w_str
            conf_candidates = []
            for ci, c in enumerate(w_str):
                if c in CONFUSABLE_MAP:
                    for alt in CONFUSABLE_MAP[c]:
                        w_alt = w_str[:ci] + alt + w_str[ci+1:]
                        conf_candidates.append(w_alt)

            if conf_candidates:
                alt_scores = []
                for alt_w in conf_candidates[:5]:
                    alt_label = alt_w[::-1] if order == "visual" else alt_w
                    alt_enc = vocab.encode(alt_label)
                    s_alt = score_word_ctc_segment(seg_lp, alt_enc, blank_idx=0)
                    alt_scores.append(s_alt)
                s_best_conf = max(alt_scores)
            else:
                s_best_conf = s_exp - 5.0  # default margin

            llr_c = (s_exp - s_best_conf) / max(1, seg_len)
            llr_conf_list.append(llr_c)

        mean_llr_free = float(np.mean(llr_free_list))
        mean_llr_conf = float(np.mean(llr_conf_list))
        return mean_llr_free, mean_llr_conf

    # Evaluate set
    def run_benchmark_on_subset(sids: List[str], corruptions: Dict[str, Dict[str, str]], set_name: str):
        print(f"\n==========================================================================================")
        print(f"DEV RESULTS: {set_name.upper()} ({len(sids)} SAMPLES)")
        print(f"==========================================================================================")

        # Baseline true scores
        true_scores_free = []
        true_scores_conf = []
        for sid in sids:
            img = rendered_images[sid]
            true_t = strokes_by_id[sid]["text"]
            sf, sc = evaluate_sample_llr(img, true_t)
            true_scores_free.append(sf)
            true_scores_conf.append(sc)

        true_free_arr = np.array(true_scores_free)
        true_conf_arr = np.array(true_scores_conf)

        print(f"{'Corruption Type':<30} | {'Metric':<10} | {'AUC':<8} | {'FRR @ FAR 1%':<14} | {'FRR @ FAR 5%':<14}")
        print("-" * 88)

        for ctype, cdesc in corruption_types:
            corr_free = []
            corr_conf = []
            for sid in sids:
                img = rendered_images[sid]
                c_text = corruptions[ctype][sid]
                sf, sc = evaluate_sample_llr(img, c_text)
                corr_free.append(sf)
                corr_conf.append(sc)

            corr_free_arr = np.array(corr_free)
            corr_conf_arr = np.array(corr_conf)

            # LLR vs Free Path
            auc_free = compute_roc_auc(true_free_arr, corr_free_arr)
            frr_far1_free = compute_frr_at_far(true_free_arr, corr_free_arr, target_far=0.01)
            frr_far5_free = compute_frr_at_far(true_free_arr, corr_free_arr, target_far=0.05)

            # LLR vs Confusable
            auc_conf = compute_roc_auc(true_conf_arr, corr_conf_arr)
            frr_far1_conf = compute_frr_at_far(true_conf_arr, corr_conf_arr, target_far=0.01)
            frr_far5_conf = compute_frr_at_far(true_conf_arr, corr_conf_arr, target_far=0.05)

            print(f"{cdesc:<30} | {'vs Free':<10} | {auc_free:<8.3f} | {frr_far1_free:<13.1f}% | {frr_far5_free:<13.1f}%")
            print(f"{'':<30} | {'vs Conf':<10} | {auc_conf:<8.3f} | {frr_far1_conf:<13.1f}% | {frr_far5_conf:<13.1f}%")
            print("-" * 88)

    # Run on Lines (40) and Words (60)
    run_benchmark_on_subset(line_ids, line_corruptions, "40 Lines (DEV)")
    run_benchmark_on_subset(word_ids, word_corruptions, "60 Word Crops (DEV)")

    print("\n" + "=" * 90)
    print("TASK 2.16 VERIFICATION V2 SUITE COMPLETED SUCCESSFULLY")
    print("=" * 90)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="/content/drive/MyDrive/quran_htr_writer_checkpoints/step2b_v3/best.pt")
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

    run_task_2_16_verification(ckpt_p)
