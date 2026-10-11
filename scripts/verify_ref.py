"""
scripts/verify_ref.py
Task 2.19 Item 3:
Pure NumPy reference implementation of the 3-state HTR verification algorithm:
- CTC Viterbi forced alignment
- Word span extraction with explicit visual-order reversal (LTR frames vs RTL script)
- Per-character normalized LLR (or per-frame normalized)
- Free-path validity check and 3-state decision policy (ACCEPT / UNCERTAIN / WRONG)
Zero PyTorch dependencies. Intended as golden reference for Flutter/Dart porting.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np


def ctc_viterbi_alignment(
    log_probs: np.ndarray,  # Shape: (T, V)
    target_indices: List[int],
    blank_idx: int = 0,
) -> Tuple[List[int], float]:
    """
    Computes Viterbi forced alignment path and total score for target sequence under CTC.
    Pure NumPy implementation with no deep learning framework dependencies.
    """
    T, V = log_probs.shape
    U = len(target_indices)
    if U == 0:
        return [blank_idx] * T, float(np.sum(log_probs[:, blank_idx]))

    # Extended target sequence with interleaved blanks: length K = 2 * U + 1
    K = 2 * U + 1
    ext_target = [blank_idx] * K
    for i, tok in enumerate(target_indices):
        ext_target[2 * i + 1] = tok

    neg_inf = -1e9
    trellis = np.full((T, K), neg_inf, dtype=np.float32)
    backpointers = np.zeros((T, K), dtype=np.int32)

    # Initialization at t = 0 (can start at blank or first character)
    trellis[0, 0] = log_probs[0, ext_target[0]]
    trellis[0, 1] = log_probs[0, ext_target[1]]

    # Forward recurrence
    for t in range(1, T):
        for k in range(K):
            tok = ext_target[k]
            emit_lp = log_probs[t, tok]

            # Transition 1: Stay at current state (k)
            best_prev = trellis[t - 1, k]
            best_p_idx = k

            # Transition 2: Move from previous state (k - 1)
            if k > 0 and trellis[t - 1, k - 1] > best_prev:
                best_prev = trellis[t - 1, k - 1]
                best_p_idx = k - 1

            # Transition 3: Skip blank if transitioning between two different tokens
            if k >= 2 and (k % 2 == 1) and (ext_target[k] != ext_target[k - 2]):
                if trellis[t - 1, k - 2] > best_prev:
                    best_prev = trellis[t - 1, k - 2]
                    best_p_idx = k - 2

            trellis[t, k] = best_prev + emit_lp
            backpointers[t, k] = best_p_idx

    # Termination: best final state at t = T - 1 (either final character or final blank)
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


def greedy_decode_segment(
    log_probs_segment: np.ndarray,  # Shape: (T_seg, V)
    idx2char: Dict[int, str],
    blank_idx: int = 0,
    order: str = "visual",
) -> str:
    """
    Greedy CTC decode over an isolated frame segment: argmax, collapse repeats & blanks.
    """
    if len(log_probs_segment) == 0:
        return ""
    best_indices = np.argmax(log_probs_segment, axis=1)
    collapsed = []
    prev = None
    for idx in best_indices:
        if idx != prev:
            if idx != blank_idx:
                collapsed.append(int(idx))
            prev = idx
    chars = [idx2char.get(i, "") for i in collapsed]
    decoded = "".join(chars).strip()
    if order == "visual":
        decoded = decoded[::-1]
    return decoded


def verify_reference(
    log_probs: np.ndarray,  # (T, V) numpy array
    expected_text: str,
    vocab_char2idx: Dict[str, int],
    vocab_idx2char: Dict[int, str],
    quran_dict: Optional[set[str]] = None,
    t_accept: float = -0.60,
    t_wrong: float = -0.95,
    order: str = "visual",
    norm: str = "char",
    blank_idx: int = 0,
) -> List[Dict[str, Any]]:
    """
    Pure NumPy reference verification logic:
    - Explicit visual-order reversal:
      * When order == 'visual', the CNN/CRNN time frames t=0..T-1 proceed horizontally Left-to-Right.
      * Arabic text is physically written Right-to-Left.
      * Therefore, the leftmost character on screen appears earliest in the time frame sequence.
      * Reversing expected_text ('label_text = expected_text[::-1]') aligns character indices with visual time frames.
      * Words are processed in reverse order ('vis_words = [w[::-1] for w in words[::-1]]').
      * Results are mapped back to original logical reading order (0 .. N-1).
    - Normalization:
      * norm == 'char': LLR = (s_exp - s_free) / max(1, num_chars). Prevents single-letter error dilution.
      * norm == 'frame': LLR = (s_exp - s_free) / seg_len.
    - 3-State policy:
      * LLR >= t_accept -> ACCEPT
      * LLR <= t_wrong  -> WRONG
      * otherwise       -> UNCERTAIN
    """
    words = expected_text.split()
    if not words:
        return []

    T, V = log_probs.shape

    # 1. Visual-order sequence alignment
    label_text = expected_text[::-1] if order == "visual" else expected_text
    enc_target = [vocab_char2idx[c] for c in label_text if c in vocab_char2idx]

    # Global CTC forced alignment
    path, _ = ctc_viterbi_alignment(log_probs, enc_target, blank_idx=blank_idx)

    # 2. Word span mapping
    if order == "visual":
        vis_words = [w[::-1] for w in words[::-1]]
    else:
        vis_words = words

    word_spans = []
    char_pos = 0
    for w in vis_words:
        w_enc = [vocab_char2idx[c] for c in w if c in vocab_char2idx]
        w_len = len(w_enc)
        word_spans.append((w, char_pos, char_pos + w_len, w_enc))
        char_pos += w_len + 1  # +1 for inter-word space

    results = []
    max_free_per_frame = np.max(log_probs, axis=1)

    for w_idx_vis, (w_vis, c_start, c_end, w_enc) in enumerate(word_spans):
        active_states = set(range(2 * c_start, 2 * c_end + 1))
        frames = [t for t in range(T) if path[t] in active_states]
        t1 = min(frames) if frames else 0
        t2 = max(frames) if frames else T - 1
        seg_len = max(1, t2 - t1 + 1)
        char_len = max(1, len(w_enc))

        seg_lp = log_probs[t1 : t2 + 1]

        # Forced alignment score on segment
        _, s_exp = ctc_viterbi_alignment(seg_lp, w_enc, blank_idx=blank_idx)
        s_free = float(np.sum(max_free_per_frame[t1 : t2 + 1]))

        # Normalized log-likelihood ratio
        norm_factor = char_len if norm == "char" else seg_len
        llr = (s_exp - s_free) / norm_factor

        # Free-path decode over the same frame segment
        free_dec = greedy_decode_segment(seg_lp, vocab_idx2char, blank_idx=blank_idx, order=order)
        orig_w = w_vis[::-1] if order == "visual" else w_vis
        orig_idx = len(words) - 1 - w_idx_vis if order == "visual" else w_idx_vis

        is_free_in_dict = bool(quran_dict and (free_dec in quran_dict) and (free_dec != ""))

        # 3-State decision policy
        if llr >= t_accept:
            state = "ACCEPT"
        elif llr <= t_wrong:
            state = "WRONG"
        else:
            state = "UNCERTAIN"

        results.append({
            "word": orig_w,
            "word_index": orig_idx,
            "t_start": int(t1),
            "t_end": int(t2),
            "llr": round(float(llr), 4),
            "free_decode": free_dec,
            "is_free_in_dict": is_free_in_dict,
            "state": state,
        })

    # Sort results by original logical reading order
    results.sort(key=lambda r: r["word_index"])
    return results


def run_golden_tests(golden_file: Path) -> bool:
    """Verifies verify_reference against golden test vectors."""
    with open(golden_file, "r", encoding="utf-8") as f:
        vectors = json.load(f)

    vocab_char2idx = vectors["vocab_char2idx"]
    vocab_idx2char = {int(k): v for k, v in vectors["vocab_idx2char"].items()}
    test_cases = vectors["test_cases"]

    print(f"Running {len(test_cases)} golden test vectors from {golden_file}...")
    all_passed = True

    for i, tc in enumerate(test_cases):
        sid = tc["sample_id"]
        exp_text = tc["text"]
        lp = np.array(tc["log_probs"], dtype=np.float32)
        norm = tc.get("norm", "char")
        t_acc = tc.get("t_accept", -0.60)
        t_wrg = tc.get("t_wrong", -0.95)

        verdicts = verify_reference(
            log_probs=lp,
            expected_text=exp_text,
            vocab_char2idx=vocab_char2idx,
            vocab_idx2char=vocab_idx2char,
            t_accept=t_acc,
            t_wrong=t_wrg,
            order="visual",
            norm=norm,
        )

        expected_verdicts = tc["expected_verdicts"]
        match = True
        if len(verdicts) != len(expected_verdicts):
            match = False
        else:
            for v_got, v_exp in zip(verdicts, expected_verdicts):
                if v_got["state"] != v_exp["state"] or abs(v_got["llr"] - v_exp["llr"]) > 0.05:
                    match = False
                    break

        status = "PASS" if match else "FAIL"
        print(f"  [{i+1:02d}/{len(test_cases)}] {sid:<20} -> {status}")
        if not match:
            all_passed = False
            print(f"       Got:      {verdicts}")
            print(f"       Expected: {expected_verdicts}")

    print("=" * 60)
    print(f"Golden Tests Result: {'ALL PASSED' if all_passed else 'FAILURES DETECTED'}")
    print("=" * 60)
    return all_passed


def main():
    p = argparse.ArgumentParser(description="NumPy-only HTR Verification Reference")
    p.add_argument("--golden", default=None, help="Path to golden test vectors JSON to validate")
    args = p.parse_args()

    if args.golden:
        ok = run_golden_tests(Path(args.golden))
        sys.exit(0 if ok else 1)
    else:
        print("verify_ref.py: Reference NumPy implementation loaded successfully.")


if __name__ == "__main__":
    main()
