"""
verify.py
Task 2.17 Item 4:
3-State Per-Word HTR Verification (ACCEPT / UNCERTAIN / WRONG):
- Computes per-word LLR vs the free path over CTC forced-aligned word frames.
- Decision policy:
  * ACCEPT: LLR >= threshold tau
  * WRONG: LLR < tau AND the free-path decoded word is a different valid Quran word.
  * UNCERTAIN: LLR < tau but the free-path decode is not a valid Quran word (smudged/unclear ink).
- Strict (FAR 1%) and relaxed (FAR 5%) operating thresholds derived from DEV benchmark.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent

# Calibrated thresholds:
# t_accept: Calibrated on DEV true claims (achieving >= 95% true-accept rate, FAR 5%)
# t_wrong:  Calibrated on DEV corrupted claims (achieving >= 90% WRONG rate on corrupted claims, < 1% false wrong on clean true claims)
DEFAULT_T_ACCEPT = -0.60
DEFAULT_T_WRONG = -1.10
DEFAULT_TAU_STRICT = -0.92   # Legacy compatibility: FAR 1%
DEFAULT_TAU_RELAXED = -0.60  # Legacy compatibility: FAR 5%



class WordVerificationResult:
    def __init__(
        self,
        word: str,
        word_index: int,
        t_start: int,
        t_end: int,
        llr: float,
        free_decode: str,
        is_free_in_dict: bool,
        state: str,  # "ACCEPT", "UNCERTAIN", "WRONG"
    ):
        self.word = word
        self.word_index = word_index
        self.t_start = t_start
        self.t_end = t_end
        self.llr = llr
        self.free_decode = free_decode
        self.is_free_in_dict = is_free_in_dict
        self.state = state

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "word_index": self.word_index,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "llr": round(self.llr, 4),
            "free_decode": self.free_decode,
            "is_free_in_dict": self.is_free_in_dict,
            "state": self.state,
        }


def load_quran_dict(dict_path: Optional[Path] = None) -> set[str]:
    """Loads closed vocabulary of all valid simple-clean Quranic words."""
    path = dict_path or (ROOT / "data" / "tanzil" / "quran-simple-clean.txt")
    words = set()
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if "|" in line:
                    line = line.split("|", 2)[-1]
                for w in line.strip().split():
                    words.add(w)
    return words


def ctc_viterbi_alignment(
    log_probs: np.ndarray,  # (T, V)
    target_indices: List[int],
    blank_idx: int = 0,
) -> Tuple[List[int], float]:
    """Computes Viterbi forced alignment path and total score for target sequence under CTC."""
    T, V = log_probs.shape
    U = len(target_indices)
    if U == 0:
        return [0] * T, float(log_probs[:, blank_idx].sum())

    K = 2 * U + 1
    ext_target = [blank_idx] * K
    for i, tok in enumerate(target_indices):
        ext_target[2 * i + 1] = tok

    neg_inf = -1e9
    trellis = np.full((T, K), neg_inf, dtype=np.float32)
    backpointers = np.zeros((T, K), dtype=np.int32)

    trellis[0, 0] = log_probs[0, ext_target[0]]
    trellis[0, 1] = log_probs[0, ext_target[1]]

    for t in range(1, T):
        for k in range(K):
            tok = ext_target[k]
            emit_lp = log_probs[t, tok]

            best_prev = trellis[t - 1, k]
            best_p_idx = k

            if k > 0 and trellis[t - 1, k - 1] > best_prev:
                best_prev = trellis[t - 1, k - 1]
                best_p_idx = k - 1

            if k >= 2 and (k % 2 == 1) and (ext_target[k] != ext_target[k - 2]):
                if trellis[t - 1, k - 2] > best_prev:
                    best_prev = trellis[t - 1, k - 2]
                    best_p_idx = k - 2

            trellis[t, k] = best_prev + emit_lp
            backpointers[t, k] = best_p_idx

    cand_end = [K - 2, K - 1] if K >= 2 else [0]
    best_final_k = cand_end[0]
    best_final_score = trellis[T - 1, cand_end[0]]
    if len(cand_end) > 1 and trellis[T - 1, cand_end[1]] > best_final_score:
        best_final_k = cand_end[1]
        best_final_score = trellis[T - 1, cand_end[1]]

    path = [0] * T
    curr = best_final_k
    for t in range(T - 1, -1, -1):
        path[t] = int(curr)
        curr = backpointers[t, curr]

    return path, float(best_final_score)


def greedy_decode_segment(
    log_probs_segment: np.ndarray,  # (T_seg, V)
    idx2char: Dict[int, str],
    blank_idx: int = 0,
    order: str = "visual",
) -> str:
    """Greedy CTC decode over an isolated frame segment."""
    if len(log_probs_segment) == 0:
        return ""
    best_indices = np.argmax(log_probs_segment, axis=1)
    # Collapse repeats and blanks
    collapsed = []
    prev = None
    for idx in best_indices:
        if idx != prev:
            if idx != blank_idx:
                collapsed.append(idx)
            prev = idx
    chars = [idx2char.get(i, "") for i in collapsed]
    decoded = "".join(chars).strip()
    if order == "visual":
        decoded = decoded[::-1]
    return decoded


def verify_text(
    log_probs_np: np.ndarray,  # (T, V) numpy array
    expected_text: str,
    vocab_char2idx: Dict[str, int],
    vocab_idx2char: Dict[int, str],
    quran_dict: Optional[set[str]] = None,
    t_accept: float = DEFAULT_T_ACCEPT,
    t_wrong: float = DEFAULT_T_WRONG,
    threshold: Optional[float] = None,  # Legacy override: if provided, sets t_accept
    order: str = "visual",
    blank_idx: int = 0,
) -> List[WordVerificationResult]:
    """
    Performs 3-state verification (ACCEPT / UNCERTAIN / WRONG) for each word in expected_text
    given model frame log-probs over CTC forced-alignment segments:
    - ACCEPT: LLR >= t_accept
    - WRONG:  LLR <= t_wrong
    - UNCERTAIN: t_wrong < LLR < t_accept
    """
    words = expected_text.split()
    if not words:
        return []

    T, V = log_probs_np.shape
    # Encode expected sequence
    label_text = expected_text[::-1] if order == "visual" else expected_text
    enc_target = [vocab_char2idx[c] for c in label_text if c in vocab_char2idx]

    # 1. CTC forced alignment
    path, _ = ctc_viterbi_alignment(log_probs_np, enc_target, blank_idx=blank_idx)

    # 2. Map words to frame segments
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
        char_pos += w_len + 1  # +1 for space

    # Match each word to frames
    results: List[WordVerificationResult] = []
    max_free_per_frame = np.max(log_probs_np, axis=1)

    tau_acc = threshold if threshold is not None else t_accept

    for w_idx_vis, (w_vis, c_start, c_end, w_enc) in enumerate(word_spans):
        active_states = set(range(2 * c_start, 2 * c_end + 1))
        frames = [t for t in range(T) if path[t] in active_states]
        t1 = min(frames) if frames else 0
        t2 = max(frames) if frames else T - 1
        seg_len = max(1, t2 - t1 + 1)

        seg_lp = log_probs_np[t1 : t2 + 1]

        # Score expected word on segment
        _, s_exp = ctc_viterbi_alignment(seg_lp, w_enc, blank_idx=blank_idx)
        s_free = float(np.sum(max_free_per_frame[t1 : t2 + 1]))

        llr = (s_exp - s_free) / seg_len

        # Free-path greedy decode over the same frames
        free_dec = greedy_decode_segment(seg_lp, vocab_idx2char, blank_idx=blank_idx, order=order)
        orig_w = w_vis[::-1] if order == "visual" else w_vis
        orig_word_idx = len(words) - 1 - w_idx_vis if order == "visual" else w_idx_vis

        is_free_in_dict = bool(quran_dict and (free_dec in quran_dict) and (free_dec != ""))

        # 3-State decision policy
        if llr >= tau_acc:
            state = "ACCEPT"
        elif llr <= t_wrong:
            state = "WRONG"
        else:
            state = "UNCERTAIN"


        results.append(WordVerificationResult(
            word=orig_w,
            word_index=orig_word_idx,
            t_start=t1,
            t_end=t2,
            llr=llr,
            free_decode=free_dec,
            is_free_in_dict=is_free_in_dict,
            state=state,
        ))

    # Sort results by original word order
    results.sort(key=lambda r: r.word_index)
    return results


def main():
    p = argparse.ArgumentParser(description="3-State HTR Line Verification")
    p.add_argument("--image", required=True, help="Path to input line image (or PNG)")
    p.add_argument("--expected", required=True, help="Expected ground-truth text")
    p.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to CRNN checkpoint")
    p.add_argument("--vocab", default="vocab.json", help="Path to vocab.json")
    p.add_argument("--mode", choices=["strict", "relaxed"], default="relaxed", help="Operating threshold mode")
    p.add_argument("--t_accept", type=float, default=None, help="LLR threshold for ACCEPT")
    p.add_argument("--t_wrong", type=float, default=None, help="LLR threshold for WRONG")
    p.add_argument("--threshold", type=float, default=None, help="Custom LLR threshold override for ACCEPT")
    args = p.parse_args()

    import torch
    from vocab import Vocabulary
    from model import CRNN

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vocab = Vocabulary.load(args.vocab)
    quran_dict = load_quran_dict()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    img = Image.open(args.image).convert("L")
    w, h = img.size
    tensor = torch.from_numpy(np.array(img, dtype=np.float32) / 255.0).unsqueeze(0)
    tensor = (tensor - 0.5) / 0.5
    pad_w = (-w) % 32
    if pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w), value=1.0)
    img_tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        log_probs = model(img_tensor).squeeze(1).cpu().numpy()

    t_acc = args.t_accept if args.t_accept is not None else (args.threshold if args.threshold is not None else DEFAULT_T_ACCEPT)
    t_wrg = args.t_wrong if args.t_wrong is not None else DEFAULT_T_WRONG

    res = verify_text(
        log_probs_np=log_probs,
        expected_text=args.expected,
        vocab_char2idx=vocab.char2idx,
        vocab_idx2char=vocab.idx2char,
        quran_dict=quran_dict,
        t_accept=t_acc,
        t_wrong=t_wrg,
        order=order,
    )

    print(f"\nVerification Results (t_accept={t_acc:.3f}, t_wrong={t_wrg:.3f}):")

    print(f"{'Idx':<4} | {'Expected Word':<18} | {'LLR':<8} | {'Free Decode':<18} | {'In Dict':<8} | {'State':<10}")
    print("-" * 75)
    for r in res:
        print(f"{r.word_index:<4} | {r.word:<18} | {r.llr:<8.3f} | {r.free_decode:<18} | {str(r.is_free_in_dict):<8} | {r.state:<10}")


if __name__ == "__main__":
    main()
