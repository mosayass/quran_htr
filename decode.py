"""
decode.py
Vocabulary-aware CTC decoding for the Quran HTR CRNN model.

model.py ships a bare index-level greedy_ctc_decode() for quick
architecture sanity checks (see its docstring). This module wraps that
idea with vocabulary awareness (raw indices -> Arabic text) and adds a
proper prefix beam search decoder. Training/validation metrics (CER/WER)
and eventual on-device inference should use the functions here, not
model.py's bare version.

Blank-token edge cases handled here:
  - An all-blank predicted path (empty collapsed sequence) decodes to ""
    (empty string), not an error -- expected early in training before the
    model has learned anything, and for genuinely blank input regions
    (e.g. padding columns).
  - Every function here takes the Vocabulary object directly rather than
    a bare blank_id int, specifically to prevent a blank-id mismatch bug
    (vocab.blank_id is always 0 in this project -- see vocab.py).
"""

from __future__ import annotations
from typing import List

import torch

from vocab import Vocabulary


def greedy_decode(log_probs: torch.Tensor, vocab: Vocabulary) -> List[str]:
    """log_probs: (T, B, V) -> list of B decoded strings.
    Best-path decode + CTC collapse (drop repeats, drop blanks)."""
    best_path = log_probs.argmax(dim=2).transpose(0, 1)  # (T,B) -> (B,T)
    texts = []
    for seq in best_path:
        seq = seq.tolist()
        collapsed, prev = [], None
        for s in seq:
            if s != prev and s != vocab.blank_id:
                collapsed.append(s)
            prev = s
        texts.append(vocab.decode(collapsed))
    return texts


def beam_search_decode(
    log_probs: torch.Tensor,
    vocab: Vocabulary,
    beam_width: int = 10,
) -> List[str]:
    """
    Prefix beam search CTC decode (Hannun et al., 2014), no external
    language model -- that's a Step 2/3 enhancement once a Quran-domain
    character/word LM exists to plug in. Even without an LM, beam search
    should meaningfully outperform greedy at high-output-entropy points
    (early training, visually ambiguous strokes).

    This is O(T * beam_width * V) per sample -- noticeably slower than
    greedy_decode. Use it for final evaluation / spot checks, not as the
    per-batch metric inside the training loop's validation pass (that
    uses greedy_decode for speed -- see train.py).

    log_probs: (T, B, V) log-probabilities from CRNN.forward().
    Returns: list of B decoded strings.
    """
    T, B, V = log_probs.shape
    probs = log_probs.exp().cpu()
    results = []

    for b in range(B):
        # beam: {prefix_tuple: (prob_ending_in_blank, prob_ending_in_non_blank)}
        beam = {(): (1.0, 0.0)}
        for t in range(T):
            next_beam: dict = {}
            probs_t = probs[t, b]
            for prefix, (p_b, p_nb) in beam.items():
                p_total = p_b + p_nb
                for c in range(V):
                    p_c = probs_t[c].item()
                    if p_c == 0.0:
                        continue
                    if c == vocab.blank_id:
                        nb, nnb = next_beam.get(prefix, (0.0, 0.0))
                        next_beam[prefix] = (nb + p_total * p_c, nnb)
                        continue
                    end_char = prefix[-1] if prefix else None
                    if c == end_char:
                        # repeated char: only extends the collapsed sequence
                        # via the path that had a blank right before it
                        nb, nnb = next_beam.get(prefix, (0.0, 0.0))
                        next_beam[prefix] = (nb, nnb + p_b * p_c)
                        new_prefix = prefix + (c,)
                        nb2, nnb2 = next_beam.get(new_prefix, (0.0, 0.0))
                        next_beam[new_prefix] = (nb2, nnb2 + p_nb * p_c)
                    else:
                        new_prefix = prefix + (c,)
                        nb2, nnb2 = next_beam.get(new_prefix, (0.0, 0.0))
                        next_beam[new_prefix] = (nb2, nnb2 + p_total * p_c)

            beam = dict(
                sorted(next_beam.items(), key=lambda kv: kv[1][0] + kv[1][1], reverse=True)[:beam_width]
            )

        best_prefix = max(beam.items(), key=lambda kv: kv[1][0] + kv[1][1])[0]
        results.append(vocab.decode(list(best_prefix)))

    return results


if __name__ == "__main__":
    # quick self-test with random logits -- no model or data needed
    torch.manual_seed(0)
    v = Vocabulary()
    T, B, V = 20, 2, len(v)
    dummy_log_probs = torch.randn(T, B, V).log_softmax(dim=2)
    print("greedy:", greedy_decode(dummy_log_probs, v))
    print("beam  :", beam_search_decode(dummy_log_probs, v, beam_width=5))
