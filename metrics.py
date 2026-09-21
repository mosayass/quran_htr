"""
metrics.py
Character Error Rate / Word Error Rate for validation.

Both are edit-distance ratios: CER operates on the character sequence,
WER on the whitespace-tokenized word sequence. No external dependency
(pure-Python DP edit distance) -- swap in python-Levenshtein's C
implementation later if this becomes a validation-loop bottleneck on
large val sets; the function signatures below won't need to change.
"""

from __future__ import annotations
from typing import List, Sequence, Tuple


def _levenshtein(a: Sequence, b: Sequence) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            temp = dp[j]
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = temp
    return dp[m]


def cer(pred: str, target: str) -> float:
    """Single-sample CER. 1.0 if target is empty and pred isn't; 0.0 if both empty."""
    if len(target) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return _levenshtein(list(pred), list(target)) / len(target)


def wer(pred: str, target: str) -> float:
    """Single-sample WER (whitespace tokenization)."""
    pred_words, target_words = pred.split(), target.split()
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return _levenshtein(pred_words, target_words) / len(target_words)


def corpus_cer_wer(preds: List[str], targets: List[str]) -> Tuple[float, float]:
    """
    Aggregate CER/WER over a batch/corpus: (total edits) / (total reference
    length), NOT the mean of per-sample ratios. This is the standard way
    to report corpus-level CER/WER -- averaging per-sample ratios would
    overweight short lines (a 1-character line with 1 error has a 100%
    CER, but contributes almost nothing to overall error count).
    """
    total_char_edits = total_char_len = 0
    total_word_edits = total_word_len = 0
    for p, t in zip(preds, targets):
        total_char_edits += _levenshtein(list(p), list(t))
        total_char_len += max(len(t), 1)
        total_word_edits += _levenshtein(p.split(), t.split())
        total_word_len += max(len(t.split()), 1)
    return total_char_edits / total_char_len, total_word_edits / total_word_len


if __name__ == "__main__":
    assert cer("abc", "abc") == 0.0
    assert cer("abd", "abc") == 1 / 3
    assert wer("hello world", "hello there world") == 1 / 3
    c, w = corpus_cer_wer(["abc", "xyz"], ["abc", "abz"])
    print(f"corpus CER={c:.4f} WER={w:.4f}")
    print("metrics.py self-test passed")
