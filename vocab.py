"""
vocab.py
Arabic character vocabulary and CTC index mapping for the Quran HTR
Step 1 base pre-training model (KHATT / AHCD open-domain Arabic text).

Notes
-----
* This is a *provisional* vocabulary sized for Step 1 (open-domain Arabic
  handwriting). It intentionally excludes tashkeel (diacritics) because
  KHATT / AHCD ground truth is undiacritized. Tashkeel + Quran-specific
  symbols (waqf marks, small alef, sukoon, etc.) get added when we build
  the Step 2 vocabulary, since the model's output layer size must match
  whichever target character set that stage trains against.
* Index 0 is reserved for the CTC "blank" token everywhere in this
  project. Never renumber this without re-training from scratch —
  checkpoints are vocab-index-dependent.
* We deliberately use plain Unicode Arabic letters, NOT presentation-form
  codepoints (initial/medial/final/isolated glyph variants). A CRNN
  learns the visual, positional glyph shape directly from pixels, so the
  *target* sequence should stay at the abstract-character level (this is
  standard practice in Arabic OCR/HTR literature).
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import List, Optional, Sequence

CTC_BLANK = "<blank>"

# Base Arabic letters (28 letters + hamza carriers + lam-alef ligature,
# which handwriting renders as a single connected glyph).
_ARABIC_LETTERS = [
    "ء", "آ", "أ", "ؤ", "إ", "ئ", "ا", "ب", "ة", "ت", "ث", "ج", "ح", "خ",
    "د", "ذ", "ر", "ز", "س", "ش", "ص", "ض", "ط", "ظ", "ع", "غ", "ف", "ق",
    "ك", "ل", "م", "ن", "ه", "و", "ى", "ي",
    "\ufefb",  # lam-alef ligature (isolated form, U+FEFB)
]

_ARABIC_DIGITS = list("٠١٢٣٤٥٦٧٨٩")   # Eastern Arabic-Indic digits (used throughout KHATT)
_LATIN_DIGITS = list("0123456789")     # KHATT forms occasionally mix in Latin digits/dates
_PUNCTUATION = list(" .,؛؟!-_()\"':،")


def build_charset() -> List[str]:
    """Deterministic ordering — do not change once training has started."""
    charset = [CTC_BLANK]
    charset += _ARABIC_LETTERS
    charset += _ARABIC_DIGITS
    charset += _LATIN_DIGITS
    charset += _PUNCTUATION
    seen = set()
    ordered = []
    for c in charset:
        if c not in seen:
            seen.add(c)
            ordered.append(c)
    return ordered


class Vocabulary:
    def __init__(self, chars: Optional[Sequence[str]] = None):
        self.chars = list(chars) if chars is not None else build_charset()
        if self.chars[0] != CTC_BLANK:
            raise ValueError("index 0 must be the CTC blank token")
        self.char2idx = {c: i for i, c in enumerate(self.chars)}
        self.idx2char = {i: c for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars)

    @property
    def blank_id(self) -> int:
        return 0

    def encode(self, text: str) -> List[int]:
        unknown = sorted({c for c in text if c not in self.char2idx})
        if unknown:
            raise KeyError(
                f"OOV characters {unknown!r} in transcription {text!r}. "
                "Either add them to vocab.py's charset (and retrain), or "
                "filter this sample out during manifest building."
            )
        return [self.char2idx[c] for c in text]

    def decode(self, indices: Sequence[int]) -> str:
        return "".join(self.idx2char[i] for i in indices if i != self.blank_id)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.chars, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Vocabulary":
        chars = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(chars)


if __name__ == "__main__":
    v = Vocabulary()
    print(f"Vocab size (incl. blank): {len(v)}")
    print(v.chars)
