"""
dataset.py
Data loading & preprocessing for Step 1 base pre-training (KHATT line images).

KHATT is distributed by KFUPM under a research license (registration
required — see README.md). Its raw folder layout differs slightly between
release versions, so this module does NOT hard-code KFUPM's directory
structure. Instead:

  1. You point `build_manifest()` at wherever you extracted KHATT.
  2. It walks the tree pairing each line image with its transcription file
     (KHATT ships one .txt per line image, matching stem — e.g.
     `AHTD3A0001_Para1_L1.tif` + `AHTD3A0001_Para1_L1.txt`) and writes a
     single manifest.csv of (image_path, transcription).
  3. LineImageDataset only ever reads manifest.csv — decoupled from
     KHATT's specifics, and reusable later for Step 2/3 data too.

AHCD is isolated *character* images (32x32, no cursive joins/ligatures),
not line images — it is not directly usable for CTC line recognition.
See README.md for how it's instead used as an optional CNN warm-start,
not plugged into this Dataset class.
"""

from __future__ import annotations
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

from vocab import Vocabulary

TARGET_HEIGHT = 64      # fixed line-image height fed to the CNN
WIDTH_MULTIPLE = 32      # pad final width to a multiple of this (stride alignment)
MAX_WIDTH = 1600         # hard cap; KHATT lines rarely exceed ~1200px at h=64


_TATWEEL = "\u0640"


def normalize_text(text: str) -> str:
    """
    Normalize KHATT ground-truth transcriptions before they're written to
    the manifest / encoded against vocab.py. Applied once here, at
    manifest-build time, so every downstream consumer (LineImageDataset,
    validate_manifest_against_vocab, train.py) sees already-clean text.

    - Strip tatweel/kashida (U+0640): a stretching stroke, not a letter --
      arbitrary-width, no fixed visual shape, adds CTC alignment noise
      for no informational value.
    - Strip '#': KHATT's own editorial/paragraph annotation marker, not
      a written character -- no corresponding stroke exists on the page.
    - Canonicalize ASCII ';' to Arabic '؛': KHATT annotators use them
      interchangeably for what a writer treats as the same mark; keeping
      both would ask the model to learn a distinction that isn't in the ink.
    - Strip tashkeel/diacritics (U+064B-U+065F, U+0670): Step 1's vocab
      excludes tashkeel by design (see vocab.py) -- diacritics like
      tanween get added properly, as a full set, in Step 2, not patched
      in reactively here. A real diacritic stroke does exist in the
      image for these (unlike tatweel/'#'), so this is a small, accepted
      image/label mismatch for Step 1 rather than a free normalization.
    - Collapse resulting double spaces, strip leading/trailing whitespace.
    """
    text = text.replace(_TATWEEL, "")
    text = text.replace("#", "")
    text = text.replace(";", "؛")
    text = re.sub(r"[\u064B-\u065F\u0670]", "", text)
    text = re.sub(r" {2,}", " ", text).strip()
    return text


def build_manifest(
    root,
    out_csv,
    image_exts: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
    encoding: str = "utf-8",
) -> int:
    """
    Walk `root` for image files that have a same-stem .txt transcription
    file next to them, and write (image_path, transcription) pairs to
    out_csv. Returns the number of pairs written.

    If your KHATT extraction keeps images and text-ground-truth in
    separate parallel trees (a common re-release layout), pre-process
    once with a small script that copies/symlinks the matching .txt next
    to each image, then run this as-is — keeping this function dumb and
    convention-based makes it robust across KHATT's release variants.
    """
    root = Path(root)
    rows = []
    skipped_oov = 0
    for img_path in root.rglob("*"):
        if img_path.suffix.lower() not in image_exts:
            continue
        txt_path = img_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = normalize_text(txt_path.read_text(encoding=encoding))
        if not text:
            continue
        rows.append((str(img_path), text))

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "transcription"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} (image, transcription) pairs to {out_csv}")
    return len(rows)


def build_manifest_from_image_label_dirs(
    images_dir,
    labels_dir,
    out_csv,
    image_exts: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
    encoding: str = "cp1256",
) -> int:
    """
    For KHATT Kaggle mirrors that ship images and transcriptions in two
    separate, same-naming folders (e.g. iraqyomar/khatt-arabic-hand-written-lines:
    "image/" + "label/") rather than an adjacent .txt per image.

    Ground truth on this mirror is windows-1256 (cp1256) encoded, not
    UTF-8 -- KHATT's original codepage. Decoding as UTF-8 will raise or
    silently mangle the Arabic text, so don't change `encoding` unless
    you've verified a different mirror's file encoding yourself.
    """
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    rows = []
    for img_path in images_dir.rglob("*"):
        if img_path.suffix.lower() not in image_exts:
            continue
        txt_path = labels_dir / img_path.relative_to(images_dir).with_suffix(".txt")
        if not txt_path.exists():
            continue
        text = normalize_text(txt_path.read_text(encoding=encoding))
        if not text:
            continue
        rows.append((str(img_path), text))

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "transcription"])
        writer.writerows(rows)

    print(f"Wrote {len(rows)} (image, transcription) pairs to {out_csv}")
    return len(rows)


def validate_manifest_against_vocab(manifest_csv, vocab: Vocabulary) -> None:
    """Run once after build_manifest() and before training — reports any
    transcriptions containing characters outside vocab.py's charset, so
    you can fix the vocab (or drop the rows) before wasting a training run."""
    bad_rows = 0
    oov_chars = set()
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                vocab.encode(row["transcription"])
            except KeyError:
                bad_rows += 1
                oov_chars |= {c for c in row["transcription"] if c not in vocab.char2idx}
    if bad_rows:
        print(f"[validate_manifest_against_vocab] {bad_rows} rows have OOV chars: {sorted(oov_chars)}")
    else:
        print("[validate_manifest_against_vocab] all transcriptions are vocab-covered.")


def _resize_pad(img: Image.Image) -> torch.Tensor:
    """Grayscale -> resize to TARGET_HEIGHT preserving aspect ratio ->
    right-pad width to a multiple of WIDTH_MULTIPLE -> normalize to [-1, 1]."""
    img = img.convert("L")
    w, h = img.size
    new_w = max(1, round(w * (TARGET_HEIGHT / h)))
    new_w = min(new_w, MAX_WIDTH)
    img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)

    tensor = TF.to_tensor(img)          # (1, H, W) in [0, 1]
    tensor = (tensor - 0.5) / 0.5        # normalize to [-1, 1]

    pad_w = (-new_w) % WIDTH_MULTIPLE
    if pad_w:
        # pad value -1.0 == "white background" under this normalization
        tensor = torch.nn.functional.pad(tensor, (0, pad_w), value=-1.0)
    return tensor  # (1, TARGET_HEIGHT, new_w + pad_w)


@dataclass
class Sample:
    image: torch.Tensor    # (1, H, W)
    target: torch.Tensor   # (L,) int64 char indices
    text: str


class LineImageDataset(Dataset):
    def __init__(self, manifest_csv, vocab: Vocabulary, max_target_len: int = 200):
        self.rows: List[Tuple[str, str]] = []
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                self.rows.append((row["image_path"], row["transcription"]))
        self.vocab = vocab
        self.max_target_len = max_target_len

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Sample:
        img_path, text = self.rows[idx]
        img = Image.open(img_path)
        image_tensor = _resize_pad(img)
        text = text[: self.max_target_len]
        target = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        return Sample(image=image_tensor, target=target, text=text)


def collate_fn(batch: List[Sample]):
    """Pads a batch of variable-width line images to the batch's max width,
    and concatenates targets flat (as nn.CTCLoss expects), along with
    per-sample raw pixel widths and target lengths."""
    max_w = max(s.image.shape[-1] for s in batch)
    images = torch.full((len(batch), 1, TARGET_HEIGHT, max_w), -1.0)
    input_lengths_px = []
    for i, s in enumerate(batch):
        w = s.image.shape[-1]
        images[i, :, :, :w] = s.image
        input_lengths_px.append(w)

    targets = torch.cat([s.target for s in batch])
    target_lengths = torch.tensor([len(s.target) for s in batch], dtype=torch.long)
    texts = [s.text for s in batch]

    return {
        "images": images,                                        # (B, 1, H, W_max)
        "input_lengths_px": torch.tensor(input_lengths_px),       # raw pixel widths, PRE-CNN
        "targets": targets,                                       # (sum(L),) flat, CTC format
        "target_lengths": target_lengths,                         # (B,)
        "texts": texts,
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("usage: python dataset.py <khatt_root> <out_manifest.csv>")
        sys.exit(1)
    n = build_manifest(sys.argv[1], sys.argv[2])
    if n:
        validate_manifest_against_vocab(sys.argv[2], Vocabulary())