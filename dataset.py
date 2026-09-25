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
from torchvision import transforms as T

from vocab import Vocabulary

TARGET_HEIGHT = 64      # fixed line-image height fed to the CNN
WIDTH_MULTIPLE = 32      # pad final width to a multiple of this (stride alignment)
MAX_WIDTH = 1600         # hard cap; KHATT lines rarely exceed ~1200px at h=64

# Mild, train-only augmentation -- rotation/shear/translate jitter to
# expose the model to writer-style variation beyond the raw dataset.
# fill=255 = white background (image is still 0-255 grayscale here, pre-
# normalization). Kept deliberately small: KHATT lines are tightly
# cropped, so aggressive rotation/shear risks clipping strokes at the
# image edges.
_TRAIN_AUGMENT = T.RandomApply(
    [T.RandomAffine(degrees=2, translate=(0.02, 0.02), shear=3, fill=255)],
    p=0.5,
)


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
    text = text.replace("%", "")
    text = text.replace("«", "\"").replace("»", "\"")
    text = text.replace("\u201c", "\"").replace("\u201d", "\"")
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("`", "'")
    text = text.replace("\u2026", "...")
    # Strip stray mathematical / formatting symbols not part of handwriting vocab
    text = re.sub(r"[%/\\*+=<>\[\]{}~^&$@|]", "", text)
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


def split_manifest_by_text(
    manifest_csv,
    out_dir,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[int, int, int]:
    """
    Group-aware train/val/test split, keyed on exact transcription text
    rather than on individual rows/images.

    KHATT writers each copy the same fixed prompt paragraphs (Para1-Para4
    per the iraqyomar mirror's AHTD3A000N_ParaK_L.jpg naming), so many
    rows across different writers share IDENTICAL transcription text. A
    plain random row-level split lets the same sentence appear in train
    (writer A) and test (writer B) -- the model is then evaluated on text
    it has already memorized, just in a different hand. This produced the
    confirmed bimodal CER: near-perfect on repeated fixed paragraphs,
    collapse on genuinely unseen text. Grouping by text ensures every
    occurrence of a given sentence, across every writer who copied it,
    lands in exactly one split. Also correctly isolates "unique" (per
    writer, non-repeated) paragraphs, since those form singleton groups.

    Writes train.csv / val.csv / test.csv into out_dir. Returns the
    (train, val, test) row counts actually written.
    """
    import random as _random

    with open(manifest_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    groups: dict = {}
    for row in rows:
        groups.setdefault(row["transcription"], []).append(row)

    group_keys = list(groups.keys())
    _random.Random(seed).shuffle(group_keys)

    n_total = len(rows)
    train_target = ratios[0] * n_total
    val_target = (ratios[0] + ratios[1]) * n_total

    train_rows, val_rows, test_rows = [], [], []
    running = 0
    for key in group_keys:
        group_rows = groups[key]
        if running < train_target:
            train_rows.extend(group_rows)
        elif running < val_target:
            val_rows.extend(group_rows)
        else:
            test_rows.extend(group_rows)
        running += len(group_rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(split_rows)

    print(f"[split_manifest_by_text] {len(group_keys)} unique text groups -> "
          f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    return len(train_rows), len(val_rows), len(test_rows)


def split_manifest_by_writer(
    manifest_csv,
    out_dir,
    ratios: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
) -> Tuple[int, int, int]:
    """
    Group-aware train/val/test split keyed on writer ID, not text.

    This Kaggle KHATT mirror turned out to contain only 4 fixed
    calibration paragraphs (Para1-Para4), copied by ~1,000 different
    writers -- there is no free/unique text to hold out, so no split
    keyed on transcription text (see split_manifest_by_text()) can ever
    produce a genuinely unseen-sentence test set on this data: every
    possible test sentence is necessarily one of the 4 sentences also
    present in train, just written by someone else.

    Splitting by writer instead asks a different, still-real question:
    can the model read these same 4 sentences' characters/ligatures when
    written by a hand it has never seen during training? That's genuine
    handwriting-style generalization (not vocabulary generalization) --
    and it's arguably closer to this project's actual downstream task
    ("known Quran text + unknown handwriting") than open-vocabulary
    reading would be anyway. Vocabulary breadth is Step 2's job (the
    synthetic Tanzil corpus), not this dataset's.

    Writer ID is taken as the token before the first '_' in the image
    filename stem (e.g. "AHTD3A0001" from "AHTD3A0001_Para1_3.jpg") --
    matches this mirror's naming convention. If you switch to a
    differently-named KHATT mirror, verify this still extracts the
    writer, not the paragraph or line number.

    Writes train.csv / val.csv / test.csv into out_dir. Returns the
    (train, val, test) row counts actually written.
    """
    import random as _random

    with open(manifest_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    groups: dict = {}
    for row in rows:
        writer_id = Path(row["image_path"]).stem.split("_")[0]
        groups.setdefault(writer_id, []).append(row)

    group_keys = list(groups.keys())
    _random.Random(seed).shuffle(group_keys)

    n_total = len(rows)
    train_target = ratios[0] * n_total
    val_target = (ratios[0] + ratios[1]) * n_total

    train_rows, val_rows, test_rows = [], [], []
    running = 0
    for key in group_keys:
        group_rows = groups[key]
        if running < train_target:
            train_rows.extend(group_rows)
        elif running < val_target:
            val_rows.extend(group_rows)
        else:
            test_rows.extend(group_rows)
        running += len(group_rows)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, split_rows in (("train", train_rows), ("val", val_rows), ("test", test_rows)):
        with open(out_dir / f"{name}.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            writer.writerows(split_rows)

    print(f"[split_manifest_by_writer] {len(group_keys)} writers -> "
          f"train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    return len(train_rows), len(val_rows), len(test_rows)


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


def _resize_pad(img: Image.Image, augment: bool = False) -> torch.Tensor:
    """Grayscale -> [train-only mild affine jitter] -> resize to TARGET_HEIGHT
    preserving aspect ratio -> right-pad width to a multiple of
    WIDTH_MULTIPLE -> normalize to [-1, 1]."""
    img = img.convert("L")
    if augment:
        img = _TRAIN_AUGMENT(img)
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
    def __init__(
        self,
        manifest_csv,
        vocab: Vocabulary,
        max_target_len: int = 200,
        augment: bool = False,
        filter_oov: bool = True,
    ):
        self.vocab = vocab
        self.max_target_len = max_target_len
        self.augment = augment
        self.rows: List[Tuple[str, str]] = []
        skipped = 0
        with open(manifest_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                img_path, text = row["image_path"], row["transcription"]
                if filter_oov:
                    try:
                        self.vocab.encode(text[: self.max_target_len])
                    except KeyError:
                        skipped += 1
                        continue
                self.rows.append((img_path, text))
        if skipped:
            print(f"[LineImageDataset] Filtered {skipped} unencodable sample(s) from {manifest_csv}.")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Sample:
        img_path, text = self.rows[idx]
        img = Image.open(img_path)
        image_tensor = _resize_pad(img, augment=self.augment)
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