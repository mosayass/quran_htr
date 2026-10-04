"""
shards.py
High-performance in-memory shard format & dataset for HTR line images.

Features:
1. Pre-resized 64px height uint8 line images with text labels.
2. ShardDataset: loads entire split into RAM; padding/normalization is identical to dataset.py.
3. Fast iteration benchmark measuring lines/sec.
4. Shard writer utilities for synthetic generation and KHATT manifests.
"""

from __future__ import annotations
import csv
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T

from vocab import Vocabulary
from dataset import Sample, collate_fn, TARGET_HEIGHT, WIDTH_MULTIPLE, _TRAIN_AUGMENT


class ShardDataset(Dataset):
    """
    In-memory dataset loaded from pre-resized 64px uint8 shard files.
    Padding to WIDTH_MULTIPLE and [-1, 1] normalization are identical to dataset.py.
    """
    def __init__(
        self,
        shard_paths: Union[str, Path, List[Union[str, Path]]],
        vocab: Vocabulary,
        augment: bool = False,
        max_target_len: int = 200,
        filter_oov: bool = True,
    ):
        self.vocab = vocab
        self.augment = augment
        self.max_target_len = max_target_len
        self.images: List[torch.Tensor] = []  # List of (1, 64, W) uint8
        self.texts: List[str] = []

        if isinstance(shard_paths, (str, Path)):
            p = Path(shard_paths)
            if p.is_dir():
                paths = sorted(list(p.glob("*.pt")))
            else:
                paths = [p]
        else:
            paths = sorted([Path(p) for p in shard_paths])

        skipped = 0
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Shard not found: {p}")
            data = torch.load(p, map_location="cpu", weights_only=False)
            imgs = data["images"]
            txts = data["texts"]
            for img, text in zip(imgs, txts):
                if filter_oov:
                    try:
                        self.vocab.encode(text[: self.max_target_len])
                    except KeyError:
                        skipped += 1
                        continue
                self.images.append(img)
                self.texts.append(text)

        if skipped:
            print(f"[ShardDataset] Filtered {skipped} unencodable sample(s).")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Sample:
        img_u8 = self.images[idx]  # (1, 64, W) uint8
        text = self.texts[idx][: self.max_target_len]

        if self.augment:
            pil_img = Image.fromarray(img_u8[0].numpy())
            pil_img = _TRAIN_AUGMENT(pil_img)
            img_u8 = torch.from_numpy(np.array(pil_img)).unsqueeze(0)

        # Padding & normalization identical to dataset.py _resize_pad:
        # 1. Normalize [0, 255] uint8 -> [-1, 1] float
        tensor = (img_u8.float() / 255.0 - 0.5) / 0.5
        w = tensor.shape[-1]
        pad_w = (-w) % WIDTH_MULTIPLE
        if pad_w:
            # Pad value 1.0 corresponds to 255 (white background)
            tensor = torch.nn.functional.pad(tensor, (0, pad_w), value=1.0)

        target = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        return Sample(image=tensor, target=target, text=text)


def write_shard_file(images: List[torch.Tensor], texts: List[str], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "images": images,
        "texts": texts,
        "count": len(images),
    }, out_path)


def convert_manifest_to_shards(
    manifest_csv: Path,
    out_dir: Path,
    split_name: str,
    max_per_shard: int = 10000,
) -> List[Path]:
    """
    Reads a CSV manifest (image_path, transcription), resizes images to 64px height,
    and writes them into uint8 .pt shard files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_paths = []

    print(f"[convert_manifest_to_shards] Processing {manifest_csv} -> {out_dir}...")
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    total = len(reader)
    shard_imgs, shard_txts = [], []
    shard_idx = 0

    for i, row in enumerate(reader):
        img_path = Path(row["image_path"])
        if not img_path.exists():
            continue
        try:
            img = Image.open(img_path).convert("L")
            w, h = img.size
            new_w = max(1, round(w * (TARGET_HEIGHT / h)))
            new_w = min(new_w, 1600)
            resized = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)
            arr = np.array(resized, dtype=np.uint8)
            img_tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, 64, W)
            shard_imgs.append(img_tensor)
            shard_txts.append(row["transcription"])
        except Exception as e:
            print(f"  [warn] Error loading {img_path}: {e}")
            continue

        if len(shard_imgs) >= max_per_shard or i == total - 1:
            shard_name = f"{split_name}_{shard_idx:03d}.pt"
            shard_path = out_dir / shard_name
            write_shard_file(shard_imgs, shard_txts, shard_path)
            print(f"  Wrote shard: {shard_path.name} ({len(shard_imgs)} samples, {shard_path.stat().st_size / (1024*1024):.1f} MB)")
            shard_paths.append(shard_path)
            shard_imgs, shard_txts = [], []
            shard_idx += 1

    return shard_paths


def benchmark_shard_iteration(
    shard_paths: List[Path],
    vocab: Vocabulary,
    batch_size: int = 16,
    num_samples: int = 2000,
) -> float:
    """Measures and reports data loading iteration speed (lines/sec) over ShardDataset."""
    ds = ShardDataset(shard_paths, vocab, augment=True)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=0
    )

    t0 = time.time()
    n_loaded = 0
    for batch in loader:
        n_loaded += len(batch["texts"])
        if n_loaded >= num_samples:
            break
    elapsed = max(time.time() - t0, 1e-4)
    speed = n_loaded / elapsed
    print(f"[benchmark_shard_iteration] Loaded {n_loaded} lines in {elapsed:.2f}s ({speed:.1f} lines/sec)")
    return speed
