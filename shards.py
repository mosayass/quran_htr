"""
shards.py
High-performance in-memory shard format & dataset for HTR line images.

Features:
1. Pre-resized 64px height uint8 line images with text labels.
2. Contiguous array storage: All images in a shard/dataset are stored in a single
   contiguous uint8 buffer (64, total_width) + offsets/widths arrays, minimizing
   Python object overhead and avoiding copy-on-write memory bloat under multi-worker DataLoader.
3. On-the-fly tensor-space affine augmentation for train shards:
   rotation +-2deg, shear +-3deg, translation +-2% (fill=1.0 for white background).
4. Fast iteration benchmark measuring lines/sec and RSS memory reporting.
"""

from __future__ import annotations
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional

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
import torchvision.transforms.functional as TF

from vocab import Vocabulary
from dataset import Sample, collate_fn, TARGET_HEIGHT, WIDTH_MULTIPLE


class ShardDataset(Dataset):
    """
    High-performance in-memory dataset storing line images in a single contiguous
    uint8 tensor (64, total_width) + int32 offsets and widths.
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

        if isinstance(shard_paths, (str, Path)):
            p = Path(shard_paths)
            if p.is_dir():
                paths = sorted(list(p.glob("*.pt")))
            else:
                paths = [p]
        else:
            paths = sorted([Path(p) for p in shard_paths])

        all_imgs: List[torch.Tensor] = []
        all_txts: List[str] = []
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
                # Ensure 2D (64, W)
                if img.dim() == 3 and img.shape[0] == 1:
                    img = img.squeeze(0)
                all_imgs.append(img)
                all_txts.append(text)

        if skipped:
            print(f"[ShardDataset] Filtered {skipped} unencodable sample(s).")

        self.num_samples = len(all_imgs)
        self.texts = all_txts

        # Pack into contiguous 2D array + offsets
        if self.num_samples > 0:
            widths_list = [img.shape[1] for img in all_imgs]
            self.widths = torch.tensor(widths_list, dtype=torch.int32)
            offsets_list = [0]
            for w in widths_list[:-1]:
                offsets_list.append(offsets_list[-1] + w)
            self.offsets = torch.tensor(offsets_list, dtype=torch.int32)
            # Single contiguous tensor: (64, total_width) uint8
            self.buffer = torch.cat(all_imgs, dim=1).contiguous()
        else:
            self.buffer = torch.zeros((TARGET_HEIGHT, 0), dtype=torch.uint8)
            self.offsets = torch.zeros(0, dtype=torch.int32)
            self.widths = torch.zeros(0, dtype=torch.int32)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Sample:
        offset = int(self.offsets[idx])
        w = int(self.widths[idx])
        text = self.texts[idx][: self.max_target_len]

        # Slice 2D array: (64, W) -> unsqueeze to (1, 64, W)
        img_u8 = self.buffer[:, offset : offset + w].unsqueeze(0)

        # Normalize [0, 255] uint8 -> [-1, 1] float
        tensor = (img_u8.float() / 255.0 - 0.5) / 0.5

        # On-the-fly tensor-space affine augmentation for train shards
        if self.augment:
            angle = random.uniform(-2.0, 2.0)
            shear = random.uniform(-3.0, 3.0)
            max_dx = 0.02 * w
            max_dy = 0.02 * TARGET_HEIGHT
            translate = [random.uniform(-max_dx, max_dx), random.uniform(-max_dy, max_dy)]
            # fill=1.0 is white background in [-1, 1] space
            tensor = TF.affine(
                tensor,
                angle=angle,
                translate=translate,
                scale=1.0,
                shear=shear,
                fill=1.0,
            )

        # Right-pad width to a multiple of WIDTH_MULTIPLE=32 with 1.0 (white background)
        cur_w = tensor.shape[-1]
        pad_w = (-cur_w) % WIDTH_MULTIPLE
        if pad_w:
            tensor = torch.nn.functional.pad(tensor, (0, pad_w), value=1.0)

        target = torch.tensor(self.vocab.encode(text), dtype=torch.long)
        return Sample(image=tensor, target=target, text=text)


def write_shard_file(images: List[torch.Tensor], texts: List[str], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Ensure images are stored as uint8
    u8_imgs = [img.squeeze(0).cpu().to(torch.uint8) if img.dim() == 3 else img.cpu().to(torch.uint8) for img in images]
    torch.save({
        "images": u8_imgs,
        "texts": texts,
        "count": len(images),
    }, out_path)


def convert_manifest_to_shards(
    manifest_csv: Path,
    out_dir: Path,
    split_name: str,
    max_per_shard: int = 10000,
) -> List[Path]:
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
            shard_imgs.append(torch.from_numpy(arr))
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


def get_current_rss_mb() -> float:
    """Returns current process RSS memory in MB."""
    import psutil
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)
