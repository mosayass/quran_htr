"""
scripts/pack_captured_shards.py
Converts captured handwriting exports (manifest.csv + PNG images from capture/index.html)
into in-memory .pt shards ready for evaluate.py and train.py.

Supports writer-level dev/test split assignment via --split_config.
Rule: Split assignment is strictly at the writer level, NEVER per sample.
"""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from PIL import Image
import torch

from vocab import Vocabulary
from dataset import TARGET_HEIGHT
from shards import write_shard_file


def load_split_config(config_path: Path) -> Dict[str, str]:
    """Loads YAML mapping of writer_id -> split (e.g. 'dev', 'test')."""
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if "writers" in data:
        return {str(k): str(v).lower() for k, v in data["writers"].items()}
    # Fallback to direct mapping
    return {str(k): str(v).lower() for k, v in data.items()}


def pack_captured_data(
    capture_dir: Path,
    out_shard_path: Path,
    vocab: Vocabulary,
    filter_writer_ids: Optional[List[str]] = None,
) -> Optional[Path]:
    manifest_csv = capture_dir / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest.csv in {capture_dir}")

    images, texts = [], []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wid = row.get("writer_id", "writer_01").strip() or "writer_01"
            if filter_writer_ids is not None and wid not in filter_writer_ids:
                continue

            img_file = row.get("image_file") or f"{row['sample_id']}.png"
            img_path = capture_dir / img_file
            if not img_path.exists():
                print(f"  [warn] Missing image file: {img_path}")
                continue

            img = Image.open(img_path).convert("L")
            w, h = img.size
            if h != TARGET_HEIGHT:
                new_w = max(1, round(w * (TARGET_HEIGHT / h)))
                img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)

            text = row["transcription"]
            try:
                vocab.encode(text)
            except KeyError as e:
                print(f"  [warn] Skipping sample '{row['sample_id']}': contains unencodable character {e}")
                continue

            arr = np.array(img, dtype=np.uint8)
            images.append(torch.from_numpy(arr))
            texts.append(text)

    if not images:
        print(f"[pack_captured_shards] No valid samples found for {filter_writer_ids or 'all'}.")
        return None

    out_shard_path.parent.mkdir(parents=True, exist_ok=True)
    write_shard_file(images, texts, out_shard_path)
    print(f"[pack_captured_shards] Packed {len(images)} samples into {out_shard_path} ({out_shard_path.stat().st_size / (1024*1024):.2f} MB)")
    return out_shard_path


def pack_with_writer_splits(
    capture_dir: Path,
    out_dir: Path,
    split_config_path: Path,
    vocab: Vocabulary,
):
    """
    Packs captured data into split shards based strictly on writer_id assignments in split_config.
    Asserts writer-level isolation (no writer split across sets).
    """
    writer_to_split = load_split_config(split_config_path)
    print(f"[pack_with_writer_splits] Loaded split config from {split_config_path}: {writer_to_split}")

    manifest_csv = capture_dir / "manifest.csv"
    with open(manifest_csv, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Check writer assignment
    split_to_writers: Dict[str, set] = {}
    for r in rows:
        wid = r.get("writer_id", "writer_01").strip() or "writer_01"
        if wid not in writer_to_split:
            raise ValueError(f"Writer '{wid}' not found in split config {split_config_path}!")
        split_name = writer_to_split[wid]
        split_to_writers.setdefault(split_name, set()).add(wid)

    # Ensure writer exclusivity across splits
    seen_writers = set()
    for s_name, writers in split_to_writers.items():
        overlap = seen_writers.intersection(writers)
        if overlap:
            raise RuntimeError(f"Writer leakage detected! Writers {overlap} appear in multiple splits.")
        seen_writers.update(writers)

    out_dir.mkdir(parents=True, exist_ok=True)
    for split_name, writers in split_to_writers.items():
        out_shard = out_dir / f"captured_{split_name}_000.pt"
        pack_captured_data(
            capture_dir=capture_dir,
            out_shard_path=out_shard,
            vocab=vocab,
            filter_writer_ids=list(writers),
        )


def main():
    p = argparse.ArgumentParser(description="Pack captured ink exports into evaluation/training shards")
    p.add_argument("--capture_dir", default="data/capture", help="Directory containing manifest.csv and PNGs")
    p.add_argument("--out_shard", default="data/shards/captured/captured_test_000.pt", help="Single output .pt shard path")
    p.add_argument("--split_config", default=None, help="YAML file with writer-level dev/test assignments")
    p.add_argument("--out_dir", default="data/shards/captured", help="Output directory when using --split_config")
    p.add_argument("--vocab_path", default="vocab.json", help="Path to vocab.json")
    args = p.parse_args()

    vocab = Vocabulary.load(args.vocab_path)
    if args.split_config:
        pack_with_writer_splits(
            capture_dir=Path(args.capture_dir),
            out_dir=Path(args.out_dir),
            split_config_path=Path(args.split_config),
            vocab=vocab,
        )
    else:
        pack_captured_data(Path(args.capture_dir), Path(args.out_shard), vocab)


if __name__ == "__main__":
    main()
