"""
scripts/pack_captured_shards.py
Converts captured handwriting exports (manifest.csv + PNG images from capture/index.html)
into an in-memory .pt shard ready for evaluate.py.
"""

from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

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
from dataset import TARGET_HEIGHT, WIDTH_MULTIPLE
from shards import write_shard_file


def pack_captured_data(capture_dir: Path, out_shard_path: Path, vocab: Vocabulary):
    manifest_csv = capture_dir / "manifest.csv"
    if not manifest_csv.exists():
        raise FileNotFoundError(f"Missing manifest.csv in {capture_dir}")

    images, texts = [], []
    with open(manifest_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_file = row.get("image_file") or f"{row['sample_id']}.png"
            img_path = capture_dir / img_file
            if not img_path.exists():
                print(f"  [warn] Missing image file: {img_path}")
                continue

            # Load and verify grayscale 64px
            img = Image.open(img_path).convert("L")
            w, h = img.size
            if h != TARGET_HEIGHT:
                new_w = max(1, round(w * (TARGET_HEIGHT / h)))
                img = img.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)

            text = row["transcription"]
            # Validate vocab encoding
            try:
                vocab.encode(text)
            except KeyError as e:
                print(f"  [warn] Skipping sample '{row['sample_id']}': contains unencodable character {e}")
                continue

            arr = np.array(img, dtype=np.uint8)
            images.append(torch.from_numpy(arr))
            texts.append(text)

    if not images:
        print("[pack_captured_shards] No valid samples found to pack.")
        return None

    out_shard_path.parent.mkdir(parents=True, exist_ok=True)
    write_shard_file(images, texts, out_shard_path)
    print(f"[pack_captured_shards] Packed {len(images)} captured samples into {out_shard_path} ({out_shard_path.stat().st_size / (1024*1024):.2f} MB)")
    return out_shard_path


def main():
    p = argparse.ArgumentParser(description="Pack captured ink exports into evaluation shard")
    p.add_argument("--capture_dir", required=True, help="Directory containing manifest.csv and PNGs")
    p.add_argument("--out_shard", default="data/shards/captured/captured_test_000.pt", help="Output .pt shard path")
    p.add_argument("--vocab_path", default="vocab.json", help="Path to vocab.json")
    args = p.parse_args()

    vocab = Vocabulary.load(args.vocab_path)
    pack_captured_data(Path(args.capture_dir), Path(args.out_shard), vocab)


if __name__ == "__main__":
    main()
