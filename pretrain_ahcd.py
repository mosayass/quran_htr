"""
pretrain_ahcd.py
Character-level CNN warm-start on the Arabic Handwritten Characters Dataset (AHCD).

Pre-trains the VGGLiteBackbone as a 28-class isolated Arabic character classifier
before plugging it into the full line-level CRNN+CTC model. This warms up the
convolutional feature extractors (edges, stroke intersections, loops, and dots)
on real human handwriting without requiring long CTC convergence from scratch.

Dataset source: Kaggle `mloey1/ahcd1` (13,440 train characters, 3,360 test characters).
Each sample is 32x32 grayscale, with 28 classes (Alif through Yaa).

Usage:
    python pretrain_ahcd.py \
        --data_dir /path/to/ahcd \
        --out_path ahcd_backbone.pt \
        --epochs 10 --batch_size 64
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

from model import VGGLiteBackbone


def find_ahcd_files(data_dir: Path) -> Tuple[Path, Path, Path, Path]:
    """
    Locates the 4 CSV files in data_dir (or any subdirectories):
      train_images, train_labels, test_images, test_labels.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

    all_csvs = list(data_dir.rglob("*.csv"))
    if not all_csvs:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    train_img = None
    train_lbl = None
    test_img = None
    test_lbl = None

    for p in all_csvs:
        stem_lower = p.stem.lower()
        if "train" in stem_lower and ("image" in stem_lower or "img" in stem_lower):
            train_img = p
        elif "train" in stem_lower and ("label" in stem_lower or "lbl" in stem_lower):
            train_lbl = p
        elif "test" in stem_lower and ("image" in stem_lower or "img" in stem_lower):
            test_img = p
        elif "test" in stem_lower and ("label" in stem_lower or "lbl" in stem_lower):
            test_lbl = p

    missing = []
    if train_img is None: missing.append("Train Images CSV")
    if train_lbl is None: missing.append("Train Labels CSV")
    if test_img is None: missing.append("Test Images CSV")
    if test_lbl is None: missing.append("Test Labels CSV")

    if missing:
        found_names = [f.name for f in all_csvs]
        raise FileNotFoundError(
            f"Could not locate all AHCD files in {data_dir}. Missing: {missing}. Found: {found_names}"
        )

    return train_img, train_lbl, test_img, test_lbl


def load_ahcd_split(
    img_csv: Path,
    lbl_csv: Path,
    max_samples: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads AHCD CSV data into PyTorch tensors:
      1. Images reshaped to (N, 1, 32, 32) and transposed to correct orientation.
      2. Inverted: AHCD raw has 0=black background, 255=white stroke. We invert
         so that background is white (255) and ink is dark (0), exactly matching
         the KHATT document polarity and dataset.py's normalization.
      3. Interpolated to (N, 1, 64, 64) matching TARGET_HEIGHT=64 of VGGLiteBackbone.
      4. Normalized to [-1.0, 1.0] (white background = +1.0, dark ink = -1.0).
      5. Labels mapped from 1..28 to 0..27.
    """
    print(f"Loading {img_csv.name} and {lbl_csv.name}...")
    # Load labels
    lbls = np.loadtxt(lbl_csv, delimiter=",", dtype=np.int64)
    # Load pixel data (N, 1024)
    imgs = np.loadtxt(img_csv, delimiter=",", dtype=np.float32)

    if max_samples is not None:
        lbls = lbls[:max_samples]
        imgs = imgs[:max_samples]

    n_samples = len(lbls)
    # AHCD was saved in Fortran/column-major order in MATLAB -> transpose axes (0, 2, 1) to upright
    imgs = imgs.reshape(n_samples, 32, 32).transpose(0, 2, 1)

    # Invert polarity so ink is 0 and paper is 255
    imgs = 255.0 - imgs

    # Normalize to [-1.0, 1.0]
    imgs = (imgs / 255.0 - 0.5) / 0.5

    # Convert to Tensor (N, 1, 32, 32)
    img_tensors = torch.from_numpy(imgs).unsqueeze(1)

    # Resize to 64x64 matching the line-image height expected by the backbone
    img_tensors = F.interpolate(img_tensors, size=(64, 64), mode="bilinear", align_corners=False)

    # Convert labels 1..28 -> 0..27
    lbl_tensors = torch.from_numpy(lbls - 1).long()

    return img_tensors, lbl_tensors


class AHCDClassifier(nn.Module):
    """
    Classifier wrapper around VGGLiteBackbone for 28-class Arabic character recognition.
    """
    def __init__(self, num_classes: int = 28):
        super().__init__()
        self.backbone = VGGLiteBackbone()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.backbone.out_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(x)           # (B, 256, 2, 16)
        pooled = self.pool(feats).flatten(1)  # (B, 256)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)   # (B, 28)
        return logits


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(targets)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_count += len(targets)

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for images, targets in loader:
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)

        total_loss += loss.item() * len(targets)
        preds = logits.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_count += len(targets)

    return total_loss / total_count, total_correct / total_count


def main():
    p = argparse.ArgumentParser(description="Pretrain CNN backbone on AHCD isolated characters")
    p.add_argument("--data_dir", required=True, help="Directory containing extracted AHCD CSVs")
    p.add_argument("--out_path", default="ahcd_backbone.pt", help="Path to save backbone weights")
    p.add_argument("--epochs", type=int, default=10, help="Number of pretraining epochs (default: 10)")
    p.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
    p.add_argument("--max_samples", type=int, default=None, help="Cap samples for fast testing")
    p.add_argument("--device", default=None, help="Device to use ('cuda' or 'cpu')")
    args = p.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_img_csv, train_lbl_csv, test_img_csv, test_lbl_csv = find_ahcd_files(Path(args.data_dir))
    print(f"Found AHCD files in {args.data_dir}:")
    print(f"  Train: {train_img_csv.name} | {train_lbl_csv.name}")
    print(f"  Test:  {test_img_csv.name} | {test_lbl_csv.name}")

    x_train, y_train = load_ahcd_split(train_img_csv, train_lbl_csv, max_samples=args.max_samples)
    x_test, y_test = load_ahcd_split(test_img_csv, test_lbl_csv, max_samples=args.max_samples)
    print(f"Loaded train: {x_train.shape} labels: {y_train.shape}")
    print(f"Loaded test:  {x_test.shape} labels: {y_test.shape}")

    train_ds = TensorDataset(x_train, y_train)
    test_ds = TensorDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    model = AHCDClassifier(num_classes=28).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n--- Starting AHCD Warm-Start ({args.epochs} epochs) ---")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, test_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        curr_lr = optimizer.param_groups[0]["lr"]
        print(f"[epoch {epoch+1:02d}/{args.epochs:02d}] lr={curr_lr:.2e} "
              f"train_loss={train_loss:.4f} train_acc={train_acc*100:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc*100:.2f}% ({elapsed:.1f}s)")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.backbone.state_dict(), out_path)
            print(f"  --> Saved new best backbone weights to {out_path} (val_acc={val_acc*100:.2f}%)")

    print(f"\n[DONE] Best AHCD accuracy: {best_acc*100:.2f}%. Backbone saved to {out_path}")
    print(f"You can now supply `--backbone_pretrained {out_path}` to train.py!")


if __name__ == "__main__":
    main()
