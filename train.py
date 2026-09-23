"""
train.py
Step 1 training loop: CRNN + CTC on KHATT.

Every path (manifests, vocab, checkpoint dir) is a CLI argument -- nothing
is hardcoded -- so this runs unmodified locally (tiny manifest, CPU,
--max_steps for a fast smoke test) and on Colab (full manifest, GPU,
Drive-mounted paths). See README.md's "Local testing before Colab"
section for the recommended pre-Colab checklist.

Usage:
    python train.py \
        --train_manifest /path/train.csv \
        --val_manifest /path/val.csv \
        --vocab_path /path/vocab.json \
        --checkpoint_dir /path/checkpoints \
        --epochs 50 --batch_size 16

Resuming:
    python train.py ... --resume /path/checkpoints/last.pt
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from vocab import Vocabulary
from dataset import LineImageDataset, collate_fn
from model import CRNN, compute_output_seq_len
from decode import greedy_decode
from metrics import corpus_cer_wer


def build_dataloaders(args, vocab: Vocabulary):
    train_ds = LineImageDataset(args.train_manifest, vocab, augment=True)
    val_ds = LineImageDataset(args.val_manifest, vocab)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn,
    )
    return train_loader, val_loader


def save_checkpoint(path, model, optimizer, scheduler, epoch, best_cer):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "best_cer": best_cer,
    }, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt.get("epoch", 0), ckpt.get("best_cer", float("inf"))


def train_one_epoch(model, loader, optimizer, ctc_loss, device, grad_clip, max_steps=None):
    model.train()
    total_loss, n_batches = 0.0, 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        optimizer.zero_grad()
        log_probs = model(images)  # (T, B, V)

        input_lengths = compute_output_seq_len(batch["input_lengths_px"]).to(device)
        # CTC requires input_lengths <= actual model output length T; clamp
        # defensively in case width-downsample rounding ever pushes one over.
        input_lengths = input_lengths.clamp(max=log_probs.shape[0])

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def validate(model, loader, ctc_loss, vocab, device, max_steps=None):
    model.eval()
    total_loss, n_batches = 0.0, 0
    all_preds, all_targets = [], []
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        images = batch["images"].to(device)
        targets = batch["targets"].to(device)
        target_lengths = batch["target_lengths"].to(device)

        log_probs = model(images)
        input_lengths = compute_output_seq_len(batch["input_lengths_px"]).to(device)
        input_lengths = input_lengths.clamp(max=log_probs.shape[0])

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item()
        n_batches += 1

        # Validation uses greedy decoding for speed -- beam_search_decode
        # (decode.py) is for final evaluation / spot checks, not every epoch.
        preds = greedy_decode(log_probs, vocab)
        all_preds.extend(preds)
        all_targets.extend(batch["texts"])

    avg_loss = total_loss / max(n_batches, 1)
    cer_val, wer_val = corpus_cer_wer(all_preds, all_targets) if all_preds else (1.0, 1.0)
    samples = list(zip(all_preds[:5], all_targets[:5]))
    return avg_loss, cer_val, wer_val, samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_manifest", required=True)
    p.add_argument("--val_manifest", required=True)
    p.add_argument("--vocab_path", required=True, help="frozen vocab.json (Vocabulary.save() output)")
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--backbone", default="vgg_lite", choices=["vgg_lite", "mobilenetv3_small"])
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--resume", default=None, help="checkpoint path to resume from")
    p.add_argument("--max_steps", type=int, default=None,
                    help="cap train/val steps per epoch -- for local smoke tests, not real training")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    vocab = Vocabulary.load(args.vocab_path)
    train_loader, val_loader = build_dataloaders(args, vocab)

    model = CRNN(vocab_size=len(vocab), backbone=args.backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    ctc_loss = torch.nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)

    start_epoch, best_cer = 0, float("inf")
    if args.resume:
        start_epoch, best_cer = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device)
        print(f"Resumed from {args.resume} at epoch {start_epoch}, best_cer={best_cer:.4f}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, ctc_loss, device, args.grad_clip, max_steps=args.max_steps
        )
        val_loss, val_cer, val_wer, samples = validate(
            model, val_loader, ctc_loss, vocab, device, max_steps=args.max_steps
        )
        scheduler.step(val_cer)

        print(f"[epoch {epoch+1}/{args.epochs}] "
              f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"val_CER={val_cer:.4f} val_WER={val_wer:.4f} "
              f"({time.time()-t0:.1f}s)")
        for pred, target in samples:
            print(f"    pred:   {pred}\n    target: {target}")

        save_checkpoint(Path(args.checkpoint_dir) / "last.pt", model, optimizer, scheduler, epoch + 1, best_cer)
        if val_cer < best_cer:
            best_cer = val_cer
            save_checkpoint(Path(args.checkpoint_dir) / "best.pt", model, optimizer, scheduler, epoch + 1, best_cer)
            print(f"    ** new best CER: {best_cer:.4f} -- saved best.pt **")


if __name__ == "__main__":
    main()