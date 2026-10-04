"""
train.py
Step 1 & Step 2 training loop: CRNN + CTC on KHATT & Synthetic Quran lines.

Features:
- Configurable training input: single source (--train_manifest) or repeatable
  weighted multi-source mixing (--mix MANIFEST:WEIGHT). Supports both CSV manifests
  and in-memory uint8 .pt ShardDataset shards.
- Model warm-start: --init_weights PATH loads model weights only, resetting
  optimizer, scheduler, and epoch (distinct from --resume).
- Configurable epoch size: --samples_per_epoch (default 40,000) for fixed-step epochs.
- Fine-tune defaults: lr 1e-4, patience 5, min_lr 1e-5.
- Dual validation: reports both synthetic val (held-out surahs) and KHATT val
  (real ink, held-out writers) each epoch.
- Dual checkpointing: saves best.pt on synthetic val CER and best_real.pt on KHATT val CER.
"""

from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Union

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import torch
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler, RandomSampler, Dataset

from vocab import Vocabulary
from dataset import LineImageDataset, collate_fn
from model import CRNN, compute_output_seq_len
from decode import greedy_decode
from metrics import corpus_cer_wer
from shards import ShardDataset


def load_dataset_source(path_str: str, vocab: Vocabulary, augment: bool = False) -> Dataset:
    """Loads either a ShardDataset (if path is a .pt file or directory with .pt files) or LineImageDataset."""
    p = Path(path_str)
    if p.suffix.lower() == ".pt" or (p.is_dir() and any(p.glob("*.pt"))):
        return ShardDataset(p, vocab, augment=augment)
    return LineImageDataset(path_str, vocab, augment=augment)


def build_dataloaders(args, vocab: Vocabulary):
    # 1. Training loader
    if args.mix:
        datasets: List[Dataset] = []
        weights: List[float] = []
        print("[build_dataloaders] Setting up weighted multi-source mixing:")
        for item in args.mix:
            if ":" not in item:
                raise ValueError(
                    f"Invalid --mix argument '{item}'. Format must be SOURCE_PATH:WEIGHT (e.g. data/shards/synth:0.75)"
                )
            m_path, w_str = item.rsplit(":", 1)
            w = float(w_str)
            ds = load_dataset_source(m_path, vocab, augment=True)
            if len(ds) == 0:
                print(f"  [warn] Source {m_path} has 0 samples; skipping.")
                continue
            datasets.append(ds)
            weights.append(w)
            print(f"  * {m_path}: {len(ds)} samples, weight={w}")

        if not datasets:
            raise RuntimeError("No valid datasets loaded from --mix arguments.")

        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]
        sample_weights = []
        for ds, nw in zip(datasets, norm_weights):
            per_sample_w = nw / len(ds)
            sample_weights.extend([per_sample_w] * len(ds))

        concat_ds = ConcatDataset(datasets)
        num_samples = args.samples_per_epoch if args.samples_per_epoch is not None else len(concat_ds)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=num_samples,
            replacement=True,
        )
        train_loader = DataLoader(
            concat_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
            drop_last=True,
        )
    else:
        if not args.train_manifest:
            raise ValueError("Either --train_manifest or repeatable --mix SOURCE:WEIGHT must be specified.")
        train_ds = load_dataset_source(args.train_manifest, vocab, augment=True)
        if args.samples_per_epoch is not None:
            sampler = RandomSampler(train_ds, replacement=True, num_samples=args.samples_per_epoch)
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                sampler=sampler,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )
        else:
            train_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=collate_fn,
                drop_last=True,
            )

    # 2. Synthetic (primary) validation loader
    val_loader = None
    if args.val_manifest:
        val_ds = load_dataset_source(args.val_manifest, vocab, augment=False)
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    # 3. Real-ink (KHATT) validation loader
    val_real_loader = None
    if args.val_real_manifest:
        val_real_ds = load_dataset_source(args.val_real_manifest, vocab, augment=False)
        val_real_loader = DataLoader(
            val_real_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    return train_loader, val_loader, val_real_loader


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
    if loader is None:
        return 0.0, float("inf"), float("inf"), []
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

        preds = greedy_decode(log_probs, vocab)
        all_preds.extend(preds)
        all_targets.extend(batch["texts"])

    avg_loss = total_loss / max(n_batches, 1)
    cer_val, wer_val = corpus_cer_wer(all_preds, all_targets) if all_preds else (1.0, 1.0)
    samples = list(zip(all_preds[:5], all_targets[:5]))
    return avg_loss, cer_val, wer_val, samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_manifest", default=None, help="Path to single training manifest CSV or shard")
    p.add_argument("--mix", action="append", default=None,
                   help="Repeatable SOURCE:WEIGHT for weighted sampling (e.g. --mix data/shards/synth:0.75 --mix data/shards/khatt:0.25)")
    p.add_argument("--val_manifest", default=None, help="Synthetic / primary validation source (CSV or shard)")
    p.add_argument("--val_real_manifest", default=None, help="Real-ink (KHATT) validation source (CSV or shard)")
    p.add_argument("--vocab_path", required=True, help="frozen vocab.json (Vocabulary.save() output)")
    p.add_argument("--checkpoint_dir", required=True)
    p.add_argument("--backbone", default="vgg_lite", choices=["vgg_lite", "mobilenetv3_small"])
    p.add_argument("--backbone_pretrained", default=None,
                   help="Path to pretrained CNN backbone weights (e.g. ahcd_backbone.pt)")
    p.add_argument("--init_weights", default=None,
                   help="Path to checkpoint for model weights only (does NOT load optimizer, scheduler, or epoch)")
    p.add_argument("--samples_per_epoch", type=int, default=40000,
                   help="Number of lines sampled per epoch (default: 40000)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4, help="Fine-tuning default learning rate (1e-4)")
    p.add_argument("--patience", type=int, default=5, help="ReduceLROnPlateau patience (default: 5)")
    p.add_argument("--min_lr", type=float, default=1e-5, help="Minimum learning rate (default: 1e-5)")
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--resume", default=None, help="Checkpoint path to resume full state from")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Cap train/val steps per epoch -- for local smoke tests, not real training")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    vocab = Vocabulary.load(args.vocab_path)
    train_loader, val_loader, val_real_loader = build_dataloaders(args, vocab)

    model = CRNN(vocab_size=len(vocab), backbone=args.backbone).to(device)

    # Model weight loading: --init_weights loads model weights only
    if args.init_weights:
        ckpt_init = torch.load(args.init_weights, map_location=device)
        state_dict = ckpt_init.get("model", ckpt_init)
        model.load_state_dict(state_dict)
        print(f"Loaded initial model weights from {args.init_weights} (optimizer, scheduler, and epoch reset)")
    elif args.backbone_pretrained and not args.resume:
        ckpt_bb = torch.load(args.backbone_pretrained, map_location=device)
        model.backbone.load_state_dict(ckpt_bb)
        print(f"Loaded pretrained backbone weights from {args.backbone_pretrained}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience, min_lr=args.min_lr
    )
    ctc_loss = torch.nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)

    start_epoch = 0
    best_cer = float("inf")
    best_real_cer = float("inf")

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
        ) if val_loader else (0.0, float("inf"), float("inf"), [])

        val_real_loss, val_real_cer, val_real_wer, real_samples = validate(
            model, val_real_loader, ctc_loss, vocab, device, max_steps=args.max_steps
        ) if val_real_loader else (0.0, float("inf"), float("inf"), [])

        if val_loader:
            scheduler.step(val_cer)

        curr_lr = optimizer.param_groups[0]["lr"]

        if val_real_loader is not None:
            print(f"[epoch {epoch+1}/{args.epochs}] lr={curr_lr:.2e} "
                  f"train_loss={train_loss:.4f} "
                  f"synth_loss={val_loss:.4f} synth_CER={val_cer:.4f} synth_WER={val_wer:.4f} | "
                  f"real_loss={val_real_loss:.4f} real_CER={val_real_cer:.4f} real_WER={val_real_wer:.4f} "
                  f"({time.time()-t0:.1f}s)")
        else:
            print(f"[epoch {epoch+1}/{args.epochs}] lr={curr_lr:.2e} "
                  f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"val_CER={val_cer:.4f} val_WER={val_wer:.4f} "
                  f"({time.time()-t0:.1f}s)")

        for pred, target in samples[:3]:
            print(f"    [synth] pred:   {pred}\n            target: {target}")
        for pred, target in real_samples[:2]:
            print(f"    [real]  pred:   {pred}\n            target: {target}")

        ckpt_dir = Path(args.checkpoint_dir)
        save_checkpoint(ckpt_dir / "last.pt", model, optimizer, scheduler, epoch + 1, best_cer)

        if val_loader and val_cer < best_cer:
            best_cer = val_cer
            save_checkpoint(ckpt_dir / "best.pt", model, optimizer, scheduler, epoch + 1, best_cer)
            print(f"    ** new best synthetic CER: {best_cer:.4f} -- saved best.pt **")

        if val_real_loader and val_real_cer < best_real_cer:
            best_real_cer = val_real_cer
            save_checkpoint(ckpt_dir / "best_real.pt", model, optimizer, scheduler, epoch + 1, best_real_cer)
            print(f"    ** new best real CER: {best_real_cer:.4f} -- saved best_real.pt **")


if __name__ == "__main__":
    main()