"""
scripts/overfit_ab.py
Task 2.8 Overfit A/B Experiment:
Tests whether CTC alignment fails on Arabic RTL lines when labels are in logical vs visual order.
512 synthetic lines (6-9 words), 500 steps, lr=3e-4:
  Run A: Logical order (unreversed)
  Run B: Visual order (reversed labels text[::-1], decoded output reversed back pred[::-1])
"""

import copy
import sys
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocab import Vocabulary
from model import CRNN, compute_output_seq_len
from decode import greedy_decode
from metrics import corpus_cer_wer
from dataset import collate_fn, Sample, WIDTH_MULTIPLE, TARGET_HEIGHT


class SimpleLineDataset(Dataset):
    def __init__(self, images, texts, vocab, label_order="logical"):
        self.images = images
        self.texts = texts
        self.vocab = vocab
        self.label_order = label_order

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_u8 = self.images[idx]
        text = self.texts[idx]
        tensor = (img_u8.float() / 255.0 - 0.5) / 0.5
        if tensor.dim() == 2:
            tensor = tensor.unsqueeze(0)

        # Pad to WIDTH_MULTIPLE
        cur_w = tensor.shape[-1]
        pad_w = (-cur_w) % WIDTH_MULTIPLE
        if pad_w:
            tensor = nn.functional.pad(tensor, (0, pad_w), value=1.0)

        label_text = text[::-1] if self.label_order == "visual" else text
        target = torch.tensor(self.vocab.encode(label_text), dtype=torch.long)
        return Sample(image=tensor, target=target, text=text)


def run_experiment(name, dataset, base_model, vocab, device, steps=500, lr=3e-4, batch_size=16):
    print(f"\n{'='*60}\nStarting {name} (500 steps, lr={lr}, batch_size={batch_size})...", flush=True)
    model = copy.deepcopy(base_model).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ctc_loss = nn.CTCLoss(blank=vocab.blank_id, zero_infinity=True)

    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    step = 0
    t0 = time.time()
    results = {}
    steps_to_10 = None

    while step < steps:
        for batch in loader:
            step += 1
            images = batch["images"].to(device)
            targets = batch["targets"].to(device)
            target_lengths = batch["target_lengths"].to(device)

            optimizer.zero_grad()
            log_probs = model(images)
            input_lengths = compute_output_seq_len(batch["input_lengths_px"]).to(device)
            input_lengths = input_lengths.clamp(max=log_probs.shape[0])

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            if step % 50 == 0 or step == steps:
                # Evaluate full 512 samples
                model.eval()
                all_preds = []
                all_targets = []
                eval_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
                with torch.no_grad():
                    for ev_batch in eval_loader:
                        ev_imgs = ev_batch["images"].to(device)
                        lp = model(ev_imgs)
                        raw_preds = greedy_decode(lp, vocab)
                        if dataset.label_order == "visual":
                            preds = [p[::-1] for p in raw_preds]
                        else:
                            preds = raw_preds
                        all_preds.extend(preds)
                        all_targets.extend(ev_batch["texts"])

                cer, wer = corpus_cer_wer(all_preds, all_targets)
                results[step] = (loss.item(), cer, wer)
                print(f"  Step {step:3d}/500: loss={loss.item():.4f} | CER={cer*100:5.2f}% | WER={wer*100:5.2f}% ({time.time()-t0:.1f}s)", flush=True)
                if cer < 0.10 and steps_to_10 is None:
                    steps_to_10 = step
                model.train()

            if step >= steps:
                break

    return results, steps_to_10


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    vocab = Vocabulary.load(ROOT / "vocab.json")
    shard_path = ROOT / "data" / "shards" / "synth" / "synth_train_000.pt"
    data = torch.load(shard_path, map_location="cpu", weights_only=False)

    # Filter 512 lines with 6-9 words
    selected_imgs = []
    selected_txts = []
    for img, txt in zip(data["images"], data["texts"]):
        n_words = len(txt.split())
        if 6 <= n_words <= 9:
            try:
                vocab.encode(txt)
                selected_imgs.append(img)
                selected_txts.append(txt)
            except KeyError:
                continue
        if len(selected_imgs) == 512:
            break

    print(f"Selected {len(selected_imgs)} lines with 6-9 words.", flush=True)

    # Base model with ahcd_backbone if available
    base_model = CRNN(vocab_size=len(vocab), backbone="vgg_lite")
    bb_path = ROOT / "ahcd_backbone.pt"
    if bb_path.exists():
        ckpt_bb = torch.load(bb_path, map_location="cpu")
        base_model.backbone.load_state_dict(ckpt_bb)
        print(f"Loaded {bb_path}", flush=True)

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="both", choices=["both", "a", "b"])
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    res_a, steps_a = {}, None
    res_b, steps_b = {}, None

    if args.mode in ("both", "a"):
        ds_logical = SimpleLineDataset(selected_imgs, selected_txts, vocab, label_order="logical")
        res_a, steps_a = run_experiment("Run A (Logical Order)", ds_logical, base_model, vocab, device, steps=args.steps)

    if args.mode in ("both", "b"):
        ds_visual = SimpleLineDataset(selected_imgs, selected_txts, vocab, label_order="visual")
        res_b, steps_b = run_experiment("Run B (Visual Order)", ds_visual, base_model, vocab, device, steps=args.steps)

    if args.mode == "both":
        print("\n" + "=" * 70, flush=True)
        print("                    OVERFIT A/B COMPARISON", flush=True)
        print("=" * 70, flush=True)
        print(f"{'Step':6s} | {'Run A (Logical) CER':20s} | {'Run B (Visual) CER':20s}", flush=True)
        print("-" * 70, flush=True)
        for st in sorted(res_a.keys()):
            cer_a = res_a[st][1] * 100
            cer_b = res_b.get(st, (0, 1.0))[1] * 100
            print(f"{st:6d} | {cer_a:18.2f}% | {cer_b:18.2f}%", flush=True)
        print("-" * 70, flush=True)
        print(f"Steps to CER < 10%: Run A = {steps_a}, Run B = {steps_b}", flush=True)
        print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
