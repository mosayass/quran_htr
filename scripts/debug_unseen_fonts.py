"""
scripts/debug_unseen_fonts.py
Task 2.10 Item 2:
- Per-font CER for the unseen-font set (Zain vs Mirza)
- Render 8 failure samples for the ل->ش confusions into data/debug/
"""

from __future__ import annotations
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
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader
import Levenshtein

from vocab import Vocabulary
from model import CRNN
from shards import ShardDataset
from dataset import collate_fn, TARGET_HEIGHT, WIDTH_MULTIPLE
from decode import greedy_decode
from metrics import corpus_cer_wer


def main():
    device = torch.device("cpu")
    vocab = Vocabulary.load("vocab.json")
    ckpt_path = Path("checkpoints/best.pt")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")

    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    shard_path = Path("data/shards/synth/synth_test_unseen_fonts_000.pt")
    raw_data = torch.load(shard_path, map_location="cpu", weights_only=False)
    raw_imgs = raw_data["images"]
    raw_txts = raw_data["texts"]
    raw_fonts = raw_data["fonts"]

    font_data = {"Mirza": {"preds": [], "targets": []}, "Zain": {"preds": [], "targets": []}}
    lam_sheen_samples = []

    print("Evaluating 2,000 unseen font lines (1,000 Mirza, 1,000 Zain)...", flush=True)

    with torch.no_grad():
        for i in range(0, len(raw_imgs), 32):
            batch_slice = slice(i, min(i + 32, len(raw_imgs)))
            batch_imgs = raw_imgs[batch_slice]
            batch_txts = raw_txts[batch_slice]
            batch_fonts = raw_fonts[batch_slice]

            # Prepare tensor batch
            tensors = []
            for img, text in zip(batch_imgs, batch_txts):
                if img.dim() == 3 and img.shape[0] == 1:
                    img = img.squeeze(0)
                tensor = (img.float() / 255.0 - 0.5) / 0.5
                tensor = tensor.unsqueeze(0)
                cur_w = tensor.shape[-1]
                pad_w = (-cur_w) % WIDTH_MULTIPLE
                if pad_w:
                    tensor = torch.nn.functional.pad(tensor, (0, pad_w), value=1.0)
                tensors.append(tensor)

            # Pad width across batch
            max_w = max(t.shape[-1] for t in tensors)
            padded = [torch.nn.functional.pad(t, (0, max_w - t.shape[-1]), value=1.0) for t in tensors]
            batch_tensor = torch.stack(padded, dim=0)

            log_probs = model(batch_tensor)
            preds = greedy_decode(log_probs, vocab, label_order=order)

            for p, t, f_name, raw_img in zip(preds, batch_txts, batch_fonts, batch_imgs):
                font_data[f_name]["preds"].append(p)
                font_data[f_name]["targets"].append(t)

                # Check if this sample has ل -> ش confusion
                ops = Levenshtein.opcodes(t, p)
                has_lam_sheen = False
                for tag, i1, i2, j1, j2 in ops:
                    if tag == "replace":
                        sub_t = t[i1:i2]
                        sub_p = p[j1:j2]
                        for c_t, c_p in zip(sub_t, sub_p):
                            if c_t == "ل" and c_p == "ش":
                                has_lam_sheen = True
                                break
                    if has_lam_sheen:
                        break

                if has_lam_sheen and len(lam_sheen_samples) < 8:
                    lam_sheen_samples.append({
                        "font": f_name,
                        "target": t,
                        "pred": p,
                        "image": raw_img.numpy(),
                    })

    print("\n" + "=" * 50)
    print("PER-FONT RESULTS ON UNSEEN FONTS")
    print("=" * 50)
    for f_name, data in font_data.items():
        cer, wer = corpus_cer_wer(data["preds"], data["targets"])
        print(f"  Font: {f_name:<8} ({len(data['targets']):4d} lines) | CER: {cer*100:.2f}% | WER: {wer*100:.2f}%")

    # Render debug samples to data/debug/
    debug_dir = Path("data/debug")
    debug_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRendering {len(lam_sheen_samples)} ل->ش failure samples into {debug_dir}...")

    manifest_lines = ["index,font,target,pred,image_file"]
    for idx, s in enumerate(lam_sheen_samples, 1):
        arr = s["image"]
        img = Image.fromarray(arr, mode="L")
        out_name = f"lam_to_sheen_{idx:02d}_{s['font']}.png"
        out_path = debug_dir / out_name
        img.save(out_path)
        manifest_lines.append(f"{idx},{s['font']},\"{s['target']}\",\"{s['pred']}\",{out_name}")
        print(f"  [{idx}] {out_name}:")
        print(f"      Font  : {s['font']}")
        print(f"      Target: {s['target']}")
        print(f"      Pred  : {s['pred']}")

    (debug_dir / "failures_manifest.csv").write_text("\n".join(manifest_lines), encoding="utf-8")
    print(f"Saved manifest to {debug_dir / 'failures_manifest.csv'}")


if __name__ == "__main__":
    main()
