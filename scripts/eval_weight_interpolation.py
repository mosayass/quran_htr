"""
scripts/eval_weight_interpolation.py
Task 2.27 Item 4: Eval-only weight interpolation between the base checkpoint
and a stroke-only fine-tuned checkpoint:
    W_alpha = (1 - alpha) * W_base + alpha * W_ft
for alpha in {0.3, 0.5, 0.7}.

Evaluates:
  1. Fold C held-out test CER (C_test, n=76)
  2. Synthetic test CER (shards/synth/synth_test_000.pt)
  3. KHATT validation CER (shards/khatt/khatt_val_000.pt)
"""

from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
import train_twofold_lowo as T


def interpolate_weights(base_state: dict, ft_state: dict, alpha: float) -> dict:
    interp = {}
    for k, w_base in base_state.items():
        if k in ft_state and ft_state[k].shape == w_base.shape:
            if w_base.dtype.is_floating_point:
                interp[k] = (1.0 - alpha) * w_base + alpha * ft_state[k]
            else:
                interp[k] = copy.deepcopy(w_base)
        else:
            interp[k] = copy.deepcopy(w_base)
    return interp


def main():
    ap = argparse.ArgumentParser(description="Eval-only weight interpolation benchmark")
    ap.add_argument("--base_checkpoint", default="checkpoints/best.pt")
    ap.add_argument("--ft_checkpoint", required=True, help="Stroke-only fine-tuned checkpoint")
    ap.add_argument("--alphas", nargs="+", type=float, default=[0.0, 0.3, 0.5, 0.7, 1.0])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--synth_test", default="data/shards/synth/synth_test_000.pt")
    ap.add_argument("--khatt_val", default="data/shards/khatt/khatt_val_000.pt")
    ap.add_argument("--forget_limit", type=int, default=0)
    ap.add_argument("--out", default="reports/weight_interpolation.json")
    args = ap.parse_args()

    for p in [args.base_checkpoint, args.ft_checkpoint, args.synth_test, args.khatt_val]:
        T.assert_no_locked_test(p)

    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    vocab = Vocabulary.load(ROOT / "vocab.json")
    sp = T.load_dev_splits()
    c_test = sp["fc_te"]

    base_p = Path(args.base_checkpoint)
    ft_p = Path(args.ft_checkpoint)
    assert base_p.exists(), f"Base checkpoint not found: {base_p}"
    assert ft_p.exists(), f"Fine-tuned checkpoint not found: {ft_p}"

    base_raw = torch.load(base_p, map_location="cpu", weights_only=False)
    base_st = base_raw.get("model", base_raw.get("model_state", base_raw))
    ft_raw = torch.load(ft_p, map_location="cpu", weights_only=False)
    ft_st = ft_raw.get("model", ft_raw.get("model_state", ft_raw))

    model = T.CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)

    forget_paths = {"synth_test_cer": Path(args.synth_test), "khatt_val_cer": Path(args.khatt_val)}
    T.ensure_shards_available(list(forget_paths.values()))

    results = []
    print(f"\nEvaluating weight interpolation (alphas={args.alphas}) on device={device}...")
    for alpha in args.alphas:
        t0 = time.time()
        w_interp = interpolate_weights(base_st, ft_st, alpha)
        model.load_state_dict(w_interp, strict=True)
        model.eval()

        c_cer = T.canonical_cer(model, c_test, vocab, device)
        forgetting = {}
        for name, p in forget_paths.items():
            if p.exists():
                forgetting[name] = T.eval_shard_cer(model, p, vocab, device, limit=args.forget_limit)
            else:
                forgetting[name] = None

        row = {
            "alpha": round(alpha, 3),
            "c_test_cer": round(c_cer, 2),
            "synth_test_cer": forgetting.get("synth_test_cer"),
            "khatt_val_cer": forgetting.get("khatt_val_cer"),
            "eval_secs": round(time.time() - t0, 1),
        }
        results.append(row)
        print(f"  alpha={alpha:0.2f} -> C_test CER: {row['c_test_cer']}% | "
              f"synth: {row['synth_test_cer']}% | khatt: {row['khatt_val_cer']}% ({row['eval_secs']}s)")

    out_p = Path(args.out)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        json.dump({"base": str(base_p), "ft": str(ft_p), "device": device, "results": results}, f, indent=2)
    print(f"\nWrote results -> {out_p}")


if __name__ == "__main__":
    main()
