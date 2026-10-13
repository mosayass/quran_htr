"""
scripts/label_free_adaptation.py
Task 2.25 item 4: label-free adaptation with NO training, dev writers only.

Conditions per writer (writer_01, writer_02; the locked-test writers are never touched):
  baseline       canonical zoom, original BatchNorm statistics
  adabn          BatchNorm running statistics recomputed on the writer's UNLABELED samples
                 (cumulative average, one sample at a time, no gradients, no labels)
  scale          writer-level global-target scale normalization (scripts/scale_policy.py), ink never clipped
  scale+adabn    both
Also reports empirical clipping counts (ink touching the top/bottom row) under the canonical and normalized zoom.
"""

from __future__ import annotations
import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from render_strokes import render_strokes
from scale_policy import writer_scale, canonical_zoom_of, sample_band, is_clipped
from train_twofold_lowo import (find_base_checkpoint, generate_all_group_splits, get_canonical_zoom_and_render,
                                load_base_model, load_dev_splits, predict_records, records_cer, assert_no_locked_test)


def adapt_bn(model: nn.Module, samples, device: str, zoom_fn=None) -> nn.Module:
    """AdaBN: recompute BatchNorm running mean/var from the writer's unlabeled inputs. No labels, no training."""
    m = copy.deepcopy(model).to(device)
    bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d)]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None  # cumulative moving average == plain mean over samples
    m.eval()
    for b in bns:
        b.train()
    with torch.no_grad():
        for s in samples:
            zo = zoom_fn(s) if zoom_fn else None
            m(get_canonical_zoom_and_render(s, augment=False, device=device, zoom_override=zo))
    m.eval()
    return m


def clip_count(samples, zoom_fn) -> int:
    n = 0
    for s in samples:
        bh, cy, bt = sample_band(s)
        z = zoom_fn(s)
        img = render_strokes(s["strokes"], zoom=z, width_after_zoom=2.2, center_y=cy, band_top_y=bt, band_height=bh, pad_x=16.0)
        n += is_clipped(np.array(img))
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_checkpoint", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="reports/label_free_adaptation.json")
    args = ap.parse_args()
    assert_no_locked_test(args.out)

    generate_all_group_splits()
    vocab = Vocabulary.load(ROOT / "vocab.json")
    ckpt = find_base_checkpoint(args.base_checkpoint)
    model, _ = load_base_model(ckpt, vocab, args.device)
    sp = load_dev_splits()
    writers = {"writer_01": sp["w1_all"], "writer_02": sp["w2_all"]}

    out = {"checkpoint": ckpt.name, "device": args.device, "writers": {}}
    for wname, samples in writers.items():
        pol = writer_scale(samples)
        canon_fn = canonical_zoom_of
        norm_fn = lambda s, z=pol["S_w"]: z
        rows = {}
        rows["baseline"] = records_cer(predict_records(model, samples, vocab, args.device))
        m_bn = adapt_bn(model, samples, args.device)
        rows["adabn"] = records_cer(predict_records(m_bn, samples, vocab, args.device))
        rows["scale"] = records_cer(predict_records(model, samples, vocab, args.device, zoom_fn=norm_fn))
        m_bn_s = adapt_bn(model, samples, args.device, zoom_fn=norm_fn)
        rows["scale+adabn"] = records_cer(predict_records(m_bn_s, samples, vocab, args.device, zoom_fn=norm_fn))
        clip = {"canonical": clip_count(samples, canon_fn), "normalized": clip_count(samples, norm_fn)}
        out["writers"][wname] = {"n": len(samples), "scale_policy": pol,
                                 "canonical_zoom": canonical_zoom_of(samples[0]),
                                 "cer": {k: round(v, 2) for k, v in rows.items()}, "clipped_samples": clip}
        print(f"\n{wname} (n={len(samples)}): S_w={pol['S_w']:.3f} (target {pol['S_target']:.3f}, "
              f"formula58 {pol['S_formula58']:.3f}, exact-clip {pol['S_clip_exact']:.3f}; canonical {canonical_zoom_of(samples[0]):.3f})")
        for k, v in rows.items():
            print(f"  {k:12s} CER {v:6.2f}%  (delta vs baseline {v - rows['baseline']:+.2f})")
        print(f"  clipped samples: canonical={clip['canonical']} normalized={clip['normalized']}")

    p = ROOT / args.out
    p.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(p, "w", encoding="utf-8"), indent=2)
    print(f"\nWrote {p}")


if __name__ == "__main__":
    main()
