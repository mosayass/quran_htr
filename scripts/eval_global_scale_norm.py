"""
scripts/eval_global_scale_norm.py
Task 2.27 Item 5: Scale normalization with one global target fill (48px in 64px tensor),
per-sample clip-safe protection, no per-writer tuning.
Compares CER for writer_01 and writer_02 against canonical zoom.
AdaBN is dropped.
"""

from __future__ import annotations
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
from scale_policy import sample_band, clip_bound, ink_extents, canonical_zoom_of


def main():
    vocab = Vocabulary.load(ROOT / "vocab.json")
    model, _ = T.load_base_model(T.find_base_checkpoint(), vocab, "cpu")
    sp = T.load_dev_splits()

    out = {"target_px": 48.0, "tensor_h": 64.0, "writers": {}}
    t0 = time.time()

    for wname, samples in (("writer_01", sp["w1_all"]), ("writer_02", sp["w2_all"])):
        canon_cer = round(T.canonical_cer(model, samples, vocab, "cpu"), 2)
        bh = sample_band(samples[0])[0]

        # Global target fill from writer's median ink extent (48px out of 64px)
        med_h = float(torch.median(torch.tensor([ink_extents(s)[2] for s in samples])))
        s_target = 48.0 / (med_h * 64.0 / bh)

        # Unclamped global target zoom
        cer_unclamped = round(T.records_cer(T.predict_records(
            model, samples, vocab, "cpu", zoom_fn=lambda s, st=s_target: st
        )), 2)

        # Clip-safe global target zoom: each sample clamped to never clip ink
        cer_clip_safe = round(T.records_cer(T.predict_records(
            model, samples, vocab, "cpu", zoom_fn=lambda s, st=s_target: min(st, clip_bound(s))
        )), 2)

        # Count samples where clipping was active
        n_clamped = sum(1 for s in samples if clip_bound(s) < s_target)

        out["writers"][wname] = {
            "n_samples": len(samples),
            "band_height_css": bh,
            "median_ink_extent_css": round(med_h, 2),
            "S_target": round(s_target, 4),
            "canonical_cer": canon_cer,
            "global_target_cer_unclamped": cer_unclamped,
            "global_target_cer_clip_safe": cer_clip_safe,
            "samples_clamped_for_clipping": n_clamped,
            "delta_vs_canonical_clip_safe": round(cer_clip_safe - canon_cer, 2),
        }
        print(f"[{wname}] (n={len(samples)}): Canonical={canon_cer}% | "
              f"Target Fill (48px) Unclamped={cer_unclamped}% | Clip-Safe={cer_clip_safe}% "
              f"(Delta: {out['writers'][wname]['delta_vs_canonical_clip_safe']:+0.2f} pt)")

    out["total_eval_secs"] = round(time.time() - t0, 1)
    rep_p = ROOT / "reports" / "task27_scale_norm.json"
    rep_p.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote scale normalization report -> {rep_p}")


if __name__ == "__main__":
    main()
