"""
scripts/measure_gate_baselines.py
Independent Step-0 gate baselines (Task 2.25 item 1).

For every test subset used by the fold trainer (A_test = writer_02 145, B_test = writer_01 100,
C_test = Fold C held-out groups, LC_test = writer_02 held-out groups) this evaluates the UNTRAINED base
checkpoint with the canonical eval (canonical zoom, greedy decode, corpus CER), `--repeats` times per
device, and records mean + spread (run-to-run nondeterminism). When both CPU and CUDA are available it
also records the cross-device difference. Results are merged into configs/gate_baselines.json; the
trainer's gate then requires |CER - baseline| <= 1.0 pt on the SAME device.

Optionally (--with_forgetting) records the base model's KHATT-val and synthetic-test CER per device,
the reference for the post-fine-tune forgetting comparison.
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from train_twofold_lowo import (GATE_BASELINE_PATH, GATE_TOL, assert_no_locked_test, canonical_cer, eval_shard_cer,
                                find_base_checkpoint, gate_subsets, generate_all_group_splits, load_base_model,
                                load_dev_splits)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", "--devices", dest="devices", nargs="+", default=["auto"], help="auto = every available device (cpu, cuda)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--base_checkpoint", default=None)
    ap.add_argument("--with_forgetting", action="store_true")
    ap.add_argument("--forget_limit", type=int, default=0)
    ap.add_argument("--khatt_val", default="data/shards/khatt/khatt_val_000.pt")
    ap.add_argument("--synth_test", default="data/shards/synth/synth_test_000.pt")
    ap.add_argument("--out", default=str(GATE_BASELINE_PATH))
    args = ap.parse_args()
    assert_no_locked_test(args.out)

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else []) if args.devices == ["auto"] else args.devices
    generate_all_group_splits()
    vocab = Vocabulary.load(ROOT / "vocab.json")
    ckpt = find_base_checkpoint(args.base_checkpoint)
    assert ckpt is not None, "base checkpoint not found"
    subsets = gate_subsets(load_dev_splits())

    out_path = Path(args.out)
    data = json.load(open(out_path, encoding="utf-8")) if out_path.exists() else {}
    data.setdefault("devices", {})
    data["tolerance_pt"] = GATE_TOL
    data["checkpoint"] = str(ckpt.name)
    data["note"] = "Independent canonical-eval baselines of the UNTRAINED base checkpoint, per device."

    for dev in devices:
        print(f"\n=== device: {dev} ===")
        model, _ = load_base_model(ckpt, vocab, dev)
        entry = {}
        for name, samples in subsets.items():
            vals = [round(canonical_cer(model, samples, vocab, dev), 4) for _ in range(args.repeats)]
            entry[name] = {"cer": round(sum(vals) / len(vals), 4), "repeats": vals,
                           "spread": round(max(vals) - min(vals), 4), "n": len(samples)}
            print(f"  {name:8s} n={len(samples):3d} CER={entry[name]['cer']:.2f}%  repeats={vals}  spread={entry[name]['spread']:.4f}")
        if args.with_forgetting:
            for key, p in (("khatt_val_cer", args.khatt_val), ("synth_test_cer", args.synth_test)):
                path = Path(p) if Path(p).is_absolute() else ROOT / p
                entry[key] = {"cer": eval_shard_cer(model, path, vocab, dev, limit=args.forget_limit),
                              "limit": args.forget_limit}
                print(f"  {key}: {entry[key]['cer']:.3f}% (limit={args.forget_limit or 'full'})")
        data["devices"][dev] = entry

    if "cpu" in data["devices"] and "cuda" in data["devices"]:
        diffs = {k: round(data["devices"]["cuda"][k]["cer"] - data["devices"]["cpu"][k]["cer"], 4)
                 for k in subsets if k in data["devices"]["cpu"] and k in data["devices"]["cuda"]}
        data["cross_device_diff_cuda_minus_cpu"] = diffs
        print(f"\nCross-device (CUDA - CPU) CER diff per subset: {diffs}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
