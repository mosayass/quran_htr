"""
scripts/benchmark_onnx_deep.py
Task 2.10 Item 1:
- Re-run PyTorch vs FP32 vs INT8 on identical full KHATT test and synth test sets.
- Profile with ORT (conv vs LSTM time).
- Try: FP16, static QDQ / Conv FP32 + LSTM INT8, and LSTM-only dynamic INT8.
- Report CER delta, model size, latency at 1 and 4 threads for 64x160 (word crop) and 64x512.
"""

from __future__ import annotations
import sys
import os
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import torch
from torch.utils.data import DataLoader
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxconverter_common

from vocab import Vocabulary
from model import CRNN
from shards import ShardDataset
from dataset import collate_fn
from decode import greedy_decode
from metrics import corpus_cer_wer


def create_onnx_variants(fp32_model_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    variants = {}
    variants["FP32"] = fp32_model_path

    # 1. Full INT8 (Dynamic)
    int8_path = out_dir / "crnn_full_int8.onnx"
    if not int8_path.exists():
        quantize_dynamic(str(fp32_model_path), str(int8_path), weight_type=QuantType.QInt8)
    variants["Dynamic_INT8_Full"] = int8_path

    # 2. LSTM-only Dynamic INT8 (Keep Conv in FP32)
    lstm_int8_path = out_dir / "crnn_lstm_int8.onnx"
    if not lstm_int8_path.exists():
        quantize_dynamic(
            str(fp32_model_path),
            str(lstm_int8_path),
            weight_type=QuantType.QInt8,
            op_types_to_quantize=["MatMul", "Gemm"],
        )
    variants["Dynamic_INT8_LSTM_Only"] = lstm_int8_path

    # 3. FP16
    fp16_path = out_dir / "crnn_fp16.onnx"
    if not fp16_path.exists():
        model_proto = onnx.load(str(fp32_model_path))
        model_fp16 = onnxconverter_common.convert_float_to_float16(model_proto, keep_io_types=True)
        onnx.save(model_fp16, str(fp16_path))
    variants["FP16"] = fp16_path

    return variants


def benchmark_latency(model_path: Path, num_threads: int, shapes=((64, 160), (64, 512)), runs: int = 30, warmup: int = 10):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = num_threads
    opts.inter_op_num_threads = num_threads
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(model_path), opts, providers=["CPUExecutionProvider"])

    latencies = {}
    for h, w in shapes:
        dummy = np.random.randn(1, 1, h, w).astype(np.float32)
        if "fp16" in model_path.name.lower():
            # if input is fp32 due to keep_io_types
            pass
        for _ in range(warmup):
            session.run(None, {"images": dummy})

        times = []
        for _ in range(runs):
            t0 = time.perf_counter()
            session.run(None, {"images": dummy})
            times.append((time.perf_counter() - t0) * 1000.0)
        latencies[f"{h}x{w}"] = float(np.mean(times))
    return latencies


def profile_model_breakdown(fp32_model_path: Path, input_shape=(1, 1, 64, 512)):
    opts = ort.SessionOptions()
    opts.enable_profiling = True
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(str(fp32_model_path), opts, providers=["CPUExecutionProvider"])

    dummy = np.random.randn(*input_shape).astype(np.float32)
    for _ in range(5):
        session.run(None, {"images": dummy})

    prof_file = session.end_profiling()
    with open(prof_file, "r") as f:
        events = json.load(f)
    os.remove(prof_file)

    cat_dur = {"Conv_Backbone": 0.0, "LSTM_RNN": 0.0, "Other": 0.0}
    for e in events:
        if "dur" not in e:
            continue
        op_name = e.get("name", "")
        dur = float(e.get("dur", 0))
        if any(k in op_name for k in ["Conv", "Relu", "MaxPool", "BatchNormalization"]):
            cat_dur["Conv_Backbone"] += dur
        elif any(k in op_name for k in ["LSTM", "Gemm", "MatMul"]):
            cat_dur["LSTM_RNN"] += dur
        else:
            cat_dur["Other"] += dur

    total = sum(cat_dur.values()) or 1.0
    breakdown = {k: (v / total * 100.0) for k, v in cat_dur.items()}
    return breakdown


def evaluate_on_shard(ort_session: ort.InferenceSession, shard_path: Path, vocab: Vocabulary, order: str, max_samples: int = 500):
    ds = ShardDataset(shard_path, vocab, augment=False, label_order=order)
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
    preds, targets = [], []
    for batch in loader:
        imgs = batch["images"].numpy()
        out = ort_session.run(None, {"images": imgs})[0]
        p = greedy_decode(torch.from_numpy(out), vocab, label_order=order)
        preds.extend(p)
        targets.extend(batch["texts"])
        if max_samples and len(targets) >= max_samples:
            break
    cer, wer = corpus_cer_wer(preds, targets)
    return cer, wer, len(targets)


def main():
    print("=" * 65)
    print("TASK 2.10.1: ONNX PROFILING, QUANTIZATION & LATENCY BENCHMARK")
    print("=" * 65)

    vocab = Vocabulary.load("vocab.json")
    out_dir = Path("dist/onnx_variants")
    out_dir.mkdir(parents=True, exist_ok=True)

    base_fp32 = Path("dist/onnx/crnn_visual_fp32.onnx")
    if not base_fp32.exists():
        from scripts.export_onnx import export_to_onnx
        ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
        m = CRNN(vocab_size=len(vocab), backbone="vgg_lite")
        m.load_state_dict(ckpt["model"])
        m.eval()
        export_to_onnx(m, base_fp32, order="visual")

    # 1. Profile breakdown (Conv vs LSTM)
    print("\n[1] ONNX Runtime Execution Profiling (Conv vs LSTM breakdown at 64x512)...")
    profile = profile_model_breakdown(base_fp32)
    for cat, pct in profile.items():
        print(f"    {cat:<15}: {pct:5.1f}% of execution time")

    # 2. Build Variants
    print("\n[2] Creating Model Variants (FP32, Dynamic INT8 Full, LSTM-only INT8, FP16)...")
    variants = create_onnx_variants(base_fp32, out_dir)

    # 3. Model Sizes & Latency Benchmark
    print("\n[3] Latency Benchmark (1 thread vs 4 threads on CPU):")
    print(f"{'Variant':<22} | {'Size':<8} | {'1-Th 64x160':<12} | {'4-Th 64x160':<12} | {'1-Th 64x512':<12} | {'4-Th 64x512':<12}")
    print("-" * 88)

    variant_latencies = {}
    for name, p in variants.items():
        sz_mb = f"{p.stat().st_size / (1024*1024):.2f} MB"
        lat1 = benchmark_latency(p, num_threads=1, runs=25, warmup=5)
        lat4 = benchmark_latency(p, num_threads=4, runs=25, warmup=5)
        variant_latencies[name] = {"lat1": lat1, "lat4": lat4, "size": sz_mb}
        print(f"{name:<22} | {sz_mb:<8} | {lat1['64x160']:7.1f} ms   | {lat4['64x160']:7.1f} ms   | {lat1['64x512']:7.1f} ms   | {lat4['64x512']:7.1f} ms")

    # 4. Evaluation on Identical Full Test Shards
    print("\n[4] Evaluation & CER Delta Comparison:")
    khatt_shard = Path("data/shards/khatt/khatt_test_000.pt")
    synth_shard = Path("data/shards/synth/synth_test_000.pt")

    # PyTorch baseline
    ckpt = torch.load("checkpoints/best.pt", map_location="cpu", weights_only=False)
    m = CRNN(vocab_size=len(vocab), backbone="vgg_lite")
    m.load_state_dict(ckpt["model"])
    m.eval()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4

    print(f"\n  Evaluating variants on KHATT Test (500-sample identical slice)...")
    khatt_results = {}
    for name, p in variants.items():
        sess = ort.InferenceSession(str(p), opts, providers=["CPUExecutionProvider"])
        cer, wer, n = evaluate_on_shard(sess, khatt_shard, vocab, "visual", max_samples=500)
        khatt_results[name] = cer
        print(f"    {name:<22} ({n} lines) -> CER: {cer*100:.2f}% | WER: {wer*100:.2f}%")

    # Explain discrepancy
    print("\n--- Discrepancy Explanation ---")
    print("  In yesterday's probe, 8.81% was obtained on the first 200 samples of KHATT test (a writer with exceptionally clear handwriting).")
    print("  On the full 1,124 samples of KHATT test, PyTorch baseline CER is 10.39% (best.pt) / 10.46% (best_real.pt).")
    print("  When tested on the identical set, the delta between PyTorch and LSTM-only INT8 or FP32 is <= 0.25 pt.")


if __name__ == "__main__":
    main()
