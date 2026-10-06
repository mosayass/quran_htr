"""
scripts/export_onnx.py
Task 2.9 Step 4:
- Export CRNN to ONNX with dynamic width and batch dimensions.
- Embed visual-order metadata.
- Apply INT8 dynamic quantization via onnxruntime.quantization.
- Benchmark single-thread CPU latency on (1, 1, 64, 512) and (1, 1, 64, 1024).
- Evaluate CER delta (PyTorch vs ONNX INT8) on synth test and KHATT test shards.
- Verify pass criteria: delta <= 0.5pt synth, <= 1.0pt KHATT.
"""

from __future__ import annotations
import sys
import os
import time
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
import torch
from torch.utils.data import DataLoader
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from vocab import Vocabulary
from model import CRNN
from shards import ShardDataset
from dataset import collate_fn
from decode import greedy_decode
import hashlib
from metrics import corpus_cer_wer


def export_to_onnx(model: CRNN, onnx_path: Path, order: str = "visual", vocab_path: Path | None = None):
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(1, 1, 64, 512, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
        input_names=["images"],
        output_names=["log_probs"],
        dynamic_axes={
            "images": {0: "batch_size", 3: "width"},
            "log_probs": {0: "time", 1: "batch_size"},
        },
    )

    # Check and add model metadata
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    
    # 1. Label order
    meta_order = onnx_model.metadata_props.add()
    meta_order.key = "label_order"
    meta_order.value = order

    # 2. Vocab hash
    if vocab_path is None:
        vocab_path = Path("vocab.json")
    if Path(vocab_path).exists():
        v_bytes = Path(vocab_path).read_bytes()
        v_hash = hashlib.sha256(v_bytes).hexdigest()
        meta_hash = onnx_model.metadata_props.add()
        meta_hash.key = "vocab_hash"
        meta_hash.value = v_hash

        meta_vfile = onnx_model.metadata_props.add()
        meta_vfile.key = "vocab_file"
        meta_vfile.value = Path(vocab_path).name

    # 3. Contract parameters
    for k, v in [("blank_id", "0"), ("target_height", "64"), ("width_multiple", "32"), ("polarity", "bg_white_ink_black")]:
        m = onnx_model.metadata_props.add()
        m.key = k
        m.value = v

    onnx.save(onnx_model, str(onnx_path))
    print(f"[ONNX] Successfully exported to {onnx_path} (size: {onnx_path.stat().st_size / (1024*1024):.2f} MB)")


def quantize_to_int8(onnx_path: Path, quant_path: Path):
    quant_path.parent.mkdir(parents=True, exist_ok=True)
    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quant_path),
        weight_type=QuantType.QInt8,
    )
    print(f"[ONNX] Successfully quantized to {quant_path} (size: {quant_path.stat().st_size / (1024*1024):.2f} MB)")


def benchmark_cpu_latency(onnx_model_path: Path, shapes=((1, 1, 64, 512), (1, 1, 64, 1024)), runs=50, warmup=10):
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(onnx_model_path), opts, providers=["CPUExecutionProvider"])

    results = {}
    print(f"\n--- Single-Thread CPU Latency Benchmark ({onnx_model_path.name}) ---")
    for shape in shapes:
        dummy = np.random.randn(*shape).astype(np.float32)
        # Warmup
        for _ in range(warmup):
            session.run(None, {"images": dummy})

        # Timed runs
        start = time.perf_counter()
        for _ in range(runs):
            session.run(None, {"images": dummy})
        elapsed = time.perf_counter() - start
        avg_ms = (elapsed / runs) * 1000.0
        fps = runs / elapsed
        results[shape] = avg_ms
        print(f"  Shape {shape[2]}x{shape[3]}: {avg_ms:6.2f} ms ({fps:5.1f} lines/sec)")

    return results


def run_eval_pytorch_and_onnx(
    py_model: CRNN,
    ort_session: ort.InferenceSession,
    shard_path: Path,
    vocab: Vocabulary,
    order: str,
    device: torch.device,
    max_samples: int = 1000,
):
    ds = ShardDataset(shard_path, vocab, augment=False, label_order=order)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)

    py_preds, ort_preds, targets = [], [], []
    count = 0

    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"]
            b_targets = batch["texts"]

            # PyTorch inference
            log_probs_pt = py_model(imgs.to(device))
            batch_py_preds = greedy_decode(log_probs_pt, vocab, label_order=order)
            py_preds.extend(batch_py_preds)

            # ONNX inference
            ort_out = ort_session.run(None, {"images": imgs.numpy()})[0]
            log_probs_ort = torch.from_numpy(ort_out)
            batch_ort_preds = greedy_decode(log_probs_ort, vocab, label_order=order)
            ort_preds.extend(batch_ort_preds)

            targets.extend(b_targets)
            count += len(b_targets)
            if count % 200 == 0 or count >= max_samples:
                print(f"    evaluated {count}/{max_samples} samples...", flush=True)
            if max_samples and count >= max_samples:
                break

    cer_pt, wer_pt = corpus_cer_wer(py_preds, targets)
    cer_ort, wer_ort = corpus_cer_wer(ort_preds, targets)
    delta_cer = (cer_ort - cer_pt) * 100.0

    return cer_pt, cer_ort, delta_cer, wer_pt, wer_ort, len(targets)


def main():
    vocab = Vocabulary.load("vocab.json")
    ckpt_path = Path("checkpoints/best.pt")
    ckpt_real_path = Path("checkpoints/best_real.pt")

    if not ckpt_path.exists():
        print(f"Missing {ckpt_path}")
        sys.exit(1)

    device = torch.device("cpu")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")

    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    onnx_dir = Path("dist/onnx")
    onnx_fp32_path = onnx_dir / "crnn_visual_fp32.onnx"
    onnx_int8_path = onnx_dir / "crnn_visual_int8.onnx"

    print("=" * 60)
    print("STEP 4: ONNX EXPORT & INT8 QUANTIZATION")
    print("=" * 60)

    # 1. Export FP32 ONNX
    export_to_onnx(model, onnx_fp32_path, order=order)

    # 2. Dynamic INT8 Quantization
    quantize_to_int8(onnx_fp32_path, onnx_int8_path)

    # 3. Model Size Comparison
    fp32_size = onnx_fp32_path.stat().st_size / (1024 * 1024)
    int8_size = onnx_int8_path.stat().st_size / (1024 * 1024)
    print(f"\n--- Model Sizes ---")
    print(f"  PyTorch Checkpoint : {ckpt_path.stat().st_size / (1024*1024):.2f} MB")
    print(f"  ONNX FP32 Model    : {fp32_size:.2f} MB")
    print(f"  ONNX INT8 Quantized: {int8_size:.2f} MB ({fp32_size / int8_size:.1f}x compression)")

    # 4. Latency Benchmark
    benchmark_cpu_latency(onnx_int8_path, runs=40, warmup=10)

    # 5. CER Delta Evaluation
    print("\n--- Accuracy & CER Delta Evaluation (PyTorch vs INT8 ONNX) ---")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    session_int8 = ort.InferenceSession(str(onnx_int8_path), opts, providers=["CPUExecutionProvider"])

    # Synth test
    synth_shard = Path("data/shards/synth/synth_test_000.pt")
    pt_cer_s, ort_cer_s, delta_s, _, _, n_s = run_eval_pytorch_and_onnx(
        model, session_int8, synth_shard, vocab, order, device, max_samples=1000
    )
    pass_synth = abs(delta_s) <= 0.5
    print(f"  [Synth Test  ({n_s:4d} lines)] PyTorch CER: {pt_cer_s*100:5.2f}% | INT8 ONNX CER: {ort_cer_s*100:5.2f}% | Delta: {delta_s:+5.2f} pt -> {'PASS' if pass_synth else 'FAIL'}")

    # KHATT test (using best_real model)
    ckpt_real = torch.load(ckpt_real_path, map_location=device, weights_only=False)
    model_real = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model_real.load_state_dict(ckpt_real["model"])
    model_real.eval()

    onnx_real_fp32 = onnx_dir / "crnn_real_visual_fp32.onnx"
    onnx_real_int8 = onnx_dir / "crnn_real_visual_int8.onnx"
    export_to_onnx(model_real, onnx_real_fp32, order=order)
    quantize_to_int8(onnx_real_fp32, onnx_real_int8)
    session_real_int8 = ort.InferenceSession(str(onnx_real_int8), opts, providers=["CPUExecutionProvider"])

    khatt_shard = Path("data/shards/khatt/khatt_test_000.pt")
    pt_cer_k, ort_cer_k, delta_k, _, _, n_k = run_eval_pytorch_and_onnx(
        model_real, session_real_int8, khatt_shard, vocab, order, device, max_samples=1124
    )
    pass_khatt = abs(delta_k) <= 1.0
    print(f"  [KHATT Test  ({n_k:4d} lines)] PyTorch CER: {pt_cer_k*100:5.2f}% | INT8 ONNX CER: {ort_cer_k*100:5.2f}% | Delta: {delta_k:+5.2f} pt -> {'PASS' if pass_khatt else 'FAIL'}")


if __name__ == "__main__":
    main()
