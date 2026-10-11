"""
scripts/generate_golden_vectors.py
Task 2.19 Item 3:
Generates 20 golden test vectors for Flutter parity testing:
- Reads 20 capture samples from data/capture/strokes.json (10 lines, 10 words)
- Renders 64px image using 4x supersampled stroke renderer
- Computes SHA-256 hash of the normalized image tensor
- Runs inference via dist/onnx/crnn_visual_fp32.onnx using ONNX Runtime
- Runs verify_reference to produce exact per-word verdicts
- Saves tests/golden/golden_vectors.json
"""

from __future__ import annotations
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from render_strokes import render_strokes
from verify_ref import verify_reference


def generate_golden_suite():
    vocab_p = ROOT / "vocab.json"
    vocab = Vocabulary.load(vocab_p)
    onnx_p = ROOT / "dist" / "onnx" / "crnn_visual_fp32.onnx"
    strokes_p = ROOT / "data" / "capture" / "strokes.json"

    if not onnx_p.exists():
        raise FileNotFoundError(f"Missing ONNX model: {onnx_p}")
    if not strokes_p.exists():
        raise FileNotFoundError(f"Missing strokes file: {strokes_p}")

    print(f"Loading ONNX model from {onnx_p}...")
    session = ort.InferenceSession(str(onnx_p), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    with open(strokes_p, "r", encoding="utf-8") as f:
        raw_strokes = json.load(f)

    # Deduplicate strokes by id
    by_id = {}
    for s in raw_strokes:
        if isinstance(s, dict) and "id" in s:
            by_id[s["id"]] = s

    # Select 10 lines and 10 words
    line_ids = sorted([k for k in by_id.keys() if "line" in k])[:10]
    word_ids = sorted([k for k in by_id.keys() if "word" in k])[:10]
    chosen_ids = line_ids + word_ids
    print(f"Generating 20 golden test vectors ({len(line_ids)} lines, {len(word_ids)} words)...")

    test_cases = []

    for sid in chosen_ids:
        item = by_id[sid]
        text = item["text"]
        strokes = item["strokes"]

        # Render 64px image
        img = render_strokes(strokes, zoom=1.4, width_after_zoom=1.8)
        w, h = img.size

        # Tensor preprocessing identical to dataset.py / inference contract
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - 0.5) / 0.5  # Shape: (64, W)
        pad_w = (-w) % 32
        if pad_w > 0:
            arr = np.pad(arr, ((0, 0), (0, pad_w)), mode="constant", constant_values=1.0)

        tensor = arr[np.newaxis, np.newaxis, :, :]  # Shape: (1, 1, 64, W_pad)
        tensor_bytes = tensor.astype(np.float32).tobytes()
        tensor_sha256 = hashlib.sha256(tensor_bytes).hexdigest()

        # Run ONNX model
        ort_inputs = {input_name: tensor}
        logits = session.run(None, ort_inputs)[0]  # Shape: (T, 1, V)
        log_probs = logits[:, 0, :]  # Shape: (T, V)

        # Run reference verification
        verdicts = verify_reference(
            log_probs=log_probs,
            expected_text=text,
            vocab_char2idx=vocab.char2idx,
            vocab_idx2char=vocab.idx2char,
            t_accept=-0.60,
            t_wrong=-0.95,
            order="visual",
            norm="char",
        )

        test_cases.append({
            "sample_id": sid,
            "text": text,
            "strokes": strokes,
            "tensor_shape": list(tensor.shape),
            "rendered_tensor_sha256": tensor_sha256,
            "log_probs_shape": list(log_probs.shape),
            "log_probs": np.round(log_probs, 4).tolist(),
            "norm": "char",
            "t_accept": -0.60,
            "t_wrong": -0.95,
            "expected_verdicts": verdicts,
        })

    out_dir = ROOT / "tests" / "golden"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "golden_vectors.json"

    suite = {
        "version": "1.0",
        "description": "20 Golden Test Vectors for HTR Verification Parity (Flutter/Dart & NumPy)",
        "model": "crnn_visual_fp32.onnx",
        "vocab_char2idx": vocab.char2idx,
        "vocab_idx2char": {str(k): v for k, v in vocab.idx2char.items()},
        "test_cases": test_cases,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(suite, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(test_cases)} golden test vectors to {out_json} ({out_json.stat().st_size / 1024:.1f} KB).")


if __name__ == "__main__":
    generate_golden_suite()
