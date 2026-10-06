# Hafiz Quran HTR — Inference Contract & Specification

This document defines the strict interface contract for the Hafiz Handwritten Text Recognition (HTR) CRNN model during inference and deployment (PyTorch and ONNX Runtime).

---

## 1. Input Tensor Specifications

| Property | Value / Rule |
| :--- | :--- |
| **Tensor Shape** | `[B, 1, 64, W]` (Batch, Channels=1, Height=64, Width=$W$) |
| **Height ($H$)** | **Fixed strictly to 64 pixels** |
| **Width ($W$)** | Dynamic; must be a positive integer multiple of 32 ($W \ge 32$) |
| **Data Type** | `float32` (or `float16` for FP16 execution) |
| **Channels** | 1 (Grayscale image) |
| **Color Polarity** | Background is white, ink is dark/black |
| **Pixel Value Range** | Normalized to $[-1.0, +1.0]$ |
| **Normalization Formula** | $x_{\text{norm}} = \frac{x_{\text{uint8}}}{127.5} - 1.0$ (where white $255 \to +1.0$, black $0 \to -1.0$) |

---

## 2. Image Preprocessing, Resizing & Padding Rules

When given a raw line/word image crop of arbitrary dimensions $(H_{\text{raw}}, W_{\text{raw}})$:

1. **Grayscale Conversion**:
   Convert color images to single-channel 8-bit grayscale (`L` mode).

2. **Aspect-Ratio Preserving Resize**:
   Scale image height to $64$ px while preserving aspect ratio:
   $$W_{\text{scaled}} = \max\left(32, \text{round}\left(W_{\text{raw}} \times \frac{64}{H_{\text{raw}}}\right)\right)$$
   Use bilinear or bicubic interpolation for clean character contours.

3. **Width Alignment & Padding**:
   The CNN backbone contains two $2\times$ pooling/stride downsamplings ($4\times$ total horizontal reduction).
   To prevent boundary artifacts and ensure integer frame time steps:
   $$W_{\text{padded}} = \left\lceil \frac{W_{\text{scaled}}}{32} \right\rceil \times 32$$
   Pad along the right edge with pure white background pixels ($255$ uint8, corresponding to $+1.0$ normalized).

4. **Batch Collation**:
   If batch size $B > 1$, pad all images in the batch to the maximum width in the batch ($W_{\text{batch}} = \max_i W_{\text{padded}}^{(i)}$).

---

## 3. Model Output Tensor Specifications

| Property | Value / Rule |
| :--- | :--- |
| **Tensor Shape** | `[T, B, V]` |
| **Time Steps ($T$)** | $T = \frac{W_{\text{padded}}}{4}$ |
| **Batch Size ($B$)** | Number of lines evaluated simultaneously |
| **Vocab Size ($V$)** | $74$ classes |
| **Output Values** | Log-probabilities ($\log P$) produced via `LogSoftmax(dim=-1)` |

---

## 4. CTC Decoding & Visual-Order Reversal Contract

### 4.1 Blank Token
- **Blank ID**: `0` (associated with token `"<blank>"` in `vocab.json`).

### 4.2 Greedy CTC Collapse
For each batch element $b \in [0, B-1]$:
1. Extract greedy argmax class at each time frame:
   $$\hat{c}_t = \arg\max_{k \in [0, V-1]} \log P(t, b, k), \quad \text{for } t = 0, \dots, T-1$$
2. Collapse consecutive repeated tokens:
   $\hat{c}_t$ is retained only if $\hat{c}_t \ne \hat{c}_{t-1}$.
3. Remove blank tokens ($\hat{c}_t \ne 0$).
4. Map remaining class IDs to character strings via `vocab.json`.

### 4.3 Visual-Order Output Reversal
- **Rationale**: The temporal dimension $t=0 \to T-1$ progresses from the left side of the image ($x=0$) to the right side ($x=W$). In right-to-left (RTL) Arabic writing, the visual leftmost characters are the *end* of the phrase, while the rightmost characters are the *beginning*. The Hafiz CRNN models are trained with `label_order="visual"` to eliminate CTC time-reversal latency penalties.
- **Contract**:
  $$\text{emitted\_visual\_string} = \text{ctc\_collapse}(\hat{c}_0, \dots, \hat{c}_{T-1})$$
  $$\text{logical\_arabic\_text} = \text{emitted\_visual\_string}[::-1]$$
  Reversing the visual sequence yields standard Unicode logical reading order.

---

## 5. Vocabulary Contract & Checksum

- **Canonical Vocab File**: `vocab.json` (74 entries).
- **Index 0**: `"<blank>"`.
- **Indices 1–73**: Arabic rasm characters (letters, alif variants, ligatures, hamzas).
- **Integrity Verification**:
  Before running inference, verify the runtime vocabulary matches the model's metadata hash:
  $$\text{SHA256}(\text{vocab.json}) = \text{onnx\_metadata}["vocab\_hash"]$$

---

## 6. ONNX Model Specifications & Embedded Metadata

### 6.1 ONNX Configuration
- **Opset Version**: `14`
- **Dynamic Axes**:
  - `images`: `{0: "batch_size", 3: "width"}`
  - `log_probs`: `{0: "time", 1: "batch_size"}`

### 6.2 Embedded Metadata Key-Value Pairs
Exported models MUST embed the following key-value pairs in `metadata_props`:

| Key | Description | Example Value |
| :--- | :--- | :--- |
| `label_order` | Reading order emitted by temporal frames | `"visual"` |
| `vocab_hash` | SHA-256 hex digest of `vocab.json` | `e2a4...` |
| `vocab_file` | Canonical vocabulary filename | `"vocab.json"` |
| `blank_id` | CTC blank class index | `"0"` |
| `target_height` | Fixed image height in pixels | `"64"` |
| `width_multiple`| Required width divisor | `"32"` |
| `polarity` | White background, black ink | `"bg_white_ink_black"` |

---

## 7. Python Reference Preprocessing & Decoding Implementation

```python
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess PIL image according to Hafiz inference contract."""
    # 1. Grayscale
    gray = img.convert("L")
    w, h = gray.size

    # 2. Aspect resize to height 64
    new_w = max(32, round(w * (64.0 / h)))
    resized = gray.resize((new_w, 64), Image.Resampling.BILINEAR)

    # 3. Pad to multiple of 32 with white background (255)
    padded_w = ((new_w + 31) // 32) * 32
    padded = Image.new("L", (padded_w, 64), color=255)
    padded.paste(resized, (0, 0))

    # 4. Normalize to [-1.0, 1.0], shape [1, 1, 64, padded_w]
    arr = (np.array(padded, dtype=np.float32) / 127.5) - 1.0
    tensor = arr[np.newaxis, np.newaxis, :, :]
    return tensor

def decode_output(log_probs: np.ndarray, vocab: list[str], label_order: str = "visual") -> str:
    """
    Greedy CTC decode log_probs (T, 1, V).
    Reverses output if label_order == 'visual'.
    """
    best_ids = np.argmax(log_probs[:, 0, :], axis=-1)  # shape (T,)
    
    # CTC collapse
    collapsed = []
    prev = -1
    for idx in best_ids:
        if idx != prev:
            if idx != 0:  # skip blank
                collapsed.append(vocab[idx])
            prev = idx
    
    text = "".join(collapsed)
    if label_order == "visual":
        text = text[::-1]
    return text
```
