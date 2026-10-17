# HTML Canvas Data Pipeline & Analysis Guide (Agent Runbook)

> **Audience**: AI Agents and Engineers receiving `.zip` exports from the HTML canvas capture tools (`capture/shareable.html` or `capture/index.html`).
> **Goal**: End-to-end instructions on unpacking, validating, rendering, evaluating, verifying, and fine-tuning on captured tablet handwriting without breaking model invariants.

---

## 1. Quick Architecture & Canvas Summary

The project provides two client-side static HTML/JS capture interfaces:
- **`capture/shareable.html`** (Production Short Protocol):
  - **40 balanced prompts**: 10 lines + 30 isolated words (takes volunteers ~10–12 minutes).
  - Balanced across all 28 Arabic letter families and includes لفظ الجلالة ("الله" / "لله").
  - Single-file, zero server dependencies, offline-capable in browser, embedded pure-JS ZIP generator (`createStoredZip`).
- **`capture/index.html`** (Full Studio):
  - **145 prompts**: 20 lines + 105 paired words + 20 extra words (comprehensive research protocol).
- **Export Trigger**: User taps **"تصدير الحزمة / Download ZIP"**, generating:
  `hafiz_capture_{writer_id}_{timestamp}.zip`

---

## 2. Ingesting & Unpacking the ZIP

### A. Directory Placement Rules
Depending on the role of the writer:
- **Active Development Writer**: Extract into `data/capture_v2/{writer_id}/` (e.g., `data/capture_v2/writer_05/`).
- **Locked Test Writer**: Extract into `data/locked_test/{writer_id}/`.
  > [!CAUTION]
  > **Locked Test Invariant**: Writers in `data/locked_test/` must NEVER be evaluated or fine-tuned on during development. All tuning scripts enforce `assert_no_locked_test()` and will abort if this path is touched.

### B. Unpacking Command
```python
import zipfile
from pathlib import Path

zip_path = Path("hafiz_capture_writer_03_20261017.zip")
target_dir = Path("data/capture_v2/writer_03")
target_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as zf:
    zf.extractall(target_dir)
print(f"Extracted to {target_dir}")
```

### C. ZIP Contents Contract
Every valid capture ZIP contains:
1. **`manifest.csv`**: Metadata index with columns:
   `sample_id,image_file,mode,transcription,writer_id,capture_version,band_top_css,band_base_css,band_bot_css,canvas_w_css,canvas_h_css,dpr`
2. **`strokes.json`**: Primary vector ink dataset. Array of objects:
   ```json
   {
     "id": "line_01",
     "text": "وقيل يا أرض ابلعي",
     "capture_version": "v2",
     "geometry": {
       "band_top_css": 72.0,
       "band_base_css": 147.0,
       "band_bot_css": 180.0,
       "band_height_css": 108.0,
       "canvas_w_css": 960.0,
       "canvas_h_css": 240.0,
       "dpr": 2.0
     },
     "strokes": [
       [
         {"x": 120.5, "y": 145.2, "p": 0.5, "t": 1728739200000},
         {"x": 121.8, "y": 145.0, "p": 0.52, "t": 1728739200016}
       ]
     ]
   }
   ```
3. **`{sample_id}.png`**: Visual reference rasters (for QA and preview montages only; training uses vector `strokes.json`).

---

## 3. Data Sanity & Coordinate Audit

Run an immediate validation on `strokes.json` before any training or evaluation:

```python
import json
from pathlib import Path
from vocab import Vocabulary

vocab = Vocabulary.load("vocab.json")
samples = json.load(open("data/capture_v2/writer_03/strokes.json", encoding="utf-8"))

# Deduplicate by ID (preserving the latest draw)
deduped = {}
for s in samples:
    deduped[s["id"]] = s
samples = list(deduped.values())

for s in samples:
    geom = s.get("geometry", {})
    # 1. Coordinate Bounds: MUST be in unscaled CSS pixels [0, 960] x [0, 240]
    for stroke in s["strokes"]:
        for pt in stroke:
            assert 0.0 <= pt["x"] <= 1200.0, f"DPR corruption detected (x={pt['x']})"
            assert -50.0 <= pt["y"] <= 350.0, f"Y coordinate out of bounds (y={pt['y']})"
    
    # 2. Geometry check (v2 standard)
    if geom:
        assert geom.get("band_height_css", 108.0) == 108.0, "Unexpected band height"
    
    # 3. Vocabulary check (text must be encodable rasm)
    clean_text = s["text"].strip()
    encoded = vocab.encode(clean_text)
    assert len(encoded) > 0, f"Unencodable text in sample {s['id']}: {clean_text}"

print(f"Validation PASSED: {len(samples)} clean, unique samples.")
```

---

## 4. Rendering Strokes to Model Tensors (`scripts/render_strokes.py`)

The ML pipeline rasterizes vector strokes dynamically using `scripts/render_strokes.py`:

```python
import numpy as np
import torch
from render_strokes import render_strokes
from scale_policy import sample_band

sample = samples[0]
bh, cy, bt = sample_band(sample)  # (108.0, 75.0, 0.0) for v2

# Render to 64px PIL Image
img = render_strokes(
    sample_strokes=sample["strokes"],
    zoom=1.0,               # Canonical zoom: 1.0 for v2 (108px band), 150/108 (~1.39) for legacy v1
    width_after_zoom=2.2,   # Nominal human pen stroke width
    center_y=cy,            # Center baseline
    band_top_y=bt,
    band_height=bh,
    pad_x=16.0              # Safety horizontal margins
)

# Convert to Model Input Tensor: shape [1, 1, 64, W_padded]
arr = np.array(img).astype(np.float32)
# Normalize to [-1.0, +1.0] where +1.0 is white background, -1.0 is ink
norm_arr = (arr / 127.5) - 1.0

# Pad width rightwards with +1.0 (white) to nearest multiple of 32
h, w = norm_arr.shape
w_padded = ((w + 31) // 32) * 32
padded = np.ones((64, w_padded), dtype=np.float32)
padded[:, :w] = norm_arr

tensor = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0)  # [1, 1, 64, W]
```

### Scale Policy Invariant (`scripts/scale_policy.py`)
- Standard zoom is canonical: **$1.0\times$ for v2**, **$150/108 \approx 1.39\times$ for v1**.
- If dynamic scale normalization is applied, ink vertical fill is capped at 48px with `MARGIN_PX = 2.5` to strictly prevent anti-aliasing edge pixel clipping.

---

## 5. Zero-Shot Baseline Evaluation

To evaluate how well the pre-trained base model recognizes the new writer out of the box:

```python
import torch
from train_twofold_lowo import load_base_model, predict_records, records_cer
from vocab import Vocabulary

device = "cuda" if torch.cuda.is_available() else "cpu"
vocab = Vocabulary.load("vocab.json")

# 1. Load canonical base checkpoint (CRNN VGG-Lite, 74 tensors)
model, msg = load_base_model("checkpoints/best.pt", vocab, device)
assert "74/74 tensors" in msg

# 2. Run greedy prediction
preds = predict_records(model, samples, vocab, device)

# Note on Visual Reading Order:
# The CNN/BiGRU processes frames Left-to-Right (LTR).
# The CTC greedy output string is in VISUAL order.
# Logical Arabic text is obtained by reversing: logical_text = visual_text[::-1]

# 3. Compute CER
cer = records_cer(preds)
print(f"Zero-shot baseline CER: {cer:.2f}%")
```

---

## 6. 3-State Verification Algorithm

The core product verification engine runs a 3-state decision policy (ACCEPT / UNCERTAIN / WRONG):

```python
# Compute CTC loss of the expected text vs free greedy decoding
# LLR is strictly character-normalized (never frame-length normalized!):
# LLR_char = (score_expected - score_free) / max(1, num_chars)

t_accept = -0.60
t_wrong = -0.95

def verify_sample(llr_char: float) -> str:
    if llr_char >= t_accept:
        return "ACCEPT"      # Verified true handwriting
    elif llr_char >= t_wrong:
        return "UNCERTAIN"   # Borderline / ambiguous dot or diacritic
    else:
        return "WRONG"       # Corrupted word, substitution, or mistake
```

---

## 7. Fine-Tuning & Personalization (`scripts/train_twofold_lowo.py`)

When fine-tuning on a writer's samples:

> [!IMPORTANT]
> **The Golden Law of Fine-Tuning**: **NEVER train on strokes alone (`stroke_only`)!**
> Unanchored stroke training triggers severe catastrophic forgetting (general synthetic Quran CER blows up from 1.2% to 6–10%).
> You MUST use anchored stream replay:
> - **`mix_a`**: 50% strokes, 30% synthetic Quran lines, 20% KHATT real ink.
> - **`mix_0.85`**: 85% strokes, 15% anchor shards (optimal personalization ratio).

### Execution Command
```bash
# Multi-seed fine-tuning with Step-0 gate and anchored replay:
python scripts/train_twofold_lowo.py \
  --folds A B C \
  --mixes mix_a \
  --seeds 0 1 2 \
  --forget_limit 500 \
  --device cuda
```

### Prompt Group Splitting Invariant
- Samples come in **prompt groups** (a line and its constituent words, e.g. `group_01_line` and `group_01_word_01..05`).
- **Never split a line and its paired words between train and test.** They must remain together in the same split to prevent lexical memorization.

---

## 8. Quick Agent Checklist & Failure Modes

| Potential Bug / Trap | Root Cause | Solution / Invariant |
| :--- | :--- | :--- |
| **Loaded model outputs random text (CER ~100%)** | Key mismatch in checkpoint (`'model'` vs `'model_state'`, `strict=False`) | Always use `load_base_model()` which asserts `init_weights matched 74/74 tensors`. |
| **Coordinates scaled by 2x or 3x** | Frontend multiplied by `window.devicePixelRatio` | Keep coordinates strictly in CSS pixels ($X \le 960, Y \le 240$). |
| **Model text is backwards** | RTL Arabic reading order vs LTR spatial frames | Model outputs visual order; reverse with `[::-1]` for logical Arabic reading. |
| **Synthetic/KHATT error blows up after fine-tune** | Trained with `stroke_only` without anchor replay | Always train with `--mixes mix_a` or `mix_0.85`. |
| **Step-0 Gate Abort** | Pre-train eval diverges from known baseline by >1.0 pt | Verify baseline in `configs/gate_baselines.json` matches the target device (CPU vs CUDA). |
| **Locked test data touched** | Attempting to tune or evaluate on locked test writers | Keep `writer_03` and `writer_04` sequestered in `data/locked_test/`. |
