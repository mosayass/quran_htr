# Quran HTR — Step 1: Base Pre-training (open-domain Arabic handwriting)

This covers items **1 (model architecture)** and **2 (dataset acquisition &
preprocessing)** of Step 1. Vocabulary (`vocab.py`) is included as a
dependency of the dataset pipeline — full CTC decoding and the training
loop with CER/WER are the next two items in this session, not yet here.

Files:
- `vocab.py` — Arabic character set + CTC index mapping
- `dataset.py` — manifest builder + `LineImageDataset` + preprocessing + collate
- `model.py` — `CRNN` (vgg_lite / mobilenetv3_small backbones) + CTC-ready forward pass

---

## 1. Dataset acquisition

### KHATT (primary dataset for Step 1)
The official KHATT release (KFUPM Handwritten Arabic TexT) requires
either KFUPM registration or an LDC membership (catalog no. LDC2015T23).
In practice, the easiest path is a community Kaggle mirror of the same
public dataset:

**[`iraqyomar/khatt-arabic-hand-written-lines`](https://www.kaggle.com/datasets/iraqyomar/khatt-arabic-hand-written-lines)**
— already line-segmented (11.4k line images), laid out as:

```
image/   line image files
label/   one .txt per image, same filename stem, windows-1256 (cp1256) encoded
```

```python
# Colab — Kaggle API auth (upload kaggle.json from kaggle.com/settings first)
from google.colab import files
files.upload()  # select kaggle.json
!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
!pip install -q kaggle
!kaggle datasets download -d iraqyomar/khatt-arabic-hand-written-lines -p /content/khatt --unzip
```

```python
from dataset import build_manifest_from_image_label_dirs, validate_manifest_against_vocab
from vocab import Vocabulary

build_manifest_from_image_label_dirs(
    '/content/khatt/image', '/content/khatt/label',
    '/content/manifests/all.csv',
)
validate_manifest_against_vocab('/content/manifests/all.csv', Vocabulary())
```

Notes:
- Use `build_manifest_from_image_label_dirs()`, not `build_manifest()`,
  for this mirror — images and labels sit in two separate folders here,
  not as an adjacent `.txt` next to each image. `build_manifest()` still
  applies if you later get a KHATT release that uses the adjacent-`.txt`
  convention.
- The ground truth is **cp1256-encoded**, not UTF-8 — that's handled by
  the `encoding="cp1256"` default in the new function, but keep it in
  mind if you fetch text from KHATT any other way (e.g. via `Vocabulary`
  or manual inspection) and see garbled/mojibake Arabic.
- This mirror ships as a single pool of images rather than KHATT's
  official train/val/test writer splits, so you'll need to split it
  yourself. Do a **random split by writer, not by line** if writer IDs
  are recoverable from filenames — otherwise the same handwriting style
  leaks across train/val and overstates validation accuracy. If writer
  IDs aren't recoverable from this mirror's filenames, a plain random
  line-level split is an acceptable fallback for Step 1 (open-domain
  pre-training doesn't need to be as rigorous as your eventual Step 3
  edge-calibration eval).
- `nizarcharrada/khattarabic` is a more complete Kaggle mirror (full
  original forms/paragraphs, not just pre-extracted lines) if you want
  more data or the official form-level structure later — it needs more
  digging to locate the line-level ground truth inside it, so start with
  `iraqyomar`'s version above.

### AHCD (Arabic Handwritten Character Dataset)
AHCD is **isolated character** images (28 classes, 32×32px, no cursive
joins or ligatures) — it is a classification dataset, not a line/sequence
dataset, and cannot be fed into `LineImageDataset`/CTC directly.

Recommended use here: **optional CNN warm-start only.** Train the
`_VGGLiteBackbone` (or an equivalent shallow copy of it) as a 28-way
character classifier on AHCD first, then load those conv weights before
starting KHATT CTC training. This can slightly speed up early
convergence by giving the backbone's early layers a head start on Arabic
stroke/curvature statistics — but it is not required, and KHATT alone is
sufficient to run Step 1. I'd suggest skipping this optional path
initially and only revisiting it if Step 1 convergence is slow.
Available via Kaggle ("Arabic Handwritten Characters Dataset").

---

## 2. Colab environment setup

```python
# Cell 1 — mount storage (KHATT is large; keep it in Drive, not Colab's ephemeral disk)
from google.colab import drive
drive.mount('/content/drive')

# Cell 2 — deps (torch/torchvision ship preinstalled on Colab; pillow too)
!pip install -q python-Levenshtein   # for CER/WER in the training step (next deliverable)

# Cell 3 — put these three files somewhere importable
import sys
sys.path.append('/content/drive/MyDrive/quran_htr')   # wherever you upload vocab.py/dataset.py/model.py
```

Build the manifest once per split (train/val/test), pointing at KHATT's
official split lists so writer identity doesn't leak across sets:

```python
from dataset import build_manifest, validate_manifest_against_vocab
from vocab import Vocabulary

build_manifest('/content/drive/MyDrive/khatt/train_lines', '/content/manifests/train.csv')
build_manifest('/content/drive/MyDrive/khatt/val_lines',   '/content/manifests/val.csv')

v = Vocabulary()
validate_manifest_against_vocab('/content/manifests/train.csv', v)
v.save('/content/drive/MyDrive/quran_htr/vocab.json')   # freeze it — re-use for Step 2/3
```

---

## 3. Preprocessing summary (implemented in `dataset.py`)

- Convert to **grayscale**.
- Resize to a **fixed height of 64px**, preserving aspect ratio (width
  varies per line — that's expected and handled by CTC + dynamic
  padding, not by squashing width).
- **Right-pad** width to a multiple of 32px per-sample, then pad again to
  the batch's max width in `collate_fn` (so batches can still have
  variable width across batches, just not within one).
- Normalize pixel values to **[-1, 1]** (pad value = -1.0, i.e. "white").
- Cap width at 1600px (`MAX_WIDTH`) as a safety net against corrupted or
  mis-segmented lines — KHATT lines at h=64 are rarely near this.

Sanity-check before training: run `python dataset.py <khatt_root>
<out.csv>` standalone — it builds the manifest and validates every
transcription against `vocab.py`'s charset, printing any OOV characters
so you can fix the vocab before a training run wastes time on encode
errors mid-epoch.

---

## 4. Model architecture summary (implemented in `model.py`)

`CRNN(vocab_size, backbone="vgg_lite")`:

```
(B,1,64,W) -> CNN backbone -> (B,256,2,W/4) -> collapse H -> (B,256,W/4)
           -> permute -> (T=W/4, B, 256) -> BiLSTM(2 layers, hidden=256)
           -> Linear(512, vocab_size) -> log_softmax -> (T, B, vocab_size)
```

- Default backbone is a small custom VGG-style stack (`vgg_lite`, ~1.2M
  conv params) — the standard, well-proven choice for line-level OCR,
  and the safest bet for a clean ONNX INT8 export later.
- `mobilenetv3_small` is provided as an alternative since you mentioned
  it, with its stride schedule patched so width only downsamples 4x
  instead of MobileNetV3's default 32x (CTC needs enough time steps to
  align against 50–100 character Quranic-length transcriptions — a
  32x width collapse would starve it). Verify the patched stride count
  against your installed torchvision version before trusting it blindly.
- `compute_output_seq_len()` converts `dataset.py`'s raw pixel widths
  into the CTC `input_lengths` argument `nn.CTCLoss` needs — this is the
  glue between the two files you'll use directly in the training loop.
- `greedy_ctc_decode()` is included only as a bare sanity check (e.g.
  "is the model outputting anything Arabic-shaped after a few epochs")
  — real CTC decoding with the frozen vocabulary is the next step.

Run `python model.py` (needs `torch`/`torchvision`) to print param counts
and confirm output shapes for both backbones on a dummy batch.

---

## Next in this session (not yet built)
3. **Vocabulary & CTC decoding** — freezing `vocab.json`, proper
   greedy/beam CTC decode utilities, blank-token handling edge cases.
4. **Training loop** — `nn.CTCLoss`, optimizer/scheduler, CER/WER
   computation (Levenshtein-based), checkpointing, and the val loop.

Say the word when you're ready for those.
