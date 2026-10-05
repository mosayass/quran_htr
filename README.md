# Quran HTR (Handwritten Text Recognition)

A high-performance CRNN + CTC pipeline for Arabic Handwritten Text Recognition, specializing in Quranic Ayah text while preserving general-domain Arabic handwriting capability.

---

## 1. Overview & Architecture

The model uses a **CRNN (Convolutional Recurrent Neural Network)** trained with **Connectionist Temporal Classification (CTC) loss**:

```
Input Line Image (B, 1, 64, W)
      │
      ▼
CNN Feature Extractor (VGG-Lite or MobileNetV3-Small)
  - Height collapsed from 64 -> 1
  - Width downsampled 4x -> T = W / 4 feature frames
      │
      ▼
Bidirectional LSTM (2 layers, 256 hidden units per direction -> 512-dim output)
      │
      ▼
Linear Classifier (512 -> Vocab Size: 72 tokens incl. CTC blank)
      │
      ▼
CTC Loss / Vocabulary-Aware Decoding (Greedy & Prefix Beam Search)
```

- **Backbone Options**:
  - `vgg_lite` (default, ~1.2M conv params): Highly stable, proven for OCR, cleanly quantizable to ONNX INT8.
  - `mobilenetv3_small`: Fast mobile backbone patched with a 4x width-stride schedule to avoid temporal starving.
- **Image Preprocessing**: Grayscale, resized to fixed 64px height preserving aspect ratio, right-padded to multiple of 32px, normalized to `[-1, 1]`.

---

## 2. Project Directory Structure

```
quran_htr/
├── configs/
│   └── synth.yaml              # Calibrated synthetic generation config (font weights, bounds)
├── data/
│   ├── derived/                # ayahs.jsonl, splits.json (surah-level train/val/test)
│   ├── fonts/                  # 28 curated Arabic Google Fonts
│   ├── fonts_manifest.json     # Font family metadata, licenses, and SHA256 hashes
│   ├── shards/                 # Fast in-memory .pt shards (synth + khatt)
│   └── tanzil/                 # Original Tanzil Quran source files (Uthmani & Simple)
├── dist/                       # Packaged zip archives for Colab / cloud training
├── scripts/
│   ├── audit_corpus.py         # 80k train audit (font drift, words/line, leakage, RSS)
│   ├── calibrate.py            # Stroke width & ink fraction calibration vs KHATT
│   ├── fetch_fonts.py          # Font downloader and acceptance verification
│   ├── generate_synth_shards.py# Quota-based shard generator
│   ├── overfit_ab.py           # Controlled A/B harness (visual vs logical order)
│   ├── prepare_derived.py      # Ayah windowing and surah splits
│   └── tanzil_stats.py         # Codepoint inventory and verse distribution
├── dataset.py                  # PyTorch Dataset for CSV manifests + preprocessing
├── decode.py                   # Greedy & prefix beam search CTC decoders
├── evaluate.py                 # Evaluation CLI for checkpoints on test sets
├── metrics.py                  # Levenshtein-based corpus CER and WER calculations
├── model.py                    # CRNN architecture & sequence length calculation
├── pretrain_ahcd.py            # AHCD 28-class character classification pretraining
├── shards.py                   # High-performance contiguous uint8 in-memory ShardDataset
├── train.py                    # Multi-source training loop with dual validation
├── vocab.py / vocab.json       # 72-character Arabic vocabulary & index mapping
└── tests/
    └── smoke_test.py           # Multi-backbone forward/backward/CTC sanity suite
```

---

## 3. The Reading-Order Breakthrough (Task 2.8)

### The Problem: Monotonic CTC vs. Right-to-Left Arabic
- **Temporal Direction**: CNN and BiLSTM feature maps scan images from **Left to Right** ($x=0 \to x=W$). Time frame $t=0$ corresponds to the **leftmost** edge of the image.
- **Arabic Text (RTL)**: In standard logical text encoding, character index 0 (`text[0]`) is the first word read, which is positioned on the **rightmost** edge of the image ($x=W$).
- **CTC Conflict**: CTC assumes a strictly monotonic alignment. Training with logical labels forced the model to predict the rightmost character at frame $t=0$ and the leftmost character at frame $t=T$. While a BiLSTM can memorize short isolated words, monotonic alignment collapses on long lines (6–9 words), stalling synthetic CER around ~41% even after 40 epochs.

### The Solution: Visual Reading Order (`--label_order visual`)
1. **At Training / Load Time**: Labels are dynamically reversed (`text[::-1]`) when loaded into memory. Character index 0 now corresponds to the leftmost glyph in the line, aligning naturally with CTC time frames.
2. **At Decoding Time**: The CTC decoder emits characters in visual order and reverses them back (`decoded[::-1]`) into standard logical Arabic.
3. **Data Integrity**: Underlying `.pt` shard files and dataset CSVs remain completely untouched in standard logical Arabic.

### Empirical Validation:

| Stage | Label Order | Epochs | Synthetic CER | Synthetic WER | Real KHATT CER |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline Run | `logical` | 40 | 40.89% | 80.33% | 40.86% |
| **New Run** | **`visual`** | **4** | **7.42%** | **26.65%** | **24.96%** |

In just 4 epochs, visual order achieved a **$5.5\times$ error reduction**, yielding 100% exact full-sentence transcriptions on held-out validation verses.

---

## 4. Dataset Pipeline & Sharding

### Tanzil Quran Text & 28-Font Manifest
- Tanzil text (`data/tanzil/`) is windowed into continuous lines of 3–9 words across Ayah boundaries within Surahs.
- Split by Surah (seed 42, ~80% train, ~10% val, ~10% test) to prevent text leakage.
- Rendered with **HarfBuzz / Raqm** across 28 open-source Arabic fonts (Naskh, Ruq'ah, Kufi, Nastaliq, Playpen).
- Acceptance tested using connected-component analysis on ligatures (`نستعين` and `لا`).

### Calibration Against Real Handwriting (KHATT)
Line images are calibrated against the 5th–95th percentiles of 2,000 real KHATT lines:
- **Median Stroke Width**: 2.00px – 4.00px (measured via Euclidean distance transform on skeletonized ink).
- **Ink Fraction**: 0.043 – 0.110.
- Calibrated sampling weights in `configs/synth.yaml` ensure realistic stroke variation while penalizing non-handwriting-like display fonts.

### High-Performance Contiguous Shards (`shards.py`)
To prevent copy-on-write RAM bloat and IPC serialization bottlenecks in PyTorch `DataLoader`:
- All line images in a shard are packed into a single contiguous uint8 tensor: `(64, total_width)`.
- Slices are retrieved via integer `offsets` and `widths` arrays.
- The entire **89,096-sample** corpus (80k synthetic + 9.1k KHATT) loads into just **1.19 GB RAM**.
- Dynamic on-the-fly tensor-space affine jitter ($\pm2^\circ$ rotation, $\pm3^\circ$ shear, $\pm2\%$ translation) is applied during training.

---

## 5. Training (`train.py`)

### Multi-Source Mixing & Dual Validation
The training loop supports weighted sampling across heterogeneous data sources:
- **80% Synthetic Quranic lines**: Expands character vocabulary and font diversity across all 79 train Surahs.
- **20% Real KHATT lines**: Grounds the model in authentic human handwriting, preventing synthetic overfitting.
- **Dual Validation Tracking**: Every epoch independently evaluates and saves:
  - `best.pt`: Best checkpoint on held-out Quranic Surahs (`synth_val`).
  - `best_real.pt`: Best checkpoint on authentic human handwriting (`khatt_val`).

### Running Training Locally:
```bash
python train.py \
    --vocab_path vocab.json \
    --mix "data/shards/synth/synth_train_*.pt:0.80" \
    --mix "data/shards/khatt/khatt_train_000.pt:0.20" \
    --val_manifest data/shards/synth/synth_val_000.pt \
    --val_real_manifest data/shards/khatt/khatt_val_000.pt \
    --label_order visual \
    --batch_size 32 \
    --samples_per_epoch 40000 \
    --epochs 40 \
    --lr 1e-4 \
    --checkpoint_dir checkpoints/run_visual
```

### Key CLI Options:
- `--label_order {logical, visual}`: Sets reading order (use `visual` for training). Saved into checkpoint metadata.
- `--init_weights PATH`: Loads model weights only (resets optimizer, scheduler, and epoch counter).
- `--resume PATH`: Resumes complete state (optimizer, scheduler, epoch, best CER) from a previous run.
- `--samples_per_epoch N`: Controls virtual epoch size for `WeightedRandomSampler` (default: 40,000).

---

## 6. Evaluation (`evaluate.py`)

Evaluate any trained checkpoint on a test manifest or shard:

```bash
python evaluate.py \
    --test_manifest data/shards/synth/synth_test_000.pt \
    --checkpoint checkpoints/run_visual/best.pt \
    --vocab_path vocab.json
```

- Automatically infers `--label_order` from the checkpoint metadata (`ckpt["label_order"]`), preventing evaluation mismatches.
- Add `--beam_search --beam_width 10` for prefix beam search decoding.

---

## 7. Google Colab Quickstart

To train on Google Colab with GPU acceleration (L4 or A100):

```python
# 1. Mount Drive & Extract Shards
from google.colab import drive
drive.mount('/content/drive')

!mkdir -p /content/data/shards/synth /content/data/shards/khatt
!unzip -q /content/drive/MyDrive/quran_htr/dist/shards_synth.zip -d /content/data/shards/synth
!unzip -q /content/drive/MyDrive/quran_htr/dist/shards_khatt.zip -d /content/data/shards/khatt

# 2. Clone Repository & Install Dependencies
!rm -rf /content/quran_htr
!git clone https://github.com/mosayass/quran_htr.git /content/quran_htr
%cd /content/quran_htr
!pip install -q pyyaml torchvision torchaudio

# 3. Launch Training Loop
!python train.py \
  --init_weights /content/drive/MyDrive/quran_htr_writer_checkpoints/best.pt \
  --vocab_path vocab.json \
  --mix "/content/data/shards/synth/synth_train_*.pt:0.80" \
  --mix "/content/data/shards/khatt/khatt_train_000.pt:0.20" \
  --val_manifest /content/data/shards/synth/synth_val_000.pt \
  --val_real_manifest /content/data/shards/khatt/khatt_val_000.pt \
  --label_order visual \
  --batch_size 64 \
  --samples_per_epoch 40000 \
  --epochs 40 \
  --lr 1e-4 \
  --checkpoint_dir /content/drive/MyDrive/quran_htr_writer_checkpoints/run_visual
```
