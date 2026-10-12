# Scripts & Tooling Guide

This guide catalogs the core operational scripts in the repository, explaining their purpose, inputs, outputs, and standard invocation patterns.

---

## 1. Core Model & Pipeline Modules

| File | Purpose & Responsibilities |
| :--- | :--- |
| **[model.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/model.py)** | Defines `CRNN` neural network (VGG-Lite backbone, BiGRU, linear projection to 72 classes) and `compute_output_seq_len()`. |
| **[vocab.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/vocab.py)** | Defines `Vocabulary` class; handles encoding/decoding, blank ID management, and character mappings. |
| **[dataset.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/dataset.py)** | Implements `StrokeDataset` (vector strokes $\to$ rasterization with canonical zoom and augmentations) and `LineImageDataset`. |
| **[shards.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/shards.py)** | Fast memory-mapped / pre-packed tensor shard loader (`ShardDataset`) for synthetic corpus and KHATT datasets. |
| **[decode.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/decode.py)** | Implements CTC greedy decoding with visual-order reversal support (`greedy_decode()`). |
| **[metrics.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/metrics.py)** | Computes Character Error Rate (`cer`), Word Error Rate (`wer`), and corpus-level aggregated error metrics (`corpus_cer_wer`). |

---

## 2. Training Scripts

### A. General Training Engine ([train.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/train.py))
The primary training runner supporting single sources or weighted multi-source mixing:
```bash
python train.py \
  --init_weights checkpoints/best.pt \
  --mix data/capture/writer_01_train_80.json:0.5 \
  --mix data/shards/synth/synth_train_000.pt:0.3 \
  --mix data/shards/khatt/khatt_train_000.pt:0.2 \
  --lr 5e-5 --min_lr 5e-6 --scheduler cosine --epochs 6 \
  --samples_per_epoch 6000 --batch_size 32 \
  --label_order visual --checkpoint_dir checkpoints/step3_lowo
```
- **Inputs**: Shards, JSON stroke splits, or CSV manifests.
- **Outputs**: `checkpoints/<dir>/best.pt`, `best_real.pt`, `last.pt`.

### B. Multi-Fold LOWO & Personalization Runner ([scripts/train_twofold_lowo.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/train_twofold_lowo.py))
Manages prompt group splitting and evaluation for cross-writer adaptation:
- **`generate_all_group_splits()`**: Splits samples by prompt group so lines and paired words stay together.
- Creates:
  - Fold A: `data/capture/w1_group_train.json` & `w1_group_calib.json`
  - Fold B: `data/capture_v2/writer_02/w2_group_train.json` & `w2_group_calib.json`
  - Fold C: `data/capture_v2/writer_02/fold_c_train.json` & `fold_c_test.json`

---

## 3. Benchmarking & Verification Scripts

### A. Canonical Baseline Benchmark ([scripts/benchmark_writer_02_canonical.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/benchmark_writer_02_canonical.py))
The comprehensive benchmark runner for `writer_02`:
```bash
python scripts/benchmark_writer_02_canonical.py
```
- **What it does**:
  1. Runs zoom grid diagnostic sweep: $\{0.7, 0.85, 1.0, 1.15, 1.3\}$.
  2. Evaluates canonical 1.0x scale: CER on words, lines, overall.
  3. Computes closed-set rank across full 14,872 Quran dictionary and 200 distractors.
  4. Evaluates 3-state verification across true claims and 6 corruption types.
  5. Renders composite worst/best montage to [reports/montage_writer_02_worst_best.png](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/reports/montage_writer_02_worst_best.png).

### B. Pure NumPy Verification Reference ([scripts/verify_ref.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/verify_ref.py))
Zero PyTorch dependencies. Intended as the algorithmic ground truth for mobile client porting:
```bash
python scripts/verify_ref.py --golden tests/golden/golden_vectors.json
```
- **Validates**: All 20 golden test vectors for exact decision state match and $|\Delta\text{LLR}| \le 0.05$.

### C. Golden Vector Generator ([scripts/generate_golden_vectors.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/generate_golden_vectors.py))
Regenerates golden test vectors from production ONNX model and canonical render path:
```bash
python scripts/generate_golden_vectors.py
```
- **Outputs**: [tests/golden/golden_vectors.json](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/tests/golden/golden_vectors.json).

### D. Self-Calibration Diagnostic ([scripts/test_self_calibration.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/test_self_calibration.py))
Tests threshold recalibration on an unseen writer without weight fine-tuning:
```bash
python scripts/test_self_calibration.py
```
- **Demonstrates**: Why threshold shifting alone fails (corruption detection drops to 64.8%).

---

## 4. Capture & Preprocessing Utilities

| Script | Purpose & Usage |
| :--- | :--- |
| **[scripts/render_strokes.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/render_strokes.py)** | Core vector-to-raster rendering function. Implements 4x supersampling, constant width dilation, baseline centering, and padding. |
| **[scripts/build_shareable_html.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/build_shareable_html.py)** | Compiles the standalone static capture web app [capture/shareable.html](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/capture/shareable.html) with 40 balanced prompts. |
| **[scripts/build_capture_prompts.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/build_capture_prompts.py)** | Selects paired capture lines and word prompts from test surahs in [data/tanzil/](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/data/tanzil/). |
| **[scripts/export_onnx.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/export_onnx.py)** | Exports PyTorch `.pt` checkpoints to dynamic-width ONNX models (`crnn_visual_fp32.onnx`). |
| **[scripts/task_2_17_suite.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/task_2_17_suite.py)** | Library containing Quranic corruption generation functions (`apply_corruption()` for dot swap, deletion/insertion, word swap, adjacent swap, skip word, mutashabih). |

---

## 5. Standard Colab Execution Snippets (<= 12 Lines Each)

```python
# Cell 1: Environment & Repository Setup (<=12 lines)
!git clone https://github.com/mosayass/quran_htr.git repo
%cd repo
!pip install -q onnxruntime editdistance pyyaml torchvision
import torch
print("CUDA Available:", torch.cuda.is_available())
```

```python
# Cell 2: Sync Checkpoints & Shards from Google Drive (<=12 lines)
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p checkpoints data/shards/synth data/shards/khatt
!cp /content/drive/MyDrive/quran_htr_writer_checkpoints/step2b_v3/best.pt checkpoints/best.pt
!cp /content/drive/MyDrive/quran_htr_shards/synth/synth_train_000.pt data/shards/synth/
!cp /content/drive/MyDrive/quran_htr_shards/khatt/khatt_train_000.pt data/shards/khatt/
```

```python
# Cell 3: Prompt Group Splits Generation (<=12 lines)
from scripts.train_twofold_lowo import generate_all_group_splits
generate_all_group_splits()
print("Splits created: Fold A (w1->w2), Fold B (w2->w1), Fold C (w2 50/50)")
```

```python
# Cell 4: Fold A Fine-Tuning: Train w1 -> Test w2 (<=12 lines)
!python train.py --init_weights checkpoints/best.pt \
  --mix data/capture/w1_group_train.json:0.5 \
  --mix data/shards/synth/synth_train_000.pt:0.3 \
  --mix data/shards/khatt/khatt_train_000.pt:0.2 \
  --lr 5e-5 --min_lr 5e-6 --scheduler cosine --epochs 6 \
  --samples_per_epoch 6000 --batch_size 32 \
  --label_order visual --checkpoint_dir checkpoints/fold_a
```

```python
# Cell 5: Fold C Personalization: w2 50% -> w2 50% (<=12 lines)
!python train.py --init_weights checkpoints/best.pt \
  --mix data/capture_v2/writer_02/fold_c_train.json:0.5 \
  --mix data/shards/synth/synth_train_000.pt:0.3 \
  --mix data/shards/khatt/khatt_train_000.pt:0.2 \
  --lr 5e-5 --min_lr 5e-6 --scheduler cosine --epochs 6 \
  --samples_per_epoch 6000 --batch_size 32 \
  --label_order visual --checkpoint_dir checkpoints/fold_c
```
