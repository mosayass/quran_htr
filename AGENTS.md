# AGENTS.md — Handover Guide & Project Knowledge Base

> **For Future Antigravity Agents**: This document provides the complete, authoritative state of the **Hafiz Quran HTR** project. Read this before starting any task in a fresh session.

---

## 1. Project Overview & Architecture

- **Project**: Arabic Quranic Handwritten Text Recognition (HTR) from digital ink and line images.
- **Model Architecture**: CRNN (VGG-lite CNN backbone $\to$ 2-layer Bidirectional GRU $\to$ Linear $\to$ CTC Loss).
- **Canonical Input**: Grayscale images of fixed height $H = 64\text{ px}$, width padded rightwards with pure white background ($255$) to a multiple of 32 ($W \ge 32$). Normalized to $[-1.0, +1.0]$.
- **Vocabulary**: 74 classes defined in `vocab.json` (Blank ID = 0, Indices 1–73 = Arabic rasm characters).
- **Visual-Order Contract (`label_order="visual"`)**: 
  - Model processes frames left-to-right ($x=0 \to W$).
  - For RTL Arabic, visual left is the *end* of the word/phrase.
  - Model outputs visual order; greedy CTC output is reversed (`text[::-1]`) to obtain logical Unicode reading order.
  - Never change `label_order` without retraining or cross-checking checkpoint metadata.

---

## 2. Hardware Constraints & Workflow Division (CRITICAL)

- **User's Local Machine**: Windows laptop with **only 8 GB RAM**.
  - **RULE**: **NEVER** run memory-heavy workloads locally (e.g., generating 66k synthetic lines, unpickling multiple large `.pt` shards simultaneously, or heavy multi-epoch CPU training). This will freeze/thrash the machine.
- **Remote Acceleration (Google Colab)**:
  - User has **Google Colab with ~100 paid compute units** (GPU, high-RAM, fast vCPUs).
  - Heavy tasks (model training, 66k Corpus shard generation, large evaluation runs) belong on **Colab**.
  - Always provide concise, ready-to-paste Colab cells for heavy workloads.
  - Backups on Drive: `/content/drive/MyDrive/quran_htr/dist/` and `/content/drive/MyDrive/quran_htr_writer_checkpoints/`.

---

## 3. Milestones & Current State (Completed up to Task 2.12)

### A. Environment & Windows FreeType Fix
- **Root Cause**: `.venv/Lib/site-packages/freetype/raw.py` had a circular import (`import freetype` inside `raw.py` while `__init__.py` did `from freetype.raw import *`), plus Windows recursive PATH scan freezes on OneDrive and Python 3.8+ DLL isolation.
- **Reproducible Script**: `scripts/patch_freetype_windows.py`. Run if `.venv` is ever re-installed.

### B. Render Regression Test (`scripts/check_render_regression.py`)
- Compared 500 lines: v2 reference vs current renderer.
- Median Stroke Width: 2.00 vs 2.00 (0.00% drift).
- Median Ink Fraction: 0.0859 vs 0.0857 (0.26% drift).
- Median Line Width: 384.0 vs 384.0 (0.00% drift).
- **Result: PASSED** (max drift 0.26% $\ll$ 10% threshold).

### C. Digital Ink Capture Studio (`capture/index.html`)
- **Stroke Rendering Rule**: Constant stroke width $W = 0.044 \times H_{\text{band}} = 2.816\text{ px}$ at 64px export height (verified in $[2.4, 3.4]\text{ px}$ real handwriting range).
- **Export Invariance**: No pressure dependence in export rasters.
- **High-Quality Downscaling**: `imageSmoothingQuality = 'high'`.
- **UI Prompt Toggle**: Switch between **"مشكول (Tanzil Simple)"** and **"رسم (Rasm-only)"**. Labels in exported manifests stay strictly rasm-only.

### D. Capture Prompts (`data/capture_prompts.json`)
- Generated via `scripts/build_capture_prompts.py` exclusively from 26 test surahs (`data/derived/splits.json`):
  - **40 lines**: 3–9 words each, with 8 containing the word "الله" ($\ge 5$).
  - **60 words**: 30 frequent words and 30 rare words.

### E. Tanzil Dataset Alignment & Codepoints
- `data/tanzil/quran-simple.txt` (vowelled) and `data/tanzil/quran-simple-clean.txt` (rasm) are 100% word-aligned (0 mismatches across 6,266 Ayahs).
- **Simple**: 9 basic diacritics (U+064B..U+0652, U+0670).
- **Uthmani**: Adds 13 tajweed/stop marks (U+0653, U+0654, U+06DC, U+06DF..U+06E8, U+06EA..U+06ED) plus wasla (U+0671).
- Manifest and training labels are **always pure rasm** (`quran-simple-clean.txt`).

### F. Continuation Baseline Evaluation Table
Evaluated across all test sets:

| Evaluation Set | `cont_best.pt` (Synth-Best) | `cont_best_real.pt` (Real-Best) | Description |
| :--- | :---: | :---: | :--- |
| **KHATT Test (Real Handwriting)** | **9.55% CER** / 35.63% WER | **9.55% CER** / **35.39% WER** | 1,123 real handwriting test lines |
| **Synth Test (Clean Rasm)** | **0.55% CER** / **2.56% WER** | 0.69% CER / 3.25% WER | 4,000 clean synthetic lines |
| **Synth Test Diac (Marks On)** | **23.99% CER** / **71.33% WER** | 24.33% CER / 72.17% WER | 2,000 vowelled lines, rasm labels |
| **Unseen Fonts Diac (Held-out)** | 26.10% CER / 78.03% WER | **25.99% CER** / **77.81% WER** | 2,000 vowelled lines (Zain & Mirza) |
| **Wordmix Test Set (1–9 words)** | **0.64% CER** / **2.84% WER** | 0.76% CER / 3.53% WER | 3,000 lines mixed word counts |

### G. Corpus v3 Generated & Backed Up
- **Tool**: `scripts/generate_corpus_v3.py`.
- **Volume**: 66,000 lines (60,000 train, 3,000 val, 3,000 test).
- **Quotas**: Exact word count distribution (1w ~37%, 2w ~19%, 3–9w uniform), relaxed ink fraction on short 1–2 word samples, multi-level diacritics (33% none, 32% light, 35% full).
- **Artifact**: `dist/shards_corpus_v3.zip` (**404.3 MB**) saved and backed up to Google Drive at `/content/drive/MyDrive/quran_htr/dist/shards_corpus_v3.zip`.

---

## 4. Key Checkpoints & Shards Locations

- **Local Checkpoints** (`checkpoints/`):
  - `checkpoints/best.pt` & `best_real.pt` (Original 8-epoch checkpoints, DO NOT OVERWRITE).
  - `checkpoints/cont_best.pt` & `cont_best_real.pt` & `cont_last.pt` (Continuation training checkpoints from Colab).
- **Drive Checkpoints**:
  - `/content/drive/MyDrive/quran_htr_writer_checkpoints/run_visual/`
- **Dist Shards Zips** (on Drive `/content/drive/MyDrive/quran_htr/dist/`):
  - `shards_corpus_v3.zip` (404.3 MB, 66k lines)
  - `shards_wordmix.zip` (25.5 MB, 3k lines)
  - `shards_synth.zip` (v2 synth shards)
  - `shards_khatt.zip` (KHATT train & val shards)

---

## 5. Important Rules of Engagement for Future Tasks

1. **Keep Manifests Strictly Rasm-Only**: Ground-truth transcriptions must match `vocab.json` (unvowelled clean Arabic letters). Never inject diacritics into the target labels.
2. **Preserve Checkpoint Integrity**: Keep `cont_` prefixes for continuation models; never overwrite baseline 8-epoch `best.pt` without explicit user instruction.
3. **Colab Execution First for Big Workloads**: When generating shards, training for multiple epochs, or running full-dataset evaluations, write Colab snippets and let the user run them on GPU.
4. **Reports**: Keep completion reports concise ($\le 25$ lines) and committed separately if requested.
