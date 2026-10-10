# Task 2.13 Comprehensive Report: Step 2b Diacritic Training, v3 Audit & Tablet Capture Evaluation

## 1. Executive Summary & Core Milestones
- **Step 2b Training**: Completed 20 epochs on Colab GPU with 80% Corpus v3 (60,000 lines) + 20% KHATT (9,096 lines), initialized from `cont_best.pt`, visual order, `CosineAnnealingLR` (1e-4 -> 1e-5), 40k samples/epoch.
- **Diacritic Recognition Breakthrough**: Vowelled synthetic CER plummeted from **23.98% -> 0.44%** on seen fonts (>54x reduction) and **26.06% -> 0.27%** on unseen fonts (nearly 100x reduction). The 44x vowelling gap is completely resolved.
- **Real Handwriting Record**: KHATT validation CER improved from 9.43% down to **8.92%** (real WER down from 34.6% to 33.3%), breaking below 9% for the first time.
- **Tablet Capture Completed & Evaluated**: 100 authentic Quranic samples (40 lines + 60 words) written with stylus on tablet via `capture/index.html`, exported via single-file ZIP, deduplicated, and evaluated (baseline CER: **27.02%**).

---

## 2. Complete Evaluation Matrix & [بتثني] Dot-Letter Confusion Analysis

| Evaluation Set | cont_best.pt (Prior Baseline) | Step 2b: best.pt (Synth-Best) | Step 2b: best_real.pt (Real-Best) | Dot Error Trend [بتثني] |
| :--- | :---: | :---: | :---: | :---: |
| **KHATT Val (Real Handwriting)** | 9.43% (316 errors) | 9.00% (296 errors) | **8.92% (298 errors)** | Reduced by 18 confusions |
| **Synth Test (Clean Rasm)** | **0.57% (91 errors)** | 1.20% (130 errors) | 1.31% (137 errors) | Modest trade-off for full vowelling |
| **Synth Test Diac (Marks On)** | 23.98% (1,851 errors) | **0.44% (47 errors)** | 0.55% (52 errors) | **97.5% drop in dot errors** |
| **Unseen Fonts Diac (Mirza & Zain)** | 26.06% (2,220 errors) | **0.27% (11 errors)** | 0.34% (21 errors) | **99.5% drop in dot errors** |
| **Wordmix Test Set (1–9 words)** | **0.66% (62 errors)** | 1.18% (86 errors) | 1.31% (83 errors) | Balanced across word lengths |
| **Corpus v3 Test Set** | 12.03% (765 errors) | **0.31% (27 errors)** | 0.43% (30 errors) | **~40x error reduction** |
| **Capture Set (Tablet Stylus, 100 samples)** | *N/A* | **27.02% (20 errors)** | 27.73% (25 errors) | Only 20 dot errors across set |

---

## 3. Word-Length Breakdown (Bucketed CER)

| Dataset | 1 Word | 2 Words | 3–5 Words | 6–9 Words |
| :--- | :---: | :---: | :---: | :---: |
| **Corpus v3 Test Set** (`best.pt`) | 0.2% (1,108 lines) | 0.3% (570 lines) | 0.3% (482 lines) | 0.3% (840 lines) |
| **Wordmix Test Set** (`best.pt`) | 0.7% (295 lines) | 1.6% (249 lines) | 1.0% (956 lines) | 1.2% (1,500 lines) |
| **Synth Test Diac** (`best.pt`) | — | — | 0.5% (853 lines) | 0.4% (1,147 lines) |
| **Unseen Fonts Diac** (`best.pt`) | — | — | 0.2% (887 lines) | 0.3% (1,113 lines) |
| **KHATT Val (Real)** (`best_real.pt`) | 20.0% (3 lines) | 22.6% (15 lines) | 12.0% (33 lines) | 8.9% (1,089 lines) |
| **Capture Set (Tablet Stylus)** (`best.pt`) | 26.1% (60 words) | — | 28.0% (28 lines) | 26.2% (12 lines) |

---

## 4. Analysis of Tablet Capture Set (27.02% CER)
- **Zero-Shot Domain Shift**: The model was evaluated without any writer adaptation or fine-tuning on digital ink strokes.
- **Very Low Dot Confusion**: Only 20 dot-letter substitutions `[بتثني]` across the entire 100-sample session (compared to 296 on KHATT), confirming the model reliably resolves Arabic diacritical pointing.
- **Error Distribution**: CER is remarkably consistent across short words (26.1%) and full lines (26.2%), indicating that line segmentation, band-cropping, and aspect ratio padding behave uniformly. The remaining error stems from personal stylus stroke dynamics (digital pen curvature and thickness vs rendered fonts / ballpoint pen), making it the ideal candidate for few-shot writer adaptation in Step 3.

---

## 5. Corpus v3 Generation & Audit
1. **Ayah Count Audit**: Confirmed canonical count is exactly **6,236 Ayahs** across all 114 Surahs. The earlier figure of 6,266 lines was raw line count in the Tanzil text file (6,236 data lines + 30 header/license lines).
2. **Quota Shift Analysis**:
   - Proposal distributions: Word count (25% 1w, 10% 2w, 65% 3–9w); Diacritics (40% none, 30% light, 30% full).
   - Observed acceptance: Word count (36.8% 1w, 19.2% 2w, ~6.3% each 3–9w); Diacritics (33.0% none, 32.1% light, 34.9% full).
   - **Root Cause**: Rejection-driven. Short lines (1–2 words) were made exempt from KHATT ink-fraction bounds (`fr_min=0.043`), yielding ~0% rejection vs ~60% on long lines. Unvowelled lines have lower ink density and occasionally dropped below `fr_min`, slightly shifting accepted unvowelled share.
3. **Contact Sheet & Montage**: Generated [data/synth_preview_v3/montage.png](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/data/synth_preview_v3/montage.png) (24 labeled samples: 8 one-word, 8 full, 8 light at native 64px and 3x zoom) via [scripts/make_v3_montage.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/make_v3_montage.py).
4. **Rasterizer Drift**: 200 identical lines compared between FreeType 2.5.1 + HarfBuzz on Windows vs Linux reference: stroke width drift 0.00%, ink fraction drift 0.37% (0.1082 vs 0.1078), line width drift 0.00% — **PASSED** (<< 10% threshold).

---

## 6. Infrastructure, Tools & Capture Pipeline
- **Extended Evaluation Tool**: Implemented [scripts/evaluate_extended.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/evaluate_extended.py) supporting multi-checkpoint evaluation, `[بتثني]` confusion matrix via sequence alignment, line-length bucketing, and automated capture manifest detection.
- **Single-File Stored ZIP Writer**: Built pure JS, zero-dependency ZIP export (PKZIP Store method 0) into [capture/index.html](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/capture/index.html), packaging all PNG crops, `manifest.csv`, and `strokes.json` into a single-click download.
- **Capture Dataset Curated**: 100 unique samples (40 lines, 60 words) stored in `data/capture/`, deduplicated, and tracked in git.
- **FreeType Patch Verification**: Tested [scripts/patch_freetype_windows.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/patch_freetype_windows.py) for idempotence. Confirmed Linux/Colab uses system dynamic linker (`ld.so`) resolving `libfreetype.so` without circular imports or PATH scan freezes.
