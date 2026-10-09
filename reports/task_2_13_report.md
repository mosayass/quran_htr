# Task 2.13 Completion Report

1. **Step 2b Completed**: Trained 20 epochs (80% Corpus v3 + 20% KHATT, visual order). Real CER reached **8.92%** (vs 9.43% cont_best). Diacritic CER dropped by >50x: **0.44%** on Synth Diac (was 23.98%) and **0.27%** on Unseen Fonts Diac (was 26.06%), completely closing the vowelling gap.
2. **Evaluation Summary Table**:

| Evaluation Set | cont_best.pt (Dot Err) | best.pt (Synth-Best) | best_real.pt (Real-Best) |
| :--- | :---: | :---: | :---: |
| **KHATT Val (Real)** | 9.43% (316) | 9.00% (296) | **8.92% (298)** |
| **Synth Test (Clean Rasm)** | 0.57% (91) | **1.20% (130)** | 1.31% (137) |
| **Synth Test Diac (Marks On)** | 23.98% (1851) | **0.44% (47)** | 0.55% (52) |
| **Unseen Fonts Diac** | 26.06% (2220) | **0.27% (11)** | 0.34% (21) |
| **Wordmix Test Set** | 0.66% (62) | **1.18% (86)** | 1.31% (83) |
| **Corpus v3 Test Set** | 12.03% (765) | **0.31% (27)** | 0.43% (30) |

3. **Montage & v3 Audit**: Generated [data/synth_preview_v3/montage.png](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/data/synth_preview_v3/montage.png) (24 samples: 8 1w, 8 full, 8 light at 64px & 3x zoom). Ayah count confirmed 6,236 (6,266 raw lines with 30 header comments). Quota shift explained as rejection-driven. 200-line Linux vs Windows rasterizer drift is 0.37% (PASSED).
4. **Extended Evaluation & Capture ZIP**: Added [scripts/evaluate_extended.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/evaluate_extended.py) with [بتثني] dot-matrix and word buckets. Pure JS zero-dependency ZIP export integrated into [capture/index.html](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/capture/index.html).
