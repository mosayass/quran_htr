# Task 2.12 Completion Report

1. **Environment Audit**: Resolved `.venv` FreeType circular import (`raw.py` importing `freetype`) and PATH scan freeze on Windows via `os.add_dll_directory`. Automated via [scripts/patch_freetype_windows.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/patch_freetype_windows.py).
2. **Render Regression**: 500 lines compared (v2 vs current renderer); median stroke width 2.0 vs 2.0 (0.00% drift), ink fraction 0.0859 vs 0.0857 (0.26% drift), line width 384.0 vs 384.0 (0.00% drift) — **PASSED** (< 10% threshold).
3. **Capture Export**: Set constant stroke width $0.044 \times \text{band height} = 2.816\text{ px}$ at 64px export resolution (verified in $2.4–3.4\text{ px}$ range). Pressure dependence stripped in export, downscaling uses `imageSmoothingQuality = 'high'`.
4. **Capture Prompts**: [scripts/build_capture_prompts.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/build_capture_prompts.py) generated [data/capture_prompts.json](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/data/capture_prompts.json) from 26 test surahs: 40 lines (8 contain "الله" $\ge 5$) and 60 words (30 frequent, 30 rare).
5. **Tanzil Alignment & Codepoints**: Perfect 1:1 word alignment (0 mismatches across 6,266 Ayahs). Simple uses 9 basic diacritic marks (U+064B..U+0652, U+0670). Uthmani adds 13 tajweed/stop marks (U+0653, U+0654, U+06DC, U+06DF..U+06E8, U+06EA..U+06ED) plus wasla U+0671. Toggle label in [capture/index.html](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/capture/index.html) updated to "مشكول (Tanzil Simple)" vs "رسم (Rasm-only)".
6. **Inference Contract**: Updated [docs/inference_contract.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/inference_contract.md) with Section 8 covering stroke rendering rule ($0.044 \times H_{\text{band}}$) and ruled-band cropping policy.
7. **Baseline Evaluation Table (Continuation Checkpoints)**:

| Evaluation Set | cont_best.pt (Synth-Best) | cont_best_real.pt (Real-Best) |
| :--- | :---: | :---: |
| **KHATT Test (Real Handwriting)** | 9.55% CER / 35.63% WER | **9.55% CER / 35.39% WER** |
| **Synth Test (Clean Rasm)** | **0.55% CER / 2.56% WER** | 0.69% CER / 3.25% WER |
| **Synth Test Diac (Marks On)** | **23.99% CER / 71.33% WER** | 24.33% CER / 72.17% WER |
| **Unseen Fonts Diac (Zain & Mirza)** | 26.10% CER / 78.03% WER | **25.99% CER / 77.81% WER** |
| **Wordmix Test Set (1–9 words)** | **0.64% CER / 2.84% WER** | 0.76% CER / 3.53% WER |

8. **Corpus v3 Complete**: 66k lines generated in 267.7s (246.4 l/s) on Colab, word quotas & balanced diacritic split (~33% none, 32% light, 35% full) verified, packaged into `dist/shards_corpus_v3.zip` (404.3 MB) and backed up to Google Drive.
