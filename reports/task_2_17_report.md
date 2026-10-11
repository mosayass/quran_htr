# Task 2.17 Completion Report

1. **Geometry & Band Calibration (Item 1)**: From `strokes.json`, old median ink extent = **78.19 CSS px** (mean 85.76 px), old band = 150 px -> ratio **52.1%**. New band set to **108 CSS px** (`top=72`, `base=147`, `bot=180` px) giving **72.4%** ratio (within 0.72-0.78 target) with generous stylus spacing (75px ascender, 33px descender). `capture/index.html` updated with per-sample geometry export (`capture_version: v2`, band top/base/bot, canvas size, DPR) and band-relative strokes; `StrokeDataset` updated to parse geometry.
2. **Stroke Renderer IoU (Item 2)**: Evaluated vs all 100 exported PNGs: **Median IoU = 0.9400 (94.0%)**, **Min IoU = 0.4549** (from redrawn duplicate `sample_line_003`).
3. **Paired Capture Prompts (Item 3)**: Built in `data/capture_prompts.json` and loaded into `capture/index.html`: **20 lines** (4-7 words, test surahs, 6 containing "الله" >= 5), **105 paired word prompts** (each line's words as separate prompts), plus **20 extra words** (10 frequent, 10 rare); vowelled/rasm mixed.
4. **3-State Verification Engine (Item 4)**: Implemented `verify.py` supporting `ACCEPT`, `UNCERTAIN`, and `WRONG`. Marks `WRONG` only if LLR < tau AND the free-path decode is a different valid Quran word in Tanzil vocabulary. Calibrated strict (FAR 1%, tau=-0.92) and relaxed (FAR 5%, tau=-0.71) operating modes.
5. **Synthetic Verification Benchmark (Item 5)**: Evaluated 6 corruption types across lines vs isolated word crops:
   - **Mutashabih Substitution**: Line AUC **0.950** (FRR@5%: 12.8%, 1.2% Accept / 73.3% Uncert / 25.5% Wrong) vs Crop AUC **0.996** (FRR@5%: 0.9%, 1.1% Accept / 71.8% Uncert / 27.2% Wrong). Pairs saved to `data/derived/mutashabih_pairs.json`.
   - **Whole-word Swap**: Line AUC **0.942** (FRR@5%: 16.4%, 2.1% Accept / 28.0% Wrong) vs Crop AUC **0.997** (FRR@5%: 1.1%, 1.2% Accept / 27.1% Wrong).
   - **Adjacent-word Swap**: Line AUC **0.981** (FRR@5%: 5.7%) vs Crop AUC **0.938** (FRR@5%: 38.3%).
   - **Dot-family Swap**: Line AUC **0.742** (FRR@5%: 53.6%) vs Crop AUC **0.994** (FRR@5%: 2.8%, 6.1% Accept / 26.7% Wrong).
   - **Letter Del/Ins**: Line AUC **0.765** (FRR@5%: 51.4%) vs Crop AUC **0.994** (FRR@5%: 2.8%, 6.4% Accept / 25.8% Wrong).
   - **Skipped Word**: Line AUC **0.824** (FRR@5%: 50.6%) vs Crop AUC **0.942** (FRR@5%: 38.3%).
