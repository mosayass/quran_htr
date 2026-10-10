# Task 2.15 Completion Report

1. **Joint Grid Sweep (20 conditions)**: Global minimum at **Zoom 1.4x, Width 1.8 px**: CER dropped from 27.02% (base) to **22.66%** (-4.36% abs), WER 64.31% (-4.46% abs). 95% CI: [-1.49%, +5.74%]. Over-zooming (>=1.6x) degrades to 25.5-29.3% due to ascender/descender clipping.
2. **Reconciliation (29.60% vs 27.02%)**: Task 2.14 re-render had +17.2% excess ink (2,119 vs 1,807 px, IoU 0.847) from PIL integer-pixel line dilation and vertex circles. Decoupling width post-zoom with 4x supersampling resolves the artifact.
3. **Adaptive Body Height vs Fixed Zoom**: Fixed 1.4x zoom (**22.66% overall, 19.52% lines, 34.15% words**) decisively beats adaptive normalization (20px: 26.56%, 22px: 28.74%, 24px: 29.91%). Adaptive height is noisy on short words (1-2 letters lack projection mass, causing over-zoom clipping).
4. **Closed-Set Word Ranking (60 Words vs 14,872 Quran Words)**: **Top-1: 48.3% (29/60)** | **Top-5: 70.0% (42/60)** | **Top-10: 76.7% (46/60)** | **Median Rank: 2.0 / 14,872**.
5. **Line Verification (40 Lines True vs Corrupted)**: Verification accuracy **97.5% (39/40)**. FAR/FRR: at tau=-1.5 (FAR 0.0%, FRR 85.0%), tau=-2.0 (FAR 0.0%, FRR 82.5%), tau=-2.5 (FAR 5.0%, FRR 77.5%).
6. **Capture Studio & Docs**: `capture/index.html` tightened band to 48px (ink extent ratio 73%, within 0.7-0.8); added `writer_id`, `device`, `stylus`, `prompt_type` to manifest. `docs/inference_contract.md` updated with provisional 1.4x zoom / 1.8-2.2px policy.
7. **Held-Out Integrity**: 100 captured samples and `strokes.json` untouched and strictly held out.
