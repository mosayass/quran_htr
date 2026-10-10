# Task 2.14 Completion Report

1. **Item 1 (Capture Error Taxonomy & Montage)**: **RAN**. Generated `data/capture/report/montage_taxonomy.png` (20 worst + 10 best samples with image, label, pred, CER) & `taxonomy_summary.md`. Line CER is 27.2% vs Word CER 26.8%. Space errors: 12 missed spaces, 15 extra spaces.
2. **Item 2 (Geometry Table at 64px)**: **RAN**. Ink extent: Capture 36.7px [35.0] vs KHATT 59.0px [63.0] vs Synth 41.1px [40.0]. Body height: Capture 15.1px vs KHATT 22.3px (KHATT text is 1.48x taller). Baseline row: Capture 37.8px vs KHATT 36.2px vs Synth 32.9px. Stroke width: Capture 2.78px vs KHATT 2.67px vs Synth 2.77px.
3. **Item 3 (Re-render Sweep from strokes.json on best.pt)**: **RAN**.
   - Current ruled-band (2.8px): 29.60% CER / 73.23% WER
   - Tight vertical crop (+8% margin): 27.18% CER / 76.21% WER
   - Band crop zoomed 1.3x (baseline centered): **24.84% CER / 69.89% WER** (-4.76% abs drop)
   - Band crop zoomed 1.6x (baseline centered): 25.70% CER / 70.26% WER
   - Stroke width 2.0px: 27.41% CER | Stroke width 3.6px: 37.46% CER | Pressure-varying: 39.80% CER
4. **Item 4 (Pipeline Tensor Check)**: **RAN**. 3 capture vs 3 synth samples dumped. Background pad is +1.0000, ink min is -1.0000 on both; right-padding to 32px multiple; visual label order verified. Zero pipeline mismatch.
5. **Item 5 (KHATT TEST for Step 2b Checkpoints, N=1,123)**: **RAN**.
   - `best.pt`: 9.35% CER / 35.30% WER / 328 [بتثني] dot errors
   - `best_real.pt`: **9.24% CER / 34.72% WER / 312 [بتثني] dot errors** (beats cont_best 9.55%).
