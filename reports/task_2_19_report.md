# Task 2.19 Report: Provenance Audit, Per-Char Norm & Golden Vectors

1. **Provenance Audit (RAN)**: Synthetic metrics (99.3% True Accept, 77.7% Corrupted Wrong) are out-of-sample (`synth_test_000.pt`, held-out surahs & fonts). All v1 CER numbers (27.02%, 25.39%, 22.66%, 24.07%) are in-sample (`writer_01` DEV).
2. **Unexplained Numbers Resolved (RAN)**:
   - **Baseline CER 24.07% vs 22.66%**: `StrokeDataset` uses `render_strokes.py` with `pad_x=16` and width mod-32 padding; 22.66% used unpadded custom crop. Both are in-sample at 1.4x zoom.
   - **Top-1 48.3% $\to$ 83.3% Jump**: Task 2.15 evaluated full 14,974 Tanzil dictionary (48.3%); 2.18 sampled 200 random distractors under 1.4x zoom (83.3%). Full-dictionary rank remains ~50%.
3. **Per-Character Normalization (RAN)**: Evaluated on synthetic test shards ($t_{\text{accept}} = -0.60, t_{\text{wrong}} = -0.95$):
   - True claims: **100.0% ACCEPT**, 0.0% UNCERTAIN, 0.0% WRONG (mean LLR -0.016).
   - Dot-family swap: **100.0% WRONG**, 0.0% ACCEPT (jump from 29.8% under frame-norm).
   - Letter del/ins: **100.0% WRONG**, 0.0% ACCEPT (jump from 38.5% under frame-norm).
4. **verify_ref.py & Golden Vectors (RAN)**: Pure NumPy reference created with explicit visual-order reversal. 20 golden vectors in `tests/golden/golden_vectors.json`; `scripts/verify_ref.py --golden` passed 20/20. Section 9 added to `docs/inference_contract.md`.
5. **data/capture_v2/ Ingestion (NOT RUN)**: Directory does not exist yet (awaiting Writer 02 export). `scripts/ingest_capture_v2.py` staged for automatic validation upon upload.
