# Hafiz Quran HTR — Knowledge Base & Technical Documentation

This documentation preserves the critical knowledge, empirical findings, architectural contracts, and operational guidelines developed for the **Hafiz Quran HTR (Handwritten Text Recognition & Verification)** system.

---

## Documentation Index

| Document | Primary Focus & Contents |
| :--- | :--- |
| **[system_overview.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/system_overview.md)** | Model architecture (CRNN, VGG-lite backbone, CTC), visual vs. logical order contract, vocabulary definition (72 classes), ONNX exports, input tensor specs, and current best checkpoint lineage. |
| **[inference_contract.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/inference_contract.md)** | Strict technical specification for PyTorch and ONNX inference: tensor dimensions, scaling/padding formulas, 3-state verification algorithm, and configuration schemas. (Scale policy & golden vectors marked PROVISIONAL). |
| **[preprocessing_and_capture.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/preprocessing_and_capture.md)** | Canonical zoom semantics (v2 108px band = 1.0 vs v1 150px band = 150/108 ~1.39), zoom root cause analysis (band double-counting + vertical clipping; dilation is only a hypothesis), augmentation policy, full capture studio (`capture/index.html`), and standalone shareable capture app (`capture/shareable.html`). |
| **[verification_and_evaluation.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/verification_and_evaluation.md)** | 3-state verification policy (ACCEPT, UNCERTAIN, WRONG), per-character LLR normalization, closed-set word ranking (14,872 full dictionary vs 200 distractors), golden test vectors (`tests/golden/golden_vectors.json`, PROVISIONAL), and pure NumPy reference implementation (`scripts/verify_ref.py`). |
| **[writer_experiments_and_benchmarks.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/writer_experiments_and_benchmarks.md)** | Empirical results on `writer_01` (in-sample DEV) and `writer_02` (held-out baseline with frozen w1 thresholds), geometry audit, zoom grid sweep, cross-writer LOWO (Folds A & B), personalization (Fold C), and valid monotonic self-calibration rerun. |
| **[numbers.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/numbers.md)** | Authoritative master metrics table: metric, value, set, in-sample/held-out provenance, preprocessing, threshold source, and date. |
| **[decisions_and_retractions.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/decisions_and_retractions.md)** | Dated changelog of architecture decisions, discarded baselines, and retracted empirical claims. |
| **[normalization_102_tokens.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/normalization_102_tokens.md)** | Complete review table of the 102 non-encodable tokens (Tanzil English copyright metadata) and expected-text comment-strip normalizer. |
| **[capture_and_preprocessing_spec.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/capture_and_preprocessing_spec.md)** | Comprehensive engineering specification for the capture interface, ML coordinate contracts, ZIP schemas, and iPad/iOS Safari error analysis for frontend engineers. |
| **[scripts_and_tooling_guide.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/scripts_and_tooling_guide.md)** | Comprehensive catalog of active scripts, their purpose, inputs, outputs, and standard CLI commands for training, evaluation, verification, and data generation. |
| **[open_questions_and_roadmap.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/open_questions_and_roadmap.md)** | Critical guidelines before continuing, open questions, unresolved trade-offs, and next implementation milestones. |

---

## Quick Reference: Key Numbers & Current State

- **Current Production Model**: `checkpoints/best.pt` / `dist/onnx/crnn_visual_fp32.onnx` (VGG-lite CRNN, 72 classes, visual frame order).
- **Scale Policy**: **PROVISIONAL** (Global ink fill target 48px / 75% in 64px tensor, clamped to prevent clipping; legacy v1 samples default to 150/108 band ratio).
- **Verification Thresholds**: **PROVISIONAL** ($t_{\text{accept}} = -0.60, t_{\text{wrong}} = -0.95$, derived in-sample from writer_01 DEV set).
- **Golden Test Vectors**: **PROVISIONAL** (`tests/golden/golden_vectors.json`, generated from uncalibrated PyTorch inference path; subject to re-generation upon finalized production scale policy).
- **writer_02 Baseline (Zero Training, Canonical 1.0x, Held-Out Provenance)**:
  - **CER**: Overall **37.24%** (Lines: **33.59%**, Words: **40.82%**, WER: **87.83%**).
  - **Closed-Set Rank (14,872 Quran Words)**: Top-1 **42.40%**, Top-5 **60.80%**, Median Rank **2.0**.
  - **3-State Verification** (Frozen writer_01 DEV thresholds $t_{\text{acc}}=-0.60, t_{\text{wrg}}=-0.95$): TRUE claims **41.3% True-Accept**, Corrupted claims **90.3%–100% Detected WRONG** across all single-letter and word error types.
- **Self-Calibration Status**: Prior claim ("proving fine-tuning is mandatory") was **RETRACTED as INVALID** due to non-monotonic thresholds ($t_{\text{accept}} = -2.99 < t_{\text{wrong}} = -1.49$). Valid monotonic rerun ($t_{\text{wrong}} \le t_{\text{accept}}$ across 20 draws) achieves 85%–87% True-Accept, with single-letter corruption detection at 29%–45% and macro word swap detection at 97%–98%.
- **Closed Quran Dictionary Size**: **14,872 unique words** from Tanzil text encodable within the 72-class `vocab.json` (102 non-encodable tokens audited and confirmed to be English copyright block metadata).
