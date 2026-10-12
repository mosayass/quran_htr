# Hafiz Quran HTR — Knowledge Base & Technical Documentation

This documentation preserves the critical knowledge, empirical findings, architectural contracts, and operational guidelines developed for the **Hafiz Quran HTR (Handwritten Text Recognition & Verification)** system.

---

## Documentation Index

| Document | Primary Focus & Contents |
| :--- | :--- |
| **[system_overview.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/system_overview.md)** | Model architecture (CRNN, VGG-lite backbone, CTC), visual vs. logical order contract, vocabulary definition (72 classes), ONNX exports, input tensor specs, and current best checkpoint lineage. |
| **[inference_contract.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/inference_contract.md)** | Strict technical specification for PyTorch and ONNX inference: tensor dimensions, scaling/padding formulas, 3-state verification algorithm, and configuration schemas. |
| **[preprocessing_and_capture.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/preprocessing_and_capture.md)** | Canonical zoom semantics (v2 108px band = 1.0 vs v1 150px band = 150/108 ~1.39), stroke dilation sensitivity, augmentation policy, full capture studio (`capture/index.html`), and standalone shareable capture app (`capture/shareable.html`). |
| **[verification_and_evaluation.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/verification_and_evaluation.md)** | 3-state verification policy (ACCEPT, UNCERTAIN, WRONG), per-character LLR normalization, closed-set word ranking (14,872 full dictionary vs 200 distractors), golden test vectors (`tests/golden/golden_vectors.json`), and the pure NumPy reference implementation (`scripts/verify_ref.py`). |
| **[writer_experiments_and_benchmarks.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/writer_experiments_and_benchmarks.md)** | Empirical results on `writer_01` and `writer_02`, geometry audit (ink-extent ratio, body height, stylus pressure), zoom grid sweep, cross-writer LOWO (Folds A & B), personalization (Fold C), prompt grouping rationale, and the self-calibration test conclusion. |
| **[scripts_and_tooling_guide.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/scripts_and_tooling_guide.md)** | Comprehensive catalog of active scripts, their purpose, inputs, outputs, and standard CLI commands for training, evaluation, verification, and data generation. |
| **[open_questions_and_roadmap.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/open_questions_and_roadmap.md)** | Critical guidelines before continuing, open questions, unresolved trade-offs, and next implementation milestones (Flutter port, partial-line verification, vocabulary expansion). |

---

## Quick Reference: Key Numbers & Current State

- **Current Production Model**: `checkpoints/best.pt` / `dist/onnx/crnn_visual_fp32.onnx` (VGG-lite CRNN, 72 classes, visual frame order).
- **Canonical Scale Definition**: v2 ruled band height is **108 CSS px** ($\text{zoom} = 1.0$). Legacy v1 band is 150 CSS px ($\text{zoom} = 150/108 \approx 1.39$). Nominal stroke width: **2.2 px**.
- **writer_02 Baseline (Zero Training, Canonical 1.0x)**:
  - **CER**: Overall **37.24%** (Lines: **33.59%**, Words: **40.82%**, WER: **87.83%**).
  - **Closed-Set Rank (14,872 Quran Words)**: Top-1 **42.40%**, Top-5 **60.80%**, Median Rank **2.0**.
  - **3-State Verification** (Frozen w1 thresholds $t_{\text{accept}}=-0.60, t_{\text{wrong}}=-0.95$): TRUE claims **41.3% ACCEPT**, Corrupted claims **90.3%–100% WRONG** across all single-letter and word error types.
- **Self-Calibration Finding**: Without fine-tuning, shifting thresholds on unseen writers degrades corruption detection to 64.8% due to baseline likelihood distribution shift. Model fine-tuning/adaptation is strictly required.
- **Closed Quran Dictionary Size**: **14,872 unique words** from Tanzil text encodable within the 72-class `vocab.json` (102 raw tokens with non-standard glyphs excluded).
