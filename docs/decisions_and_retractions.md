# Decisions, Retractions & Provenance Log

This document records architectural decisions, dated policy changes, and explicitly retired or retracted claims in the Hafiz Quran HTR project. Maintaining this log prevents past misconceptions from resurfacing.

---

## 1. Dated Retractions & Corrected Misconceptions

### [2026-10-12] Retraction: 'Catastrophic Forgetting' Diagnosis on Folds A, B, C Was a Silent Weight Loading Bug
- **Prior Claim (Task 2.22)**: Stated that fine-tuning on small stroke batches (60-119 samples) down to loss ~0.02 caused catastrophic forgetting / loss of general HTR representation, blowing up held-out CER to ~100%.
- **Retraction Reason**: The checkpoint `best.pt` stored model parameters under dictionary key `'model'`, whereas `scripts/train_twofold_lowo.py` queried `ckpt.get('model_state', ckpt)`. Because `model.load_state_dict(..., strict=False)` ran with `strict=False`, 0/65 tensors were matched. The model was trained from completely random uninitialized weights from epoch 1 (initial CTC loss 3.32 vs 0.89 for pre-trained weights), memorizing 80 samples and outputting random garbage on test samples.
- **Corrected Status**: Marked **INVALID**. Fixed checkpoint loading, added `init_weights matched N/M tensors` assertion, and instituted a mandatory Step-0 gate aborting training if pre-train evaluation fails to match known baseline within 1 pt.

### [2026-10-12] Retraction: Initial Self-Calibration Claim ("Fine-Tuning Mandatory") Was Based on an Invalid Sweep
- **Prior Claim (Task 2.21)**: Stated that self-calibrating thresholds from writer_02's first 20 samples caused corruption detection to collapse to $64.8\%$, "proving that fine-tuning is mandatory."
- **Retraction Reason**: The initial sweep unconstrained percentiles produced $t_{\text{accept}} = -2.9919$ and $t_{\text{wrong}} = -1.4860$. Because $t_{\text{accept}} < t_{\text{wrong}}$, the decision policy was **MATHEMATICALLY INVALID** (the UNCERTAIN interval was inverted, creating simultaneous accept/reject conflicts).
- **Corrected Status**: Marked **INVALID pending valid monotonic self-calibration** ($t_{\text{wrong}} \le t_{\text{accept}}$) with 20 repeated draws at $N \in \{10, 20, 40\}$ samples.

### [2026-10-12] Retraction: The "102 Out-of-Vocabulary Quran Words" Were Actually Tanzil Copyright Text
- **Prior Claim (Tasks 2.15–2.20)**: Assumed that out of 14,974 tokens in [data/tanzil/quran-simple-clean.txt](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/data/tanzil/quran-simple-clean.txt), exactly 102 tokens were "non-standard Quranic glyphs or unnormalized dagger alifs" that failed vocabulary encoding.
- **Retraction Reason**: Auditing codepoints revealed that lines 1–6236 contain 14,868 unique Arabic Quran words, and **100.0% of actual Quran words are already fully encodable in `vocab.json`**. The 102 unencodable tokens were English copyright text appended to the bottom of the Tanzil file (e.g. `Tanzil`, `Project`, `Copyright`, `http://tanzil.net/updates/`).
- **Corrected Status**: No Quranic vocabulary expansion or retraining is needed. A simple comment-stripping pre-pass (`if line.startswith('#'): continue`) resolves the 102 non-Arabic tokens completely.

### [2026-10-11] Retraction: 1.4x Fixed Zoom Policy on v2 Captures
- **Prior Policy**: Applied $1.4\times$ zoom universally to tablet ink based on Task 2.15 findings.
- **Retraction Reason**: $1.4\times$ was calibrated for `writer_01`'s wide $150\text{ px}$ band (where handwriting occupied only $52.1\%$ of height). Applying $1.4\times$ to `writer_02`'s tighter $108\text{ px}$ band caused double-counting and severe ascender clipping, blowing up CER from $37.24\%$ to $53.66\%$.
- **Corrected Status**: Replaced by canonical zoom ($1.0\times$ for v2, $150/108 \approx 1.39$ for v1) and global target fill normalization. Stroke dilation remains a hypothesis, while geometric clipping and band double-counting are the proven causes.

### [2026-10-10] Retraction: Temporal Frame-Length Normalization in LLR
- **Prior Policy**: Normalized Log-Likelihood Ratio by temporal frame count: $\text{LLR} = \Delta / (t_2 - t_1 + 1)$.
- **Retraction Reason**: Words span $25–40$ frames. A single substituted dot or inserted letter only penalizes $2–3$ frames; dividing by 35 frames diluted single-letter penalties, dropping dot-swap rejection to $29.8\%$.
- **Corrected Status**: Replaced by per-character normalization: $\text{LLR} = \Delta / \max(1, \text{num\_chars})$, achieving $100.0\%$ corruption rejection on synthetic test shards and $\ge 90.3\%$ on real ink.

---

## 2. Provisional Status & Provenance Notice

The following system components are currently marked **PROVISIONAL**:

| Component | Current Value / Rule | Provenance | Status | Condition for Freezing |
| :--- | :---: | :--- | :---: | :--- |
| **Operating Thresholds** | $t_{\text{accept}} = -0.60$<br>$t_{\text{wrong}} = -0.95$ | Calibrated on `writer_01` DEV ink (in-sample) only. | **PROVISIONAL** | Re-freeze after completing 2-fold cross-writer fine-tuning (Folds A & B). |
| **Scale Normalization** | Canonical: v2 = $1.0$, v1 = $150/108$ | Calibrated on `writer_01` (100) and `writer_02` (145). | **PROVISIONAL** | Validate across $\ge 5$ volunteer writers using `capture/shareable.html`. |
| **Golden Reference Vectors** | [tests/golden/golden_vectors.json](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/tests/golden/golden_vectors.json) | Generated using provisional thresholds ($t_{\text{acc}}=-0.6, t_{\text{wrg}}=-0.95$) on v1 ink. | **PROVISIONAL** | Re-export final golden vectors after multi-writer threshold freeze. |

---

## 3. Permanent Architectural Decisions

1. **Visual-Order Target Labels**: Left-to-Right CNN spatial feature processing requires reversing RTL Arabic target strings ($S_{\text{vis}} = S_{\text{logical}}[::-1]$) to maintain monotonic CTC alignment.
2. **Prompt Group Unit of Splitting**: Splits must never partition lines and their paired words independently. A line and its constituent words must remain together to prevent lexical memorization.
3. **Zero-Dependency Static Client**: Capture tools ([capture/shareable.html](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/capture/shareable.html)) must remain standalone single-file HTML/JS with pure in-browser ZIP packaging to run on any tablet without server infrastructure.
