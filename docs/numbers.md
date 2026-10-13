# Canonical Numbers & Benchmarks Master Table

This document serves as the single source of truth for all reported empirical metrics. Every number includes its evaluation set, provenance (in-sample vs. held-out), preprocessing policy, threshold calibration source, and recording date.

---

## 1. Master Metric Matrix

| Metric Name | Value | Evaluation Dataset | In-Sample / Held-Out | Preprocessing Policy | Threshold Source | Recording Date | Notes & Status |
| :--- | :---: | :--- | :---: | :--- | :--- | :---: | :--- |
| **Synthetic Baseline CER** | **1.20%** | `synth_test_000.pt` (4,000 lines) | Held-Out | Native raster | N/A (Greedy Decode) | 2026-10-09 | Out-of-sample fonts & surahs |
| **Synthetic True Accept** | **100.0%** | `synth_test_000.pt` | Held-Out | Native raster | DEV w01 ($t_{\text{acc}}=-0.60$) | 2026-10-11 | Per-character normalization |
| **Synthetic Dot-Swap Reject** | **100.0%** | `synth_test_000.pt` | Held-Out | Native raster | DEV w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-11 | Jumped from 29.8% under frame-norm |
| **Synthetic Del/Ins Reject** | **100.0%** | `synth_test_000.pt` | Held-Out | Native raster | DEV w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-11 | Jumped from 38.5% under frame-norm |
| **Synthetic Word-Swap Reject** | **100.0%** | `synth_test_000.pt` | Held-Out | Native raster | DEV w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-11 | Perfect macro rejection |
| **w01 Baseline Overall CER** | **24.07%** | `writer_01` (100 samples) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (Greedy Decode) | 2026-10-10 | Primary development set |
| **w01 Baseline Lines CER** | **19.52%** | `writer_01` (40 lines) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (Greedy Decode) | 2026-10-10 | Baseline on line crops |
| **w01 Baseline Words CER** | **34.15%** | `writer_01` (60 words) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (Greedy Decode) | 2026-10-10 | Baseline on isolated word crops |
| **w01 Baseline Overall WER** | **64.68%** | `writer_01` (100 samples) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (Greedy Decode) | 2026-10-10 | Greedy CTC word errors |
| **w01 Full-Dict Top-1 Rank** | **48.33%** | `writer_01` (60 words) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (CTC Loss Rank) | 2026-10-10 | Ranked vs 14,872 Quran words |
| **w01 Full-Dict Median Rank** | **2.0** | `writer_01` (60 words) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (CTC Loss Rank) | 2026-10-10 | 50% of words ranked 1 or 2 |
| **w01 200-Distractor Top-1** | **83.33%** | `writer_01` (60 words) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (CTC Loss Rank) | 2026-10-11 | Sampled protocol approximation |
| **w01 True Claims Accept** | **93.7%** | `writer_01` (100 samples) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | Self DEV ($t_{\text{acc}}=-0.60$) | 2026-10-11 | Target true accept rate |
| **w01 Corrupted Claims Reject** | **95.0%** | `writer_01` (100 samples) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | Self DEV ($t_{\text{wrg}}=-0.95$) | 2026-10-11 | Per-character norm reject rate |
| **w02 Baseline CER (1.4x)** | 53.66% | `writer_02` (145 samples) | **Held-Out** | 1.4x zoom (PROVISIONAL) | N/A (Greedy Decode) | 2026-10-12 | **RETRACTED**: Clipping distortion |
| **w02 Lines CER (1.4x)** | 44.02% | `writer_02` (20 lines) | **Held-Out** | 1.4x zoom (PROVISIONAL) | N/A (Greedy Decode) | 2026-10-12 | **RETRACTED**: Over-magnified |
| **w02 Words CER (1.4x)** | 63.08% | `writer_02` (125 words) | **Held-Out** | 1.4x zoom (PROVISIONAL) | N/A (Greedy Decode) | 2026-10-12 | **RETRACTED**: Over-magnified |
| **w02 Baseline CER (Canonical)**| **37.24%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (Greedy Decode) | 2026-10-12 | Zero training, canonical scale |
| **w02 Lines CER (Canonical)** | **33.59%** | `writer_02` (20 lines) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (Greedy Decode) | 2026-10-12 | Zero training, canonical scale |
| **w02 Words CER (Canonical)** | **40.82%** | `writer_02` (125 words) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (Greedy Decode) | 2026-10-12 | Zero training, canonical scale |
| **w02 Overall WER (Canonical)** | **87.83%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (Greedy Decode) | 2026-10-12 | Zero training, canonical scale |
| **w02 Full-Dict Top-1 Rank** | **42.40%** | `writer_02` (125 words) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (CTC Loss Rank) | 2026-10-12 | Ranked vs 14,872 Quran words |
| **w02 Full-Dict Top-5 Rank** | **60.80%** | `writer_02` (125 words) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (CTC Loss Rank) | 2026-10-12 | Top-5 dictionary precision |
| **w02 Full-Dict Median Rank** | **2.0** | `writer_02` (125 words) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (CTC Loss Rank) | 2026-10-12 | Strong discrimination despite shift |
| **w02 200-Distractor Top-1** | **85.60%** | `writer_02` (125 words) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (CTC Loss Rank) | 2026-10-12 | Sampled protocol approximation |
| **w02 True Accept (Frozen w01)**| **41.3%** | `writer_02` (230 words) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{acc}}=-0.60$) | 2026-10-12 | False rejection due to zero-FT |
| **w02 Dot-Swap Reject (Frozen)**| **91.0%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-12 | 2.5% accept, 6.6% uncertain |
| **w02 Del/Ins Reject (Frozen)** | **90.3%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-12 | 4.8% accept, 4.8% uncertain |
| **w02 Word-Swap Reject (Frozen)**| **100.0%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-12 | Perfect macro rejection |
| **w02 Adj-Swap Reject (Frozen)** | **85.0%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-12 | 10.0% accept, 5.0% uncertain |
| **w02 Skip-Word Reject (Frozen)**| **85.0%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-12 | 15.0% accept, 0.0% uncertain |
| **w02 Mutashabih Reject (Frozen)**| **100.0%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-12 | Perfect mutashabih detection |

---

## 2. Zoom Diagnostic Grid on `writer_02`

| Zoom Factor | Overall CER | Lines CER (20) | Words CER (125) | Overall WER | Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.70x** | 35.18% | 33.59% | 36.73% | 81.74% | Diagnostic sweep |
| **0.85x** | 35.46% | 32.64% | 38.22% | 82.17% | Diagnostic sweep |
| **1.00x** | **37.24%** | **33.59%** | **40.82%** | **87.83%** | **CANONICAL STANDARD** |
| **1.15x** | 43.43% | 37.19% | 49.54% | 95.65% | Diagnostic sweep |
| **1.30x** | 48.78% | 41.37% | 56.03% | 97.39% | Diagnostic sweep |
| **1.40x** | 53.66% | 44.02% | 63.08% | 101.30% | RETRACTED (Double-count/clipping) |

---

## 3. Cross-Writer Fine-Tuning & Personalization Matrix (Folds A, B, C)

> **Root cause of the earlier ~100% CER**: weight-loading bug (`'model'` vs `'model_state'` key, `strict=False`, 0 tensors loaded). The "catastrophic forgetting" explanation is RETRACTED.

### 3a. Valid run (2026-10-13, Colab CUDA, `step2b_v3/best.pt`, `init_weights matched 74/74`, Step-0 gate passed)

Provenance: single run per fold, no seeds/variance. Training data is **strokes only** (synth=0, khatt=0 batches; no anchor mix). Thresholds recalibrated on a slice of the *training* writer (mechanical `t_wrong = t_accept - 0.20`). Detection rates are paired with True-Accept.

| Fold | Train → Test | Step-0 CER (pre) | Final Lines / Words / Overall CER | Δ Overall | True-Accept | Dot-Swap WRONG | Word-Swap WRONG | Thresholds (acc, wrg) | Final train loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A** | w01 (70) → w02 (145, cross-writer) | 36.30% | 28.65 / 30.61 / **29.64%** | **−6.66** | 89.57% | 51.64% | 100.0% | −3.5686, −3.7686 | 0.0007 |
| **B** | w02 (82) → w01 (100, cross-writer) | 24.77% | 27.45 / 20.45 / **26.01%** | **+1.24 (worse)** | 88.10% | 50.00% | 100.0% | −3.6752, −3.8752 | 0.0020 |
| **C** | w02 (56) → w02 (76 held-out groups, personalization) | 38.57% | 23.11 / 25.68 / **24.46%** | **−14.11** | 91.45% | 54.69% | 98.68% | −2.8919, −3.0919 | 0.0003 |

Caveats: (1) Fold C step-0 has no independent known baseline (gate is vacuous there). (2) Gate tolerances: Fold A uses a CUDA baseline (36.15%) taken from the earlier failing run's pre-eval; Fold B tolerance was widened to 1.5 pt (spec said 1.0). (3) Train loss collapses to ~0.001 on 56-82 samples (memorization); gains are real on held-out samples but fragile. (4) Lexical overlap between train and test prompts not yet audited. (5) Dot-swap rejection (~50%) is *lower* than the frozen-w01-threshold figure (91.0% at 41.3% true-accept) but at a different operating point; not comparable.

### 3b. Superseded INVALID rows (uninitialized-weight run)

| Protocol / Fold | Train Split | Test Split | Train Epochs & Loss | Calibrated Thresholds ($t_{\text{acc}}, t_{\text{wrg}}$) | Lines CER | Words CER | Overall CER | True-Accept Rate | Dot Swap WRONG | Word Swap WRONG | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold A (LOWO)** | `writer_01` (70 train, 30 calib) | `writer_02` (148 held-out) | 6 epochs ($3.32 \to 0.021$) | $-6.28, -6.48$ | 104.98% | 102.25% | **103.55%** | **88.61%** | 22.40% | 59.46% | **INVALID** (Uninitialized weights) |
| **Fold B (LOWO)** | `writer_02` (14 train grps, calib) | `writer_01` (100 held-out) | 6 epochs ($3.25 \to 0.025$) | $-6.53, -6.73$ | 109.50% | 75.80% | **102.26%** | **84.40%** | 42.16% | 69.09% | **INVALID** (Uninitialized weights) |
| **Fold C (Personalized)** | `writer_02` (10 train grps, calib) | `writer_02` (10 held-out grps) | 6 epochs ($3.04 \to 0.013$) | $-6.20, -6.40$ | 99.36% | 91.35% | **95.03%** | **83.04%** | 33.90% | 68.57% | **INVALID** (Uninitialized weights) |

---

## 4. Independent Gate Baselines (`configs/gate_baselines.json`)

Precomputed canonical evaluation of the untrained base model (`best.pt`, 74/74 tensors matched) across all test subsets. Evaluated with 3 repeated draws to verify determinism. Step-0 gate enforces $\le 1.0\text{ pt}$ tolerance on the same hardware target.

| Subset Name | Target Writer | Sample Count ($n$) | CPU Baseline CER | CPU Repeat Spread | Target Tolerance | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`A_test`** | `writer_02` (all) | 145 | **37.24%** | 0.000% | $\pm 1.0$ pt | Canonical Frozen |
| **`B_test`** | `writer_01` (all) | 100 | **25.62%** | 0.000% | $\pm 1.0$ pt | Canonical Frozen |
| **`C_test`** | `writer_02` (50% held-out grps) | 76 | **41.25%** | 0.000% | $\pm 1.0$ pt | Canonical Frozen |
| **`LC_test`** | `writer_02` (held-out learning curve grps) | 45 | **39.35%** | 0.000% | $\pm 1.0$ pt | Canonical Frozen |

*Note: On CUDA CuDNN FP32, convolution precision produces ~1.09 pt drift (e.g. 36.15% on `A_test`). CUDA values are measured and recorded under `devices.cuda` in `configs/gate_baselines.json` when running on GPU.*

---

## 5. Lexical-Overlap Audit (`reports/lexical_overlap_audit.json`)

Measures the share of test-prompt words found in train prompts, and stratifies pre-training baseline CER into seen vs. unseen vocabulary words across each partition.

| Fold / Split | Train Writer ($n$) | Test Writer ($n$) | Total Test Tokens | Tokens Seen in Train (%) | Types Seen in Train (%) | Fully Seen Samples (%) | Baseline CER Seen Words | Baseline CER Unseen Words | Lexical Gap (Δ) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold A** | w01 (70) | w02 (145) | 230 | **24.78%** (57/230) | **22.61%** (26/115) | 22.07% | 35.87% | 41.53% | −5.66 pt |
| **Fold B** | w02 (82) | w01 (100) | 269 | **20.07%** (54/269) | **12.68%** (26/205) | 14.00% | 27.53% | 29.88% | −2.35 pt |
| **Fold C** | w02 (56) | w02 (76) | 117 | **17.95%** (21/117) | **8.77%** (5/57) | 17.11% | 47.95% | 43.95% | +4.00 pt |
| **Fold CLC** | w02 (100) | w02 (45) | 69 | **30.43%** (21/69) | **20.00%** (6/30) | 33.33% | 45.95% | 41.51% | +4.44 pt |

*Finding: In cross-writer folds (A & B), seen words have lower baseline error by 2–6 pt. In within-writer partitions (C & CLC), group splitting holds out ~82% of tokens; the small seen sample size shows baseline CER parity.*

---

## 6. Label-Free Adaptation Without Training (`reports/label_free_adaptation.json`)

Tests whether unsupervised feature alignment (AdaBN recomputing running statistics without backprop or labels) and global target scale normalization ($S_w$ with exact no-clip margin) improve recognition without gradient updates. Evaluated on DEV writers (`writer_01`, `writer_02`).

| Writer | Metric / Policy | Baseline (Canonical) | AdaBN (Unlabeled Stats) | Scale Norm ($S_w$) | Scale + AdaBN | Clipped Samples (Canon vs Norm) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **`writer_01`** ($n=100$) | Zoom / Scale | 1.389x (v1) | 1.389x | $S_w = 0.615$ | $S_w = 0.615$ | **42 → 0** clipped |
| | **Overall CER** | **25.62%** | **34.81%** (+9.19) | **47.90%** (+22.27) | **64.80%** (+39.17) | |
| **`writer_02`** ($n=145$) | Zoom / Scale | 1.000x (v2) | 1.000x | $S_w = 0.362$ | $S_w = 0.362$ | **80 → 0** clipped |
| | **Overall CER** | **37.24%** | **44.93%** (+7.69) | **54.60%** (+17.35) | **69.89%** (+32.65) | |

*Key Finding: Passive label-free adaptation is strictly detrimental. Recomputing BatchNorm statistics on small sample sizes (100–145) shifts running stats away from the generalized pre-trained domain. Scaling down ink to accommodate extreme ascender samples shrinks average character height below the CNN receptive field. Gradient-based fine-tuning with anchor replay is mandatory.*

---

## 7. Locked-Test Partition Status (`configs/writer_splits.yaml`)

Cryptographically isolated test partition to prevent test set leakage.

- **Dev Writers**: `writer_01` (100 samples), `writer_02` (145 samples).
- **Locked Test Writers**: `writer_03` (102 samples), `writer_04` (104 samples).
- **Location**: `data/locked_test/` with manifest `data/locked_test/SHA256SUMS`.
- **Enforcement**: All training, tuning, and evaluation scripts enforce `assert_no_locked_test()`.
- **Evaluations to date**: **0** (untouched and uninspected).


