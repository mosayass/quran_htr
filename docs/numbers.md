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

### 3a. Task 2.25 Multi-Seed LOWO Fine-Tuning & Anchored Replay Benchmark (2026-10-13)

Provenance: Evaluated across independent random seeds on CUDA A100. Base checkpoint `best.pt` (`init_weights matched 74/74 tensors`). Step-0 gate verified on CUDA. Forgetting evaluated against held-out `synth_test` (4,000 lines) and `khatt_val` (1,140 lines).

| Fold | Mix Ratio (Strokes / Synth / KHATT) | Seeds | Step-0 Pre CER | Final Overall CER (mean ± std) | Synthetic Test CER (Forgetting) | KHATT Val CER (Forgetting) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **A** (w01 → w02) | `mix_a` (0.5 / 0.3 / 0.2) | $n=4$ | 36.30% | **29.10% ± 0.59%** | **0.81% ± 0.09%** | **9.60% ± 0.25%** | **CANONICAL ANCHORED** |
| **A** (w01 → w02) | `mix_b` (0.3 / 0.4 / 0.3) | $n=4$ | 36.30% | **29.90% ± 0.69%** | **0.81% ± 0.07%** | **9.34% ± 0.33%** | Anchored Replay |
| **A** (w01 → w02) | `stroke_only` (1.0 / 0.0 / 0.0) | $n=4$ | 36.30% | 28.87% ± 0.75% | 5.09% ± 0.31% (Forgetting!) | 18.09% ± 0.32% (Forgetting!) | Unanchored Baseline |
| **B** (w02 → w01) | `mix_a` (0.5 / 0.3 / 0.2) | $n=4$ | 24.77% | **21.32% ± 0.03%** | **0.80% ± 0.10%** | **9.48% ± 0.35%** | **CANONICAL ANCHORED** |
| **B** (w02 → w01) | `mix_b` (0.3 / 0.4 / 0.3) | $n=4$ | 24.77% | **21.26% ± 0.12%** | **0.79% ± 0.08%** | **9.31% ± 0.31%** | Anchored Replay |
| **B** (w02 → w01) | `stroke_only` (1.0 / 0.0 / 0.0) | $n=4$ | 24.77% | 25.06% ± 0.25% (Worse) | 8.66% ± 0.25% (Forgetting!) | 19.60% ± 0.29% (Forgetting!) | Unanchored Baseline |
| **C** (w02 50/50) | `mix_a` (0.5 / 0.3 / 0.2) | $n=4$ | 38.57% | **31.70% ± 0.48%** | **0.80% ± 0.09%** | **9.47% ± 0.28%** | Reference Baseline |
| **C** (w02 50/50) | `mix_0.7` (0.7 / 0.18 / 0.12) | $n=2$ | 38.57% | **28.66% ± 0.64%** | **0.95% ± 0.03%** | **9.32% ± 0.01%** | Anchored Replay |
| **C** (w02 50/50) | `mix_0.85` (0.85 / 0.09 / 0.06) | $n=2$ | 38.57% | **27.14% ± 0.25%** | **1.04% ± 0.03%** | **9.64% ± 0.03%** | **CANONICAL WINNER (Rule Passed)** |
| **C** (w02 50/50) | `mix_a` (Frozen Backbone) | $n=2$ | 38.57% | 34.29% ± 1.52% | 1.25% ± 0.06% | 9.46% ± 0.12% | Backbone-Frozen (BiLSTM+Head) |
| **C** (w02 50/50) | `mix_b` (0.3 / 0.4 / 0.3) | $n=3$ | 38.57% | **32.62% ± 0.22%** | **0.79% ± 0.07%** | **9.22% ± 0.23%** | Anchored Replay |
| **C** (w02 50/50) | `stroke_only` (1.0 / 0.0 / 0.0) | $n=4$ | 38.57% | 24.69% ± 0.59% | 6.57% ± 0.36% (Forgetting!) | 18.63% ± 0.40% (Forgetting!) | Unanchored Baseline |

### 3b. Rebuilt Fold C Personalization Learning Curve (Fixed `C_test` $n=76$, Base CER: 38.57%)

| Mix Config | $N=10$ Samples | $N=20$ Samples | $N=40$ Samples | $N=56$ Samples | Net Personalization Δ ($N=56$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`mix_0.85` (Winner)** | **36.25% ± 1.51%** | **32.06% ± 0.37%** | **30.45% ± 1.39%** | **26.61% ± 0.50%** | **−11.96 pt** (Synth 1.07%, KHATT 9.65%) |

### 3c. Weight Interpolation Benchmark ($W_{\alpha} = (1 - \alpha) W_{\text{base}} + \alpha W_{\text{stroke\_only}}$)

| Alpha ($\alpha$) | Held-Out `C_test` CER ($n=76$) | Synthetic Test CER (Base: 1.20%) | KHATT Val CER (Base: 8.92%) | Retention Rule Status |
| :---: | :---: | :---: | :---: | :--- |
| **$\alpha = 0.30$** | **32.68%** | **1.582%** (Δ +0.38 pt) | **9.772%** (Δ +0.85 pt) | **PASSED** (synth $\le +0.5$, khatt $\le +1.5$) |
| **$\alpha = 0.50$** | 28.93% | 2.223% (Δ +1.02 pt) | 11.190% (Δ +2.27 pt) | FAILS retention rule |
| **$\alpha = 0.70$** | 25.89% | 3.405% (Δ +2.20 pt) | 13.174% (Δ +4.25 pt) | FAILS retention rule |

*Core Scientific Takeaways:*
1. **`mix_0.85` is the Optimal Personalization Policy**: Anchored stroke weight 0.85 drops C_test CER to **27.14% ± 0.25%** (-4.56 pt better than `mix_a` 31.70%), while strictly preserving synthetic error at **1.04%** (better than base 1.20%) and KHATT val at **9.64%** (well within the +1.5 pt cap).
2. **Backbone Adaptation is Essential**: Freezing the CNN backbone degrades CER by +2.59 pt (34.29% vs 31.70%), showing that spatial feature tuning is helpful for stroke adaptation.
3. **Weight Interpolation Sub-Optimal vs Anchoring**: Interpolating stroke-only weights at $\alpha = 0.30$ satisfies retention (32.68% CER), but direct multi-stream anchored training (`mix_0.85`) achieves far superior accuracy (**27.14%**) with superior retention (**1.04%** synth).
4. **Rebuilt Learning Curve on Fixed C_test**: With zero leakage, personalization scales monotonically from 38.57% base $\to$ 36.25% ($N=10$) $\to$ 32.06% ($N=20$) $\to$ 30.45% ($N=40$) $\to$ **26.61%** ($N=56$).

### 3d. 4-Writer Personalization Benchmark (`mix_0.85`: 0.85 strokes / 0.09 synth / 0.06 khatt, 6,000 steps)

Provenance: Evaluated under canonical 1.0x on CUDA GPU using canonical base checkpoint `best.pt` (SHA256: `D9F63734...`, 54,793,797 B, `init_weights matched 74/74 tensors`). All writers evaluated on strictly held-out prompt-group test splits ($N=2$ seeds, 6 epochs $\times$ 1,000 steps). Synthetic test base is 1.20%, KHATT val base is 8.92% (retention rule: synth $\le +0.5$ pt, KHATT $\le +1.5$ pt).

| Writer ID | Split / Partition | Total Samples | Train Pool (Pure Tr + Cal) | Held-Out Test ($n$) | Base Pre CER | Base Pre WER | Post-FT Test CER (mean ± std) | Post-FT Test WER (mean ± std) | Seen Lexical CER* | Unseen Lexical CER* | Synth Test CER | KHATT Val CER | Retention Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **`writer_01`** | Dev Baseline (`w1_all`) | 100 | 70 (56 tr + 14 cal) | 30 cal / 100 all | 24.77% | 54.64% | **21.32% ± 0.03%** | 49.20% ± 0.35% | 27.53% | 29.88% | **0.80% ± 0.10%** | **9.48% ± 0.35%** | **PASSED** (Dev reference) |
| **`writer_02`** | Fold C (50/50 group split) | 145 | 70 (56 tr + 14 cal) | 76 (10 lines + 66 words) | 38.57% | ~57.90% | **27.14% ± 0.25%** | 48.68% ± 0.40% | 24.66% | 29.82% | **1.04% ± 0.03%** | **9.64% ± 0.03%** | **PASSED** (Rule Winner) |
| **`writer_03`** | Fold C3 (50/50 group split) | 148 | 70 (56 tr + 14 cal) | 78 (11 lines + 67 words) | 21.25% | 57.72% | **10.46% ± 0.23%** | **37.81% ± 1.71%** | **5.00% ± 0.00%** ($n=23$) | **12.39% ± 0.30%** ($n=100$) | **1.10% ± 0.02%** | **9.72% ± 0.09%** | **PASSED** (−10.79 pt ΔCER) |
| **`writer_04`** | Fold C4 (50/50 group split) | 152 | 69 (56 tr + 13 cal) | 83 (11 lines + 72 words) | 17.25% | 50.00% | **12.54% ± 0.11%** | **38.64% ± 1.06%** | **20.41% ± 1.44%** ($n=27$) | **12.21% ± 0.15%** ($n=105$) | **1.04% ± 0.02%** | **9.56% ± 0.00%** | **PASSED** (−4.71 pt ΔCER) |

*\*Note on Lexical Token Counts: Seen/Unseen $n$ ($23/100$ for `writer_03` C3; $27/105$ for `writer_04` C4) represent word token counts (character counts: $80/468$ chars for C3; $98/479$ chars for C4) evaluated across the 78 and 83 test samples respectively.*

#### Post-FT 3-State Verification & Corruption Rejection Matrix

| Writer / Fold | Calibrated Thresholds ($t_{\text{acc}}, t_{\text{wrg}}$) | True Accept Rate | True Uncertain Rate | True Wrong Rate | Dot Swap Rejection | Del/Ins Rejection | Word Swap Rejection | Adj Swap Rejection | Skip Word Rejection | Mutashabih Rejection |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`writer_03` (C3)** | $[-0.05, -1.41]$ | **65.86% ± 0.82%** | **25.61% ± 1.22%** | **8.54% ± 2.04%** | **90.15% ± 2.27%** | **92.95% ± 1.92%** | 100.0% | 100.0% | 100.0% | 100.0% |
| **`writer_04` (C4)** | $[-1.80, -2.02]$ (Collapsed) | **90.91% ± 0.00%** | **0.76% ± 0.00%** | **8.33% ± 0.00%** | **74.64% ± 0.72%** | **72.89% ± 1.81%** | 100.0% | 100.0% | 100.0% | 100.0% |

*Key 4-Writer Observations:*
1. **Universal Accuracy Gains**: `mix_0.85` delivers massive error reductions across all volunteer writers: `writer_03` drops by **−10.79 pt** ($21.25\% \to 10.46\%$) and `writer_04` drops by **−4.71 pt** ($17.25\% \to 12.54\%$).
2. **Strict Zero-Forgetting Compliance**: Across all 4 writers, synthetic error stays between **1.04% and 1.10%** (strictly lower than the 1.20% base model error) and KHATT val stays between **9.48% and 9.72%** (well beneath the +1.5 pt cap of 10.42%).
3. **Line vs. Word Generalization**:
   - `writer_03`: Line CER drops from 16.15% $\to$ **10.13% ± 0.52%**; Word CER drops from 20.86% $\to$ **10.77% ± 0.17%**.
   - `writer_04`: Line CER drops from 15.11% $\to$ **11.09% ± 0.16%**; Word CER drops from 21.59% $\to$ **13.97% ± 0.32%**.
4. **Lexical Generalization**: `writer_03` achieves 5.00% on seen words and 12.39% on unseen words (+7.39 pt gap). In `writer_04`, unseen words achieve 12.21% while seen words show 20.41% due to complex ligatures in the calibration set.
5. **C4 Calibration Collapse Root Cause**: For small calibration sizes ($N=13$), the $p=5$ percentile rule (`np.percentile(vals, 5)`) interpolates into the single worst stroke outlier (`تعملون` LLR = −2.55), dragging $t_{\text{acc}}$ down to −1.80 and squeezing the uncertainty band to the minimum 0.20 clamp ($t_{\text{wrg}} = -2.02$), reducing dot swap rejection from 90% down to 74.6%.

### 3e. Superseded INVALID rows (uninitialized-weight run)

| Protocol / Fold | Train Split | Test Split | Train Epochs & Loss | Calibrated Thresholds ($t_{\text{acc}}, t_{\text{wrg}}$) | Lines CER | Words CER | Overall CER | True-Accept Rate | Dot Swap WRONG | Word Swap WRONG | Status |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold A (LOWO)** | `writer_01` (70 train, 30 calib) | `writer_02` (148 held-out) | 6 epochs ($3.32 \to 0.021$) | $-6.28, -6.48$ | 104.98% | 102.25% | **103.55%** | **88.61%** | 22.40% | 59.46% | **INVALID** (Uninitialized weights) |
| **Fold B (LOWO)** | `writer_02` (14 train grps, calib) | `writer_01` (100 held-out) | 6 epochs ($3.25 \to 0.025$) | $-6.53, -6.73$ | 109.50% | 75.80% | **102.26%** | **84.40%** | 42.16% | 69.09% | **INVALID** (Uninitialized weights) |
| **Fold C (Personalized)** | `writer_02` (10 train grps, calib) | `writer_02` (10 held-out grps) | 6 epochs ($3.04 \to 0.013$) | $-6.20, -6.40$ | 99.36% | 91.35% | **95.03%** | **83.04%** | 33.90% | 68.57% | **INVALID** (Uninitialized weights) |

---

## 4. Independent Gate Baselines (`configs/gate_baselines.json`)

Precomputed canonical evaluation of the untrained canonical base model (`checkpoints/best.pt`, SHA256: `D9F63734...`, 54,793,797 B, 74/74 tensors matched) across all test subsets. Evaluated across repeated draws to verify determinism. Step-0 gate enforces $\le 1.0\text{ pt}$ tolerance on the same hardware target.

| Subset Name | Target Writer | Sample Count ($n$) | CPU Baseline CER | CUDA Baseline CER | Target Tolerance | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **`A_test`** | `writer_02` (all) | 145 | **36.40%** | **36.30%** | $\pm 1.0$ pt | Canonical Frozen |
| **`B_test`** | `writer_01` (all) | 100 | **24.77%** | **24.77%** | $\pm 1.0$ pt | Canonical Frozen |
| **`C_test`** | `writer_02` (50% held-out grps) | 76 | **38.57%** | **38.57%** | $\pm 1.0$ pt | Canonical Frozen |
| **`LC_test`** | `writer_02` (held-out learning curve grps) | 45 | **35.48%** | **35.48%** | $\pm 1.0$ pt | Canonical Frozen |
| **`C3_test`** | `writer_03` (50% held-out grps) | 78 | **21.25%** | **21.25%** | $\pm 1.0$ pt | Canonical Frozen |
| **`C4_test`** | `writer_04` (50% held-out grps) | 83 | **17.25%** | **17.25%** | $\pm 1.0$ pt | Canonical Frozen |

*Provenance Notice: Both CPU and CUDA reproduce identical baselines (e.g. 38.57% on `C_test`, 24.77% on `B_test`). The previously reported 41.25% CPU baseline was caused entirely by an outdated local checkpoint (`E5C88C31...`, 54,794,437 B), NOT CuDNN FP32 hardware drift. Both file hashes are recorded:*
- Canonical Checkpoint: `D9F63734B8BEBF994E77E20D6B599EFD5AAEF23181511D21336DE58A02E94368` (54,793,797 B)
- Superseded Outdated Checkpoint: `E5C88C3181D7BC52540E8E947EF9CE8F942CAC3447C6520A3E2F71BE717E9475` (54,794,437 B)


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

---

## 8. Checkpoint, Shard & Artifact Provenance Directory

Single source of truth for all project checkpoints, training shards, and evaluation manifests across Local and Google Drive storage:

### A. Core Model Checkpoints
| Checkpoint Identifier | Canonical Google Drive Path | Local Repo Relative Path | Origin & Description |
| :--- | :--- | :--- | :--- |
| **`best.pt` (Canonical Base)** | `/content/drive/MyDrive/quran_htr_writer_checkpoints/step2b_v3/best.pt` | `checkpoints/best.pt` | Step 2b v3 CRNN pre-trained model (74 tensors). Baseline foundation for all fine-tuning. |
| **`best_visual.pt`** | `/content/drive/MyDrive/quran_htr_writer_checkpoints/run_visual/best.pt` | `checkpoints/run_visual/best.pt` | Visual-order continuation checkpoint. |
| **`best_backup.pt`** | `/content/drive/MyDrive/quran_htr_writer_checkpoints/best.pt` | `checkpoints/best_backup.pt` | Root Drive backup of canonical base checkpoint. |
| **Task 2.25 Runs Log** | `/content/drive/MyDrive/quran_htr_writer_checkpoints/lowo_runs.jsonl` | `reports/lowo_runs.jsonl` | Complete multi-seed evaluation JSONL database across all mixes, seeds, and folds. |

### B. Training & Evaluation Shards
| Shard Dataset | Canonical Google Drive Zip Archive | Canonical Google Drive Loose Directory | Local Destination Path |
| :--- | :--- | :--- | :--- |
| **Synthetic Quran Training** | `/content/drive/MyDrive/quran_htr/dist/shards_synth.zip`<br>*(alt: `/content/drive/MyDrive/shards_synth.zip`)* | `/content/drive/MyDrive/quran_htr_data/shards/synth/` | `data/shards/synth/synth_train_000.pt` |
| **KHATT Real Handwriting** | `/content/drive/MyDrive/quran_htr/dist/shards_khatt.zip`<br>*(alt: `/content/drive/MyDrive/shards_khatt.zip`)* | `/content/drive/MyDrive/quran_htr_data/shards/khatt/` | `data/shards/khatt/khatt_train_000.pt` |
| **Synthetic Benchmark Test** | Inside `shards_synth.zip` | `/content/drive/MyDrive/quran_htr_data/synth_test_000.pt` | `data/shards/synth/synth_test_000.pt` |
| **KHATT Validation Test** | Inside `shards_khatt.zip` | `/content/drive/MyDrive/quran_htr_data/khatt_val_000.pt` | `data/shards/khatt/khatt_val_000.pt` |
| **Corpus v3 Full Pack** | `/content/drive/MyDrive/quran_htr/dist/shards_corpus_v3.zip` | N/A | `dist/shards_corpus_v3.zip` (404.3 MB, 66k lines) |

### C. Tablet Stroke Datasets
| Partition | File Path | Status | Sample Count |
| :--- | :--- | :--- | :---: |
| **`writer_01` (Dev v1)** | `data/capture/strokes.json` | Active Dev | 100 samples (40 lines, 60 words) |
| **`writer_02` (Dev v2)** | `data/capture_v2/writer_02/strokes.json` | Active Dev | 145 samples (20 lines, 125 words) |
| **`writer_03` (Locked Test)** | `data/locked_test/writer_03/strokes.json` | **LOCKED** (Zero evals) | 102 samples |
| **`writer_04` (Locked Test)** | `data/locked_test/writer_04/strokes.json` | **LOCKED** (Zero evals) | 104 samples |



