# Writer Experiments, Benchmarks & Error Analysis

## 1. Demographic & Physical Geometry Comparison

Two human writers have contributed real tablet ink to date. Their physical and dynamic properties differ substantially:

| Property | `writer_01` (Capture v1) | `writer_02` (Capture v2) | Analytical Significance |
| :--- | :---: | :---: | :--- |
| **Dataset Path** | `data/capture/strokes.json` | `data/capture_v2/writer_02/strokes.json` | Distinct schema versions |
| **Total Samples** | 100 unique (40 lines, 60 words) | 145 unique (20 lines, 125 words) | Deduplicated, latest-save wins |
| **Ruled Band Height** | 150.0 CSS px | **108.0 CSS px** (Canonical) | Band redesigned for v2 |
| **Ink-Extent in Band** | Median **52.1%** (mean 57.2%) | Median **86.0%** (mean 86.2%) | w02 fills almost entire band |
| **Body-Height Proxy** | Median **78.2 px** | Median **92.9 px** | w02 writes physically larger |
| **Stylus Pressure** | Median **0.698** (Heavy) | Median **0.216** (Very Light) | w02 writes with gentle touch |
| **Sampling Rate** | Median **90.9 Hz** | Median **90.9 Hz** | Identical PointerEvents rate |
| **Device / Canvas** | 960x240 CSS px, DPR ~1.0 | 956x240 CSS px, DPR 1.5 | Windows / Surface active pen |

---

## 2. Baseline Performance on Held-Out `writer_02` (Zero Training)

Evaluated with the un-adapted production model ([checkpoints/best.pt](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/checkpoints/best.pt)) under frozen `writer_01` DEV thresholds ($t_{\text{accept}} = -0.60, t_{\text{wrong}} = -0.95$):

### Accuracy & CER Comparison by Preprocessing Policy
| Metric | `writer_01` (DEV, 1.4x zoom) | `writer_02` (Provisional 1.4x) | `writer_02` (Canonical 1.0x, 2.2px) |
| :--- | :---: | :---: | :---: |
| **Overall CER** | **24.07%** | 53.66% | **37.24%** |
| **Lines CER** | **19.52%** | 44.02% | **33.59%** |
| **Words CER** | **34.15%** | 63.08% | **40.82%** |
| **Overall WER** | **64.68%** | 101.30% | **87.83%** |

### Closed-Set Word Ranking (125 Words on `writer_02`)
| Protocol | Top-1 Accuracy | Top-5 Accuracy | Top-10 Accuracy | Median Rank |
| :--- | :---: | :---: | :---: | :---: |
| **Full Dict (14,872 words)** | **42.40%** | **60.80%** | **68.80%** | **2.0** |
| **200-Distractor Protocol** | **85.60%** | 85.60% | 85.60% | **1.0** |

*Under canonical 1.0x, median rank dropped from 19.0 down to 2.0, showing strong lexical discrimination despite handwriting domain shift.*

### 3-State Verification Outcomes (Canonical 1.0x, Provenance: writer_02 Held-Out, Frozen w01 DEV Thresholds)
- **TRUE Claims** (230 words in 145 samples):
  - **True-Accept Rate**: **41.3% ACCEPT**
  - **UNCERTAIN**: **10.4%**
  - **WRONG**: **48.3%** *(False reject caused by domain shift of unadapted writer)*
- **CORRUPTED Claims Detection Rate (WRONG)** (Paired with 41.3% True-Accept):
  - **Dot Swap**: **91.0% WRONG** (2.5% ACCEPT, 6.6% UNC)
  - **Del / Ins**: **90.3% WRONG** (4.8% ACCEPT, 4.8% UNC)
  - **Word Swap**: **100.0% WRONG** (0.0% ACCEPT)
  - **Adjacent Swap**: **85.0% WRONG** (10.0% ACCEPT, 5.0% UNC)
  - **Skip Word**: **85.0% WRONG** (15.0% ACCEPT, 0.0% UNC)
  - **Mutashabih**: **100.0% WRONG** (0.0% ACCEPT)

---

## 3. Self-Calibration Analysis: Retraction of Invalid Sweep & Valid Protocol Results

### A. Retraction of Task 2.21 Inverted Sweep
The previous claim that self-calibration "proved fine-tuning is mandatory" was **RETRACTED as INVALID**:
- In the initial sweep, independent percentiles yielded $t_{\text{accept}} = -2.9919$ and $t_{\text{wrong}} = -1.4860$.
- Because $t_{\text{accept}} < t_{\text{wrong}}$, the monotonic ordering contract ($t_{\text{wrong}} \le t_{\text{accept}}$) was violated. Scores between $-2.99$ and $-1.49$ were logically undefined / simultaneously triggered contradictory rules.

### B. Valid Monotonic Self-Calibration Protocol (Task 2.22 Item C)
A mathematically valid evaluation was executed ([scripts/test_valid_self_calibration.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/test_valid_self_calibration.py)) enforcing strict monotonicity:
$$t_{\text{wrong}} \le t_{\text{accept}}$$
- **Monte Carlo Setup**: 20 independent random draws evaluated across calibration sample sizes $N \in \{10, 20, 40\}$ drawn from `writer_02` prompt groups, evaluated on remaining held-out groups with zero neural network fine-tuning.
- **Results with Spread (Mean ± Std over 20 draws)**:

| Metric / Error Type | $N = 10$ Calibration Samples | $N = 20$ Calibration Samples | $N = 40$ Calibration Samples |
| :--- | :---: | :---: | :---: |
| **Calibrated $t_{\text{accept}}$** | $-2.50 \pm 0.46$ | $-2.32 \pm 0.35$ | $-2.47 \pm 0.27$ |
| **Calibrated $t_{\text{wrong}}$** | $-2.70 \pm 0.46$ | $-2.52 \pm 0.35$ | $-2.67 \pm 0.27$ |
| **TRUE Claims (True-Accept Rate)** | **$87.5\% \pm 8.1\%$** | **$84.8\% \pm 6.5\%$** | **$87.2\% \pm 5.5\%$** |
| **Dot Swap Detection (WRONG)** | $40.3\% \pm 12.6\%$ | $45.0\% \pm 10.2\%$ | $40.5\% \pm 8.5\%$ |
| **Letter Del / Ins (WRONG)** | $29.4\% \pm 12.9\%$ | $33.3\% \pm 10.1\%$ | $29.1\% \pm 7.8\%$ |
| **Adjacent Word Swap (WRONG)** | $63.1\% \pm 13.0\%$ | $70.0\% \pm 11.7\%$ | $64.0\% \pm 11.9\%$ |
| **Skip Word (WRONG)** | $68.9\% \pm 5.7\%$ | $70.9\% \pm 5.5\%$ | $72.6\% \pm 10.4\%$ |
| **Mutashabih Substitution (WRONG)** | $92.3\% \pm 2.9\%$ | $93.4\% \pm 2.2\%$ | $92.7\% \pm 2.3\%$ |
| **Whole Word Swap (WRONG)** | **$97.0\% \pm 1.5\%$** | **$97.8\% \pm 1.1\%$** | **$97.2\% \pm 1.6\%$** |

### Empirical Takeaway:
When thresholds are recalibrated monotonically to achieve acceptable True-Accept rates ($\sim 85\%–87\%$), **macro word errors remain reliably detected ($97\%–98\%$)**, but **subtle single-character corruptions (dot swaps $\sim 40\%–45\%$, deletions $\sim 29\%–33\%$) suffer substantial degradation**. This confirms that without weight fine-tuning, threshold tuning alone cannot resolve fine-grained character boundary confusion on new handwriting styles.

---

## 4. Cross-Writer Fine-Tuning & Personalization Protocol

To adapt the model to new writers, three training folds are defined in [scripts/train_twofold_lowo.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/train_twofold_lowo.py):

### Prompt Group Splitting Rationale:
In capture sets, lines and their constituent words are paired (e.g. line `وقيل يا أرض ابلعي` is paired with words `وقيل`, `يا`, `أرض`, `ابلعي`). If line 1 is in Train and word 1 is in Test, the model memorizes the exact lexical ink.
**Rule**: Splitting is strictly performed **by prompt group**. A line and all its paired words are permanently assigned to the same split.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FOLD A (Leave-One-Writer-Out):                                              │
│ - Train: writer_01 80% groups (32 lines / 48 words)                         │
│ - Calib: writer_01 20% groups (8 lines / 12 words) -> threshold tuning      │
│ - Test:  writer_02 100% held-out (20 lines / 125 words, zero training leak) │
├─────────────────────────────────────────────────────────────────────────────┤
│ FOLD B (Leave-One-Writer-Out Reversed):                                     │
│ - Train: writer_02 80% groups (16 lines / 100 words)                        │
│ - Calib: writer_02 20% groups (4 lines / 25 words) -> threshold tuning      │
│ - Test:  writer_01 100% held-out (40 lines / 60 words, zero training leak)  │
├─────────────────────────────────────────────────────────────────────────────┤
│ FOLD C (Personalization on writer_02):                                      │
│ - Train: writer_02 50% groups (10 lines + 65 words = 75 samples)            │
│ - Test:  writer_02 50% groups (10 lines + 60 words = 70 samples)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Training Mix & Hyperparameters:
- **Mix Ratio**: `0.5` online strokes / `0.3` synthetic v3 shards / `0.2` KHATT real ink.
- **Schedule**: Fixed **6 epochs $\times$ 6,000 samples** (36,000 samples total).
- **Optimizer & LR**: AdamW, Cosine annealing from **$5\times 10^{-5} \to 5\times 10^{-6}$**.
- **No Test Selection**: Checkpoints are taken strictly at the final epoch to prevent test-set over-fitting.

---

## 5. Visual Error Analysis (Worst 20 vs. Best 10 Samples)

A diagnostic montage ranking all 145 samples of `writer_02` by CER under canonical 1.0x is preserved in [reports/montage_writer_02_worst_best.png](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/reports/montage_writer_02_worst_best.png).

### Characteristics of Best Samples (CER 0.0%):
- Distinct vertical strokes with standard baselines (e.g. `بِسْمِ`, `خَالِدِينَ`).
- Clear separation between words in line crops.
- Ample horizontal spacing between connected components.

### Characteristics of Worst Samples (CER 60%–100%):
1. **Light Pressure Hairlines**: Extremely low stylus pressure ($p < 0.15$) produces fine, faint strokes where ink points occasionally fragment during fast handwriting.
2. **Ascender/Descender Collisions**: In dense verses, non-standard descender loops (e.g. final `ي` or `ر`) plunge below the band boundary, causing vertical truncation.
3. **Cursive Sub-word Ligatures**: Cursively connected character clusters (e.g. `ليستخفوا`) where character bodies blend without clear intermediate baseline segments.
