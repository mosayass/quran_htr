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

### 3-State Verification Outcomes (Canonical 1.0x, Frozen Thresholds)
- **TRUE Claims** (230 words in 145 samples):
  - **ACCEPT**: **41.3%**
  - **UNCERTAIN**: **10.4%**
  - **WRONG**: **48.3%** *(False reject caused by lack of writer adaptation)*
- **CORRUPTED Claims Rejection Rate (WRONG)**:
  - **Dot Swap**: **91.0% WRONG** (2.5% ACCEPT, 6.6% UNC)
  - **Del / Ins**: **90.3% WRONG** (4.8% ACCEPT, 4.8% UNC)
  - **Word Swap**: **100.0% WRONG** (0.0% ACCEPT)
  - **Adjacent Swap**: **85.0% WRONG** (10.0% ACCEPT, 5.0% UNC)
  - **Skip Word**: **85.0% WRONG** (15.0% ACCEPT, 0.0% UNC)
  - **Mutashabih**: **100.0% WRONG** (0.0% ACCEPT)

---

## 3. The Self-Calibration Experiment: Why Fine-Tuning Is Mandatory

To test whether the 48.3% false-rejection rate on `writer_02` could be resolved simply by recalibrating thresholds without retraining the neural network, a self-calibration experiment was conducted ([scripts/test_self_calibration.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/test_self_calibration.py)):

### Protocol:
1. Calibrate thresholds exclusively from the **first 20 samples of `writer_02`**:
   - $t_{\text{accept}} = -2.9919$ (5th percentile of true claims)
   - $t_{\text{wrong}} = -1.4860$ (90th percentile of corrupted claims)
   - *Notice that $t_{\text{accept}} < t_{\text{wrong}}$, indicating extreme distribution overlap!*
2. Evaluate the **remaining 125 samples without any model fine-tuning**.

### Results:
- **TRUE Claims Acceptance**: Jumped from $41.3\% \to \mathbf{94.4\%}$ ACCEPT.
- **CORRUPTED Claims Detection**: **Collapsed from $95.1\% \to \mathbf{64.8\%}$ WRONG**.
  - Dot-family swaps: WRONG dropped to **32.4%** (67.6% missed!).
  - Letter deletions/insertions: WRONG dropped to **24.8%** (75.2% missed!).

### Critical Conclusion:
Domain shift between different human writers shifts the entire un-adapted feature space. Simply loosening detection thresholds on an unadapted model creates catastrophic false accepts on corrupted claims. **Weight fine-tuning (few-shot personalization or LOWO) is mathematically required.**

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
