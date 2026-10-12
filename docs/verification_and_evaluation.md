# Verification Algorithm & Evaluation Protocols

## 1. Why Recitation Verification Differs from Standard HTR

In standard OCR/HTR, the goal is greedy sequence decoding: $\arg\max_W P(W \mid X)$. In Quranic recitation and memorization checking, this is insufficient because:
1. **Model Bias Towards High-Frequency Words**: Standard decoders frequently "correct" minor user handwriting flaws into common dictionary words, masking spelling errors.
2. **Dilution of Single-Letter Errors**: A student who writes "ت" instead of "ب" or drops a single letter in a 6-word verse still gets $95\%$ of characters right. A naive CER threshold or line-level score accepts the line as correct, failing the core educational requirement.
3. **Discrete Decision Requirement**: Mobile apps require explicit 3-state feedback:
   - **ACCEPT** (Green): High confidence match.
   - **UNCERTAIN** (Amber): Ambiguous ink; prompt user to inspect or re-write.
   - **WRONG** (Red): Definitive recitation error detected.

---

## 2. The 3-State Verification Algorithm

The verification pipeline is formally specified in Section 9 of [inference_contract.md](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/docs/inference_contract.md) and implemented in [verify_ref.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/verify_ref.py).

```
Model Log-Probabilities L [T, V]    Expected Text: "وقيل يا أرض ابلعي"
               │                                      │
               │ (Visual-order target)                │
               ▼                                      ▼
┌─────────────────────────────────────────────────────────────┐
│ 1. Global CTC Forced Alignment (Viterbi)                    │
│    - Compute optimal path π* over interleaved blank trellis │
│    - Map character indices to temporal frame spans [t1, t2] │
└─────────────────────────────────────────────────────────────┘
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
┌───────────────────────────────┐ ┌───────────────────────────┐
│ 2. Forced Alignment Score     │ │ 3. Free-Path Score        │
│    s_exp = Viterbi(L[t1:t2+1])│ │    s_free = Σ max_v L_t,v │
└───────────────────────────────┘ └───────────────────────────┘
               │                               │
               └───────────────┬───────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 4. Per-Character Normalized LLR                             │
│    LLR = (s_exp - s_free) / max(1, num_chars)               │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│ 5. 3-State Decision Policy                                  │
│    - LLR >= t_accept (-0.60) ──> ACCEPT                     │
│    - LLR <= t_wrong  (-0.95) ──> WRONG                      │
│    - Otherwise               ──> UNCERTAIN                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. The Normalization Breakthrough: Per-Character vs. Per-Frame

During earlier iterations (Task 2.17–2.18), LLR was normalized by temporal frame span: $\text{LLR}_{\text{frame}} = \frac{s_{\text{exp}} - s_{\text{free}}}{t_2 - t_1 + 1}$.

### The Failure of Frame Normalization
In handwriting, a single word spans $25–40$ temporal frames. A single substituted dot or inserted letter only causes a penalty at $2–3$ frames (e.g. $\Delta \approx -10.0$). Dividing $-10.0$ by $35$ frames yielded $\text{LLR} \approx -0.28$, which easily cleared $t_{\text{wrong}} = -0.95$ and was accepted as correct.
- Frame-length normalization caught only **$29.8\%$** of dot-family swaps and **$38.5\%$** of letter deletions on synthetic tests.

### The Resolution: Per-Character Normalization
Normalizing by the number of characters in the word:
$$\text{LLR}_{\text{char}} = \frac{s_{\text{exp}} - s_{\text{free}}}{\max(1, \text{len}(\text{enc}(w_i)))}$$

Dividing the same $\Delta = -10.0$ by $4$ characters yields $\text{LLR} = -2.50$, cleanly falling below $t_{\text{wrong}} = -0.95$.
- **Empirical Impact**:
  - Synthetic test shards: Dot-family detection jumped from **$29.8\% \to 100.0\%$ WRONG**.
  - Letter deletion/insertion jumped from **$38.5\% \to 100.0\%$ WRONG**.
  - True claim acceptance remained unaffected at **$100.0\%$ ACCEPT**.
  - Real writer 02 data: Single-letter corruptions caught at **$90.3\%–91.0\%$ WRONG**.

---

## 4. Operating Threshold Sets

| Parameter | Relaxed Mode (Default / Everyday) | Strict Mode (Rigorous Certification) |
| :--- | :---: | :---: |
| **$t_{\text{accept}}$** | **$-0.60$** | **$-0.40$** |
| **$t_{\text{wrong}}$** | **$-0.95$** | **$-0.80$** |
| **Calibration Target** | 95% True Acceptance on DEV ink | 99% Corruption Rejection |
| **True Accept (w01)** | **93.7%** | 84.1% (excessive false reject) |
| **Corrupted Wrong** | **$\ge 90.3\%$** | $\ge 98.5\%$ |

---

## 5. Closed-Set Word Ranking Protocol

To measure how well the acoustic/visual model discriminates words within a closed Quran vocabulary without language model bias, word crops are ranked against candidate dictionaries:

### A. Full Dictionary Protocol (14,872 Words)
- Candidate set: All **14,872 unique words** in Tanzil Quran encodable in `vocab.json`.
- Each candidate $W_c$ is scored via CTC loss against the image feature map.
- The rank is the 1-based position of the true word among all 14,872 candidates sorted by ascending CTC loss.
- **Batched PyTorch Acceleration**: Evaluating 14,872 candidates serially in Python took ~20 seconds per crop. Batching candidates in chunks of 1024 through `torch.nn.CTCLoss(reduction="none")` evaluates the entire dictionary for 125 words in **2.5 seconds**.

### B. 200-Distractor Protocol (Labeled)
- Introduced in Task 2.18 as an approximation: 200 random distractors are sampled from the dictionary alongside the ground truth.
- Scaled rank formula: $\text{Rank}_{\text{scaled}} = 1 + \text{round}\left(\text{higher\_distractors} \times \frac{14872}{200}\right)$.
- **Empirical Relationship on writer_02 (Canonical 1.0x)**:
  - Full Dictionary: Top-1 **42.40%**, Top-5 **60.80%**, Median Rank **2.0**.
  - 200-Distractor: Top-1 **85.60%**, Median Rank **1.0**.
  - *Note*: The 200-distractor metric is substantially more lenient than full-dictionary evaluation because random distractors rarely include hard orthographic neighbors. Both metrics are always reported side-by-side.

---

## 6. Golden Reference Vectors (`tests/golden/golden_vectors.json`)

To ensure that mobile client ports (Flutter/Dart or iOS Swift) achieve bit-level behavioral parity with the Python reference, 20 test vectors (10 lines, 10 words) are committed in [tests/golden/golden_vectors.json](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/tests/golden/golden_vectors.json).

### Validation Invariants:
1. **Exact Decision State Match**: Every word verdict (`ACCEPT`, `UNCERTAIN`, `WRONG`) must match the golden verdict identically.
2. **Numerical Tolerance**: Computed LLR must satisfy $|\Delta\text{LLR}| \le 0.05$ (accounting for FP32/FP16 SIMD differences across architectures).
3. **Execution**:
   ```bash
   python scripts/verify_ref.py --golden tests/golden/golden_vectors.json
   # Output: 20/20 PASS
   ```
