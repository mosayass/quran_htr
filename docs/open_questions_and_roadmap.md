# Open Questions, Unresolved Gaps & Roadmap

This document outlines key technical invariants to maintain, known gaps in the current implementation, and the recommended roadmap for future engineering phases.

---

## 1. Critical Rules & Invariants to Remember Before Continuing

When working on this repository, future developers and AI agents must preserve the following architectural contracts:

1. **Canonical Zoom Invariant**:
   - For all **v2 captures and future tablet collections**, the canonical band is strictly **$108\text{ CSS px}$** ($\text{zoom} = 1.0$).
   - Never apply legacy $1.4\times$ zoom to v2 data. Stroke dilation directly degrades the VGG-lite receptive field.
   - Legacy **v1 data** (`writer_01`) defaults to $\text{zoom} = \frac{150.0}{108.0} \approx 1.3889$ to normalize its $150\text{ px}$ band down to the $108\text{ px}$ reference.
2. **Per-Character Normalization Invariant**:
   - Always normalize verification LLR by character length ($\max(1, \text{num\_chars})$).
   - Never revert to frame-length normalization; doing so dilutes single-letter and dot-swap penalties across long time spans, destroying error sensitivity.
3. **Visual-Order Invariant**:
   - The model consumes Left-to-Right visual time steps.
   - Arabic text must be reversed prior to CTC loss and forced alignment: $\text{target}_{\text{visual}} = \text{target}_{\text{logical}}[::-1]$.
4. **Prompt Group Invariant**:
   - Never randomly split word and line samples independently.
   - A line crop and all its constituent paired word crops must be assigned to the same train or test split to prevent lexical leakage.
5. **Client-Side Export Invariant**:
   - Capture studios must remain 100% client-side with zero external server dependencies, enabling volunteers to participate offline on any tablet browser.

---

## 2. Identified Technical Gaps & Status

### A. GPU Execution of Cross-Writer Fine-Tuning Folds (A, B, C)
- **Current Status**: Staged and verified. The configuration, data mixing pipeline, prompt group split JSONs, and Colab cells are implemented and tested locally.
- **Gap**: Running the 36,000-sample training loop (6 epochs $\times$ 6k samples) on CPU is computationally prohibitive (~3–4 hours per fold). Running these folds on a Colab GPU instance (T4 / V100, ~3 minutes per fold) is required to obtain empirical post-adaptation CER and verify that `writer_02` false rejections drop from 48.3% to below 5%.

### B. Vocabulary Reconcilement for 102 Tanzil Tokens
- **Current Status**: 14,872 out of 14,974 Tanzil words are encodable in `vocab.json` (72 classes).
- **Gap**: Exactly 102 tokens in [quran-simple-clean.txt](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/data/tanzil/quran-simple-clean.txt) contain archaic non-standard Unicode glyphs (e.g., specific dagger alif forms or Qur'anic stop marks).
- **Next Step**: Either implement a canonical Unicode normalization pre-pass mapping these 102 tokens into the 72-class space, or expand `vocab.json` by 2–4 classes to achieve 100.0% coverage of all 14,974 tokens in the Tanzil text.

### C. Streaming / Real-Time "Growing Line" Verification
- **Current Status**: Verification evaluates complete static line or word crops after pen-up.
- **Gap**: In the full Hafiz application, a user writes continuously across a ruled notebook page. The engine needs to detect word-end pauses (e.g. inter-word pen lifts $>400\text{ ms}$ or horizontal gaps $>30\text{ px}$) and perform incremental forced alignment over the growing stroke sequence without waiting for the entire verse to finish.

### D. Flutter / Dart Client Porting
- **Current Status**: Pure NumPy reference implementation [scripts/verify_ref.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/verify_ref.py) and 20 golden test vectors [tests/golden/golden_vectors.json](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/tests/golden/golden_vectors.json) are complete and pass 20/20.
- **Gap**: The production Flutter mobile client needs a Dart implementation of `ctc_viterbi_alignment` and `verify_reference`, backed by ONNX Runtime Mobile or TFLite.

---

## 3. Recommended Next Implementation Steps

1. **Execute Fold A, B, and C in Google Colab**:
   Using the ready-to-run 12-line notebook cells, fine-tune the model with the 0.5 strokes / 0.3 synth / 0.2 KHATT mix, and record post-personalization CER on held-out test splits.
2. **Collect Additional Writers via `capture/shareable.html`**:
   Deploy [capture/shareable.html](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/capture/shareable.html) statically and gather short-protocol datasets from 5–10 independent tablet users to expand cross-writer generalization.
3. **Implement Dart Verification Package**:
   Port the verification logic from `scripts/verify_ref.py` to Dart and run automated tests asserting exact state and LLR tolerance ($|\Delta\text{LLR}| \le 0.05$) against `tests/golden/golden_vectors.json`.
