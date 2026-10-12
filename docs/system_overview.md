# System Overview & Architecture

## 1. High-Level Objective

The **Hafiz Quran HTR** system is designed for interactive Quranic memorization and recitation verification. As a user writes Quranic ayahs and words on a digital tablet canvas using a stylus, the system verifies whether the written ink matches the expected Quranic text, detecting:
- Whole-word swaps and omissions
- Mutashabihat (verses with close lexical similarities) substitutions
- Single-letter insertions, deletions, and dot-family substitutions (e.g. ب vs ت vs ث vs ن vs ي)
- Harakat / vowelling discrepancies

Rather than merely outputting an unconstrained transcription, the system operates as a **verification engine**, producing discrete 3-state decisions (`ACCEPT`, `UNCERTAIN`, `WRONG`) accompanied by normalized confidence metrics.

---

## 2. Neural Architecture (CRNN + CTC)

The primary model is a Convolutional Recurrent Neural Network (CRNN) trained with Connectionist Temporal Classification (CTC) loss:

```
Input Image [B, 1, 64, W]
       │
       ▼
┌────────────────────────────────────────┐
│  CNN Feature Extractor (VGG-Lite)      │
│  - 4 Conv Blocks with BatchNorm & ReLU │
│  - MaxPool2d (vertical 64 -> 1)        │
│  - Horizontal reduction: 4x (W -> W/4) │
└────────────────────────────────────────┘
       │
       ▼ Feature Map [B, 512, 1, T] where T = W / 4
┌────────────────────────────────────────┐
│  Sequence Modeling (Bidirectional GRU) │
│  - 2-layer BiGRU, hidden_size=256      │
│  - Temporal dropout = 0.1              │
└────────────────────────────────────────┘
       │
       ▼ Feature Sequence [T, B, 512]
┌────────────────────────────────────────┐
│  Linear Projection & LogSoftmax        │
│  - Linear(512 -> VocabSize = 72)       │
│  - LogSoftmax(dim=-1)                  │
└────────────────────────────────────────┘
       │
       ▼ Log-Probabilities [T, B, 72]
```

### Key Dimensions
- **Input Height**: Strictly fixed to $64\text{ px}$.
- **Input Width ($W$)**: Dynamic, must be an integer multiple of $32$ ($\text{pad\_w} = (-W) \pmod{32}$).
- **Time Steps ($T$)**: $T = \frac{W}{4}$.
- **Vocab Size ($V$)**: $72$ classes (loaded from [vocab.json](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/vocab.json)).

### Alternative Backbone
- `mobilenetv3_small`: Supported in [model.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/model.py) for lightweight edge execution, but the production standard remains `vgg_lite` due to significantly superior character-level feature discrimination on thin Arabic strokes.

---

## 3. The Visual-Order vs. Logical-Order Contract

Arabic is written Right-to-Left (RTL), but deep learning CNN backbones process spatial images horizontally from Left-to-Right ($x=0 \to x=W$). 

### The Problem
If the target text is fed in logical reading order ($w_1, w_2, \dots, w_N$), the CTC alignment must align character 1 (appearing at the far right of the image) with frame $t=T-1$, and the last character with frame $t=0$. This violates CTC's fundamental assumption that target sequences progress monotonically forward in time ($t=0 \to T-1$).

### The Solution: Visual-Order Reversal
The model is strictly trained and evaluated using **Visual Order**:
- **Image Orientation**: Images are fed in natural un-flipped orientation (the visual start of the text crop is at the right, but the leftmost ink appears at $x=0$, $t=0$).
- **Label Reversal**: The logical Arabic string is reversed:
  $$\text{target}_{\text{visual}} = \text{target}_{\text{logical}}[::-1]$$
- **Decoding**: Greedy CTC decoding produces visual-order text. Output strings are reversed before display:
  $$\text{pred}_{\text{logical}} = \text{pred}_{\text{visual}}[::-1]$$
- **Forced Alignment**: Target word sequences are indexed in visual order ($k = N - 1 - i$).

This contract is locked across all training loops ([train.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/train.py)), ONNX exports ([export_onnx.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/export_onnx.py)), and the verification reference ([verify_ref.py](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/scripts/verify_ref.py)).

---

## 4. Vocabulary & Tokenization

The vocabulary is defined in [vocab.json](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/vocab.json) and comprises **72 classes**:
- **Index 0**: `<blank>` (CTC blank token)
- **Index 1**: `<pad>`
- **Index 2**: `<unk>`
- **Index 3**: ` ` (Space / word delimiter)
- **Indices 4–71**: 68 normalized Arabic glyphs, basic letters, and standard Quranic diacritics (Fathah, Dammah, Kasrah, Sukun, Shaddah, Tanween, Maddah).

### Tanzil Dictionary Coverage (14,872 vs. 14,974 Words)
- Raw Tanzil clean text ([quran-simple-clean.txt](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/data/tanzil/quran-simple-clean.txt)) contains **14,974** unique whitespace-delimited word tokens.
- Exactly **102 tokens** contain non-standard Unicode glyphs (e.g., unnormalized dagger alif variants or archaic presentation forms).
- Filtering for 100% encodability within the 72-class vocabulary leaves exactly **14,872 valid Quran words**. This forms the closed dictionary for all closed-set ranking evaluations.

---

## 5. Model Lineage & Current Production Weights

The production model has evolved through four sequential training phases:

```
[AHCD Dataset]
  │ (Pretrained on Arabic handwritten characters)
  ▼
[Step 1: KHATT Real Ink]
  │ (Pretrained on real Arabic handwritten lines; CER ~18% on KHATT)
  ▼
[Step 2: Synthetic Corpus v3]
  │ (Trained on 40k+ synthetic Quran line shards with 20+ fonts, ruled bands, and diacritics)
  ▼
[Step 2b / Step 3 Checkpoint: checkpoints/best.pt]
  │ (Visual order, VGG-lite backbone, best synthetic validation CER ~1.2%)
  ▼
[Export: dist/onnx/crnn_visual_fp32.onnx]
```

### Production Artifacts
- **PyTorch Weights**: [checkpoints/best.pt](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/checkpoints/best.pt)
- **Production ONNX Model**: [dist/onnx/crnn_visual_fp32.onnx](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/dist/onnx/crnn_visual_fp32.onnx) (18.2 MB, FP32, dynamic width)
- **Quantized Edge Model**: [dist/onnx/crnn_visual_int8.onnx](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/dist/onnx/crnn_visual_int8.onnx) (4.6 MB, INT8 dynamic quantization)
