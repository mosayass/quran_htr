# Preprocessing, Geometry & Ink Capture

## 1. The Geometry Shift: v1 Legacy vs. v2 Canonical Band

A major source of confusion in earlier evaluation phases was the discrepancy in physical canvas and ruled-band dimensions across capture versions.

```
       v1 Capture Geometry (writer_01)                 v2 Capture Geometry (writer_02)
┌───────────────────────────────────────────┐   ┌───────────────────────────────────────────┐
│ Canvas: 960 x 240 CSS px                  │   │ Canvas: 956 x 240 CSS px (DPR = 1.5)      │
│                                           │   │                                           │
│ ─── Y_top = 50.0 px ───────────────────── │   │ ─── Y_top = 72.0 px ───────────────────── │
│                                           │   │                                           │
│                                           │   │     Band Height = 108.0 CSS px            │
│     Band Height = 150.0 CSS px            │   │     (CANONICAL REFERENCE)                 │
│                                           │   │                                           │
│ ─── Y_base = 160.0 px ─────────────────── │   │ ─── Y_base = 147.0 px ─────────────────── │
│ ─── Y_bot = 200.0 px ──────────────────── │   │ ─── Y_bot = 180.0 px ──────────────────── │
└───────────────────────────────────────────┘   └───────────────────────────────────────────┘
  * Ink Extent: Median 78.2 px (52.1% of band)    * Ink Extent: Median 92.9 px (86.0% of band)
  * Stylus Pressure: Median 0.698 (heavy ink)     * Stylus Pressure: Median 0.216 (light ink)
```

### Why 1.4x Zoom Worked on v1 but Failed on v2
- In **v1**, the wide $150\text{ px}$ band caused `writer_01` to write relatively small strokes that occupied only **52.1%** of the band height. Applying a $1.4\times$ zoom magnified the small strokes to occupy $\sim 73\%$ of the $64\text{ px}$ CNN input height, dropping CER from $27.02\%$ to **$22.66\%$**.
- In **v2**, the band was redesigned to a tighter **$108\text{ px}$**. `writer_02` naturally filled **$86.0\%$** of the band. When an unthinking $1.4\times$ zoom was applied to `writer_02`, the strokes were magnified to over $120\%$ of the input window, clipping ascenders/descenders and dilating stroke bodies into heavy blobs. This blew up `writer_02` CER to **$53.66\%$**.
- Evaluating `writer_02` at its **canonical $1.0\times$ scale** immediately lowered CER to **$37.24\%$** (lines CER dropped from $44.02\%$ to $33.59\%$, words CER from $63.08\%$ to $40.82\%$).

---

## 2. Canonical Zoom Semantics Contract

To ensure unified behavior across all past and future capture sets, the **canonical reference band is locked to the v2 band ($108\text{ CSS px}$)**:

$$\text{Canonical Band Height} = 108.0\text{ CSS px}$$

### Rules by Sample Version:
1. **v2 Native Samples (`data/capture_v2/`)**:
   $$\text{zoom}_{\text{canonical}} = 1.0$$
2. **v1 Legacy Samples (`data/capture/`)**:
   $$\text{zoom}_{\text{v1}} = \frac{150.0}{108.0} \approx 1.388889 \quad (\sim 1.39)$$
   *(This exactly scales the legacy 150px band to parity with the canonical 108px band).*

### Zoom Grid Diagnostic Sweep on `writer_02`:
| Zoom Factor | Overall CER | Lines CER (20) | Words CER (125) | Overall WER | Observations |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.70x** | 35.18% | 33.59% | 36.73% | 81.74% | Compact, minimal clipping |
| **0.85x** | 35.46% | 32.64% | 38.22% | 82.17% | Well-proportioned |
| **1.00x** | **37.24%** | **33.59%** | **40.82%** | **87.83%** | **Canonical standard** |
| **1.15x** | 43.43% | 37.19% | 49.54% | 95.65% | Slight dilation begins |
| **1.30x** | 48.78% | 41.37% | 56.03% | 97.39% | Severe ascender crowding |
| **1.40x** | 53.66% | 44.02% | 63.08% | 101.30% | Severe clipping & dilation |

---

## 3. Stroke Rendering Pipeline (`scripts/render_strokes.py`)

Raw captured samples consist of timestamped, pressure-sensitive 2D points $[(x_i, y_i, p_i, t_i)]$. To achieve parity with real-ink training corpora (KHATT), the rendering engine enforces strict invariants:

1. **Pressure Invariance**:
   Stylus pressure is removed during rasterization. All strokes are rendered with **constant stroke width**. Pressure variation creates inconsistent ink mass that degrades CNN character features.
2. **4x Supersampling**:
   Vector paths are rendered on an internal canvas at $4\times$ resolution ($H = 256\text{ px}$) using high-quality anti-aliasing, then downscaled using Lanczos/Bicubic resampling to $H = 64\text{ px}$. This prevents pixel aliasing on thin diacritics.
3. **Stroke Width After Zoom**:
   Nominal stroke width is locked to **$2.2\text{ px}$** post-zoom (within the human handwriting window of $1.8–2.6\text{ px}$).
4. **Baseline Centering**:
   Vertical coordinates are centered relative to the writing baseline:
   $$Y_{\text{center}} = Y_{\text{base}} - Y_{\text{top}}$$
5. **Horizontal Safety Margins & Multiple-of-32 Alignment**:
   A horizontal safety margin of $\text{pad\_x} = 16.0\text{ px}$ is added to the left and right of the bounding box. The resulting width $W$ is padded on the right with white background ($255$) to a multiple of 32.

---

## 4. Training Augmentations (`StrokeDataset` in `dataset.py`)

During few-shot and cross-writer fine-tuning, [StrokeDataset](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/dataset.py) applies online augmentations around the canonical scale:

```python
# Canonical zoom resolution
canonical_zoom = 1.0 if geom else (150.0 / 108.0)

# Training Augmentation
zoom = canonical_zoom * random.uniform(0.8, 1.3)
width = random.uniform(1.8, 2.6)
rotation = random.uniform(-3.0, 3.0)       # degrees
shear_x  = random.uniform(-0.15, 0.15)
wobble   = random.uniform(0.0, 1.2)        # sinusoidal baseline wobble
jitter   = random.uniform(0.0, 0.4)        # coordinate anchor jitter
```

---

## 5. Capture Studios & Applications

The repository contains two independent web-based capture interfaces:

### A. Full Capture Studio ([capture/index.html](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/capture/index.html))
- **Scope**: Complete 145-prompt protocol (20 lines, 105 paired words, 20 extra words).
- **Features**: Toggle between fully vowelled text (مشكول) and rasm-only (رسم), line vs. word mode switches, prompt grouping, real-time guidelines rendering, duplicate-resolving save-and-next, and pure JS ZIP export.
- **Used For**: Comprehensive benchmarks (`writer_01`, `writer_02`).

### B. Shareable Short-Protocol Studio ([capture/shareable.html](file:///c:/Users/mosa/OneDrive/Desktop/Hafiz/quran_htr/capture/shareable.html))
- **Scope**: Streamlined **40-sample protocol** (10 lines + 30 words), estimated at **10–12 minutes** completion time.
- **Alphabet Balance**: Covers all 28 Arabic letter families and contains multiple occurrences of لفظ الجلالة ("الله" / "لله").
- **Consent Modal**: Includes an introductory Arabic consent screen detailing research purpose, anonymity, and data collection scope.
- **Anonymous Writer ID**: Automatically assigns persistent IDs (`writer_anon_xxxxxx`) via `localStorage`.
- **Zero Server Dependencies**: Embedded lightweight pure-JavaScript ZIP builder (`createStoredZip`). Generates and downloads `hafiz_capture_{anonId}_{timestamp}.zip` directly in the tablet browser. Deployable as a single static file on GitHub Pages, tablet file managers, or static CDNs.
