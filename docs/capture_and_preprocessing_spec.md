# Hafiz Digital Ink Capture & Preprocessing Engineering Specification

> **Target Audience**: Frontend, Web, and Mobile Engineers maintaining or debugging the web capture interface (`capture/index.html` and `capture/shareable.html`).
> **Purpose**: This document defines the strict ML data contract, coordinate system, export schema, and known iPadOS/Safari failure modes so that web client fixes do not break downstream machine learning pipelines.

---

## 1. Executive Summary & System Contract

The Hafiz Quran HTR project collects handwritten Arabic text samples (lines and isolated words) using a browser-based stylus canvas. The output files are packaged as a `.zip` archive on the client device and subsequently fed into a PyTorch CRNN (Convolutional Recurrent Neural Network) training and verification pipeline.

```
┌─────────────────────────────────┐       ┌───────────────────────────────┐
│ Tablet Browser (iPad / Surface) │       │ Downstream ML Pipeline        │
│ capture/index.html              │ ───>  │ Python / PyTorch CRNN         │
│ capture/shareable.html          │  ZIP  │ scripts/render_strokes.py     │
│ [Canvas -> PointerEvents -> JS] │       │ dataset.py (StrokeDataset)    │
└─────────────────────────────────┘       └───────────────────────────────┘
```

### The Cardinal Rule for Frontend Developers
**Do not alter the coordinate system, band ratios, or JSON schema without reading this document.** Downstream models rely on strict physical scale invariants. If coordinates are scaled incorrectly (e.g. multiplied by DPR before saving), the entire dataset becomes invalid for model training.

---

## 2. Downstream Machine Learning Invariants

The neural network model has strict physical expectations:

| Property | Value / Constraint | Rationale |
| :--- | :--- | :--- |
| **Input Tensor Shape** | `[B, 1, 64, W]` | Batch, 1-channel Grayscale, Height strictly **64 px**, Width $W$ multiple of 32 ($W \ge 32$). |
| **Pixel Value Range** | `[-1.0, +1.0]` | Normalized float: $+1.0 = \text{pure white background } (255)$, $-1.0 = \text{black ink } (0)$. |
| **Visual Reading Order** | RTL script, LTR frame sequence | Model processes temporal frames left-to-right. Greedy output is reversed (`text[::-1]`) for logical Unicode reading. |
| **Canonical Ruled Band** | **108.0 CSS px** height | The v2 capture reference band. Baseline at $Y=147$, Top at $Y=72$, Bottom at $Y=180$. |
| **Nominal Stroke Width** | **2.2 px** post-downscale | Natural human writing range is $1.8–2.6\text{ px}$. Thin or bloated strokes degrade character recognition. |
| **Global Target Fill** | **48.0 px** in 64px tensor | Ink should fill ~75% of the 64px tensor height. **Ink must never be clipped vertically.** |
| **Pressure Invariance** | Constant stroke width in rasters | Stylus pressure variation ($p_i$) is recorded in `strokes.json` for research, but flattened during training rasterization to prevent character distortion. |

---

## 3. Ruled Band & Geometry Definition

The capture canvas operates with a ruled 3-guideline system:

```
Y = 0 px   ┌─────────────────────────────────────────────────────────┐
           │                                                         │
Y = 72 px  ├┈┈┈┈┈┈┈┈┈┈ Top Guideline (band_top_css) ┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┈┤ ──┐
           │                                                         │   │
           │                                                         │   │ Band Height = 108 px
Y = 147 px ├───━───━─── Blue Baseline (band_base_css) ─━───━───━─────┤   │ (CANONICAL REFERENCE)
           │                                                         │   │
Y = 180 px ├┈┈┈┈┈┈┈┈┈┈ Bottom Guideline (band_bot_css) ┈┈┈┈┈┈┈┈┈┈┈┈┈┤ ──┘
           │                                                         │
Y = 240 px └─────────────────────────────────────────────────────────┘
```

- **Canvas Size**: `width = 960 CSS px` (or container width), `height = 240 CSS px`.
- **Top Guideline (`band_top_css`)**: $72.0\text{ CSS px}$
- **Baseline (`band_base_css`)**: $147.0\text{ CSS px}$ (where Arabic letters rest)
- **Bottom Guideline (`band_bot_css`)**: $180.0\text{ CSS px}$ (limit for descenders like ر, ز, و)
- **Band Height (`band_height_css`)**: $180.0 - 72.0 = 108.0\text{ CSS px}$

---

## 4. Exported Dataset Schema (The ZIP Package Contract)

When the user taps **"تصدير الحزمة / Download ZIP"**, the browser must generate a `.zip` archive named:
`hafiz_capture_{writer_id}_{timestamp}.zip`

The archive **MUST** contain exactly:

### 1. `manifest.csv`
A comma-separated file with header:
```csv
sample_id,image_file,mode,transcription,writer_id,capture_version,band_top_css,band_base_css,band_bot_css,canvas_w_css,canvas_h_css,dpr
line_01,line_01.png,line,"وقيل يا أرض ابلعي",writer_03,v2,72,147,180,960,240,2.0
word_01,word_01.png,word,"وقيل",writer_03,v2,72,147,180,960,240,2.0
```

### 2. `strokes.json`
A JSON array of all unique sample records:
```json
[
  {
    "id": "line_01",
    "text": "وقيل يا أرض ابلعي",
    "capture_version": "v2",
    "geometry": {
      "band_top_css": 72.0,
      "band_base_css": 147.0,
      "band_bot_css": 180.0,
      "canvas_w_css": 960.0,
      "canvas_h_css": 240.0,
      "dpr": 2.0
    },
    "strokes": [
      [
        {"x": 120.5, "y": 145.2, "p": 0.5, "t": 1728739200000},
        {"x": 121.8, "y": 145.0, "p": 0.52, "t": 1728739200016}
      ]
    ]
  }
]
```

### 3. `{sample_id}.png` files
Exported image rasters for every sample for visual QA, review montages, and quick inspection.

---

## 5. Critical Coordinate Rules (DO NOT BREAK)

> [!CAUTION]
> **Points `(x, y)` in `strokes.json` MUST be in CSS PIXELS relative to the canvas top-left corner.**

1. **Do NOT multiply by DPR**: If the canvas backing store is scaled by $2\times$ for Retina (`canvas.width = rect.width * 2`), the points stored in JSON must **still be unscaled CSS pixels** ($x \in [0, 960]$, $y \in [0, 240]$).
2. **Do NOT use screen or client coordinates**: Always subtract `canvas.getBoundingClientRect()`:
   ```javascript
   const rect = canvas.getBoundingClientRect();
   const x = event.clientX - rect.left;
   const y = event.clientY - rect.top;
   ```
3. **Timestamp `t`**: High-resolution integer timestamp (e.g. `Date.now()`).
4. **Pressure `p`**: Value between $0.0$ and $1.0$. If the browser/device does not support pressure, default to `0.5`.

---

## 6. Apple iPad / iPadOS / Safari Common Issues & Error Analysis

When volunteers attempt to use Apple iPads with Apple Pencils, they commonly encounter one of four specific issues:

### Issue A: Touch Gestures Interfere with Drawing (Scrolling / Pinching)
- **Symptom**: Drawing a stroke scrolls the web page or triggers Safari's navigation gestures; the stroke cuts off abruptly or disappears.
- **Cause**: Safari on iPadOS defaults to gesture handling unless explicitly disabled.
- **Fix in CSS & HTML**:
  ```css
  /* Must be on canvas AND container */
  #inkCanvas, .canvas-container {
    touch-action: none;
    -webkit-touch-callout: none;
    -webkit-user-select: none;
    user-select: none;
  }
  ```
- **Fix in Event Listeners**:
  Call `e.preventDefault()` on `pointerdown`, `pointermove`, and `touchstart`.

### Issue B: Retina Display DPR Misalignment / Offset Ink
- **Symptom**: The ink appears 2x away from where the Apple Pencil tip touches the screen, or the canvas appears blurry.
- **Cause**: iPad screens have `window.devicePixelRatio = 2.0` (or 3.0). If backing store resizing and 2D context scaling are out of sync, coordinates will be offset.
- **Correct Pattern**:
  ```javascript
  function resizeCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    // Set internal backing store dimensions to physical device pixels
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    
    // Scale 2D context so drawing commands continue to use CSS pixels
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    
    // Redraw guidelines and ink
    redrawCanvas();
  }
  ```

### Issue C: Palm Rejection (Hand Touching Screen Alongside Apple Pencil)
- **Symptom**: Random dots or erratic lines appear under the user's resting wrist.
- **Cause**: iPadOS emits `PointerEvent` for both the Apple Pencil (`e.pointerType === "pen"`) and the resting palm (`e.pointerType === "touch"`).
- **Correct Pattern**:
  ```javascript
  function handlePointerDown(e) {
    // If Apple Pencil is active, ignore touches from fingers/palm
    if (e.pointerType === "touch" && hasActiveStylus) {
      return;
    }
    // ...
  }
  ```

### Issue D: ZIP Download Fails or Silently Does Nothing on iPad Safari
- **Symptom**: Tapping "Download ZIP" does not trigger a download, or Safari asks "Do you want to download?" and fails.
- **Root Causes on iOS**:
  1. **`file:///` Sandbox Restriction**: If the volunteer opened `index.html` directly from the iPad Files app (via `file:///...`), Safari's strict security sandbox **disables `a.download` programmatic downloads and blocks `localStorage`**.
     - **Requirement**: The page **must be served over HTTP/HTTPS** (e.g. GitHub Pages or a local server via Wi-Fi `http://192.168.x.x:8000`).
  2. **Asynchronous User Gesture Expiration**: If `createStoredZip` is slow or wrapped in an asynchronous callback, Safari cancels the download permission because it was not directly attached to the user's tap gesture.
  3. **Blob MIME Type**: Safari requires `application/zip` or `application/octet-stream`.
  4. **Fallback for iOS**: If `a.download` fails, provide an alternative fallback that opens the blob in a new tab:
     ```javascript
     try {
       const a = document.createElement('a');
       a.href = URL.createObjectURL(zipBlob);
       a.download = filename;
       document.body.appendChild(a);
       a.click();
       document.body.removeChild(a);
     } catch (err) {
       // Fallback: Open in new window
       window.open(URL.createObjectURL(zipBlob), '_blank');
     }
     ```

---

## 7. How the Python ML Pipeline Processes the Data

To understand why the above constraints exist, here is the exact transformation performed by Python during training and evaluation:

```
strokes.json + manifest.csv
       │
       ▼
scripts/render_strokes.py
  1. Center coordinates on baseline: Y_center = band_base - band_top
  2. Scale by canonical zoom: Z = 1.0 (v2) or 150/108 (v1)
  3. Render vector path on 4x supersampled canvas (H = 256 px) with constant width (2.2 px)
  4. Anti-aliased downscale to H = 64 px
  5. Add pad_x = 16 px safety margins, pad width rightwards with 255 to multiple of 32
  6. Normalize array: (arr / 127.5) - 1.0 -> [-1.0, +1.0]
       │
       ▼
Input Tensor [1, 1, 64, W_padded]
       │
       ▼
CRNN Model Forward (VGG-Lite Backbone -> BiGRU -> Linear -> CTC)
```

### Scale Normalization Formula (Global Target Fill)
In `dataset.py`, the training engine scales strokes dynamically so ink occupies 48px (~75%) of the 64px tensor without per-writer manual tuning:

$$S_w = \min\left( \frac{48.0}{H_{\text{ink}, w} \times (64.0 / H_{\text{band}, w})}, \; \frac{58.0}{H_{\text{ink\_max}, w} \times (64.0 / H_{\text{band}, w})} \right)$$

This formula mathematically guarantees that **ink is never clipped vertically**, provided the frontend records true, uncorrupted CSS coordinates.

---

## 8. Verification Checklist for Frontend Engineers

Before submitting changes to `capture/index.html` or `capture/shareable.html`, verify the following on a physical tablet:

- [ ] **Physical Device Test**: Tested on physical tablet (iPad with Apple Pencil and/or Android/Surface tablet).
- [ ] **Touch Action**: Drawing on the canvas does NOT scroll or zoom the webpage.
- [ ] **Coordinate Sanity**: `strokes.json` coordinates are within $[0, 960]$ for $X$ and $[0, 240]$ for $Y$ regardless of screen DPR ($1\times, 2\times, 3\times$).
- [ ] **Ruled Band Centering**: Baseline is drawn at $Y = 147$, Top guideline at $Y = 72$, Bottom guideline at $Y = 180$.
- [ ] **Prompt Groups**: Lines and words maintain paired correspondence (`parent_line_id` matching).
- [ ] **ZIP Download**: Tapping "تصدير الحزمة / Download ZIP" immediately downloads a valid ZIP file containing `manifest.csv`, `strokes.json`, and all `.png` files.
- [ ] **No Server Dependency**: Page operates fully offline in the browser once loaded.
- [ ] **HTTP/HTTPS Hosting**: If testing on iPad, page is loaded via HTTP server or HTTPS host (not raw `file:///` path).
