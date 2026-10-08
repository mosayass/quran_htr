"""
scripts/update_capture_html.py
Updates capture/index.html with:
1. Curated prompts from data/capture_prompts.json (40 lines, 60 words).
2. Header toggle buttons for display switch (Tanzil Simple vs Rasm-only).
3. Export stroke rendering rule (constant 0.044 * cropH = 2.8 px at 64, no pressure dependence).
4. Downscaling with imageSmoothingQuality = 'high'.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = ROOT / "data" / "capture_prompts.json"
HTML_PATH = ROOT / "capture" / "index.html"

def main():
    prompts = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
    lines_data = [
        {"rasm": l["rasm"], "vowelled": l["vowelled"], "surah": l["surah_name"]}
        for l in prompts["lines"]
    ]
    words_data = [
        {"rasm": w["rasm"], "vowelled": w["vowelled"], "surah": w["surah_name"]}
        for w in prompts["words"]
    ]

    lines_js = json.dumps(lines_data, ensure_ascii=False, indent=6)
    words_js = json.dumps(words_data, ensure_ascii=False, indent=6)

    html = HTML_PATH.read_text(encoding="utf-8")

    # 1. Update header HTML
    old_header = """  <div class="header">
    <div class="title">Hafiz HTR — Stopgap Ink Capture Studio</div>
    <div class="mode-switch">
      <button class="mode-btn active" id="modeLineBtn" onclick="setMode('line')">وضع السطر (Line: 3-9 كلمات)</button>
      <button class="mode-btn" id="modeWordBtn" onclick="setMode('word')">وضع الكلمة (Word: كلمة واحدة)</button>
    </div>
  </div>"""

    new_header = """  <div class="header">
    <div class="title">Hafiz HTR — Stopgap Ink Capture Studio</div>
    <div class="display-switch">
      <button class="toggle-btn active" id="toggleVowelledBtn" onclick="setDisplayMode('vowelled')">مشكول (Tanzil Simple)</button>
      <button class="toggle-btn" id="toggleRasmBtn" onclick="setDisplayMode('rasm')">رسم (Rasm-only)</button>
    </div>
    <div class="mode-switch">
      <button class="mode-btn active" id="modeLineBtn" onclick="setMode('line')">وضع السطر (Line: 3-9 كلمات)</button>
      <button class="mode-btn" id="modeWordBtn" onclick="setMode('word')">وضع الكلمة (Word: كلمة واحدة)</button>
    </div>
  </div>"""

    assert old_header in html, "Could not find old_header in capture/index.html"
    html = html.replace(old_header, new_header)

    # 2. Update prompts
    js_start = html.find("const PROMPTS_LINE = [")
    js_end = html.find("let promptDisplayMode =", js_start)
    assert js_start != -1 and js_end != -1, "Could not find PROMPTS_LINE in HTML"

    new_prompts_code = f"const PROMPTS_LINE = {lines_js};\n\n    const PROMPTS_WORD = {words_js};\n\n    "
    html = html[:js_start] + new_prompts_code + html[js_end:]

    # 3. Update exportRuledBandPNG
    old_export = """      for (const stroke of strokes) {
        if (stroke.length < 2) continue;
        for (let i = 1; i < stroke.length; i++) {
          const p1 = stroke[i - 1];
          const p2 = stroke[i];
          offCtx.lineWidth = Math.max(1.5, (p2.p || 0.5) * 3.5) * scale;
          offCtx.beginPath();
          offCtx.moveTo(p1.x * scale - cropX1, p1.y * scale - cropY1);
          offCtx.lineTo(p2.x * scale - cropX1, p2.y * scale - cropY1);
          offCtx.stroke();
        }
      }

      // Resize to TARGET_HEIGHT = 64px
      const targetH = 64;
      const targetW = Math.max(32, Math.round(cropW * (targetH / cropH)));
      const finalCanvas = document.createElement('canvas');
      finalCanvas.width = targetW;
      finalCanvas.height = targetH;
      const fCtx = finalCanvas.getContext('2d');
      fCtx.fillStyle = '#ffffff';
      fCtx.fillRect(0, 0, targetW, targetH);
      fCtx.drawImage(offscreen, 0, 0, targetW, targetH);"""

    new_export = """      // Constant stroke width = 0.044 * band height in export pixels (2.8 px at 64)
      // No pressure dependence in export
      offCtx.lineWidth = 0.044 * cropH;

      for (const stroke of strokes) {
        if (stroke.length < 2) continue;
        for (let i = 1; i < stroke.length; i++) {
          const p1 = stroke[i - 1];
          const p2 = stroke[i];
          offCtx.beginPath();
          offCtx.moveTo(p1.x * scale - cropX1, p1.y * scale - cropY1);
          offCtx.lineTo(p2.x * scale - cropX1, p2.y * scale - cropY1);
          offCtx.stroke();
        }
      }

      // Resize to TARGET_HEIGHT = 64px using high quality smoothing
      const targetH = 64;
      const targetW = Math.max(32, Math.round(cropW * (targetH / cropH)));
      const finalCanvas = document.createElement('canvas');
      finalCanvas.width = targetW;
      finalCanvas.height = targetH;
      const fCtx = finalCanvas.getContext('2d');
      fCtx.fillStyle = '#ffffff';
      fCtx.fillRect(0, 0, targetW, targetH);
      fCtx.imageSmoothingEnabled = true;
      fCtx.imageSmoothingQuality = 'high';
      fCtx.drawImage(offscreen, 0, 0, targetW, targetH);"""

    assert old_export in html, "Could not find old_export in capture/index.html"
    html = html.replace(old_export, new_export)

    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Successfully updated {HTML_PATH.name} with {len(lines_data)} line prompts and {len(words_data)} word prompts.")

if __name__ == "__main__":
    main()
