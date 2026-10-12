"""
scripts/build_shareable_html.py
Generates the static shareable capture page: capture/shareable.html
Features:
- Short protocol: 40 samples (10 lines + 30 words, 10-12 mins, balanced letters, includes Allah)
- Consent screen
- Anonymous writer ID (stored in localStorage)
- In-browser ZIP export with zero server dependencies
"""

from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

with open(ROOT / "data" / "capture_prompts_short.json", encoding="utf-8") as f:
    short_data = json.load(f)

lines_js = json.dumps(short_data["lines"], ensure_ascii=False, indent=2)
words_js = json.dumps(short_data["words"], ensure_ascii=False, indent=2)

html_content = f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
  <title>مشروع حافظ — استوديو جمع خط اليد القرآني</title>
  <style>
    :root {{
      --bg: #0f172a;
      --card-bg: #1e293b;
      --border: #334155;
      --primary: #38bdf8;
      --primary-hover: #0ea5e9;
      --text: #f8fafc;
      --text-muted: #94a3b8;
      --accent: #10b981;
      --danger: #ef4444;
      --canvas-bg: #ffffff;
      --ink: #000000;
      --guide-line: #cbd5e1;
      --guide-base: #93c5fd;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Amiri", "Traditional Arabic", sans-serif;
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 1rem;
      touch-action: manipulation;
    }}
    .header {{
      width: 100%;
      max-width: 960px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 0.75rem;
      border-bottom: 1px solid var(--border);
      padding-bottom: 0.5rem;
      flex-wrap: wrap;
      gap: 0.5rem;
    }}
    .title {{ font-size: 1.15rem; font-weight: 700; color: var(--primary); }}
    .badge {{
      background: #0369a1;
      color: #e0f2fe;
      padding: 0.2rem 0.6rem;
      border-radius: 999px;
      font-size: 0.8rem;
      font-weight: 600;
    }}
    .mode-switch, .display-switch {{
      display: flex;
      gap: 0.25rem;
      background: var(--card-bg);
      padding: 0.2rem;
      border-radius: 8px;
      border: 1px solid var(--border);
    }}
    .toggle-btn, .mode-btn {{
      background: transparent;
      border: none;
      color: var(--text-muted);
      padding: 0.35rem 0.75rem;
      border-radius: 6px;
      cursor: pointer;
      font-size: 0.8rem;
      font-weight: 600;
      transition: all 0.2s;
    }}
    .toggle-btn.active, .mode-btn.active {{
      background: var(--primary);
      color: #0f172a;
    }}
    .prompt-box {{
      width: 100%;
      max-width: 960px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 1rem;
      margin-bottom: 0.75rem;
      text-align: center;
    }}
    .prompt-info {{
      display: flex;
      justify-content: space-between;
      color: var(--text-muted);
      font-size: 0.85rem;
      margin-bottom: 0.5rem;
    }}
    .prompt-text {{
      font-size: 2.2rem;
      font-weight: 700;
      color: #38bdf8;
      letter-spacing: 1px;
      direction: rtl;
    }}
    .canvas-container {{
      width: 100%;
      max-width: 960px;
      position: relative;
      margin-bottom: 0.75rem;
      box-shadow: 0 4px 12px rgba(0,0,0,0.5);
      border-radius: 12px;
      overflow: hidden;
      border: 2px solid var(--border);
    }}
    #inkCanvas {{
      display: block;
      width: 100%;
      height: 240px;
      background: var(--canvas-bg);
      cursor: crosshair;
      touch-action: none;
    }}
    .controls {{
      width: 100%;
      max-width: 960px;
      display: flex;
      justify-content: space-between;
      gap: 0.5rem;
      flex-wrap: wrap;
      margin-bottom: 0.75rem;
    }}
    .btn-group {{ display: flex; gap: 0.4rem; }}
    button {{
      padding: 0.5rem 1rem;
      border-radius: 8px;
      border: none;
      font-weight: 600;
      font-size: 0.9rem;
      cursor: pointer;
      transition: all 0.2s;
    }}
    .btn-primary {{ background: var(--primary); color: #0f172a; }}
    .btn-secondary {{ background: var(--card-bg); color: var(--text); border: 1px solid var(--border); }}
    .btn-danger {{ background: #dc2626; color: white; }}
    .btn-success {{ background: #16a34a; color: white; }}
    .stats-card {{
      width: 100%;
      max-width: 960px;
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 0.75rem 1rem;
      display: flex;
      justify-content: space-around;
      font-size: 0.85rem;
      color: var(--text-muted);
      text-align: center;
    }}
    .stat-val {{ font-size: 1.1rem; font-weight: 700; color: var(--text); margin-top: 2px; }}

    /* Consent Modal */
    .modal-overlay {{
      position: fixed;
      inset: 0;
      background: rgba(0,0,0,0.8);
      backdrop-filter: blur(4px);
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 1.5rem;
      z-index: 1000;
    }}
    .modal-content {{
      background: var(--card-bg);
      border: 1px solid var(--border);
      border-radius: 16px;
      max-width: 600px;
      width: 100%;
      padding: 1.75rem;
      text-align: right;
    }}
    .modal-title {{ font-size: 1.3rem; font-weight: 700; color: var(--primary); margin-bottom: 1rem; }}
    .modal-body {{ font-size: 0.95rem; line-height: 1.6; color: #cbd5e1; margin-bottom: 1.5rem; }}
    .modal-body ul {{ margin: 0.75rem 1.25rem; }}
    .modal-body li {{ margin-bottom: 0.4rem; }}
  </style>
</head>
<body>

  <!-- Consent Modal -->
  <div id="consentModal" class="modal-overlay">
    <div class="modal-content">
      <div class="modal-title">مشروع حافظ — دراسة التحقق من خط اليد القرآني</div>
      <div class="modal-body">
        <p>أهلاً بك! تهدف هذه الدراسة إلى تدريب وتطوير خوارزميات ذكاء اصطناعي (HTR) للتعرف على خطوط اليد القرآنية ودعم حفظة كتاب الله.</p>
        <ul style="list-style: square;">
          <li><strong>البروتوكول القصير:</strong> 40 عينة فقط (10 أسطر + 30 كلمة).</li>
          <li><strong>الوقت المتوقع:</strong> 10 إلى 12 دقيقة تقريباً.</li>
          <li><strong>الخصوصية التامة:</strong> المعرف مجهول تماماً ولا تُجمع أي بيانات شخصية (فقط إحداثيات القلم وسرعة الحركة).</li>
          <li><strong>التصدير الفوري:</strong> عند الانتهاء يمكنك تنزيل ملف ZIP وحفظه بضغطة زر.</li>
        </ul>
        <p style="color: #38bdf8;">باستخدامك للقلم والكتابة على الشاشة، فإنك تساهم في خدمة كتاب الله تعالى.</p>
      </div>
      <div style="display: flex; justify-content: flex-end; gap: 0.75rem;">
        <button class="btn-primary" style="padding: 0.75rem 1.5rem; font-size: 1rem;" onclick="acceptConsent()">أوافق وأبدأ الكتابة (Agree &amp; Begin)</button>
      </div>
    </div>
  </div>

  <div class="header">
    <div class="title">استوديو حافظ للخط القرآني (البروتوكول القصير)</div>
    <div style="display: flex; gap: 0.5rem; align-items: center;">
      <span class="badge" id="writerBadge">المعرف: جاري التحميل...</span>
      <div class="display-switch">
        <button class="toggle-btn active" id="toggleVowelledBtn" onclick="setDisplayMode('vowelled')">مشكول</button>
        <button class="toggle-btn" id="toggleRasmBtn" onclick="setDisplayMode('rasm')">رسم</button>
      </div>
      <div class="mode-switch">
        <button class="mode-btn active" id="modeLineBtn" onclick="setMode('line')">أسطر (10)</button>
        <button class="mode-btn" id="modeWordBtn" onclick="setMode('word')">كلمات (30)</button>
      </div>
    </div>
  </div>

  <div class="prompt-box">
    <div class="prompt-info">
      <span id="promptIndex">العينة: 1 / 10</span>
      <span id="promptSurah">سورة هود</span>
    </div>
    <div class="prompt-text" id="promptText">وقيل يا أرض ابلعي</div>
  </div>

  <div class="canvas-container">
    <canvas id="inkCanvas"></canvas>
  </div>

  <div class="controls">
    <div class="btn-group">
      <button class="btn-danger" onclick="clearCanvas()">مسح (Clear)</button>
      <button class="btn-secondary" onclick="undoStroke()">تراجع (Undo)</button>
    </div>
    <div class="btn-group">
      <button class="btn-secondary" onclick="prevPrompt()">السابق (Prev)</button>
      <button class="btn-primary" onclick="saveAndNext()">حفظ والتالي (Save &amp; Next)</button>
    </div>
    <div class="btn-group">
      <button class="btn-success" onclick="exportCapturedDataset()">تصدير الحزمة (Download ZIP)</button>
    </div>
  </div>

  <div class="stats-card">
    <div>
      <div>العينات المحفوظة</div>
      <div class="stat-val" id="savedCount">0 / 40</div>
    </div>
    <div>
      <div>الوقت المقدر</div>
      <div class="stat-val">10–12 دقيقة</div>
    </div>
    <div>
      <div>تغطية الحروف</div>
      <div class="stat-val">100% متوازنة + لفظ الجلالة</div>
    </div>
  </div>

  <script>
    const PROMPTS_LINE = {lines_js};
    const PROMPTS_WORD = {words_js};

    // Anonymous Writer ID
    let anonWriterId = localStorage.getItem('hafiz_anon_writer_id');
    if (!anonWriterId) {{
      anonWriterId = 'writer_anon_' + Math.random().toString(36).substring(2, 8);
      localStorage.setItem('hafiz_anon_writer_id', anonWriterId);
    }}
    document.getElementById('writerBadge').innerText = 'المعرف: ' + anonWriterId;

    // Consent Screen check
    if (localStorage.getItem('hafiz_consent_accepted') === 'true') {{
      document.getElementById('consentModal').style.display = 'none';
    }}
    function acceptConsent() {{
      localStorage.setItem('hafiz_consent_accepted', 'true');
      document.getElementById('consentModal').style.display = 'none';
    }}

    let currentMode = 'line';
    let displayMode = 'vowelled';
    let lineIndex = 0;
    let wordIndex = 0;
    let currentStrokes = [];
    let currentStroke = null;
    let isDrawing = false;
    let savedSamples = [];

    // LocalStorage restore
    const savedLocal = localStorage.getItem('hafiz_short_saved_' + anonWriterId);
    if (savedLocal) {{
      try {{ savedSamples = JSON.parse(savedLocal); }} catch(e) {{}}
    }}

    const canvas = document.getElementById('inkCanvas');
    const ctx = canvas.getContext('2d');

    const GUIDE_TOP = 72;
    const GUIDE_BASE = 147;
    const GUIDE_BOT = 180;
    const BAND_HEIGHT = 108;

    function resizeCanvas() {{
      const rect = canvas.getBoundingClientRect();
      const dpr = window.devicePixelRatio || 1;
      canvas.width = rect.width * dpr;
      canvas.height = 240 * dpr;
      ctx.scale(dpr, dpr);
      redrawCanvas();
    }}
    window.addEventListener('resize', resizeCanvas);

    function redrawCanvas() {{
      const w = canvas.getBoundingClientRect().width;
      const h = 240;
      ctx.clearRect(0, 0, w, h);

      // Ruled guidelines
      ctx.lineWidth = 1;
      ctx.strokeStyle = '#e2e8f0';
      ctx.beginPath();
      ctx.moveTo(0, GUIDE_TOP); ctx.lineTo(w, GUIDE_TOP);
      ctx.stroke();

      ctx.strokeStyle = '#93c5fd';
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(0, GUIDE_BASE); ctx.lineTo(w, GUIDE_BASE);
      ctx.stroke();

      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, GUIDE_BOT); ctx.lineTo(w, GUIDE_BOT);
      ctx.stroke();

      // Render strokes
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      for (const stroke of currentStrokes) {{
        if (stroke.length < 2) continue;
        for (let i = 1; i < stroke.length; i++) {{
          ctx.beginPath();
          ctx.moveTo(stroke[i-1].x, stroke[i-1].y);
          ctx.lineTo(stroke[i].x, stroke[i].y);
          const p = stroke[i].p || 0.5;
          ctx.lineWidth = 1.2 + p * 2.2;
          ctx.strokeStyle = '#0f172a';
          ctx.stroke();
        }}
      }}
    }}

    function getCanvasCoords(e) {{
      const rect = canvas.getBoundingClientRect();
      return {{
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
        p: e.pressure || 0.5,
        t: Date.now()
      }};
    }}

    canvas.addEventListener('pointerdown', (e) => {{
      canvas.setPointerCapture(e.pointerId);
      isDrawing = true;
      const pt = getCanvasCoords(e);
      currentStroke = [pt];
      currentStrokes.push(currentStroke);
      redrawCanvas();
    }});

    canvas.addEventListener('pointermove', (e) => {{
      if (!isDrawing || !currentStroke) return;
      const pt = getCanvasCoords(e);
      currentStroke.push(pt);
      redrawCanvas();
    }});

    canvas.addEventListener('pointerup', (e) => {{
      isDrawing = false;
      currentStroke = null;
    }});

    function clearCanvas() {{
      currentStrokes = [];
      redrawCanvas();
    }}

    function undoStroke() {{
      if (currentStrokes.length > 0) {{
        currentStrokes.pop();
        redrawCanvas();
      }}
    }}

    function setMode(mode) {{
      currentMode = mode;
      document.getElementById('modeLineBtn').classList.toggle('active', mode === 'line');
      document.getElementById('modeWordBtn').classList.toggle('active', mode === 'word');
      updatePromptUI();
      clearCanvas();
    }}

    function setDisplayMode(disp) {{
      displayMode = disp;
      document.getElementById('toggleVowelledBtn').classList.toggle('active', disp === 'vowelled');
      document.getElementById('toggleRasmBtn').classList.toggle('active', disp === 'rasm');
      updatePromptUI();
    }}

    function getCurrentPrompt() {{
      const arr = currentMode === 'line' ? PROMPTS_LINE : PROMPTS_WORD;
      const idx = currentMode === 'line' ? lineIndex : wordIndex;
      return arr[idx];
    }}

    function updatePromptUI() {{
      const p = getCurrentPrompt();
      const total = currentMode === 'line' ? PROMPTS_LINE.length : PROMPTS_WORD.length;
      const idx = (currentMode === 'line' ? lineIndex : wordIndex) + 1;
      document.getElementById('promptIndex').innerText = `العينة: ${{idx}} / ${{total}}`;
      document.getElementById('promptSurah').innerText = p.surah || '';
      document.getElementById('promptText').innerText = displayMode === 'vowelled' ? (p.vowelled || p.rasm) : p.rasm;
      document.getElementById('savedCount').innerText = `${{savedSamples.length}} / 40`;
    }}

    function prevPrompt() {{
      if (currentMode === 'line') {{
        if (lineIndex > 0) lineIndex--;
      }} else {{
        if (wordIndex > 0) wordIndex--;
      }}
      updatePromptUI();
      clearCanvas();
    }}

    function saveAndNext() {{
      if (currentStrokes.length === 0) {{
        alert('الرجاء كتابة العينة قبل الانتقال!');
        return;
      }}
      const p = getCurrentPrompt();
      const sId = `sample_${{currentMode}}_${{String(Date.now()).slice(-6)}}`;
      const rect = canvas.getBoundingClientRect();

      // Band-relative strokes
      const relStrokes = currentStrokes.map(st => st.map(pt => ({{
        x: pt.x,
        y: pt.y - GUIDE_TOP,
        p: pt.p,
        t: pt.t
      }})));

      // Save sample
      const sample = {{
        id: sId,
        text: p.rasm,
        prompt_type: displayMode,
        mode: currentMode,
        writer_id: anonWriterId,
        geometry: {{
          capture_version: 'v2',
          band_top_css: GUIDE_TOP,
          band_base_css: GUIDE_BASE,
          band_bot_css: GUIDE_BOT,
          band_height_css: BAND_HEIGHT,
          canvas_w_css: rect.width,
          canvas_h_css: 240,
          dpr: window.devicePixelRatio || 1
        }},
        strokes: relStrokes,
        dataUrl: canvas.toDataURL('image/png')
      }};

      savedSamples.push(sample);
      localStorage.setItem('hafiz_short_saved_' + anonWriterId, JSON.stringify(savedSamples));

      // Advance
      if (currentMode === 'line') {{
        if (lineIndex < PROMPTS_LINE.length - 1) {{
          lineIndex++;
        }} else {{
          alert('أحسنت! أكملت كل الأسطر (10). ننتقل الآن إلى قسم الكلمات (30).');
          setMode('word');
          return;
        }}
      }} else {{
        if (wordIndex < PROMPTS_WORD.length - 1) {{
          wordIndex++;
        }} else {{
          alert('مبروك! أنهيت كافة العينات الـ 40 بنجاح! الرجاء الضغط على زر "تصدير الحزمة" لتحميل ملف ZIP.');
        }}
      }}
      updatePromptUI();
      clearCanvas();
    }}

    // In-browser Pure JS ZIP generator (Zero external dependencies)
    function createStoredZip(files) {{
      const parts = [];
      let cdOffset = 0;
      let cdSize = 0;
      const fileRecords = [];

      for (const file of files) {{
        const nameBytes = new TextEncoder().encode(file.name);
        const dataBytes = file.data;
        const offset = cdOffset;

        const header = new Uint8Array(30 + nameBytes.length);
        const hView = new DataView(header.buffer);
        hView.setUint32(0, 0x04034b50, true);
        hView.setUint16(4, 20, true);
        hView.setUint16(6, 0, true);
        hView.setUint16(8, 0, true);
        hView.setUint16(10, 0, true);
        hView.setUint16(12, 0, true);
        hView.setUint32(14, 0, true);
        hView.setUint32(18, dataBytes.length, true);
        hView.setUint32(22, dataBytes.length, true);
        hView.setUint16(26, nameBytes.length, true);
        hView.setUint16(28, 0, true);
        header.set(nameBytes, 30);

        parts.push(header);
        parts.push(dataBytes);
        cdOffset += header.length + dataBytes.length;

        fileRecords.push({{ nameBytes, dataLength: dataBytes.length, offset }});
      }}

      for (const rec of fileRecords) {{
        const cdHeader = new Uint8Array(46 + rec.nameBytes.length);
        const view = new DataView(cdHeader.buffer);
        view.setUint32(0, 0x02014b50, true);
        view.setUint16(4, 20, true);
        view.setUint16(6, 20, true);
        view.setUint16(8, 0, true);
        view.setUint16(10, 0, true);
        view.setUint16(12, 0, true);
        view.setUint16(14, 0, true);
        view.setUint32(16, 0, true);
        view.setUint32(20, rec.dataLength, true);
        view.setUint32(24, rec.dataLength, true);
        view.setUint16(28, rec.nameBytes.length, true);
        view.setUint16(30, 0, true);
        view.setUint16(32, 0, true);
        view.setUint16(34, 0, true);
        view.setUint16(36, 0, true);
        view.setUint32(38, 0, true);
        view.setUint32(42, rec.offset, true);
        cdHeader.set(rec.nameBytes, 46);

        parts.push(cdHeader);
        cdSize += cdHeader.length;
      }}

      const eocd = new Uint8Array(22);
      const eView = new DataView(eocd.buffer);
      eView.setUint32(0, 0x06054b50, true);
      eView.setUint16(4, 0, true);
      eView.setUint16(6, 0, true);
      eView.setUint16(8, fileRecords.length, true);
      eView.setUint16(10, fileRecords.length, true);
      eView.setUint32(12, cdSize, true);
      eView.setUint32(16, cdOffset, true);
      eView.setUint16(20, 0, true);
      parts.push(eocd);

      return new Blob(parts, {{ type: "application/zip" }});
    }}

    function dataUrlToUint8(dataUrl) {{
      const b64 = dataUrl.split(',')[1];
      const bin = atob(b64);
      const bytes = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      return bytes;
    }}

    function exportCapturedDataset() {{
      if (savedSamples.length === 0) {{
        alert('لا توجد عينات محفوظة للتصدير بعد!');
        return;
      }}
      const files = [];
      let csv = 'sample_id,image_file,mode,transcription,writer_id,capture_version,band_top_css,band_base_css,band_bot_css,canvas_w_css,canvas_h_css,dpr\\n';
      for (const s of savedSamples) {{
        const g = s.geometry || {{}};
        csv += `${{s.id}},${{s.id}}.png,${{s.mode}},"${{s.text}}",${{s.writer_id}},v2,${{g.band_top_css}},${{g.band_base_css}},${{g.band_bot_css}},${{g.canvas_w_css}},${{g.canvas_h_css}},${{g.dpr}}\\n`;
      }}
      files.push({{ name: 'manifest.csv', data: new TextEncoder().encode(csv) }});

      const strokesData = savedSamples.map(s => ({{
        id: s.id,
        text: s.text,
        capture_version: 'v2',
        geometry: s.geometry,
        strokes: s.strokes
      }}));
      files.push({{ name: 'strokes.json', data: new TextEncoder().encode(JSON.stringify(strokesData, null, 2)) }});

      for (const s of savedSamples) {{
        if (s.dataUrl) files.push({{ name: `${{s.id}}.png`, data: dataUrlToUint8(s.dataUrl) }});
      }}

      const zipBlob = createStoredZip(files);
      const a = document.createElement('a');
      a.href = URL.createObjectURL(zipBlob);
      a.download = `hafiz_${{anonWriterId}}_${{Date.now()}}.zip`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      alert(`تم تنزيل الحزمة بنجاح (${{savedSamples.length}} عينة) في ملف ZIP واحد.`);
    }}

    resizeCanvas();
    updatePromptUI();
  </script>
</body>
</html>
"""

out_file = ROOT / "capture" / "shareable.html"
with open(out_file, "w", encoding="utf-8") as f:
    f.write(html_content)
print(f"Generated {out_file} ({len(html_content)} bytes).")
