import os
import sys
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True, encoding="utf-8")
print("STARTING MAKE_DIAC_CONTACT_SHEET...", flush=True)

import uharfbuzz as hb
import freetype
from PIL import Image, ImageDraw
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
FONTS_DIR = ROOT / "data" / "fonts"
out_path = ROOT / "data" / "font_samples" / "diac_contact_sheet.png"

def render_arabic_text(
    font_path: Path,
    text: str,
    font_size: int = 34,
    word_spacing_factor: float = 1.0,
) -> Image.Image:
    words = text.split(" ")
    blob = hb.Blob.from_file_path(str(font_path))
    face = hb.Face(blob)
    hb_font = hb.Font(face)
    scale = int(font_size * 64)
    hb_font.scale = (scale, scale)

    ft_face = freetype.Face(str(font_path))
    ft_face.set_char_size(scale, scale)

    space_buf = hb.Buffer()
    space_buf.add_str(" ")
    space_buf.guess_segment_properties()
    hb.shape(hb_font, space_buf)
    base_space_advance = (space_buf.glyph_positions[0].x_advance >> 6) if space_buf.glyph_positions else int(font_size * 0.3)
    space_advance = max(4, int(base_space_advance * word_spacing_factor))

    word_bitmaps = []
    total_w = 0
    max_h = int(font_size * 2.2)
    baseline = int(font_size * 1.5)

    for word in words:
        buf = hb.Buffer()
        buf.add_str(word)
        buf.guess_segment_properties()
        hb.shape(hb_font, buf)

        w_adv = sum(pos.x_advance for pos in buf.glyph_positions) >> 6
        w_img_w = max(w_adv + 16, 20)
        w_arr = np.zeros((max_h, w_img_w), dtype=np.uint8)

        pen_x = 4
        for info, pos in zip(buf.glyph_infos, buf.glyph_positions):
            ft_face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
            slot = ft_face.glyph
            bm = slot.bitmap
            bx = (pen_x + (pos.x_offset >> 6)) + slot.bitmap_left
            by = (baseline - (pos.y_offset >> 6)) - slot.bitmap_top
            if bm.width > 0 and bm.rows > 0:
                bm_arr = np.array(bm.buffer, dtype=np.uint8).reshape((bm.rows, bm.width))
                y1, y2 = max(0, by), min(max_h, by + bm.rows)
                x1, x2 = max(0, bx), min(w_img_w, bx + bm.width)
                sy1, sy2 = y1 - by, y1 - by + (y2 - y1)
                sx1, sx2 = x1 - bx, x1 - bx + (x2 - x1)
                if y2 > y1 and x2 > x1:
                    w_arr[y1:y2, x1:x2] = np.maximum(w_arr[y1:y2, x1:x2], bm_arr[sy1:sy2, sx1:sx2])
            pen_x += (pos.x_advance >> 6)

        ink_cols = np.where(w_arr.sum(axis=0) > 0)[0]
        if len(ink_cols) > 0:
            w_arr = w_arr[:, ink_cols[0]:ink_cols[-1] + 1]
        word_bitmaps.append(w_arr)
        total_w += w_arr.shape[1]

    total_w += space_advance * max(0, len(words) - 1)
    full_w = max(total_w + 30, 40)
    full_arr = np.zeros((max_h, full_w), dtype=np.uint8)

    cur_x = full_w - 15
    for w_arr in word_bitmaps:
        bw = w_arr.shape[1]
        x1 = cur_x - bw
        if x1 >= 0:
            full_arr[:, x1:cur_x] = np.maximum(full_arr[:, x1:cur_x], w_arr)
        cur_x = x1 - space_advance

    img = Image.fromarray(255 - full_arr)
    return img

fonts_to_test = [
    "Amiri.ttf",
    "Scheherazade_New.ttf",
    "Aref_Ruqaa.ttf",
    "Harmattan.ttf",
    "Lateef.ttf",
    "Reem_Kufi_Fun.ttf",
    "Zain.ttf",
    "Mirza.ttf"
]
sample_text = "بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ اقْرَأْ بِاسْمِ رَبِّكَ"

out_path.parent.mkdir(parents=True, exist_ok=True)
row_h = 80
sheet_w = 960
sheet_h = len(fonts_to_test) * row_h + 60

canvas = Image.new("RGB", (sheet_w, sheet_h), color=(255, 255, 255))
draw = ImageDraw.Draw(canvas)
draw.text((30, 15), "DIACRITIC RENDERING CONTACT SHEET (8 FONTS WITH VOWEL MARKS ON)", fill=(20, 20, 20))
draw.line([(30, 38), (sheet_w - 30, 38)], fill=(200, 200, 210), width=1)

y_cur = 50
for f_name in fonts_to_test:
    fp = FONTS_DIR / f_name
    print(f"Rendering font {fp.stem}...", flush=True)
    if not fp.exists():
        print(f"  Missing font {fp.name}", flush=True)
        continue
    try:
        rendered = render_arabic_text(fp, sample_text, font_size=32, word_spacing_factor=1.0)
        draw.rectangle([30, y_cur, sheet_w - 30, y_cur + row_h - 10], fill=(250, 250, 252), outline=(220, 225, 235))
        draw.text((45, y_cur + 24), f"{fp.stem}", fill=(100, 100, 110))
        paste_x = sheet_w - 45 - rendered.width
        paste_y = y_cur + (row_h - 10 - rendered.height) // 2
        canvas.paste(rendered.convert("RGB"), (paste_x, paste_y))
        print(f"  Success {fp.stem}", flush=True)
    except Exception as e:
        print(f"  Error {fp.stem}: {e}", flush=True)
        draw.text((200, y_cur + 20), f"Error: {e}", fill=(200, 40, 40))

    y_cur += row_h

canvas.save(out_path)
print(f"[Contact Sheet] Successfully saved 8-font contact sheet to {out_path}", flush=True)
