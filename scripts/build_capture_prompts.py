"""
scripts/build_capture_prompts.py
Task 2.17 Item 3:
Builds paired capture prompts exclusively from test surahs (data/derived/splits.json):
- 20 lines (4-7 words, test surahs, >=5 containing "الله")
- Each line's words as separate paired prompts (enables line vs isolated word comparison)
- 20 extra words (balanced frequent and rare words)
- Vowelled and rasm mixed

Outputs:
- data/capture_prompts.json
"""

from __future__ import annotations
import sys
import json
import random
from pathlib import Path
from collections import Counter

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = ROOT / "data" / "derived" / "splits.json"
SIMPLE_PATH = ROOT / "data" / "tanzil" / "quran-simple.txt"
CLEAN_NUMBERED_PATH = ROOT / "data" / "tanzil" / "quran-simple-clean-numbered.txt"
OUT_JSON = ROOT / "data" / "capture_prompts.json"

SURAH_NAMES = {
    1: "الفاتحة", 11: "هود", 25: "الفرقان", 32: "السجدة", 41: "فصلت",
    43: "الزخرف", 44: "الدخان", 52: "الطور", 53: "النجم", 62: "الجمعة",
    64: "التغابن", 68: "القلم", 71: "نوح", 76: "الإنسان", 78: "النبأ",
    84: "الانشقاق", 89: "الفجر", 90: "البلد", 92: "الليل", 96: "العلق",
    100: "العاديات", 101: "القارعة", 103: "العصر", 107: "الماعون",
    111: "المسد", 112: "الإخلاص",
}


def load_test_ayahs(test_surahs: set[int]):
    s_lines = SIMPLE_PATH.read_text(encoding="utf-8").strip().splitlines()
    c_lines = CLEAN_NUMBERED_PATH.read_text(encoding="utf-8").strip().splitlines()

    ayahs = []
    for s_l, c_l in zip(s_lines, c_lines):
        c_parts = c_l.split("|")
        if len(c_parts) != 3:
            continue
        s_id, a_id, c_text = int(c_parts[0]), int(c_parts[1]), c_parts[2].strip()
        s_text = s_l.strip()
        if s_id in test_surahs:
            sw = s_text.split()
            cw = c_text.split()
            if len(sw) == len(cw):
                ayahs.append({
                    "surah_id": s_id,
                    "ayah_id": a_id,
                    "surah_name": f"سورة {SURAH_NAMES.get(s_id, str(s_id))}",
                    "sw": sw,
                    "cw": cw,
                })
    return ayahs


def build_prompts():
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    test_surahs = set(splits["test"]["surahs"])
    ayahs = load_test_ayahs(test_surahs)

    rng = random.Random(42)

    # 1. 20 Lines (4-7 words, >= 5 containing "الله")
    cand_lines = []
    for a in ayahs:
        nw = len(a["cw"])
        if 4 <= nw <= 7:
            cand_lines.append({
                "surah_id": a["surah_id"],
                "ayah_id": a["ayah_id"],
                "surah_name": a["surah_name"],
                "words_cw": a["cw"],
                "words_sw": a["sw"],
            })
        elif nw > 7:
            for st in range(0, nw - 4, 3):
                ln = rng.randint(4, min(7, nw - st))
                cand_lines.append({
                    "surah_id": a["surah_id"],
                    "ayah_id": a["ayah_id"],
                    "surah_name": a["surah_name"],
                    "words_cw": a["cw"][st : st + ln],
                    "words_sw": a["sw"][st : st + ln],
                })

    rng.shuffle(cand_lines)
    with_allah = [l for l in cand_lines if "الله" in " ".join(l["words_cw"])]
    without_allah = [l for l in cand_lines if "الله" not in " ".join(l["words_cw"])]

    # 6 lines with "الله" (>= 5 guaranteed) + 14 without
    selected_raw_lines = with_allah[:6] + without_allah[:14]
    rng.shuffle(selected_raw_lines)

    lines_prompts = []
    paired_word_prompts = []

    for idx, l in enumerate(selected_raw_lines, 1):
        line_id = f"line_{idx:02d}"
        rasm_line = " ".join(l["words_cw"])
        vow_line = " ".join(l["words_sw"])
        prompt_type = "vowelled" if (idx % 2 == 1) else "rasm"

        lines_prompts.append({
            "id": line_id,
            "mode": "line",
            "rasm": rasm_line,
            "vowelled": vow_line,
            "prompt_type": prompt_type,
            "surah": l["surah_name"],
            "words_count": len(l["words_cw"]),
        })

        # Generate each line's words as separate prompts
        for w_idx, (cw, sw) in enumerate(zip(l["words_cw"], l["words_sw"])):
            paired_id = f"paired_{line_id}_w{w_idx+1:02d}"
            w_prompt_type = "vowelled" if ((idx + w_idx) % 2 == 1) else "rasm"
            paired_word_prompts.append({
                "id": paired_id,
                "mode": "word",
                "parent_line_id": line_id,
                "word_index_in_line": w_idx,
                "rasm": cw,
                "vowelled": sw,
                "prompt_type": w_prompt_type,
                "surah": l["surah_name"],
            })

    # 2. 20 Extra words (from test surahs: 10 frequent, 10 rare)
    word_freq = Counter()
    word_map = {}
    for a in ayahs:
        for sw, cw in zip(a["sw"], a["cw"]):
            word_freq[cw] += 1
            if cw not in word_map:
                word_map[cw] = (sw, a["surah_name"])

    frequent_cand = [w for w, cnt in word_freq.most_common(100) if len(w) >= 3 and w not in [p["rasm"] for p in paired_word_prompts]]
    rare_cand = [w for w, cnt in word_freq.items() if cnt in (1, 2) and len(w) >= 3 and w not in [p["rasm"] for p in paired_word_prompts]]
    rng.shuffle(frequent_cand)
    rng.shuffle(rare_cand)

    extra_word_prompts = []
    for idx, w in enumerate(frequent_cand[:10], 1):
        sw, s_name = word_map[w]
        ptype = "vowelled" if (idx % 2 == 1) else "rasm"
        extra_word_prompts.append({
            "id": f"extra_word_freq_{idx:02d}",
            "mode": "word",
            "category": "frequent",
            "rasm": w,
            "vowelled": sw,
            "prompt_type": ptype,
            "surah": s_name,
        })

    for idx, w in enumerate(rare_cand[:10], 1):
        sw, s_name = word_map[w]
        ptype = "vowelled" if (idx % 2 == 0) else "rasm"
        extra_word_prompts.append({
            "id": f"extra_word_rare_{idx:02d}",
            "mode": "word",
            "category": "rare",
            "rasm": w,
            "vowelled": sw,
            "prompt_type": ptype,
            "surah": s_name,
        })

    all_word_prompts = paired_word_prompts + extra_word_prompts

    out_data = {
        "metadata": {
            "version": "2.0_paired",
            "lines_count": len(lines_prompts),
            "paired_words_count": len(paired_word_prompts),
            "extra_words_count": len(extra_word_prompts),
            "total_words_count": len(all_word_prompts),
            "allah_lines_count": sum(1 for l in lines_prompts if "الله" in l["rasm"]),
        },
        "lines": lines_prompts,
        "words": all_word_prompts,
    }

    OUT_JSON.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(lines_prompts)} lines and {len(all_word_prompts)} words to {OUT_JSON}.")
    print(f"Lines containing 'الله': {out_data['metadata']['allah_lines_count']}")


if __name__ == "__main__":
    build_prompts()
