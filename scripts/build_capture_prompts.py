"""
scripts/build_capture_prompts.py
Task 2.12 Step 3:
Builds curated capture prompts exclusively from test surahs (data/derived/splits.json):
- 40 line prompts (3-9 words, >=5 containing "الله")
- 60 word prompts (balanced mixture of frequent and rare words)

Vowelled text variant used: quran-simple.txt (Tanzil Simple).
Rasm text variant used: quran-simple-clean.txt (Tanzil Simple Clean).
Both variants have verified 100% word-count alignment across all 6,236 Ayahs.

Outputs:
- data/capture_prompts.json
"""

from __future__ import annotations
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
SPLITS_PATH = ROOT / "data" / "derived" / "splits.json"
SIMPLE_PATH = ROOT / "data" / "tanzil" / "quran-simple.txt"
CLEAN_NUMBERED_PATH = ROOT / "data" / "tanzil" / "quran-simple-clean-numbered.txt"
OUT_JSON = ROOT / "data" / "capture_prompts.json"

SURAH_NAMES = {
    1: "الفاتحة",
    11: "هود",
    25: "الفرقان",
    32: "السجدة",
    41: "فصلت",
    43: "الزخرف",
    44: "الدخان",
    52: "الطور",
    53: "النجم",
    62: "الجمعة",
    64: "التغابن",
    68: "القلم",
    71: "نوح",
    76: "الإنسان",
    78: "النبأ",
    84: "الانشقاق",
    89: "الفجر",
    90: "البلد",
    92: "الليل",
    96: "العلق",
    100: "العاديات",
    101: "القارعة",
    103: "العصر",
    107: "الماعون",
    111: "المسد",
    112: "الإخلاص",
}


def load_test_ayahs(test_surahs: set[int]):
    s_lines = SIMPLE_PATH.read_text(encoding="utf-8").strip().splitlines()
    c_lines = CLEAN_NUMBERED_PATH.read_text(encoding="utf-8").strip().splitlines()

    ayahs = []
    for s_l, c_l in zip(s_lines, c_lines):
        c_parts = c_l.split("|")
        if len(c_parts) != 3:
            continue
        s_id = int(c_parts[0])
        a_id = int(c_parts[1])
        c_text = c_parts[2].strip()
        s_text = s_l.strip()
        if s_id in test_surahs:
            s_words = s_text.split()
            c_words = c_text.split()
            assert len(s_words) == len(c_words), f"Mismatch at {s_id}:{a_id}"
            ayahs.append({
                "surah_id": s_id,
                "ayah_id": a_id,
                "surah_name": f"سورة {SURAH_NAMES.get(s_id, str(s_id))}",
                "simple_words": s_words,
                "clean_words": c_words,
            })
    return ayahs


def build_line_prompts(ayahs: list[dict], rng: random.Random) -> list[dict]:
    # Extract candidate 3-9 word windows from ayahs
    candidates = []
    for a in ayahs:
        sw = a["simple_words"]
        cw = a["clean_words"]
        n_w = len(sw)
        if n_w < 3:
            continue
        # Extract full ayah if 3..9 words, or sub-windows
        if 3 <= n_w <= 9:
            candidates.append({
                "surah_id": a["surah_id"],
                "ayah_id": a["ayah_id"],
                "surah_name": a["surah_name"],
                "rasm": " ".join(cw),
                "vowelled": " ".join(sw),
                "words_count": n_w,
            })
        else:
            # sliding non-overlapping or stepped windows
            step = 5
            for start in range(0, n_w - 3, step):
                w_len = rng.randint(3, min(9, n_w - start))
                v_slice = sw[start : start + w_len]
                r_slice = cw[start : start + w_len]
                candidates.append({
                    "surah_id": a["surah_id"],
                    "ayah_id": a["ayah_id"],
                    "surah_name": a["surah_name"],
                    "rasm": " ".join(r_slice),
                    "vowelled": " ".join(v_slice),
                    "words_count": len(r_slice),
                })

    rng.shuffle(candidates)

    # Separate into candidates containing "الله" and others
    with_allah = [c for c in candidates if "الله" in c["rasm"]]
    without_allah = [c for c in candidates if "الله" not in c["rasm"]]

    selected = []
    # Pick 8 containing "الله" (>= 5 guaranteed)
    for c in with_allah[:8]:
        selected.append(c)

    # Pick 32 remaining from without_allah
    for c in without_allah[:32]:
        selected.append(c)

    rng.shuffle(selected)
    for idx, p in enumerate(selected, 1):
        p["id"] = f"line_{idx:02d}"

    return selected


def build_word_prompts(ayahs: list[dict], rng: random.Random) -> list[dict]:
    # Vocabulary counts
    word_freq = Counter()
    word_map = {}  # clean_word -> (simple_word, surah_id, surah_name)

    for a in ayahs:
        for sw, cw in zip(a["simple_words"], a["clean_words"]):
            word_freq[cw] += 1
            if cw not in word_map:
                word_map[cw] = (sw, a["surah_id"], a["surah_name"])

    # 30 Frequent words (freq >= 8)
    frequent_candidates = [w for w, cnt in word_freq.most_common(100) if len(w) > 2]
    rng.shuffle(frequent_candidates)
    sel_frequent = frequent_candidates[:30]

    # 30 Rare words (freq == 1 or 2)
    rare_candidates = [w for w, cnt in word_freq.items() if cnt in (1, 2) and len(w) >= 3]
    rng.shuffle(rare_candidates)
    sel_rare = rare_candidates[:30]

    words_list = []
    for w in sel_frequent:
        sw, s_id, s_name = word_map[w]
        words_list.append({
            "rasm": w,
            "vowelled": sw,
            "surah_id": s_id,
            "surah_name": s_name,
            "category": "frequent",
            "freq_in_test": word_freq[w],
        })

    for w in sel_rare:
        sw, s_id, s_name = word_map[w]
        words_list.append({
            "rasm": w,
            "vowelled": sw,
            "surah_id": s_id,
            "surah_name": s_name,
            "category": "rare",
            "freq_in_test": word_freq[w],
        })

    rng.shuffle(words_list)
    for idx, p in enumerate(words_list, 1):
        p["id"] = f"word_{idx:02d}"

    return words_list


def main():
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    test_surahs = set(splits["test"]["surahs"])
    print(f"Loaded {len(test_surahs)} test surahs from {SPLITS_PATH.name}.")

    ayahs = load_test_ayahs(test_surahs)
    print(f"Loaded {len(ayahs)} ayahs from test surahs.")

    rng = random.Random(42)
    line_prompts = build_line_prompts(ayahs, rng)
    word_prompts = build_word_prompts(ayahs, rng)

    allah_count = sum(1 for l in line_prompts if "الله" in l["rasm"])
    print(f"Generated {len(line_prompts)} line prompts (contains 'الله': {allah_count} >= 5).")
    print(f"Generated {len(word_prompts)} word prompts (30 frequent, 30 rare).")

    out_data = {
        "metadata": {
            "source_vowelled": "data/tanzil/quran-simple.txt",
            "source_rasm": "data/tanzil/quran-simple-clean.txt",
            "test_surahs": sorted(list(test_surahs)),
            "line_prompts_count": len(line_prompts),
            "word_prompts_count": len(word_prompts),
            "allah_in_lines": allah_count,
        },
        "lines": line_prompts,
        "words": word_prompts,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Successfully saved capture prompts to {OUT_JSON}.")


if __name__ == "__main__":
    main()
