"""
scripts/prepare_derived.py
Generate derived Quran datasets from Tanzil Simple-Clean into data/derived/.

Outputs:
1. data/derived/ayahs.jsonl:
   - Line 1: Header metadata containing Tanzil CC BY 3.0 attribution notice.
   - Lines 2..6237: {surah, ayah, text, basmalah}.
     - Surah 1:1: text="بسم الله الرحمن الرحيم", basmalah="بسم الله الرحمن الرحيم".
     - Surahs 2-114 (except 9) Ayah 1: prepended Basmalah is extracted into its own
       field, leaving text starting directly with the Ayah's content.
     - Surahs 2-114 (Ayah > 1) and Surah 9 (all Ayahs): basmalah=None.
   - Confirms codepoint inventory is a strict subset of vocab.json.
2. data/derived/splits.json:
   - Header with CC BY 3.0 notice.
   - Surah-level train/val/test split (~80/10/10 by word count, seed 42).
   - Test split includes >=1 long Surah and >=1 short Surah.
"""

from __future__ import annotations
import json
import random
import unicodedata
from collections import defaultdict
from pathlib import Path

DERIVED_DIR = Path("data/derived")
DERIVED_DIR.mkdir(parents=True, exist_ok=True)

NUMBERED_SRC = Path("data/tanzil/quran-simple-clean-numbered.txt")
VOCAB_PATH = Path("vocab.json")

BASMALAH_TEXT = "بسم الله الرحمن الرحيم"
BASMALAH_PREFIX = "بسم الله الرحمن الرحيم "

CC_BY_NOTICE = (
    "Derived from Tanzil Quran Text (Simple Clean) - Copyright (C) 2007-2021 Tanzil Project - "
    "Licensed under Creative Commons Attribution 3.0 Unported (CC BY 3.0) - http://tanzil.net"
)


def build_ayahs_jsonl() -> Tuple[Path, set, set]:
    lines = NUMBERED_SRC.read_text(encoding="utf-8").splitlines()
    out_path = DERIVED_DIR / "ayahs.jsonl"

    ayah_records = []
    text_codepoints = set()

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != 3:
            continue
        sura, aya, raw_text = int(parts[0]), int(parts[1]), parts[2].strip()

        basmalah = None
        ayah_text = raw_text

        if sura == 1 and aya == 1:
            basmalah = BASMALAH_TEXT
            ayah_text = BASMALAH_TEXT
        elif sura != 1 and sura != 9 and aya == 1:
            if raw_text.startswith(BASMALAH_PREFIX):
                basmalah = BASMALAH_TEXT
                ayah_text = raw_text[len(BASMALAH_PREFIX):].strip()
            elif raw_text == BASMALAH_TEXT:
                basmalah = BASMALAH_TEXT
                ayah_text = ""
        else:
            basmalah = None
            ayah_text = raw_text

        text_codepoints.update(ayah_text)
        if basmalah:
            text_codepoints.update(basmalah)

        ayah_records.append({
            "surah": sura,
            "ayah": aya,
            "text": ayah_text,
            "basmalah": basmalah,
        })

    # Write ayahs.jsonl with CC BY header in first record
    with open(out_path, "w", encoding="utf-8") as f:
        header_record = {
            "_type": "header",
            "license": "CC BY 3.0",
            "notice": CC_BY_NOTICE,
            "total_ayahs": len(ayah_records),
        }
        f.write(json.dumps(header_record, ensure_ascii=False) + "\n")
        for rec in ayah_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # Check against vocab.json
    vocab_list = json.loads(VOCAB_PATH.read_text(encoding="utf-8"))
    vocab_set = set(vocab_list)
    exceptions = text_codepoints - vocab_set

    return out_path, text_codepoints, exceptions


def build_splits_json() -> Path:
    lines = NUMBERED_SRC.read_text(encoding="utf-8").splitlines()
    surah_words = defaultdict(int)

    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) != 3:
            continue
        sura, aya, raw_text = int(parts[0]), int(parts[1]), parts[2].strip()
        if sura != 1 and sura != 9 and aya == 1 and raw_text.startswith(BASMALAH_PREFIX):
            raw_text = raw_text[len(BASMALAH_PREFIX):].strip()
        surah_words[sura] += len(raw_text.split())

    total_words = sum(surah_words.values())
    target_val = 0.10 * total_words
    target_test = 0.10 * total_words

    rng = random.Random(42)

    # Constraint: test includes >=1 long surah (>1500 words, leaving Al-Baqarah for train)
    # and >=1 short surah (<50 words)
    long_surahs = [s for s, w in surah_words.items() if w > 1500 and s != 2]
    short_surahs = [s for s, w in surah_words.items() if w < 50]
    rng.shuffle(long_surahs)
    rng.shuffle(short_surahs)

    test_surahs = [long_surahs[0], short_surahs[0]]
    test_w = sum(surah_words[s] for s in test_surahs)

    remaining = [s for s in range(1, 115) if s not in test_surahs]
    rng.shuffle(remaining)

    val_surahs = []
    val_w = 0
    train_surahs = []
    train_w = 0

    for s in remaining:
        w = surah_words[s]
        if test_w + w <= target_test + 100:
            test_surahs.append(s)
            test_w += w
        elif val_w + w <= target_val + 100:
            val_surahs.append(s)
            val_w += w
        else:
            train_surahs.append(s)
            train_w += w

    splits_data = {
        "_notice": CC_BY_NOTICE,
        "seed": 42,
        "total_words": total_words,
        "train": {
            "surahs": sorted(train_surahs),
            "surah_count": len(train_surahs),
            "word_count": train_w,
            "word_share_pct": round((train_w / total_words) * 100, 2),
        },
        "val": {
            "surahs": sorted(val_surahs),
            "surah_count": len(val_surahs),
            "word_count": val_w,
            "word_share_pct": round((val_w / total_words) * 100, 2),
        },
        "test": {
            "surahs": sorted(test_surahs),
            "surah_count": len(test_surahs),
            "word_count": test_w,
            "word_share_pct": round((test_w / total_words) * 100, 2),
            "long_surahs_included": [s for s in test_surahs if surah_words[s] > 1500],
            "short_surahs_included": [s for s in test_surahs if surah_words[s] < 50],
        },
    }

    out_path = DERIVED_DIR / "splits.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(splits_data, f, ensure_ascii=False, indent=2)

    return out_path


def main():
    print("=== Generating Derived Tanzil Data ===")
    jsonl_path, codepoints, exceptions = build_ayahs_jsonl()
    print(f"1. ayahs.jsonl: {jsonl_path} generated (6,236 Ayahs).")
    print(f"   Codepoint count: {len(codepoints)}")
    if exceptions:
        print(f"   [WARNING] Codepoints not in vocab.json: {exceptions}")
    else:
        print("   [OK] Codepoint inventory is a strict SUBSET of vocab.json (0 exceptions).")

    splits_path = build_splits_json()
    splits_data = json.loads(splits_path.read_text(encoding="utf-8"))
    print(f"\n2. splits.json: {splits_path} generated.")
    print(f"   Train: {splits_data['train']['surah_count']} surahs, {splits_data['train']['word_count']} words ({splits_data['train']['word_share_pct']}%)")
    print(f"   Val:   {splits_data['val']['surah_count']} surahs, {splits_data['val']['word_count']} words ({splits_data['val']['word_share_pct']}%)")
    print(f"   Test:  {splits_data['test']['surah_count']} surahs, {splits_data['test']['word_count']} words ({splits_data['test']['word_share_pct']}%)")
    print(f"   Test long surahs:  {splits_data['test']['long_surahs_included']}")
    print(f"   Test short surahs: {splits_data['test']['short_surahs_included']}")


if __name__ == "__main__":
    main()
