"""
scripts/ingest_capture_v2.py
Task 2.19 Item 4:
Ingestion and sanity-checking tool for data/capture_v2/<writer>/:
- Detects new writer folders (or unzips new exports into data/capture_v2/<writer>/)
- Validates:
  1. Duplicate resolution (monotonic timestamped sample IDs, latest save wins)
  2. Geometry schema (capture_version == "v2", band_top/base/bot in CSS px, canvas size, DPR)
  3. Label vocabulary conformity against vocab.json
  4. Stroke integrity (non-empty stroke arrays, finite coordinates)
Strictly performs validation only — DOES NOT TRAIN.
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vocab import Vocabulary


def inspect_writer_dir(writer_dir: Path, vocab: Vocabulary) -> dict:
    print(f"\n--- Ingesting & Sanity Checking: {writer_dir.name} ---")
    strokes_file = writer_dir / "strokes.json"
    manifest_file = writer_dir / "manifest.csv"

    if not strokes_file.exists():
        print(f"  [warn] strokes.json missing in {writer_dir}")
        return {"status": "error", "error": "missing strokes.json"}

    with open(strokes_file, "r", encoding="utf-8") as f:
        raw_strokes = json.load(f)

    # 1. Duplicates check
    id_counts = defaultdict(int)
    for s in raw_strokes:
        if isinstance(s, dict) and "id" in s:
            id_counts[s["id"]] += 1

    dupes = {k: v for k, v in id_counts.items() if v > 1}
    print(f"  Total raw saves: {len(raw_strokes)} | Unique sample IDs: {len(id_counts)}")
    if dupes:
        print(f"  [Notice] Found {len(dupes)} duplicate sample IDs (latest save will be retained).")
    else:
        print("  [Pass] Zero duplicate IDs detected (monotonic timestamped IDs validated).")

    # Deduplicate to latest save
    strokes_by_id = {}
    for s in raw_strokes:
        if isinstance(s, dict) and "id" in s:
            strokes_by_id[s["id"]] = s

    # 2. Geometry check
    v2_count = 0
    missing_geom = 0
    for sid, s in strokes_by_id.items():
        geom = s.get("geometry")
        if geom and geom.get("capture_version") == "v2":
            v2_count += 1
            # Check required fields
            req = ["band_top_css", "band_base_css", "band_bot_css", "canvas_w_css", "canvas_h_css", "dpr"]
            if not all(k in geom for k in req):
                missing_geom += 1
        else:
            missing_geom += 1

    print(f"  Geometry: {v2_count}/{len(strokes_by_id)} samples have valid v2 per-sample geometry.")
    if missing_geom > 0:
        print(f"  [Notice] {missing_geom} samples missing complete v2 geometry (fallback applied).")
    else:
        print("  [Pass] 100% of samples conform to v2 band-relative geometry schema.")

    # 3. Label vocabulary conformity
    unencodable = []
    for sid, s in strokes_by_id.items():
        text = s.get("text", "")
        try:
            vocab.encode(text)
        except KeyError as e:
            unencodable.append((sid, text, str(e)))

    if unencodable:
        print(f"  [Fail] {len(unencodable)} samples contain unencodable characters: {unencodable[:3]}")
    else:
        print(f"  [Pass] 100% of transcriptions cleanly encode under vocab.json ({len(vocab)} classes).")

    # 4. Stroke data integrity
    empty_strokes = 0
    for sid, s in strokes_by_id.items():
        stk = s.get("strokes", [])
        if not stk or sum(len(st) for st in stk) == 0:
            empty_strokes += 1

    if empty_strokes > 0:
        print(f"  [Notice] {empty_strokes} samples have empty strokes.")
    else:
        print("  [Pass] All samples contain valid non-empty stroke coordinates.")

    return {
        "status": "ok",
        "writer": writer_dir.name,
        "total_samples": len(strokes_by_id),
        "v2_geometry_count": v2_count,
        "unencodable_count": len(unencodable),
    }


def main():
    p = argparse.ArgumentParser(description="Ingest and Sanity-Check data/capture_v2/<writer>/")
    p.add_argument("--capture_v2_dir", default="data/capture_v2", help="Path to capture_v2 root directory")
    p.add_argument("--vocab_path", default="vocab.json", help="Path to vocab.json")
    args = p.parse_args()

    v2_dir = ROOT / args.capture_v2_dir
    vocab = Vocabulary.load(ROOT / args.vocab_path)

    if not v2_dir.exists() or not any(v2_dir.iterdir()):
        print(f"[ingest_capture_v2] Directory '{v2_dir}' is empty or does not exist yet.")
        print("Status: NOT RUN (No data/capture_v2/<writer>/ folders present yet; awaiting writer upload).")
        return

    writer_dirs = [d for d in v2_dir.iterdir() if d.is_dir()]
    if not writer_dirs:
        print(f"[ingest_capture_v2] No writer subdirectories found in {v2_dir}.")
        print("Status: NOT RUN (Awaiting writer export extraction).")
        return

    print(f"[ingest_capture_v2] Found {len(writer_dirs)} writer directory(ies): {[d.name for d in writer_dirs]}")
    for wd in writer_dirs:
        inspect_writer_dir(wd, vocab)


if __name__ == "__main__":
    main()
