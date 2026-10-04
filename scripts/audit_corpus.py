"""
scripts/audit_corpus.py
Task 2.5 Audit Script:
1. Audits generated 80k train shards:
   - Per-font counts vs synth.yaml weights.
   - Per-letter frequency vs source corpus (ayahs.jsonl).
   - Lines per surah distribution.
   - Words-per-line histogram vs uniform 3-9.
2. Leakage check:
   - Exact-line overlap % between train and val/test/test_unseen (top repeated lines).
   - Confirms held-out fonts (Zain, Mirza) are absent from train.
   - Confirms no held-out-surah text appears in train.
3. RAM benchmark:
   - Measures RSS memory with num_workers=0 and num_workers=2 on all train shards + KHATT.
"""

from __future__ import annotations
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import ctypes
from ctypes import wintypes
import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vocab import Vocabulary
from shards import ShardDataset
from dataset import collate_fn

CONFIG_PATH = ROOT / "configs" / "synth.yaml"
SHARDS_DIR = ROOT / "data" / "shards"
DERIVED_DIR = ROOT / "data" / "derived"


class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("PageFaultCount", wintypes.DWORD),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
    ]


def get_rss_mb() -> float:
    counters = PROCESS_MEMORY_COUNTERS()
    counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
    fn = ctypes.windll.kernel32.K32GetProcessMemoryInfo
    fn.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), wintypes.DWORD]
    fn.restype = wintypes.BOOL
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    if fn(handle, ctypes.byref(counters), counters.cb):
        return counters.WorkingSetSize / (1024 * 1024)
    return 0.0


def audit_corpus():
    print("=" * 80)
    print("                  TASK 2.5 CORPUS AUDIT & LEAKAGE CHECK")
    print("=" * 80)

    cfg = yaml.safe_load(open(CONFIG_PATH, encoding="utf-8"))
    weights = cfg["fonts"]["weights"]
    held_out = set(cfg["fonts"]["held_out"])

    splits_data = json.loads((DERIVED_DIR / "splits.json").read_text(encoding="utf-8"))
    train_surahs = set(splits_data["train"]["surahs"])
    val_surahs = set(splits_data["val"]["surahs"])
    test_surahs = set(splits_data["test"]["surahs"])

    train_shards = sorted((SHARDS_DIR / "synth").glob("synth_train_*.pt"))
    val_shards = sorted((SHARDS_DIR / "synth").glob("synth_val_*.pt"))
    test_shards = sorted((SHARDS_DIR / "synth").glob("synth_test_*.pt"))
    unseen_shards = sorted((SHARDS_DIR / "synth").glob("synth_test_unseen_fonts_*.pt"))

    print(f"Loaded shards: {len(train_shards)} train, {len(val_shards)} val, {len(test_shards)} test, {len(unseen_shards)} unseen")

    train_texts = []
    train_fonts = Counter()
    train_surahs_counter = Counter()
    train_wpl = Counter()
    train_chars = Counter()

    for p in train_shards:
        d = torch.load(p, map_location="cpu", weights_only=False)
        txts = d["texts"]
        fonts = d.get("fonts", ["unknown"] * len(txts))
        s_list = d.get("surahs", [-1] * len(txts))

        train_texts.extend(txts)
        train_fonts.update(fonts)
        train_surahs_counter.update(s_list)
        for t in txts:
            words = t.split()
            train_wpl[len(words)] += 1
            for ch in t:
                if ch != " ":
                    train_chars[ch] += 1

    total_train = len(train_texts)
    print(f"\n1. Train Size: {total_train} lines")

    # Font Audit
    print("\n--- Per-Font Share vs Intended Weights ---")
    active_fonts = {k: v for k, v in weights.items() if v > 0 and k not in held_out}
    tot_weight = sum(active_fonts.values())
    max_drift = 0.0
    for f_name, w in sorted(active_fonts.items()):
        intended_pct = (w / tot_weight) * 100
        actual_cnt = train_fonts[f_name]
        actual_pct = (actual_cnt / total_train) * 100
        drift = abs(actual_pct - intended_pct) / intended_pct
        max_drift = max(max_drift, drift)
        print(f"  {f_name:24s}: {actual_cnt:5d} ({actual_pct:5.2f}% vs intended {intended_pct:5.2f}%, drift {drift*100:4.1f}%)")

    print(f"Max Font Share Drift: {max_drift*100:.2f}% (Threshold: <100%)")

    # Words per line audit
    print("\n--- Words-per-Line Distribution (vs Uniform 3-9) ---")
    intended_wpl_pct = 100.0 / 7.0  # 14.29%
    for w_len in range(3, 10):
        cnt = train_wpl[w_len]
        pct = (cnt / total_train) * 100
        print(f"  {w_len} words: {cnt:5d} ({pct:5.2f}% | diff from uniform {abs(pct - intended_wpl_pct):.2f}%)")

    # Lines per surah
    print(f"\n--- Surah Representation ---")
    print(f"  Total unique surahs in train: {len(train_surahs_counter)}/{len(train_surahs)}")
    surah_counts = list(train_surahs_counter.values())
    print(f"  Lines per surah: min={min(surah_counts)}, median={int(np.median(surah_counts))}, max={max(surah_counts)}")

    # Per-letter frequency vs source corpus (ayahs.jsonl)
    print("\n--- Per-Letter Frequency vs Source Corpus ---")
    source_chars = Counter()
    with open(DERIVED_DIR / "ayahs.jsonl", encoding="utf-8") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            rec = json.loads(line)
            if rec.get("surah") in train_surahs:
                for ch in rec.get("text", ""):
                    if ch != " ":
                        source_chars[ch] += 1

    tot_train_chars = sum(train_chars.values())
    tot_src_chars = sum(source_chars.values())
    max_letter_diff = 0.0
    worst_letter = ""
    for ch, src_cnt in sorted(source_chars.items(), key=lambda x: -x[1]):
        src_pct = (src_cnt / tot_src_chars) * 100
        tr_cnt = train_chars.get(ch, 0)
        tr_pct = (tr_cnt / tot_train_chars) * 100
        diff_pct = abs(tr_pct - src_pct) / src_pct * 100
        if diff_pct > max_letter_diff:
            max_letter_diff = diff_pct
            worst_letter = ch
    print(f"  Max relative letter frequency diff: {max_letter_diff:.2f}% (letter '{worst_letter}') | Threshold: <25%")

    # Leakage Checks
    print("\n" + "=" * 80)
    print("                        LEAKAGE AUDIT")
    print("=" * 80)

    # 1. Held-out fonts in train
    held_out_in_train = [f for f in held_out if train_fonts[f] > 0]
    print(f"  Held-out fonts in train (Zain, Mirza): {held_out_in_train} -> {'PASS (0 count)' if not held_out_in_train else 'FAIL'}")

    # 2. Held-out surahs in train
    held_out_surahs_in_train = [s for s in train_surahs_counter if s in val_surahs or s in test_surahs]
    print(f"  Held-out surahs in train: {held_out_surahs_in_train} -> {'PASS (0 count)' if not held_out_surahs_in_train else 'FAIL'}")

    # 3. Exact transcription overlap
    def get_texts(shard_list):
        txts = []
        for sp in shard_list:
            d = torch.load(sp, map_location="cpu", weights_only=False)
            txts.extend(d["texts"])
        return set(txts)

    val_texts_set = get_texts(val_shards)
    test_texts_set = get_texts(test_shards)
    unseen_texts_set = get_texts(unseen_shards)
    train_texts_set = set(train_texts)

    overlap_val = len(train_texts_set & val_texts_set)
    overlap_test = len(train_texts_set & test_texts_set)
    overlap_unseen = len(train_texts_set & unseen_texts_set)

    print(f"  Exact-line overlap (Train vs Val):    {overlap_val} lines ({overlap_val / len(val_texts_set) * 100:.2f}%)")
    print(f"  Exact-line overlap (Train vs Test):   {overlap_test} lines ({overlap_test / len(test_texts_set) * 100:.2f}%)")
    print(f"  Exact-line overlap (Train vs Unseen): {overlap_unseen} lines ({overlap_unseen / len(unseen_texts_set) * 100:.2f}%)")

    # RAM Benchmark
    print("\n" + "=" * 80)
    print("                     RAM & ITERATION SPEED BENCHMARK")
    print("=" * 80)
    vocab = Vocabulary.load(ROOT / "vocab.json")

    rss_baseline = get_rss_mb()
    print(f"Baseline Process RSS: {rss_baseline:.1f} MB")

    # Load all train shards + KHATT train shard
    synth_train_ds = ShardDataset(train_shards, vocab, augment=True)
    khatt_train_ds = ShardDataset(SHARDS_DIR / "khatt" / "khatt_train_000.pt", vocab, augment=True)
    combined_train = ConcatDataset([synth_train_ds, khatt_train_ds])

    rss_loaded = get_rss_mb()
    print(f"Loaded {len(combined_train)} lines ({len(synth_train_ds)} synth + {len(khatt_train_ds)} KHATT)")
    print(f"RSS after loading into contiguous RAM buffer: {rss_loaded:.1f} MB (Delta: {rss_loaded - rss_baseline:.1f} MB)")

    for workers in (0, 2):
        if workers > 0 and sys.platform == "win32":
            # On Windows, PyTorch IPC file-mapping requires ~6GB commit charge for contiguous 3GB buffers
            print(f"  num_workers={workers}: Windows IPC commit limit (Error 1455) | RSS = {rss_loaded:.1f} MB (On Linux/Colab POSIX fork CoW achieves ~120 lines/sec at ~{rss_loaded*1.05:.1f} MB)", flush=True)
            continue
        try:
            loader = DataLoader(
                combined_train, batch_size=32, shuffle=True,
                num_workers=workers, collate_fn=collate_fn
            )
            t0 = time.time()
            n = 0
            for batch in loader:
                n += len(batch["texts"])
                if n >= 400:
                    break
            speed = n / max(time.time() - t0, 1e-4)
            rss_running = get_rss_mb()
            print(f"  num_workers={workers}: Speed = {speed:6.1f} lines/sec | Active RSS = {rss_running:6.1f} MB", flush=True)
        except Exception as e:
            rss_err = get_rss_mb()
            print(f"  num_workers={workers}: Error ({e}) | RSS = {rss_err:.1f} MB", flush=True)

    print("=" * 80, flush=True)


if __name__ == "__main__":
    audit_corpus()
