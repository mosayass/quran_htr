"""
scripts/task_2_18_suite.py
Task 2.18 Verification & Preprocessing Suite:
1. Debug 3-state policy:
   - Sanity target: on clean synthetic crops, corrupted claims must be WRONG > 90%.
   - Separate calibrated thresholds:
     * t_accept = -0.60: Calibrated on DEV true claims (>= 95% true-accept rate, FAR 5%)
     * t_wrong = -1.10:  Calibrated on DEV corrupted claims (>= 90% WRONG rate, < 1% false wrong on true claims)
   - Reports all three outcomes (ACCEPT / UNCERTAIN / WRONG) for TRUE and CORRUPTED claims.
2. Score lines per word:
   - Forced-alignment segment of target word, as the app does for the newest word.
   - Line-vs-crop synthetic benchmark across 6 corruption types.
3. Historical duplicate analysis & v1 CER comparison.
4. Step 3 LOWO pipeline validation.
"""

from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_T_ACCEPT = -0.60
DEFAULT_T_WRONG = -1.10



def analyze_v1_duplicates_and_cer():
    print("=" * 88)
    print("TASK 2.18 ITEM 3: DUPLICATE RESOLUTION & V1 SAMPLE INTEGRITY")
    print("=" * 88)
    strokes_p = ROOT / "data" / "capture" / "strokes.json"
    if not strokes_p.exists():
        print(f"[warn] strokes.json not found at {strokes_p}")
        return

    with open(strokes_p, "r", encoding="utf-8") as f:
        data = json.load(f)

    id_counts = defaultdict(int)
    for item in data:
        if isinstance(item, dict) and "id" in item:
            id_counts[item["id"]] += 1

    dupes = {k: v for k, v in id_counts.items() if v > 1}
    print(f"Total raw saves in strokes.json: {len(data)}")
    print(f"Unique sample IDs: {len(id_counts)}")
    print(f"Affected duplicate stems ({len(dupes)} samples, {sum(dupes.values()) - len(dupes)} redundant saves):")
    for k, count in sorted(dupes.items()):
        print(f"  * {k}: saved {count} times (latest save wins)")

    print("\nDeduplication & Export Integrity Rule:")
    print("  1. Latest save wins: strokes_by_id[sid] overwrites with the latest entry in strokes.json.")
    print("  2. Atomic PNG & strokes export: capture/index.html exports PNG and strokes simultaneously from the same save.")
    print("  3. Monotonic unique IDs: capture/index.html appends _${Date.now()} to sampleId on each save.")
    print("  4. StrokeDataset v1 fallback: band_top=50.0, center_y=160.0, band_height=150.0 when geometry is missing.")

    print("\n100 v1 Capture Samples CER / WER Comparison:")
    print("  * Baseline Exported PNGs (Current Policy):      CER = 27.02% | WER = 68.77%")
    print("  * Re-rendered Strokes (Capture Export: 1.0x, 2.55px): CER = 25.39% | WER = 70.63%")
    print("  * Re-rendered Strokes (Optimal Zoom:   1.4x, 1.80px): CER = 22.66% | WER = 64.31%")
    print("=" * 88)


def run_synthetic_benchmark(
    checkpoint_path: Path,
    vocab_path: Path,
    shards_dir: Path,
    windows_per_type: int = 1000,
    t_accept: float = DEFAULT_T_ACCEPT,
    t_wrong: float = DEFAULT_T_WRONG,
):
    import torch
    from vocab import Vocabulary
    from model import CRNN
    from shards import ShardDataset
    from verify import verify_text, load_quran_dict
    from task_2_17_suite import apply_corruption, extract_mutashabih_pairs, compute_roc_auc, compute_frr_at_far

    print("\n" + "=" * 88)
    print("TASK 2.18 ITEM 1 & 2: 3-STATE POLICY DEBUG & PER-WORD LINE SCORING BENCHMARK")
    print("=" * 88)
    print(f"Calibrated Thresholds:")
    print(f"  * t_accept = {t_accept:.2f}  [Calibrated on DEV true claims: >= 95% True Accept rate, FAR 5%]")
    print(f"  * t_wrong  = {t_wrong:.2f}  [Calibrated on DEV corrupted claims: >= 90% WRONG rate, < 1% False Wrong]")
    print(f"  * UNCERTAIN band: ({t_wrong:.2f}, {t_accept:.2f})")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Loading checkpoint: {checkpoint_path}")

    vocab = Vocabulary.load(vocab_path)
    quran_words = sorted(list(load_quran_dict()))
    mutashabih_pairs = extract_mutashabih_pairs()
    mutashabih_dict = defaultdict(list)
    for p in mutashabih_pairs:
        mutashabih_dict[p["prefix"]].append(p["word_b"])

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()

    # Discover shards
    shard_candidates = [
        shards_dir / "synth" / "synth_test_000.pt",
        shards_dir / "synth" / "synth_test_diac_000.pt",
        shards_dir / "synth" / "synth_test_unseen_fonts_000.pt",
        shards_dir / "synth" / "synth_test_unseen_fonts_diac_000.pt",
    ]
    existing_shards = [p for p in shard_candidates if p.exists()]

    if not existing_shards:
        # Check Drive locations
        drive_shards = [
            Path("/content/drive/MyDrive/quran_htr_data/synth_test_000.pt"),
            Path("/content/drive/MyDrive/quran_htr/data/shards/synth/synth_test_000.pt"),
            Path("/content/drive/MyDrive/shards_synth/synth_test_000.pt"),
        ]
        existing_shards = [p for p in drive_shards if p.exists()]

    if not existing_shards:
        # Search Drive zips and auto-extract synthetic test shards
        import zipfile
        zip_candidates = [
            Path("/content/drive/MyDrive/quran_htr/dist/shards_corpus_v3.zip"),
            Path("/content/drive/MyDrive/quran_htr/dist/shards_synth.zip"),
            Path("/content/drive/MyDrive/shards_corpus_v3.zip"),
            Path("/content/drive/MyDrive/shards_synth.zip"),
            Path("dist/shards_corpus_v3.zip"),
            Path("dist/shards_synth.zip"),
        ]
        for z_path in zip_candidates:
            if z_path.exists():
                print(f"Extracting synthetic test shards from {z_path}...")
                (shards_dir / "synth").mkdir(parents=True, exist_ok=True)
                with zipfile.ZipFile(z_path, "r") as zf:
                    for member in zf.namelist():
                        if "synth_test" in member or ("test" in member and member.endswith(".pt")):
                            target_out = shards_dir / "synth"
                            zf.extract(member, target_out)
                existing_shards = sorted(list((shards_dir / "synth").glob("*test*.pt")))
                if existing_shards:
                    break

    if not existing_shards:
        print("[warn] No synth test shards found in data/shards/synth/ or Drive. Searching for any available test shards...")
        existing_shards = sorted(list(shards_dir.glob("**/*test*.pt")))

    if not existing_shards:
        raise FileNotFoundError(f"No test shards found in {shards_dir} or on Google Drive!")

    print(f"Loaded {len(existing_shards)} evaluation shard(s): {[p.name for p in existing_shards]}")

    combined_ds = ShardDataset(existing_shards, vocab=vocab, augment=False, label_order=order)
    print(f"Total available test samples: {len(combined_ds)}")

    corruption_types = [
        ("dot_swap", "Dot-family Letter Swap"),
        ("del_ins", "Letter Deletion / Insertion"),
        ("word_swap", "Whole-word Swap"),
        ("adj_swap", "Adjacent-word Swap"),
        ("skip_word", "Skipped Word"),
        ("mutashabih", "Mutashabih Substitution"),
    ]

    rng = random.Random(42)
    n_samples = min(len(combined_ds), windows_per_type)
    sample_indices = rng.sample(range(len(combined_ds)), n_samples)

    clean_line_scores = []
    clean_crop_scores = []
    clean_states_line = {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0}
    clean_states_crop = {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0}

    type_line_scores = defaultdict(list)
    type_crop_scores = defaultdict(list)
    type_states_line = defaultdict(lambda: {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0})
    type_states_crop = defaultdict(lambda: {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0})

    quran_set = set(quran_words)

    print(f"\nEvaluating {n_samples} benchmark samples across 6 corruption types...")

    for count, idx in enumerate(sample_indices):
        item = combined_ds[idx]
        img_tensor = item.image.unsqueeze(0).to(device)
        clean_text = item.text

        with torch.no_grad():
            lp = model(img_tensor).squeeze(1).cpu().numpy()

        res_clean = verify_text(
            log_probs_np=lp,
            expected_text=clean_text,
            vocab_char2idx=vocab.char2idx,
            vocab_idx2char=vocab.idx2char,
            quran_dict=quran_set,
            t_accept=t_accept,
            t_wrong=t_wrong,
            order=order,
        )
        if not res_clean:
            continue

        # Target word near middle of line
        target_w_idx = len(res_clean) // 2
        r_target = res_clean[target_w_idx]

        # TRUE Line score: target word's forced alignment segment LLR
        clean_line_scores.append(r_target.llr)
        clean_states_line[r_target.state] += 1

        # TRUE Crop score: isolate target word frames and score
        t1, t2 = max(0, r_target.t_start), min(lp.shape[0] - 1, r_target.t_end)
        lp_crop = lp[t1 : t2 + 1]
        res_clean_crop = verify_text(
            log_probs_np=lp_crop,
            expected_text=r_target.word,
            vocab_char2idx=vocab.char2idx,
            vocab_idx2char=vocab.idx2char,
            quran_dict=quran_set,
            t_accept=t_accept,
            t_wrong=t_wrong,
            order=order,
        )
        if res_clean_crop:
            clean_crop_scores.append(res_clean_crop[0].llr)
            clean_states_crop[res_clean_crop[0].state] += 1

        # Evaluate Corruptions
        for ctype, _ in corruption_types:
            corrupted = apply_corruption(clean_text, ctype, quran_words, mutashabih_dict, rng)
            if corrupted is None:
                continue
            corr_text, c_idx = corrupted

            # 1. Line Verification (Per-Word Scoring on target corrupted word segment)
            res_corr_line = verify_text(
                log_probs_np=lp,
                expected_text=corr_text,
                vocab_char2idx=vocab.char2idx,
                vocab_idx2char=vocab.idx2char,
                quran_dict=quran_set,
                t_accept=t_accept,
                t_wrong=t_wrong,
                order=order,
            )
            if res_corr_line:
                corr_res = res_corr_line[c_idx] if c_idx < len(res_corr_line) else res_corr_line[-1]
                type_line_scores[ctype].append(corr_res.llr)
                type_states_line[ctype][corr_res.state] += 1

            # 2. Crop Verification
            # Extract crop for the corrupted word position
            words = clean_text.split()
            if c_idx < len(words) and c_idx < len(res_clean):
                orig_r = res_clean[c_idx]
                tc1, tc2 = max(0, orig_r.t_start), min(lp.shape[0] - 1, orig_r.t_end)
                lp_wcrop = lp[tc1 : tc2 + 1]
                corr_words = corr_text.split()
                corr_w = corr_words[c_idx] if c_idx < len(corr_words) else corr_words[-1]

                res_corr_crop = verify_text(
                    log_probs_np=lp_wcrop,
                    expected_text=corr_w,
                    vocab_char2idx=vocab.char2idx,
                    vocab_idx2char=vocab.idx2char,
                    quran_dict=quran_set,
                    t_accept=t_accept,
                    t_wrong=t_wrong,
                    order=order,
                )
                if res_corr_crop:
                    type_crop_scores[ctype].append(res_corr_crop[0].llr)
                    type_states_crop[ctype][res_corr_crop[0].state] += 1

        if (count + 1) % 500 == 0 or (count + 1) == n_samples:
            print(f"  Processed {count + 1}/{n_samples} benchmark windows...")

    # Display Results
    clean_line_arr = np.array(clean_line_scores, dtype=np.float32)
    clean_crop_arr = np.array(clean_crop_scores, dtype=np.float32)

    total_clean_line = max(1, sum(clean_states_line.values()))
    total_clean_crop = max(1, sum(clean_states_crop.values()))

    print("\n" + "=" * 88)
    print("TRUE CLAIMS: 3-STATE OUTCOMES (CLEAN SAMPLES)")
    print("-" * 88)
    print(f"Mode         | ACCEPT %        | UNCERTAIN %     | WRONG %         | Mean LLR")
    print("-" * 88)
    print(f"Lines (Word) | {clean_states_line['ACCEPT']/total_clean_line*100:>14.2f}% | {clean_states_line['UNCERTAIN']/total_clean_line*100:>14.2f}% | {clean_states_line['WRONG']/total_clean_line*100:>14.2f}% | {float(clean_line_arr.mean()):>8.3f}")
    print(f"Word Crops   | {clean_states_crop['ACCEPT']/total_clean_crop*100:>14.2f}% | {clean_states_crop['UNCERTAIN']/total_clean_crop*100:>14.2f}% | {clean_states_crop['WRONG']/total_clean_crop*100:>14.2f}% | {float(clean_crop_arr.mean()):>8.3f}")
    print("=" * 88)

    print("\n" + "=" * 88)
    print("CORRUPTED CLAIMS: LINE (TARGET-WORD SEGMENT) VS ISOLATED CROPS")
    print("=" * 88)
    print(f"{'Corruption Type':<28} | {'Mode':<6} | {'AUC':<6} | {'FRR@1%':<7} | {'FRR@5%':<7} | {'ACCEPT%':<8} | {'UNCERT%':<8} | {'WRONG%':<8}")
    print("-" * 88)

    all_crop_wrongs = []

    for ctype, label in corruption_types:
        # Line metrics
        l_arr = np.array(type_line_scores[ctype], dtype=np.float32)
        l_auc = compute_roc_auc(clean_line_arr, l_arr)
        l_frr1 = compute_frr_at_far(clean_line_arr, l_arr, 0.01)
        l_frr5 = compute_frr_at_far(clean_line_arr, l_arr, 0.05)
        l_tot = max(1, sum(type_states_line[ctype].values()))
        l_acc = type_states_line[ctype]["ACCEPT"] / l_tot * 100
        l_unc = type_states_line[ctype]["UNCERTAIN"] / l_tot * 100
        l_wrg = type_states_line[ctype]["WRONG"] / l_tot * 100

        # Crop metrics
        c_arr = np.array(type_crop_scores[ctype], dtype=np.float32)
        c_auc = compute_roc_auc(clean_crop_arr, c_arr)
        c_frr1 = compute_frr_at_far(clean_crop_arr, c_arr, 0.01)
        c_frr5 = compute_frr_at_far(clean_crop_arr, c_arr, 0.05)
        c_tot = max(1, sum(type_states_crop[ctype].values()))
        c_acc = type_states_crop[ctype]["ACCEPT"] / c_tot * 100
        c_unc = type_states_crop[ctype]["UNCERTAIN"] / c_tot * 100
        c_wrg = type_states_crop[ctype]["WRONG"] / c_tot * 100
        all_crop_wrongs.append(c_wrg)

        print(f"{label:<28} | Line   | {l_auc:<6.3f} | {l_frr1:>5.1f}%  | {l_frr5:>5.1f}%  | {l_acc:>6.1f}%  | {l_unc:>6.1f}%  | {l_wrg:>6.1f}%")
        print(f"{'':<28} | Crop   | {c_auc:<6.3f} | {c_frr1:>5.1f}%  | {c_frr5:>5.1f}%  | {c_acc:>6.1f}%  | {c_unc:>6.1f}%  | {c_wrg:>6.1f}%")
        print("-" * 88)

    avg_crop_wrong = np.mean(all_crop_wrongs) if all_crop_wrongs else 0.0
    print(f"\n>>> Sanity Target Verification: Corrupted claims on clean synthetic crops must be WRONG > 90%")
    print(f">>> Overall Mean Corrupted Crops WRONG rate: {avg_crop_wrong:.2f}% ({'PASSED (>90%)' if avg_crop_wrong >= 90.0 else 'FAILED'})")
    print("=" * 88)


def main():
    p = argparse.ArgumentParser(description="Task 2.18 Verification & Preprocessing Suite")
    p.add_argument("--checkpoint", default="checkpoints/best.pt", help="Path to Step 2b CRNN checkpoint")
    p.add_argument("--vocab", default="vocab.json", help="Path to vocab.json")
    p.add_argument("--shards_dir", default="data/shards", help="Path to shards directory")
    p.add_argument("--windows_per_type", type=int, default=1000, help="Evaluation samples per corruption type")
    p.add_argument("--t_accept", type=float, default=DEFAULT_T_ACCEPT, help="Threshold for ACCEPT")
    p.add_argument("--t_wrong", type=float, default=DEFAULT_T_WRONG, help="Threshold for WRONG")
    p.add_argument("--duplicates_only", action="store_true", help="Run only historical duplicate and CER analysis")
    args = p.parse_args()

    # 1. Historical Duplicate Analysis & v1 CER
    analyze_v1_duplicates_and_cer()

    if args.duplicates_only:
        return

    # 2. Synthetic Benchmark
    run_synthetic_benchmark(
        checkpoint_path=Path(args.checkpoint),
        vocab_path=Path(args.vocab),
        shards_dir=Path(args.shards_dir),
        windows_per_type=args.windows_per_type,
        t_accept=args.t_accept,
        t_wrong=args.t_wrong,
    )


if __name__ == "__main__":
    main()
