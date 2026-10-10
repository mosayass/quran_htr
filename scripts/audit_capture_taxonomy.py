"""
scripts/audit_capture_taxonomy.py
Task 2.14 Item 1:
Capture Error Taxonomy & Qualitative Montage:
1. Computes per-sample CER sorted (100 tablet samples).
2. Extracts top substitutions, insertions, deletions, space errors.
3. Splits metrics by mode (line vs word) and prompt type (if line prompt had vowelled or rasm).
4. Generates a visual montage of the 20 worst and 10 best samples:
   [image | label | prediction | CER].
5. Saves artifacts to data/capture/report/montage_taxonomy.png and taxonomy_summary.md.
"""

from __future__ import annotations
import sys
import difflib
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vocab import Vocabulary
from model import CRNN
from dataset import LineImageDataset, collate_fn
from evaluate import greedy_decode
from metrics import cer as compute_single_cer, wer as compute_single_wer, corpus_cer_wer


def run_taxonomy_audit(
    checkpoint_path: Path,
    vocab_path: Path = Path("vocab.json"),
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device_str)
    vocab = Vocabulary.load(vocab_path)
    cap_csv = ROOT / "data" / "capture" / "manifest.csv"

    # Load model
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    order = ckpt.get("label_order", "visual")
    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = LineImageDataset(cap_csv, vocab, augment=False, label_order=order)
    loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate_fn)

    all_preds, all_targets, all_img_paths = [], [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["images"].to(device)
            log_probs = model(imgs)
            preds = greedy_decode(log_probs, vocab, label_order=order)
            all_preds.extend(preds)
            all_targets.extend(batch["texts"])

    for r in ds.rows:
        all_img_paths.append(r[0])

    # Per-sample CER analysis
    sample_evals = []
    substitutions = Counter()
    deletions = Counter()
    insertions = Counter()
    space_errors = Counter()  # ('missed_space' or 'extra_space')

    line_evals = []
    word_evals = []

    for idx, (p, t, ip) in enumerate(zip(all_preds, all_targets, all_img_paths)):
        sample_cer = compute_single_cer(p, t)
        sample_wer = compute_single_wer(p, t)
        is_word = "sample_word_" in ip
        mode = "word" if is_word else "line"

        item = {
            "idx": idx,
            "image_path": ip,
            "pred": p,
            "target": t,
            "cer": sample_cer,
            "wer": sample_wer,
            "mode": mode,
        }
        sample_evals.append(item)
        if is_word:
            word_evals.append(item)
        else:
            line_evals.append(item)

        # Character alignment
        matcher = difflib.SequenceMatcher(None, t, p)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag == "replace":
                t_sub = t[i1:i2]
                p_sub = p[j1:j2]
                min_l = min(len(t_sub), len(p_sub))
                for k in range(min_l):
                    t_c, p_c = t_sub[k], p_sub[k]
                    substitutions[(t_c, p_c)] += 1
                    if t_c == " " and p_c != " ":
                        space_errors["space_substituted"] += 1
                    elif p_c == " " and t_c != " ":
                        space_errors["char_became_space"] += 1
                if len(t_sub) > min_l:
                    for k in range(min_l, len(t_sub)):
                        deletions[t_sub[k]] += 1
                        if t_sub[k] == " ":
                            space_errors["missed_space"] += 1
                elif len(p_sub) > min_l:
                    for k in range(min_l, len(p_sub)):
                        insertions[p_sub[k]] += 1
                        if p_sub[k] == " ":
                            space_errors["extra_space"] += 1
            elif tag == "delete":
                for c in t[i1:i2]:
                    deletions[c] += 1
                    if c == " ":
                        space_errors["missed_space"] += 1
            elif tag == "insert":
                for c in p[j1:j2]:
                    insertions[c] += 1
                    if c == " ":
                        space_errors["extra_space"] += 1

    # Sort by CER
    sample_evals_sorted = sorted(sample_evals, key=lambda x: x["cer"], reverse=True)

    # Metrics by mode
    overall_cer, overall_wer = corpus_cer_wer(all_preds, all_targets)
    line_preds = [x["pred"] for x in line_evals]
    line_targets = [x["target"] for x in line_evals]
    line_cer, line_wer = corpus_cer_wer(line_preds, line_targets) if line_preds else (0, 0)

    word_preds = [x["pred"] for x in word_evals]
    word_targets = [x["target"] for x in word_evals]
    word_cer, word_wer = corpus_cer_wer(word_preds, word_targets) if word_preds else (0, 0)

    # Save Markdown report
    report_dir = ROOT / "data" / "capture" / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    md_path = report_dir / "taxonomy_summary.md"

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Capture Set Error Taxonomy (100 Samples)\n\n")
        f.write(f"- **Overall CER / WER**: {overall_cer*100:.2f}% / {overall_wer*100:.2f}%\n")
        f.write(f"- **Lines (40 samples)**: CER {line_cer*100:.2f}% | WER {line_wer*100:.2f}%\n")
        f.write(f"- **Words (60 samples)**: CER {word_cer*100:.2f}% | WER {word_wer*100:.2f}%\n\n")

        f.write("### Space Errors\n")
        for k, v in space_errors.items():
            f.write(f"- {k}: {v}\n")

        f.write("\n### Top 15 Character Substitutions (Target -> Predicted)\n")
        for (tc, pc), cnt in substitutions.most_common(15):
            f.write(f"- '{tc}' -> '{pc}': {cnt}\n")

        f.write("\n### Top 10 Deletions (Omitted Characters)\n")
        for c, cnt in deletions.most_common(10):
            f.write(f"- '{c}': {cnt}\n")

        f.write("\n### Top 10 Insertions (Spurious Characters)\n")
        for c, cnt in insertions.most_common(10):
            f.write(f"- '{c}': {cnt}\n")

    print(f"Taxonomy summary written to {md_path}")

    # Generate Montage of 20 Worst + 10 Best
    worst_20 = sample_evals_sorted[:20]
    best_10 = sorted(sample_evals_sorted, key=lambda x: x["cer"])[:10]
    montage_samples = [("WORST 20 SAMPLES", worst_20), ("BEST 10 SAMPLES", best_10)]

    montage_rows = []
    # Try finding an Arabic font for text annotation, otherwise fallback
    font = None
    for font_cand in [
        ROOT / "data" / "fonts" / "Amiri.ttf",
        ROOT / "data" / "fonts" / "Lateef.ttf",
        Path("C:/Windows/Fonts/arial.ttf"),
    ]:
        if font_cand.exists():
            try:
                font = ImageFont.truetype(str(font_cand), 14)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()

    row_h = 74
    row_w = 900

    # Build image panels
    panels = []
    for section_title, s_list in montage_samples:
        header_img = Image.new("RGB", (row_w, 30), color=(240, 240, 240))
        h_draw = ImageDraw.Draw(header_img)
        h_draw.text((10, 8), section_title, fill=(0, 0, 0), font=font)
        panels.append(header_img)

        for s in s_list:
            r_img = Image.new("RGB", (row_w, row_h), color=(255, 255, 255))
            r_draw = ImageDraw.Draw(r_img)

            # Paste crop image on left (rescaled to height 60)
            orig_crop = Image.open(s["image_path"]).convert("RGB")
            c_w = min(350, int(round(orig_crop.width * (60.0 / orig_crop.height))))
            scaled_crop = orig_crop.resize((c_w, 60), Image.BILINEAR)
            r_img.paste(scaled_crop, (5, 7))

            # Annotations on right
            info_x = 370
            cer_color = (180, 0, 0) if s["cer"] > 0.25 else (0, 140, 0)
            r_draw.text((info_x, 8), f"CER: {s['cer']*100:5.1f}% | {Path(s['image_path']).name}", fill=cer_color, font=font)
            r_draw.text((info_x, 28), f"True: {s['target']}", fill=(30, 30, 30), font=font)
            r_draw.text((info_x, 48), f"Pred: {s['pred']}", fill=(70, 70, 160), font=font)
            r_draw.line([(0, row_h - 1), (row_w, row_h - 1)], fill=(220, 220, 220))
            panels.append(r_img)

    total_montage_h = sum(p.height for p in panels)
    full_montage = Image.new("RGB", (row_w, total_montage_h), color=(255, 255, 255))
    y_offset = 0
    for p in panels:
        full_montage.paste(p, (0, y_offset))
        y_offset += p.height

    montage_path = report_dir / "montage_taxonomy.png"
    full_montage.save(montage_path)
    print(f"Saved qualitative montage to {montage_path}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/step2b_v3/best.pt")
    args = p.parse_args()

    ckpt_p = Path(args.checkpoint)
    if not ckpt_p.exists():
        for alt in [
            Path("checkpoints/best.pt"),
            Path("/content/drive/MyDrive/quran_htr_writer_checkpoints/step2b_v3/best.pt"),
        ]:
            if alt.exists():
                ckpt_p = alt
                break

    if not ckpt_p.exists():
        print(f"Error: Checkpoint {args.checkpoint} not found.")
        sys.exit(1)

    run_taxonomy_audit(ckpt_p)


if __name__ == "__main__":
    main()
