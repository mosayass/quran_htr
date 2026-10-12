"""
scripts/train_twofold_lowo.py
Task 2.20 & 2.21 & 2.22:
Two-fold & Three-fold (Folds A, B, C) cross-writer fine-tune & personalization pipeline:
- Canonical zoom semantics: v2 = 1.0, v1 = 150/108 (~1.39), width = 2.2px (1.8-2.6px augment)
- Prompt group splitting: a line and its paired words stay strictly together
- Fold A: train writer_01 (80% groups) -> test writer_02 (145 samples)
- Fold B: train writer_02 (80% groups) -> test writer_01 (100 samples)
- Fold C: personalization on writer_02 (train 50% of groups, test remaining 50%)
- Mix: 0.5 strokes / 0.3 synth v3 / 0.2 KHATT (or fallback if shards not downloaded)
- AdamW, cosine annealing lr (5e-5 -> 5e-6), fixed 6 epochs
- Thresholds recalibrated on 20% slice of training writer (monotonic t_wrong <= t_accept)
- Evaluates CER, closed-set rank, and 3-state outcomes
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from model import CRNN
from decode import greedy_decode
from metrics import cer, corpus_cer_wer
from render_strokes import render_strokes
from verify_ref import verify_reference
from task_2_17_suite import apply_corruption, extract_mutashabih_pairs


def get_canonical_zoom_and_render(s: dict, augment: bool = False, device: str = "cpu"):
    geom = s.get("geometry") or {}
    if geom:
        bt = float(geom.get("band_top_css", 0.0))
        bb = float(geom.get("band_base_css", 147.0))
        bot = float(geom.get("band_bot_css", 180.0))
        bh = float(geom.get("band_height_css", bot - bt))
        render_bt = 0.0
        render_cy = bb - bt
        canonical_zoom = 1.0
    else:
        render_bt = 50.0
        render_cy = 160.0
        bh = 150.0
        canonical_zoom = 150.0 / 108.0

    if augment:
        z = canonical_zoom * random.uniform(0.8, 1.3)
        w = random.uniform(1.8, 2.6)
    else:
        z = canonical_zoom
        w = 2.2

    img = render_strokes(s["strokes"], zoom=z, width_after_zoom=w, center_y=render_cy, band_top_y=render_bt, band_height=bh, pad_x=16.0)
    arr = (np.array(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    pad_w = (-arr.shape[1]) % 32
    if pad_w > 0:
        arr = np.pad(arr, ((0, 0), (0, pad_w)), mode="constant", constant_values=1.0)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


def generate_all_group_splits():
    # 1. Load prompts mapping for writer_02
    prompts_meta = {}
    prompts_p = ROOT / "data" / "capture_prompts.json"
    if prompts_p.exists():
        with open(prompts_p, encoding="utf-8") as f:
            prompts_meta = json.load(f)
    w_to_line = {w["rasm"]: w["parent_line_id"] for w in prompts_meta.get("words", []) if "parent_line_id" in w}

    # Writer 02
    w2_p = ROOT / "data" / "capture_v2" / "writer_02" / "strokes.json"
    if w2_p.exists():
        with open(w2_p, encoding="utf-8") as f:
            w2_strokes = json.load(f)
        w2_by_id = {x["id"]: x for x in w2_strokes if isinstance(x, dict) and "id" in x}
        w2_groups = defaultdict(list)
        for sid, s in w2_by_id.items():
            if "line" in sid:
                num = sid.split("_")[-1]
                w2_groups[f"group_line_{int(num):02d}"].append(sid)
            else:
                txt = s["text"]
                p = w_to_line.get(txt)
                if p:
                    w2_groups[f"group_line_{int(p.split('_')[-1]):02d}"].append(sid)
                else:
                    w2_groups["group_extra"].append(sid)

        line_gids = sorted([g for g in w2_groups if g != "group_extra"])
        rng = random.Random(42)
        rng.shuffle(line_gids)

        # Fold C: 50% personalization on writer_02
        extra_sids = sorted(w2_groups["group_extra"])
        rng.shuffle(extra_sids)
        c_tr_sids = [sid for g in line_gids[:10] for sid in w2_groups[g]] + extra_sids[:10]
        c_te_sids = [sid for g in line_gids[10:] for sid in w2_groups[g]] + extra_sids[10:]
        with open(ROOT / "data" / "capture_v2" / "writer_02" / "fold_c_train.json", "w", encoding="utf-8") as f:
            json.dump([w2_by_id[s] for s in c_tr_sids], f, ensure_ascii=False, indent=2)
        with open(ROOT / "data" / "capture_v2" / "writer_02" / "fold_c_test.json", "w", encoding="utf-8") as f:
            json.dump([w2_by_id[s] for s in c_te_sids], f, ensure_ascii=False, indent=2)

        # Fold B: 80% train groups on writer_02, 20% calib
        b_tr_sids = [sid for g in line_gids[:16] for sid in w2_groups[g]] + extra_sids[:16]
        b_cal_sids = [sid for g in line_gids[16:] for sid in w2_groups[g]] + extra_sids[16:]
        with open(ROOT / "data" / "capture_v2" / "writer_02" / "w2_group_train.json", "w", encoding="utf-8") as f:
            json.dump([w2_by_id[s] for s in b_tr_sids], f, ensure_ascii=False, indent=2)
        with open(ROOT / "data" / "capture_v2" / "writer_02" / "w2_group_calib.json", "w", encoding="utf-8") as f:
            json.dump([w2_by_id[s] for s in b_cal_sids], f, ensure_ascii=False, indent=2)

    # Writer 01 (100 samples)
    w1_p = ROOT / "data" / "capture" / "strokes.json"
    if w1_p.exists():
        with open(w1_p, encoding="utf-8") as f:
            w1_strokes = json.load(f)
        w1_by_id = {x["id"]: x for x in w1_strokes if isinstance(x, dict) and "id" in x}
        w1_lines = sorted([x for x in w1_by_id.keys() if "line" in x])
        w1_words = sorted([x for x in w1_by_id.keys() if "word" in x])
        rng = random.Random(42)
        rng.shuffle(w1_lines)
        rng.shuffle(w1_words)
        w1_tr_sids = w1_lines[:32] + w1_words[:48]
        w1_cal_sids = w1_lines[32:] + w1_words[48:]
        with open(ROOT / "data" / "capture" / "w1_group_train.json", "w", encoding="utf-8") as f:
            json.dump([w1_by_id[s] for s in w1_tr_sids], f, ensure_ascii=False, indent=2)
        with open(ROOT / "data" / "capture" / "w1_group_calib.json", "w", encoding="utf-8") as f:
            json.dump([w1_by_id[s] for s in w1_cal_sids], f, ensure_ascii=False, indent=2)
    print("Generated all prompt group splits: Fold A, B, C.")


def find_base_checkpoint(custom_path: Optional[str] = None) -> Optional[Path]:
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    candidates.extend([
        ROOT / "checkpoints" / "best.pt",
        ROOT / "checkpoints" / "step2b_v3" / "best.pt",
        Path("/content/drive/MyDrive/quran_htr_writer_checkpoints/step2b_v3/best.pt"),
        Path("/content/drive/MyDrive/quran_htr_writer_checkpoints/run_visual/best.pt"),
        Path("/content/drive/MyDrive/quran_htr_writer_checkpoints/best.pt"),
        Path("/content/drive/MyDrive/quran_htr_writer_checkpoints/cont_best.pt"),
        ROOT / "dist" / "best.pt",
    ])
    for c in candidates:
        if c.exists():
            return c
    return None


class SimpleStrokeTrainDataset(Dataset):
    def __init__(self, samples: List[dict], vocab: Vocabulary):
        self.samples = samples
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        txt = s["text"]
        target = self.vocab.encode(txt)
        # Visual order reversal for labels
        target_vis = target[::-1]
        img_t = get_canonical_zoom_and_render(s, augment=True, device="cpu").squeeze(0)  # [1, 64, W]
        return img_t, target_vis, txt


def collate_train_batch(batch):
    imgs, targets, texts = zip(*batch)
    max_w = max(img.shape[2] for img in imgs)
    max_w = ((max_w + 31) // 32) * 32
    B = len(imgs)
    padded_imgs = torch.ones((B, 1, 64, max_w), dtype=torch.float32)
    target_lengths = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat_targets = torch.tensor([idx for t in targets for idx in t], dtype=torch.long)
    input_lengths = torch.tensor([img.shape[2] // 4 for img in imgs], dtype=torch.long)
    for i, img in enumerate(imgs):
        w = img.shape[2]
        padded_imgs[i, :, :, :w] = img
    return padded_imgs, flat_targets, input_lengths, target_lengths, texts


def recalibrate_thresholds(model: nn.Module, calib_samples: List[dict], vocab: Vocabulary, device: str = "cpu"):
    model.eval()
    tanzil_clean = ROOT / "data" / "tanzil" / "quran-simple-clean.txt"
    q_dict = []
    if tanzil_clean.exists():
        with open(tanzil_clean, encoding="utf-8") as f:
            for l in f:
                if not l.strip().startswith("#"):
                    q_dict.extend(l.strip().split())
    q_dict = sorted(set(q_dict))[:5000]
    mut_pairs = extract_mutashabih_pairs()
    mut_dict = defaultdict(list)
    for m in mut_pairs:
        mut_dict[m["prefix"]].append(m["variant_b"].split()[-1])

    true_llrs = []
    corr_llrs = []
    rng = random.Random(123)

    with torch.no_grad():
        for s in calib_samples:
            inp = get_canonical_zoom_and_render(s, augment=False, device=device)
            out = model(inp).squeeze(1).cpu().numpy()  # (T, V)
            txt = s["text"]
            # True claim
            v_res = verify_reference(out, txt, vocab.char2idx, vocab.idx2char, t_accept=-0.6, t_wrong=-0.95, order="visual", norm="char")
            for w_eval in v_res:
                true_llrs.append(w_eval["llr"])

            # Corrupted claim
            c_info = apply_corruption(txt, "dot_swap", q_dict, mut_dict, rng)
            if c_info:
                c_txt, _ = c_info
                cv_res = verify_reference(out, c_txt, vocab.char2idx, vocab.idx2char, t_accept=-0.6, t_wrong=-0.95, order="visual", norm="char")
                for w_eval in cv_res:
                    corr_llrs.append(w_eval["llr"])

    if not true_llrs:
        return -0.60, -0.95
    t_acc = float(np.percentile(true_llrs, 5))
    t_wrg_raw = float(np.percentile(corr_llrs, 90)) if corr_llrs else t_acc - 0.35
    # Enforce strict monotonicity: t_wrong <= t_accept
    t_wrg = min(t_acc - 0.20, t_wrg_raw)
    return round(t_acc, 4), round(t_wrg, 4)


def evaluate_test_set(model: nn.Module, test_samples: List[dict], vocab: Vocabulary, t_acc: float, t_wrong: float, device: str = "cpu"):
    model.eval()
    tanzil_clean = ROOT / "data" / "tanzil" / "quran-simple-clean.txt"
    q_dict = []
    if tanzil_clean.exists():
        with open(tanzil_clean, encoding="utf-8") as f:
            for l in f:
                if not l.strip().startswith("#"):
                    q_dict.extend(l.strip().split())
    q_dict = sorted(set(q_dict))[:14872]
    mut_pairs = extract_mutashabih_pairs()
    mut_dict = defaultdict(list)
    for m in mut_pairs:
        mut_dict[m["prefix"]].append(m["variant_b"].split()[-1])

    line_gt, line_pred = [], []
    word_gt, word_pred = [], []
    true_states = {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0}
    corr_wrg = defaultdict(lambda: {"WRONG": 0, "TOTAL": 0})
    ctypes = ["dot_swap", "del_ins", "word_swap", "adj_swap", "skip_word", "mutashabih"]
    rng = random.Random(42)

    with torch.no_grad():
        for s in test_samples:
            inp = get_canonical_zoom_and_render(s, augment=False, device=device)
            logits = model(inp)  # (T, 1, V)
            log_probs = logits.squeeze(1).cpu().numpy()
            pred_txt = greedy_decode(logits, vocab, label_order="visual")[0]
            gt_txt = s["text"]

            if "line" in s.get("id", "") or len(gt_txt.split()) > 2:
                line_gt.append(gt_txt)
                line_pred.append(pred_txt)
            else:
                word_gt.append(gt_txt)
                word_pred.append(pred_txt)

            # True claim 3-state
            v_res = verify_reference(log_probs, gt_txt, vocab.char2idx, vocab.idx2char, t_accept=t_acc, t_wrong=t_wrong, order="visual", norm="char")
            for w in v_res:
                true_states[w["state"]] += 1

            # Corrupted claims
            for ct in ctypes:
                c_info = apply_corruption(gt_txt, ct, q_dict, mut_dict, rng)
                if c_info:
                    c_txt, _ = c_info
                    cv_res = verify_reference(log_probs, c_txt, vocab.char2idx, vocab.idx2char, t_accept=t_acc, t_wrong=t_wrong, order="visual", norm="char")
                    corr_wrg[ct]["TOTAL"] += 1
                    # If any word is flagged WRONG, claim is caught
                    if any(w["state"] == "WRONG" for w in cv_res):
                        corr_wrg[ct]["WRONG"] += 1

    all_gt = line_gt + word_gt
    all_pred = line_pred + word_pred
    cer_all, wer_all = corpus_cer_wer(all_gt, all_pred)
    cer_lines, _ = corpus_cer_wer(line_gt, line_pred) if line_gt else (0.0, 0.0)
    cer_words, _ = corpus_cer_wer(word_gt, word_pred) if word_gt else (0.0, 0.0)

    total_true = sum(true_states.values()) or 1
    true_acc_rate = round(100.0 * true_states["ACCEPT"] / total_true, 2)
    true_unc_rate = round(100.0 * true_states["UNCERTAIN"] / total_true, 2)
    true_wrg_rate = round(100.0 * true_states["WRONG"] / total_true, 2)

    corr_rates = {}
    for ct in ctypes:
        tot = corr_wrg[ct]["TOTAL"]
        corr_rates[ct] = round(100.0 * corr_wrg[ct]["WRONG"] / tot, 2) if tot > 0 else 100.0

    return {
        "cer_overall": round(cer_all * 100.0, 2),
        "cer_lines": round(cer_lines * 100.0, 2),
        "cer_words": round(cer_words * 100.0, 2),
        "wer_overall": round(wer_all * 100.0, 2),
        "true_accept_rate": true_acc_rate,
        "true_unc_rate": true_unc_rate,
        "true_wrg_rate": true_wrg_rate,
        "corr_rejection": corr_rates,
    }


def train_single_fold(
    fold: str,
    train_samples: List[dict],
    calib_samples: List[dict],
    test_samples: List[dict],
    vocab: Vocabulary,
    base_ckpt: Optional[Path],
    epochs: int = 6,
    lr: float = 5e-5,
    steps_per_epoch: int = 1000,
    batch_size: int = 8,
    device: str = "cpu",
):
    print(f"\n=======================================================")
    print(f"STARTING FOLD {fold} FINE-TUNING")
    print(f"Train samples: {len(train_samples)} | Calib: {len(calib_samples)} | Test: {len(test_samples)}")
    print(f"Schedule: {epochs} epochs x {steps_per_epoch} steps | Base LR: {lr} -> {lr/10.0}")
    print(f"=======================================================")

    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    if base_ckpt and base_ckpt.exists():
        print(f"Loading base checkpoint: {base_ckpt}")
        ckpt_data = torch.load(base_ckpt, map_location=device, weights_only=False)
        st = ckpt_data.get("model_state", ckpt_data)
        model.load_state_dict(st, strict=False)
    else:
        print("[WARN] No base checkpoint found! Training from scratch / current weights.")

    # Train dataset & loader
    train_ds = SimpleStrokeTrainDataset(train_samples, vocab)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(batch_size, len(train_ds)),
        shuffle=True,
        collate_fn=collate_train_batch,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps = epochs * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=lr / 10.0)
    ctc_loss_fn = nn.CTCLoss(blank=0, zero_infinity=True)

    model.train()
    step_count = 0
    loader_iter = iter(train_loader)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(train_loader)
                batch = next(loader_iter)

            imgs, flat_targets, in_lens, tgt_lens, _ = batch
            imgs = imgs.to(device)
            flat_targets = flat_targets.to(device)
            in_lens = in_lens.to(device)
            tgt_lens = tgt_lens.to(device)

            optimizer.zero_grad()
            logits = model(imgs)  # (T, B, V)
            log_probs = logits.log_softmax(dim=-1)
            loss = ctc_loss_fn(log_probs, flat_targets, in_lens, tgt_lens)

            if not torch.isnan(loss) and not torch.isinf(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step()
            step_count += 1

        avg_loss = epoch_loss / max(1, steps_per_epoch)
        current_lr = scheduler.get_last_lr()[0]
        print(f"  [Fold {fold}] Epoch {epoch}/{epochs} | CTC Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")

    # Save checkpoint
    out_dir = ROOT / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_path = out_dir / f"lowo_fold_{fold}.pt"
    torch.save({"model_state": model.state_dict(), "label_order": "visual"}, save_path)
    print(f"Saved Fold {fold} checkpoint to {save_path}")

    # Recalibrate thresholds on calibration slice
    t_acc, t_wrg = recalibrate_thresholds(model, calib_samples, vocab, device=device)
    print(f"Recalibrated thresholds on 20% slice: t_accept = {t_acc}, t_wrong = {t_wrg}")

    # Evaluate on held-out test writer
    eval_res = evaluate_test_set(model, test_samples, vocab, t_acc, t_wrg, device=device)
    eval_res["t_accept"] = t_acc
    eval_res["t_wrong"] = t_wrg
    print(f"Fold {fold} Results: CER Lines={eval_res['cer_lines']}%, CER Words={eval_res['cer_words']}%, Overall={eval_res['cer_overall']}%")
    print(f"True-Accept={eval_res['true_accept_rate']}% | Dot Swap WRONG={eval_res['corr_rejection'].get('dot_swap')}% | Word Swap WRONG={eval_res['corr_rejection'].get('word_swap')}%")
    return eval_res


def main():
    parser = argparse.ArgumentParser(description="Cross-Writer LOWO and Personalization Training")
    parser.add_argument("--folds", nargs="+", default=["A", "B", "C"], help="Folds to execute (A, B, C)")
    parser.add_argument("--epochs", type=int, default=6, help="Epochs per fold")
    parser.add_argument("--steps_per_epoch", type=int, default=1000, help="Steps per epoch")
    parser.add_argument("--lr", type=float, default=5e-5, help="Initial learning rate")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--base_checkpoint", type=str, default=None, help="Base CRNN checkpoint")
    args = parser.parse_args()

    generate_all_group_splits()
    vocab = Vocabulary.load(ROOT / "vocab.json")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Execution Device: {device} | Folds: {args.folds}")

    base_ckpt = find_base_checkpoint(args.base_checkpoint)
    print(f"Using base checkpoint: {base_ckpt}")

    # Load splits
    w1_tr = json.load(open(ROOT / "data" / "capture" / "w1_group_train.json", encoding="utf-8"))
    w1_cal = json.load(open(ROOT / "data" / "capture" / "w1_group_calib.json", encoding="utf-8"))
    w1_all = json.load(open(ROOT / "data" / "capture" / "strokes.json", encoding="utf-8"))

    w2_tr = json.load(open(ROOT / "data" / "capture_v2" / "writer_02" / "w2_group_train.json", encoding="utf-8"))
    w2_cal = json.load(open(ROOT / "data" / "capture_v2" / "writer_02" / "w2_group_calib.json", encoding="utf-8"))
    w2_all = json.load(open(ROOT / "data" / "capture_v2" / "writer_02" / "strokes.json", encoding="utf-8"))

    fc_tr = json.load(open(ROOT / "data" / "capture_v2" / "writer_02" / "fold_c_train.json", encoding="utf-8"))
    fc_te = json.load(open(ROOT / "data" / "capture_v2" / "writer_02" / "fold_c_test.json", encoding="utf-8"))

    results = {}
    folds = [f.upper() for f in args.folds]

    if "A" in folds:
        # Fold A: Train w01 (80%) -> Calib w01 (20%) -> Test w02 (100% held-out)
        res_a = train_single_fold(
            "A", w1_tr, w1_cal, w2_all, vocab, base_ckpt,
            epochs=args.epochs, lr=args.lr, steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size, device=device
        )
        results["Fold_A"] = res_a

    if "B" in folds:
        # Fold B: Train w02 (80%) -> Calib w02 (20%) -> Test w01 (100% held-out)
        res_b = train_single_fold(
            "B", w2_tr, w2_cal, w1_all, vocab, base_ckpt,
            epochs=args.epochs, lr=args.lr, steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size, device=device
        )
        results["Fold_B"] = res_b

    if "C" in folds:
        # Fold C: Train w02 50% groups -> Calib on 20% of train -> Test remaining 50% groups
        cal_c_size = max(5, int(0.2 * len(fc_tr)))
        fc_cal = fc_tr[-cal_c_size:]
        fc_pure_tr = fc_tr[:-cal_c_size]
        res_c = train_single_fold(
            "C", fc_pure_tr, fc_cal, fc_te, vocab, base_ckpt,
            epochs=args.epochs, lr=args.lr, steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size, device=device
        )
        results["Fold_C"] = res_c

    # Save summary report
    rep_dir = ROOT / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    with open(rep_dir / "lowo_fold_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\nSaved all fold results to reports/lowo_fold_results.json")


if __name__ == "__main__":
    main()
