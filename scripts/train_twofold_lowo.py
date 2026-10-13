"""
scripts/train_twofold_lowo.py
Cross-writer fine-tune / personalization pipeline (dev writers only: writer_01, writer_02).

Task 2.25 version:
- Folds: A (w01 -> w02), B (w02 -> w01), C (personalization on w02, 50% groups),
  CLC (Fold C learning curve: n in {10,20,40,75} train samples, tested on w02's held-out groups).
- Data mixes (per-SAMPLE multinomial draws, strokes/synth/KHATT):
    stroke_only = 1.0/0.0/0.0   mix_a = 0.5/0.3/0.2   mix_b = 0.3/0.4/0.3
  Each epoch asserts nonzero synth AND khatt sample counts when the mix requires them.
- Step-0 gate: the loaded model is evaluated on the fold's test subset BEFORE any optimizer step and
  must match the INDEPENDENTLY measured same-device baseline in configs/gate_baselines.json within
  1.0 pt (measure with scripts/measure_gate_baselines.py). Missing baseline => abort.
- Weight loading asserts every state_dict entry matched ("init_weights matched N/M tensors").
- Seeds: torch/numpy/random + DataLoader worker streams; results append to a JSONL file and runs
  already present are skipped (Colab disconnect-safe).
- Post-fine-tune forgetting evals: KHATT val CER and synthetic test CER (same loader as
  evaluate_continuation.py).
- Locked-test writers are never loadable: every path/id is checked by assert_no_locked_test().
"""

from __future__ import annotations
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, IterableDataset, Dataset, Subset, get_worker_info

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

GATE_TOL = 1.0  # percentage points, same device
GATE_BASELINE_PATH = ROOT / "configs" / "gate_baselines.json"
SOURCES = ("strokes", "synth", "khatt")
MIXES: Dict[str, Tuple[float, float, float]] = {
    "stroke_only": (1.0, 0.0, 0.0),
    "mix_a": (0.5, 0.3, 0.2),
    "mix_b": (0.3, 0.4, 0.3),
}
LC_SIZES = (10, 20, 40, 75)


# ----------------------------------------------------------------------------------------------
# Safety: locked-test refusal
# ----------------------------------------------------------------------------------------------
def assert_no_locked_test(path_or_str: Any):
    p = str(path_or_str).replace("\\", "/").lower()
    assert "locked_test" not in p, (
        f"SECURITY VIOLATION: Tuning script attempted to access locked-test path: {path_or_str}. "
        "Locked test writers are strictly reserved for final frozen evaluation and must never be touched during training/CV!"
    )


# ----------------------------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------------------------
def get_canonical_zoom_and_render(s: dict, augment: bool = False, device: str = "cpu",
                                  zoom_override: Optional[float] = None):
    """Renders one stroke sample to a (1,1,64,W) tensor. zoom_override (label-free scale policy)
    replaces the canonical zoom when given (only for non-augmented evaluation)."""
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
        z = canonical_zoom if zoom_override is None else float(zoom_override)
        w = 2.2

    img = render_strokes(s["strokes"], zoom=z, width_after_zoom=w, center_y=render_cy, band_top_y=render_bt, band_height=bh, pad_x=16.0)
    arr = (np.array(img, dtype=np.float32) / 255.0 - 0.5) / 0.5
    pad_w = (-arr.shape[1]) % 32
    if pad_w > 0:
        arr = np.pad(arr, ((0, 0), (0, pad_w)), mode="constant", constant_values=1.0)
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)


# ----------------------------------------------------------------------------------------------
# Splits (prompt-group based; a line and its paired words stay together)
# ----------------------------------------------------------------------------------------------
def _dedup(samples: Sequence[dict]) -> List[dict]:
    return list({x["id"]: x for x in samples if isinstance(x, dict) and "id" in x}.values())


def build_w2_groups() -> Tuple[Dict[str, dict], Dict[str, List[str]]]:
    prompts_p = ROOT / "data" / "capture_prompts.json"
    prompts_meta = json.load(open(prompts_p, encoding="utf-8")) if prompts_p.exists() else {}
    w_to_line = {w["rasm"]: w["parent_line_id"] for w in prompts_meta.get("words", []) if "parent_line_id" in w}
    w2_p = ROOT / "data" / "capture_v2" / "writer_02" / "strokes.json"
    w2_by_id = {x["id"]: x for x in json.load(open(w2_p, encoding="utf-8")) if isinstance(x, dict) and "id" in x}
    groups: Dict[str, List[str]] = defaultdict(list)
    for sid, s in w2_by_id.items():
        if "line" in sid:
            groups[f"group_line_{int(sid.split('_')[-1]):02d}"].append(sid)
        else:
            p = w_to_line.get(s["text"])
            if p:
                groups[f"group_line_{int(p.split('_')[-1]):02d}"].append(sid)
            else:
                groups["group_extra"].append(sid)
    return w2_by_id, groups


def generate_all_group_splits():
    w2_p = ROOT / "data" / "capture_v2" / "writer_02" / "strokes.json"
    if w2_p.exists():
        w2_by_id, w2_groups = build_w2_groups()
        line_gids = sorted([g for g in w2_groups if g != "group_extra"])
        train_gids = [g for g in line_gids if int(g.split("_")[-1]) <= 14]
        held_out_gids = [g for g in line_gids if int(g.split("_")[-1]) > 14]
        assert len(held_out_gids) == 6, f"Expected 6 held-out groups (~30%), got {len(held_out_gids)}"
        extra_sids = sorted(w2_groups["group_extra"])

        b_tr_sids = [sid for g in train_gids[:12] for sid in w2_groups[g]] + extra_sids[:14]
        b_cal_sids = [sid for g in train_gids[12:] for sid in w2_groups[g]] + extra_sids[14:]
        held_sids = [sid for g in held_out_gids for sid in w2_groups[g]]
        c_tr_sids = [sid for g in line_gids[:10] for sid in w2_groups[g]] + extra_sids[:10]
        c_te_sids = [sid for g in line_gids[10:] for sid in w2_groups[g]] + extra_sids[10:]
        assert not (set(b_tr_sids + b_cal_sids) & set(held_sids)), "held-out groups leaked into train pool"

        d = ROOT / "data" / "capture_v2" / "writer_02"
        for name, sids in (("w2_group_train.json", b_tr_sids), ("w2_group_calib.json", b_cal_sids),
                           ("w2_group_heldout.json", held_sids),
                           ("fold_c_train.json", c_tr_sids), ("fold_c_test.json", c_te_sids)):
            with open(d / name, "w", encoding="utf-8") as f:
                json.dump([w2_by_id[s] for s in sids], f, ensure_ascii=False, indent=2)

    w1_p = ROOT / "data" / "capture" / "strokes.json"
    if w1_p.exists():
        w1_by_id = {x["id"]: x for x in json.load(open(w1_p, encoding="utf-8")) if isinstance(x, dict) and "id" in x}
        w1_lines = sorted([x for x in w1_by_id if "line" in x])
        w1_words = sorted([x for x in w1_by_id if "word" in x])
        rng = random.Random(42)
        rng.shuffle(w1_lines)
        rng.shuffle(w1_words)
        w1_tr_sids = w1_lines[:28] + w1_words[:42]
        w1_cal_sids = w1_lines[28:] + w1_words[42:]
        with open(ROOT / "data" / "capture" / "w1_group_train.json", "w", encoding="utf-8") as f:
            json.dump([w1_by_id[s] for s in w1_tr_sids], f, ensure_ascii=False, indent=2)
        with open(ROOT / "data" / "capture" / "w1_group_calib.json", "w", encoding="utf-8") as f:
            json.dump([w1_by_id[s] for s in w1_cal_sids], f, ensure_ascii=False, indent=2)
    print("Generated all prompt group splits: Fold A, B, C (+ w2 held-out groups for the learning curve).")


def load_dev_splits() -> Dict[str, List[dict]]:
    c = ROOT / "data" / "capture"
    c2 = ROOT / "data" / "capture_v2" / "writer_02"
    j = lambda p: json.load(open(p, encoding="utf-8"))
    s = {
        "w1_tr": j(c / "w1_group_train.json"), "w1_cal": j(c / "w1_group_calib.json"),
        "w1_all": _dedup(j(c / "strokes.json")),
        "w2_tr": j(c2 / "w2_group_train.json"), "w2_cal": j(c2 / "w2_group_calib.json"),
        "w2_all": _dedup(j(c2 / "strokes.json")), "w2_held": j(c2 / "w2_group_heldout.json"),
        "fc_tr": j(c2 / "fold_c_train.json"), "fc_te": j(c2 / "fold_c_test.json"),
    }
    cal_c = max(5, int(0.2 * len(s["fc_tr"])))
    s["fc_cal"], s["fc_pure_tr"] = s["fc_tr"][-cal_c:], s["fc_tr"][:-cal_c]
    return s


def gate_subsets(sp: Dict[str, List[dict]]) -> Dict[str, List[dict]]:
    """Test subsets that the Step-0 gate / baseline measurement use. Single source of truth."""
    return {"A_test": sp["w2_all"], "B_test": sp["w1_all"], "C_test": sp["fc_te"], "LC_test": sp["w2_held"]}


def lc_order(w2_pool_ids_by_group: Dict[str, List[str]], seed: int) -> List[str]:
    """Group-respecting, seed-dependent order; nested prefixes give the learning-curve training sets."""
    gids = sorted(w2_pool_ids_by_group)
    random.Random(seed).shuffle(gids)
    return [sid for g in gids for sid in w2_pool_ids_by_group[g]]


def find_base_checkpoint(custom_path: Optional[str] = None) -> Optional[Path]:
    candidates = []
    if custom_path:
        candidates.append(Path(custom_path))
    candidates.extend([
        ROOT / "checkpoints" / "best.pt",
        ROOT / "checkpoints" / "step2b_v3" / "best.pt",
        Path("/content/drive/MyDrive/quran_htr_writer_checkpoints/step2b_v3/best.pt"),
    ])
    for c in candidates:
        if c.exists():
            return c
    return None


def load_base_model(base_ckpt: Path, vocab: Vocabulary, device: str) -> Tuple[nn.Module, str]:
    """Loads weights STRICTLY and logs 'init_weights matched N/M tensors' (N must equal M)."""
    assert_no_locked_test(str(base_ckpt))
    model = CRNN(vocab_size=len(vocab), backbone="vgg_lite").to(device)
    ckpt = torch.load(base_ckpt, map_location=device, weights_only=False)
    st = ckpt["model"] if "model" in ckpt else ckpt.get("model_state", ckpt)
    ms = model.state_dict()
    matched = [k for k in st if k in ms and st[k].shape == ms[k].shape]
    msg = f"init_weights matched {len(matched)}/{len(ms)} tensors"
    print(msg)
    assert len(matched) == len(ms), f"Weight matching failure: {msg}; missing {set(ms) - set(matched)}"
    model.load_state_dict(st, strict=True)
    return model, msg


# ----------------------------------------------------------------------------------------------
# Canonical eval, per-word error attribution, lexical stratification
# ----------------------------------------------------------------------------------------------
def predict_records(model: nn.Module, samples: List[dict], vocab: Vocabulary, device: str = "cpu",
                    zoom_fn=None) -> List[dict]:
    model.eval()
    recs = []
    with torch.no_grad():
        for s in samples:
            zo = zoom_fn(s) if zoom_fn else None
            inp = get_canonical_zoom_and_render(s, augment=False, device=device, zoom_override=zo)
            lp = model(inp)
            recs.append({"id": s["id"], "text": s["text"], "pred": greedy_decode(lp, vocab, label_order="visual")[0]})
    return recs


def records_cer(records: List[dict]) -> float:
    return 100.0 * corpus_cer_wer([r["pred"] for r in records], [r["text"] for r in records])[0]


def canonical_cer(model: nn.Module, samples: List[dict], vocab: Vocabulary, device: str = "cpu") -> float:
    """Canonical eval (zoom policy of docs/inference_contract, greedy decode, corpus CER, %)."""
    return records_cer(predict_records(model, samples, vocab, device))


def per_word_errors(ref: str, hyp: str) -> List[Tuple[str, int, int]]:
    """Character edit errors attributed to REFERENCE words. Sub/del -> that ref char; insertion -> the
    next ref char (clamped); space errors -> the word before the space. Sum of errors == edit distance."""
    n, m = len(ref), len(hyp)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]))
    err = [0] * max(n, 1)
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (ref[i - 1] != hyp[j - 1]):
            if ref[i - 1] != hyp[j - 1]:
                err[i - 1] += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            err[i - 1] += 1
            i -= 1
        else:
            err[min(i, n - 1) if n > 0 else 0] += 1
            j -= 1
    out, pos = [], 0
    for w in ref.split(" "):
        end = pos + len(w)
        e = sum(err[pos:end]) + (err[end] if end < n else 0)
        out.append((w, e, len(w)))
        pos = end + 1
    return out


def stratify_cer(records: List[dict], seen_words: set) -> Dict[str, Dict[str, Any]]:
    agg = {"seen": [0, 0, 0], "unseen": [0, 0, 0]}  # errors, chars, words
    for r in records:
        for w, e, l in per_word_errors(r["text"], r["pred"]):
            k = "seen" if w in seen_words else "unseen"
            agg[k][0] += e
            agg[k][1] += l
            agg[k][2] += 1
    return {k: {"cer": (100.0 * v[0] / v[1]) if v[1] else None, "chars": v[1], "words": v[2]} for k, v in agg.items()}


def words_of(samples: Sequence[dict]) -> set:
    return {w for s in samples for w in s["text"].split()}


# ----------------------------------------------------------------------------------------------
# Gate baselines (measured independently by scripts/measure_gate_baselines.py)
# ----------------------------------------------------------------------------------------------
def load_gate_baselines(path: Path = GATE_BASELINE_PATH) -> dict:
    return json.load(open(path, encoding="utf-8")) if path.exists() else {"devices": {}}


def gate_check(device: str, subset: str, step0_cer: float, baselines: Optional[dict] = None) -> float:
    baselines = baselines if baselines is not None else load_gate_baselines()
    dev = "cuda" if str(device).startswith("cuda") else "cpu"
    entry = baselines.get("devices", {}).get(dev, {}).get(subset)
    if entry is None:
        raise AssertionError(
            f"STEP-0 GATE ABORT: no independent baseline for device={dev} subset={subset} in {GATE_BASELINE_PATH}. "
            f"Run scripts/measure_gate_baselines.py on this device first.")
    base = float(entry["cer"])
    diff = abs(step0_cer - base)
    print(f"[Step-0 Gate] {subset} on {dev}: evaluated {step0_cer:.2f}% vs baseline {base:.2f}% -> |diff| {diff:.2f} (tol {GATE_TOL})")
    assert diff <= GATE_TOL, (f"STEP-0 GATE ABORT: {subset}/{dev} CER {step0_cer:.2f}% deviates from baseline "
                              f"{base:.2f}% by {diff:.2f} pt (> {GATE_TOL}). Aborting training!")
    print("[Step-0 Gate] PASSED")
    return base


# ----------------------------------------------------------------------------------------------
# Mixed-source training stream
# ----------------------------------------------------------------------------------------------
class MixedStream(IterableDataset):
    """Each yielded sample is drawn from strokes / synth / khatt with the given per-sample weights.
    Used by the actual training loop (and by the tests). Yields (img[1,64,W], target(list[int]), src_idx)."""

    def __init__(self, stroke_samples: List[dict], synth_ds, khatt_ds, weights: Sequence[float],
                 vocab: Vocabulary, steps: int, batch_size: int, seed: int):
        super().__init__()
        self.stroke_samples, self.synth_ds, self.khatt_ds = stroke_samples, synth_ds, khatt_ds
        self.weights, self.vocab = list(weights), vocab
        self.steps, self.batch_size, self.seed = steps, batch_size, seed
        for w, ds, name in ((self.weights[0], stroke_samples, "strokes"), (self.weights[1], synth_ds, "synth"),
                            (self.weights[2], khatt_ds, "khatt")):
            if w > 0 and (ds is None or len(ds) == 0):
                raise ValueError(f"mix requires source '{name}' (weight {w}) but it is empty/unavailable")

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        batches = self.steps // nw + (1 if wid < self.steps % nw else 0)
        rng = random.Random(self.seed * 1_000_003 + wid)
        random.seed(rng.random())  # module-level RNG drives on-the-fly augmentation
        for _ in range(batches * self.batch_size):
            src = rng.choices((0, 1, 2), weights=self.weights, k=1)[0]
            if src == 0:
                s = self.stroke_samples[rng.randrange(len(self.stroke_samples))]
                img = get_canonical_zoom_and_render(s, augment=True, device="cpu").squeeze(0)
                tgt = self.vocab.encode(s["text"][::-1])
            else:
                ds = self.synth_ds if src == 1 else self.khatt_ds
                item = ds[rng.randrange(len(ds))]
                img, tgt = item.image, item.target.tolist()
            yield img, tgt, src


def collate_mixed(batch):
    imgs, targets, srcs = zip(*batch)
    max_w = ((max(i.shape[-1] for i in imgs) + 31) // 32) * 32
    padded = torch.ones((len(imgs), 1, 64, max_w), dtype=torch.float32)
    for k, im in enumerate(imgs):
        padded[k, :, :, :im.shape[-1]] = im
    in_lens = torch.tensor([i.shape[-1] // 4 for i in imgs], dtype=torch.long)
    tgt_lens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    flat = torch.tensor([x for t in targets for x in t], dtype=torch.long)
    return padded, flat, in_lens, tgt_lens, list(srcs)


# ----------------------------------------------------------------------------------------------
# Threshold calibration and full test evaluation (CER + 3-state verification)
# ----------------------------------------------------------------------------------------------
def _quran_dict(limit: int) -> List[str]:
    p = ROOT / "data" / "tanzil" / "quran-simple-clean.txt"
    words: List[str] = []
    if p.exists():
        with open(p, encoding="utf-8") as f:
            for l in f:
                if not l.strip().startswith("#"):
                    words.extend(l.strip().split())
    return sorted(set(words))[:limit]


def _mut_dict():
    d = defaultdict(list)
    for m in extract_mutashabih_pairs():
        d[m["prefix"]].append(m["variant_b"].split()[-1])
    return d


def recalibrate_thresholds(model: nn.Module, calib_samples: List[dict], vocab: Vocabulary, device: str = "cpu"):
    model.eval()
    q_dict, mut_dict = _quran_dict(5000), _mut_dict()
    true_llrs, corr_llrs = [], []
    rng = random.Random(123)
    with torch.no_grad():
        for s in calib_samples:
            inp = get_canonical_zoom_and_render(s, augment=False, device=device)
            out = model(inp).squeeze(1).cpu().numpy()
            txt = s["text"]
            for w_eval in verify_reference(out, txt, vocab.char2idx, vocab.idx2char, t_accept=-0.6, t_wrong=-0.95, order="visual", norm="char"):
                true_llrs.append(w_eval["llr"])
            c_info = apply_corruption(txt, "dot_swap", q_dict, mut_dict, rng)
            if c_info:
                for w_eval in verify_reference(out, c_info[0], vocab.char2idx, vocab.idx2char, t_accept=-0.6, t_wrong=-0.95, order="visual", norm="char"):
                    corr_llrs.append(w_eval["llr"])
    if not true_llrs:
        return -0.60, -0.95
    t_acc = float(np.percentile(true_llrs, 5))
    t_wrg_raw = float(np.percentile(corr_llrs, 90)) if corr_llrs else t_acc - 0.35
    t_wrg = min(t_acc - 0.20, t_wrg_raw)  # monotonic: t_wrong <= t_accept
    return round(t_acc, 4), round(t_wrg, 4)


def evaluate_test_set(model: nn.Module, test_samples: List[dict], vocab: Vocabulary, t_acc: float, t_wrong: float,
                      device: str = "cpu", seen_words: Optional[set] = None, skip_verification: bool = False):
    for s in test_samples:
        assert_no_locked_test(s.get("id", ""))
    model.eval()
    q_dict = _quran_dict(14872)
    mut_dict = _mut_dict() if not skip_verification else {}

    records, line_gt, line_pred, word_gt, word_pred = [], [], [], [], []
    true_states = {"ACCEPT": 0, "UNCERTAIN": 0, "WRONG": 0}
    corr_wrg = defaultdict(lambda: {"WRONG": 0, "TOTAL": 0})
    ctypes = ["dot_swap", "del_ins", "word_swap", "adj_swap", "skip_word", "mutashabih"]
    rng = random.Random(42)

    with torch.no_grad():
        for s in test_samples:
            inp = get_canonical_zoom_and_render(s, augment=False, device=device)
            logits = model(inp)
            pred_txt = greedy_decode(logits, vocab, label_order="visual")[0]
            gt_txt = s["text"]
            records.append({"id": s["id"], "text": gt_txt, "pred": pred_txt})
            if "line" in s.get("id", "") or len(gt_txt.split()) > 2:
                line_gt.append(gt_txt); line_pred.append(pred_txt)
            else:
                word_gt.append(gt_txt); word_pred.append(pred_txt)
            if skip_verification:
                continue
            log_probs = logits.squeeze(1).cpu().numpy()
            for w in verify_reference(log_probs, gt_txt, vocab.char2idx, vocab.idx2char, t_accept=t_acc, t_wrong=t_wrong, order="visual", norm="char"):
                true_states[w["state"]] += 1
            for ct in ctypes:
                c_info = apply_corruption(gt_txt, ct, q_dict, mut_dict, rng)
                if c_info:
                    cv = verify_reference(log_probs, c_info[0], vocab.char2idx, vocab.idx2char, t_accept=t_acc, t_wrong=t_wrong, order="visual", norm="char")
                    corr_wrg[ct]["TOTAL"] += 1
                    if any(w["state"] == "WRONG" for w in cv):
                        corr_wrg[ct]["WRONG"] += 1

    cer_all, wer_all = corpus_cer_wer(line_pred + word_pred, line_gt + word_gt)
    cer_lines = corpus_cer_wer(line_pred, line_gt)[0] if line_gt else 0.0
    cer_words = corpus_cer_wer(word_pred, word_gt)[0] if word_gt else 0.0
    res: Dict[str, Any] = {
        "cer_overall": round(cer_all * 100.0, 2), "cer_lines": round(cer_lines * 100.0, 2),
        "cer_words": round(cer_words * 100.0, 2), "wer_overall": round(wer_all * 100.0, 2),
    }
    if not skip_verification:
        tot = sum(true_states.values()) or 1
        res.update({
            "true_accept_rate": round(100.0 * true_states["ACCEPT"] / tot, 2),
            "true_unc_rate": round(100.0 * true_states["UNCERTAIN"] / tot, 2),
            "true_wrg_rate": round(100.0 * true_states["WRONG"] / tot, 2),
            "corr_rejection": {ct: (round(100.0 * corr_wrg[ct]["WRONG"] / corr_wrg[ct]["TOTAL"], 2)
                                    if corr_wrg[ct]["TOTAL"] else None) for ct in ctypes},
        })
    if seen_words is not None:
        res["lexical"] = stratify_cer(records, seen_words)
    return res


# ----------------------------------------------------------------------------------------------
# Forgetting evals (KHATT val, synthetic test) -- same loader/metric as evaluate_continuation.py
# ----------------------------------------------------------------------------------------------
def eval_shard_cer(model: nn.Module, shard_path: Path, vocab: Vocabulary, device: str, limit: int = 0,
                   batch_size: int = 32) -> float:
    from shards import ShardDataset
    from dataset import collate_fn as shard_collate
    from evaluate import evaluate as eval_loader
    assert_no_locked_test(str(shard_path))
    ds = ShardDataset(shard_path, vocab, augment=False, label_order="visual")
    if limit:
        ds = Subset(ds, list(range(min(limit, len(ds)))))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=shard_collate)
    cer_v, _, _ = eval_loader(model, loader, vocab, torch.device(device), label_order="visual", max_samples_display=0)
    return round(cer_v * 100.0, 3)


# ----------------------------------------------------------------------------------------------
# One training run
# ----------------------------------------------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_run(
    fold: str, mix_name: str, seed: int, train_samples: List[dict], calib_samples: List[dict],
    test_samples: List[dict], subset: str, vocab: Vocabulary, base_ckpt: Path, synth_ds, khatt_ds,
    forget_paths: Dict[str, Path], args, device: str, lc_size: Optional[int] = None,
) -> dict:
    weights = MIXES[mix_name]
    print(f"\n{'=' * 70}\nRUN fold={fold}{'' if lc_size is None else f'(n={lc_size})'} mix={mix_name} {weights} seed={seed} "
          f"| train={len(train_samples)} calib={len(calib_samples)} test={len(test_samples)} | {args.epochs}x{args.steps_per_epoch} steps\n{'=' * 70}")
    for s in train_samples + calib_samples + test_samples:
        assert_no_locked_test(s.get("id", ""))
    set_seed(seed)
    model, matched_msg = load_base_model(base_ckpt, vocab, device)

    # Step-0 gate (before any optimizer step)
    step0_cer = canonical_cer(model, test_samples, vocab, device)
    baseline = gate_check(device, subset, step0_cer)

    seen_words = words_of(train_samples)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=args.lr / 10.0)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    epoch_log = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        stream = MixedStream(train_samples, synth_ds, khatt_ds, weights, vocab, args.steps_per_epoch,
                             args.batch_size, seed=seed * 10_000 + epoch)
        loader = DataLoader(stream, batch_size=args.batch_size, num_workers=args.num_workers, collate_fn=collate_mixed)
        counts = {k: 0 for k in SOURCES}
        loss_sum, steps_done = 0.0, 0
        for imgs, flat, in_lens, tgt_lens, srcs in loader:
            for s in srcs:
                counts[SOURCES[s]] += 1
            optimizer.zero_grad()
            lp = model(imgs.to(device)).log_softmax(dim=-1)
            loss = ctc(lp, flat.to(device), in_lens, tgt_lens)
            if torch.isfinite(loss):
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                loss_sum += loss.item()
            scheduler.step()
            steps_done += 1
        assert steps_done == args.steps_per_epoch, f"expected {args.steps_per_epoch} steps, ran {steps_done}"
        if weights[1] > 0:
            assert counts["synth"] > 0 and counts["khatt"] > 0, f"epoch {epoch}: zero synth/khatt samples {counts}"
        if weights[0] > 0:
            assert counts["strokes"] > 0, f"epoch {epoch}: zero stroke samples {counts}"
        avg = loss_sum / max(1, steps_done)
        epoch_log.append({"epoch": epoch, "loss": round(avg, 5), "samples_by_source": dict(counts)})
        print(f"  [{fold}/{mix_name}/s{seed}] epoch {epoch}/{args.epochs} loss={avg:.4f} lr={scheduler.get_last_lr()[0]:.6f} "
              f"samples by source: strokes={counts['strokes']} synth={counts['synth']} khatt={counts['khatt']}")
    train_secs = round(time.time() - t0, 1)

    thresholds = None
    if not args.skip_verification and calib_samples:
        t_acc, t_wrg = recalibrate_thresholds(model, calib_samples, vocab, device=device)
        thresholds = [t_acc, t_wrg]
    else:
        t_acc, t_wrg = -0.60, -0.95
    res = evaluate_test_set(model, test_samples, vocab, t_acc, t_wrg, device=device, seen_words=seen_words,
                            skip_verification=args.skip_verification or not calib_samples)

    forgetting = {}
    if not args.skip_forgetting:
        for name, p in forget_paths.items():
            if p.exists():
                forgetting[name] = eval_shard_cer(model, p, vocab, device, limit=args.forget_limit)
            else:
                forgetting[name] = None
    rec = {
        "fold": fold, "mix": mix_name, "weights": list(weights), "seed": seed, "lc_size": lc_size, "device": device,
        "epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size, "lr": args.lr,
        "init_weights": matched_msg, "subset": subset, "n_train": len(train_samples), "n_test": len(test_samples),
        "step0_cer": round(step0_cer, 2), "gate_baseline": baseline, "epoch_log": epoch_log, "train_secs": train_secs,
        "thresholds": thresholds, "result": res, "forgetting": forgetting, "forget_limit": args.forget_limit,
    }
    print(f"  RESULT {fold}/{mix_name}/s{seed}: CER {step0_cer:.2f}% -> {res['cer_overall']}% "
          f"(lines {res['cer_lines']} / words {res['cer_words']}) forgetting={forgetting}")
    return rec


def run_key(r: dict) -> tuple:
    return (r["fold"], r["mix"], r["seed"], r.get("lc_size"), r["epochs"], r["steps_per_epoch"], r["device"])


def ensure_shards_available(shards: Sequence[str | Path]):
    import shutil
    import zipfile
    for sp in shards:
        p = Path(sp) if Path(sp).is_absolute() else ROOT / sp
        if p.exists():
            continue
        kind = "synth" if "synth" in p.name else ("khatt" if "khatt" in p.name else None)
        if not kind:
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        # 1. Direct copy candidate files
        direct_candidates = [
            Path(f"/content/drive/MyDrive/quran_htr_data/shards/{kind}/{p.name}"),
            Path(f"/content/drive/MyDrive/quran_htr_data/{p.name}"),
            Path(f"/content/drive/MyDrive/quran_htr/data/shards/{kind}/{p.name}"),
            Path(f"/content/drive/MyDrive/shards_{kind}/{p.name}"),
            Path(f"/content/drive/MyDrive/{p.name}"),
        ]
        for dc in direct_candidates:
            if dc.exists():
                print(f"[Auto-Sync] Copying {dc} -> {p}...")
                shutil.copy2(dc, p)
                break
        if p.exists():
            continue

        # 2. Extract from zip archives
        zip_candidates = [
            Path(f"/content/drive/MyDrive/quran_htr/dist/shards_{kind}.zip"),
            Path(f"/content/drive/MyDrive/shards_{kind}.zip"),
            ROOT / f"dist/shards_{kind}.zip",
            Path("/content/drive/MyDrive/quran_htr/dist/shards_corpus_v3.zip"),
            Path("/content/drive/MyDrive/shards_corpus_v3.zip"),
            ROOT / "dist/shards_corpus_v3.zip",
        ]
        for z_path in zip_candidates:
            if z_path.exists():
                try:
                    with zipfile.ZipFile(z_path, "r") as zf:
                        matches = [m for m in zf.namelist() if m.endswith(p.name) or Path(m).name == p.name]
                        if matches:
                            print(f"[Auto-Extract] Extracting {matches[0]} from {z_path} -> {p}...")
                            with zf.open(matches[0]) as sf, open(p, "wb") as df:
                                shutil.copyfileobj(sf, df)
                            print(f"[Auto-Extract] Successfully extracted {p}")
                            break
                except Exception as e:
                    print(f"[Auto-Extract Warn] Failed extracting from {z_path}: {e}")
            if p.exists():
                break


def main():
    ap = argparse.ArgumentParser(description="Dev-writer cross-writer fine-tuning with anchored mix, seeds and gate")
    ap.add_argument("--folds", nargs="+", default=["A", "B", "C"], help="A B C CLC")
    ap.add_argument("--mixes", nargs="+", default=list(MIXES), choices=list(MIXES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--lc_sizes", nargs="+", type=int, default=list(LC_SIZES))
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--steps_per_epoch", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--base_checkpoint", default=None)
    ap.add_argument("--synth_shards", nargs="+", default=["data/shards/synth/synth_train_000.pt"])
    ap.add_argument("--khatt_shards", nargs="+", default=["data/shards/khatt/khatt_train_000.pt"])
    ap.add_argument("--khatt_val", default="data/shards/khatt/khatt_val_000.pt")
    ap.add_argument("--synth_test", default="data/shards/synth/synth_test_000.pt")
    ap.add_argument("--forget_limit", type=int, default=0, help="0 = full eval sets")
    ap.add_argument("--skip_forgetting", action="store_true")
    ap.add_argument("--skip_verification", action="store_true", help="CER only (no thresholds / 3-state)")
    ap.add_argument("--results", default="reports/lowo_runs.jsonl")
    args = ap.parse_args()

    for p in list(args.synth_shards) + list(args.khatt_shards) + [args.khatt_val, args.synth_test, args.results]:
        assert_no_locked_test(p)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    print(f"Device: {device} | folds={args.folds} mixes={args.mixes} seeds={args.seeds}")

    generate_all_group_splits()
    vocab = Vocabulary.load(ROOT / "vocab.json")
    base_ckpt = find_base_checkpoint(args.base_checkpoint)
    assert base_ckpt is not None, "base checkpoint not found"
    print(f"Base checkpoint: {base_ckpt}")
    sp = load_dev_splits()
    subsets = gate_subsets(sp)

    need_anchor = any(MIXES[m][1] > 0 for m in args.mixes)
    synth_ds = khatt_ds = None
    if need_anchor:
        from shards import ShardDataset
        synth_files = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in args.synth_shards]
        khatt_files = [ROOT / p if not Path(p).is_absolute() else Path(p) for p in args.khatt_shards]
        ensure_shards_available(synth_files + khatt_files)
        missing = [str(f) for f in synth_files + khatt_files if not f.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing anchor shard(s) for mix training: {missing}. "
                f"Ensure shards_synth.zip and shards_khatt.zip are in Drive /dist/ or run with '--mixes stroke_only'.")
        synth_ds = ShardDataset(synth_files, vocab, augment=True, label_order="visual")
        khatt_ds = ShardDataset(khatt_files, vocab, augment=True, label_order="visual")
        print(f"Anchor pools: synth={len(synth_ds)} khatt={len(khatt_ds)}")
    forget_paths = {}
    rp = lambda p: Path(p) if Path(p).is_absolute() else ROOT / p
    if not args.skip_forgetting:
        forget_paths = {"khatt_val_cer": rp(args.khatt_val), "synth_test_cer": rp(args.synth_test)}
        ensure_shards_available(list(forget_paths.values()))

    results_path = rp(args.results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if results_path.exists():
        for line in open(results_path, encoding="utf-8"):
            if line.strip():
                done.add(run_key(json.loads(line)))
    print(f"Resuming: {len(done)} runs already recorded in {results_path}")

    # nested learning-curve pool: w02 train+calib (14 groups) in group order
    w2_by_id, w2_groups = build_w2_groups()
    pool_ids = {s["id"] for s in sp["w2_tr"] + sp["w2_cal"]}
    pool_by_group = {g: [sid for sid in sids if sid in pool_ids] for g, sids in w2_groups.items()}
    pool_by_group = {g: v for g, v in pool_by_group.items() if v and g != "group_extra"}
    for sid in w2_groups.get("group_extra", []):
        if sid in pool_ids:
            pool_by_group[f"extra::{sid}"] = [sid]

    folds = [f.upper() for f in args.folds]
    for seed in args.seeds:
        jobs = []
        for mix in args.mixes:
            if "A" in folds:
                jobs.append(("A", mix, None, sp["w1_tr"], sp["w1_cal"], sp["w2_all"], "A_test"))
            if "B" in folds:
                jobs.append(("B", mix, None, sp["w2_tr"], sp["w2_cal"], sp["w1_all"], "B_test"))
            if "C" in folds:
                jobs.append(("C", mix, None, sp["fc_pure_tr"], sp["fc_cal"], sp["fc_te"], "C_test"))
            if "CLC" in folds:
                order = lc_order(pool_by_group, seed)
                for n in args.lc_sizes:
                    jobs.append(("CLC", mix, n, [w2_by_id[s] for s in order[:n]], [], sp["w2_held"], "LC_test"))
        for fold, mix, n, tr, cal, te, subset in jobs:
            probe = {"fold": fold, "mix": mix, "seed": seed, "lc_size": n, "epochs": args.epochs,
                     "steps_per_epoch": args.steps_per_epoch, "device": device}
            if run_key(probe) in done:
                print(f"[skip] already done: {run_key(probe)}")
                continue
            if fold == "CLC":
                a = argparse.Namespace(**vars(args))
                a.skip_verification = True
            else:
                a = args
            rec = train_single_run(fold, mix, seed, tr, cal, te, subset, vocab, base_ckpt, synth_ds, khatt_ds,
                                   forget_paths, a, device, lc_size=n)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done.add(run_key(rec))
    print(f"\nAll requested runs complete -> {results_path}")


if __name__ == "__main__":
    main()
