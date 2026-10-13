"""
tests/test_mix_training.py  (Task 2.25)
Tests of the REAL code paths (not a re-implementation of the sampler):
  (a) visual-order labels: ShardDataset(label_order="visual") == stroke path target for the same text
  (b) mix composition over 1,000 draws through the actual MixedStream class, for both mixes
  (c) per-word error attribution preserves the edit distance
  (d) learning-curve order is nested and group-respecting
  (e) gate: passes within 1.0 pt, aborts beyond, aborts without an independent baseline
  (f) a 20-step run of the ACTUAL script (scripts/train_twofold_lowo.py) asserts nonzero synth+khatt each epoch
  (g) label-free scale policy never clips ink on the dev writers
  (h) locked_test paths are refused
Run: python tests/test_mix_training.py   (or pytest)
"""

from __future__ import annotations
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
from metrics import _levenshtein
from shards import ShardDataset, write_shard_file
import train_twofold_lowo as T
from scale_policy import writer_scale, is_clipped, sample_band
from render_strokes import render_strokes

VOCAB = Vocabulary.load(ROOT / "vocab.json")


def _w2():
    return T._dedup(json.load(open(ROOT / "data/capture_v2/writer_02/strokes.json", encoding="utf-8")))


def _tiny_shard(tmp: Path, name: str, texts):
    imgs = [torch.full((64, 40 + 8 * i), 255, dtype=torch.uint8) for i in range(len(texts))]
    p = tmp / name
    write_shard_file(imgs, list(texts), p)
    return p


def test_a_visual_order_shard_equals_stroke_path():
    sample = _w2()[0]
    text = sample["text"]
    with tempfile.TemporaryDirectory() as d:
        p = _tiny_shard(Path(d), "s.pt", [text])
        ds = ShardDataset(p, VOCAB, augment=False, label_order="visual")
        shard_target = ds[0].target.tolist()
    stream = T.MixedStream([sample], None, None, (1.0, 0.0, 0.0), VOCAB, steps=1, batch_size=1, seed=0)
    _, stroke_target, src = next(iter(stream))
    assert src == 0
    assert shard_target == stroke_target == VOCAB.encode(text[::-1]), "visual-order labels differ between paths"


def test_b_mix_composition_actual_stream():
    w2 = _w2()[:5]
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        synth = ShardDataset(_tiny_shard(d, "a.pt", [w2[0]["text"], w2[1]["text"]]), VOCAB, label_order="visual")
        khatt = ShardDataset(_tiny_shard(d, "b.pt", [w2[2]["text"], w2[3]["text"]]), VOCAB, label_order="visual")
        for weights in ((0.5, 0.3, 0.2), (0.3, 0.4, 0.3)):
            stream = T.MixedStream(w2, synth, khatt, weights, VOCAB, steps=125, batch_size=8, seed=7)
            counts = [0, 0, 0]
            for _, _, src in stream:
                counts[src] += 1
            n = sum(counts)
            assert n == 1000, n
            fr = [c / n for c in counts]
            print(f"mix {weights}: observed {[round(x, 3) for x in fr]}")
            assert all(abs(f - w) < 0.05 for f, w in zip(fr, weights)), (fr, weights)


def test_c_per_word_errors_preserve_edit_distance():
    pairs = [("وقيل يا ارض ابلعي", "وقيل ارض ابلعي"), ("الله", "اله"), ("بسم الله الرحمن", "بسم الله الرحيم x"), ("ab cd", "")]
    for ref, hyp in pairs:
        pw = T.per_word_errors(ref, hyp)
        assert sum(e for _, e, _ in pw) == _levenshtein(list(ref), list(hyp)), (ref, hyp, pw)
        assert [w for w, _, _ in pw] == ref.split(" ")


def test_d_learning_curve_order():
    groups = {f"g{i}": [f"g{i}_s{j}" for j in range(i % 3 + 1)] for i in range(8)}
    o1, o2 = T.lc_order(groups, 0), T.lc_order(groups, 1)
    assert sorted(o1) == sorted(o2) == sorted(s for v in groups.values() for s in v)
    assert o1 != o2
    for n in (3, 5):  # nested prefixes
        assert o1[:n] == T.lc_order(groups, 0)[:n]
    seen_groups = []
    for sid in o1:
        g = sid.split("_")[0]
        if not seen_groups or seen_groups[-1] != g:
            assert g not in seen_groups, "group interleaved"
            seen_groups.append(g)


def test_e_gate():
    bl = {"devices": {"cpu": {"A_test": {"cer": 37.24}}}}
    T.gate_check("cpu", "A_test", 37.9, bl)
    for bad in (lambda: T.gate_check("cpu", "A_test", 38.5, bl), lambda: T.gate_check("cuda", "A_test", 37.2, bl),
                lambda: T.gate_check("cpu", "C_test", 37.2, bl)):
        try:
            bad()
        except AssertionError:
            continue
        raise AssertionError("gate should have aborted")


def test_f_actual_script_20_steps():
    sv, kv = ROOT / "data/shards/synth/synth_val_000.pt", ROOT / "data/shards/khatt/khatt_val_000.pt"
    if not (sv.exists() and kv.exists() and T.GATE_BASELINE_PATH.exists()):
        print("SKIP test_f: small shards / gate baselines missing")
        return
    with tempfile.TemporaryDirectory() as d:
        res = Path(d) / "runs.jsonl"
        cmd = [sys.executable, str(ROOT / "scripts/train_twofold_lowo.py"), "--folds", "A", "--mixes", "mix_a",
               "--seeds", "0", "--epochs", "2", "--steps_per_epoch", "10", "--batch_size", "4", "--device", "cpu",
               "--synth_shards", str(sv), "--khatt_shards", str(kv), "--synth_test", str(sv), "--khatt_val", str(kv),
               "--forget_limit", "16", "--skip_verification", "--results", str(res)]
        out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
        assert out.returncode == 0, out.stdout[-2000:] + out.stderr[-2000:]
        rec = json.loads(res.read_text(encoding="utf-8").strip().splitlines()[-1])
    assert rec["init_weights"] == "init_weights matched 74/74 tensors"
    assert len(rec["epoch_log"]) == 2
    total = 0
    for e in rec["epoch_log"]:
        c = e["samples_by_source"]
        assert c["synth"] > 0 and c["khatt"] > 0 and c["strokes"] > 0, c
        assert sum(c.values()) == 10 * 4, c
        total += 10
    assert total == 20
    assert set(rec["forgetting"]) == {"khatt_val_cer", "synth_test_cer"}
    print("20-step script run OK:", [e["samples_by_source"] for e in rec["epoch_log"]], "step0", rec["step0_cer"])


def test_g_scale_policy_never_clips():
    sp = T.load_dev_splits()
    for name, samples in (("writer_01", sp["w1_all"]), ("writer_02", sp["w2_all"])):
        pol = writer_scale(samples)
        clipped = 0
        for s in samples:
            bh, cy, bt = sample_band(s)
            img = render_strokes(s["strokes"], zoom=pol["S_w"], width_after_zoom=2.2, center_y=cy, band_top_y=bt,
                                 band_height=bh, pad_x=16.0)
            clipped += is_clipped(np.array(img))
        print(f"{name}: S_w={pol['S_w']:.3f} clipped={clipped}/{len(samples)}")
        assert clipped == 0, f"{name}: {clipped} clipped samples at S_w={pol['S_w']}"


def test_h_locked_test_refused():
    try:
        T.assert_no_locked_test("data/locked_test/writer_03/strokes.json")
    except AssertionError:
        return
    raise AssertionError("locked_test path was not refused")


if __name__ == "__main__":
    for fn in [v for k, v in sorted(globals().items()) if k.startswith("test_")]:
        print(f"-- {fn.__name__}")
        fn()
    print("ALL TESTS PASSED")
