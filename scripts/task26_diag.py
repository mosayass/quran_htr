"""
scripts/task26_diag.py
Brief #26 local, CPU-only diagnostics (dev writers only; locked-test writers are never touched).

Sections (run with --section):
  partition  item 1a: Fold C vs learning-curve (LC) N=75 -- train/test composition, overlap, base CER per subset
  seeds      item 2 : do seeds change sampling + augmentation? (MixedStream draw log, stub anchor pools)
  timing     item 3 : measured CPU seconds/step of the fine-tune loop (stroke_only) -> run-time budget
  scale      item 4a: scale-normalization term breakdown (which term makes S_w) + CER at fixed zooms
  adabn      item 4b: AdaBN as implemented vs a corrected layer-wise pooled-statistics AdaBN
Writes reports/task26_<section>.json.
"""

from __future__ import annotations
import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from vocab import Vocabulary
import train_twofold_lowo as T
from scale_policy import writer_scale, canonical_zoom_of, ink_extents, clip_bound, sample_band


def _setup(device="cpu"):
    T.generate_all_group_splits()
    vocab = Vocabulary.load(ROOT / "vocab.json")
    ckpt = T.find_base_checkpoint(None)
    model, msg = T.load_base_model(ckpt, vocab, device)
    sp = T.load_dev_splits()
    return vocab, model, sp, msg


def _kind(s):
    return "line" if "line" in s["id"] else "word"


def _comp(samples):
    c = Counter(_kind(s) for s in samples)
    return {"n": len(samples), "lines": c["line"], "words": c["word"],
            "chars": sum(len(s["text"]) for s in samples)}


def _save(name, obj):
    p = ROOT / "reports" / f"task26_{name}.json"
    p.parent.mkdir(exist_ok=True)
    json.dump(obj, open(p, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"wrote {p}")


# ------------------------------------------------------------------------------------------------
def sec_partition(dev):
    vocab, model, sp, _ = _setup(dev)
    ids = lambda xs: {s["id"] for s in xs}
    fc_tr, fc_cal, fc_te = sp["fc_pure_tr"], sp["fc_cal"], sp["fc_te"]
    pool = sp["w2_tr"] + sp["w2_cal"]
    lc_te = sp["w2_held"]
    out = {"device": dev}

    out["fold_C"] = {"train(fc_pure_tr)": _comp(fc_tr), "calib": _comp(fc_cal), "test(fc_te)": _comp(fc_te)}
    out["LC"] = {"pool(w2_tr+w2_cal)": _comp(pool), "test(w2_held)": _comp(lc_te)}

    inter = ids(lc_te) & ids(fc_te)
    only_c = [s for s in fc_te if s["id"] not in ids(lc_te)]
    out["test_overlap"] = {"LC_test_subset_of_C_test": ids(lc_te) <= ids(fc_te), "n_common": len(inter),
                           "C_test_only": _comp(only_c)}
    out["train_vs_test_leak"] = {
        "C_train&C_test": len(ids(fc_tr) & ids(fc_te)), "C_cal&C_test": len(ids(fc_cal) & ids(fc_te)),
        "LCpool&LC_test": len(ids(pool) & ids(lc_te)),
        "LCpool&C_test(samples of C_test that LC trains on)": len(ids(pool) & ids(fc_te)),
        "C_train&LC_test": len(ids(fc_tr) & ids(lc_te)),
    }
    # exact-text duplicates between a train set and a test set (same prompt written twice = lexical leak)
    def dup(train, test):
        tt = {s["text"] for s in train}
        n = sum(1 for s in test if s["text"] in tt)
        return {"test_samples_with_identical_text_in_train": n, "of": len(test)}
    def tok_seen(train, test):
        sw = T.words_of(train)
        toks = [w for s in test for w in s["text"].split()]
        return {"tokens": len(toks), "seen_pct": round(100.0 * sum(w in sw for w in toks) / max(1, len(toks)), 2)}
    out["text_dup_and_token_seen"] = {
        "C_train->C_test": {**dup(fc_tr, fc_te), **tok_seen(fc_tr, fc_te)},
        "LCpool->LC_test": {**dup(pool, lc_te), **tok_seen(pool, lc_te)},
        "C_train->LC_test": {**dup(fc_tr, lc_te), **tok_seen(fc_tr, lc_te)},
    }
    # LC N=75 composition per seed (nested prefixes of a seed-dependent group order)
    w2_by_id, w2_groups = T.build_w2_groups()
    pool_ids = ids(pool)
    pbg = {g: [x for x in v if x in pool_ids] for g, v in w2_groups.items()}
    pbg = {g: v for g, v in pbg.items() if v and g != "group_extra"}
    for sid in w2_groups.get("group_extra", []):
        if sid in pool_ids:
            pbg[f"extra::{sid}"] = [sid]
    out["LC_N75_by_seed"] = {}
    for seed in range(5):
        order = T.lc_order(pbg, seed)
        tr = [w2_by_id[x] for x in order[:75]]
        out["LC_N75_by_seed"][seed] = {**_comp(tr), **tok_seen(tr, lc_te),
                                       "n_groups_touched": len({x.split('::')[0] if x.startswith('extra') else
                                                                next(g for g, v in pbg.items() if x in v) for x in order[:75]})}

    # base-model CER (same canonical eval as gate), per subset
    t0 = time.time()
    subsets = {"C_test": fc_te, "LC_test": lc_te, "C_test_only(31)": only_c, "C_train(56)": fc_tr,
               "C_calib": fc_cal, "LC_pool(100)": pool}
    out["base_cer_pct"] = {k: round(T.canonical_cer(model, v, vocab, dev), 2) for k, v in subsets.items()}
    out["base_cer_eval_secs"] = round(time.time() - t0, 1)
    print(json.dumps(out, indent=1, ensure_ascii=False))
    _save("partition", out)


# ------------------------------------------------------------------------------------------------
class _Stub:
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        from types import SimpleNamespace
        return SimpleNamespace(image=torch.zeros(1, 64, 32), target=torch.tensor([1]))


def sec_seeds(dev):
    vocab, model, sp, _ = _setup(dev)
    train = sp["w2_tr"]  # Fold B train pool
    n_draw = 8 * 200
    logs = {}
    for seed in (0, 1, 2, 3):
        # replicate train_single_run: set_seed(seed) then MixedStream(seed=seed*10000+epoch)
        T.set_seed(seed)
        st = T.MixedStream(train, _Stub(1000), _Stub(1000), T.MIXES["mix_a"], vocab, 200, 8, seed=seed * 10_000 + 1)
        # capture which stroke sample + augmentation params get used (patch renderer to record z,w)
        rec = []
        orig = T.render_strokes

        def spy(strokes, zoom, width_after_zoom, **kw):
            rec.append((round(zoom, 4), round(width_after_zoom, 4)))
            return orig(strokes, zoom=zoom, width_after_zoom=width_after_zoom, **kw)
        T.render_strokes = spy
        idx_log = []
        try:
            rng = random.Random(st.seed * 1_000_003 + 0)
            # mirror MixedStream.__iter__ draws for index bookkeeping (separate RNG object, same seed)
            rng2 = random.Random(st.seed * 1_000_003 + 0)
            random.seed(rng2.random())
            for _ in range(n_draw):
                src = rng2.choices((0, 1, 2), weights=st.weights, k=1)[0]
                if src == 0:
                    idx_log.append(rng2.randrange(len(train)))
                else:
                    rng2.randrange(1000)
            # now run the true iterator and capture render params + sources
            srcs = []
            for k, (img, tgt, src) in enumerate(st):
                srcs.append(src)
                if k + 1 >= n_draw:
                    break
        finally:
            T.render_strokes = orig
        h = lambda x: hashlib.md5(json.dumps(x).encode()).hexdigest()[:10]
        logs[seed] = {"first_20_sources": srcs[:20], "src_hash": h(srcs), "stroke_idx_hash": h(idx_log),
                      "first_5_aug(zoom,width)": rec[:5], "aug_hash": h(rec), "n_stroke_renders": len(rec),
                      "stroke_ids_distinct_in_800_draws": len(set(idx_log)), "pool": len(train)}
    out = {"fold_B_train_pool": len(train), "draws_per_seed_logged": n_draw, "per_seed": logs}
    # all-distinct check
    for k in ("src_hash", "aug_hash", "stroke_idx_hash"):
        out[f"all_seeds_distinct_{k}"] = len({v[k] for v in logs.values()}) == len(logs)
    # exposure: total stroke-sample draws in a 6x1000x8 run under mix_a
    total_draws = 6 * 1000 * 8 * 0.5
    out["exposure"] = {"mix_a_stroke_draws_per_run": total_draws,
                       "avg_draws_per_train_sample": round(total_draws / len(train), 1)}
    # CER quantum of the Fold B test set
    w1 = sp["w1_all"]
    chars = sum(len(s["text"]) for s in w1)
    out["fold_B_test"] = {"n": len(w1), "ref_chars": chars, "one_char_error_in_CER_pct": round(100.0 / chars, 3),
                          "std_0.03pt_in_chars": round(0.03 * chars / 100.0, 2)}
    print(json.dumps(out, indent=1))
    _save("seeds", out)


# ------------------------------------------------------------------------------------------------
def sec_timing(dev, steps):
    import torch.nn as nn
    from torch.utils.data import DataLoader
    vocab, model, sp, _ = _setup(dev)
    train = sp["fc_pure_tr"]
    torch.set_num_threads(max(1, torch.get_num_threads()))
    st = T.MixedStream(train, None, None, T.MIXES["stroke_only"], vocab, steps, 8, seed=1)
    loader = DataLoader(st, batch_size=8, num_workers=0, collate_fn=T.collate_mixed)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-5)
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)
    model.train()
    t_data = t_step = 0.0
    t_last = time.time()
    n = 0
    for imgs, flat, il, tl, _ in loader:
        t_data += time.time() - t_last
        t1 = time.time()
        opt.zero_grad()
        loss = ctc(model(imgs).log_softmax(-1), flat, il, tl)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        t_step += time.time() - t1
        n += 1
        t_last = time.time()
    # eval timing
    t0 = time.time()
    T.canonical_cer(model, sp["fc_te"], vocab, dev)
    ev = time.time() - t0
    out = {"device": dev, "threads": torch.get_num_threads(), "steps_timed": n,
           "render_s_per_step": round(t_data / n, 3), "fwd_bwd_s_per_step": round(t_step / n, 3),
           "total_s_per_step": round((t_data + t_step) / n, 3),
           "extrapolated_6x1000_stroke_only_min": round(6000 * (t_data + t_step) / n / 60, 1),
           "canonical_eval_C_test_76_secs": round(ev, 1)}
    print(json.dumps(out, indent=1))
    _save("timing", out)


# ------------------------------------------------------------------------------------------------
def sec_scale(dev):
    vocab, model, sp, _ = _setup(dev)
    out = {"device": dev, "writers": {}}
    for wname, samples in (("writer_01", sp["w1_all"]), ("writer_02", sp["w2_all"])):
        pol = writer_scale(samples)
        tot = np.array([ink_extents(s)[2] for s in samples])
        bh = pol["band_height"]
        clip = np.array([clip_bound(s) for s in samples])
        kinds = np.array([_kind(s) for s in samples])
        o = {"policy": pol, "ink_extent_css_px": {"p10": float(np.percentile(tot, 10)), "median": float(np.median(tot)),
                                                  "p90": float(np.percentile(tot, 90)), "max": float(tot.max())},
             "ink_px_at_zoom1_median(64px tensor)": float(np.median(tot) * 64 / bh),
             "clip_bound_percentiles": {f"p{q}": round(float(np.percentile(clip, q)), 3) for q in (0, 1, 5, 10, 25, 50)},
             "median_ink_extent_by_kind": {k: round(float(np.median(tot[kinds == k])), 1) for k in ("line", "word")},
             "n_samples_with_clip_bound_below_0.8": int((clip < 0.8).sum()),
             "argmin_clip_sample": samples[int(clip.argmin())]["id"],
             "binding_term": min(("S_target", "S_formula58", "S_clip_exact"), key=lambda k: pol[k])}
        # CER at fixed zooms (relative to canonical): this is the empirical check on the 0.8 expectation
        grid = {}
        for z in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            zz = z * canonical_zoom_of(samples[0])
            grid[f"{z:.1f}x_canonical"] = round(T.records_cer(T.predict_records(model, samples, vocab, dev,
                                                                              zoom_fn=lambda s, zz=zz: zz)), 2)
        o["cer_vs_zoom_multiplier"] = grid
        out["writers"][wname] = o
        print(wname, json.dumps(o, indent=1))
    _save("scale", out)


# ------------------------------------------------------------------------------------------------
def adabn_pooled(model, samples, device, zoom_fn=None):
    """Corrected AdaBN: process BN layers in network order; for layer k use already-adapted layers < k in eval
    mode, and accumulate POOLED per-channel mean / E[x^2] over every pixel of every sample (batch size irrelevant,
    no per-sample self-normalisation, no cumulative-mean-of-variances bias)."""
    import copy
    import torch.nn as nn
    m = copy.deepcopy(model).to(device).eval()
    bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d)]
    inputs = [T.get_canonical_zoom_and_render(s, augment=False, device=device,
                                              zoom_override=(zoom_fn(s) if zoom_fn else None)) for s in samples]
    for b in bns:
        acc = {"n": 0, "s": 0.0, "q": 0.0}

        def hook(mod, inp, acc=acc):
            x = inp[0].double()
            acc["n"] += x.numel() // x.shape[1]
            acc["s"] = acc["s"] + x.sum(dim=(0, 2, 3))
            acc["q"] = acc["q"] + (x * x).sum(dim=(0, 2, 3))
        h = b.register_forward_pre_hook(hook)
        with torch.no_grad():
            for x in inputs:
                m(x)
        h.remove()
        mean = acc["s"] / acc["n"]
        var = (acc["q"] / acc["n"] - mean * mean).clamp_min(0)
        b.running_mean.copy_(mean.float())
        b.running_var.copy_(var.float())
    m.eval()
    return m


def sec_adabn(dev):
    import label_free_adaptation as L
    vocab, model, sp, _ = _setup(dev)
    out = {"device": dev, "writers": {}}
    for wname, samples in (("writer_01", sp["w1_all"]), ("writer_02", sp["w2_all"])):
        cer = lambda m: round(T.records_cer(T.predict_records(m, samples, vocab, dev)), 2)
        row = {"baseline": cer(model), "adabn_as_implemented(per-sample batch stats, mean of variances)": cer(L.adapt_bn(model, samples, dev)),
               "adabn_pooled_layerwise": cer(adabn_pooled(model, samples, dev))}
        # sanity: adapting on the TRAINING-domain-like data must be ~no-op. Use half of samples for stats, other half
        # for test to expose a stats-sample-size effect.
        half = len(samples) // 2
        mh = adabn_pooled(model, samples[:half], dev)
        row["adabn_pooled_stats_from_first_half"] = round(T.records_cer(T.predict_records(mh, samples[half:], vocab, dev)), 2)
        row["baseline_on_second_half"] = round(T.records_cer(T.predict_records(model, samples[half:], vocab, dev)), 2)
        out["writers"][wname] = row
        print(wname, json.dumps(row, indent=1))
    _save("adabn", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", required=True, choices=["partition", "seeds", "timing", "scale", "adabn"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()
    {"partition": lambda: sec_partition(a.device), "seeds": lambda: sec_seeds(a.device),
     "timing": lambda: sec_timing(a.device, a.steps), "scale": lambda: sec_scale(a.device),
     "adabn": lambda: sec_adabn(a.device)}[a.section]()
