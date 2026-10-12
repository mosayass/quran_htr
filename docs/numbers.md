# Canonical Numbers & Benchmarks Master Table

This document serves as the single source of truth for all reported empirical metrics. Every number includes its evaluation set, provenance (in-sample vs. held-out), preprocessing policy, threshold calibration source, and recording date.

---

## 1. Master Metric Matrix

| Metric Name | Value | Evaluation Dataset | In-Sample / Held-Out | Preprocessing Policy | Threshold Source | Recording Date | Notes & Status |
| :--- | :---: | :--- | :---: | :--- | :--- | :---: | :--- |
| **Synthetic Baseline CER** | **1.20%** | `synth_test_000.pt` (4,000 lines) | Held-Out | Native raster | N/A (Greedy Decode) | 2026-10-09 | Out-of-sample fonts & surahs |
| **Synthetic True Accept** | **100.0%** | `synth_test_000.pt` | Held-Out | Native raster | DEV w01 ($t_{\text{acc}}=-0.60$) | 2026-10-11 | Per-character normalization |
| **Synthetic Dot-Swap Reject** | **100.0%** | `synth_test_000.pt` | Held-Out | Native raster | DEV w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-11 | Jumped from 29.8% under frame-norm |
| **Synthetic Del/Ins Reject** | **100.0%** | `synth_test_000.pt` | Held-Out | Native raster | DEV w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-11 | Jumped from 38.5% under frame-norm |
| **Synthetic Word-Swap Reject** | **100.0%** | `synth_test_000.pt` | Held-Out | Native raster | DEV w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-11 | Perfect macro rejection |
| **w01 Baseline Overall CER** | **24.07%** | `writer_01` (100 samples) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (Greedy Decode) | 2026-10-10 | Primary development set |
| **w01 Baseline Lines CER** | **19.52%** | `writer_01` (40 lines) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (Greedy Decode) | 2026-10-10 | Baseline on line crops |
| **w01 Baseline Words CER** | **34.15%** | `writer_01` (60 words) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (Greedy Decode) | 2026-10-10 | Baseline on isolated word crops |
| **w01 Baseline Overall WER** | **64.68%** | `writer_01` (100 samples) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (Greedy Decode) | 2026-10-10 | Greedy CTC word errors |
| **w01 Full-Dict Top-1 Rank** | **48.33%** | `writer_01` (60 words) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (CTC Loss Rank) | 2026-10-10 | Ranked vs 14,872 Quran words |
| **w01 Full-Dict Median Rank** | **2.0** | `writer_01` (60 words) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (CTC Loss Rank) | 2026-10-10 | 50% of words ranked 1 or 2 |
| **w01 200-Distractor Top-1** | **83.33%** | `writer_01` (60 words) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | N/A (CTC Loss Rank) | 2026-10-11 | Sampled protocol approximation |
| **w01 True Claims Accept** | **93.7%** | `writer_01` (100 samples) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | Self DEV ($t_{\text{acc}}=-0.60$) | 2026-10-11 | Target true accept rate |
| **w01 Corrupted Claims Reject** | **95.0%** | `writer_01` (100 samples) | **In-Sample (DEV)** | 1.4x zoom, 1.8px width, pad 16 | Self DEV ($t_{\text{wrg}}=-0.95$) | 2026-10-11 | Per-character norm reject rate |
| **w02 Baseline CER (1.4x)** | 53.66% | `writer_02` (145 samples) | **Held-Out** | 1.4x zoom (PROVISIONAL) | N/A (Greedy Decode) | 2026-10-12 | **RETRACTED**: Clipping distortion |
| **w02 Lines CER (1.4x)** | 44.02% | `writer_02` (20 lines) | **Held-Out** | 1.4x zoom (PROVISIONAL) | N/A (Greedy Decode) | 2026-10-12 | **RETRACTED**: Over-magnified |
| **w02 Words CER (1.4x)** | 63.08% | `writer_02` (125 words) | **Held-Out** | 1.4x zoom (PROVISIONAL) | N/A (Greedy Decode) | 2026-10-12 | **RETRACTED**: Over-magnified |
| **w02 Baseline CER (Canonical)**| **37.24%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (Greedy Decode) | 2026-10-12 | Zero training, canonical scale |
| **w02 Lines CER (Canonical)** | **33.59%** | `writer_02` (20 lines) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (Greedy Decode) | 2026-10-12 | Zero training, canonical scale |
| **w02 Words CER (Canonical)** | **40.82%** | `writer_02` (125 words) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (Greedy Decode) | 2026-10-12 | Zero training, canonical scale |
| **w02 Overall WER (Canonical)** | **87.83%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (Greedy Decode) | 2026-10-12 | Zero training, canonical scale |
| **w02 Full-Dict Top-1 Rank** | **42.40%** | `writer_02` (125 words) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (CTC Loss Rank) | 2026-10-12 | Ranked vs 14,872 Quran words |
| **w02 Full-Dict Top-5 Rank** | **60.80%** | `writer_02` (125 words) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (CTC Loss Rank) | 2026-10-12 | Top-5 dictionary precision |
| **w02 Full-Dict Median Rank** | **2.0** | `writer_02` (125 words) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (CTC Loss Rank) | 2026-10-12 | Strong discrimination despite shift |
| **w02 200-Distractor Top-1** | **85.60%** | `writer_02` (125 words) | **Held-Out** | **Canonical 1.0x, 2.2px width** | N/A (CTC Loss Rank) | 2026-10-12 | Sampled protocol approximation |
| **w02 True Accept (Frozen w01)**| **41.3%** | `writer_02` (230 words) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{acc}}=-0.60$) | 2026-10-12 | False rejection due to zero-FT |
| **w02 Dot-Swap Reject (Frozen)**| **91.0%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-12 | 2.5% accept, 6.6% uncertain |
| **w02 Del/Ins Reject (Frozen)** | **90.3%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-12 | 4.8% accept, 4.8% uncertain |
| **w02 Word-Swap Reject (Frozen)**| **100.0%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-12 | Perfect macro rejection |
| **w02 Adj-Swap Reject (Frozen)** | **85.0%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-12 | 10.0% accept, 5.0% uncertain |
| **w02 Skip-Word Reject (Frozen)**| **85.0%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-12 | 15.0% accept, 0.0% uncertain |
| **w02 Mutashabih Reject (Frozen)**| **100.0%** | `writer_02` (145 samples) | **Held-Out** | **Canonical 1.0x, 2.2px width** | Frozen w01 ($t_{\text{wrg}}=-0.95$) | 2026-10-12 | Perfect mutashabih detection |

---

## 2. Zoom Diagnostic Grid on `writer_02`

| Zoom Factor | Overall CER | Lines CER (20) | Words CER (125) | Overall WER | Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.70x** | 35.18% | 33.59% | 36.73% | 81.74% | Diagnostic sweep |
| **0.85x** | 35.46% | 32.64% | 38.22% | 82.17% | Diagnostic sweep |
| **1.00x** | **37.24%** | **33.59%** | **40.82%** | **87.83%** | **CANONICAL STANDARD** |
| **1.15x** | 43.43% | 37.19% | 49.54% | 95.65% | Diagnostic sweep |
| **1.30x** | 48.78% | 41.37% | 56.03% | 97.39% | Diagnostic sweep |
| **1.40x** | 53.66% | 44.02% | 63.08% | 101.30% | RETRACTED (Double-count/clipping) |
