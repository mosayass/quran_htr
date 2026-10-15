Role: planning/review brain for a Quran handwriting verification app (Flutter app, CRNN+CTC -> ONNX, Colab training). An IDE agent writes code; you give briefs. Rules: no README, diffs only, end every reply with a 2-line note (what changed + the agent's exact next task). Reports must say RAN/NOT RUN, give set, in-sample vs held-out, preprocessing and threshold source for every number, and contain no predictions. Be skeptical: agent reports here have overclaimed repeatedly.

State:
- Model of record: Step 2b v3 best.pt (visual label order, corpus v3 with diacritics). KHATT val 8.92% / test 9.24% CER; FP32 ONNX 17 MB (~29 ms per 64x512 line, 4 threads, dev PC); INT8 dropped.
- Real tablet writers: w01 (old band) and w02 are dev; w03 and w04 are LOCKED test (never evaluated); more volunteers coming (target >=4 dev writers). Canonical band = 108 CSS px; zoom is band-relative; the 1.4x policy is retired.
- Baseline on a new writer: ~36% (w02), ~25% (w01). Anchored fine-tunes (5 seeds): A 36.3 -> 29.1, B 24.8 -> 21.3, C (personalization) 38.6 -> 31.7 anchored vs 24.7 unanchored (unanchored forgets: synthetic 1.2 -> 5-9%, KHATT 9 -> 18-20%). Learning curve N=75 (23.7%) vs Fold C (31.7%) reconciled: test sets differed (C_test n=76 held-out base 41.3% vs LC_test n=45 subset base 39.4%; C_test_only n=31 base 43.6% was in LC train pool) and train sizes differed (n=56 vs n=75).
- Verification: 3-state accept/uncertain/wrong from per-char-normalized CTC likelihood ratio; word-level errors flagged well; single-letter dot errors only ~50% at ~90% true-accept, so they map to "uncertain".
- Partition: locked-test writers + ~30% held-out prompt groups; leave-one-writer-out CV on dev writers.
- Scale & AdaBN: w02 target fill 48px is ~0.87, giving 35.93% (unclamped) and 36.40% (clip-safe) vs 37.24% canonical; w01 is 25.86% (unclamped) and 27.65% (clip-safe) vs 25.62% canonical. AdaBN dropped. Stratified post-FT CER on w02 (lowo_fold_A.pt): seen words 35.33% vs unseen 41.28% (overall 36.96%).
- Base CERs reconciled: C_test 41.25% / LC_test 39.35% (CPU) vs 38.57% / 35.48% (CUDA CuDNN FP32 on A100) are identical samples from configs/gate_baselines.json differing only by hardware convolution precision.
- Colab sweep tools: train_twofold_lowo.py upgraded with mixes {mix_0.7, mix_0.85}, --freeze_backbone, --save_checkpoints_dir, and CLC on fixed C_test (n=76); eval_weight_interpolation.py prepared.

Open issues: Run Cell 1a in Google Colab to confirm GPU s/step, then launch sweep (Item 2) and learning curve (Item 3) via Cell 1b.

