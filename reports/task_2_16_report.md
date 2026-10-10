# Task 2.16 Completion Report

1. **Capture Band & Geometry (Item 4)**: CSS guides at `top=117px`, `base=152px`, `bot=165px`. "48px" is ruled band height (`165-117=48px`). Median ink extent 35px gives **73%** extent ratio (within 0.7-0.8), scaling body height to ~47px at 64px.
2. **Stroke Renderer & Unit Test (Item 2)**: `scripts/render_strokes.py` (zoom, post-zoom width, rot +-3 deg, shear +-0.15, wobble, jitter, seeded). Unit test: pixel IoU = **0.9612 >= 0.95** against exported PNG (`sample_line_020`).
3. **StrokeDataset (Item 3)**: Added `StrokeDataset` in `dataset.py` (reads `strokes.json` + manifest `writer_id`/`split`, renders on-the-fly with zoom 1.0-1.8, width 1.4-2.6). Accepted by `train.py` `--mix`. Not trained.
4. **Pack Script & Writer Split (Item 5)**: `scripts/pack_captured_shards.py` updated with `configs/writer_splits.yaml`; assigns splits strictly per-writer, asserting zero writer leakage across sets.
5. **Verification v2 (Item 1 - DEV)**: Evaluated 40 lines & 60 words via CTC forced alignment per-word LLR (vs free path & vs confusable):
   - **40 Lines (DEV, vs Free)**: Whole-word swap AUC **0.940** (FRR@1%: 27.5%, @5%: 10.0%) | Missing/extra AUC **0.927** (FRR@1%: 27.5%, @5%: 27.5%) | Del/Ins AUC **0.711** (FRR@5%: 67.5%) | Dot swap AUC **0.672** (FRR@5%: 70.0%).
   - **40 Lines (DEV, vs Conf)**: Dot swap AUC **0.704** (FRR@5%: 80.0%) | Whole-word AUC **0.676** | Missing/extra AUC **0.611** | Del/Ins AUC **0.528**.
   - **60 Words (DEV, vs Free)**: Missing/extra AUC **1.000** (FRR@1%: 0.0%, @5%: 0.0%) | Whole-word swap AUC **0.991** (FRR@1%: 30.0%, @5%: 5.0%) | Del/Ins AUC **0.810** (FRR@5%: 46.7%) | Dot swap AUC **0.730** (FRR@5%: 65.0%).
   - **60 Words (DEV, vs Conf)**: Whole-word AUC **0.890** (FRR@5%: 48.3%) | Dot swap AUC **0.874** (FRR@5%: 83.3%) | Missing/extra AUC **0.782** | Del/Ins AUC **0.549**.
