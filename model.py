"""
model.py
CRNN + CTC architecture for Step 1 base Arabic handwriting pre-training.

Pipeline: CNN backbone (collapses height to ~1, preserves usable width
resolution) -> reshape to a (T, B, C) sequence -> BiLSTM -> Linear ->
log_softmax over the vocabulary (CTC-ready).

Two backbones are provided:

  - "vgg_lite" (default, recommended): a small custom VGG-style stack,
    the standard choice for line-level OCR (Shi et al., CRNN 2015,
    "An End-to-End Trainable Neural Network for Image-based Sequence
    Recognition"). ~1.2M conv params, trains fast, exports cleanly to
    ONNX INT8.

  - "mobilenetv3_small": torchvision's MobileNetV3-Small, adapted for
    1-channel input. Its default stride schedule downsamples width by
    32x, which would collapse a long handwritten line to only ~15-20
    time steps — too few for CTC to align a 50-100 character
    transcription (CTC requires T >= 2 * max_target_len + 1, and in
    practice wants real headroom above that minimum). We patch most of
    its stride-2 convs to (2,1) so width only drops by 4x, matching
    vgg_lite. NOTE: verify the stride-2 count against your installed
    torchvision version before trusting this blindly — print
    `_make_mobilenetv3_small_backbone()`'s modules and confirm the
    output width factor with `compute_output_seq_len` on a known input.

Both backbones downsample height by 32x with the settings below, which is
why dataset.TARGET_HEIGHT=64 maps to a final feature height of 2 — keep
dataset.py's TARGET_HEIGHT and this file's height math in sync if you
ever change either.
"""

from __future__ import annotations
from typing import List

import torch
import torch.nn as nn

WIDTH_DOWNSAMPLE_FACTOR = 4  # must match the backbone's actual stride product


class _VGGLiteBackbone(nn.Module):
    """~1.2M params. Height 64 -> 2 (collapsed to 1 by CRNN). Width /4 total."""

    def __init__(self, in_channels: int = 1):
        super().__init__()

        def conv_bn_relu(c_in, c_out, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, k, s, p, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv_bn_relu(in_channels, 32),
            conv_bn_relu(32, 32),
            nn.MaxPool2d(2, 2),                                 # H/2, W/2   -> 32, W/2

            conv_bn_relu(32, 64),
            conv_bn_relu(64, 64),
            nn.MaxPool2d(2, 2),                                 # H/4, W/4   -> 16, W/4

            conv_bn_relu(64, 128),
            conv_bn_relu(128, 128),
            nn.MaxPool2d((2, 1), (2, 1)),                        # H/8, W/4   -> 8,  W/4 (W kept)

            conv_bn_relu(128, 256),
            conv_bn_relu(256, 256),
            nn.MaxPool2d((2, 1), (2, 1)),                        # H/16, W/4  -> 4,  W/4

            conv_bn_relu(256, 256, k=(3, 3), s=1, p=(0, 1)),       # H: 4->2 (valid), W kept
        )
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)   # (B, 256, 2, W/4)


def _make_mobilenetv3_small_backbone(in_channels: int = 1) -> nn.Module:
    from torchvision.models import mobilenet_v3_small
    m = mobilenet_v3_small(weights=None)
    features = m.features

    old_stem = features[0][0]
    features[0][0] = nn.Conv2d(
        in_channels, old_stem.out_channels,
        kernel_size=old_stem.kernel_size, stride=old_stem.stride,
        padding=old_stem.padding, bias=False,
    )

    stride2_seen = 0
    for module in features.modules():
        if isinstance(module, nn.Conv2d) and module.stride == (2, 2):
            stride2_seen += 1
            if stride2_seen > 2:  # first two stay 2x2 (contributes to the 4x width drop)
                module.stride = (2, 1)

    return features  # (B, 576, H/32, W/4)


class CRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        backbone: str = "vgg_lite",
        rnn_hidden: int = 256,
        rnn_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        if backbone == "vgg_lite":
            self.backbone = _VGGLiteBackbone()
            backbone_out_ch = self.backbone.out_channels
            self.backbone_final_h = 2
        elif backbone == "mobilenetv3_small":
            self.backbone = _make_mobilenetv3_small_backbone()
            backbone_out_ch = 576
            self.backbone_final_h = 2   # 64 / 32 = 2
        else:
            raise ValueError(f"unknown backbone {backbone!r}")

        self.height_collapse = nn.Conv2d(
            backbone_out_ch, backbone_out_ch,
            kernel_size=(self.backbone_final_h, 1), stride=1, padding=0,
        )

        self.rnn = nn.LSTM(
            input_size=backbone_out_ch,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            bidirectional=True,
            batch_first=False,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(rnn_hidden * 2, vocab_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: (B, 1, H, W) -> log_probs: (T, B, vocab_size)"""
        feats = self.backbone(images)          # (B, C, H', W')
        feats = self.height_collapse(feats)      # (B, C, 1, W')
        feats = feats.squeeze(2)                 # (B, C, W')
        feats = feats.permute(2, 0, 1)             # (T=W', B, C)
        rnn_out, _ = self.rnn(feats)                # (T, B, 2*hidden)
        logits = self.classifier(rnn_out)             # (T, B, vocab)
        return logits.log_softmax(dim=2)


def compute_output_seq_len(pixel_widths: torch.Tensor) -> torch.Tensor:
    """Converts raw image pixel widths (dataset.py's input_lengths_px,
    measured BEFORE batch padding) into the CTC input_lengths the loss
    needs — i.e. the backbone's downsampled time-step count. Must be kept
    in sync with WIDTH_DOWNSAMPLE_FACTOR / whichever backbone you use."""
    return torch.div(pixel_widths, WIDTH_DOWNSAMPLE_FACTOR, rounding_mode="floor").clamp(min=1)


def greedy_ctc_decode(log_probs: torch.Tensor, blank_id: int = 0) -> List[List[int]]:
    """Minimal greedy (best-path) CTC decode, only for architecture sanity
    checks during Step 1. Production decoding — proper vocab-aware
    decode/scoring, batching edge cases, possibly beam search — is
    covered in the CTC-decoding step of this session, not here."""
    best_path = log_probs.argmax(dim=2).transpose(0, 1)  # (T,B,V) -> (B,T)
    decoded = []
    for seq in best_path:
        seq = seq.tolist()
        collapsed, prev = [], None
        for s in seq:
            if s != prev and s != blank_id:
                collapsed.append(s)
            prev = s
        decoded.append(collapsed)
    return decoded


if __name__ == "__main__":
    from vocab import Vocabulary
    v = Vocabulary()

    for backbone in ("vgg_lite", "mobilenetv3_small"):
        model = CRNN(vocab_size=len(v), backbone=backbone)
        n_params = sum(p.numel() for p in model.parameters())
        dummy = torch.randn(4, 1, 64, 512)
        out = model(dummy)
        lens = compute_output_seq_len(torch.tensor([512, 480, 300, 512]))
        print(f"[{backbone}] params={n_params/1e6:.2f}M  log_probs shape={tuple(out.shape)}  "
              f"CTC input_lengths={lens.tolist()}")
