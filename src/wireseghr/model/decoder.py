"""SegFormer-like multi-scale decoder heads for coarse and fine branches.

Fuse four feature maps from MiT encoder via 1x1 projections, upsample to the
highest spatial resolution (stage 0), concatenate, and predict 2-class logits.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class _SegFormerHead(nn.Module):
    def __init__(self, in_chs: List[int], embed_dim: int = 128, num_classes: int = 2):
        super().__init__()
        assert len(in_chs) == 4
        self.proj = nn.ModuleList([nn.Conv2d(c, embed_dim, kernel_size=1) for c in in_chs])
        self.fuse = _ConvBNReLU(embed_dim * 4, embed_dim, k=3, p=1)
        self.cls = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, feats: List[torch.Tensor]) -> torch.Tensor:
        assert len(feats) == 4
        h, w = feats[0].shape[2], feats[0].shape[3]
        xs = []
        for f, proj in zip(feats, self.proj):
            x = proj(f)
            if x.shape[2] != h or x.shape[3] != w:
                x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
            xs.append(x)
        x = torch.cat(xs, dim=1)
        x = self.fuse(x)
        x = self.cls(x)
        return x


class CoarseDecoder(_SegFormerHead):
    def __init__(self, in_chs: List[int] = (64, 128, 320, 512), embed_dim: int = 128, num_classes: int = 2):
        super().__init__(list(in_chs), embed_dim, num_classes)


class FineDecoder(_SegFormerHead):
    def __init__(self, in_chs: List[int] = (64, 128, 320, 512), embed_dim: int = 128, num_classes: int = 2):
        super().__init__(list(in_chs), embed_dim, num_classes)
