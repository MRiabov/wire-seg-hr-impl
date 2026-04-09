"""Loss functions for wire segmentation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.smooth = float(smooth)
        self.eps = float(eps)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.ndim != 4:
            raise ValueError(f"Expected logits BCHW, got {tuple(logits.shape)}")
        if target.ndim != 3:
            raise ValueError(f"Expected target BHW, got {tuple(target.shape)}")

        probs = torch.softmax(logits, dim=1)
        target_onehot = F.one_hot(target.long(), num_classes=logits.shape[1]).permute(0, 3, 1, 2).float()

        # Binary case: focus Dice on the foreground channel only.
        if probs.shape[1] == 2:
            probs = probs[:, 1:2]
            target_onehot = target_onehot[:, 1:2]

        dims = (0, 2, 3)
        intersection = torch.sum(probs * target_onehot, dim=dims)
        denom = torch.sum(probs, dim=dims) + torch.sum(target_onehot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denom + self.smooth + self.eps)
        return 1.0 - dice.mean()
