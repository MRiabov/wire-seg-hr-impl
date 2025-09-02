# 1x1 conv to collapse 2-ch coarse logits into 1-ch conditioning map
# TODO: Wire with coarse decoder outputs and proper resize/cropping.

import torch
import torch.nn as nn


class Conditioning1x1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=1, bias=True)

    def forward(self, coarse_logits: torch.Tensor) -> torch.Tensor:
        return self.conv(coarse_logits)
