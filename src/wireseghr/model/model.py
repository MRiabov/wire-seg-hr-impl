from typing import Tuple

import torch
import torch.nn as nn

from .encoder import SegFormerEncoder
from .decoder import CoarseDecoder, FineDecoder
from .condition import Conditioning1x1


class WireSegHR(nn.Module):
    """
    Two-stage WireSegHR model wrapper with shared encoder.

    Expects callers to prepare input channel stacks according to the plan:
    - Coarse input: RGB + MinMax (and any extra channels per config), shape (B, Cc, Hc, Wc)
    - Fine input: RGB + MinMax + cond_crop + binary_location_mask, shape (B, Cf, p, p)

    Conditioning 1x1 is applied to coarse logits to produce a single-channel map.
    """

    def __init__(
        self, backbone: str = "mit_b2", in_channels: int = 6, pretrained: bool = True
    ):
        super().__init__()
        self.encoder = SegFormerEncoder(
            backbone=backbone, in_channels=in_channels, pretrained=pretrained
        )
        # Use encoder-exposed feature dims for decoder projections
        in_chs = tuple(self.encoder.feature_dims)
        self.coarse_head = CoarseDecoder(in_chs=in_chs, embed_dim=128, num_classes=2)
        self.fine_head = FineDecoder(in_chs=in_chs, embed_dim=128, num_classes=2)
        self.cond1x1 = Conditioning1x1()

    def forward_coarse(
        self, x_coarse: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert x_coarse.dim() == 4
        feats = self.encoder(x_coarse)
        logits_coarse = self.coarse_head(feats)
        cond_map = self.cond1x1(logits_coarse)
        return logits_coarse, cond_map

    def forward_fine(self, x_fine: torch.Tensor) -> torch.Tensor:
        assert x_fine.dim() == 4
        feats = self.encoder(x_fine)
        logits_fine = self.fine_head(feats)
        return logits_fine
