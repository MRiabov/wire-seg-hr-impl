"""SegFormer MiT encoder wrapper with adjustable input channels.

Uses timm to instantiate MiT (e.g., mit_b3) and returns a list of multi-scale
features [C1, C2, C3, C4].
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import timm


class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "mit_b3",
        in_channels: int = 7,
        pretrained: bool = True,
        out_indices: Tuple[int, int, int, int] = (0, 1, 2, 3),
    ):
        super().__init__()
        self.backbone_name = backbone
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.out_indices = out_indices

        # Create MiT with features_only to obtain multi-scale feature maps.
        # in_chans allows expanded inputs (RGB + minmax + cond + loc)
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
            in_chans=in_channels,
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.encoder(x)
        # Ensure list of tensors is returned
        assert isinstance(feats, (list, tuple)) and len(feats) == len(self.out_indices)
        return list(feats)
