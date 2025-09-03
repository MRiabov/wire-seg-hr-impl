"""SegFormer MiT encoder wrapper with adjustable input channels.

Uses timm to instantiate MiT (e.g., mit_b2) and returns a list of multi-scale
features [C1, C2, C3, C4].
"""

from typing import List, Tuple

import torch
import torch.nn as nn
import timm


class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "mit_b2",
        in_channels: int = 7,
        pretrained: bool = True,
        out_indices: Tuple[int, int, int, int] = (0, 1, 2, 3),
    ):
        super().__init__()
        self.backbone_name = backbone
        self.in_channels = in_channels
        self.pretrained = pretrained
        self.out_indices = out_indices

        # Prefer HuggingFace SegFormer for 'mit_*' backbones.
        # Otherwise try timm features_only. Always have Tiny CNN fallback.
        self.encoder = None
        self.hf = None
        prefer_hf = backbone.startswith("mit_") or backbone.startswith("segformer")
        if prefer_hf:
            # HF -> timm -> tiny
            try:
                self.hf = _HFEncoderWrapper(in_channels, backbone, pretrained)
                self.feature_dims = self.hf.feature_dims
            except Exception:
                try:
                    self.encoder = timm.create_model(
                        backbone,
                        pretrained=pretrained,
                        features_only=True,
                        out_indices=out_indices,
                        in_chans=in_channels,
                    )
                    self.feature_dims = list(self.encoder.feature_info.channels())
                except Exception:
                    self.encoder = None
                    self.fallback = _TinyEncoder(in_channels)
                    self.feature_dims = [64, 128, 320, 512]
        else:
            # timm -> HF -> tiny
            try:
                self.encoder = timm.create_model(
                    backbone,
                    pretrained=pretrained,
                    features_only=True,
                    out_indices=out_indices,
                    in_chans=in_channels,
                )
                self.feature_dims = list(self.encoder.feature_info.channels())
            except Exception:
                try:
                    self.hf = _HFEncoderWrapper(in_channels, backbone, pretrained)
                    self.feature_dims = self.hf.feature_dims
                except Exception:
                    self.encoder = None
                    self.fallback = _TinyEncoder(in_channels)
                    self.feature_dims = [64, 128, 320, 512]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.encoder is not None:
            feats = self.encoder(x)
            assert isinstance(feats, (list, tuple)) and len(feats) == len(
                self.out_indices
            )
            return list(feats)
        elif self.hf is not None:
            return self.hf(x)
        else:
            return self.fallback(x)


class _TinyEncoder(nn.Module):
    def __init__(self, in_chans: int):
        super().__init__()
        # Output strides: 4, 8, 16, 32 with channels 64,128,320,512
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=7, stride=4, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.stage1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(128, 320, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(320),
            nn.ReLU(inplace=True),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(320, 512, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        c0 = self.stem(x)  # 1/4
        c1 = self.stage1(c0)  # 1/8
        c2 = self.stage2(c1)  # 1/16
        c3 = self.stage3(c2)  # 1/32
        return [c0, c1, c2, c3]


class _HFEncoderWrapper(nn.Module):
    def __init__(self, in_chans: int, backbone: str, pretrained: bool):
        super().__init__()
        # Lazy import to avoid hard dependency during tests if not used
        from transformers import SegformerModel, SegformerConfig

        name_map = {
            "mit_b0": "nvidia/mit-b0",
            "mit_b1": "nvidia/mit-b1",
            "mit_b2": "nvidia/mit-b2",
            "mit_b2": "nvidia/mit-b3",
            "mit_b4": "nvidia/mit-b4",
            "mit_b5": "nvidia/mit-b5",
        }
        model_id = name_map.get(backbone, "nvidia/mit-b0")

        if pretrained:
            base_cfg = SegformerConfig.from_pretrained(model_id)
            base_cfg.num_channels = in_chans
            self.model = SegformerModel.from_pretrained(
                model_id, config=base_cfg, ignore_mismatched_sizes=True
            )
        else:
            cfg = SegformerConfig()  # default config (B0-like)
            cfg.num_channels = in_chans
            self.model = SegformerModel(cfg)

        # Expose channel dims per stage
        self.feature_dims = list(self.model.config.hidden_sizes)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = self.model(
            pixel_values=x, output_hidden_states=True, return_dict=True
        )
        feats = list(outputs.hidden_states)
        assert len(feats) == 4
        return feats
