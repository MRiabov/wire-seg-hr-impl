"""Encoder wrappers with adjustable input channels.

Supports two backbone families:
- HuggingFace Transformers SegFormer (e.g., "mit_b2")
- TorchVision ResNet-50 (use backbone "resnet50" | "resnet-50" | "resnet_50")

Both return a list of 4 multi-scale feature maps [C1, C2, C3, C4] at strides
1/4, 1/8, 1/16, 1/32 respectively.
"""

from typing import List, Tuple

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


class SegFormerEncoder(nn.Module):
    def __init__(
        self,
        backbone: str = "mit_b2",
        in_channels: int = 6,
        pretrained: bool = True,
    ):
        super().__init__()
        self.backbone_name = backbone
        self.in_channels = in_channels
        self.pretrained = pretrained

        self.hf = None
        self.resnet = None

        # SegFormer path
        if backbone.startswith("mit_") or backbone.startswith("segformer"):
            self.hf = _HFEncoderWrapper(in_channels, backbone, pretrained)
            self.feature_dims = self.hf.feature_dims
        # ResNet-50 path
        elif backbone in ("resnet50", "resnet-50", "resnet_50"):
            self.resnet = _ResNetEncoderWrapper(in_channels, pretrained)
            self.feature_dims = self.resnet.feature_dims
        else:
            raise ValueError(
                f"Unsupported backbone '{backbone}'. Use one of: mit_b[0-5], segformer*, resnet50."
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self.hf is not None:
            return self.hf(x)
        if self.resnet is not None:
            return self.resnet(x)
        raise AssertionError("No encoder instantiated")


class _ResNetEncoderWrapper(nn.Module):
    def __init__(self, in_chans: int, pretrained: bool):
        super().__init__()
        # Build base ResNet-50
        if pretrained:
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        else:
            self.model = resnet50(weights=None)

        # Adjust input stem for arbitrary channel count
        if in_chans != 3:
            old_conv = self.model.conv1
            new_conv = nn.Conv2d(
                in_chans, old_conv.out_channels, kernel_size=old_conv.kernel_size[0],
                stride=old_conv.stride[0], padding=old_conv.padding[0], bias=False
            )
            with torch.no_grad():
                if pretrained and old_conv.weight.shape[1] == 3:
                    w = old_conv.weight  # [64, 3, 7, 7]
                    if in_chans > 3:
                        w_mean = w.mean(dim=1, keepdim=True)
                        new_w = w_mean.repeat(1, in_chans, 1, 1)
                    else:
                        new_w = w[:, :in_chans, :, :]
                    new_conv.weight.copy_(new_w)
                else:
                    nn.init.kaiming_normal_(new_conv.weight, mode="fan_out", nonlinearity="relu")
            self.model.conv1 = new_conv

        self.feature_dims = [256, 512, 1024, 2048]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Stem
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)  # 1/4

        # Stages
        c1 = self.model.layer1(x)  # 1/4, 256
        c2 = self.model.layer2(c1)  # 1/8, 512
        c3 = self.model.layer3(c2)  # 1/16, 1024
        c4 = self.model.layer4(c3)  # 1/32, 2048
        return [c1, c2, c3, c4]


class _HFEncoderWrapper(nn.Module):
    def __init__(self, in_chans: int, backbone: str, pretrained: bool):
        super().__init__()
        # Lazy import to avoid hard dependency during tests if not used
        from transformers import SegformerModel, SegformerConfig

        name_map = {
            "mit_b0": "nvidia/mit-b0",
            "mit_b1": "nvidia/mit-b1",
            "mit_b2": "nvidia/mit-b2",
            "mit_b3": "nvidia/mit-b3",
            "mit_b4": "nvidia/mit-b4",
            "mit_b5": "nvidia/mit-b5",
        }
        model_id = name_map[backbone]

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
