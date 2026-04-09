"""Training-time transforms for wire segmentation."""

from __future__ import annotations

from dataclasses import fields
from typing import Any, Dict, Optional

from .augmentations import TrainAugmentConfig, TrainAugmenter


class TrainTransforms:
    """Lightweight augmentation wrapper used by the training loop."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = TrainAugmentConfig()
        if config:
            allowed = {f.name for f in fields(TrainAugmentConfig)}
            for key, value in config.items():
                if key in allowed:
                    setattr(cfg, key, value)
        self.cfg = cfg
        self.augmenter = TrainAugmenter(cfg)

    def __call__(self, image, mask):
        return self.augmenter(image, mask)
