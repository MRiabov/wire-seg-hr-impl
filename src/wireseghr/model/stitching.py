"""Helpers for weighted patch stitching during tiled inference."""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=16)
def _base_window(patch_size: int, min_weight: float = 0.2) -> torch.Tensor:
    if patch_size <= 1:
        return torch.ones((1, 1), dtype=torch.float32)
    w = torch.hann_window(patch_size, periodic=False, dtype=torch.float32)
    w2d = torch.outer(w, w)
    w2d = min_weight + (1.0 - min_weight) * w2d
    return w2d


def get_patch_window(patch_size: int, device: torch.device, min_weight: float = 0.2) -> torch.Tensor:
    """Return a 2D patch blending window on the requested device."""
    return _base_window(patch_size, min_weight=min_weight).to(device=device)
