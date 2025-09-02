# Balanced patch sampler (>=1% wire pixels)
"""Balanced patch sampling with >= ``min_wire_ratio`` positives.

Sampling is uniform over valid top-left positions for up to ``max_tries``.
If no patch meets ``min_wire_ratio``, it falls back to the best observed
candidate (highest wire ratio) instead of raising.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class BalancedPatchSampler:
    patch_size: int = 768
    min_wire_ratio: float = 0.01
    max_tries: int = 200

    def sample(self, image: np.ndarray, mask: np.ndarray) -> tuple[int, int]:
        h, w = mask.shape
        p = self.patch_size
        assert h >= p and w >= p, "Image smaller than patch size"
        best_ratio = -1.0
        best_y = 0
        best_x = 0
        for _ in range(self.max_tries):
            y = np.random.randint(0, h - p + 1)
            x = np.random.randint(0, w - p + 1)
            m = mask[y : y + p, x : x + p]
            ratio = float(m.sum()) / float(p * p)
            if ratio > best_ratio:
                best_ratio = ratio
                best_y, best_x = y, x
            if ratio >= self.min_wire_ratio:
                return int(y), int(x)
        # Fallback: return best candidate even if below threshold
        return int(best_y), int(best_x)
