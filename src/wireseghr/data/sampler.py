# Balanced patch sampler (>=1% wire pixels)
"""Balanced patch sampling with >= min_wire_ratio positives.

Sampling is uniform over valid top-left positions; tries a fixed number of
attempts and asserts if none meet the threshold.
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
        for _ in range(self.max_tries):
            y = np.random.randint(0, h - p + 1)
            x = np.random.randint(0, w - p + 1)
            m = mask[y : y + p, x : x + p]
            ratio = float(m.sum()) / float(p * p)
            if ratio >= self.min_wire_ratio:
                return int(y), int(x)
        raise AssertionError("Failed to sample a patch meeting min_wire_ratio")
