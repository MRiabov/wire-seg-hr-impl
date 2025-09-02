# Balanced patch sampler (>=1% wire pixels)
# TODO: Implement logic over mask to pick patches with wire ratio >= threshold.

from dataclasses import dataclass


@dataclass
class BalancedPatchSampler:
    patch_size: int = 768
    min_wire_ratio: float = 0.01

    def sample(self, image, mask):
        # TODO: sample and return top-left (y, x) of a valid patch
        return 0, 0
