# MaxPool-based downsampling for coarse labels
"""Downsample binary masks preserving thin positives.

We use area-based resize on float32 masks followed by a >0 threshold.
This emulates block-wise max pooling: any positive in the source region
produces a positive in the target pixel.
"""

import numpy as np


def downsample_label_maxpool(mask: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Args:
        mask: HxW binary (0/1) numpy array
        out_h, out_w: target size
    Returns:
        H'xW' binary array via max-pooling-like downsample
    """
    assert mask.ndim == 2
    # Convert to float32 so area resize yields fractional averages > 0 if any positive present
    import cv2

    m = mask.astype(np.float32)
    r = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_AREA)
    out = (r > 0.0).astype(np.uint8)
    return out
