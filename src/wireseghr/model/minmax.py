# MinMax luminance feature computation (6x6 window)
# Implemented with OpenCV morphology (erode=min, dilate=max) using 6x6 kernel and replicate border.

from typing import Tuple

import numpy as np


class MinMaxLuminance:
    def __init__(self, kernel: int = 6):
        assert kernel == 6, "Per plan, kernel is fixed to 6x6"
        self.kernel = kernel

    def __call__(self, img_rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            img_rgb: HxWx3 uint8 or float32 in [0,255] or [0,1]
        Returns:
            (Y_min, Y_max): two HxW float32 arrays
        """
        assert img_rgb.ndim == 3 and img_rgb.shape[2] == 3
        r, g, b = img_rgb[..., 0], img_rgb[..., 1], img_rgb[..., 2]
        y = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)

        import cv2  # lazy import to avoid test-time dependency at module import
        kernel = np.ones((self.kernel, self.kernel), dtype=np.uint8)
        y_min = cv2.erode(y, kernel, borderType=cv2.BORDER_REPLICATE)
        y_max = cv2.dilate(y, kernel, borderType=cv2.BORDER_REPLICATE)
        return y_min.astype(np.float32), y_max.astype(np.float32)
