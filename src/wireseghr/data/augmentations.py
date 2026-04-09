"""Lightweight image-mask augmentations for wire segmentation.

These augmentations are intentionally simple and CPU-friendly so they can be used
in the existing training loop without adding another dependency stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


def _ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    img = image.astype(np.float32)
    if img.max() <= 1.5:
        img = img * 255.0
    return np.clip(img, 0.0, 255.0).astype(np.uint8)


def _ensure_binary_mask(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8)


def _warp_affine(
    image: np.ndarray, mask: np.ndarray, angle: float, scale: float
) -> Tuple[np.ndarray, np.ndarray]:
    import cv2

    h, w = image.shape[:2]
    center = (w * 0.5, h * 0.5)
    mat = cv2.getRotationMatrix2D(center, angle, scale)
    img_aug = cv2.warpAffine(
        image,
        mat,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    mask_aug = cv2.warpAffine(
        mask,
        mat,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    return img_aug, mask_aug


def _apply_color_jitter(
    image: np.ndarray,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> np.ndarray:
    img = image.astype(np.float32)

    if brightness > 0:
        delta = np.random.uniform(-brightness, brightness) * 255.0
        img = img + delta

    if contrast > 0:
        factor = np.random.uniform(1.0 - contrast, 1.0 + contrast)
        img = (img - 127.5) * factor + 127.5

    img = np.clip(img, 0.0, 255.0).astype(np.uint8)

    if saturation > 0 or hue > 0:
        import cv2

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        if saturation > 0:
            sat_factor = np.random.uniform(1.0 - saturation, 1.0 + saturation)
            hsv[..., 1] *= sat_factor
        if hue > 0:
            hue_shift = np.random.uniform(-hue, hue) * 180.0
            hsv[..., 0] = (hsv[..., 0] + hue_shift) % 180.0
        hsv[..., 1] = np.clip(hsv[..., 1], 0.0, 255.0)
        hsv[..., 2] = np.clip(hsv[..., 2], 0.0, 255.0)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    return img


def _apply_blur_noise_and_jpeg(
    image: np.ndarray,
    blur_prob: float,
    noise_std: float,
    jpeg_prob: float,
    jpeg_quality: Tuple[int, int],
) -> np.ndarray:
    img = image

    if blur_prob > 0 and np.random.rand() < blur_prob:
        import cv2

        k = int(np.random.choice([3, 5]))
        if np.random.rand() < 0.5:
            img = cv2.GaussianBlur(img, (k, k), 0)
        else:
            img = cv2.medianBlur(img, k)

    if noise_std > 0:
        noise = np.random.normal(0.0, noise_std, size=img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)

    if jpeg_prob > 0 and np.random.rand() < jpeg_prob:
        import cv2

        q_min, q_max = jpeg_quality
        q = int(np.random.randint(q_min, q_max + 1))
        ok, enc = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

    return img


def _apply_cutout(image: np.ndarray, cutout_prob: float) -> np.ndarray:
    if cutout_prob <= 0 or np.random.rand() >= cutout_prob:
        return image
    img = image.copy()
    h, w = img.shape[:2]
    cut_w = max(8, int(np.random.uniform(0.05, 0.18) * w))
    cut_h = max(8, int(np.random.uniform(0.05, 0.18) * h))
    x0 = int(np.random.randint(0, max(1, w - cut_w + 1)))
    y0 = int(np.random.randint(0, max(1, h - cut_h + 1)))
    fill = int(np.median(img.reshape(-1, 3), axis=0).mean())
    img[y0 : y0 + cut_h, x0 : x0 + cut_w] = fill
    return img


@dataclass
class TrainAugmentConfig:
    enable: bool = True
    hflip_prob: float = 0.5
    rotation_deg: float = 8.0
    scale_min: float = 0.90
    scale_max: float = 1.10
    brightness: float = 0.10
    contrast: float = 0.10
    saturation: float = 0.08
    hue: float = 0.03
    blur_prob: float = 0.12
    noise_std: float = 3.0
    jpeg_prob: float = 0.15
    jpeg_quality_min: int = 70
    jpeg_quality_max: int = 100
    cutout_prob: float = 0.05


class TrainAugmenter:
    def __init__(self, cfg: Optional[TrainAugmentConfig] = None):
        self.cfg = cfg or TrainAugmentConfig()

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.cfg.enable:
            return _ensure_uint8_image(image), _ensure_binary_mask(mask)

        img = _ensure_uint8_image(image)
        msk = _ensure_binary_mask(mask)

        if np.random.rand() < self.cfg.hflip_prob:
            img = np.ascontiguousarray(img[:, ::-1])
            msk = np.ascontiguousarray(msk[:, ::-1])

        if self.cfg.rotation_deg > 0 or self.cfg.scale_min != 1.0 or self.cfg.scale_max != 1.0:
            angle = float(np.random.uniform(-self.cfg.rotation_deg, self.cfg.rotation_deg))
            scale = float(np.random.uniform(self.cfg.scale_min, self.cfg.scale_max))
            img, msk = _warp_affine(img, msk, angle=angle, scale=scale)

        if any(v > 0 for v in (self.cfg.brightness, self.cfg.contrast, self.cfg.saturation, self.cfg.hue)):
            img = _apply_color_jitter(
                img,
                brightness=self.cfg.brightness,
                contrast=self.cfg.contrast,
                saturation=self.cfg.saturation,
                hue=self.cfg.hue,
            )

        img = _apply_blur_noise_and_jpeg(
            img,
            blur_prob=self.cfg.blur_prob,
            noise_std=self.cfg.noise_std,
            jpeg_prob=self.cfg.jpeg_prob,
            jpeg_quality=(self.cfg.jpeg_quality_min, self.cfg.jpeg_quality_max),
        )
        img = _apply_cutout(img, self.cfg.cutout_prob)

        return np.ascontiguousarray(img), np.ascontiguousarray(msk)
