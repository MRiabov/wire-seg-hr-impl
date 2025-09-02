# Dataset placeholder for wire segmentation
"""WireSeg dataset indexing and loading.

Pairs images in `images_dir` with masks in `masks_dir` by matching filename stems.
Mask is loaded as single-channel 0/1.
"""

from typing import Any, Dict, List

from pathlib import Path

import numpy as np
import cv2


class WireSegDataset:
    def __init__(self, images_dir: str, masks_dir: str, split: str = "train"):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.split = split
        assert self.images_dir.exists(), f"Missing images_dir: {self.images_dir}"
        assert self.masks_dir.exists(), f"Missing masks_dir: {self.masks_dir}"
        self._items: List[tuple[Path, Path]] = self._index_pairs()

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path = self._items[idx]
        img_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        assert img_bgr is not None, f"Failed to read image: {img_path}"
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        assert mask is not None, f"Failed to read mask: {mask_path}"
        mask_bin = (mask > 0).astype(np.uint8)
        return {"image": img, "mask": mask_bin, "image_path": str(img_path), "mask_path": str(mask_path)}

    def _index_pairs(self) -> List[tuple[Path, Path]]:
        exts_img = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        exts_mask = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        imgs: Dict[str, Path] = {}
        for p in sorted(self.images_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts_img:
                imgs[p.stem] = p
        masks: Dict[str, Path] = {}
        for p in sorted(self.masks_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in exts_mask:
                masks[p.stem] = p
        pairs: List[tuple[Path, Path]] = []
        for stem, ip in imgs.items():
            if stem in masks:
                pairs.append((ip, masks[stem]))
        assert len(pairs) > 0, f"No image-mask pairs found in {self.images_dir} and {self.masks_dir}"
        return pairs
