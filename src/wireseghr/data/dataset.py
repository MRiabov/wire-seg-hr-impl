# Dataset placeholder for wire segmentation
"""WireSeg dataset indexing and loading.

Pairs images in `images_dir` with masks in `masks_dir` by matching filename stems.
Mask is loaded as single-channel 0/1.
"""

from typing import Any, Dict, List, Tuple

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
        self._items: List[Tuple[Path, Path]] = self._index_pairs()
        # Precompute sizes from masks (lighter to load) and build bins
        self._sizes: List[Tuple[int, int]] = []  # (H, W) per item
        self._size_bins: Dict[Tuple[int, int], List[int]] = {}
        self._compute_size_bins()

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
        return {
            "image": img,
            "mask": mask_bin,
            "image_path": str(img_path),
            "mask_path": str(mask_path),
        }

    def _index_pairs(self) -> List[Tuple[Path, Path]]:
        # Convention: numeric filenames; images are .jpg/.jpeg; masks (gts) are .png
        img_files = sorted([p for p in self.images_dir.glob("*.jpg") if p.exists()])
        img_files += sorted([p for p in self.images_dir.glob("*.jpeg") if p.exists()])
        assert len(img_files) > 0, f"No .jpg/.jpeg images in {self.images_dir}"
        pairs: List[Tuple[Path, Path]] = []
        ids: List[int] = []
        for p in img_files:
            stem = p.stem
            assert stem.isdigit(), f"Non-numeric filename encountered: {p.name}"
            ids.append(int(stem))
        ids = sorted(ids)
        for i in ids:
            # Prefer .jpg, else .jpeg
            ip_jpg = self.images_dir / f"{i}.jpg"
            ip_jpeg = self.images_dir / f"{i}.jpeg"
            ip = ip_jpg if ip_jpg.exists() else ip_jpeg
            assert ip.exists(), f"Missing image for {i}: {ip_jpg} or {ip_jpeg}"
            mp = self.masks_dir / f"{i}.png"
            assert mp.exists(), f"Missing mask for {i}: {mp}"
            pairs.append((ip, mp))
        assert len(pairs) > 0, (
            f"No numeric pairs found in {self.images_dir} and {self.masks_dir}"
        )
        return pairs

    def _compute_size_bins(self) -> None:
        sizes: List[Tuple[int, int]] = []
        bins: Dict[Tuple[int, int], List[int]] = {}
        for idx, (_ip, mp) in enumerate(self._items):
            m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            assert m is not None, f"Failed to read mask for size scan: {mp}"
            H, W = int(m.shape[0]), int(m.shape[1])
            sizes.append((H, W))
            bins.setdefault((H, W), []).append(idx)
        self._sizes = sizes
        self._size_bins = bins

    @property
    def size_bins(self) -> Dict[Tuple[int, int], List[int]]:
        return self._size_bins
