from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from PIL import Image, ImageDraw
import numpy as np


def _rasterize_cable_mask(
    shapes: List[dict], height: int, width: int, label: str
) -> np.ndarray:
    """Rasterize polygons with given label into a binary mask of shape (H, W), values {0,255}.

    Expects LabelMe-style annotations with shape entries containing keys:
      - label: str
      - shape_type: "polygon"
      - points: [[x,y], ...]
    """
    assert height > 0 and width > 0
    # PIL uses (W, H) for image size
    mask_img = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask_img)

    for s in shapes:
        if s.get("label") != label:
            continue
        assert s.get("shape_type") == "polygon", "Only polygon shapes are supported"
        pts = np.asarray(s.get("points"), dtype=np.float32)
        assert pts.ndim == 2 and pts.shape[1] == 2, "Invalid points array"
        # Round to nearest pixel and clip to image bounds
        pts = np.rint(pts)
        pts[:, 0] = np.clip(pts[:, 0], 0, width - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, height - 1)
        # PIL expects list of (x, y) tuples
        pts_list = [(int(p[0]), int(p[1])) for p in pts]
        draw.polygon(pts_list, outline=255, fill=255)

    mask = np.asarray(mask_img, dtype=np.uint8)
    return mask


def _convert_one(json_path: Path, out_dir: Path, label: str) -> Path | None:
    with open(json_path, "r") as f:
        data = json.load(f)

    shapes = data["shapes"]
    H = int(data["imageHeight"])  # required by given JSON
    W = int(data["imageWidth"])  # required by given JSON
    image_path = Path(data["imagePath"])  # e.g. "1_00186.jpg"
    # WireSegDataset expects numeric filename stems. Derive a numeric-only stem.
    stem_raw = image_path.stem
    out_stem = "".join([c for c in stem_raw if c.isdigit()])
    assert out_stem.isdigit() and len(out_stem) > 0, (
        f"Non-numeric stem derived from {stem_raw}"
    )

    mask = _rasterize_cable_mask(shapes, H, W, label)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{out_stem}.png"
    # Write with Pillow
    Image.fromarray(mask, mode="L").save(str(out_path))
    return out_path


def convert_ttpla_jsons_to_masks(
    input_path: str | Path,
    output_dir: str | Path,
    label: str = "cable",
    recursive: bool = True,
) -> List[Path]:
    """Convert TTPLA LabelMe JSON annotations into binary masks matching WireSegHR conventions.

    - input_path: directory containing JSONs (or a single .json file)
    - output_dir: directory where .png masks will be written
    - label: which label to rasterize (default: "cable")
    - recursive: when input_path is a directory, whether to search recursively

    Returns a list of written mask paths.
    """
    input_p = Path(input_path)
    output_p = Path(output_dir)

    if input_p.is_file():
        assert input_p.suffix.lower() == ".json", (
            f"Expected a .json file, got: {input_p}"
        )
        out = _convert_one(input_p, output_p, label)
        return [out] if out else []

    assert input_p.is_dir(), (
        f"Input path must be a directory or a .json file: {input_p}"
    )

    json_iter: Iterable[Path]
    if recursive:
        json_iter = input_p.rglob("*.json")
    else:
        json_iter = input_p.glob("*.json")

    written: List[Path] = []
    for jp in sorted(json_iter):
        w = _convert_one(jp, output_p, label)
        if w is not None:
            written.append(w)
    return written


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Convert TTPLA LabelMe JSONs to WireSegHR-style binary masks"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to a directory of JSONs or a single JSON file",
    )
    parser.add_argument(
        "--output", required=True, help="Output directory for PNG masks"
    )
    parser.add_argument(
        "--label", default="cable", help="Label to rasterize (default: cable)"
    )
    parser.add_argument(
        "--no-recursive", action="store_true", help="Do not search subdirectories"
    )
    args = parser.parse_args(argv)

    convert_ttpla_jsons_to_masks(
        args.input,
        args.output,
        label=args.label,
        recursive=(not args.no_recursive),
    )


if __name__ == "__main__":
    main()
