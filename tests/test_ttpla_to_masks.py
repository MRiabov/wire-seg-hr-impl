from pathlib import Path
import json
import numpy as np
from PIL import Image

from wireseghr.data.ttpla_to_masks import convert_ttpla_jsons_to_masks


def _read_dims(json_path: Path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return (
        int(data["imageHeight"]),
        int(data["imageWidth"]),
        Path(data["imagePath"]).stem,
    )


def test_convert_single_json_cable_only(tmp_path: Path):
    # Use the provided example JSON at repo root
    src_json = Path("/workspace/wire-seg-hr-impl/tests/1_00186_from_ttpla_dataset.json")
    assert src_json.exists()

    H, W, stem = _read_dims(src_json)

    out_dir = tmp_path / "masks"
    written = convert_ttpla_jsons_to_masks(src_json, out_dir, label="cable")

    assert len(written) == 1
    out_path = written[0]
    # Converter writes numeric-only stems
    expected_stem = "".join([c for c in stem if c.isdigit()])
    assert out_path.name == f"{expected_stem}.png"
    assert out_path.exists()

    mask = np.array(Image.open(out_path).convert("L"))
    assert mask is not None
    assert mask.shape == (H, W)
    assert mask.dtype == np.uint8

    # Binary with values in {0,255}
    uniq = np.unique(mask)
    assert all(int(v) in (0, 255) for v in uniq)
    assert (mask > 0).any(), "Expected some positive pixels for cable"


def test_convert_different_labels(tmp_path: Path):
    src_json = Path("/workspace/wire-seg-hr-impl/tests/1_00186_from_ttpla_dataset.json")
    assert src_json.exists()

    out_dir_cable = tmp_path / "masks_cable"
    out_dir_tower = tmp_path / "masks_tower"

    written_cable = convert_ttpla_jsons_to_masks(src_json, out_dir_cable, label="cable")
    written_tower = convert_ttpla_jsons_to_masks(
        src_json, out_dir_tower, label="tower_wooden"
    )

    mc = np.array(Image.open(written_cable[0]).convert("L"))
    mt = np.array(Image.open(written_tower[0]).convert("L"))

    # Both masks should have some positives and should not be identical
    assert (mc > 0).any()
    assert (mt > 0).any()
    assert not np.array_equal(mc, mt)
