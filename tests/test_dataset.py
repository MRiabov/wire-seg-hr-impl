from pathlib import Path
from wireseghr.data.dataset import WireSegDataset


def test_real_dataset_masks_ids_1_to_5_have_positive_pixels():
    images_dir = "dataset/train/images"
    masks_dir = "dataset/train/gts"

    ds = WireSegDataset(images_dir, masks_dir)

    # Map numeric id -> dataset index by inspecting dataset's indexed pairs
    id_to_idx = {}
    for idx, (_ip, mp) in enumerate(ds._items):
        stem = Path(mp).stem
        assert stem.isdigit(), f"Non-numeric mask filename encountered: {mp}"
        id_to_idx[int(stem)] = idx

    # Check masks 1..5 specifically
    for i in range(1, 6):
        assert i in id_to_idx, f"Missing mask for id {i} in {masks_dir}"
        sample = ds[id_to_idx[i]]
        mask = sample["mask"]
        print(mask)
        assert mask.any(), f"Mask {i}.png has no positive pixels"
