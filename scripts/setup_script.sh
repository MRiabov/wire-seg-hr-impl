#!/usr/bin/env bash
set -euo pipefail
# This script downloads WireSegHR and TTPLA, converts TTPLA to masks, combines both,
# and creates an 85/5/10 train/val/test split under dataset/.

# 0) Setup env (includes gdown used by scripts/pull_ttpla.sh)
pip install uv
uv venv || true
source .venv/bin/activate
pip install uv
uv pip install -r requirements.txt
uv pip install gdown 

# 1) Pull WireSegHR dataset from Google Drive (default folder-id provided in script)
#    This writes under dataset/wireseghr_raw/ (adjust if you want another dir)
python3 scripts/pull_and_preprocess_wireseghr_dataset.py pull \
  --output-dir dataset/wireseghr_raw

# 2) Pull TTPLA dataset zip and unzip under dataset/ttpla_dataset/
# Pass OUT_DIR explicitly to avoid nested dataset/dataset/ttpla_dataset
bash scripts/pull_ttpla.sh "" "" ttpla_dataset

# 3) Convert TTPLA JSON annotations to binary masks with numeric-only filenames
#    Set these two to your actual TTPLA paths (after unzip).
TTPLA_JSON_ROOT="dataset/ttpla_dataset"     # directory containing LabelMe-style JSONs (recursively)
mkdir -p dataset/ttpla_flat/gts
python3 scripts/ttpla_to_masks.py \
  --input "$TTPLA_JSON_ROOT" \
  --output dataset/ttpla_flat/gts \
  --label cable

# 4) Flatten TTPLA images to numeric-only stems to match the masks
#    Set TTPLA_IMG_ROOT to the folder under which all TTPLA images can be found (recursively).
TTPLA_IMG_ROOT="dataset/ttpla_dataset"      # directory where the images referenced by JSONs reside (recursively)
mkdir -p dataset/ttpla_flat/images
python3 - <<'PY'
from pathlib import Path
import json, os, shutil
ttpla_json_root = Path("dataset/ttpla_dataset")
img_root = Path(os.environ.get("TTPLA_IMG_ROOT","dataset/ttpla_dataset"))
out_img = Path("dataset/ttpla_flat/images")
out_img.mkdir(parents=True, exist_ok=True)

jsons = sorted(ttpla_json_root.rglob("*.json"))
assert len(jsons) > 0, f"No JSONs under {ttpla_json_root}"
for jp in jsons:
    data = json.loads(jp.read_text())
    image_path = Path(data["imagePath"])  # e.g. "1_00186.jpg"
    stem_raw = image_path.stem
    num = "".join([c for c in stem_raw if c.isdigit()])
    assert num.isdigit() and len(num) > 0, f"Non-numeric from {stem_raw}"
    # locate the actual image file somewhere under img_root by filename
    cands = list(img_root.rglob(image_path.name))
    assert len(cands) == 1, f"Ambiguous or missing image for {image_path.name}: {cands}"
    src = cands[0]
    ext = src.suffix.lower()  # keep original .jpg/.jpeg
    dst = out_img / f"{num}{ext}"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    # Prefer hardlink for speed and space efficiency; fallback to copy
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
print(f"TTPLA flat images written to: {out_img}")
PY

# 5) Point to WireSegHR raw images/masks (adjust these to match what was downloaded in step 1)
#    After the Drive pull, inspect to find these two folders:
#    They must contain numeric-only image stems (.jpg/.jpeg) and PNG masks.
#    Example placeholders below — update them to your actual locations:
export WSHR_IMAGES="dataset/wireseghr_raw/images"
export WSHR_MASKS="dataset/wireseghr_raw/gts"

# 6) Build a combined pool (WireSegHR + TTPLA) and reindex to a single contiguous numeric ID space
mkdir -p dataset/combined_pool_fix/images dataset/combined_pool_fix/gts
python3 - <<'PY'
import os
from pathlib import Path

def index_pairs(images_dir: Path, masks_dir: Path):
    imgs = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.jpeg"))
    pairs = {}
    for ip in imgs:
        assert ip.stem.isdigit(), f"Non-numeric image name: {ip.name}"
        mp = masks_dir / f"{ip.stem}.png"
        assert mp.exists(), f"Missing mask for {ip.stem}: {mp}"
        pairs[int(ip.stem)] = (ip, mp)
    return [pairs[k] for k in sorted(pairs.keys())]

w_images = Path(os.environ["WSHR_IMAGES"])
w_masks  = Path(os.environ["WSHR_MASKS"])
t_images = Path("dataset/ttpla_flat/images")
t_masks  = Path("dataset/ttpla_flat/gts")

w_pairs = index_pairs(w_images, w_masks)
t_pairs = index_pairs(t_images, t_masks)
print("w_pairs:", len(w_pairs), "t_pairs:", len(t_pairs))

all_pairs = w_pairs + t_pairs  # deterministic order: WireSegHR first, then TTPLA
out_img = Path("dataset/combined_pool_fix/images")
out_msk = Path("dataset/combined_pool_fix/gts")
out_img.mkdir(parents=True, exist_ok=True)
out_msk.mkdir(parents=True, exist_ok=True)

# Reindex to 1..N, preserving each image's original extension
i = 1
for ip, mp in all_pairs:
    ext = ip.suffix.lower()  # .jpg or .jpeg
    dst_i = out_img / f"{i}{ext}"
    dst_m = out_msk / f"{i}.png"
    if dst_i.exists() or dst_i.is_symlink(): dst_i.unlink()
    if dst_m.exists() or dst_m.is_symlink(): dst_m.unlink()
    # Prefer hardlinks; fallback to copy if cross-device or unsupported
    try:
        os.link(ip, dst_i)
    except OSError:
        import shutil; shutil.copy2(ip, dst_i)
    try:
        os.link(mp, dst_m)
    except OSError:
        import shutil; shutil.copy2(mp, dst_m)
    i += 1

print(f"Combined pool: {i-1} pairs -> {out_img} and {out_msk}")
PY

# 7) Split the combined pool into train/val/test = 85/5/10
python3 scripts/pull_and_preprocess_wireseghr_dataset.py split_test_train_val \
  --images-dir dataset/combined_pool_fix/images \
  --masks-dir dataset/combined_pool_fix/gts \
  --out-dir dataset \
  --seed 42 \
  --link-method copy

# Done. Your config at configs/default.yaml already points to dataset/train|val|test.