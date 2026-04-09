#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ -f /venv/main/bin/activate ]]; then
  source /venv/main/bin/activate
else
  echo "Missing /venv/main/bin/activate" >&2
  exit 1
fi

SERVICE_ACCOUNT_JSON="${SERVICE_ACCOUNT_JSON:-secrets/drive-json.json}"
WIRESEGHR_OUTPUT_DIR="${WIRESEGHR_OUTPUT_DIR:-dataset/wireseghr_raw}"
TTPLA_RAW_DIR="${TTPLA_RAW_DIR:-dataset/ttpla_dataset}"
TTPLA_FLAT_DIR="${TTPLA_FLAT_DIR:-dataset/ttpla_flat}"
COMBINED_DIR="${COMBINED_DIR:-dataset/combined_pool_fix}"
SEED="${SEED:-42}"
LINK_METHOD="${LINK_METHOD:-copy}"
TTPLA_LABEL="${TTPLA_LABEL:-cable}"

if [[ ! -f "$SERVICE_ACCOUNT_JSON" ]]; then
  echo "Missing service account JSON: $SERVICE_ACCOUNT_JSON" >&2
  exit 1
fi

uv pip install -r requirements.txt

# The Drive pull is incremental and skips files that already match size.
python3 scripts/pull_and_preprocess_wireseghr_dataset.py pull \
  --output-dir "$WIRESEGHR_OUTPUT_DIR" \
  --service-account "$SERVICE_ACCOUNT_JSON"

# TTPLA does not have an incremental pull helper, so only download if the raw
# JSONs are not already present.
if ! find "$TTPLA_RAW_DIR" -name '*.json' -print -quit >/dev/null 2>&1; then
  bash scripts/pull_ttpla.sh "" "" "$(basename "$TTPLA_RAW_DIR")"
fi

mkdir -p "$TTPLA_FLAT_DIR/gts" "$TTPLA_FLAT_DIR/images"
python3 scripts/ttpla_to_masks.py \
  --input "$TTPLA_RAW_DIR" \
  --output "$TTPLA_FLAT_DIR/gts" \
  --label "$TTPLA_LABEL"

export TTPLA_JSON_ROOT="$TTPLA_RAW_DIR"
export TTPLA_IMG_ROOT="$TTPLA_RAW_DIR"
export TTPLA_FLAT_IMAGES_DIR="$TTPLA_FLAT_DIR/images"
export TTPLA_FLAT_GTS_DIR="$TTPLA_FLAT_DIR/gts"
python3 - <<'PY'
from pathlib import Path
import json, os, shutil

ttpla_json_root = Path(os.environ["TTPLA_JSON_ROOT"])
img_root = Path(os.environ["TTPLA_IMG_ROOT"])
out_img = Path(os.environ["TTPLA_FLAT_IMAGES_DIR"])
out_img.mkdir(parents=True, exist_ok=True)

jsons = sorted(ttpla_json_root.rglob("*.json"))
assert len(jsons) > 0, f"No JSONs under {ttpla_json_root}"
for jp in jsons:
    data = json.loads(jp.read_text())
    image_path = Path(data["imagePath"])
    stem_raw = image_path.stem
    num = "".join([c for c in stem_raw if c.isdigit()])
    assert num.isdigit() and len(num) > 0, f"Non-numeric from {stem_raw}"
    cands = list(img_root.rglob(image_path.name))
    assert len(cands) == 1, f"Ambiguous or missing image for {image_path.name}: {cands}"
    src = cands[0]
    ext = src.suffix.lower()
    dst = out_img / f"{num}{ext}"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
print(f"TTPLA flat images written to: {out_img}")
PY

mkdir -p "$COMBINED_DIR/images" "$COMBINED_DIR/gts"
export WSHR_IMAGES="$WIRESEGHR_OUTPUT_DIR/images"
export WSHR_MASKS="$WIRESEGHR_OUTPUT_DIR/gts"
export COMBINED_IMAGES_DIR="$COMBINED_DIR/images"
export COMBINED_GTS_DIR="$COMBINED_DIR/gts"
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
w_masks = Path(os.environ["WSHR_MASKS"])
t_images = Path(os.environ["TTPLA_FLAT_IMAGES_DIR"])
t_masks = Path(os.environ["TTPLA_FLAT_GTS_DIR"])

w_pairs = index_pairs(w_images, w_masks)
t_pairs = index_pairs(t_images, t_masks)
print("w_pairs:", len(w_pairs), "t_pairs:", len(t_pairs))

all_pairs = w_pairs + t_pairs
out_img = Path(os.environ["COMBINED_IMAGES_DIR"])
out_msk = Path(os.environ["COMBINED_GTS_DIR"])
out_img.mkdir(parents=True, exist_ok=True)
out_msk.mkdir(parents=True, exist_ok=True)

i = 1
for ip, mp in all_pairs:
    ext = ip.suffix.lower()
    dst_i = out_img / f"{i}{ext}"
    dst_m = out_msk / f"{i}.png"
    if dst_i.exists() or dst_i.is_symlink():
        dst_i.unlink()
    if dst_m.exists() or dst_m.is_symlink():
        dst_m.unlink()
    try:
        os.link(ip, dst_i)
    except OSError:
        import shutil
        shutil.copy2(ip, dst_i)
    try:
        os.link(mp, dst_m)
    except OSError:
        import shutil
        shutil.copy2(mp, dst_m)
    i += 1

print(f"Combined pool: {i-1} pairs -> {out_img} and {out_msk}")
PY

python3 scripts/pull_and_preprocess_wireseghr_dataset.py split_test_train_val \
  --images-dir "$COMBINED_DIR/images" \
  --masks-dir "$COMBINED_DIR/gts" \
  --out-dir dataset \
  --seed "$SEED" \
  --link-method "$LINK_METHOD"
