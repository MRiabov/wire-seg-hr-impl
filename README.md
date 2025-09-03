# WireSegHR (Segmentation Only)

This repository contains the segmentation-only implementation plan and code skeleton for the two-stage WireSegHR model (global-to-local, shared encoder).

- Paper sources live under `paper-tex/`.
- Long-term navigation plan: `SEGMENTATION_PLAN.md`.

## Quick Start (skeleton)

1) Create a virtual environment and install requirements:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) Print configuration and verify the skeleton runs:

```bash
python src/wireseghr/train.py --config configs/default.yaml
python src/wireseghr/infer.py --config configs/default.yaml --image /path/to/image.png
```

3) Next steps:
- Implement encoder/decoders/condition/minmax/label downsampling per `SEGMENTATION_PLAN.md`.
- Implement training and inference logic, then metrics and ablations.

## Notes
- This is a segmentation-only codebase. Inpainting is out of scope here.
- Defaults locked: SegFormer MiT-B3 encoder, patch size 768, MinMax 6×6, global+binary mask conditioning with patch-cropped global map.

### Backbone Source
- HuggingFace Transformers SegFormer (e.g., `nvidia/mit-b3`). We set `num_channels` to match input channels.
- Fallback: a small internal CNN that preserves 1/4, 1/8, 1/16, 1/32 strides with channels [64, 128, 320, 512].

## Dataset Convention
- Flat directories with numeric filenames; images are `.jpg`/`.jpeg`, masks are `.png`.
- Example (after split 85/5/10):
  - `dataset/train/images/1.jpg, 2.jpg, ...` and `dataset/train/gts/1.png, 2.png, ...`
  - `dataset/val/images/...` and `dataset/val/gts/...`
  - `dataset/test/images/...` and `dataset/test/gts/...`
- Masks are binary: foreground = white (255), background = black (0).
- The loader strictly enforces numeric stems and 1:1 pairing and will assert on mismatches.

Update `configs/default.yaml` with your paths under `data.train_images`, `data.train_masks`, etc. Defaults point to `dataset/train/images`, `dataset/train/gts`, and validation to `dataset/val/...`.
