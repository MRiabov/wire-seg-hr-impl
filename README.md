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
- Defaults locked: MiT-B3 encoder, patch size 768, MinMax 6×6, global+binary mask conditioning with patch-cropped global map.
