# WireSegHR (Segmentation Only)

This repository contains the segmentation-only implementation of the two-stage [WireSegHR model](https://arxiv.org/abs/2304.00221), training on the WireSegHR dataset plus the [TTPLA dataset](https://github.com/R3ab/ttpla_dataset).

## Quick Start

1) Get secrets necessary for fetching of the dataset: 

You'll need a GDrive service account to fetch the WireSegHR dataset using scripts in this repo. Get a GDrive key as described [in this short README](scripts/drive-viewer-key-readme.md), and put it in `/secrets/drive-json.json`

2) Run:

```bash
scripts/setup.sh
```
This installs dependencies and merges the TTPLA dataset into the WireSegHR dataset format.

3) Train and run a quick inference check:

```bash
python3 train.py --config configs/default.yaml
python3 infer.py --config configs/default.yaml --image /path/to/image.jpg
```

The default config `default.yaml` is suitable for a 24GB VRAM GPU with support for bf16 (e.g., RTX 3090/4090). 
<!-- For a quick RTX GPU setup, I recommend [vast.ai](https://cloud.vast.ai/?ref_id=162850) -->

## Project Overview
- Two-stage, global-to-local segmentation with a shared encoder and a fine decoder conditioned on the coarse stage.
- Full training loop with AMP (optional), Poly LR, periodic evaluation, checkpointing, and test visualizations (`train.py`).
- Dataset utilities under `src/wireseghr/data/` and model components under `src/wireseghr/model/`.
- Paper text and figures live in `paper-tex/` (`paper-tex/sections/` contains the Method, Results, etc.).

## Notes
- This is a segmentation-only codebase. Inpainting is out of scope here.
- Defaults locked: SegFormer MiT-B3 encoder, patch size 768, MinMax 6×6, global+binary mask conditioning with patch-cropped global map.

### Backbone Source
- HuggingFace Transformers SegFormer (e.g., `nvidia/mit-b3`). We set `num_channels` to match input channels.
- Alternative: TorchVision ResNet-50 (`backbone: resnet50`). The stem is adapted to the requested `in_channels`, and we expose features from `layer1`..`layer4` at strides 1/4, 1/8, 1/16, 1/32 with channels [256, 512, 1024, 2048].

## Dataset Convention
- Flat directories with numeric filenames; images are `.jpg`/`.jpeg`, masks are `.png`.
- Example (after split 85/5/10):
  - `dataset/train/images/1.jpg, 2.jpg, ...` and `dataset/train/gts/1.png, 2.png, ...`
  - `dataset/val/images/...` and `dataset/val/gts/...`
  - `dataset/test/images/...` and `dataset/test/gts/...`
- Masks are binary: foreground = white (255), background = black (0).
- The loader strictly enforces numeric stems and 1:1 pairing of naming and will raise on file name mismatches.

Update `configs/default.yaml` with your paths under `data.train_images`, `data.train_masks`, etc. Defaults point to `dataset/train/images`, `dataset/train/gts`, and validation to `dataset/val/...`.

## Inference

- Single image (optionally save outputs to a directory):

```bash
python3 infer.py \
  --config configs/default.yaml \
  --ckpt ckpt_5000.pt \
  --image dataset/test/images/123.jpg \
  --out outputs/infer
```

- Compute metrics for a single image (requires a GT mask):

```bash
python3 infer.py \
  --config configs/default.yaml \
  --ckpt ckpt_5000.pt \
  --image dataset/test/images/123.jpg \
  --out outputs/infer \
  --metrics \
  --mask dataset/test/gts/123.png
```

- Run inference over the entire directory with metrics (images_dir sets the image directory, masks_dir sets the ground truth mask directory):

```bash
python3 infer.py \
  --config configs/default.yaml \
  --ckpt ckpt_5000.pt \
  --images_dir dataset/test/images \
  --out outputs/infer \
  --metrics \
  --masks_dir dataset/test/gts
```

Notes:
- Predictions are saved as 0/255 PNGs. For metrics, predictions are binarized with `> 0` to match training logic.
- Masks are matched by filename stem: `images/123.jpg` ↔ `gts/123.png`.

## Benchmarking and Metrics

Benchmark mode times the model on a directory of images and reports coarse/fine/total latency statistics. When `--metrics` is provided, it also computes IoU/F1/Precision/Recall over the benchmark set (both fine and coarse outputs).

Example (uses `data.test_images` and `data.test_masks` from the config by default):

```bash
python3 infer.py \
  --config configs/default.yaml \
  --benchmark \
  --ckpt ckpt_5000.pt \
  --bench_warmup 2 \
  --bench_limit 0 \
  --bench_report_json outputs/bench_report.json \
  --metrics
```

If your ground truth directory is different from `data.test_masks`, please override it with `--bench_masks_dir`:

```bash
python3 infer.py \
  --config configs/default.yaml \
  --benchmark \
  --ckpt ckpt_5000.pt \
  --bench_warmup 2 \
  --bench_limit 0 \
  --bench_report_json outputs/bench_report.json \
  --metrics \
  --bench_masks_dir /path/to/gts
```

You will see output like:

```
[WireSegHR][bench] Results (ms):
  Coarse  avg=50.16  p50=44.48  p95=76.78
  Fine    avg=534.38  p50=419.52  p95=1187.66
  Total   avg=584.54  p50=464.73  p95=1300.07
  Target  < 1000 ms per 3000x4000 image: YES
[WireSegHR][bench][Fine]   IoU=0.6098 F1=0.7576 P=0.6418 R=0.9244
[WireSegHR][bench][Coarse] IoU=0.5315 F1=0.6941 P=0.5467 R=0.9502
```
**These metrics were obtained after 5000 iterations*

Optional: you can save a JSON timing report with `--bench_report_json`. Schema:
- `summary`
  - `avg_ms`, `p50_ms`, `p95_ms`
  - `avg_coarse_ms`, `avg_fine_ms`
  - `images`
- `per_image`: list of objects with
  - `path`, `H`, `W`, `t_coarse_ms`, `t_fine_ms`, `t_total_ms`

Utils: 
- Export your model to inference-only weights by scripts/strip_checkpoint.py