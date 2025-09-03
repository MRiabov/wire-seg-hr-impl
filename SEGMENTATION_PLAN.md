# WireSegHR Segmentation-Only Implementation Plan

This plan distills the model and pipeline described in the paper sources:
- `paper-tex/sections/method.tex`
- `paper-tex/sections/method_yq.tex`
- `paper-tex/figure_tex/pipeline.tex`
- `paper-tex/tables/{component,logit,thresholds}.tex`

Focus: segmentation only (no dataset collection or inpainting).

## Decisions and Defaults (locked)
- Backbone: SegFormer MiT-B3 via HuggingFace Transformers (shared encoder `E`), with `timm` or tiny CNN fallback.
- Fine/local patch size p: 768.
- Conditioning: global map + binary location mask by default (Table `tables/logit.tex`).
- Conditioning map scope: patch-cropped from the global map per `paper-tex/sections/method_yq.tex` (no full-image concatenation variant).
- MinMax feature augmentation: luminance min and max with a fixed 6×6 window; channels concatenated to inputs (Figure `figure_tex/pipeline.tex`, Sec. “Wire Feature Preservation” in `method_yq.tex`).
- Loss: CE on both branches, λ = 1 (`method_yq.tex`).
- α-threshold for refining windows: default 0.01 (Table `tables/thresholds.tex`).
- Coarse input size: train 512×512; test 1024×1024 (`method.tex`).
- Optim: AdamW (lr=6e-5, wd=0.01, poly schedule with power=1), ~40k iters, batch size ~8 (`method.tex`).

## Project Structure
- `configs/`
  - `default.yaml` (backbone=mit_b2, p=768, coarse_train=512, coarse_test=1024, alpha=0.01, minmax=true, kernel=6, maxpool_label=true, cond_variant=global+binary_mask)
- `src/wireseghr/`
  - `model/`
    - `encoder.py` (SegFormer MiT-B3, N_in channels expansion)
    - `decoder.py` (two MLP decoders `D_C`, `D_F` for 2 classes)
    - `condition.py` (1×1 conv to collapse coarse 2-ch logits → 1-ch cond)
    - `minmax.py` (6×6 luminance min/max filtering)
    - `label_downsample.py` (MaxPool-based coarse GT downsampling)
  - `data/`
    - `dataset.py` (image/mask loading, full-res to coarse/fine inputs)
    - `sampler.py` (balanced patch sampling with ≥1% wire pixels)
    - `transforms.py` (scaling, rotation, flip, photometric distortion)
  - `train.py` (end-to-end two-branch training)
  - `infer.py` (coarse-to-fine sliding-window inference + stitching)
  - `metrics.py` (IoU, F1, Precision, Recall)
  - `utils.py` (misc: overlap blending, seeding, logging)
- `tests/` (unit tests for channel wiring, cond alignment, stitching)
- `README.md` (segmentation-only usage)

## Model Specification
- Shared encoder `E`: SegFormer MiT-B3 (HF Transformers preferred).
  - Input channels (default): 3 (RGB) + 2 (MinMax) + 1 (global cond) + 1 (binary location) = 7.
  - For the coarse pass, the cond and location channels are zeros to keep channel count consistent (`method_yq.tex`).
  - Weight init for extra channels: copy mean of RGB conv weights or zero-init.
- Decoders: two SegFormer MLP decoders
  - `D_C`: coarse logits (2 channels) at coarse resolution.
  - `D_F`: fine logits (2 channels) at patch resolution p×p.
- Conditioning to fine branch (default):
  - Take coarse pre-softmax logits (2-ch), apply 1×1 conv → 1-ch cond map (`method.tex`).
  - Binary location mask: 1 inside current patch region (in full-image coordinates), 0 elsewhere.
  - Pass patch-aligned cond crop and binary mask as channels to the fine branch input.
- Notes:
  - We expose a config toggle to switch conditioning variant between: `global+binary_mask` (default) and `global_only` (Table `tables/logit.tex`).
  - We follow the published version (`paper-tex/sections/method_yq.tex`) and use patch-cropped conditioning exclusively; no full-image conditioning variant will be implemented.

## Data and Preprocessing
- MinMax luminance features (both branches):
  - Y = 0.299R + 0.587G + 0.114B.
  - Y_min = min filter (6×6), Y_max = max filter (6×6).
  - Concat [Y_min, Y_max] to the input image channels.
- Coarse GT label generation (MaxPool):
  - Downsample full-res mask to coarse size with max-pooling to prevent wire vanishing (`method_yq.tex`).
- Normalization: standard mean/std per backbone; apply consistently across channels (new channels can be mean=0, std=1 by convention, or min-max scaled).

### Dataset Convention (project-specific)
- Flat directories with numeric filenames; images are `.jpg`/`.jpeg`, masks are `.png`.
- Example:
  - `dataset/images/1.jpg, 2.jpg, ..., N.jpg` (or `.jpeg`)
  - `dataset/gts/1.png, 2.png, ..., N.png`
- Masks are binary: foreground = white (255), background = black (0).
- The loader (`data/dataset.py`) strictly enforces numeric stems and 1:1 pairing and will assert on mismatch.

## Training Pipeline
- Augment the full-res image (scaling, rotation, horizontal flip, photometric distortion) before constructing coarse/fine inputs (`method.tex`).
- Coarse input: downsample augmented full image to 512×512; build channels [RGB+MinMax+zeros(2)] → `E` → `D_C`.
- Fine input (per iteration select 1–k patches):
  - Sample p×p patch (p=768) with ≥1% wire pixels (`method.tex`, `method_yq.tex`).
  - Build cond map from coarse logits via 1×1 conv; crop cond to patch region.
  - Build binary location mask for patch region.
  - Build channels [RGB + MinMax + cond + location] → `E` → `D_F`.
- Losses:
  - L_glo = CE(Softmax(`D_C(E(coarse))`), G_glo), where G_glo uses MaxPool downsample.
  - L_loc = CE(Softmax(`D_F(E(fine))`), G_loc).
  - L = L_glo + λ L_loc, λ=1 (`method_yq.tex`).
- Optimization:
  - AdamW (lr=6e-5, wd=0.01), poly schedule (power=1.0), ~40k iterations, batch ≈8 (tune by memory).
  - AMP and grad accumulation recommended for stability/memory.

## Inference Pipeline
- Coarse pass:
  - Downsample to 1024×1024; predict coarse probability/logits.
- Window proposal (sliding window on full-res):
  - Tile with patch size p=768. Overlap ~128px (configurable). Compute wire fraction within each window from coarse prediction (prob>0.5).
  - If fraction ≥ α (default 0.01), run fine refinement on that patch; else skip (Table `tables/thresholds.tex`).
- Fine refinement + stitching:
  - For selected windows, build fine input with cond crop + location mask; predict logits.
  - Stitch logits into full-res canvas; average in overlaps; final argmax over classes.
- Outputs: full-res binary mask, plus optional probability map.

## Metrics and Reporting
- Implement: IoU, F1, Precision, Recall (global, and optionally per-size bins if available) matching `tables/component.tex`.
- Validate α trade-offs following `tables/thresholds.tex`.
- Ablations: MinMax on/off, MaxPool on/off, conditioning variant (Table `tables/logit.tex`).

## Configuration Surface (key)
- Backbone/weights: `mit_b2` (pretrained ImageNet-1K).
- Sizes: `p=768`, `coarse_train=512`, `coarse_test=1024`, `overlap=128`.
- Conditioning: `cond_from='coarse_logits_1x1'`, `cond_crop='patch'`.
- MinMax: `enable=true`, `kernel=6`.
- Label: `coarse_label_downsample='maxpool'`.
- Training: `iters=40000`, `batch=8`, `lr=6e-5`, `wd=0.01`, `schedule='poly'`, `power=1.0`.
- Inference: `alpha=0.01`, `prob_threshold=0.5` for wire fraction, `stitch='avg_logits'`.

## Risks / Gotchas
- Channel expansion requires careful initialization; confirm no NaNs and stable early training.
- Precise spatial alignment of cond and location mask with the patch is critical. Add assertions/tests.
- Even-sized MinMax window (6×6) requires careful padding to maintain alignment.
- Memory with p=768 and MiT-B3 may need tuning (AMP, batch size, overlap).

## Milestones
1) Skeleton + configs + metrics.
2) Encoder channel expansion + two decoders + 1×1 cond.
3) MinMax (6×6) + MaxPool label downsampling.
4) Training loop with ≥1% wire patch sampling.
5) Inference α-threshold + stitching.
6) Ablations toggles + scripts + README.
7) Tests (channel wiring, cond/mask alignment, stitching correctness).

## References (paper sources)
- `paper-tex/sections/method.tex`: Two-stage design, shared encoder, 1×1 cond, training/inference sizes, optimizer/schedule.
- `paper-tex/sections/method_yq.tex`: CE losses, λ, sliding-window with α, MinMax & MaxPool rationale.
- `paper-tex/figure_tex/pipeline.tex`: System overview; MinMax concatenation.
- `paper-tex/tables/component.tex`: Ablation of MinMax/MaxPool/coarse.
- `paper-tex/tables/logit.tex`: Conditioning variants.
- `paper-tex/tables/thresholds.tex`: α vs speed/quality.
