# WireSegHR Modernization Plan

This repository already has the right macro-architecture for the problem:
high-resolution wire segmentation with a coarse-to-fine router and tiled inference.
The main modernization opportunity is to replace the 2023-era representation and
training heuristics with newer foundation-model tooling while keeping the high-res
pipeline intact.

This plan is split into two buckets:

- **Definite improvements**: low-risk changes that should land directly.
- **Experiment matrix**: promising changes that need ablation before we lock them in.

Shipping constraint:

- Target experiments that can finish in roughly 5-15 minutes per run.
- Keep the 12-hour ship window in mind and prefer small, measurable steps over
  large refactors until the immediate release is stable.

## What To Keep

- Two-stage coarse-to-fine design.
- Sliding-window / tiled inference for full-resolution images.
- Binary wire segmentation as the core output.
- Sparse-positive patch sampling as the basic data strategy.

These are still good choices for wires because the task is sparse, thin, and
resolution-sensitive.

## Definite Improvements

These are the changes I would treat as direct upgrades, not experiments.

### 1. Implement real data augmentation

Current code:

- `src/wireseghr/data/transforms.py`

This module is currently a no-op placeholder. That is a clear gap.

Land now:

- random scale and crop
- horizontal flip
- small rotation / perspective jitter
- color jitter and exposure shifts
- blur, sharpening, JPEG artifacts, noise
- light synthetic occlusion

Why this is a direct improvement:

- It fills an obvious missing piece in the current pipeline.
- It should improve generalization on a small, appearance-diverse dataset.

### 2. Add a stronger baseline loss

Current code:

- `train.py`

Training is currently plain cross entropy on both branches.

Land now:

- keep CE
- add Dice or Tversky as an auxiliary term

Why this is a direct improvement:

- The task is extremely class-imbalanced.
- Dice/Tversky is a standard, low-risk upgrade for thin foreground segmentation.

### 3. Replace equal overlap averaging with weighted stitching

Current code:

- `infer.py`

The current tiled inference averages overlapping windows equally.

Land now:

- Hann / cosine window blending, or
- confidence-weighted blending over logits

Why this is a direct improvement:

- It reduces patch boundary seams.
- It is a standard fix for tiled high-resolution segmentation.

## Experiment Matrix

These items are likely beneficial, but I would not ship them without an ablation.

| Order | Experiment | What changes | Why it is worth testing | Measure |
|---|---|---|---|---|
| 1 | Encoder upgrade | Swap SegFormer/ResNet for DINOv2 or DINOv3, possibly ConvNeXt-style | Recent dense foundation models should give better thin-structure features | Wire IoU, F1, latency, VRAM |
| 2 | Conditioning upgrade | Replace the 1x1 coarse-logit projector with coarse uncertainty or coarse feature conditioning | The current conditioning path is very weak for ambiguous wire-like structures | Precision, recall, seam artifacts |
| 3 | Topology-aware supervision | Add boundary loss, centerline loss, clDice, or multi-morphological supervision | Post-2023 wire/guide-wire papers are converging on structure-aware priors | Connectedness, recall, clDice, boundary F1 |
| 4 | Routing strategy | Replace the hard 1% patch rule with uncertainty-driven or top-k patch selection | Lets compute follow the coarse branch’s uncertainty instead of uniform retries | IoU vs latency |
| 5 | Promptable or pseudo-labeled extension | Use SAM2-style prompting or pseudo-labeling to expand supervision | The field is moving toward promptable generalized segmentation | Cross-domain IoU, manual QA rate |
| 6 | Domain adaptation | Add synthetic-to-real adaptation or background-aware training | Citation graph includes domain adaptation and background-aware wire segmentation | Cross-domain IoU |
| 7 | Inpainting modernization | If the removal branch returns, test diffusion / transformer inpainting instead of LaMa alone | 2025 wire inpainting papers are already more redundancy-aware and temporal | PSNR, LPIPS, FID, subjective quality |

### Suggested experiment order

1. Encoder upgrade.
2. Conditioning upgrade.
3. Topology-aware supervision.
4. Routing strategy.
5. Promptable or pseudo-labeled extension.
6. Domain adaptation.
7. Inpainting modernization, only if the removal branch is revived.

### Notes from the citation graph

The citing papers point to the following directions:

- topology-aware losses and morphology-sensitive supervision
- prompt engineering and generalized segmentation
- multi-task wire-like perception for drone / aircraft / robotics settings
- inpainting that is redundancy-aware or hierarchical rather than plain LaMa

Relevant citations are collected in [CITATION_REVIEW.md](/home/maksym/Work/proj/wire-seg-hr-impl/CITATION_REVIEW.md).

## File-Level Change Map

Most likely files to touch:

- `src/wireseghr/data/transforms.py`
- `src/wireseghr/model/encoder.py`
- `src/wireseghr/model/model.py`
- `src/wireseghr/model/condition.py`
- `src/wireseghr/model/decoder.py`
- `src/wireseghr/data/sampler.py`
- `src/wireseghr/model/label_downsample.py`
- `train.py`
- `infer.py`
- `configs/default.yaml`

Possible new files:

- `src/wireseghr/model/losses.py`
- `src/wireseghr/model/encoders/dinov3.py`
- `src/wireseghr/data/augmentations.py`
- `src/wireseghr/model/stitching.py`
- `src/wireseghr/model/topology.py`

## Success Criteria

Good signs that the modernization is working:

- Higher wire IoU on thin and occluded structures.
- Better recall without a large precision collapse.
- Fewer broken or disconnected wire segments.
- More stable performance on panoramas and very large images.
- Less sensitivity to threshold choice at inference.

## Practical Recommendation

If you want the highest ROI, do this first:

1. Add real augmentation.
2. Add CE + Dice/Tversky.
3. Add weighted stitching.
4. Run the encoder / conditioning / topology experiments in the order above.

That keeps the repo aligned with the original paper idea, but brings the system
much closer to current segmentation practice.

## Reference Links

- DINOv3: https://github.com/facebookresearch/dinov3
- DINOv2: https://github.com/facebookresearch/dinov2
- SAM 2: https://github.com/facebookresearch/sam2
- BrushNet: https://github.com/TencentARC/BrushNet
