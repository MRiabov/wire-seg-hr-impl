import argparse
import os
import pprint
import yaml
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.amp import GradScaler
from tqdm import tqdm
import random
import torch.backends.cudnn as cudnn
import cv2
from torch.utils.data import DataLoader
import time

from src.wireseghr.model import WireSegHR
from src.wireseghr.model.minmax import MinMaxLuminance
from src.wireseghr.data.dataset import WireSegDataset
from src.wireseghr.model.label_downsample import downsample_label_maxpool
from src.wireseghr.data.sampler import BalancedPatchSampler
from src.wireseghr.metrics import compute_metrics
from infer import _coarse_forward, _tiled_fine_forward
from pathlib import Path


class SizeBatchSampler:
    """Batch sampler that groups indices by exact (H, W) so all samples in a batch share size.

    This enables DataLoader prefetching while preserving the existing assumption
    in `_prepare_batch()` that all items in a batch have the same full resolution.
    """

    def __init__(self, dset: WireSegDataset, batch_size: int):
        self.dset = dset
        self.batch_size = batch_size
        # Precompute epoch length as the total number of full batches across bins
        bins = self.dset.size_bins
        self._len = 0
        for hw, idxs in bins.items():
            _ = hw  # unused, clarity
            self._len += len(idxs) // self.batch_size

    def __len__(self) -> int:
        return self._len

    def __iter__(self):
        # Create randomized batches per epoch across size bins
        bins = self.dset.size_bins
        keys = list(bins.keys())
        random.shuffle(keys)
        for hw in keys:
            pool = list(bins[hw])
            random.shuffle(pool)
            # Yield only full batches to keep fixed batch size and same-size assumption
            for i in range(
                0, len(pool) - (len(pool) % self.batch_size), self.batch_size
            ):
                yield pool[i : i + self.batch_size]


def collate_train(batch: List[Dict]):
    """Collate function that returns lists of numpy arrays to match existing pipeline."""
    imgs = [b["image"] for b in batch]
    masks = [b["mask"] for b in batch]
    return imgs, masks


def main():
    parser = argparse.ArgumentParser(description="WireSegHR training (skeleton)")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    cfg_path = args.config
    if not Path(cfg_path).is_absolute():
        cfg_path = str(Path.cwd() / cfg_path)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("[WireSegHR][train] Loaded config from:", cfg_path)
    pprint.pprint(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[WireSegHR][train] Device: {device}")

    # Config
    coarse_train = int(cfg["coarse"]["train_size"])  # 512
    coarse_test = int(cfg["coarse"]["test_size"])  # use higher res for eval/infer
    patch_size = int(cfg["fine"]["patch_size"])  # training fine patch size
    overlap = int(cfg["fine"]["overlap"])  # e.g., 128
    eval_patch_size = int(cfg["inference"]["fine_patch_size"])  # 1024 for eval/infer
    eval_cfg = cfg.get("eval", {})
    eval_fine_batch = int(eval_cfg.get("fine_batch", 16))
    assert eval_fine_batch >= 1
    eval_max_samples = int(eval_cfg.get("max_samples", 16))
    assert eval_max_samples >= 1
    iters = int(cfg["optim"]["iters"])  # 40000
    batch_size = int(cfg["optim"]["batch_size"])  # 8
    base_lr = float(cfg["optim"]["lr"])  # 6e-5
    weight_decay = float(cfg["optim"]["weight_decay"])  # 0.01
    power = float(cfg["optim"]["power"])  # 1.0
    precision = str(cfg["optim"].get("precision", "fp32")).lower()
    assert precision in ("fp32", "fp16", "bf16")
    # Enable AMP only when requested and on CUDA
    amp_enabled = (device.type == "cuda") and (precision in ("fp16", "bf16"))
    # Fail fast on unsupported hardware if mixed precision is requested
    if amp_enabled:
        cc_major, cc_minor = torch.cuda.get_device_capability()
        if precision == "fp16":
            assert cc_major >= 7, (
                f"fp16 requires Volta (SM 7.0)+; current SM {cc_major}.{cc_minor}"
            )
        elif precision == "bf16":
            assert cc_major >= 8, (
                f"bf16 requires Ampere (SM 8.0)+; current SM {cc_major}.{cc_minor}"
            )
    amp_dtype = (
        torch.float16
        if precision == "fp16"
        else (torch.bfloat16 if precision == "bf16" else None)
    )

    # Housekeeping
    seed = int(cfg.get("seed", 42))
    out_dir = cfg.get("out_dir", "runs/wireseghr")
    eval_interval = int(cfg["eval_interval"])
    ckpt_interval = int(cfg["ckpt_interval"])
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    # Dataset
    train_images = cfg["data"]["train_images"]
    train_masks = cfg["data"]["train_masks"]
    dset = WireSegDataset(train_images, train_masks, split="train")
    # DataLoader with prefetching and size-aware batching
    loader_cfg = cfg.get("loader", {})
    num_workers = int(loader_cfg.get("num_workers", 4))
    prefetch_factor = int(loader_cfg.get("prefetch_factor", 2))
    pin_memory = bool(loader_cfg.get("pin_memory", True))
    persistent_workers = (
        bool(loader_cfg.get("persistent_workers", True)) if num_workers > 0 else False
    )
    batch_sampler = SizeBatchSampler(dset, batch_size)
    loader_kwargs = dict(
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        collate_fn=collate_train,
    )
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    train_loader = DataLoader(dset, **loader_kwargs)
    # Validation and test
    val_images = cfg["data"].get("val_images", None)
    val_masks = cfg["data"].get("val_masks", None)
    test_images = cfg["data"].get("test_images", None)
    test_masks = cfg["data"].get("test_masks", None)
    dset_val = (
        WireSegDataset(val_images, val_masks, split="val")
        if val_images and val_masks
        else None
    )
    dset_test = (
        WireSegDataset(test_images, test_masks, split="test")
        if test_images and test_masks
        else None
    )
    sampler = BalancedPatchSampler(patch_size=patch_size, min_wire_ratio=0.01)
    minmax = (
        MinMaxLuminance(kernel=cfg["minmax"]["kernel"])
        if cfg["minmax"]["enable"]
        else None
    )

    # Inference/eval settings from config
    prob_thresh = float(cfg["inference"]["prob_threshold"])
    mm_enable = bool(cfg["minmax"]["enable"])
    mm_kernel = int(cfg["minmax"]["kernel"])

    # Model
    # Channel definition: RGB(3) + MinMax(2) + cond(1) = 6
    pretrained_flag = bool(cfg.get("pretrained", False))
    model = WireSegHR(
        backbone=cfg["backbone"], in_channels=6, pretrained=pretrained_flag
    )
    model = model.to(device)

    # Optimizer and loss
    optim = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda" and precision == "fp16"))
    ce = nn.CrossEntropyLoss()

    # Resume
    start_step = 0
    best_f1 = -1.0
    resume_path = cfg.get("resume", None)
    if resume_path and Path(resume_path).is_file():
        print(f"[WireSegHR][train] Resuming from {resume_path}")
        start_step, best_f1 = _load_checkpoint(
            resume_path, model, optim, scaler, device
        )

    # Training loop
    model.train()
    step = start_step
    pbar = tqdm(total=iters - step, initial=0, desc="Train", ncols=100)
    data_iter = iter(train_loader)
    while step < iters:
        optim.zero_grad(set_to_none=True)
        try:
            imgs, masks = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            imgs, masks = next(data_iter)
        batch = _prepare_batch(
            imgs, masks, coarse_train, patch_size, sampler, minmax, device
        )

        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits_coarse, cond_map = model.forward_coarse(
                batch["x_coarse"]
            )  # (B,2,Hc/4,Wc/4) and (B,1,Hc/4,Wc/4)

        # Build fine inputs: crop cond from low-res map to patch, concat with patch RGB+MinMax and loc mask
        B, _, hc4, wc4 = cond_map.shape
        x_fine = _build_fine_inputs(batch, cond_map, device)
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            logits_fine = model.forward_fine(x_fine)

            # Targets
            y_coarse = _build_coarse_targets(batch["mask_full"], hc4, wc4, device)
            y_fine = _build_fine_targets(
                batch["mask_patches"],
                logits_fine.shape[2],
                logits_fine.shape[3],
                device,
            )

            loss_coarse = ce(logits_coarse, y_coarse)
            loss_fine = ce(logits_fine, y_fine)
            loss = loss_coarse + loss_fine

        scaler.scale(loss).backward()
        scaler.step(optim)
        scaler.update()

        # Poly LR schedule (per optimizer step)
        lr = base_lr * ((1.0 - float(step) / float(iters)) ** power)
        for pg in optim.param_groups:
            pg["lr"] = lr

        if step % 50 == 0:
            print(f"[Iter {step}/{iters}] lr={lr:.6e}")

        # Eval & Checkpoint
        if (step % eval_interval == 0) and (dset_val is not None):
            # Free training-step tensors before eval to lower peak memory
            del (
                x_fine,
                logits_coarse,
                cond_map,
                logits_fine,
                y_coarse,
                y_fine,
                loss_coarse,
                loss_fine,
                loss,
            )
            torch.cuda.empty_cache()
            model.eval()
            print(
                f"[WireSegHR][train] Eval starting... val_size={len(dset_val)} max={eval_max_samples} patch={eval_patch_size} overlap={overlap} stride={eval_patch_size - overlap} fine_batch={eval_fine_batch}",
                flush=True,
            )
            val_stats = validate(
                model,
                dset_val,
                coarse_test,
                device,
                amp_enabled,
                amp_dtype,
                prob_thresh,
                mm_enable,
                mm_kernel,
                eval_patch_size,
                overlap,
                eval_fine_batch,
                eval_max_samples,
            )
            print(
                f"[Val @ {step}][Fine]   IoU={val_stats['iou']:.4f} F1={val_stats['f1']:.4f} P={val_stats['precision']:.4f} R={val_stats['recall']:.4f}"
            )
            print(
                f"[Val @ {step}][Coarse] IoU={val_stats['iou_coarse']:.4f} F1={val_stats['f1_coarse']:.4f} P={val_stats['precision_coarse']:.4f} R={val_stats['recall_coarse']:.4f}"
            )
            # Save best
            if val_stats["f1"] > best_f1:
                best_f1 = val_stats["f1"]
                _save_checkpoint(
                    str(Path(out_dir) / "best.pt"),
                    step,
                    model,
                    optim,
                    scaler,
                    best_f1,
                )
            # Save periodic ckpt
            if ckpt_interval > 0 and (step % ckpt_interval == 0):
                _save_checkpoint(
                    str(Path(out_dir) / f"ckpt_{step}.pt"),
                    step,
                    model,
                    optim,
                    scaler,
                    best_f1,
                )
            # Save test visualizations
            if dset_test is not None:
                save_test_visuals(
                    model,
                    dset_test,
                    coarse_test,
                    device,
                    str(Path(out_dir) / f"test_vis_{step}"),
                    amp_enabled,
                    mm_enable,
                    mm_kernel,
                    prob_thresh,
                    max_samples=8,
                )
            model.train()

        step += 1
        pbar.update(1)

    # Save a final checkpoint upon completion
    _save_checkpoint(
        str(Path(out_dir) / f"ckpt_{iters}.pt"), step, model, optim, scaler, best_f1
    )

    # Final test evaluation
    if dset_test is not None:
        torch.cuda.empty_cache()
        model.eval()
        print(
            f"[WireSegHR][train] Final test starting... test_size={len(dset_test)} patch={eval_patch_size} overlap={overlap} stride={eval_patch_size - overlap} fine_batch={eval_fine_batch}",
            flush=True,
        )
        test_stats = validate(
            model,
            dset_test,
            coarse_test,
            device,
            amp_enabled,
            amp_dtype,
            prob_thresh,
            mm_enable,
            mm_kernel,
            eval_patch_size,
            overlap,
            eval_fine_batch,
            len(dset_test),
        )
        print(
            f"[Test Final][Fine]   IoU={test_stats['iou']:.4f} F1={test_stats['f1']:.4f} P={test_stats['precision']:.4f} R={test_stats['recall']:.4f}"
        )
        print(
            f"[Test Final][Coarse] IoU={test_stats['iou_coarse']:.4f} F1={test_stats['f1_coarse']:.4f} P={test_stats['precision_coarse']:.4f} R={test_stats['recall_coarse']:.4f}"
        )
        # Save final evaluation artifacts
        final_out = Path(out_dir) / f"final_vis_{step}"
        final_out.mkdir(parents=True, exist_ok=True)
        # Dump metrics for record
        with open(final_out / "metrics.yaml", "w") as f:
            yaml.safe_dump({**test_stats, "step": step}, f, sort_keys=False)
        # Save predictions (fine + coarse) for the whole test set
        save_final_visuals(
            model,
            dset_test,
            coarse_test,
            device,
            str(final_out),
            amp_enabled,
            amp_dtype,
            prob_thresh,
            mm_enable,
            mm_kernel,
            eval_patch_size,
            overlap,
            eval_fine_batch,
        )
        model.train()

    print("[WireSegHR][train] Done.")


def _sample_batch_same_size(
    dset: WireSegDataset, batch_size: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    # Use precomputed size bins to sample a batch from a single (H, W) bin
    assert len(dset) > 0
    bins = dset.size_bins
    keys = list(bins.keys())
    random.shuffle(keys)
    chosen_key = None
    for hw in keys:
        if len(bins[hw]) >= batch_size:
            chosen_key = hw
            break
    assert chosen_key is not None, f"No size bin with at least {batch_size} samples"
    pool = bins[chosen_key]
    idxs = np.random.choice(pool, size=batch_size, replace=False)
    imgs: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for idx in idxs:
        item = dset[int(idx)]
        imgs.append(item["image"])
        masks.append(item["mask"])
    return imgs, masks


def _prepare_batch(
    imgs: List[np.ndarray],
    masks: List[np.ndarray],
    coarse_train: int,
    patch_size: int,
    sampler: BalancedPatchSampler,
    minmax: Optional[MinMaxLuminance],
    device: torch.device,
):
    B = len(imgs)
    assert B == len(masks)
    # Keep numpy versions for geometry and torch versions for model inputs

    full_h = imgs[0].shape[0]
    full_w = imgs[0].shape[1]
    for im, m in zip(imgs, masks):
        assert im.shape[0] == full_h and im.shape[1] == full_w
        assert m.shape[0] == full_h and m.shape[1] == full_w

    xs_coarse = []
    patches_rgb = []
    patches_mask = []
    patches_min = []
    patches_max = []
    yx_list: List[tuple[int, int]] = []

    for img, mask in zip(imgs, masks):
        # Float32 [0,1] on CPU, then move to GPU for heavy ops
        imgf = img.astype(np.float32) / 255.0
        t_img = (
            torch.from_numpy(np.transpose(imgf, (2, 0, 1))).unsqueeze(0).to(device)
        )  # 1x3xHxW

        # Luminance and Min/Max (6x6 replicate) on GPU
        y_t = (
            0.299 * t_img[:, 0:1] + 0.587 * t_img[:, 1:2] + 0.114 * t_img[:, 2:3]
        )  # 1x1xHxW
        if minmax is not None:
            # Asymmetric padding for even kernel to keep same HxW
            y_p = F.pad(y_t, (2, 3, 2, 3), mode="replicate")
            y_max_full = F.max_pool2d(y_p, kernel_size=6, stride=1)
            y_min_full = -F.max_pool2d(-y_p, kernel_size=6, stride=1)
        else:
            y_min_full = y_t
            y_max_full = y_t

        # Coarse input: resize on GPU, build 6-ch tensor (RGB + min + max + cond0)
        rgb_coarse_t = F.interpolate(
            t_img,
            size=(coarse_train, coarse_train),
            mode="bilinear",
            align_corners=False,
        )[0]
        y_min_c_t = F.interpolate(
            y_min_full,
            size=(coarse_train, coarse_train),
            mode="bilinear",
            align_corners=False,
        )[0]
        y_max_c_t = F.interpolate(
            y_max_full,
            size=(coarse_train, coarse_train),
            mode="bilinear",
            align_corners=False,
        )[0]
        zeros_coarse = torch.zeros(1, coarse_train, coarse_train, device=device)
        c_t = torch.cat(
            [rgb_coarse_t, y_min_c_t, y_max_c_t, zeros_coarse], dim=0
        )  # 6xHc x Wc
        xs_coarse.append(c_t)

        # Sample fine patch (CPU mask), then slice GPU min/max and transfer only patches
        y0, x0 = sampler.sample(imgf, mask)
        patch_rgb = imgf[y0 : y0 + patch_size, x0 : x0 + patch_size, :]
        patch_mask = mask[y0 : y0 + patch_size, x0 : x0 + patch_size]
        patches_rgb.append(patch_rgb)
        patches_mask.append(patch_mask)
        ymin_patch = (
            y_min_full[0, 0, y0 : y0 + patch_size, x0 : x0 + patch_size]
            .detach()
            .cpu()
            .numpy()
        )
        ymax_patch = (
            y_max_full[0, 0, y0 : y0 + patch_size, x0 : x0 + patch_size]
            .detach()
            .cpu()
            .numpy()
        )
        patches_min.append(ymin_patch)
        patches_max.append(ymax_patch)
        yx_list.append((y0, x0))

    x_coarse = torch.stack(xs_coarse, dim=0)  # already on device

    # Store numpy arrays for fine build
    return {
        "x_coarse": x_coarse,
        "full_h": full_h,
        "full_w": full_w,
        "rgb_patches": patches_rgb,
        "mask_patches": patches_mask,
        "ymin_patches": patches_min,
        "ymax_patches": patches_max,
        "patch_yx": yx_list,
        "mask_full": masks,
    }


def _build_fine_inputs(
    batch, cond_map: torch.Tensor, device: torch.device
) -> torch.Tensor:
    # Build fine input tensor Bx6xP x P; crop cond from low-res map, upsample to P
    B = cond_map.shape[0]
    P = batch["rgb_patches"][0].shape[0]
    full_h, full_w = batch["full_h"], batch["full_w"]
    hc4, wc4 = cond_map.shape[2], cond_map.shape[3]

    xs: List[torch.Tensor] = []
    for i in range(B):
        rgb = batch["rgb_patches"][i]
        ymin = batch["ymin_patches"][i]
        ymax = batch["ymax_patches"][i]
        y0, x0 = batch["patch_yx"][i]

        # Map full-res patch box to low-res cond grid, crop and upsample to P
        y1, x1 = y0 + P, x0 + P
        y0c = (y0 * hc4) // full_h
        y1c = ((y1 * hc4) + full_h - 1) // full_h
        x0c = (x0 * wc4) // full_w
        x1c = ((x1 * wc4) + full_w - 1) // full_w
        cond_sub = cond_map[i : i + 1, :, y0c:y1c, x0c:x1c].float()  # 1x1xhxw
        cond_patch = F.interpolate(
            cond_sub, size=(P, P), mode="bilinear", align_corners=False
        ).squeeze(1)  # 1xPxP

        # Convert numpy channels to torch and concat
        rgb_t = (
            torch.from_numpy(np.transpose(rgb, (2, 0, 1))).to(device).float()
        )  # 3xPxP
        ymin_t = torch.from_numpy(ymin)[None, ...].to(device).float()  # 1xPxP
        ymax_t = torch.from_numpy(ymax)[None, ...].to(device).float()  # 1xPxP
        x = torch.cat([rgb_t, ymin_t, ymax_t, cond_patch], dim=0)  # 6xPxP
        xs.append(x)
    x_fine = torch.stack(xs, dim=0)
    return x_fine


def _build_coarse_targets(
    masks: List[np.ndarray], out_h: int, out_w: int, device: torch.device
) -> torch.Tensor:
    ys: List[torch.Tensor] = []
    for m in masks:
        dm = downsample_label_maxpool(m, out_h, out_w)
        ys.append(torch.from_numpy(dm.astype(np.int64)))
    y = torch.stack(ys, dim=0).to(device)  # BxHc4xWc4 with values {0,1}
    return y


def _build_fine_targets(
    mask_patches: List[np.ndarray], out_h: int, out_w: int, device: torch.device
) -> torch.Tensor:
    ys: List[torch.Tensor] = []
    for m in mask_patches:
        dm = downsample_label_maxpool(m, out_h, out_w)
        ys.append(torch.from_numpy(dm.astype(np.int64)))
    y = torch.stack(ys, dim=0).to(device)  # BxHf4xWf4 with values {0,1}
    return y


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # cudnn.benchmark = False
    # cudnn.deterministic = True
    cudnn.benchmark = True
    cudnn.deterministic = False


def _save_checkpoint(
    path: str,
    step: int,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
    best_f1: float,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {
        "step": step,
        "model": model.state_dict(),
        "optim": optim.state_dict(),
        "scaler": scaler.state_dict(),
        "best_f1": best_f1,
    }
    torch.save(state, path)
    print(f"[WireSegHR][train] Saved checkpoint: {path}")


def _load_checkpoint(
    path: str,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
) -> Tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optim.load_state_dict(ckpt["optim"])
    try:
        scaler.load_state_dict(ckpt["scaler"])  # may not exist
    except Exception:
        pass
    step = int(ckpt.get("step", 0))
    best_f1 = float(ckpt.get("best_f1", -1.0))
    return step, best_f1


@torch.no_grad()
def validate(
    model: WireSegHR,
    dset_val: WireSegDataset,
    coarse_size: int,
    device: torch.device,
    amp_flag: bool,
    amp_dtype,
    prob_thresh: float,
    minmax_enable: bool,
    minmax_kernel: int,
    fine_patch_size: int,
    fine_overlap: int,
    fine_batch: int,
    max_images: int,
) -> Dict[str, float]:
    # Coarse-only validation: resize image to coarse_size, predict coarse logits, upsample to full and compute metrics
    model = model.to(device)
    metrics_sum = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    coarse_sum = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    n = 0
    t0 = time.time()
    total_tiles = 0
    target_n = min(len(dset_val), max_images)
    idxs = random.sample(range(len(dset_val)), k=target_n)
    print(
        f"[Eval] Started: N={target_n}/{len(dset_val)} coarse={coarse_size} patch={fine_patch_size} overlap={fine_overlap} stride={fine_patch_size - fine_overlap} fine_batch={fine_batch}",
        flush=True,
    )
    for j, i in enumerate(idxs):
        if (j % 2) == 0:
            print(f"[Eval] Running... {j}/{target_n}", flush=True)
        item = dset_val[i]
        img = item["image"].astype(np.float32) / 255.0  # HxWx3
        mask = item["mask"].astype(np.uint8)
        H, W = mask.shape
        # Reuse inference coarse pass
        prob_up, cond_map, t_img, y_min_full, y_max_full = _coarse_forward(
            model,
            img,
            coarse_size,
            minmax_enable,
            int(minmax_kernel),
            device,
            amp_flag,
            amp_dtype,
        )
        # Coarse metrics
        pred_coarse = (prob_up > prob_thresh).to(torch.uint8).cpu().numpy()
        m_c = compute_metrics(pred_coarse, mask)
        for k in coarse_sum:
            coarse_sum[k] += m_c[k]

        # Fine-stage via helper (batched and stitched)
        prob_full = _tiled_fine_forward(
            model,
            t_img,
            cond_map,
            y_min_full,
            y_max_full,
            int(fine_patch_size),
            int(fine_overlap),
            int(fine_batch),
            device,
            amp_flag,
            amp_dtype,
        )
        # Track tiles for throughput parity
        P = int(fine_patch_size)
        stride = P - int(fine_overlap)
        ys = list(range(0, H - P + 1, stride))
        if ys[-1] != (H - P):
            ys.append(H - P)
        xs = list(range(0, W - P + 1, stride))
        if xs[-1] != (W - P):
            xs.append(W - P)
        total_tiles += len(ys) * len(xs)
        pred_fine = (prob_full > prob_thresh).to(torch.uint8).cpu().numpy()
        m_f = compute_metrics(pred_fine, mask)
        for k in metrics_sum:
            metrics_sum[k] += m_f[k]
        n += 1
    if n > 0:
        for k in metrics_sum:
            metrics_sum[k] /= n
        for k in coarse_sum:
            coarse_sum[k] /= n
    dt = time.time() - t0
    tp_img = (n / dt) if dt > 0 else 0.0
    tp_tile = (total_tiles / dt) if dt > 0 else 0.0
    print(
        f"[Eval] Done in {dt:.2f}s | imgs={n}, tiles={total_tiles}, imgs/s={tp_img:.2f}, tiles/s={tp_tile:.2f}",
        flush=True,
    )
    out = {k: v for k, v in metrics_sum.items()}
    out.update(
        {
            "iou_coarse": coarse_sum["iou"],
            "f1_coarse": coarse_sum["f1"],
            "precision_coarse": coarse_sum["precision"],
            "recall_coarse": coarse_sum["recall"],
        }
    )
    return out


@torch.no_grad()
def save_test_visuals(
    model: WireSegHR,
    dset_test: WireSegDataset,
    coarse_size: int,
    device: torch.device,
    out_dir: str,
    amp_flag: bool,
    minmax_enable: bool,
    minmax_kernel: int,
    prob_thresh: float,
    max_samples: int = 8,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i in range(min(max_samples, len(dset_test))):
        item = dset_test[i]
        img = item["image"].astype(np.float32) / 255.0
        H, W = img.shape[:2]
        prob_up, _cond_map, _t_img, _ymin, _ymax = _coarse_forward(
            model,
            img,
            int(coarse_size),
            bool(minmax_enable),
            int(minmax_kernel),
            device,
            bool(amp_flag),
            None,
        )
        pred = ((prob_up > prob_thresh).to(torch.uint8) * 255).cpu().numpy()
        # Save input and prediction
        img_bgr = (img[..., ::-1] * 255.0).astype(np.uint8)
        cv2.imwrite(str(Path(out_dir) / f"{i:03d}_input.jpg"), img_bgr)
        cv2.imwrite(str(Path(out_dir) / f"{i:03d}_pred.png"), pred)


@torch.no_grad()
def save_final_visuals(
    model: WireSegHR,
    dset_test: WireSegDataset,
    coarse_size: int,
    device: torch.device,
    out_dir: str,
    amp_flag: bool,
    amp_dtype,
    prob_thresh: float,
    minmax_enable: bool,
    minmax_kernel: int,
    fine_patch_size: int,
    fine_overlap: int,
    fine_batch: int,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for i in range(len(dset_test)):
        item = dset_test[i]
        img = item["image"].astype(np.float32) / 255.0
        H, W = img.shape[:2]
        # Coarse pass
        prob_up, cond_map, t_img, y_min_full, y_max_full = _coarse_forward(
            model,
            img,
            int(coarse_size),
            bool(minmax_enable),
            int(minmax_kernel),
            device,
            bool(amp_flag),
            amp_dtype,
        )
        pred_coarse = ((prob_up > prob_thresh).to(torch.uint8) * 255).cpu().numpy()
        # Fine pass (tiled)
        prob_full = _tiled_fine_forward(
            model,
            t_img,
            cond_map,
            y_min_full,
            y_max_full,
            int(fine_patch_size),
            int(fine_overlap),
            int(fine_batch),
            device,
            bool(amp_flag),
            amp_dtype,
        )
        pred_fine = ((prob_full > prob_thresh).to(torch.uint8) * 255).cpu().numpy()
        # Save input and predictions
        img_bgr = (img[..., ::-1] * 255.0).astype(np.uint8)
        base = f"{i:03d}"
        cv2.imwrite(str(Path(out_dir) / f"{base}_input.jpg"), img_bgr)
        cv2.imwrite(str(Path(out_dir) / f"{base}_coarse_pred.png"), pred_coarse)
        cv2.imwrite(str(Path(out_dir) / f"{base}_fine_pred.png"), pred_fine)


if __name__ == "__main__":
    main()
