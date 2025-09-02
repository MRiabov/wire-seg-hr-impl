import argparse
import os
import pprint
import yaml
from typing import Tuple, List, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from tqdm import tqdm
import random
import torch.backends.cudnn as cudnn
import cv2

from src.wireseghr.model import WireSegHR
from src.wireseghr.model.minmax import MinMaxLuminance
from src.wireseghr.data.dataset import WireSegDataset
from src.wireseghr.model.label_downsample import downsample_label_maxpool
from src.wireseghr.data.sampler import BalancedPatchSampler
from src.wireseghr.metrics import compute_metrics


def main():
    parser = argparse.ArgumentParser(description="WireSegHR training (skeleton)")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(os.getcwd(), cfg_path)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("[WireSegHR][train] Loaded config from:", cfg_path)
    pprint.pprint(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[WireSegHR][train] Device: {device}")

    # Config
    coarse_train = int(cfg["coarse"]["train_size"])  # 512
    patch_size = int(cfg["fine"]["patch_size"])  # 768
    iters = int(cfg["optim"]["iters"])  # 40000
    batch_size = int(cfg["optim"]["batch_size"])  # 8
    base_lr = float(cfg["optim"]["lr"])  # 6e-5
    weight_decay = float(cfg["optim"]["weight_decay"])  # 0.01
    power = float(cfg["optim"]["power"])  # 1.0
    amp_flag = bool(cfg["optim"].get("amp", True))

    # Housekeeping
    seed = int(cfg.get("seed", 42))
    out_dir = cfg.get("out_dir", "runs/wireseghr")
    eval_interval = int(cfg.get("eval_interval", 500))
    ckpt_interval = int(cfg.get("ckpt_interval", 1000))
    os.makedirs(out_dir, exist_ok=True)
    set_seed(seed)

    # Dataset
    train_images = cfg["data"]["train_images"]
    train_masks = cfg["data"]["train_masks"]
    dset = WireSegDataset(train_images, train_masks, split="train")
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

    # Model
    # Channel definition: RGB(3) + MinMax(2) + cond(1) + loc(1) = 7
    pretrained_flag = bool(cfg.get("pretrained", False))
    model = WireSegHR(
        backbone=cfg["backbone"], in_channels=7, pretrained=pretrained_flag
    )
    model = model.to(device)

    # Optimizer and loss
    optim = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scaler = GradScaler("cuda", enabled=(device.type == "cuda" and amp_flag))
    ce = nn.CrossEntropyLoss()

    # Resume
    start_step = 0
    best_f1 = -1.0
    resume_path = cfg.get("resume", None)
    if resume_path and os.path.isfile(resume_path):
        print(f"[WireSegHR][train] Resuming from {resume_path}")
        start_step, best_f1 = _load_checkpoint(
            resume_path, model, optim, scaler, device
        )

    # Training loop
    model.train()
    step = start_step
    pbar = tqdm(total=iters - step, initial=0, desc="Train", ncols=100)
    while step < iters:
        optim.zero_grad(set_to_none=True)
        imgs, masks = _sample_batch_same_size(dset, batch_size)
        batch = _prepare_batch(
            imgs, masks, coarse_train, patch_size, sampler, minmax, device
        )

        with autocast(enabled=(device.type == "cuda" and amp_flag)):
            logits_coarse, cond_map = model.forward_coarse(
                batch["x_coarse"]
            )  # (B,2,Hc/4,Wc/4) and (B,1,Hc/4,Wc/4)

        # Build fine inputs: crop cond from low-res map to patch, concat with patch RGB+MinMax and loc mask
        B, _, hc4, wc4 = cond_map.shape
        x_fine = _build_fine_inputs(batch, cond_map, device)
        with autocast(enabled=(device.type == "cuda" and amp_flag)):
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
            model.eval()
            val_stats = validate(model, dset_val, coarse_train, device, amp_flag)
            print(
                f"[Val @ {step}] IoU={val_stats['iou']:.4f} F1={val_stats['f1']:.4f} P={val_stats['precision']:.4f} R={val_stats['recall']:.4f}"
            )
            # Save best
            if val_stats["f1"] > best_f1:
                best_f1 = val_stats["f1"]
                _save_checkpoint(
                    os.path.join(out_dir, "best.pt"),
                    step,
                    model,
                    optim,
                    scaler,
                    best_f1,
                )
            # Save periodic ckpt
            if ckpt_interval > 0 and (step % ckpt_interval == 0):
                _save_checkpoint(
                    os.path.join(out_dir, f"ckpt_{step}.pt"),
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
                    coarse_train,
                    device,
                    os.path.join(out_dir, f"test_vis_{step}"),
                    amp_flag,
                    max_samples=8,
                )
            model.train()

        step += 1
        pbar.update(1)

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
    loc_masks = []
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

        # Coarse input: resize on GPU, build 7-ch tensor
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
            [rgb_coarse_t, y_min_c_t, y_max_c_t, zeros_coarse, zeros_coarse], dim=0
        )  # 7xHc x Wc
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
        # Binary location mask (ones inside the patch)
        loc_masks.append(np.ones((patch_size, patch_size), dtype=np.float32))
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
        "loc_patches": loc_masks,
        "patch_yx": yx_list,
        "mask_full": masks,
    }


def _build_fine_inputs(
    batch, cond_map: torch.Tensor, device: torch.device
) -> torch.Tensor:
    # Build fine input tensor Bx7xP x P; crop cond from low-res map, upsample to P
    B = cond_map.shape[0]
    P = batch["loc_patches"][0].shape[0]
    full_h, full_w = batch["full_h"], batch["full_w"]
    hc4, wc4 = cond_map.shape[2], cond_map.shape[3]

    xs: List[torch.Tensor] = []
    for i in range(B):
        rgb = batch["rgb_patches"][i]
        ymin = batch["ymin_patches"][i]
        ymax = batch["ymax_patches"][i]
        loc = batch["loc_patches"][i]
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
        loc_t = torch.from_numpy(loc)[None, ...].to(device).float()  # 1xPxP
        x = torch.cat([rgb_t, ymin_t, ymax_t, cond_patch, loc_t], dim=0)  # 7xPxP
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
    cudnn.benchmark = False
    cudnn.deterministic = True


def _save_checkpoint(
    path: str,
    step: int,
    model: nn.Module,
    optim: torch.optim.Optimizer,
    scaler: GradScaler,
    best_f1: float,
):
    os.makedirs(os.path.dirname(path), exist_ok=True)
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
) -> Dict[str, float]:
    # Coarse-only validation: resize image to coarse_size, predict coarse logits, upsample to full and compute metrics
    model = model.to(device)
    metrics_sum = {"iou": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}
    n = 0
    for i in range(len(dset_val)):
        item = dset_val[i]
        img = item["image"].astype(np.float32) / 255.0  # HxWx3
        mask = item["mask"].astype(np.uint8)
        H, W = mask.shape
        # Build coarse input (zeros for cond+loc) on GPU
        t_img = (
            torch.from_numpy(np.transpose(img, (2, 0, 1)))
            .unsqueeze(0)
            .to(device)
            .float()
        )
        rgb_c = F.interpolate(
            t_img, size=(coarse_size, coarse_size), mode="bilinear", align_corners=False
        )[0]
        y_t = 0.299 * t_img[:, 0:1] + 0.587 * t_img[:, 1:2] + 0.114 * t_img[:, 2:3]
        y_c = F.interpolate(
            y_t, size=(coarse_size, coarse_size), mode="bilinear", align_corners=False
        )[0]
        zeros_c = torch.zeros(1, coarse_size, coarse_size, device=device)
        x_t = torch.cat([rgb_c, y_c, y_c, zeros_c, zeros_c], dim=0).unsqueeze(0)
        with autocast(enabled=(device.type == "cuda" and amp_flag)):
            logits_c, _ = model.forward_coarse(x_t)
        prob = torch.softmax(logits_c, dim=1)[:, 1:2]
        prob_up = (
            F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            .detach()
            .cpu()
            .numpy()
        )
        pred = (prob_up > 0.5).astype(np.uint8)
        m = compute_metrics(pred, mask)
        for k in metrics_sum:
            metrics_sum[k] += m[k]
        n += 1
    if n == 0:
        return {k: 0.0 for k in metrics_sum}
    return {k: v / float(n) for k, v in metrics_sum.items()}


@torch.no_grad()
def save_test_visuals(
    model: WireSegHR,
    dset_test: WireSegDataset,
    coarse_size: int,
    device: torch.device,
    out_dir: str,
    amp_flag: bool,
    max_samples: int = 8,
):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(min(max_samples, len(dset_test))):
        item = dset_test[i]
        img = item["image"].astype(np.float32) / 255.0
        H, W = img.shape[:2]
        t_img = (
            torch.from_numpy(np.transpose(img, (2, 0, 1)))
            .unsqueeze(0)
            .to(device)
            .float()
        )
        rgb_c = F.interpolate(
            t_img, size=(coarse_size, coarse_size), mode="bilinear", align_corners=False
        )[0]
        y_t = 0.299 * t_img[:, 0:1] + 0.587 * t_img[:, 1:2] + 0.114 * t_img[:, 2:3]
        y_c = F.interpolate(
            y_t, size=(coarse_size, coarse_size), mode="bilinear", align_corners=False
        )[0]
        zeros_c = torch.zeros(1, coarse_size, coarse_size, device=device)
        x_t = torch.cat([rgb_c, y_c, y_c, zeros_c, zeros_c], dim=0).unsqueeze(0)
        with autocast(enabled=(device.type == "cuda" and amp_flag)):
            logits_c, _ = model.forward_coarse(x_t)
        prob = torch.softmax(logits_c, dim=1)[:, 1:2]
        prob_up = (
            F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
            .detach()
            .cpu()
            .numpy()
        )
        pred = (prob_up > 0.5).astype(np.uint8) * 255
        # Save input and prediction
        img_bgr = (img[..., ::-1] * 255.0).astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, f"{i:03d}_input.jpg"), img_bgr)
        cv2.imwrite(os.path.join(out_dir, f"{i:03d}_pred.png"), pred)


if __name__ == "__main__":
    main()
