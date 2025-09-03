import argparse
import os
import pprint
from typing import List, Tuple, Optional
import yaml

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.amp import autocast

from src.wireseghr.model import WireSegHR


def _pad_for_minmax(kernel: int) -> Tuple[int, int, int, int]:
    # Replicate the padding logic from train.validate for even/odd kernels
    if (kernel % 2) == 0:
        return (kernel // 2 - 1, kernel // 2, kernel // 2 - 1, kernel // 2)
    else:
        return (kernel // 2, kernel // 2, kernel // 2, kernel // 2)


@torch.no_grad()
def _coarse_forward(
    model: WireSegHR,
    img_rgb: np.ndarray,
    coarse_size: int,
    minmax_enable: bool,
    minmax_kernel: int,
    device: torch.device,
    amp_flag: bool,
    amp_dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # Convert to tensor on device
    t_img = (
        torch.from_numpy(np.transpose(img_rgb, (2, 0, 1)))
        .unsqueeze(0)
        .to(device)
        .float()
    )  # 1x3xHxW
    H = img_rgb.shape[0]
    W = img_rgb.shape[1]

    rgb_c = F.interpolate(
        t_img, size=(coarse_size, coarse_size), mode="bilinear", align_corners=False
    )[0]
    y_t = 0.299 * t_img[:, 0:1] + 0.587 * t_img[:, 1:2] + 0.114 * t_img[:, 2:3]
    if minmax_enable:
        pad = _pad_for_minmax(minmax_kernel)
        y_p = F.pad(y_t, pad, mode="replicate")
        y_max_full = F.max_pool2d(y_p, kernel_size=minmax_kernel, stride=1)
        y_min_full = -F.max_pool2d(-y_p, kernel_size=minmax_kernel, stride=1)
    else:
        y_min_full = y_t
        y_max_full = y_t
    y_min_c = F.interpolate(
        y_min_full,
        size=(coarse_size, coarse_size),
        mode="bilinear",
        align_corners=False,
    )[0]
    y_max_c = F.interpolate(
        y_max_full,
        size=(coarse_size, coarse_size),
        mode="bilinear",
        align_corners=False,
    )[0]
    zeros_c = torch.zeros(1, coarse_size, coarse_size, device=device)
    x_t = torch.cat([rgb_c, y_min_c, y_max_c, zeros_c], dim=0).unsqueeze(0)

    with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_flag):
        logits_c, cond_map = model.forward_coarse(x_t)
    prob = torch.softmax(logits_c, dim=1)[:, 1:2]
    prob_up = (
        F.interpolate(prob, size=(H, W), mode="bilinear", align_corners=False)[0, 0]
        .detach()
        .cpu()
        .float()
    )  # HxW torch.Tensor on CPU
    return prob_up, cond_map, t_img, y_min_full, y_max_full


@torch.no_grad()
def _tiled_fine_forward(
    model: WireSegHR,
    t_img: torch.Tensor,  # 1x3xHxW on device
    cond_map: torch.Tensor,  # 1x1xhxw
    y_min_full: torch.Tensor,  # 1x1xHxW
    y_max_full: torch.Tensor,  # 1x1xHxW
    patch_size: int,
    overlap: int,
    fine_batch: int,
    device: torch.device,
    amp_flag: bool,
    amp_dtype,
) -> torch.Tensor:
    H = int(t_img.shape[2])
    W = int(t_img.shape[3])
    P = patch_size
    stride = P - overlap
    assert stride > 0
    assert H >= P and W >= P

    prob_sum_t = torch.zeros((H, W), device=device, dtype=torch.float32)
    weight_t = torch.zeros((H, W), device=device, dtype=torch.float32)

    hc4, wc4 = cond_map.shape[2], cond_map.shape[3]

    ys = list(range(0, H - P + 1, stride))
    if ys[-1] != (H - P):
        ys.append(H - P)
    xs = list(range(0, W - P + 1, stride))
    if xs[-1] != (W - P):
        xs.append(W - P)

    coords: List[Tuple[int, int]] = []
    for y0 in ys:
        for x0 in xs:
            coords.append((y0, x0))

    for i0 in range(0, len(coords), fine_batch):
        batch_coords = coords[i0 : i0 + fine_batch]
        xs_list: List[torch.Tensor] = []
        for y0, x0 in batch_coords:
            y1, x1 = y0 + P, x0 + P
            # Map to cond grid
            y0c = (y0 * hc4) // H
            y1c = ((y1 * hc4) + H - 1) // H
            x0c = (x0 * wc4) // W
            x1c = ((x1 * wc4) + W - 1) // W
            cond_sub = cond_map[:, :, y0c:y1c, x0c:x1c].float()
            cond_patch = F.interpolate(
                cond_sub, size=(P, P), mode="bilinear", align_corners=False
            ).squeeze(1)  # 1xPxP

            rgb_t = t_img[0, :, y0:y1, x0:x1]  # 3xPxP
            ymin_t = y_min_full[0, 0, y0:y1, x0:x1].float().unsqueeze(0)  # 1xPxP
            ymax_t = y_max_full[0, 0, y0:y1, x0:x1].float().unsqueeze(0)  # 1xPxP
            x_f = torch.cat([rgb_t, ymin_t, ymax_t, cond_patch], dim=0).unsqueeze(0)
            xs_list.append(x_f)

        x_f_batch = torch.cat(xs_list, dim=0)  # Bx6xPxP
        with autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_flag):
            logits_f = model.forward_fine(x_f_batch)
            prob_f = torch.softmax(logits_f, dim=1)[:, 1:2]
            prob_f_up = F.interpolate(
                prob_f, size=(P, P), mode="bilinear", align_corners=False
            )[:, 0, :, :]  # BxPxP

        for bi, (y0, x0) in enumerate(batch_coords):
            y1, x1 = y0 + P, x0 + P
            prob_sum_t[y0:y1, x0:x1] += prob_f_up[bi]
            weight_t[y0:y1, x0:x1] += 1.0

    prob_full = (prob_sum_t / weight_t).detach().cpu().float()
    return prob_full  # HxW torch.Tensor on CPU


def _build_model_from_cfg(cfg: dict, device: torch.device) -> WireSegHR:
    pretrained_flag = bool(cfg.get("pretrained", False))
    model = WireSegHR(
        backbone=cfg["backbone"], in_channels=6, pretrained=pretrained_flag
    )
    model = model.to(device)
    return model


@torch.no_grad()
def infer_image(
    model: WireSegHR,
    img_path: str,
    cfg: dict,
    device: torch.device,
    amp_flag: bool,
    amp_dtype,
    out_dir: Optional[str] = None,
    save_prob: bool = False,
    prob_thresh: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    assert os.path.isfile(img_path), f"Image not found: {img_path}"
    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
    assert bgr is not None, f"Failed to read {img_path}"
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    coarse_size = int(cfg["coarse"]["test_size"])
    patch_size = int(cfg["inference"]["fine_patch_size"])  # 1024 for inference
    overlap = int(cfg["fine"]["overlap"])
    minmax_enable = bool(cfg["minmax"]["enable"])
    minmax_kernel = int(cfg["minmax"]["kernel"])
    if prob_thresh is None:
        prob_thresh = float(cfg["inference"]["prob_threshold"])

    prob_c, cond_map, t_img, y_min_full, y_max_full = _coarse_forward(
        model,
        rgb,
        coarse_size,
        minmax_enable,
        minmax_kernel,
        device,
        amp_flag,
        amp_dtype,
    )

    prob_f = _tiled_fine_forward(
        model,
        t_img,
        cond_map,
        y_min_full,
        y_max_full,
        patch_size,
        overlap,
        int(cfg.get("eval", {}).get("fine_batch", 16)),
        device,
        amp_flag,
        amp_dtype,
    )

    # Threshold with torch on CPU; convert to numpy only for saving/returning
    pred_t = (prob_f > prob_thresh).to(torch.uint8) * 255  # HxW uint8 torch
    pred = pred_t.detach().cpu().numpy()

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        stem = os.path.splitext(os.path.basename(img_path))[0]
        out_mask = os.path.join(out_dir, f"{stem}_pred.png")
        cv2.imwrite(out_mask, pred)
        if save_prob:
            out_prob = os.path.join(out_dir, f"{stem}_prob.npy")
            np.save(out_prob, prob_f.detach().cpu().float().numpy())

    # Return numpy arrays for external consumers, computed via torch
    return pred, prob_f.detach().cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="WireSegHR inference")
    parser.add_argument(
        "--config", type=str, default="configs/default.yaml", help="Path to YAML config"
    )
    parser.add_argument("--image", type=str, required=False, help="Path to input image")
    parser.add_argument(
        "--images_dir",
        type=str,
        required=False,
        help="Directory with .jpg/.jpeg images",
    )
    parser.add_argument(
        "--out", type=str, default="outputs/infer", help="Directory to save predictions"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="Optional checkpoint (.pt) with model state",
    )
    parser.add_argument(
        "--save_prob", action="store_true", help="Also save probability .npy"
    )

    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(os.getcwd(), cfg_path)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("[WireSegHR][infer] Loaded config from:", cfg_path)
    pprint.pprint(cfg)

    assert (args.image is not None) ^ (args.images_dir is not None), (
        "Provide exactly one of --image or --images_dir"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = str(cfg["optim"].get("precision", "fp32")).lower()
    assert precision in ("fp32", "fp16", "bf16")
    amp_enabled = (device.type == "cuda") and (precision in ("fp16", "bf16"))
    amp_dtype = (
        torch.float16
        if precision == "fp16"
        else (torch.bfloat16 if precision == "bf16" else None)
    )

    model = _build_model_from_cfg(cfg, device)

    ckpt_path = args.ckpt if args.ckpt else cfg.get("resume", "")
    if ckpt_path:
        assert os.path.isfile(ckpt_path), f"Checkpoint not found: {ckpt_path}"
        print(f"[WireSegHR][infer] Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])
    model.eval()

    if args.image is not None:
        infer_image(
            model,
            args.image,
            cfg,
            device,
            amp_enabled,
            amp_dtype,
            out_dir=args.out,
            save_prob=args.save_prob,
        )
        print("[WireSegHR][infer] Done.")
        return

    # Directory mode
    img_dir = args.images_dir
    assert os.path.isdir(img_dir), f"Not a directory: {img_dir}"
    img_files = sorted(
        [p for p in os.listdir(img_dir) if p.lower().endswith((".jpg", ".jpeg"))]
    )
    assert len(img_files) > 0, f"No .jpg/.jpeg in {img_dir}"
    os.makedirs(args.out, exist_ok=True)
    for name in img_files:
        path = os.path.join(img_dir, name)
        infer_image(
            model,
            path,
            cfg,
            device,
            amp_enabled,
            amp_dtype,
            out_dir=args.out,
            save_prob=args.save_prob,
        )
    print("[WireSegHR][infer] Done.")


if __name__ == "__main__":
    main()
