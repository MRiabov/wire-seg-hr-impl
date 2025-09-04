import argparse
import os
import pprint
import time
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import cv2

# TensorRT + CUDA
try:
    import tensorrt as trt
    import pycuda.autoinit  # noqa: F401  # initializes CUDA driver context
    import pycuda.driver as cuda
except Exception as e:  # pragma: no cover
    raise RuntimeError(
        "TensorRT runner requires 'tensorrt' and 'pycuda' Python packages and a valid CUDA/TensorRT install"
    ) from e

import yaml
from pathlib import Path


# ---- Utility: TRT engine wrapper ----
class TrtEngine:
    def __init__(self, engine_path: str):
        assert Path(engine_path).is_file(), f"Engine not found: {engine_path}"
        logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, 'rb') as f, trt.Runtime(logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        assert self.engine is not None, f"Failed to load engine: {engine_path}"
        self.context = self.engine.create_execution_context()
        # Default to profile 0
        try:
            self.context.active_optimization_profile = 0
        except Exception:
            pass
        self.stream = cuda.Stream()
        self.bindings: List[int] = [0] * self.engine.num_bindings
        self.host_mem: Dict[int, Any] = {}
        self.device_mem: Dict[int, Any] = {}

    def _allocate_binding(self, idx: int, shape: Tuple[int, ...]):
        dtype = trt.nptype(self.engine.get_binding_dtype(idx))
        nbytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        if idx in self.device_mem:
            # Reuse if same size; else reallocate
            old = self.device_mem[idx]
            if old.size >= nbytes:
                self.host_mem[idx] = np.empty(shape, dtype=dtype)
                return
            # free old and reallocate
            del old
        self.host_mem[idx] = np.empty(shape, dtype=dtype)
        self.device_mem[idx] = cuda.mem_alloc(nbytes)

    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        # Map names -> indices
        name_to_idx = {self.engine.get_binding_name(i): i for i in range(self.engine.num_bindings)}
        # Set input shapes (handle dynamic batch if present)
        for name, arr in inputs.items():
            idx = name_to_idx[name]
            assert self.engine.binding_is_input(idx)
            # Set shape if dynamic
            shape = tuple(arr.shape)
            try:
                self.context.set_binding_shape(idx, shape)
            except Exception:
                # Static shape engines won't allow setting; assert it matches
                eshape = tuple(self.engine.get_binding_shape(idx))
                assert eshape == shape, f"Static engine expects {eshape}, got {shape} for input {name}"
            self._allocate_binding(idx, shape)

        # Allocate outputs for resolved shapes
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                shape = tuple(self.context.get_binding_shape(i))
                assert all(s > 0 for s in shape), f"Unresolved output shape at binding {i}: {shape}"
                self._allocate_binding(i, shape)

        # Copy inputs H2D
        for name, arr in inputs.items():
            idx = name_to_idx[name]
            h_arr = self.host_mem[idx]
            assert h_arr.shape == arr.shape and h_arr.dtype == arr.dtype
            cuda.memcpy_htod_async(self.device_mem[idx], arr, self.stream)
            self.bindings[idx] = int(self.device_mem[idx])

        # Set output bindings
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                self.bindings[i] = int(self.device_mem[i])

        # Execute
        self.context.execute_async_v2(self.bindings, self.stream.handle)

        # D2H outputs
        outputs: Dict[str, np.ndarray] = {}
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                name = self.engine.get_binding_name(i)
                h_arr = self.host_mem[i]
                cuda.memcpy_dtoh_async(h_arr, self.device_mem[i], self.stream)
                outputs[name] = h_arr

        self.stream.synchronize()
        # Return copies to detach from internal buffers
        return {k: np.array(v) for k, v in outputs.items()}


# ---- Pre/post processing consistent with infer.py ----

def _pad_for_minmax(kernel: int) -> Tuple[int, int, int, int]:
    if (kernel % 2) == 0:
        return (kernel // 2 - 1, kernel // 2, kernel // 2 - 1, kernel // 2)
    else:
        return (kernel // 2, kernel // 2, kernel // 2, kernel // 2)


def _build_6ch_coarse(rgb: np.ndarray, coarse_size: int, minmax_enable: bool, minmax_kernel: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # rgb: HxWx3 float32 [0,1]
    H, W = int(rgb.shape[0]), int(rgb.shape[1])
    # To match training/minmax in torch, we replicate exact pad + pool logic via torch CPU
    import torch
    import torch.nn.functional as F

    t_img = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).float()  # 1x3xHxW
    y_t = 0.299 * t_img[:, 0:1] + 0.587 * t_img[:, 1:2] + 0.114 * t_img[:, 2:3]
    if minmax_enable:
        pad = _pad_for_minmax(minmax_kernel)
        y_p = F.pad(y_t, pad, mode="replicate")
        y_max_full = F.max_pool2d(y_p, kernel_size=minmax_kernel, stride=1)
        y_min_full = -F.max_pool2d(-y_p, kernel_size=minmax_kernel, stride=1)
    else:
        y_min_full = y_t
        y_max_full = y_t

    # Resize for coarse
    rgb_c = cv2.resize(rgb, (coarse_size, coarse_size), interpolation=cv2.INTER_LINEAR)
    y_min_c = cv2.resize(y_min_full[0, 0].numpy(), (coarse_size, coarse_size), interpolation=cv2.INTER_LINEAR)
    y_max_c = cv2.resize(y_max_full[0, 0].numpy(), (coarse_size, coarse_size), interpolation=cv2.INTER_LINEAR)

    zeros_c = np.zeros((coarse_size, coarse_size), dtype=np.float32)
    x6 = np.stack([
        rgb_c[:, :, 0], rgb_c[:, :, 1], rgb_c[:, :, 2], y_min_c, y_max_c, zeros_c
    ], axis=0)  # 6xHc x Wc
    return x6.astype(np.float32), y_min_full[0, 0].numpy().astype(np.float32), y_max_full[0, 0].numpy().astype(np.float32), t_img.numpy().astype(np.float32)


def _softmax_channel(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x_max = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - x_max)
    return e / np.sum(e, axis=axis, keepdims=True)


def _tiled_fine_trt(
    fine: TrtEngine,
    t_img: np.ndarray,          # 1x3xHxW float32
    cond_map: np.ndarray,       # 1x1xhxw float32
    y_min_full: np.ndarray,     # HxW float32
    y_max_full: np.ndarray,     # HxW float32
    patch_size: int,
    overlap: int,
    fine_batch: int,
) -> np.ndarray:
    H, W = int(t_img.shape[2]), int(t_img.shape[3])
    P = patch_size
    stride = P - overlap
    assert stride > 0
    assert H >= P and W >= P

    prob_sum = np.zeros((H, W), dtype=np.float32)
    weight = np.zeros((H, W), dtype=np.float32)

    hc4, wc4 = int(cond_map.shape[2]), int(cond_map.shape[3])

    ys = list(range(0, H - P + 1, stride))
    if ys[-1] != (H - P):
        ys.append(H - P)
    xs = list(range(0, W - P + 1, stride))
    if xs[-1] != (W - P):
        xs.append(W - P)

    coords: List[Tuple[int, int]] = [(y0, x0) for y0 in ys for x0 in xs]

    # Run with batches supported by engine if dynamic; otherwise enforce 1
    input_name = None
    for i in range(fine.engine.num_bindings):
        if fine.engine.binding_is_input(i):
            input_name = fine.engine.get_binding_name(i)
            shape_decl = fine.engine.get_binding_shape(i)
            break
    assert input_name is not None

    dynamic_batch = -1 in list(shape_decl)
    batch_allowed = fine_batch if dynamic_batch else 1

    for i0 in range(0, len(coords), batch_allowed):
        batch_coords = coords[i0 : i0 + batch_allowed]
        B = len(batch_coords)
        xs_list: List[np.ndarray] = []
        for (y0, x0) in batch_coords:
            y1, x1 = y0 + P, x0 + P
            y0c = (y0 * hc4) // H
            y1c = ((y1 * hc4) + H - 1) // H
            x0c = (x0 * wc4) // W
            x1c = ((x1 * wc4) + W - 1) // W
            cond_sub = cond_map[:, :, y0c:y1c, x0c:x1c][0, 0]
            cond_patch = cv2.resize(cond_sub, (P, P), interpolation=cv2.INTER_LINEAR)

            rgb_patch = t_img[0, :, y0:y1, x0:x1]  # 3xPxP
            ymin_patch = y_min_full[y0:y1, x0:x1][None, ...]  # 1xPxP
            ymax_patch = y_max_full[y0:y1, x0:x1][None, ...]  # 1xPxP
            x6 = np.concatenate([rgb_patch, ymin_patch, ymax_patch, cond_patch[None, ...]], axis=0)
            xs_list.append(x6)

        x_batch = np.stack(xs_list, axis=0).astype(np.float32)  # Bx6xPxP
        outputs = fine.infer({input_name: x_batch})

        # Assume single output named 'logits' or similar; take the first one
        out_name = [n for n in outputs.keys()][0]
        logits = outputs[out_name]  # Bx2xPxP
        prob = _softmax_channel(logits, axis=1)[:, 1, :, :]  # BxPxP

        for bi, (y0, x0) in enumerate(batch_coords):
            y1, x1 = y0 + P, x0 + P
            prob_sum[y0:y1, x0:x1] += prob[bi]
            weight[y0:y1, x0:x1] += 1.0

    prob_full = prob_sum / weight
    return prob_full.astype(np.float32)


def _coarse_trt(coarse: TrtEngine, rgb: np.ndarray, coarse_size: int, minmax_enable: bool, minmax_kernel: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x6, y_min_full, y_max_full, t_img = _build_6ch_coarse(rgb, coarse_size, minmax_enable, minmax_kernel)
    # Engine input name
    input_name = None
    for i in range(coarse.engine.num_bindings):
        if coarse.engine.binding_is_input(i):
            input_name = coarse.engine.get_binding_name(i)
            break
    assert input_name is not None
    x = x6[None, ...].astype(np.float32)  # 1x6xHc x Wc
    outputs = coarse.infer({input_name: x})
    # Identify outputs: we expect 2 outputs (logits 1x2xHc x Wc, cond 1x1xHc x Wc)
    assert len(outputs) == 2, f"Coarse engine must have 2 outputs, got {list(outputs.keys())}"
    # Determine which is cond by channel dim =1
    names = list(outputs.keys())
    a, b = outputs[names[0]], outputs[names[1]]
    if a.shape[1] == 1:
        cond = a
        logits = b
    else:
        cond = b
        logits = a
    # Coarse prob upsampled to full HxW (optional)
    prob_c = _softmax_channel(logits, axis=1)[:, 1:2]
    H, W = int(t_img.shape[2]), int(t_img.shape[3])
    prob_up = cv2.resize(prob_c[0, 0], (W, H), interpolation=cv2.INTER_LINEAR)
    return prob_up.astype(np.float32), cond.astype(np.float32), t_img.astype(np.float32), y_min_full, y_max_full


# ---- Inference API ----

def infer_image_trt(
    coarse: TrtEngine,
    fine: TrtEngine,
    img_path: str,
    cfg: dict,
    out_dir: Optional[str] = None,
    save_prob: bool = False,
    prob_thresh: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    assert Path(img_path).is_file(), f"Image not found: {img_path}"
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

    prob_c, cond_map, t_img, y_min_full, y_max_full = _coarse_trt(
        coarse, rgb, coarse_size, minmax_enable, minmax_kernel
    )

    prob_f = _tiled_fine_trt(
        fine,
        t_img,
        cond_map,
        y_min_full,
        y_max_full,
        patch_size,
        overlap,
        int(cfg.get("eval", {}).get("fine_batch", 16)),
    )

    pred = (prob_f > prob_thresh).astype(np.uint8) * 255

    if out_dir is not None:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        stem = Path(img_path).stem
        out_mask = Path(out_dir) / f"{stem}_pred.png"
        cv2.imwrite(str(out_mask), pred)
        if save_prob:
            out_prob = Path(out_dir) / f"{stem}_prob.npy"
            np.save(str(out_prob), prob_f.astype(np.float32))

    return pred, prob_f


def main():
    parser = argparse.ArgumentParser(description="WireSegHR TensorRT Inference")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--coarse_engine", type=str, required=True)
    parser.add_argument("--fine_engine", type=str, required=True)
    parser.add_argument("--image", type=str, default="", help="Path to single image")
    parser.add_argument("--images_dir", type=str, default="", help="Directory with images")
    parser.add_argument("--out", type=str, default="outputs/trt_infer")
    parser.add_argument("--save_prob", action="store_true")
    # Benchmarking
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--bench_images_dir", type=str, default="")
    parser.add_argument("--bench_limit", type=int, default=0)
    parser.add_argument("--bench_warmup", type=int, default=2)
    parser.add_argument("--bench_size_filter", type=str, default="")
    parser.add_argument("--bench_report_json", type=str, default="")

    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    print("[TRT][infer] Loaded config:")
    pprint.pprint(cfg)

    coarse = TrtEngine(args.coarse_engine)
    fine = TrtEngine(args.fine_engine)

    if args.benchmark:
        bench_dir = args.bench_images_dir or cfg["data"]["test_images"]
        assert Path(bench_dir).is_dir(), f"Not a directory: {bench_dir}"
        size_filter: Optional[Tuple[int, int]] = None
        if args.bench_size_filter:
            try:
                h_str, w_str = args.bench_size_filter.lower().split("x")
                size_filter = (int(h_str), int(w_str))
            except Exception:
                raise AssertionError(
                    f"Invalid --bench_size_filter format: {args.bench_size_filter} (use HxW)"
                )
        img_files = sorted(
            [
                str(Path(bench_dir) / p)
                for p in os.listdir(bench_dir)
                if p.lower().endswith((".jpg", ".jpeg"))
            ]
        )
        assert len(img_files) > 0, f"No .jpg/.jpeg in {bench_dir}"

        if size_filter is not None:
            sel: List[str] = []
            for p in img_files:
                im = cv2.imread(p, cv2.IMREAD_COLOR)
                assert im is not None
                if im.shape[0] == size_filter[0] and im.shape[1] == size_filter[1]:
                    sel.append(p)
            img_files = sel
            assert len(img_files) > 0, (
                f"No images matching {size_filter[0]}x{size_filter[1]} in {bench_dir}"
            )

        if args.bench_limit > 0:
            img_files = img_files[: args.bench_limit]

        print(f"[TRT][bench] Images: {len(img_files)} from {bench_dir}")
        print(f"[TRT][bench] Warmup: {args.bench_warmup}")

        timings: List[Dict[str, Any]] = []
        # Warmup
        for i in range(min(args.bench_warmup, len(img_files))):
            infer_image_trt(coarse, fine, img_files[i], cfg, out_dir=None, save_prob=False)

        # Timed runs
        for p in img_files[args.bench_warmup :]:
            t0 = time.perf_counter()
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            assert bgr is not None
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

            coarse_size = int(cfg["coarse"]["test_size"])
            minmax_enable = bool(cfg["minmax"]["enable"])
            minmax_kernel = int(cfg["minmax"]["kernel"])

            c0 = time.perf_counter()
            prob_c, cond_map, t_img, y_min_full, y_max_full = _coarse_trt(
                coarse, rgb, coarse_size, minmax_enable, minmax_kernel
            )
            c1 = time.perf_counter()

            patch_size = int(cfg["inference"]["fine_patch_size"])  # 1024
            overlap = int(cfg["fine"]["overlap"])

            prob_f = _tiled_fine_trt(
                fine,
                t_img,
                cond_map,
                y_min_full,
                y_max_full,
                patch_size,
                overlap,
                int(cfg.get("eval", {}).get("fine_batch", 16)),
            )
            c2 = time.perf_counter()

            timings.append(
                {
                    "path": p,
                    "H": int(t_img.shape[2]),
                    "W": int(t_img.shape[3]),
                    "t_coarse_ms": (c1 - c0) * 1000.0,
                    "t_fine_ms": (c2 - c1) * 1000.0,
                    "t_total_ms": (c2 - t0) * 1000.0,
                }
            )

        if len(timings) == 0:
            print("[TRT][bench] Nothing to benchmark after warmup.")
            return

        def _agg(key: str) -> Tuple[float, float, float]:
            vals = sorted([t[key] for t in timings])
            n = len(vals)
            p50 = vals[n // 2]
            p95 = vals[min(n - 1, int(0.95 * (n - 1)))]
            avg = sum(vals) / n
            return avg, p50, p95

        avg_c, p50_c, p95_c = _agg("t_coarse_ms")
        avg_f, p50_f, p95_f = _agg("t_fine_ms")
        avg_t, p50_t, p95_t = _agg("t_total_ms")

        print("[TRT][bench] Results (ms):")
        print(f"  Coarse  avg={avg_c:.2f}  p50={p50_c:.2f}  p95={p95_c:.2f}")
        print(f"  Fine    avg={avg_f:.2f}  p50={p50_f:.2f}  p95={p95_f:.2f}")
        print(f"  Total   avg={avg_t:.2f}  p50={p50_t:.2f}  p95={p95_t:.2f}")
        print(f"  Target  < 1000 ms per 3000x4000 image: {'YES' if p50_t < 1000.0 else 'NO'}")

        if args.bench_report_json:
            import json
            report = {
                "summary": {
                    "avg_ms": avg_t,
                    "p50_ms": p50_t,
                    "p95_ms": p95_t,
                    "avg_coarse_ms": avg_c,
                    "avg_fine_ms": avg_f,
                    "images": len(timings),
                },
                "per_image": timings,
            }
            with open(args.bench_report_json, "w") as f:
                json.dump(report, f, indent=2)
        return

    # Non-benchmark single/directory
    assert (args.image != "") ^ (args.images_dir != ""), "Provide exactly one of --image or --images_dir"
    if args.image:
        infer_image_trt(coarse, fine, args.image, cfg, out_dir=args.out, save_prob=args.save_prob)
        print("[TRT][infer] Done.")
        return

    img_dir = args.images_dir
    assert Path(img_dir).is_dir()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    img_files = sorted([p for p in os.listdir(img_dir) if p.lower().endswith((".jpg", ".jpeg"))])
    assert len(img_files) > 0
    for name in img_files:
        p = str(Path(img_dir) / name)
        infer_image_trt(coarse, fine, p, cfg, out_dir=args.out, save_prob=args.save_prob)
    print("[TRT][infer] Done.")


if __name__ == "__main__":
    main()
