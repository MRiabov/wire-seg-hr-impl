import argparse
import os
import pprint
import shutil
import subprocess
from typing import Tuple

import torch
import tensorrt as trt

from src.wireseghr.model import WireSegHR
from pathlib import Path


class CoarseModule(torch.nn.Module):
    def __init__(self, core: WireSegHR):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, cond = self.core.forward_coarse(x)
        return logits, cond


class FineModule(torch.nn.Module):
    def __init__(self, core: WireSegHR):
        super().__init__()
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.core.forward_fine(x)
        return logits


def build_model(cfg: dict, device: torch.device) -> WireSegHR:
    pretrained_flag = bool(cfg.get("pretrained", False))
    model = WireSegHR(backbone=cfg["backbone"], in_channels=6, pretrained=pretrained_flag)
    model = model.to(device)
    return model


def main():
    parser = argparse.ArgumentParser(description="Export WireSegHR to ONNX and TensorRT")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--ckpt", type=str, default="", help="Path to checkpoint .pt")
    parser.add_argument("--out_dir", type=str, default="exports")
    parser.add_argument("--coarse_size", type=int, default=1024)
    parser.add_argument("--fine_patch_size", type=int, default=1024)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--trtexec", type=str, default="", help="Optional path to trtexec to build TRT engines")
    parser.add_argument("--build_trt", action="store_true", help="Build TensorRT engines after ONNX export")

    args = parser.parse_args()

    import yaml

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    print("[export] Loaded config:")
    pprint.pprint(cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, device)

    ckpt_path = args.ckpt if args.ckpt else cfg.get("resume", "")
    if ckpt_path:
        assert Path(ckpt_path).is_file(), f"Checkpoint not found: {ckpt_path}"
        print(f"[export] Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"])  # expects dict with key 'model'
    model.eval()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # Prepare dummy inputs (static shapes for best TRT performance)
    coarse_in = torch.randn(1, 6, args.coarse_size, args.coarse_size, device=device)
    fine_in = torch.randn(1, 6, args.fine_patch_size, args.fine_patch_size, device=device)

    # Coarse export
    coarse_wrapper = CoarseModule(model).to(device).eval()
    coarse_onnx = Path(args.out_dir) / f"wireseghr_coarse_{args.coarse_size}.onnx"
    print(f"[export] Exporting COARSE to {coarse_onnx}")
    torch.onnx.export(
        coarse_wrapper,
        coarse_in,
        str(coarse_onnx),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["x_coarse"],
        output_names=["logits", "cond"],
        dynamic_axes=None,
        dynamo=True
    )

    # Fine export
    fine_wrapper = FineModule(model).to(device).eval()
    fine_onnx = Path(args.out_dir) / f"wireseghr_fine_{args.fine_patch_size}.onnx"
    print(f"[export] Exporting FINE to {fine_onnx}")
    torch.onnx.export(
        fine_wrapper,
        fine_in,
        str(fine_onnx),
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["x_fine"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    # Optional TensorRT building via trtexec; fallback to Python API if unavailable
    if args.build_trt:
        trtexec_path = args.trtexec if args.trtexec else shutil.which("trtexec")
        coarse_engine = Path(args.out_dir) / f"wireseghr_coarse_{args.coarse_size}.engine"
        fine_engine = Path(args.out_dir) / f"wireseghr_fine_{args.fine_patch_size}.engine"
        if trtexec_path:
            def build_engine_cli(onnx_path: str, engine_path: str):
                print(f"[export] Building TRT engine (trtexec): {engine_path}")
                cmd = [
                    trtexec_path,
                    f"--onnx={onnx_path}",
                    f"--saveEngine={engine_path}",
                    "--explicitBatch",
                    "--fp16",
                ]
                subprocess.run(cmd, check=True)

            build_engine_cli(str(coarse_onnx), str(coarse_engine))
            build_engine_cli(str(fine_onnx), str(fine_engine))
        else:
            print("[export] trtexec not found; building engines via TensorRT Python API")

            def build_engine_py(onnx_path: str, engine_path: str):
                logger = trt.Logger(trt.Logger.WARNING)
                builder = trt.Builder(logger)
                network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
                parser = trt.OnnxParser(network, logger)
                with open(str(onnx_path), "rb") as f:
                    data = f.read()
                ok = parser.parse(data)
                if not ok:
                    for i in range(parser.num_errors):
                        print(f"[TRT][parser] {parser.get_error(i)}")
                    raise RuntimeError("ONNX parse failed")

                config = builder.create_builder_config()
                config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
                if builder.platform_has_fast_fp16:
                    config.set_flag(trt.BuilderFlag.FP16)

                print(f"[export] Building TRT engine (Python): {engine_path}")
                serialized = builder.build_serialized_network(network, config)
                assert serialized is not None, "Failed to build TensorRT engine"
                with open(str(engine_path), "wb") as f:
                    f.write(serialized)

            build_engine_py(coarse_onnx, coarse_engine)
            build_engine_py(fine_onnx, fine_engine)
    else:
        print("[export] Skipping TensorRT engine build (use --build_trt to enable)")

    print("[export] Done.")


if __name__ == "__main__":
    main()
