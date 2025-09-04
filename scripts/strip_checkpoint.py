#!/usr/bin/env python3

import argparse
from pathlib import Path
import torch
from safetensors.torch import save_file as safetensors_save_file


def main():
    parser = argparse.ArgumentParser(
        description="Strip training checkpoint to inference-only weights (FP32)."
    )
    parser.add_argument("--in", dest="inp", type=str, required=True, help="Path to training checkpoint .pt")
    parser.add_argument("--out", dest="out", type=str, required=True, help="Path to save weights-only .pt or .safetensors")
    # Output format is inferred from --out extension
    args = parser.parse_args()

    in_path = Path(args.inp)
    out_path = Path(args.out)

    assert in_path.is_file(), f"Input file does not exist: {in_path}"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(in_path), map_location="cpu")

    # Primary (project) format: {'step', 'model', 'optim', 'scaler', 'best_f1'}
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
    # Secondary common format: {'state_dict': model.state_dict(), ...}
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # Fallback: checkpoint is already a pure state_dict
        assert isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()), (
            "Checkpoint is not a recognized format: expected keys 'model' or 'state_dict', "
            "or a pure state_dict (name->Tensor)."
        )
        state_dict = ckpt

    #in the future, can cast to bfloat if necessary.
    # state_dict = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in state_dict.items()}

    suffix = out_path.suffix.lower()
    if suffix == ".safetensors":
        safetensors_save_file(state_dict, str(out_path))
        print(f"[strip_checkpoint] Saved safetensors (pure state_dict) to: {out_path}")
    else:
        to_save = {"model": state_dict}
        torch.save(to_save, str(out_path))
        print(f"[strip_checkpoint] Saved dict with only 'model' to: {out_path}")


if __name__ == "__main__":
    main()
