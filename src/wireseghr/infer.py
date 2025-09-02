import argparse
import os
import pprint
import yaml


def main():
    parser = argparse.ArgumentParser(description="WireSegHR inference (skeleton)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--image", type=str, required=False, help="Path to input image")
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(os.getcwd(), cfg_path)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("[WireSegHR][infer] Loaded config from:", cfg_path)
    pprint.pprint(cfg)
    print("[WireSegHR][infer] Image:", args.image)
    print("[WireSegHR][infer] Skeleton OK. Implement inference per SEGMENTATION_PLAN.md.")


if __name__ == "__main__":
    main()
