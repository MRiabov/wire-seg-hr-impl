import argparse
import os
import pprint
import yaml


def main():
    parser = argparse.ArgumentParser(description="WireSegHR training (skeleton)")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg_path = args.config
    if not os.path.isabs(cfg_path):
        cfg_path = os.path.join(os.getcwd(), cfg_path)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    print("[WireSegHR][train] Loaded config from:", cfg_path)
    pprint.pprint(cfg)
    print("[WireSegHR][train] Skeleton OK. Implement training per SEGMENTATION_PLAN.md.")


if __name__ == "__main__":
    main()
