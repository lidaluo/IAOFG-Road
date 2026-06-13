"""快速检查上海 filtered 四目录：stem 对齐、首样本 shape。"""
import argparse
import os
import sys

import numpy as np
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.shanghai_filtered_dataset import list_image_stems


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/config_shanghai.yaml")
    args = p.parse_args()
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    d = cfg["data"]
    root = d["root"]
    images_dir = os.path.join(root, d["images_subdir"])
    stems = list_image_stems(images_dir)
    print(f"[OK] images: {len(stems)} stems under {images_dir}")
    if not stems:
        return
    stem = stems[0]
    ext = d.get("orientation_ext", ".npy")
    paths = [
        os.path.join(images_dir, stem + ".png"),
        os.path.join(root, d["masks_subdir"], stem + ".png"),
        os.path.join(root, d["heatmap_subdir"], stem + ".png"),
        os.path.join(root, d["orientations_subdir"], stem + ext),
    ]
    for path in paths:
        print(f"  exists {os.path.isfile(path)} {path}")
    arr = np.load(paths[3])
    print(f"  orientation shape {arr.shape} dtype {arr.dtype}")


if __name__ == "__main__":
    main()
