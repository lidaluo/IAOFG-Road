"""
快速检查道路 mask 的骨架与粗略线宽（面积/骨架长）。

上海数据 mask 为 PNG（非 tif）。依赖: scikit-image, scipy

用法:
  python scripts/mask_width_skeleton_check.py --stem SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1813
  python scripts/mask_width_skeleton_check.py --mask E:/Code/spacenet_filtered/masks/xxx.png
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from skimage.morphology import skeletonize
except ImportError as e:
    print("请安装: pip install scikit-image scipy", file=sys.stderr)
    raise SystemExit(1) from e

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="E:/Code/spacenet_filtered")
    ap.add_argument(
        "--stem",
        type=str,
        default="SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1813",
        help="不含扩展名时自动找 masks/{stem}.png",
    )
    ap.add_argument("--mask", type=str, default=None, help="直接指定 mask 文件路径")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--save", type=str, default=None, help="保存 PNG，默认 logs_shanghai/label_qc/mask_skeleton_*.png")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    if args.mask:
        mask_path = Path(args.mask)
    else:
        stem = args.stem
        if not stem.endswith((".png", ".tif", ".tiff")):
            mask_path = Path(args.root) / "masks" / f"{stem}.png"
        else:
            mask_path = Path(args.root) / "masks" / Path(stem).name

    if not mask_path.is_file():
        raise FileNotFoundError(mask_path)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Cannot read {mask_path}")

    tw = th = int(args.size)
    mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    road_pixels = mask > 127

    road_skeleton = skeletonize(road_pixels)
    skeleton_pixels = int(np.sum(road_skeleton))

    road_area = int(np.sum(road_pixels))
    if skeleton_pixels > 0:
        estimated_width = road_area / skeleton_pixels
    else:
        estimated_width = 0.0

    print(f"[mask] {mask_path}")
    print(f"道路像素总数: {road_area}")
    print(f"道路骨架像素数: {skeleton_pixels}")
    print(f"估计平均线宽(面积/骨架): {estimated_width:.2f} 像素")
    print(f"道路像素占比: {100.0 * road_area / (tw * th):.2f}%")

    h, w = mask.shape
    cy, cx = h // 2, w // 2
    r = 20
    crop = mask[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(mask, cmap="gray")
    axes[0].set_title(f"Original Mask\n{road_area} road pixels")

    axes[1].imshow(road_skeleton, cmap="gray")
    axes[1].set_title(f"Skeleton\n{skeleton_pixels} skel px")

    axes[2].imshow(crop, cmap="gray")
    axes[2].set_title(f"Zoom center {crop.shape[0]}x{crop.shape[1]}")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"est. width ≈ {estimated_width:.2f} px", fontsize=11)
    fig.tight_layout()

    out = args.save
    if out is None:
        out = PROJECT_ROOT / "logs_shanghai" / "label_qc" / f"{mask_path.stem}_mask_skeleton.png"
    else:
        out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    print(f"[Saved] {out.resolve()}")
    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
