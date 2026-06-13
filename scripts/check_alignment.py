"""
数据集对齐检查：2×2 与「RGB+mask / RGB+heatmap / mask+heatmap / RGB+orient_conf」一致。

上海 filtered 实际格式：
  images/*.png, masks/*.png, heatmap/*.png, orientations/*.npy (H,W,3)

heatmap 若为 .npy（2D）也会尝试读取。

用法:
  python scripts/check_alignment.py --root E:/Code/spacenet_filtered --stem SN3_roads_train_AOI_4_Shanghai_PS-RGB_img10
  python scripts/check_alignment.py --stem SN3_roads_train_AOI_4_Shanghai_PS-RGB_img10 --save alignment_check.png
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_heatmap(path: Path) -> np.ndarray:
    if path.suffix.lower() in (".png", ".jpg", ".tif", ".tiff"):
        h = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if h is None:
            raise FileNotFoundError(path)
        return h.astype(np.float32) / 255.0
    if path.suffix.lower() == ".npy":
        h = np.load(str(path))
        if h.ndim != 2:
            h = np.squeeze(h)
        return h.astype(np.float32) if h.max() <= 1.5 else (h.astype(np.float32) / 255.0)
    raise ValueError(f"Unsupported heatmap: {path}")


def resolve_paths(root: Path, stem: str) -> tuple[Path, Path, Path, Path]:
    img = None
    for ext in (".png", ".jpg", ".tif", ".tiff"):
        p = root / "images" / f"{stem}{ext}"
        if p.is_file():
            img = p
            break
    if img is None:
        raise FileNotFoundError(f"No image for stem {stem} under {root / 'images'}")

    mask = root / "masks" / f"{stem}.png"
    if not mask.is_file():
        raise FileNotFoundError(mask)

    heat = root / "heatmap" / f"{stem}.png"
    if not heat.is_file():
        alt = root / "heatmap" / f"{stem}.npy"
        if alt.is_file():
            heat = alt
        else:
            raise FileNotFoundError(f"Neither {heat} nor {alt}")

    orient = root / "orientations" / f"{stem}.npy"
    if not orient.is_file():
        raise FileNotFoundError(orient)

    return img, mask, heat, orient


def check_alignment(
    img_path: Path,
    mask_path: Path,
    heatmap_path: Path,
    orient_path: Path,
    size: int = 224,
    save_path: str | Path | None = None,
    show: bool = False,
):
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(img_path)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST)
    mask_bin = mask > 127

    heatmap = load_heatmap(heatmap_path)
    heatmap = cv2.resize(heatmap, (size, size), interpolation=cv2.INTER_LINEAR)
    hm_min, hm_max = float(heatmap.min()), float(heatmap.max())
    heatmap_norm = (heatmap - hm_min) / (hm_max - hm_min + 1e-8)

    orient = np.load(str(orient_path))
    if orient.ndim != 3 or orient.shape[2] < 3:
        raise ValueError(f"orient expected (H,W,3)+, got {orient.shape}")
    orient_r = cv2.resize(orient, (size, size), interpolation=cv2.INTER_LINEAR)
    orient_conf = orient_r[..., 2]

    fig, axes = plt.subplots(2, 2, figsize=(12, 12))

    overlay_mask = img.copy()
    overlay_mask[mask_bin] = [0, 255, 0]
    axes[0, 0].imshow(cv2.cvtColor(overlay_mask, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("RGB + Road Mask (green)")

    overlay_heat = img.copy()
    overlay_heat[heatmap_norm > 0.5] = [255, 0, 0]
    axes[0, 1].imshow(cv2.cvtColor(overlay_heat, cv2.COLOR_BGR2RGB))
    axes[0, 1].set_title("RGB + Heatmap thr=0.5 (norm min-max, red)")

    overlay_mask_heat = np.zeros_like(img)
    overlay_mask_heat[mask_bin] = [0, 255, 0]
    heatmap_mask = np.zeros((size, size), dtype=bool)
    heatmap_mask[heatmap_norm > 0.5] = True
    overlay_mask_heat[heatmap_mask] = [255, 0, 0]
    axes[1, 0].imshow(cv2.cvtColor(overlay_mask_heat, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title("Mask (green) + Heatmap>0.5 (red)")

    overlay_orient = img.copy()
    overlay_orient[orient_conf > 0.5] = [0, 0, 255]
    axes[1, 1].imshow(cv2.cvtColor(overlay_orient, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title("RGB + Orient conf>0.5 (blue)")

    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle(f"{img_path.stem}  |  heat [{hm_min:.3f},{hm_max:.3f}]", fontsize=10)
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[Saved] {save_path.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="E:/Code/spacenet_filtered")
    ap.add_argument("--stem", type=str, default="SN3_roads_train_AOI_4_Shanghai_PS-RGB_img10")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument(
        "--save",
        type=str,
        default=None,
        help="保存路径，默认 logs_shanghai/label_qc/{stem}_alignment_2x2.png",
    )
    ap.add_argument("--show", action="store_true", help="弹窗显示（默认只保存）")
    args = ap.parse_args()

    root = Path(args.root)
    img_p, mask_p, heat_p, orient_p = resolve_paths(root, args.stem)
    out = args.save
    if out is None:
        out = PROJECT_ROOT / "logs_shanghai" / "label_qc" / f"{args.stem}_alignment_2x2.png"

    print(f"[image]    {img_p}")
    print(f"[mask]     {mask_p}")
    print(f"[heatmap]  {heat_p}")
    print(f"[orient]   {orient_p}")

    check_alignment(
        img_p,
        mask_p,
        heat_p,
        orient_p,
        size=args.size,
        save_path=out,
        show=args.show,
    )


if __name__ == "__main__":
    main()
