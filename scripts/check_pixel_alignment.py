"""
像素级对齐检查：原图 | mask | heatmap | 三色轮廓叠加（绿=道路，红=交叉口热力，蓝=方向置信）。

数据格式（上海 filtered）：
  images/*.png, masks/*.png, heatmap/*.png
  orientations/*.npy 形状 (H,W,3)，第 3 通道为 conf（与训练一致）

heatmap 为 PNG 灰度，读入后归一化到 [0,1]；若存在 heatmap.npy 亦可。

简写 stem：371 / img371 / 完整 SN3_roads_train_AOI_4_Shanghai_PS-RGB_img371

用法:
  python scripts/check_pixel_alignment.py --samples 371 1942 637 749 1138
  python scripts/check_pixel_alignment.py --stems SN3_roads_train_AOI_4_Shanghai_PS-RGB_img10
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

DEFAULT_PREFIX = "SN3_roads_train_AOI_4_Shanghai_PS-RGB_"


def normalize_stem(raw: str, prefix: str) -> str:
    raw = raw.strip()
    if Path(raw).suffix:
        raw = Path(raw).stem
    if raw.startswith("SN3_"):
        return raw
    if raw.startswith("img"):
        return prefix + raw
    if raw.isdigit():
        return prefix + "img" + raw
    return prefix + raw


def resolve_image_path(root: Path, stem: str) -> Path:
    for ext in (".png", ".jpg", ".tif", ".tiff"):
        p = root / "images" / f"{stem}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(root / "images" / f"{stem}.png")


def load_heatmap(root: Path, stem: str) -> np.ndarray:
    png = root / "heatmap" / f"{stem}.png"
    npy = root / "heatmap" / f"{stem}.npy"
    if png.is_file():
        h = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if h is None:
            raise FileNotFoundError(png)
        return h.astype(np.float32) / 255.0
    if npy.is_file():
        h = np.load(str(npy))
        h = np.squeeze(h)
        if h.ndim != 2:
            raise ValueError(f"heatmap npy need 2D, got {h.shape}")
        return h.astype(np.float32) if h.max() <= 1.5 else (h.astype(np.float32) / 255.0)
    raise FileNotFoundError(f"No heatmap for {stem}: {png} or {npy}")


def check_pixel_alignment(
    img_path: Path,
    mask_path: Path,
    heatmap: np.ndarray,
    orient_conf: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    save_path: Path | None = None,
    show: bool = False,
) -> None:
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(img_path)
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)

    tw, th = target_size[0], target_size[1]
    img_bgr = cv2.resize(img_bgr, (tw, th), interpolation=cv2.INTER_LINEAR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    mask = cv2.resize(mask, (tw, th), interpolation=cv2.INTER_NEAREST)
    heatmap_r = cv2.resize(heatmap, (tw, th), interpolation=cv2.INTER_LINEAR)
    orient_conf_r = cv2.resize(orient_conf, (tw, th), interpolation=cv2.INTER_LINEAR)

    overlay = img_bgr.copy()

    mask_u8 = (mask > 127).astype(np.uint8) * 255
    mask_contour, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, mask_contour, -1, (0, 255, 0), 2)

    heatmap_mask = (heatmap_r > 0.5).astype(np.uint8) * 255
    heatmap_points, _ = cv2.findContours(heatmap_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, heatmap_points, -1, (255, 0, 0), 2)

    orient_mask = (orient_conf_r > 0.5).astype(np.uint8) * 255
    orient_contour, _ = cv2.findContours(orient_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, orient_contour, -1, (0, 0, 255), 2)

    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original Image")
    axes[1].imshow(mask, cmap="gray")
    axes[1].set_title("Road Mask")
    axes[2].imshow(heatmap_r, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Intersection Heatmap")
    axes[3].imshow(overlay_rgb)
    axes[3].set_title("Overlay: Green=road, Red=inter, Blue=orient conf")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[Saved] {save_path.resolve()}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="E:/Code/spacenet_filtered")
    ap.add_argument(
        "--prefix",
        type=str,
        default=DEFAULT_PREFIX,
        help="stem 前缀，如 SN3_roads_train_AOI_4_Shanghai_PS-RGB_",
    )
    ap.add_argument(
        "--samples",
        type=str,
        nargs="*",
        default=["371", "1942", "637", "749", "1138"],
        help="简写：371 或 img371；或完整 stem",
    )
    ap.add_argument(
        "--stems",
        type=str,
        nargs="*",
        default=None,
        help="若指定则覆盖 --samples，直接给完整 stem 列表",
    )
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="默认 logs_shanghai/label_qc/pixel_align",
    )
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "logs_shanghai" / "label_qc" / "pixel_align"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.stems:
        stems = args.stems
    else:
        stems = [normalize_stem(s, args.prefix) for s in args.samples]

    tw = th = int(args.size)
    for stem in stems:
        img_path = resolve_image_path(root, stem)
        mask_path = root / "masks" / f"{stem}.png"
        orient_path = root / "orientations" / f"{stem}.npy"
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        if not orient_path.is_file():
            raise FileNotFoundError(orient_path)

        heatmap = load_heatmap(root, stem)
        orient = np.load(str(orient_path))
        if orient.ndim != 3 or orient.shape[2] < 3:
            raise ValueError(f"{orient_path}: expected (H,W,3)+, got {orient.shape}")
        orient_conf = orient[..., 2]

        save_path = out_dir / f"{stem}_pixel_align.png"
        print(f"[Run] {stem}")
        check_pixel_alignment(
            img_path,
            mask_path,
            heatmap,
            orient_conf.astype(np.float32),
            target_size=(tw, th),
            save_path=save_path,
            show=args.show,
        )


if __name__ == "__main__":
    main()
