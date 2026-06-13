"""检查上海 filtered：影像/mask/heatmap/orientation 对齐与叠图。

目录：images/*.png, masks/*.png, heatmap/*.png, orientations/*.npy
用法见 argparse 或：python scripts/inspect_label_alignment.py --root E:/Code/spacenet_filtered
"""
from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.shanghai_filtered_dataset import list_image_stems


def load_triplet(root: Path, stem: str, orientation_ext: str):
    img_path = None
    for ext in (".png", ".jpg", ".tif", ".tiff"):
        p = root / "images" / f"{stem}{ext}"
        if p.is_file():
            img_path = p
            break
    if img_path is None:
        raise FileNotFoundError(f"No image for stem {stem} under {root / 'images'}")

    mask_path = root / "masks" / f"{stem}.png"
    heat_png = root / "heatmap" / f"{stem}.png"
    heat_npy = root / "heatmap" / f"{stem}.npy"
    orient_path = root / "orientations" / f"{stem}{orientation_ext}"

    img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(img_path)

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_path)

    if heat_png.is_file():
        heatmap = cv2.imread(str(heat_png), cv2.IMREAD_GRAYSCALE)
        if heatmap is None:
            raise FileNotFoundError(heat_png)
        heatmap = heatmap.astype(np.float32) / 255.0
    elif heat_npy.is_file():
        heatmap = np.load(str(heat_npy))
        if heatmap.ndim != 2:
            heatmap = heatmap.squeeze()
            if heatmap.ndim != 2:
                raise ValueError(f"heatmap npy expected 2D, got {heatmap.shape}")
        heatmap = heatmap.astype(np.float32)
        if heatmap.max() > 1.5:
            heatmap = heatmap / 255.0
    else:
        raise FileNotFoundError(f"Neither {heat_png} nor {heat_npy}")

    orient = None
    if orient_path.is_file():
        orient = np.load(str(orient_path))

    return img, mask.astype(np.float32) / 255.0, heatmap, orient, {
        "image": img_path,
        "mask": mask_path,
        "heatmap": heat_png if heat_png.is_file() else heat_npy,
        "orientation": orient_path if orient_path.is_file() else None,
    }


def stats(name: str, arr: np.ndarray):
    arr = np.asarray(arr, dtype=np.float64)
    return f"{name}: shape={arr.shape}, min={arr.min():.4f}, max={arr.max():.4f}, mean={arr.mean():.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="E:/Code/spacenet_filtered")
    ap.add_argument("--stem", type=str, default=None, help="指定 stem；默认随机抽 1 个")
    ap.add_argument("--size", type=int, default=224, help="resize 边长（与训练一致）")
    ap.add_argument("--heat_thresh", type=float, default=0.5, help="交叉口热力图二值阈值")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None, help="保存 PNG 目录，默认 logs_shanghai/label_qc")
    ap.add_argument("--orientation_ext", type=str, default=".npy")
    args = ap.parse_args()

    root = Path(args.root)
    stems = list_image_stems(str(root / "images"))
    if not stems:
        print(f"[Error] 无影像: {root / 'images'}")
        sys.exit(1)

    rng = np.random.default_rng(args.seed)
    stem = args.stem if args.stem else str(rng.choice(stems))
    print(f"[Stem] {stem}")

    img_bgr, mask, heatmap, orient, paths = load_triplet(root, stem, args.orientation_ext)
    print("[Paths]", paths)
    print(stats("mask", mask))
    print(stats("heatmap", heatmap))

    h = w = int(args.size)
    img_r = cv2.resize(img_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    heat_r = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    # 一致性：交叉口是否落在道路内
    road = mask_r > 0.5
    inter_high = heat_r > args.heat_thresh
    inter_on_road = np.logical_and(inter_high, road)
    inter_off_road = np.logical_and(inter_high, np.logical_not(road))
    n_inter = int(inter_high.sum())
    n_on = int(inter_on_road.sum())
    n_off = int(inter_off_road.sum())
    print(
        f"[Align] heat>{args.heat_thresh}: pixels={n_inter}, "
        f"on_road={n_on} ({100.0 * n_on / max(n_inter, 1):.1f}%), "
        f"off_road={n_off} ({100.0 * n_off / max(n_inter, 1):.1f}%)"
    )

    rgb = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
    overlay = img_r.copy()
    contours, _ = cv2.findContours((mask_r > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
    ys, xs = np.where(heat_r > args.heat_thresh)
    for y, x in zip(ys, xs):
        cv2.circle(overlay, (int(x), int(y)), 2, (255, 0, 0), -1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(rgb)
    axes[0].set_title("Image (RGB)")
    axes[1].imshow(mask_r, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Mask (road)")
    axes[2].imshow(heat_r, cmap="hot", vmin=0, vmax=1)
    axes[2].set_title("Intersection heatmap")
    axes[3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[3].set_title(f"Overlay green=road, red=inter (thr={args.heat_thresh})")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(f"{stem}  |  inter on road: {100.0 * n_on / max(n_inter, 1):.1f}%", fontsize=10)
    fig.tight_layout()

    out_dir = Path(args.out) if args.out else Path(PROJECT_ROOT) / "logs_shanghai" / "label_qc"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_png = out_dir / f"{stem}_label_check.png"
    fig.savefig(out_png, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {out_png}")

    if orient is not None:
        print(stats("orientation", orient))
        if orient.ndim == 3 and orient.shape[2] >= 2:
            o_r = cv2.resize(orient, (w, h), interpolation=cv2.INTER_LINEAR)
            dx, dy = o_r[:, :, 0], o_r[:, :, 1]
            conf = o_r[:, :, 2] if o_r.shape[2] > 2 else np.ones_like(dx)
            mag = np.sqrt(dx * dx + dy * dy) * np.clip(conf, 0, 1)
            fig2, ax2 = plt.subplots(1, 2, figsize=(8, 3.5))
            ax2[0].imshow(mag, cmap="viridis")
            ax2[0].set_title("|orient| × conf (GT)")
            ax2[1].imshow(rgb)
            ax2[1].imshow(mag, cmap="viridis", alpha=0.45)
            ax2[1].set_title("RGB + direction mag")
            for a in ax2:
                a.axis("off")
            fig2.suptitle(stem, fontsize=9)
            fig2.tight_layout()
            out2 = out_dir / f"{stem}_orientation_gt.png"
            fig2.savefig(out2, dpi=160, bbox_inches="tight")
            plt.close(fig2)
            print(f"[Saved] {out2}")


if __name__ == "__main__":
    main()
