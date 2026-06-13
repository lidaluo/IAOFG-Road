"""
从 spacenet_filtered 生成形态学膨胀后的 spacenet_filtered_thick（不修改原目录）。

mask：3×3 椭圆核，膨胀 1 次 → 约 3px 线宽
heatmap：对 heatmap>0.5 二值做同样膨胀，可选高斯平滑，保存 float32 .npy
orientation：conf>0.5 区域膨胀，dx/dy 用最近邻有效像素填充，再限制在厚 mask 内并单位化

用法:
  python scripts/regenerate_thick_dataset.py --src E:/Code/spacenet_filtered --dst E:/Code/spacenet_filtered_thick
  python scripts/regenerate_thick_dataset.py --src E:/Code/spacenet_filtered --dst E:/Code/spacenet_filtered_thick --skip_images
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from scipy.ndimage import distance_transform_edt
except ImportError as e:
    print("需要: pip install scipy", file=sys.stderr)
    raise SystemExit(1) from e

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.shanghai_filtered_dataset import list_image_stems


def get_kernel():
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))


def thicken_mask_u8(mask_u8: np.ndarray, kernel) -> np.ndarray:
    """mask 0-255 二值道路"""
    m = (mask_u8 > 127).astype(np.uint8) * 255
    return cv2.dilate(m, kernel, iterations=1)


def load_mask(src_masks: Path, stem: str) -> np.ndarray | None:
    for ext in (".png", ".tif", ".tiff", ".jpg"):
        p = src_masks / f"{stem}{ext}"
        if p.is_file():
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            return m
    return None


def load_heatmap_float(src_hm: Path, stem: str) -> np.ndarray | None:
    png = src_hm / f"{stem}.png"
    npy = src_hm / f"{stem}.npy"
    if png.is_file():
        h = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if h is None:
            return None
        return (h.astype(np.float32) / 255.0).clip(0, 1)
    if npy.is_file():
        h = np.load(str(npy))
        h = np.squeeze(h)
        if h.ndim != 2:
            return None
        h = h.astype(np.float32)
        if h.max() > 1.5:
            h = h / 255.0
        return np.clip(h, 0, 1)
    return None


def thicken_heatmap(heat: np.ndarray, kernel, gaussian_ksize: int = 0) -> np.ndarray:
    binary = (heat > 0.5).astype(np.uint8)
    dil = cv2.dilate(binary, kernel, iterations=1).astype(np.float32)
    if gaussian_ksize and gaussian_ksize >= 3 and gaussian_ksize % 2 == 1:
        dil = cv2.GaussianBlur(dil, (gaussian_ksize, gaussian_ksize), 0)
    return np.clip(dil, 0, 1)


def thicken_orientation(orient: np.ndarray, mask_thick_u8: np.ndarray, kernel) -> np.ndarray:
    """orient (H,W,3) float; mask_thick 0-255"""
    dx = orient[:, :, 0].astype(np.float32)
    dy = orient[:, :, 1].astype(np.float32)
    conf = np.clip(orient[:, :, 2].astype(np.float32), 0, 1)
    road_thick = mask_thick_u8 > 127

    valid = conf > 0.5
    valid_dil = cv2.dilate(valid.astype(np.uint8), kernel, iterations=1).astype(bool)

    input_dt = np.where(valid, 0, 1).astype(np.uint8)
    _, indices = distance_transform_edt(input_dt, return_indices=True)
    ni, nj = indices[0], indices[1]
    dx_nn = dx[ni, nj]
    dy_nn = dy[ni, nj]
    conf_nn = conf[ni, nj]

    dx_th = np.where(valid_dil, dx_nn, 0.0).astype(np.float32)
    dy_th = np.where(valid_dil, dy_nn, 0.0).astype(np.float32)
    c_th = np.where(valid_dil, np.maximum(conf_nn, 0.5), 0.0).astype(np.float32)

    m = road_thick.astype(np.float32)
    dx_th *= m
    dy_th *= m
    c_th *= m

    norm = np.sqrt(dx_th * dx_th + dy_th * dy_th) + 1e-6
    on = (c_th > 0.5) & road_thick
    dx_th = np.where(on, dx_th / norm, dx_th)
    dy_th = np.where(on, dy_th / norm, dy_th)

    return np.stack([dx_th, dy_th, c_th], axis=-1)


def copy_images(src_img: Path, dst_img: Path, stems: list[str], log: logging.Logger):
    dst_img.mkdir(parents=True, exist_ok=True)
    n = 0
    for stem in tqdm(stems, desc="copy images"):
        copied = False
        for ext in (".png", ".jpg", ".tif", ".tiff"):
            p = src_img / f"{stem}{ext}"
            if p.is_file():
                shutil.copy2(p, dst_img / f"{stem}{ext}")
                n += 1
                copied = True
                break
        if not copied:
            log.warning("no image for stem %s", stem)
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=str, default="E:/Code/spacenet_filtered")
    ap.add_argument("--dst", type=str, default="E:/Code/spacenet_filtered_thick")
    ap.add_argument("--skip_images", action="store_true", help="跳过复制 images（已拷过）")
    ap.add_argument("--gaussian", type=int, default=0, help="heatmap 膨胀后高斯核边长，0 关闭，建议 5")
    ap.add_argument("--val_samples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="0=全部；>0 只处理前 N 个 stem（试跑）")
    args = ap.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    kernel = get_kernel()

    log_path = dst / "regenerate_log.txt"
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    log = logging.getLogger("thick")

    stems = list_image_stems(str(src / "images"))
    if not stems:
        print(f"[Error] 无影像: {src / 'images'}")
        sys.exit(1)
    if args.limit and args.limit > 0:
        stems = stems[: args.limit]

    masks_out = dst / "masks_thick"
    hm_out = dst / "heatmap_thick"
    ori_out = dst / "orientations_thick"
    val_dir = dst / "validation_samples"
    for d in (masks_out, hm_out, ori_out, val_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_img = 0
    if not args.skip_images:
        n_img = copy_images(src / "images", dst / "images", stems, log)
    else:
        n_img = len(list((dst / "images").glob("*"))) if (dst / "images").is_dir() else 0

    ok_m, ok_h, ok_o = 0, 0, 0
    errors: list[str] = []

    for stem in tqdm(stems, desc="masks/heatmap/orient"):
        try:
            mask = load_mask(src / "masks", stem)
            if mask is None:
                errors.append(f"{stem}: mask missing")
                continue
            heat = load_heatmap_float(src / "heatmap", stem)
            if heat is None:
                errors.append(f"{stem}: heatmap missing")
                continue
            op = src / "orientations" / f"{stem}.npy"
            if not op.is_file():
                errors.append(f"{stem}: orientation missing")
                continue
            orient = np.load(str(op))
            if orient.ndim != 3 or orient.shape[2] < 3:
                errors.append(f"{stem}: bad orient shape")
                continue

            mask_t = thicken_mask_u8(mask, kernel)
            if orient.shape[:2] != mask.shape[:2]:
                errors.append(f"{stem}: shape mismatch mask {mask.shape} vs orient {orient.shape}")
                continue
            if heat.shape[:2] != mask.shape[:2]:
                errors.append(f"{stem}: shape mismatch heat {heat.shape} vs mask {mask.shape}")
                continue

            heat_t = thicken_heatmap(heat, kernel, gaussian_ksize=args.gaussian)
            ori_t = thicken_orientation(orient, mask_t, kernel)

            cv2.imwrite(str(masks_out / f"{stem}.png"), mask_t)
            np.save(str(hm_out / f"{stem}.npy"), heat_t.astype(np.float32))
            np.save(str(ori_out / f"{stem}.npy"), ori_t.astype(np.float32))
            ok_m += 1
            ok_h += 1
            ok_o += 1
        except Exception as e:
            errors.append(f"{stem}: {e}")

    with open(dst / "errors.log", "w", encoding="utf-8") as f:
        for line in errors:
            f.write(line + "\n")

    rng = np.random.default_rng(args.seed)
    val_stems = list(rng.choice(stems, size=min(args.val_samples, len(stems)), replace=False))

    for i, stem in enumerate(val_stems):
        mp = src / "masks"
        m0 = load_mask(mp, stem)
        mt = cv2.imread(str(masks_out / f"{stem}.png"), cv2.IMREAD_GRAYSCALE)
        ht = np.load(str(hm_out / f"{stem}.npy"))
        ot = np.load(str(ori_out / f"{stem}.npy"))
        img_p = None
        for ext in (".png", ".jpg", ".tif", ".tiff"):
            p = dst / "images" / f"{stem}{ext}"
            if p.is_file():
                img_p = p
                break
        rgb = None
        if img_p:
            bgr = cv2.imread(str(img_p))
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        if m0 is not None and mt is not None:
            axes[0].imshow(np.hstack([m0, mt]), cmap="gray")
            axes[0].set_title("Left: orig mask | Right: thick mask")
        if rgb is not None:
            ov = rgb.copy().astype(np.float32) / 255.0
            h, w = ov.shape[:2]
            red = np.zeros_like(ov)
            red[:, :, 0] = (ht > 0.5).astype(np.float32)
            ov = np.clip(ov * 0.55 + red * 0.45, 0, 1)
            axes[1].imshow(ov)
            axes[1].set_title("RGB + thick heatmap (red)")
        oc = ot[:, :, 2]
        if rgb is not None:
            ov2 = (rgb.astype(np.float32) / 255.0).copy()
            blue = np.zeros_like(ov2)
            blue[:, :, 2] = (oc > 0.5).astype(np.float32)
            ov2 = np.clip(ov2 * 0.55 + blue * 0.45, 0, 1)
            axes[2].imshow(ov2)
            axes[2].set_title("RGB + thick orient conf (blue)")
        for ax in axes:
            ax.axis("off")
        fig.suptitle(stem, fontsize=9)
        fig.tight_layout()
        fig.savefig(val_dir / f"val_{i:02d}_{stem}.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    print("")
    print("数据集重构完成！")
    print(f"原始样本数: {len(stems)}")
    print(f"images 复制: {n_img}")
    print(f"厚 mask 生成: {ok_m}/{len(stems)}")
    print(f"厚 heatmap 生成: {ok_h}/{len(stems)}")
    print(f"厚 orientation 生成: {ok_o}/{len(stems)}")
    print(f"验证样本可视化: {val_dir}")
    print(f"错误记录: {dst / 'errors.log'} ({len(errors)} 条)")
    if errors:
        print("前 5 条:", *errors[:5], sep="\n  ")


if __name__ == "__main__":
    main()
