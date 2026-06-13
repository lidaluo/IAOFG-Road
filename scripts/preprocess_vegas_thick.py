"""
拉斯维加斯 AOI2：对原始二值掩膜做形态学膨胀，生成与上海 thick 类似的「厚掩膜」目录。
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, **kwargs):
        return x


def create_thick_masks(
    input_mask_dir: str,
    output_mask_dir: str,
    kernel_size: int = 5,
    verbose: bool = True,
) -> bool:
    input_path = Path(input_mask_dir)
    output_path = Path(output_mask_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
    mask_files: list[Path] = []
    for ext in exts:
        mask_files.extend(input_path.glob(f"*{ext}"))
        mask_files.extend(input_path.glob(f"*{ext.upper()}"))
    mask_files = sorted(set(mask_files))

    if not mask_files:
        print(f"[ERROR] 在 {input_mask_dir} 中未找到掩码文件")
        return False

    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    print(f"[INFO] 掩码数量: {len(mask_files)}，膨胀核: {kernel_size}x{kernel_size}（椭圆）")
    processed = 0
    it = tqdm(mask_files, desc="dilate", disable=not verbose)
    for mask_file in it:
        mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"[WARN] 无法读取: {mask_file}")
            continue
        if mask.max() <= 1:
            mask = (mask.astype(np.float32) * 255.0).astype(np.uint8)
        thick = cv2.dilate(mask, kernel, iterations=1)
        out = output_path / mask_file.name
        cv2.imwrite(str(out), thick)
        processed += 1

    print(f"[OK] 已写入 {processed}/{len(mask_files)} -> {output_path}")
    return processed > 0


def compare_masks_save(
    original: np.ndarray,
    thick: np.ndarray,
    save_path: Path,
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] 未安装 matplotlib，跳过对比图")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original mask")
    axes[0].axis("off")
    axes[1].imshow(thick, cmap="gray")
    axes[1].set_title("Thick mask")
    axes[1].axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] 对比图: {save_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Vegas 厚掩膜预处理（形态学膨胀）")
    p.add_argument("--input_dir", type=str, default="E:/Code/spacenet/train/AOI2_Vegas/masks")
    p.add_argument("--output_dir", type=str, default="E:/Code/spacenet/train/AOI2_Vegas/masks_thick")
    p.add_argument("--kernel_size", type=int, default=5, help="奇数；与上海 thick 常见 3~7px 量级一致时可试 5")
    p.add_argument("--visualize", action="store_true", help="保存一张随机样本对比图到 Vegas 根目录")
    args = p.parse_args()

    if not os.path.isdir(args.input_dir):
        print(f"[ERROR] 输入目录不存在: {args.input_dir}")
        return

    ok = create_thick_masks(
        input_mask_dir=args.input_dir,
        output_mask_dir=args.output_dir,
        kernel_size=args.kernel_size,
    )
    if not ok:
        return

    if args.visualize:
        inp = Path(args.input_dir)
        pngs = list(inp.glob("*.png"))
        if pngs:
            import random

            sample = random.choice(pngs)
            o = cv2.imread(str(sample), cv2.IMREAD_GRAYSCALE)
            t = cv2.imread(str(Path(args.output_dir) / sample.name), cv2.IMREAD_GRAYSCALE)
            if o is not None and t is not None:
                viz = Path(args.output_dir).parent / "thick_mask_comparison.png"
                compare_masks_save(o, t, viz)


if __name__ == "__main__":
    main()
