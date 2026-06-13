"""
Read GeoTIFF / TIFF as RGB from a folder (e.g. 百度网盘 对比实验), run road segmentation.

- 使用 tifffile 读取 16 位 TIFF（PIL 常误读成 0–4 的 uint8 导致全黑），再按百分位拉伸到 8 位 RGB。
- 输出：原图 RGB、GT（若有）、道路预测；以及三联图 *_triptych.png。

Default input: E:/BaiduNetdiskDownload/对比实验
Default output: <input>/推理结果

依赖: pip install tifffile
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import tifffile
except ImportError as e:
    raise ImportError("请安装: pip install tifffile") from e

from models.road_extraction_model import RoadExtractionModel

CKPT = ROOT / "checkpoints_shanghai_thick" / "model_best_val_iou.pth"
MASK_THICK = Path("E:/Code/spacenet_filtered_thick/masks_thick")
MASK_THIN = Path("E:/Code/spacenet_filtered/masks")

DEFAULT_INPUT = Path("E:/BaiduNetdiskDownload") / "\u5bf9\u6bd4\u5b9e\u9a8c"  # 对比实验
OUT_SUBDIR = "\u63a8\u7406\u7ed3\u679c"  # 推理结果

PRED_THRESH = 0.26
PREVIEW_MAX_SIDE = 1400


def load_model(device: torch.device) -> RoadExtractionModel:
    model = RoadExtractionModel(encoder="swin_tiny", num_classes=1, input_size=224)
    state = torch.load(str(CKPT), map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _percentile_stretch_u8(vol: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """HxWxC -> uint8 RGB，uint16/float 等."""
    if vol.ndim == 2:
        vol = np.stack([vol, vol, vol], axis=-1)
    if vol.shape[-1] > 3:
        vol = vol[..., :3].copy()
    out = np.zeros((vol.shape[0], vol.shape[1], 3), dtype=np.uint8)
    for c in range(3):
        ch = vol[..., c].astype(np.float64)
        lo, hi = np.percentile(ch, [p_lo, p_hi])
        if hi <= lo:
            hi = lo + 1.0
        ch = (ch - lo) / (hi - lo) * 255.0
        out[..., c] = np.clip(ch, 0, 255).astype(np.uint8)
    return out


def load_tiff_as_rgb_u8(path: Path) -> np.ndarray:
    """
    优先 tifffile 读原始位深；PIL 对 16bit 常截断成近全黑 uint8。
    """
    a = tifffile.imread(path)
    if a.ndim == 2:
        a = np.stack([a, a, a], axis=-1)
    elif a.ndim == 3 and a.shape[-1] > 3:
        a = a[..., :3]

    if a.dtype == np.uint8:
        if float(a.max()) <= 32.0:
            # 疑似被错误量化：重新读一次并做 min-max stretch
            lo, hi = float(a.min()), float(a.max())
            if hi > lo:
                a = ((a.astype(np.float32) - lo) / (hi - lo) * 255.0).astype(np.uint8)
            else:
                a = np.zeros_like(a, dtype=np.uint8)
    elif a.dtype in (np.uint16, np.uint32, np.int32):
        a = _percentile_stretch_u8(a)
    elif np.issubdtype(a.dtype, np.floating):
        a = _percentile_stretch_u8(np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0))
    else:
        a = np.asarray(a, dtype=np.uint8)
    return np.asarray(a, dtype=np.uint8)


def infer_mask_224(rgb: np.ndarray, model: RoadExtractionModel, device: torch.device) -> np.ndarray:
    pil = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.sigmoid(out["segmentation"])[0, 0].detach().cpu().numpy()
    return (prob > PRED_THRESH).astype(np.uint8)


def upsample_mask(mask_224: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray((mask_224 * 255).astype(np.uint8), mode="L")
    up = pil.resize((size[0], size[1]), Image.NEAREST)
    return (np.asarray(up) > 127).astype(np.uint8)


def sample_id_from_stem(stem: str) -> str | None:
    if "SN3_roads_train_AOI_4_Shanghai" in stem or re.match(r"SN3_roads_train_.*_img\d+", stem):
        return stem
    return None


def load_gt_optional(sample_id: str) -> np.ndarray | None:
    for base in (MASK_THICK, MASK_THIN):
        p = base / f"{sample_id}.png"
        if p.exists():
            gt = np.asarray(Image.open(p).convert("L"), dtype=np.uint8)
            return (gt > 127).astype(np.uint8)
    return None


def save_overlay(rgb: np.ndarray, mask: np.ndarray, out_path: Path) -> None:
    h, w = rgb.shape[:2]
    fig, ax = plt.subplots(1, 1, figsize=(w / 200, h / 200), dpi=200, facecolor="white")
    ax.imshow(rgb)
    ax.contour(mask, levels=[0.5], colors=["#cc0000"], linewidths=1.6)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout(pad=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0, facecolor="white")
    plt.close(fig)


def resize_preview(rgb: np.ndarray, mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= PREVIEW_MAX_SIDE:
        return rgb, mask
    scale = PREVIEW_MAX_SIDE / m
    nh, nw = int(h * scale), int(w * scale)
    r = np.asarray(Image.fromarray(rgb).resize((nw, nh), Image.BILINEAR))
    if mask is None:
        return r, None
    mo = np.asarray(
        Image.fromarray((mask * 255).astype(np.uint8), mode="L").resize((nw, nh), Image.NEAREST)
    )
    mo = (mo > 127).astype(np.uint8)
    return r, mo


def save_triptych_figure(
    rgb: np.ndarray,
    pred: np.ndarray,
    gt: np.ndarray | None,
    out_path: Path,
    title: str,
) -> None:
    """固定 1×3：原图 | GT | 道路提取（无 GT 时中列提示）。"""
    rgb_p, pred_p = resize_preview(rgb, pred)
    gt_p = None
    if gt is not None:
        _, gt_p = resize_preview(rgb, gt)

    fig, axes = plt.subplots(1, 3, figsize=(14.4, 4.6), facecolor="white")

    axes[0].imshow(rgb_p)
    axes[0].set_title("(a) Original (RGB)", fontsize=11)

    if gt_p is not None:
        axes[1].imshow(rgb_p, alpha=0.96)
        axes[1].contour(gt_p, levels=[0.5], colors=["#00aa44"], linewidths=1.8)
    else:
        axes[1].imshow(np.ones((*rgb_p.shape[:2], 3), dtype=np.float32) * 0.94)
        axes[1].text(
            0.5,
            0.5,
            "GT not available\n(no mask in dataset)",
            ha="center",
            va="center",
            transform=axes[1].transAxes,
            fontsize=10,
            color="#666666",
        )
    axes[1].set_title("(b) Ground truth", fontsize=11)

    axes[2].imshow(rgb_p, alpha=0.96)
    axes[2].contour(pred_p, levels=[0.5], colors=["#cc0000"], linewidths=1.8)
    axes[2].set_title("(c) Road extraction (ours)", fontsize=11)

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def process_one(
    tif_path: Path,
    out_dir: Path,
    model: RoadExtractionModel,
    device: torch.device,
) -> None:
    stem = tif_path.stem
    rgb = load_tiff_as_rgb_u8(tif_path)
    if int(rgb.size) == 0 or int(rgb.max()) == 0:
        print(
            f"[warn] {tif_path.name}: 影像全为 0（文件可能损坏或未正确导出），"
            f"请检查网盘源文件。"
        )
    h, w = rgb.shape[:2]

    pred_224 = infer_mask_224(rgb, model, device)
    pred_full = upsample_mask(pred_224, (w, h))

    sid = sample_id_from_stem(stem)
    gt_full: np.ndarray | None = None
    if sid:
        gt_raw = load_gt_optional(sid)
        if gt_raw is not None:
            if gt_raw.shape[0] != h or gt_raw.shape[1] != w:
                gt_full = (
                    np.asarray(
                        Image.fromarray((gt_raw * 255).astype(np.uint8), mode="L").resize(
                            (w, h), Image.NEAREST
                        )
                    )
                    > 127
                ).astype(np.uint8)
            else:
                gt_full = gt_raw

    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_dir / f"{stem}_rgb.png")
    if gt_full is not None:
        Image.fromarray((gt_full * 255).astype(np.uint8), mode="L").save(out_dir / f"{stem}_gt_mask.png")
    Image.fromarray((pred_full * 255).astype(np.uint8), mode="L").save(out_dir / f"{stem}_pred_mask.png")
    save_overlay(rgb, pred_full, out_dir / f"{stem}_overlay.png")
    save_triptych_figure(rgb, pred_full, gt_full, out_dir / f"{stem}_triptych.png", stem)

    print(f"OK {stem} -> {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT, help="含 TIFF 的文件夹")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录（默认 <input-dir>/推理结果）",
    )
    args = ap.parse_args()

    inp = args.input_dir
    if not inp.is_absolute():
        inp = ROOT / inp
    if not inp.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {inp}")

    out = args.out_dir
    if out is None:
        out = inp / OUT_SUBDIR
    elif not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    if not CKPT.exists():
        raise FileNotFoundError(CKPT)

    seen: set[Path] = set()
    tifs: list[Path] = []
    for pat in ("*.tif", "*.tiff", "*.TIF"):
        for p in sorted(inp.glob(pat)):
            rp = p.resolve()
            if rp in seen:
                continue
            seen.add(rp)
            tifs.append(p)
    if not tifs:
        raise FileNotFoundError(f"未找到 TIFF: {inp}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    for p in tifs:
        if p.is_dir():
            continue
        process_one(p, out, model, device)

    print(f"Done. Outputs under: {out}")


if __name__ == "__main__":
    main()
