"""
For each sample that has figures under paper_materials/anchor_direction/,
run segmentation inference and save a 1×3 row: RGB | GT (thick mask) | Predicted road mask.

Reads existing filenames like: anchor_dir_<sample_id>_apls1.00_gn178.png
Outputs: paper_materials/anchor_direction/seg_triptych_<sample_id>.png
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

from models.road_extraction_model import RoadExtractionModel

ANCHOR_DIR = ROOT / "paper_materials" / "anchor_direction"
OUT_DIR = ANCHOR_DIR
# 避免在终端传中文路径乱码，用开关在脚本内写死 UTF-8 目录名
COMPARE_RESULT_DIR = ROOT / "paper_materials" / "\u5bf9\u6bd4\u7ed3\u679c"  # 对比结果
CKPT = ROOT / "checkpoints_shanghai_thick" / "model_best_val_iou.pth"

SHANGHAI_IMG = Path("E:/Code/spacenet_filtered/images")
MASK_THICK = Path("E:/Code/spacenet_filtered_thick/masks_thick")

IMAGE_CANDIDATES = [SHANGHAI_IMG, Path("E:/Code/spacenet_filtered_thick/images")]

PRED_THRESH = 0.26

# anchor_dir_<sample_id>_apls<float>_gn<int>.png
FNAME_RE = re.compile(r"^anchor_dir_(.+)_apls[\d.]+_gn\d+\.png$")


def find_image(sample_id: str) -> Path | None:
    for d in IMAGE_CANDIDATES:
        p = d / f"{sample_id}.png"
        if p.exists():
            return p
    return None


def load_gt_thick(sample_id: str) -> np.ndarray | None:
    p = MASK_THICK / f"{sample_id}.png"
    if not p.exists():
        return None
    gt = np.asarray(Image.open(p).convert("L").resize((224, 224), Image.NEAREST), dtype=np.uint8)
    return (gt > 127).astype(np.uint8)


def load_model(device: torch.device) -> RoadExtractionModel:
    model = RoadExtractionModel(encoder="swin_tiny", num_classes=1, input_size=224)
    state = torch.load(str(CKPT), map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def infer_road_mask(
    img_rgb: np.ndarray, model: RoadExtractionModel, device: torch.device
) -> np.ndarray:
    pil = Image.fromarray(img_rgb).resize((224, 224), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.sigmoid(out["segmentation"])[0, 0].detach().cpu().numpy()
    return (prob > PRED_THRESH).astype(np.uint8)


def parse_sample_ids_from_anchor_dir(anchor_dir: Path) -> list[str]:
    ids: list[str] = []
    for p in sorted(anchor_dir.glob("anchor_dir_*.png")):
        m = FNAME_RE.match(p.name)
        if m:
            ids.append(m.group(1))
    return ids


def render_triptych(sample_id: str, model: RoadExtractionModel, device: torch.device, out_path: Path) -> None:
    ip = find_image(sample_id)
    gt = load_gt_thick(sample_id)
    if ip is None or gt is None:
        raise FileNotFoundError(f"missing image or GT mask: {sample_id}")

    img = np.asarray(Image.open(ip).convert("RGB").resize((224, 224), Image.BILINEAR))
    pred = infer_road_mask(img, model, device)
    h, w = img.shape[:2]

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.0), facecolor="white")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    axes[0].imshow(img)
    axes[0].set_title("(a) Satellite image", fontsize=11)

    axes[1].imshow(img, alpha=0.92)
    axes[1].contour(gt, levels=[0.5], colors=["#00aa44"], linewidths=1.8)
    axes[1].set_title("(b) Ground truth (thick mask)", fontsize=11)

    axes[2].imshow(img, alpha=0.92)
    axes[2].contour(pred, levels=[0.5], colors=["#cc0000"], linewidths=2.0)
    axes[2].set_title("(c) Predicted road (segmentation, τ={:.2f})".format(PRED_THRESH), fontsize=11)

    fig.suptitle(f"Road extraction — {sample_id}", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 与 generate_anchor_direction_visuals 同一批样本（anchor 目录为空时兜底）
FALLBACK_SAMPLE_IDS = [
    "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1607",
    "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1791",
    "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img525",
    "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1681",
    "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img919",
]

SHANGHAI_PREFIX = "SN3_roads_train_AOI_4_Shanghai_PS-RGB_"


def resolve_sample_id(token: str) -> str:
    """Accept full stem or shorthand like img1582 / 1582."""
    s = token.strip()
    if not s:
        return s
    if s.startswith("SN3_roads_train_"):
        return s
    m = re.match(r"^img(\d+)$", s, re.I)
    if m:
        return f"{SHANGHAI_PREFIX}img{m.group(1)}"
    if s.isdigit():
        return f"{SHANGHAI_PREFIX}img{s}"
    return s


def merge_ids(base: list[str], extra: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for sid in base + [resolve_sample_id(x) for x in extra]:
        if not sid or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="1×3 seg triptych for anchor_direction samples (+ optional extras).")
    ap.add_argument(
        "--extra",
        nargs="*",
        default=[],
        help="额外样本：完整 sample_id，或简写 img1582 / 1582（上海 AOI-4）",
    )
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="仅处理列出的样本（简写或全名）；若设置则忽略 anchor 目录与 fallback",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="输出目录（相对路径相对项目根；默认 paper_materials/anchor_direction）",
    )
    ap.add_argument(
        "--use-compare-folder",
        action="store_true",
        help="输出到 paper_materials/对比结果（无需在命令行输入中文路径）",
    )
    args = ap.parse_args()

    if not CKPT.exists():
        raise FileNotFoundError(CKPT)

    if args.only is not None and len(args.only) > 0:
        sample_ids = merge_ids([], list(args.only))
    else:
        sample_ids = parse_sample_ids_from_anchor_dir(ANCHOR_DIR)
        if not sample_ids:
            sample_ids = FALLBACK_SAMPLE_IDS
            print(f"[warn] no anchor_dir_*.png in {ANCHOR_DIR}, using fallback list ({len(sample_ids)} ids).")
        sample_ids = merge_ids(sample_ids, list(args.extra))

    if args.use_compare_folder:
        out_root = COMPARE_RESULT_DIR
    else:
        out_root = args.out_dir if args.out_dir is not None else OUT_DIR
        if not out_root.is_absolute():
            out_root = ROOT / out_root
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    failed: list[str] = []
    for sid in sample_ids:
        out = out_root / f"seg_triptych_{sid}.png"
        try:
            render_triptych(sid, model, device, out)
            print(f"Saved {out}")
        except FileNotFoundError as e:
            failed.append(sid)
            print(f"[skip] {e}")

    if failed:
        print("\n以下样本在下列路径未找到影像或厚掩膜（请确认是否已放入 filtered 数据集）：")
        print(f"  影像候选: {IMAGE_CANDIDATES}")
        print(f"  厚掩膜: {MASK_THICK}")
        for sid in failed:
            print(f"  - {sid}")


if __name__ == "__main__":
    main()
