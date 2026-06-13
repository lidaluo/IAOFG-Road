from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from models.road_extraction_model import RoadExtractionModel
CSV_PATH = ROOT / "logs_shanghai_thick_optimized_final" / "eval" / "topology_eval_per_sample.csv"
GEO_DIR = ROOT / "logs_shanghai_thick_optimized_final" / "eval" / "vectors_geojson"
OUT_DIR = ROOT / "paper_materials" / "best_cases"
CKPT = ROOT / "checkpoints_shanghai_thick" / "model_best_val_iou.pth"
PRED_THRESH = 0.26

IMAGE_CANDIDATES = [
    Path("E:/Code/spacenet_filtered/images"),
    Path("E:/Code/spacenet_filtered_thick/images"),
    Path("E:/Code/spacenet/train/AOI2_Vegas/images"),
]


def _f(v: str) -> float:
    vv = (v or "").strip().lower()
    if vv in {"", "nan", "none"}:
        return float("nan")
    return float(v)


def load_rows() -> List[Dict[str, str]]:
    with CSV_PATH.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _image_black_ratio(img: np.ndarray) -> float:
    # 近黑像素比例，用于过滤“半张黑块”样本
    gray = np.mean(img.astype(np.float32), axis=2)
    return float((gray < 8.0).mean())


def _mask_road_fraction(mask: np.ndarray) -> float:
    return float((mask > 0).mean())


def select_best(rows: List[Dict[str, str]], top_k: int = 5) -> List[Dict[str, str]]:
    scored = []
    for r in rows:
        sid = r["sample_id"]
        img_path = find_image(sid)
        gt_mask = load_gt_mask(sid)
        if img_path is None or gt_mask is None:
            continue

        img = np.array(Image.open(img_path).convert("RGB").resize((224, 224), Image.BILINEAR))
        black_ratio = _image_black_ratio(img)
        road_frac = _mask_road_fraction(gt_mask)

        # 审美硬筛：去掉黑块占比高、道路过少样本
        if black_ratio > 0.12:
            continue
        if road_frac < 0.06:
            continue

        iou = _f(r.get("pixel_iou", "nan"))
        apls = _f(r.get("apls", "nan"))
        if np.isnan(iou) and np.isnan(apls):
            continue

        # 联合评分：优先 APLS/IoU，同时偏好道路丰富且非黑块影像
        score = (
            (apls if not np.isnan(apls) else 0.0) * 0.45
            + (iou if not np.isnan(iou) else 0.0) * 0.35
            + road_frac * 0.25
            - black_ratio * 0.60
        )
        scored.append((score, r, apls, iou, road_frac, black_ratio))

    scored.sort(key=lambda x: x[0], reverse=True)
    picked = [x[1] for x in scored[:top_k]]

    if len(picked) < top_k:
        # 兜底：放宽条件，但仍按黑块比例从低到高优先
        loose = []
        for r in rows:
            sid = r["sample_id"]
            img_path = find_image(sid)
            gt_mask = load_gt_mask(sid)
            if img_path is None or gt_mask is None:
                continue
            img = np.array(Image.open(img_path).convert("RGB").resize((224, 224), Image.BILINEAR))
            black_ratio = _image_black_ratio(img)
            road_frac = _mask_road_fraction(gt_mask)
            iou = _f(r.get("pixel_iou", "nan"))
            apls = _f(r.get("apls", "nan"))
            if np.isnan(iou) and np.isnan(apls):
                continue
            score = (apls if not np.isnan(apls) else 0.0) * 0.5 + (iou if not np.isnan(iou) else 0.0) * 0.3 + road_frac * 0.2 - black_ratio * 0.8
            loose.append((score, r))
        loose.sort(key=lambda x: x[0], reverse=True)
        exist = {p["sample_id"] for p in picked}
        for _, r in loose:
            if r["sample_id"] in exist:
                continue
            picked.append(r)
            exist.add(r["sample_id"])
            if len(picked) >= top_k:
                break

    return picked[:top_k]


def find_image(sample_id: str) -> Path | None:
    for d in IMAGE_CANDIDATES:
        p = d / f"{sample_id}.png"
        if p.exists():
            return p
    return None


def load_geo(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_features(fc: Dict) -> Tuple[List[np.ndarray], List[Tuple[float, float]]]:
    lines: List[np.ndarray] = []
    points: List[Tuple[float, float]] = []
    for feat in fc.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates")
        if gtype == "LineString" and isinstance(coords, list) and len(coords) >= 2:
            arr = np.array(coords, dtype=np.float32)
            lines.append(arr)
        elif gtype == "Point" and isinstance(coords, list) and len(coords) >= 2:
            points.append((float(coords[0]), float(coords[1])))
    return lines, points


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadExtractionModel(encoder="swin_tiny", num_classes=1, input_size=224)
    state = torch.load(str(CKPT), map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, device


def infer_pred_mask(img: np.ndarray, model: RoadExtractionModel, device: torch.device) -> np.ndarray:
    # 与训练口径一致：仅缩放到 224，不做额外标准化
    pil = Image.fromarray(img).resize((224, 224), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.sigmoid(out["segmentation"])[0, 0].detach().cpu().numpy()
    return (prob > PRED_THRESH).astype(np.uint8)


def load_gt_mask(sample_id: str) -> np.ndarray | None:
    p = Path("E:/Code/spacenet_filtered/masks") / f"{sample_id}.png"
    if not p.exists():
        return None
    gt = np.asarray(Image.open(p).convert("L").resize((224, 224), Image.NEAREST), dtype=np.uint8)
    return (gt > 127).astype(np.uint8)


def draw_lines(ax, lines: List[np.ndarray], color: str, lw: float, alpha: float = 1.0):
    for arr in lines:
        ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=lw, alpha=alpha)


def draw_nodes(ax, points: List[Tuple[float, float]], color: str = "#0066ff", size: int = 18):
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, s=size, c=color, marker="o")


def make_match_mask(h: int, w: int, pred_lines: List[np.ndarray], gt_lines: List[np.ndarray]) -> np.ndarray:
    pred_img = Image.new("L", (w, h), 0)
    gt_img = Image.new("L", (w, h), 0)
    draw_pred = ImageDraw.Draw(pred_img)
    draw_gt = ImageDraw.Draw(gt_img)

    for arr in pred_lines:
        pts = [tuple(map(float, p)) for p in arr.tolist()]
        if len(pts) >= 2:
            draw_pred.line(pts, fill=255, width=2)
    for arr in gt_lines:
        pts = [tuple(map(float, p)) for p in arr.tolist()]
        if len(pts) >= 2:
            draw_gt.line(pts, fill=255, width=1)

    pred_mask = np.array(pred_img, dtype=np.uint8)
    gt_mask = np.array(gt_img, dtype=np.uint8)
    return np.bitwise_and(pred_mask, gt_mask)


def render_case(row: Dict[str, str], model: RoadExtractionModel, device: torch.device) -> Path | None:
    sid = row["sample_id"]
    pred_path = GEO_DIR / f"{sid}_pred.geojson"
    gt_path = GEO_DIR / f"{sid}_gt.geojson"
    img_path = find_image(sid)
    if (not pred_path.exists()) or (not gt_path.exists()) or (img_path is None):
        return None

    img = np.array(Image.open(img_path).convert("RGB"))
    img = np.asarray(Image.fromarray(img).resize((224, 224), Image.BILINEAR))
    h, w = img.shape[:2]
    pred_mask = infer_pred_mask(img, model, device)
    gt_mask = load_gt_mask(sid)
    if gt_mask is None:
        return None

    pred_fc = load_geo(pred_path)
    gt_fc = load_geo(gt_path)
    pred_lines, pred_nodes = parse_features(pred_fc)
    gt_lines, _ = parse_features(gt_fc)

    match_mask = np.logical_and(pred_mask > 0, gt_mask > 0).astype(np.float32)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="white")
    for ax in axes:
        ax.set_facecolor("white")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    # panel 1: raw image
    axes[0].imshow(img)
    axes[0].set_title("Satellite Image", fontsize=11)

    # panel 2: predicted road network overlay
    axes[1].imshow(img, alpha=0.9)
    axes[1].contour(pred_mask, levels=[0.5], colors=["#ff0000"], linewidths=2.0)
    draw_nodes(axes[1], pred_nodes, color="#0066ff", size=18)
    axes[1].set_title("Predicted Road Network", fontsize=11)

    # panel 3: 先铺黄色半透明填充（GT∩Pred），再画轮廓，避免“看不见黄点”
    axes[2].imshow(img, alpha=0.92)
    yellow = np.zeros((h, w, 4), dtype=np.float32)
    yellow[..., 0] = 1.0
    yellow[..., 1] = 0.9
    yellow[..., 2] = 0.05
    yellow[..., 3] = match_mask * 0.5
    axes[2].imshow(yellow)
    axes[2].contour(gt_mask, levels=[0.5], colors=["#00aa00"], linewidths=1.3, alpha=0.9)
    axes[2].contour(pred_mask, levels=[0.5], colors=["#cc0000"], linewidths=1.5, alpha=0.95)
    draw_nodes(axes[2], pred_nodes, color="#0066ff", size=18)
    axes[2].set_title("Overlay (yellow fill = GT ∩ Pred)", fontsize=11)

    iou = _f(row.get("pixel_iou", "nan"))
    apls = _f(row.get("apls", "nan"))
    ann = f"IoU={iou:.2f}, APLS={apls:.2f}" if (not np.isnan(iou) and not np.isnan(apls)) else "IoU/APLS=N/A"
    fig.text(
        0.015,
        0.965,
        ann,
        fontsize=11,
        color="black",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="gray", alpha=0.95),
    )
    fig.suptitle(f"Best-Case Topology Visualization: {sid}", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"topo_{sid}_iou{iou:.2f}_apls{apls:.2f}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out


def main():
    rows = load_rows()
    best = select_best(rows, top_k=5)
    model, device = load_model()
    saved = []
    print("Selected samples:")
    for r in best:
        print(f"- {r['sample_id']} (IoU={_f(r.get('pixel_iou','nan')):.3f}, APLS={_f(r.get('apls','nan')):.3f})")
    for r in best:
        out = render_case(r, model, device)
        if out is not None:
            saved.append(out)
    print("Saved best-case figures:")
    for p in saved:
        print(f"- {p}")


if __name__ == "__main__":
    main()
