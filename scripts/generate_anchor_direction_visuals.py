"""
Select samples with many intersections + high APLS, then export 4-panel figures:
  (a) RGB  (b) Ground-truth road mask  (c) Predicted intersection heatmap
  (d) Schematic: dim RGB + heatmap + direction field arrows (toward anchors)

Outputs: paper_materials/anchor_direction/*.png
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import maximum_filter

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.road_extraction_model import RoadExtractionModel

CSV_PATH = ROOT / "logs_shanghai_thick_optimized_final" / "eval" / "topology_eval_per_sample.csv"
OUT_DIR = ROOT / "paper_materials" / "anchor_direction"
CKPT = ROOT / "checkpoints_shanghai_thick" / "model_best_val_iou.pth"

SHANGHAI_IMG = Path("E:/Code/spacenet_filtered/images")
MASK_THICK = Path("E:/Code/spacenet_filtered_thick/masks_thick")

IMAGE_CANDIDATES = [
    SHANGHAI_IMG,
    Path("E:/Code/spacenet_filtered_thick/images"),
]


def _f(v: str) -> float:
    try:
        x = float(v)
        if math.isnan(x):
            return float("nan")
        return x
    except (TypeError, ValueError):
        return float("nan")


def _black_ratio(img: np.ndarray) -> float:
    gray = np.mean(img.astype(np.float32), axis=2)
    return float((gray < 8.0).mean())


def _road_frac(mask: np.ndarray) -> float:
    return float((mask > 0).mean())


def find_image(sample_id: str) -> Optional[Path]:
    for d in IMAGE_CANDIDATES:
        p = d / f"{sample_id}.png"
        if p.exists():
            return p
    return None


def load_gt_thick(sample_id: str) -> Optional[np.ndarray]:
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


def infer_maps(
    img_rgb: np.ndarray, model: RoadExtractionModel, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """seg prob, inter prob, u, v (tanh, same as eval_topology)."""
    pil = Image.fromarray(img_rgb).resize((224, 224), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        seg = torch.sigmoid(out["segmentation"])[0, 0].detach().cpu().numpy()
        inter = torch.sigmoid(out["intersection"])[0, 0].detach().cpu().numpy()
        ori = torch.tanh(out["orientation"][0, 0:2]).detach().cpu().numpy()
    u, v = ori[0], ori[1]
    return seg, inter, u, v


def select_samples(rows: List[dict], top_k: int = 5, min_apls: float = 0.92, min_nodes: int = 40) -> List[dict]:
    """Prefer many intersections (nodes/edges) and high APLS; filter bad images."""
    scored: List[Tuple[float, dict]] = []
    for r in rows:
        apls = _f(r.get("apls", "nan"))
        if np.isnan(apls) or apls < min_apls:
            continue
        gn = int(float(r.get("gt_num_nodes", 0) or 0))
        ge = int(float(r.get("gt_num_edges", 0) or 0))
        if gn < min_nodes:
            continue
        sid = r["sample_id"]
        ip = find_image(sid)
        m = load_gt_thick(sid)
        if ip is None or m is None:
            continue
        img = np.asarray(Image.open(ip).convert("RGB").resize((224, 224), Image.BILINEAR))
        br = _black_ratio(img)
        rf = _road_frac(m)
        # 厚掩膜道路占比常偏低；交叉口极多时仍值得展示
        if br > 0.12:
            continue
        if rf < 0.06 and gn < 80:
            continue
        if rf < 0.015:
            continue
        score = float(gn) * float(max(ge, 1)) * apls * (1.0 - br * 0.5)
        scored.append((score, r))

    scored.sort(key=lambda x: x[0], reverse=True)
    out = [x[1] for x in scored[:top_k]]
    if len(out) >= top_k:
        return out
    # 补足：放宽 APLS / 节点数门槛
    exist = {r["sample_id"] for r in out}
    relaxed = select_samples_relaxed(rows, top_k * 2, min_apls=max(0.85, min_apls - 0.06))
    for r in relaxed:
        if r["sample_id"] in exist:
            continue
        out.append(r)
        exist.add(r["sample_id"])
        if len(out) >= top_k:
            break
    return out[:top_k]


def select_samples_relaxed(rows: List[dict], top_k: int, min_apls: float) -> List[dict]:
    scored = []
    for r in rows:
        apls = _f(r.get("apls", "nan"))
        if np.isnan(apls) or apls < min_apls:
            continue
        gn = int(float(r.get("gt_num_nodes", 0) or 0))
        ge = int(float(r.get("gt_num_edges", 0) or 0))
        if gn < 25:
            continue
        sid = r["sample_id"]
        ip = find_image(sid)
        m = load_gt_thick(sid)
        if ip is None or m is None:
            continue
        img = np.asarray(Image.open(ip).convert("RGB").resize((224, 224), Image.BILINEAR))
        br = _black_ratio(img)
        rf = _road_frac(m)
        if br > 0.16:
            continue
        if rf < 0.05 and gn < 50:
            continue
        if rf < 0.012:
            continue
        score = float(gn) * float(max(ge, 1)) * apls
        scored.append((score, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:top_k]]


def render_figure(
    row: dict,
    model: RoadExtractionModel,
    device: torch.device,
    out_path: Path,
    stride: int = 10,
) -> None:
    sid = row["sample_id"]
    ip = find_image(sid)
    gt = load_gt_thick(sid)
    if ip is None or gt is None:
        raise FileNotFoundError(sid)

    img = np.asarray(Image.open(ip).convert("RGB").resize((224, 224), Image.BILINEAR))
    h, w = img.shape[:2]
    seg, inter, u, v = infer_maps(img, model, device)
    road = (seg > 0.26).astype(np.float32)

    fig, axes = plt.subplots(1, 4, figsize=(16.5, 4.2), facecolor="white")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    axes[0].imshow(img)
    axes[0].set_title("(a) Satellite image", fontsize=11)

    axes[1].imshow(img, alpha=0.92)
    axes[1].contour(gt, levels=[0.5], colors=["#00aa44"], linewidths=1.6)
    axes[1].set_title("(b) Ground truth (thick mask)", fontsize=11)

    axes[2].imshow(img, alpha=0.55)
    hi = axes[2].imshow(inter, cmap="inferno", vmin=0, vmax=1, alpha=0.92)
    axes[2].set_title("(c) Intersection heatmap (predicted)", fontsize=11)
    plt.colorbar(hi, ax=axes[2], fraction=0.046, pad=0.02)

    # (d) Combined: dim image + heatmap + quiver (direction toward anchors)
    base = (img.astype(np.float32) * 0.38 + 255 * 0.62).clip(0, 255).astype(np.uint8)
    axes[3].imshow(base)
    axes[3].imshow(inter, cmap="magma", vmin=0, vmax=1, alpha=0.42)
    # 交叉口锚点：热图局部极大（示意“锚定”位置）
    mx = maximum_filter(inter, size=9)
    local_max = (inter >= mx - 1e-5) & (inter > 0.35) & (road > 0.4)
    py, px = np.where(local_max)
    if len(px) > 0:
        order = np.argsort(-inter[py, px])[:16]
        axes[3].scatter(
            px[order],
            py[order],
            s=22,
            facecolors="white",
            edgecolors="#003344",
            linewidths=0.35,
            zorder=5,
        )
    yy, xx = np.meshgrid(np.arange(0, h, stride), np.arange(0, w, stride), indexing="ij")
    U = u[yy, xx]
    V = v[yy, xx]
    m = (road[yy, xx] > 0) & (seg[yy, xx] > 0.12)
    if m.any():
        axes[3].quiver(
            xx[m],
            yy[m],
            U[m],
            V[m],
            color="#00e5ff",
            angles="xy",
            scale_units="xy",
            scale=2.8,
            width=0.004,
            headwidth=3.2,
            headlength=4,
            alpha=0.92,
            zorder=4,
        )
    axes[3].set_title("(d) Anchor heatmap + direction field", fontsize=11)

    iou = _f(row.get("pixel_iou", "nan"))
    apls = _f(row.get("apls", "nan"))
    gn = int(float(row.get("gt_num_nodes", 0) or 0))
    ge = int(float(row.get("gt_num_edges", 0) or 0))
    ann = f"IoU={iou:.2f}, APLS={apls:.2f}, GT nodes={gn}, edges={ge}"
    fig.suptitle(f"Intersection-anchored direction guidance — {sid}\n{ann}", fontsize=11, y=1.02)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    if not CKPT.exists():
        raise FileNotFoundError(CKPT)
    rows = list(csv.DictReader(CSV_PATH.open(encoding="utf-8")))
    picked = select_samples(rows, top_k=5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    print("Selected (intersection-rich + high APLS):")
    for r in picked:
        print(
            f"  {r['sample_id']}  APLS={_f(r.get('apls','nan')):.3f}  "
            f"nodes={r.get('gt_num_nodes')} edges={r.get('gt_num_edges')}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for r in picked:
        sid = r["sample_id"]
        apls = _f(r.get("apls", "nan"))
        gn = int(float(r.get("gt_num_nodes", 0) or 0))
        safe = f"anchor_dir_{sid}_apls{apls:.2f}_gn{gn}.png"
        out = OUT_DIR / safe
        render_figure(r, model, device, out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
