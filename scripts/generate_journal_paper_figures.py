"""
Generate four journal-style figures (SpaceNet Shanghai + Vegas) for paper submission.
Outputs to paper_materials/journal_figures/ at 300 DPI.

Requires:
  - logs_shanghai/training_log.json, logs_shanghai_thick/training_log.json
  - eval_results/extended_optimization_v2/optimization_results.csv
  - checkpoints_shanghai/model_best_val_iou.pth, checkpoints_shanghai_thick/model_best_val_iou.pth
  - E:/Code/spacenet_filtered/{images,masks}, E:/Code/spacenet_filtered_thick/masks
  - E:/Code/spacenet/train/AOI2_Vegas/{images,masks_thick}

Optional:
  --sat2graph-png  Path to a pre-rendered Sat2Graph panel for Fig.4(c).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.road_extraction_model import RoadExtractionModel

# --- paths (override via env if needed) ---
SHANGHAI_IMG = Path("E:/Code/spacenet_filtered/images")
MASK_THIN = Path("E:/Code/spacenet_filtered/masks")
MASK_THICK = Path("E:/Code/spacenet_filtered_thick/masks_thick")
VEGAS_ROOT = Path("E:/Code/spacenet/train/AOI2_Vegas")
VEGAS_IMG = VEGAS_ROOT / "images"
VEGAS_MASK_THICK = VEGAS_ROOT / "masks_thick"

LOG_THIN = ROOT / "logs_shanghai" / "training_log.json"
LOG_THICK = ROOT / "logs_shanghai_thick" / "training_log.json"
OPT_CSV = ROOT / "eval_results" / "extended_optimization_v2" / "optimization_results.csv"
TOPO_CSV = ROOT / "logs_shanghai_thick_optimized_final" / "eval" / "topology_eval_per_sample.csv"

CKPT_THIN = ROOT / "checkpoints_shanghai" / "model_best_val_iou.pth"
CKPT_THICK = ROOT / "checkpoints_shanghai_thick" / "model_best_val_iou.pth"

OUT_DIR = ROOT / "paper_materials" / "journal_figures"
PRED_THRESH = 0.26

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica", "Liberation Sans"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.linewidth": 0.8,
    }
)


def _f(v: str) -> float:
    try:
        x = float(v)
        if math.isnan(x):
            return float("nan")
        return x
    except (TypeError, ValueError):
        return float("nan")


def _image_black_ratio(img: np.ndarray) -> float:
    gray = np.mean(img.astype(np.float32), axis=2)
    return float((gray < 8.0).mean())


def _mask_road_fraction(mask: np.ndarray) -> float:
    return float((mask > 0).mean())


def load_training_log(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_gt_mask(path: Path, size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    gt = np.asarray(Image.open(path).convert("L").resize(size, Image.NEAREST), dtype=np.uint8)
    return (gt > 127).astype(np.uint8)


def load_rgb(path: Path, size: Tuple[int, int] = (224, 224)) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    return np.asarray(Image.open(path).convert("RGB").resize(size, Image.BILINEAR))


def load_model(ckpt: Path) -> Tuple[RoadExtractionModel, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadExtractionModel(encoder="swin_tiny", num_classes=1, input_size=224)
    state = torch.load(str(ckpt), map_location=device)
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model, device


def infer_pred_mask(img: np.ndarray, model: RoadExtractionModel, device: torch.device) -> np.ndarray:
    pil = Image.fromarray(img).resize((224, 224), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.sigmoid(out["segmentation"])[0, 0].detach().cpu().numpy()
    return (prob > PRED_THRESH).astype(np.uint8)


def add_panel_label(ax, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        color="white",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="black", edgecolor="none", alpha=0.55),
    )


def build_postprocess_heatmap(
    csv_path: Path,
    stage: str = "coarse_subset",
    dir_stop: float = 0.2,
    min_path_len: int = 8,
) -> Tuple[np.ndarray, List[float], List[float]]:
    """strict APLS grid: rows = post_threshold, cols = endpoint_dist."""
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    pts_set: set = set()
    eds_set: set = set()
    cells: Dict[Tuple[float, float], float] = {}
    for r in rows:
        if r.get("stage") != stage:
            continue
        if abs(_f(r.get("dir_stop_eps", "nan")) - dir_stop) > 1e-6:
            continue
        if int(float(r.get("min_path_len", -1))) != min_path_len:
            continue
        pt = _f(r.get("post_threshold", "nan"))
        ed = _f(r.get("endpoint_dist", "nan"))
        sa = _f(r.get("strict_apls", "nan"))
        if math.isnan(pt) or math.isnan(ed) or math.isnan(sa):
            continue
        pts_set.add(round(pt, 4))
        eds_set.add(round(ed, 4))
        cells[(round(pt, 4), round(ed, 4))] = sa

    post_thresholds = sorted(pts_set)
    endpoint_dists = sorted(eds_set)
    if not post_thresholds or not endpoint_dists:
        raise ValueError("Empty heatmap from optimization CSV — check filters.")

    H = np.full((len(post_thresholds), len(endpoint_dists)), np.nan, dtype=np.float64)
    for i, pt in enumerate(post_thresholds):
        for j, ed in enumerate(endpoint_dists):
            v = cells.get((pt, ed))
            if v is not None:
                H[i, j] = v
    return H, post_thresholds, endpoint_dists


def select_complex_shanghai_sample(
    topo_csv: Path,
    prefer_id: Optional[str] = None,
) -> str:
    if prefer_id:
        return prefer_id
    rows = list(csv.DictReader(topo_csv.open(encoding="utf-8")))
    scored: List[Tuple[float, str]] = []
    for r in rows:
        sid = r["sample_id"]
        img_path = SHANGHAI_IMG / f"{sid}.png"
        mpath = MASK_THICK / f"{sid}.png"
        img = load_rgb(img_path)
        m = load_gt_mask(mpath)
        if img is None or m is None:
            continue
        br = _image_black_ratio(img)
        rf = _mask_road_fraction(m)
        if br > 0.12 or rf < 0.06:
            continue
        gn = int(float(r.get("gt_num_nodes", 0) or 0))
        ge = int(float(r.get("gt_num_edges", 0) or 0))
        score = gn * max(ge, 1) * (1.0 + rf) * (1.0 - br)
        scored.append((score, sid))
    scored.sort(key=lambda x: x[0], reverse=True)
    if not scored:
        raise RuntimeError("No valid Shanghai sample found for Fig.2/3 — check data paths.")
    return scored[0][1]


def select_full_network_sample(topo_csv: Path, exclude: str) -> str:
    """Prefer high road coverage + topology among filtered samples."""
    rows = list(csv.DictReader(topo_csv.open(encoding="utf-8")))
    best_sid = exclude
    best = -1.0
    for r in rows:
        sid = r["sample_id"]
        if sid == exclude:
            continue
        img_path = SHANGHAI_IMG / f"{sid}.png"
        mpath = MASK_THICK / f"{sid}.png"
        img = load_rgb(img_path)
        m = load_gt_mask(mpath)
        if img is None or m is None:
            continue
        br = _image_black_ratio(img)
        rf = _mask_road_fraction(m)
        if br > 0.14 or rf < 0.08:
            continue
        gn = int(float(r.get("gt_num_nodes", 0) or 0))
        ge = int(float(r.get("gt_num_edges", 0) or 0))
        score = rf * 3.0 + math.log1p(gn * max(ge, 1)) - br
        if score > best:
            best = score
            best_sid = sid
    return best_sid


def figure1_training_and_heatmap(out_path: Path) -> None:
    lt = load_training_log(LOG_THIN)
    lk = load_training_log(LOG_THICK)
    e_t = np.arange(1, len(lt) + 1)
    e_k = np.arange(1, len(lk) + 1)
    vt = [float(x["val_iou"]) for x in lt]
    vk = [float(x["val_iou"]) for x in lk]
    trt = [float(x["train_iou"]) for x in lt]
    trk = [float(x["train_iou"]) for x in lk]

    H, pts, eds = build_postprocess_heatmap(OPT_CSV)

    fig = plt.figure(figsize=(12.5, 4.8), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.plot(e_t, vt, color="#1f77b4", linewidth=1.8, label="Val IoU (thin labels)")
    ax0.plot(e_t, trt, color="#aec7e8", linewidth=1.2, linestyle="--", label="Train IoU (thin labels)")
    ax0.plot(e_k, vk, color="#d62728", linewidth=1.8, label="Val IoU (thick labels, ours)")
    ax0.plot(e_k, trk, color="#ff9896", linewidth=1.2, linestyle="--", label="Train IoU (thick labels, ours)")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("IoU")
    ax0.grid(True, alpha=0.35, linestyle=":")
    ax0.legend(loc="lower right", fontsize=8, framealpha=0.92)
    add_panel_label(ax0, "(a)")
    ax0.set_title("Training convergence (Shanghai AOI-4)")

    ax1 = fig.add_subplot(gs[0, 1])
    vmin = np.nanmin(H)
    vmax = np.nanmax(H)
    im = ax1.imshow(
        H,
        aspect="auto",
        cmap="viridis",
        norm=Normalize(vmin=vmin, vmax=vmax),
        origin="upper",
    )
    ax1.set_xticks(np.arange(len(eds)))
    ax1.set_xticklabels([str(int(x)) if float(x).is_integer() else str(x) for x in eds])
    ax1.set_yticks(np.arange(len(pts)))
    ax1.set_yticklabels([f"{x:.2f}" for x in pts])
    ax1.set_xlabel(r"Endpoint merge distance $d_{\mathrm{end}}$ (px)")
    ax1.set_ylabel(r"Post threshold $\tau$")
    cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label("Strict APLS (val subset)")
    add_panel_label(ax1, "(b)")
    ax1.set_title("Post-processing search (coarse grid, $\\varepsilon_{\\mathrm{dir}}=0.2$)")

    fig.suptitle("Fig.1  Training dynamics and post-processing sensitivity", fontsize=11, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure2_intersection_compare(
    sample_id: str,
    out_path: Path,
    model_thin: RoadExtractionModel,
    model_thick: RoadExtractionModel,
    dev: torch.device,
) -> None:
    img_path = SHANGHAI_IMG / f"{sample_id}.png"
    gt_thin = load_gt_mask(MASK_THIN / f"{sample_id}.png")
    gt_thick = load_gt_mask(MASK_THICK / f"{sample_id}.png")
    img = load_rgb(img_path)
    if img is None or gt_thin is None or gt_thick is None:
        raise FileNotFoundError(f"Missing image/masks for {sample_id}")

    p_thin = infer_pred_mask(img, model_thin, dev)
    p_thick = infer_pred_mask(img, model_thick, dev)

    fig, axes = plt.subplots(2, 2, figsize=(10.2, 10.0), facecolor="white")
    panels = [
        (axes[0, 0], img, None, "Satellite image"),
        (axes[0, 1], img, gt_thick, "Ground truth (thick mask)"),
        (axes[1, 0], img, p_thin, "Baseline (thin-label training)"),
        (axes[1, 1], img, p_thick, "Ours (thick-label + intersection-anchored)"),
    ]
    labels = ["(a)", "(b)", "(c)", "(d)"]
    for k, (ax, im, mask, title) in enumerate(panels):
        ax.imshow(im, alpha=1.0)
        if mask is not None:
            if title.startswith("Ground"):
                ax.contour(mask, levels=[0.5], colors=["#00aa44"], linewidths=1.4)
            else:
                ax.contour(mask, levels=[0.5], colors=["#cc0000"], linewidths=1.8)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=10)
        add_panel_label(ax, labels[k])

    fig.suptitle(
        "Fig.2  Complex urban topology (multi-leg junctions, shadows)\n"
        f"Sample: {sample_id}",
        fontsize=11,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure3_full_network(
    sample_id: str,
    out_path: Path,
    model_thick: RoadExtractionModel,
    dev: torch.device,
) -> None:
    img_path = SHANGHAI_IMG / f"{sample_id}.png"
    gt_thick = load_gt_mask(MASK_THICK / f"{sample_id}.png")
    img = load_rgb(img_path)
    if img is None or gt_thick is None:
        raise FileNotFoundError(f"Missing data for {sample_id}")
    pred = infer_pred_mask(img, model_thick, dev)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), facecolor="white")
    axes[0].imshow(img)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    for s in axes[0].spines.values():
        s.set_visible(False)
    axes[0].set_title("Satellite image")
    add_panel_label(axes[0], "(a)")

    axes[1].imshow(img, alpha=0.92)
    axes[1].contour(gt_thick, levels=[0.5], colors=["#00aa44"], linewidths=1.0, alpha=0.45)
    axes[1].contour(pred, levels=[0.5], colors=["#cc0000"], linewidths=2.0)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    for s in axes[1].spines.values():
        s.set_visible(False)
    axes[1].set_title("Ours — full road network (overlay: GT green faint, prediction red)")
    add_panel_label(axes[1], "(b)")

    fig.suptitle(
        "Fig.3  Dense road network extraction (Shanghai AOI-4)\n" f"Sample: {sample_id}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure4_vegas(
    vegas_id: str,
    out_path: Path,
    model_thick: RoadExtractionModel,
    dev: torch.device,
    sat2graph_png: Optional[Path],
) -> None:
    img_path = VEGAS_IMG / f"{vegas_id}.png"
    gt = load_gt_mask(VEGAS_MASK_THICK / f"{vegas_id}.png")
    img = load_rgb(img_path)
    if img is None or gt is None:
        raise FileNotFoundError(f"Vegas data missing for {vegas_id}")

    pred = infer_pred_mask(img, model_thick, dev)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.8), facecolor="white")

    axes[0].imshow(img, alpha=0.95)
    axes[0].contour(gt, levels=[0.5], colors=["#00aa44"], linewidths=1.6)
    axes[0].set_title("Ground truth (Las Vegas AOI-2)")
    add_panel_label(axes[0], "(a)")

    axes[1].imshow(img, alpha=0.95)
    axes[1].contour(pred, levels=[0.5], colors=["#cc0000"], linewidths=2.0)
    axes[1].set_title("Ours (Shanghai-trained generalization)")
    add_panel_label(axes[1], "(b)")

    if sat2graph_png is not None and sat2graph_png.exists():
        s2 = np.asarray(Image.open(sat2graph_png).convert("RGB").resize((224, 224), Image.BILINEAR))
        axes[2].imshow(s2)
        axes[2].set_title("Sat2Graph")
    else:
        axes[2].imshow(img, alpha=0.35)
        axes[2].text(
            0.5,
            0.5,
            "Sat2Graph\n(export baseline PNG and pass\n--sat2graph-png)",
            ha="center",
            va="center",
            fontsize=10,
            color="0.1",
            transform=axes[2].transAxes,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.5", alpha=0.92),
        )
        axes[2].set_title("Sat2Graph (placeholder)")
    add_panel_label(axes[2], "(c)")

    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle(
        "Fig.4  Cross-city generalization (Las Vegas grid roads)\n" f"Sample: {vegas_id}",
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig2-sample", default=None, help="Shanghai sample_id for Fig.2 (auto if omitted)")
    ap.add_argument("--fig3-sample", default=None, help="Shanghai sample_id for Fig.3 (auto if omitted)")
    ap.add_argument("--vegas-sample", default="SN3_roads_train_AOI_2_Vegas_PS-RGB_img601")
    ap.add_argument("--sat2graph-png", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()

    sid2 = args.fig2_sample or select_complex_shanghai_sample(TOPO_CSV)
    sid3 = args.fig3_sample or select_full_network_sample(TOPO_CSV, exclude=sid2)

    if not CKPT_THIN.exists() or not CKPT_THICK.exists():
        raise FileNotFoundError(f"Need checkpoints: {CKPT_THIN} and {CKPT_THICK}")

    m_thin, dev = load_model(CKPT_THIN)
    m_thick, _ = load_model(CKPT_THICK)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)

    figure1_training_and_heatmap(out / "Fig1_training_and_postprocess_heatmap.png")
    print(f"[OK] {out / 'Fig1_training_and_postprocess_heatmap.png'}")

    figure2_intersection_compare(sid2, out / "Fig2_complex_intersection_compare.png", m_thin, m_thick, dev)
    print(f"[OK] {out / 'Fig2_complex_intersection_compare.png'}  sample={sid2}")

    figure3_full_network(sid3, out / "Fig3_full_network_shanghai.png", m_thick, dev)
    print(f"[OK] {out / 'Fig3_full_network_shanghai.png'}  sample={sid3}")

    figure4_vegas(args.vegas_sample, out / "Fig4_vegas_generalization.png", m_thick, dev, args.sat2graph_png)
    print(f"[OK] {out / 'Fig4_vegas_generalization.png'}  vegas={args.vegas_sample}")

    print("\nDone. Replace Fig.4(c) by running with --sat2graph-png <your_baseline.png> if available.")


if __name__ == "__main__":
    main()
