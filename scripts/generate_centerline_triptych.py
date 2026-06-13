"""
Five (or N) triptychs: (a) RGB  (b) GT centerline (green)  (c) Predicted centerline (yellow edges, red nodes).

Pipeline: thick binary mask → medial-axis skeleton → NetworkX graph → merge nearby nodes →
collapse degree-2 chain nodes. See postprocess/centerline_from_mask.py.

Outputs: paper_materials/centerline_figures/
Optional: *.graph.json per sample (nodes + edges + polylines)
"""

from __future__ import annotations

import argparse
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
from postprocess.centerline_from_mask import (
    draw_centerline_overlay,
    mask_to_centerline_graph,
    save_graph_json,
)

CKPT = ROOT / "checkpoints_shanghai_thick" / "model_best_val_iou.pth"
SHANGHAI_IMG = Path("E:/Code/spacenet_filtered/images")
MASK_THICK = Path("E:/Code/spacenet_filtered_thick/masks_thick")
IMAGE_CANDIDATES = [SHANGHAI_IMG, Path("E:/Code/spacenet_filtered_thick/images")]
OUT_DIR = ROOT / "paper_materials" / "centerline_figures"

PRED_THRESH = 0.26

DEFAULT_FIVE = [
    "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1607",
    "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1791",
    "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img525",
    "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1681",
    "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img919",
]


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


def infer_mask(img_rgb: np.ndarray, model: RoadExtractionModel, device: torch.device) -> np.ndarray:
    pil = Image.fromarray(img_rgb).resize((224, 224), Image.BILINEAR)
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    x = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        prob = torch.sigmoid(out["segmentation"])[0, 0].detach().cpu().numpy()
    return (prob > PRED_THRESH).astype(np.uint8)


def render_triptych(
    sample_id: str,
    img: np.ndarray,
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
    out_png: Path,
    save_json: bool,
) -> None:
    G_gt = mask_to_centerline_graph(gt_mask, open_kernel=0)
    G_pr = mask_to_centerline_graph(pred_mask, open_kernel=0)

    fig, axes = plt.subplots(1, 3, figsize=(12.9, 4.1), facecolor="white")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    axes[0].imshow(img)
    axes[0].set_title("(a) Satellite image", fontsize=11)

    axes[1].imshow(img, alpha=0.96)
    draw_centerline_overlay(
        axes[1],
        G_gt,
        edge_color="#00cc66",
        edge_lw=1.15,
        node_color="#006633",
        node_size=16.0,
    )
    axes[1].set_title("(b) GT centerline (graph)", fontsize=11)

    axes[2].imshow(img, alpha=0.96)
    draw_centerline_overlay(
        axes[2],
        G_pr,
        edge_color="#ffcc00",
        edge_lw=1.15,
        node_color="#ff2222",
        node_size=16.0,
    )
    axes[2].set_title("(c) Ours — centerline graph", fontsize=11)

    fig.suptitle(
        f"Centerline topology — {sample_id}\n"
        f"(skeleton / medial axis + node merge + degree-2 simplification)",
        fontsize=10,
        y=1.02,
    )
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if save_json:
        save_graph_json(G_gt, out_png.with_suffix("").as_posix() + "_gt.graph.json")
        save_graph_json(G_pr, out_png.with_suffix("").as_posix() + "_pred.graph.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--save-json", action="store_true", help="Export NetworkX as JSON (nodes + edges)")
    ap.add_argument(
        "--samples",
        nargs="*",
        default=DEFAULT_FIVE,
        help="Sample IDs (default: 5 Shanghai AOI-4 stems)",
    )
    args = ap.parse_args()

    if not CKPT.exists():
        raise FileNotFoundError(CKPT)

    out = args.out_dir
    if not out.is_absolute():
        out = ROOT / out

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(device)

    for sid in args.samples:
        ip = find_image(sid)
        gt = load_gt_thick(sid)
        if ip is None or gt is None:
            print(f"[skip] missing data {sid}")
            continue
        img = np.asarray(Image.open(ip).convert("RGB").resize((224, 224), Image.BILINEAR))
        pred = infer_mask(img, model, device)
        png = out / f"centerline_triptych_{sid}.png"
        render_triptych(sid, img, gt, pred, png, save_json=args.save_json)
        print(f"Saved {png}")

    print("Done.")


if __name__ == "__main__":
    main()
