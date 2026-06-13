"""
WidthConsistentConnector 单 tile 演示：读 seg.npy（或二值图）→ 桥接 → 可视化与指标（可选 GT）。

示例：
  python scripts/demo_width_consistent_connector.py ^
    --seg eval_outputs/npy_tiles/region_18_seg.npy ^
    --gt-graph E:/datasets/cityscaledataset/cityscale/20cities/region_18_refine_gt_graph.p ^
    --out eval_outputs/wcc_region_18.png
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from iaof_postprocessing.width_consistent_connector import WidthConsistentConnector
from utils.cityscale_graph import default_gt_pickle_path, load_cityscale_gt_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=str, required=True, help="道路概率/二值 .npy 路径，H×W")
    ap.add_argument("--gt-graph", type=str, default=None, help="GT pickle（算 APLS/TOPO）")
    ap.add_argument("--cityscale-root", type=str, default=None)
    ap.add_argument("--region", type=int, default=None)
    ap.add_argument("--hw", type=int, nargs=2, default=[2048, 2048])
    ap.add_argument("--out", type=str, default="eval_outputs/wcc_demo.png")
    ap.add_argument(
        "--original-image",
        type=str,
        default=None,
        help="可选：原始遥感 RGB 影像路径（OpenCV 读入 BGR 后转 RGB），叠加在右侧主视图中",
    )
    ap.add_argument("--max-bridge-gap", type=float, default=10.0)
    ap.add_argument("--width-rel-tol", type=float, default=0.3)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    seg = np.load(args.seg)
    gt_path = args.gt_graph
    if gt_path is None and args.cityscale_root is not None and args.region is not None:
        gt_path = default_gt_pickle_path(args.cityscale_root, int(args.region))
    gt = load_cityscale_gt_graph(gt_path) if gt_path else None

    h, w = int(args.hw[0]), int(args.hw[1])
    outp = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)

    orig_rgb: np.ndarray | None = None
    if args.original_image:
        ip = args.original_image if os.path.isabs(args.original_image) else os.path.join(PROJECT_ROOT, args.original_image)
        bgr = cv2.imread(ip, cv2.IMREAD_COLOR)
        if bgr is None:
            raise SystemExit(f"无法读取影像: {ip}")
        orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    conn = WidthConsistentConnector(
        seg,
        max_bridge_gap=float(args.max_bridge_gap),
        width_rel_tol=float(args.width_rel_tol),
        thresh=float(args.thresh),
        debug=True,
    )
    nodes, edges, vis_path, metrics = conn.process(
        gt_graph=gt,
        canvas_hw=(h, w),
        visualization_path=outp,
        original_image=orig_rgb,
    )
    print(f"nodes={len(nodes)} edges={len(edges)}")
    print(f"metrics={metrics}")
    print(f"[Vis] {vis_path}")


if __name__ == "__main__":
    main()
