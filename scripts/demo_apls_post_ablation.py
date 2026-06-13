"""
单张 seg/orient 上消融：baseline / +DP / +soft / +both（不改 evaluate_cityscale）。

用法（项目根）：
  python scripts/demo_apls_post_ablation.py \\
    --seg eval_outputs/npy_tiles/region_18_seg.npy \\
    --orient eval_outputs/npy_tiles/region_18_orient.npy \\
    --gt-graph <path>/region_18_refine_gt_graph.p \\
    --hw 2048 2048 --thr 0.25
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from iaof_postprocessing.pipeline import compute_topology_metrics
from iaof_postprocessing.pipeline_apls_opt import run_pipeline_with_apls_post
from utils.cityscale_graph import load_cityscale_gt_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--orient", required=True)
    ap.add_argument("--gt-graph", required=True)
    ap.add_argument("--hw", type=int, nargs=2, default=[2048, 2048])
    ap.add_argument("--thr", type=float, default=0.25)
    ap.add_argument("--orient-compat", type=float, default=0.10)
    ap.add_argument("--orient-radius", type=float, default=48.0)
    ap.add_argument("--dbscan-eps", type=float, default=8.0)
    args = ap.parse_args()

    seg = np.load(args.seg)
    orient = np.load(args.orient)
    h, w = int(args.hw[0]), int(args.hw[1])

    gt = load_cityscale_gt_graph(args.gt_graph)

    base_kw = dict(
        connection_mode="orient",
        thresh=float(args.thr),
        orient_compat=float(args.orient_compat),
        orient_neighbor_radius=float(args.orient_radius),
        dbscan_eps=float(args.dbscan_eps),
        debug=False,
    )

    runs = [
        ("Baseline（原版 orient 硬阈值）", dict(enable_soft_direction=False, enable_dp_simplify=False)),
        ("+ DP simplify", dict(enable_soft_direction=False, enable_dp_simplify=True)),
        ("+ Soft direction", dict(enable_soft_direction=True, enable_dp_simplify=False)),
        ("+ Both", dict(enable_soft_direction=True, enable_dp_simplify=True)),
    ]

    rows = []
    for name, flags in runs:
        G = run_pipeline_with_apls_post(seg, orient, **base_kw, **flags)
        apls, topo = compute_topology_metrics(G, gt, (h, w))
        rows.append((name, G.number_of_nodes(), G.number_of_edges(), apls, topo))

    print("")
    print("| 配置 | nodes | edges | APLS | TOPO-F1 |")
    print("|------|------:|------:|-----:|--------:|")
    for name, nv, ne, apls, topo in rows:
        print(f"| {name} | {nv} | {ne} | {apls:.4f} | {topo:.4f} |")
    print("")


if __name__ == "__main__":
    main()
