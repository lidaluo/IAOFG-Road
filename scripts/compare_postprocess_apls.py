"""
同一张 seg/orient 上对比两种预测路网，看 APLS / TOPO-F1（与 evaluate_cityscale 默认 centerline 对齐）。

用法（项目根）：
  python scripts/compare_postprocess_apls.py \\
    --seg eval_outputs/npy_tiles/region_18_seg.npy \\
    --orient eval_outputs/npy_tiles/region_18_orient.npy \\
    --gt-graph E:/datasets/.../region_18_refine_gt_graph.p \\
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

from iaof_postprocessing.pipeline import compute_topology_metrics, run_pipeline
from metrics.metrics import MetricsCalculator
from postprocess.centerline_from_mask import mask_to_centerline_graph
from utils.cityscale_graph import ensure_edge_lengths, load_cityscale_gt_graph, relabel_graph_continuous


def _score(
    pred: object,
    gt: object,
    hw: tuple[int, int],
    buf: int,
    max_gt_nodes: int,
) -> tuple[float, float, int, int]:
    calc = MetricsCalculator(max_gt_nodes_for_apls=max_gt_nodes)
    pg = relabel_graph_continuous(ensure_edge_lengths(pred.copy()))
    gg = relabel_graph_continuous(ensure_edge_lengths(gt.copy()))
    apls = calc.calculate_apls(pg, gg, canvas_hw=hw)
    topo = calc.calculate_topo_f1(pg, gg, hw, buffer_px=buf)
    return float(apls), float(topo), pg.number_of_nodes(), pg.number_of_edges()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", required=True)
    ap.add_argument("--orient", required=True)
    ap.add_argument("--gt-graph", required=True)
    ap.add_argument("--hw", type=int, nargs=2, default=[2048, 2048])
    ap.add_argument("--thr", type=float, default=0.25, help="二值化阈值（与 configs 里 evaluation.post_threshold 一致）")
    ap.add_argument("--topo-buffer", type=int, default=5)
    ap.add_argument("--max-gt-nodes-apls", type=int, default=64)
    ap.add_argument("--orient-compat", type=float, default=0.10)
    ap.add_argument("--orient-radius", type=float, default=48.0)
    args = ap.parse_args()

    seg = np.load(args.seg)
    orient = np.load(args.orient)
    gt = load_cityscale_gt_graph(args.gt_graph)
    h, w = int(args.hw[0]), int(args.hw[1])
    hw = (h, w)

    pred_mask = (seg > float(args.thr)).astype(np.uint8) * 255
    G_line = mask_to_centerline_graph(pred_mask, open_kernel=0, merge_node_dist=4.0, use_medial_axis=True)

    G_iaof = run_pipeline(
        seg,
        orient,
        orient_compat=float(args.orient_compat),
        orient_neighbor_radius=float(args.orient_radius),
    )

    buf = int(args.topo_buffer)
    max_n = int(args.max_gt_nodes_apls)

    a1, t1, n1, e1 = _score(G_line, gt, hw, buf, max_n)
    a2, t2, n2, e2 = _score(G_iaof, gt, hw, buf, max_n)

    print("")
    print("========== 同图对比（单 region）==========")
    print(f"阈值 thr={args.thr}  |  GT nodes={gt.number_of_nodes()} edges={gt.number_of_edges()}")
    print("------------------------------------------")
    print(f"[A] mask→centerline（与 evaluate_cityscale 默认一致）")
    print(f"    nodes={n1} edges={e1}  |  APLS={a1:.4f}  TOPO-F1={t1:.4f}")
    print(f"[B] iaof_postprocessing（方向+补全+拓扑精简）")
    print(f"    nodes={n2} edges={e2}  |  APLS={a2:.4f}  TOPO-F1={t2:.4f}")
    print("==========================================")
    print("")
    print("整数据集均值请跑: python scripts/evaluate_cityscale.py --split valid ...")
    print("若 [A] 明显高于 [B]，优先继续打磨 centerline 线或训练；实验性管线再调 --orient-compat / --orient-radius。")


if __name__ == "__main__":
    main()
