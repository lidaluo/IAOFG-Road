"""
对 ``region_*_seg.npy`` 批量运行 ``WidthConsistentConnector``，输出指标 CSV 与遥感叠加图。

指标列：TOPO-P、TOPO-R、TOPO-F1、APLS、Infer.Time（``process`` 墙钟时间，含可视化保存）。

**推荐目录布局**：把 npy 与汇总都放在 ``testoutput/`` 下，例如 ``testoutput/npy_tiles/`` + ``testoutput/results_summary.csv`` + 各 ``region_*_wcc_overlay.png``。

**先导出 20 张 npy（示例 A：从 train 划分取前 20 个 tile）**::

    python scripts/evaluate_cityscale.py --config configs/cityscale_iaofpp.yaml \\
      --split train --max-tiles 20 --export-npy-dir testoutput/npy_tiles \\
      --iaofpp-pipeline

**示例 B：显式指定 20 个 region ID（与 valid 的 18,38,… 错开时常用步长 20）**::

    python scripts/evaluate_cityscale.py --config configs/cityscale_iaofpp.yaml \\
      --regions 198,218,238,258,278,298,318,338,358,378,398,418,438,458,478,498,518,538,558,578 \\
      --export-npy-dir testoutput/npy_tiles --iaofpp-pipeline

**再批处理 WCC**（叠加图道路画成细单线、色谱不变时加 ``--viz-single-line``）::

    python scripts/batch_wcc_testoutput.py \\
      --npy-dir testoutput/npy_tiles \\
      --cityscale-root E:/datasets/cityscaledataset/cityscale \\
      --out-dir testoutput \\
      --viz-single-line
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from typing import Any

import cv2
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.cityscale_dataset import resolve_cityscale_rgb_path
from iaof_postprocessing.width_consistent_connector import WidthConsistentConnector
from utils.cityscale_graph import default_gt_pickle_path, load_cityscale_gt_graph


def _discover_region_ids(npy_dir: str) -> list[int]:
    pat = re.compile(r"region_(\d+)_seg\.npy$")
    out: list[int] = []
    for name in os.listdir(npy_dir):
        m = pat.match(name)
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def _fmt_metric(x: Any) -> str:
    if x is None:
        return ""
    try:
        return f"{float(x):.6f}"
    except (TypeError, ValueError):
        return ""


def main() -> None:
    ap = argparse.ArgumentParser(description="批量 WCC：CSV 汇总 + 每区域叠加可视化")
    ap.add_argument(
        "--npy-dir",
        type=str,
        default="eval_outputs/npy_tiles",
        help="含 region_{id}_seg.npy 的目录",
    )
    ap.add_argument("--cityscale-root", type=str, required=True, help="City-Scale 数据集根（含 20cities、GT .p）")
    ap.add_argument("--out-dir", type=str, default="testoutput", help="输出目录：results_summary.csv 与叠加 PNG")
    ap.add_argument(
        "--regions",
        type=str,
        default=None,
        help="逗号分隔 region ID；默认扫描 npy-dir 下全部 region_*_seg.npy",
    )
    ap.add_argument("--limit", type=int, default=None, help="最多处理前 N 个（在排序后的列表上截断）")
    ap.add_argument("--sat-subdir", type=str, default="20cities")
    ap.add_argument("--images-subdir", type=str, default="images")
    ap.add_argument("--thresh", type=float, default=0.5)
    ap.add_argument("--max-bridge-gap", type=float, default=10.0)
    ap.add_argument("--width-rel-tol", type=float, default=0.3)
    ap.add_argument("--hw", type=int, nargs=2, default=None, help="canvas_hw；默认使用 seg 数组形状")
    ap.add_argument("--summary-name", type=str, default="results_summary.csv")
    ap.add_argument(
        "--viz-single-line",
        action="store_true",
        help="叠加图中道路以细单线绘制：边线更细、节点更小；宽度色谱（plasma 等）不变",
    )
    ap.add_argument(
        "--viz-edge-width",
        type=float,
        default=None,
        help="叠加图上网边的 matplotlib linewidth；未指定时默认 1.75，或与 --viz-single-line 时为 0.85",
    )
    ap.add_argument(
        "--viz-node-size",
        type=float,
        default=None,
        help="叠加图上节点 scatter 大小；未指定时默认 36，或与 --viz-single-line 时为 8",
    )
    args = ap.parse_args()

    npy_dir = args.npy_dir if os.path.isabs(args.npy_dir) else os.path.join(PROJECT_ROOT, args.npy_dir)
    out_dir = args.out_dir if os.path.isabs(args.out_dir) else os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.regions:
        regions = sorted({int(x.strip()) for x in args.regions.split(",") if x.strip()})
    else:
        regions = _discover_region_ids(npy_dir)
    if args.limit is not None:
        regions = regions[: int(args.limit)]

    if not regions:
        raise SystemExit(f"未找到可处理的 region（目录 {npy_dir}）")

    csv_path = os.path.join(out_dir, args.summary_name)
    fieldnames = [
        "region_id",
        "TOPO_P",
        "TOPO_R",
        "TOPO_F1",
        "APLS",
        "Infer_Time_s",
        "num_nodes",
        "num_edges",
        "seg_path",
        "rgb_path",
        "overlay_path",
    ]
    rows: list[dict[str, str]] = []

    root = args.cityscale_root
    for rid in regions:
        seg_path = os.path.join(npy_dir, f"region_{rid}_seg.npy")
        if not os.path.isfile(seg_path):
            print(f"[Skip] 无 seg: {seg_path}")
            continue

        seg = np.load(seg_path)
        h, w = int(seg.shape[0]), int(seg.shape[1])
        canvas = (h, w)
        if args.hw is not None:
            canvas = (int(args.hw[0]), int(args.hw[1]))

        rgb_path = resolve_cityscale_rgb_path(
            root, rid, sat_subdir=args.sat_subdir, images_subdir=args.images_subdir
        )
        orig_rgb: np.ndarray | None = None
        if rgb_path and os.path.isfile(rgb_path):
            bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if bgr is not None:
                orig_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        gt_path = default_gt_pickle_path(root, rid)
        gt = load_cityscale_gt_graph(gt_path) if os.path.isfile(gt_path) else None
        if gt is None:
            print(f"[Warn] region {rid} 无 GT，拓扑/APLS 列为空")

        overlay_path = os.path.join(out_dir, f"region_{rid}_wcc_overlay.png")
        ew = args.viz_edge_width
        ns = args.viz_node_size
        if args.viz_single_line:
            if ew is None:
                ew = 0.85
            if ns is None:
                ns = 8.0
        if ew is None:
            ew = 1.75
        if ns is None:
            ns = 36.0
        conn = WidthConsistentConnector(
            seg,
            max_bridge_gap=float(args.max_bridge_gap),
            width_rel_tol=float(args.width_rel_tol),
            thresh=float(args.thresh),
            debug=False,
            viz_edge_width=float(ew),
            viz_node_size=float(ns),
        )
        t0 = time.perf_counter()
        nodes, edges, vis_path, metrics = conn.process(
            gt_graph=gt,
            canvas_hw=canvas,
            visualization_path=overlay_path,
            original_image=orig_rgb,
        )
        dt = time.perf_counter() - t0

        row = {
            "region_id": str(rid),
            "TOPO_P": _fmt_metric(metrics.get("topo_precision")),
            "TOPO_R": _fmt_metric(metrics.get("topo_recall")),
            "TOPO_F1": _fmt_metric(metrics.get("topo_f1")),
            "APLS": _fmt_metric(metrics.get("apls")),
            "Infer_Time_s": f"{dt:.6f}",
            "num_nodes": str(len(nodes)),
            "num_edges": str(len(edges)),
            "seg_path": seg_path,
            "rgb_path": rgb_path or "",
            "overlay_path": vis_path or overlay_path,
        }
        rows.append(row)
        print(
            f"[{rid}] TOPO-P={row['TOPO_P'] or '-'} TOPO-R={row['TOPO_R'] or '-'} "
            f"TOPO-F1={row['TOPO_F1'] or '-'} APLS={row['APLS'] or '-'} time={dt:.3f}s -> {overlay_path}"
        )

    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"[Done] 共 {len(rows)} 行 -> {csv_path}")


if __name__ == "__main__":
    main()
