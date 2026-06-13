"""
对单张 seg 概率图 +（可选）K 维方向图跑 iaof_postprocessing，并可对比简单距离连边。

先导出 npy：
  python scripts/evaluate_cityscale.py ... --export-npy-dir eval_outputs\\npy_tiles

调试连接 + 方向图模式（需 --orient）：
  python scripts/demo_iaof_postprocessing.py --seg ... --orient ... --gt-graph ... --hw 2048 2048 --debug

仅距离连边（易得到 600+ 边，用于 sanity check）：
  python scripts/demo_iaof_postprocessing.py --seg ... --gt-graph ... --hw 2048 2048 --simple-dist --simple-max-dist 48

宽度分级分层建图（无方向场，EDT+KMeans 分级后从宽到细连边；--visualize 时保存按 L1–L4 着色图，默认文件名加 _colored）：
  python scripts/demo_iaof_postprocessing.py --seg ... --gt-graph ... --hw 2048 2048 --hierarchical --debug --visualize eval_outputs\\pp_hierarchical.png

骨架中心线（推荐：长蓝线、少碎网，对齐 evaluate_cityscale）：
  python scripts/demo_iaof_postprocessing.py --seg ... --gt-graph ... --hw 2048 2048 --centerline --no-path-completion --visualize eval_outputs\\pp_centerline.png

APLS 导向组合（seg 预处理 + 转角护度2 + 保守 A* / 同连通域 / 折线过滤；端点半径默认 48 时改为 24）：
  python scripts/demo_iaof_postprocessing.py --seg ... --gt-graph ... --hw 2048 2048 --centerline --optimize-apls --visualize eval_outputs\\pp_opt.png

漏检/重复边调参示例（放宽 walkable、端点半径、连通域、转角门控 + edge_deduplicate + 高亮候选）：
  python scripts/demo_iaof_postprocessing.py --seg ... --gt-graph ... --hw 2048 2048 --centerline --optimize-apls ^
    --endpoint-radius 30 --astar-walkable-thresh 0.38 --path-component-prob-thresh 0.40 ^
    --degree2-guard-turn-max-deviation 35 --remove-parallel-duplicates --show-duplicate-edges ^
    --parallel-dup-angle-thresh 12 --parallel-dup-dist-thresh 2.5 --parallel-dup-len-frac-thresh 0.85 ^
    --medial-axis-reskeletonize --visualize eval_outputs\\pp_optimized_v3_dedup.png

APLS baseline（关 optimize、强调平滑与曲率惩罚、可关去重）：
  python scripts/demo_iaof_postprocessing.py --seg ... --gt-graph ... --hw 2048 2048 --centerline --debug ^
    --no-optimize-apls --no-remove-parallel-duplicate-edges --parallel-dedup-timing none ^
    --endpoint-radius 25 --astar-walkable-thresh 0.35 --astar-curvature-penalty 4 ^
    --polyline-smooth-window 9 --visualize eval_outputs\\pp_baseline_apls.png
  # 或省略 --gt-graph，用数据集根目录 + tile 编号（自动拼 20cities/region_{id}_refine_gt_graph.p）：
  python scripts/demo_iaof_postprocessing.py --seg ... --cityscale-root E:/datasets/cityscaledataset/cityscale --region 18 --hw 2048 2048 --centerline --optimize-apls

可视化：
  python scripts/demo_iaof_postprocessing.py ... --visualize eval_outputs\\pp_vis.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from iaof_postprocessing.pipeline import compute_topology_metrics, run_pipeline
from iaof_postprocessing.visualize import visualize_hierarchical_results, visualize_postprocessing
from utils.cityscale_graph import default_gt_pickle_path, load_cityscale_gt_graph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg", type=str, required=True, help="seg 概率 .npy 形状 H×W")
    ap.add_argument("--orient", type=str, default=None, help="方向 .npy；simple/centerline/hierarchical 时可省略")
    ap.add_argument(
        "--gt-graph",
        type=str,
        default=None,
        help="region_*_refine_gt_graph.p 的完整路径；若省略则需同时指定 --cityscale-root 与 --region",
    )
    ap.add_argument(
        "--cityscale-root",
        type=str,
        default=None,
        help="City-Scale 数据集根目录（其下应有 20cities/）；与 --region 联用可自动拼 GT 路径",
    )
    ap.add_argument(
        "--region",
        type=int,
        default=None,
        help="tile 编号 0..179；与 --cityscale-root 联用，加载 20cities/region_{region}_refine_gt_graph.p",
    )
    ap.add_argument("--hw", type=int, nargs=2, default=[2048, 2048], help="画布高 宽（TOPO-F1）")
    ap.add_argument("--no-path-completion", action="store_true")
    ap.add_argument("--no-topology-refine", action="store_true")
    ap.add_argument("--debug", action="store_true", help="打印方向连边 / 简单连边 / A* 调试信息")
    ap.add_argument("--centerline", action="store_true", help="骨架/中轴路径追踪建图（长折线边，缓解碎网）；无需 --orient")
    ap.add_argument(
        "--centerline-merge-dist",
        type=float,
        default=15.0,
        help="交叉口附近合并半径（像素），约 node_spacing/2；增大可减少冗余节点（如 15~18）",
    )
    ap.add_argument("--no-smooth-polylines", action="store_true", help="关闭边折线滑动平均平滑")
    ap.add_argument(
        "--smooth-window",
        "--polyline-smooth-window",
        type=int,
        default=3,
        dest="smooth_window",
        help="折线平滑窗口（奇数≥3）",
    )
    ap.add_argument("--smooth-iters", type=int, default=1, help="折线平滑迭代次数")
    ap.add_argument(
        "--centerline-min-edge",
        type=float,
        default=0.0,
        help="centerline 阶段删短边阈值（像素）；默认 0=不删，利于 APLS 连通。需压噪时再设 15~25",
    )
    ap.add_argument(
        "--centerline-refine-min-edge",
        type=float,
        default=0.0,
        help="refine 阶段删短边；centerline 默认 0 与上项一致，避免二次切碎",
    )
    ap.add_argument(
        "--node-spacing",
        type=float,
        default=None,
        help="若指定，覆盖 merge_dist ≈ spacing×0.5（目标间距 15~20 时可设 18）",
    )
    ap.add_argument(
        "--centerline-full-refine",
        action="store_true",
        help="centerline 时使用完整 refine（合并近邻+小夹角删边）；默认关闭以免边数从 ~700 掉到 ~550",
    )
    ap.add_argument(
        "--optimize-apls",
        action="store_true",
        help="开启几何+语义保守组合（可与下方单项开关叠加覆盖）",
    )
    ap.add_argument(
        "--no-optimize-apls",
        action="store_true",
        help="与 --optimize-apls 互斥；强制关闭 optimize 组合（便于 APLS baseline）",
    )
    ap.add_argument(
        "--seg-preprocess",
        action="store_true",
        help="中心线前对二值 mask 闭运算+高斯+再二值",
    )
    ap.add_argument(
        "--degree2-guard-turn",
        action="store_true",
        help="centerline 度2链压缩使用转角门控（减轻弯道拉直）",
    )
    ap.add_argument(
        "--refine-degree2-guard",
        action="store_true",
        help="refine 阶段度2压缩也使用转角门控",
    )
    ap.add_argument(
        "--degree2-guard-turn-max-deviation",
        type=float,
        default=None,
        help="度2转角门控阈值（度）；指定时同时作用于 centerline 与 refine；默认 28",
    )
    ap.add_argument(
        "--astar-walkable-thresh",
        type=float,
        default=None,
        help="A* 扩展时邻居道路概率下限（仅路上搜索）；不设则与历史行为一致",
    )
    ap.add_argument(
        "--path-same-component",
        action="store_true",
        help="仅当两端点落在 seg 同一连通域（prob>=阈值）时才尝试路径补全",
    )
    ap.add_argument(
        "--path-component-thresh",
        "--path-component-prob-thresh",
        type=float,
        default=0.45,
        dest="path_component_thresh",
        help="同连通域判定时 seg 概率二值阈值（与 --optimize-apls / --path-same-component 联用）",
    )
    ap.add_argument(
        "--path-polyline-min-frac",
        type=float,
        default=None,
        help="补全新边：沿折线采样中 prob>=--path-polyline-min-prob 的占比下限",
    )
    ap.add_argument(
        "--path-polyline-min-prob",
        type=float,
        default=0.45,
        help="折线采样判“在路上”的概率阈值",
    )
    ap.add_argument(
        "--path-polyline-min-mid-prob",
        type=float,
        default=None,
        help="若设置：新边折线上最小 prob 须不低于该值",
    )
    ap.add_argument("--simple-dist", action="store_true", help="使用纯距离+线段道路占比连边")
    ap.add_argument(
        "--hierarchical",
        action="store_true",
        help="宽度分级分层建图：EDT 估宽 + KMeans 分级，从宽到细逐层距离连边（无方向场）",
    )
    ap.add_argument("--seg-thresh", type=float, default=0.5, help="二值化阈值（节点、分级、道路占比采样）")
    ap.add_argument("--n-road-classes", type=int, default=4, help="[hierarchical] 道路宽度等级数（KMeans 簇数上限）")
    ap.add_argument(
        "--max-dist-l1",
        type=float,
        default=30.0,
        help="[hierarchical] L1（最宽）级内最大连边距离（像素）",
    )
    ap.add_argument("--max-dist-l2", type=float, default=26.0, help="[hierarchical] L2 级内最大连边距离")
    ap.add_argument("--max-dist-l3", type=float, default=22.0, help="[hierarchical] L3 级内最大连边距离")
    ap.add_argument("--max-dist-l4", type=float, default=18.0, help="[hierarchical] L4（最细）级内最大连边距离")
    ap.add_argument(
        "--connect-dist-l2",
        type=float,
        default=25.0,
        help="[hierarchical] L2 骨架节点连到已建图的最大距离",
    )
    ap.add_argument("--connect-dist-l3", type=float, default=20.0, help="[hierarchical] L3 连到已建图的最大距离")
    ap.add_argument("--connect-dist-l4", type=float, default=16.0, help="[hierarchical] L4 连到已建图的最大距离")
    ap.add_argument(
        "--hierarchical-max-bridge-l4",
        type=float,
        default=None,
        help="[hierarchical] 若设置：最细级与已有图连边超过该长度则丢弃（防长斜边）",
    )
    ap.add_argument(
        "--hierarchical-min-road",
        type=float,
        default=0.30,
        help="[hierarchical] 线段道路占比下限（与 simple 类似）",
    )
    ap.add_argument(
        "--hierarchical-node-dbscan-eps",
        type=float,
        default=12.0,
        help="[hierarchical] 关键节点 DBSCAN 合并半径（像素）",
    )
    ap.add_argument(
        "--hierarchical-chain-min-turn-deg",
        type=float,
        default=18.0,
        help="[hierarchical] 骨架直链上保留拐点的角度阈值（度）；越大直段上节点越少",
    )
    ap.add_argument(
        "--hierarchical-no-chain-simplify",
        action="store_true",
        help="[hierarchical] 关闭链角度简化（仅端点+交叉口像素，节点更密）",
    )
    ap.add_argument("--simple-max-dist", type=float, default=45.0, help="简单模式 KDTree 半径")
    ap.add_argument("--simple-min-road", type=float, default=0.30, help="简单模式线段上道路像素最小占比")
    ap.add_argument(
        "--orient-compat",
        type=float,
        default=0.10,
        help="方向兼容性阈值 0.5*(p_i+p_j)；越小边越多（建议 0.08~0.20）",
    )
    ap.add_argument("--orient-radius", type=float, default=48.0, help="方向连边 KDTree 半径（像素）")
    ap.add_argument("--dbscan-eps", type=float, default=8.0, help="节点 DBSCAN eps（像素）")
    ap.add_argument(
        "--endpoint-radius",
        type=float,
        default=48.0,
        help="A* 端点搜索半径（像素）；略小更保守，APLS 易稳",
    )
    ap.add_argument(
        "--road-cost-coef",
        type=float,
        default=9.0,
        help="A* 代价 1+coef*(1-p)；越大越贴高 prob 区域，减少乱连",
    )
    ap.add_argument(
        "--path-pairs-per-node",
        type=int,
        default=24,
        help="每端点最多尝试的候选对数（降低可减弱补全强度）",
    )
    ap.add_argument(
        "--max-astar-edges",
        type=int,
        default=120,
        help="全局 A* 新边上限；0 表示不限制",
    )
    ap.add_argument("--min-edge-len", type=float, default=6.0, help="拓扑精简：删短边阈值（像素）")
    ap.add_argument(
        "--medial-axis-reskeletonize",
        action="store_true",
        help="medial_axis 后再 skeletonize，减轻局部双脊线（略增断线风险）",
    )
    ap.add_argument(
        "--remove-parallel-duplicate-edges",
        "--remove-parallel-duplicates",
        action="store_true",
        dest="remove_parallel_duplicate_edges",
        help="启用平行边去重（见 --parallel-dedup-method / --parallel-dedup-timing）",
    )
    ap.add_argument(
        "--no-remove-parallel-duplicate-edges",
        action="store_true",
        dest="no_remove_parallel_duplicate_edges",
        help="显式关闭去重（即使其它配置曾打开）",
    )
    ap.add_argument(
        "--parallel-dedup-method",
        type=str,
        choices=["geometric", "density"],
        default="geometric",
        help="geometric：几何规则；density：边采样点 DBSCAN（易误伤交叉口，可调 eps/min_samples）",
    )
    ap.add_argument(
        "--parallel-dedup-timing",
        type=str,
        choices=["none", "connect", "refine", "both"],
        default="refine",
        help="none=不去重；connect=平滑后、补全前；refine=原 refine 末尾（几何）或 refine 后（密度）；both=两阶段",
    )
    ap.add_argument("--density-dedup-eps", type=float, default=5.0, help="DBSCAN eps（像素）")
    ap.add_argument("--density-dedup-min-samples", type=int, default=6, help="DBSCAN min_samples")
    ap.add_argument("--density-dedup-n-samples", type=int, default=12, help="每条边采样点数")
    ap.add_argument(
        "--astar-curvature-penalty",
        type=float,
        default=0.0,
        help="路径补全 A* 转折角惩罚系数（>0 更平滑，建议 2~8 试）",
    )
    ap.add_argument(
        "--parallel-dup-angle",
        "--parallel-dup-angle-thresh",
        type=float,
        default=15.0,
        dest="parallel_dup_angle",
        help="平行边：方向角差阈值（度），默认 15",
    )
    ap.add_argument(
        "--parallel-dup-strip",
        "--parallel-dup-dist-thresh",
        type=float,
        default=3.0,
        dest="parallel_dup_strip",
        help="平行边：四端点到另一线段距离均值上限（像素），默认 3",
    )
    ap.add_argument("--parallel-dup-min-len", type=float, default=20.0, help="仅对长度≥该值的边参与去重")
    ap.add_argument(
        "--parallel-dup-len-frac",
        "--parallel-dup-len-frac-thresh",
        type=float,
        default=0.8,
        dest="parallel_dup_len_frac",
        help="沿向重叠 / 平均边长 的下限（0~1），默认 0.8",
    )
    ap.add_argument(
        "--show-duplicate-edges",
        action="store_true",
        help="可视化时增加第 4 列，红色虚线标出近平行冗余候选边",
    )
    ap.add_argument("--visualize", type=str, default=None, help="若指定路径，保存三合一或四合一 PNG")
    ap.add_argument("--stats-json", type=str, default=None, help="若指定，写入各阶段边数等统计 JSON")
    args = ap.parse_args()

    _nconn = int(bool(args.centerline)) + int(bool(args.simple_dist)) + int(bool(args.hierarchical))
    if _nconn > 1:
        ap.error("--centerline、--simple-dist、--hierarchical 三者只能选其一")

    if bool(args.optimize_apls) and bool(args.no_optimize_apls):
        ap.error("--optimize-apls 与 --no-optimize-apls 不能同时使用")

    if args.gt_graph:
        gt_path = args.gt_graph
    elif args.cityscale_root is not None and args.region is not None:
        gt_path = default_gt_pickle_path(args.cityscale_root, int(args.region))
    else:
        ap.error("请提供 --gt-graph，或同时提供 --cityscale-root 与 --region（整数）")

    seg = np.load(args.seg)
    orient = np.load(args.orient) if args.orient else None
    if not args.simple_dist and not args.centerline and not args.hierarchical and orient is None:
        ap.error("方向模式需要 --orient，或改用 --simple-dist / --centerline / --hierarchical")

    gt = load_cityscale_gt_graph(gt_path)

    if args.centerline:
        mode = "centerline"
    elif args.simple_dist:
        mode = "simple"
    elif args.hierarchical:
        mode = "hierarchical"
    else:
        mode = "orient"

    cm = float(args.centerline_merge_dist)
    if args.node_spacing is not None:
        cm = max(4.0, float(args.node_spacing) * 0.5)
    max_astar = None if int(args.max_astar_edges) <= 0 else int(args.max_astar_edges)
    stats: dict = {}
    ep = float(args.endpoint_radius)
    if bool(args.optimize_apls) and not bool(args.no_optimize_apls) and abs(ep - 48.0) < 1e-6:
        ep = 24.0

    centerline_seg_preprocess = False
    centerline_degree2_simplify = "default"
    refine_simplify_degree2_turn_guard = False
    astar_walkable = None
    path_same_comp = False
    path_poly_frac = None
    path_poly_mid = None

    if bool(args.optimize_apls) and not bool(args.no_optimize_apls):
        centerline_seg_preprocess = True
        centerline_degree2_simplify = "guard_turn"
        refine_simplify_degree2_turn_guard = True
        astar_walkable = 0.42
        path_same_comp = True
        path_poly_frac = 0.55

    if bool(args.seg_preprocess):
        centerline_seg_preprocess = True
    if bool(args.degree2_guard_turn):
        centerline_degree2_simplify = "guard_turn"
    if bool(args.refine_degree2_guard):
        refine_simplify_degree2_turn_guard = True
    if args.astar_walkable_thresh is not None:
        astar_walkable = float(args.astar_walkable_thresh)
    if bool(args.path_same_component):
        path_same_comp = True
    if args.path_polyline_min_frac is not None:
        path_poly_frac = float(args.path_polyline_min_frac)
    if args.path_polyline_min_mid_prob is not None:
        path_poly_mid = float(args.path_polyline_min_mid_prob)

    ctd = (
        float(args.degree2_guard_turn_max_deviation)
        if args.degree2_guard_turn_max_deviation is not None
        else 28.0
    )

    remove_parallel = bool(args.remove_parallel_duplicate_edges) and not bool(
        args.no_remove_parallel_duplicate_edges
    )

    hier_chain_turn = None if bool(args.hierarchical_no_chain_simplify) else float(args.hierarchical_chain_min_turn_deg)

    G = run_pipeline(
        seg,
        orient,
        connection_mode=mode,
        thresh=float(args.seg_thresh),
        debug=bool(args.debug),
        stats_out=stats if args.stats_json else None,
        do_path_completion=not args.no_path_completion,
        do_topology_refine=not args.no_topology_refine,
        orient_compat=float(args.orient_compat),
        orient_neighbor_radius=float(args.orient_radius),
        dbscan_eps=float(args.dbscan_eps),
        endpoint_radius=ep,
        road_cost_coef=float(args.road_cost_coef),
        path_completion_max_pairs_per_node=int(args.path_pairs_per_node),
        path_completion_max_astar_edges=max_astar,
        min_edge_len=float(args.min_edge_len),
        simple_max_distance=float(args.simple_max_dist),
        simple_min_road_fraction=float(args.simple_min_road),
        centerline_merge_node_dist=cm,
        centerline_min_edge_length=float(args.centerline_min_edge),
        centerline_refine_min_edge=float(args.centerline_refine_min_edge),
        centerline_smooth_polylines=not args.no_smooth_polylines,
        centerline_smooth_window=int(args.smooth_window),
        centerline_smooth_iters=int(args.smooth_iters),
        centerline_refine_light=not bool(args.centerline_full_refine),
        centerline_seg_preprocess=centerline_seg_preprocess,
        centerline_degree2_simplify=centerline_degree2_simplify,
        refine_simplify_degree2_turn_guard=refine_simplify_degree2_turn_guard,
        astar_walkable_thresh=astar_walkable,
        path_completion_require_same_component=path_same_comp,
        path_completion_component_thresh=float(args.path_component_thresh),
        path_completion_polyline_min_road_frac=path_poly_frac,
        path_completion_polyline_min_prob=float(args.path_polyline_min_prob),
        path_completion_polyline_min_mid_prob=path_poly_mid,
        centerline_max_turn_deviation_deg=ctd,
        refine_max_turn_deviation_deg=ctd,
        centerline_medial_axis_reskeletonize=bool(args.medial_axis_reskeletonize),
        refine_remove_parallel_duplicates=remove_parallel,
        refine_parallel_dup_angle_deg=float(args.parallel_dup_angle),
        refine_parallel_dup_strip_px=float(args.parallel_dup_strip),
        refine_parallel_dup_min_edge_len=float(args.parallel_dup_min_len),
        refine_parallel_dup_len_frac_thresh=float(args.parallel_dup_len_frac),
        parallel_dedup_method=str(args.parallel_dedup_method),
        parallel_dedup_timing=str(args.parallel_dedup_timing),
        density_dedup_eps=float(args.density_dedup_eps),
        density_dedup_min_samples=int(args.density_dedup_min_samples),
        density_dedup_n_samples=int(args.density_dedup_n_samples),
        astar_curvature_penalty_coef=float(args.astar_curvature_penalty),
        hierarchical_n_classes=int(args.n_road_classes),
        hierarchical_max_dist_l1=float(args.max_dist_l1),
        hierarchical_max_dist_l2=float(args.max_dist_l2),
        hierarchical_max_dist_l3=float(args.max_dist_l3),
        hierarchical_max_dist_l4=float(args.max_dist_l4),
        hierarchical_connect_dist_l2=float(args.connect_dist_l2),
        hierarchical_connect_dist_l3=float(args.connect_dist_l3),
        hierarchical_connect_dist_l4=float(args.connect_dist_l4),
        hierarchical_max_bridge_l4=(
            float(args.hierarchical_max_bridge_l4) if args.hierarchical_max_bridge_l4 is not None else None
        ),
        hierarchical_min_road_fraction=float(args.hierarchical_min_road),
        hierarchical_node_dbscan_eps=float(args.hierarchical_node_dbscan_eps),
        hierarchical_chain_min_turn_deg=hier_chain_turn,
    )
    h, w = int(args.hw[0]), int(args.hw[1])
    apls, topo = compute_topology_metrics(G, gt, (h, w))
    print(f"nodes={G.number_of_nodes()} edges={G.number_of_edges()}")
    print(f"APLS={apls:.4f}  TOPO-F1={topo:.4f}")

    if args.visualize:
        outp = args.visualize if os.path.isabs(args.visualize) else os.path.join(PROJECT_ROOT, args.visualize)
        if mode == "hierarchical":
            root, ext = os.path.splitext(outp)
            if "_colored" not in os.path.basename(root):
                outp = root + "_colored" + ext
            visualize_hierarchical_results(
                seg,
                G,
                outp,
                n_classes_max=int(args.n_road_classes),
            )
        else:
            visualize_postprocessing(
                seg,
                G,
                outp,
                show_duplicate_edges=bool(args.show_duplicate_edges),
                dup_angle_deg=float(args.parallel_dup_angle),
                dup_strip_px=float(args.parallel_dup_strip),
                dup_min_edge_len=float(args.parallel_dup_min_len),
                dup_len_frac_thresh=float(args.parallel_dup_len_frac),
                dup_method=str(args.parallel_dedup_method),
                dup_density_eps=float(args.density_dedup_eps),
                dup_density_min_samples=int(args.density_dedup_min_samples),
                dup_density_n_samples=int(args.density_dedup_n_samples),
            )
        print(f"[Vis] saved {outp}")

    if args.stats_json:
        outp = args.stats_json if os.path.isabs(args.stats_json) else os.path.join(PROJECT_ROOT, args.stats_json)
        os.makedirs(os.path.dirname(outp) or ".", exist_ok=True)
        stats["apls"] = apls
        stats["topo_f1"] = topo
        stats["connection_mode"] = mode
        with open(outp, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"[Stats] saved {outp}")


if __name__ == "__main__":
    main()
