"""
与仓库根目录 ``calculate_topo_metrics.py``、``standard_apls.py`` 对齐的拓扑与 APLS 实现。

- TOPO-P/R/F1：骨架二值图 → 均匀随机采样点 → cKDTree 半径匹配（同 ``topo_precision_recall_f1``）。
- APLS：二值 mask → skeletonize → sknw 建图 → SpaceNet 风格惩罚（同 ``compute_apls``）。
"""

from __future__ import annotations

import math
import random
from itertools import combinations
from typing import Any, Optional

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

from utils import cv_io

try:
    import sknw
except ImportError:  # pragma: no cover
    sknw = None  # type: ignore[misc, assignment]
    _sknw_err: Optional[BaseException] = None
else:
    _sknw_err = None


def _ensure_sknw() -> None:
    global _sknw_err
    if sknw is None:
        raise ImportError(
            "标准 APLS 需要 sknw：pip install sknw\n"
            "（与 standard_apls.py 一致）"
        )


def sample_points_from_skeleton(skel_u8: np.ndarray, n_max: int, rng: np.random.Generator) -> np.ndarray:
    """与 calculate_topo_metrics.py 一致。"""
    ys, xs = np.where(skel_u8 > 127)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    n = min(int(n_max), len(xs))
    pick = rng.choice(len(xs), size=n, replace=False)
    return np.stack([xs[pick], ys[pick]], axis=1).astype(np.float64)


def topo_precision_recall_f1(
    pred_pts: np.ndarray, gt_pts: np.ndarray, radius: float
) -> tuple[float, float, float]:
    """与 calculate_topo_metrics.py 一致。"""
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    d_p, _ = cKDTree(gt_pts).query(pred_pts, k=1)
    prec = float(np.mean(d_p <= radius))
    d_g, _ = cKDTree(pred_pts).query(gt_pts, k=1)
    rec = float(np.mean(d_g <= radius))
    if prec + rec < 1e-12:
        f1 = 0.0
    else:
        f1 = float(2.0 * prec * rec / (prec + rec))
    return prec, rec, f1


def graph_to_binary_mask(graph: nx.Graph, hw: tuple[int, int], line_width: int = 3) -> np.ndarray:
    """NetworkX 图 → H×W 前景 bool（栅格化）。"""
    u8 = cv_io.rasterize_graph_canvas(graph, (int(hw[0]), int(hw[1])), line_width=int(line_width))
    return u8 > 0


def skeleton_u8_from_mask(binmask: np.ndarray) -> np.ndarray:
    """bool/0-1 mask → 骨架 uint8 0/255（与 calculate_topo_metrics 中 skeleton 可视化一致）。"""
    sk = skeletonize(np.asarray(binmask, dtype=bool))
    return (sk.astype(np.uint8) * 255)


def topo_precision_recall_f1_from_graphs(
    pred_graph: nx.Graph,
    gt_graph: nx.Graph,
    canvas_hw: tuple[int, int],
    *,
    radius: float,
    n_samples: int,
    seed: int,
    raster_line_width: int = 3,
) -> tuple[float, float, float]:
    """
    将图栅格化 → 骨架 → 采样 → ``topo_precision_recall_f1``（与 calculate_topo_metrics 流程对齐）。
    """
    pm = graph_to_binary_mask(pred_graph, canvas_hw, line_width=raster_line_width)
    gm = graph_to_binary_mask(gt_graph, canvas_hw, line_width=raster_line_width)
    ps = skeleton_u8_from_mask(pm)
    gs = skeleton_u8_from_mask(gm)
    if not np.any(ps > 0) or not np.any(gs > 0):
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(int(seed))
    pred_pts = sample_points_from_skeleton(ps, int(n_samples), rng)
    gt_pts = sample_points_from_skeleton(gs, int(n_samples), rng)
    p, r, f1 = topo_precision_recall_f1(pred_pts, gt_pts, float(radius))
    if not math.isfinite(p):
        return 0.0, 0.0, 0.0
    return p, r, f1


def mask_to_graph(mask: np.ndarray) -> nx.Graph:
    """与 standard_apls.mask_to_graph 一致。"""
    _ensure_sknw()
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("mask 须为 2D")
    skel = skeletonize(m > 0).astype(np.uint16)
    G = sknw.build_sknw(skel)
    if hasattr(G, "is_directed") and G.is_directed() and hasattr(G, "to_undirected"):
        G = G.to_undirected()
    for u, v in list(G.edges()):
        pts = G[u][v].get("pts")
        if pts is None or len(pts) < 2:
            length = 1e-6
        else:
            length = 0.0
            for i in range(1, len(pts)):
                dy = float(pts[i][0] - pts[i - 1][0])
                dx = float(pts[i][1] - pts[i - 1][1])
                length += math.sqrt(dy * dy + dx * dx)
            length = max(length, 1e-6)
        G[u][v]["length"] = length
    return G


def get_node_coords(G: nx.Graph) -> dict[Any, np.ndarray]:
    coords: dict[Any, np.ndarray] = {}
    for n in G.nodes():
        o = G.nodes[n].get("o")
        if o is None:
            continue
        coords[n] = np.asarray(o, dtype=np.float64).reshape(-1)
        if coords[n].size < 2:
            continue
    return coords


def snap_point_to_graph(
    point: np.ndarray, node_coords: dict[Any, np.ndarray]
) -> tuple[Any | None, float]:
    if not node_coords:
        return None, float("inf")
    pt = np.asarray(point, dtype=np.float64).reshape(2)
    best: Any = None
    min_d = float("inf")
    for nid, coord in node_coords.items():
        c = np.asarray(coord, dtype=np.float64).reshape(2)
        d = float(np.linalg.norm(c - pt))
        if d < min_d:
            min_d = d
            best = nid
    return best, min_d


def compute_apls(
    pred_mask: np.ndarray,
    gt_mask: np.ndarray,
    *,
    n_pairs: int = 200,
    snap_threshold: float = 50.0,
    seed: int = 42,
) -> float:
    """与 standard_apls.compute_apls 一致。"""
    _ensure_sknw()
    random.seed(int(seed))
    np.random.seed(int(seed))

    pred_mask = np.asarray(pred_mask)
    gt_mask = np.asarray(gt_mask)
    if pred_mask.shape != gt_mask.shape:
        raise ValueError("pred_mask / gt_mask 形状须一致")

    if not np.any(gt_mask > 0) or not np.any(pred_mask > 0):
        return float("nan")

    G_gt = mask_to_graph(gt_mask)
    G_pred = mask_to_graph(pred_mask)

    if G_gt.number_of_nodes() < 2 or G_pred.number_of_nodes() < 2:
        return float("nan")

    gt_coords = get_node_coords(G_gt)
    pred_coords = get_node_coords(G_pred)
    if len(gt_coords) < 2 or len(pred_coords) < 2:
        return float("nan")

    gt_nodes = list(G_gt.nodes())
    all_pairs = list(combinations(gt_nodes, 2))
    n_sample = min(int(n_pairs), len(all_pairs))
    if n_sample < 1:
        return float("nan")
    if len(all_pairs) > n_sample:
        sampled_pairs = random.sample(all_pairs, n_sample)
    else:
        sampled_pairs = all_pairs

    penalties: list[float] = []

    for s_gt, t_gt in sampled_pairs:
        try:
            d_gt = nx.shortest_path_length(G_gt, s_gt, t_gt, weight="length")
        except nx.NetworkXNoPath:
            continue
        if not math.isfinite(d_gt) or d_gt < 1e-3:
            continue

        s_coord = gt_coords.get(s_gt)
        t_coord = gt_coords.get(t_gt)
        if s_coord is None or t_coord is None:
            continue

        s_pred, s_dist = snap_point_to_graph(s_coord, pred_coords)
        t_pred, t_dist = snap_point_to_graph(t_coord, pred_coords)
        if s_pred is None or t_pred is None:
            penalties.append(1.0)
            continue

        if s_dist > snap_threshold or t_dist > snap_threshold:
            penalties.append(1.0)
            continue

        if s_pred == t_pred:
            d_pred = float(s_dist + t_dist)
            penalty = min(1.0, abs(d_pred - float(d_gt)) / max(float(d_gt), 1e-9))
            penalties.append(float(penalty))
            continue

        try:
            d_path = nx.shortest_path_length(G_pred, s_pred, t_pred, weight="length")
        except nx.NetworkXNoPath:
            penalties.append(1.0)
            continue

        if not math.isfinite(d_path):
            penalties.append(1.0)
            continue

        d_pred = float(d_path) + float(s_dist) + float(t_dist)
        penalty = min(1.0, abs(d_pred - float(d_gt)) / max(float(d_gt), 1e-9))
        penalties.append(float(penalty))

    if not penalties:
        return float("nan")

    mean_pen = float(np.mean(np.asarray(penalties, dtype=np.float64)))
    apls = 1.0 - mean_pen
    return float(max(apls, 0.0))


def compute_apls_from_graphs(
    pred_graph: nx.Graph,
    gt_graph: nx.Graph,
    canvas_hw: tuple[int, int],
    *,
    n_pairs: int,
    snap_threshold: float,
    seed: int,
    raster_line_width: int = 3,
) -> float:
    """图 → 二值 mask → ``compute_apls``。"""
    pm = graph_to_binary_mask(pred_graph, canvas_hw, line_width=raster_line_width).astype(np.uint8) * 255
    gm = graph_to_binary_mask(gt_graph, canvas_hw, line_width=raster_line_width).astype(np.uint8) * 255
    s = compute_apls(pm, gm, n_pairs=n_pairs, snap_threshold=snap_threshold, seed=seed)
    return float(s) if math.isfinite(s) else 0.0


def infer_canvas_hw(
    pred_graph: nx.Graph,
    gt_graph: nx.Graph,
    *,
    default_hw: tuple[int, int] = (2048, 2048),
) -> tuple[int, int]:
    """由节点 pos (x,y) 推断画布 (H,W)；无节点时返回 default。"""
    mx = my = 0.0
    for G in (pred_graph, gt_graph):
        for _, d in G.nodes(data=True):
            p = d.get("pos")
            if p is None:
                continue
            mx = max(mx, float(p[0]))
            my = max(my, float(p[1]))
    if mx <= 0 and my <= 0:
        return int(default_hw[0]), int(default_hw[1])
    w = max(int(default_hw[1]), int(np.ceil(mx)) + 1)
    h = max(int(default_hw[0]), int(np.ceil(my)) + 1)
    return h, w
