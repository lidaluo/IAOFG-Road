"""
标准栅格 APLS（SpaceNet 风格）：mask → sknw 骨架图 → NetworkX 最短路。

公式::
    APLS = 1 - (1/N) * Σ min(1, |d_pred - d_gt| / d_gt)

返回值越大越好（满分 1）。无效图返回 ``nan``。

依赖: ``pip install sknw networkx``
"""

from __future__ import annotations

import math
import random
from itertools import combinations
from typing import Any

import numpy as np
import networkx as nx
from skimage.morphology import skeletonize

try:
    import sknw
except ImportError as e:  # pragma: no cover
    sknw = None  # type: ignore[misc, assignment]
    _sknw_import_error = e
else:
    _sknw_import_error = None


def _ensure_sknw() -> None:
    if sknw is None:
        raise ImportError(
            "需要安装 sknw 与 networkx：pip install sknw networkx\n"
            f"原始错误: {_sknw_import_error}"
        )


def mask_to_graph(mask: np.ndarray) -> nx.Graph:
    """二值 mask ``[H,W]`` → 骨架 ``uint16`` → sknw → NetworkX；边上 ``length`` 为欧氏弧长。"""
    _ensure_sknw()
    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("mask 须为 2D")
    skel = skeletonize(m > 0).astype(np.uint16)
    G = sknw.build_sknw(skel)
    if hasattr(G, "is_directed") and G.is_directed() and hasattr(G, "to_undirected"):
        G = G.to_undirected()
    # 保留 sknw 边属性 ``pts``，仅补充 ``length``
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
    """``{node_id: np.array([y,x])}``（sknw 用 ``'o'`` 存节点像素坐标）。"""
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
    """图上距 ``point``（y,x）最近的节点。"""
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
    n_pairs: int = 200,
    snap_threshold: float = 50.0,
    seed: int = 42,
) -> float:
    """
    标准 APLS（越大越好）::

        APLS = 1 - (1/N) * Σ min(1, |d_pred - d_gt| / d_gt)

    其中 ``d_gt``、``d_pred`` 为对应骨架网上带 ``length`` 权的最短路径长；
    Pred 端点由 GT 图节点坐标 snap 到 Pred 图，并可将 snap 距离计入 ``d_pred``。
    """
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


def evaluate_apls_batch(pred_masks: np.ndarray, gt_masks: np.ndarray, **kwargs: Any) -> float:
    """一批 ``[N,H,W]`` 二值图，返回平均 APLS。"""
    n = int(pred_masks.shape[0])
    scores: list[float] = []
    for i in range(n):
        s = compute_apls(pred_masks[i], gt_masks[i], **kwargs)
        if math.isfinite(s):
            scores.append(s)
    return float(np.nanmean(scores)) if scores else float("nan")
