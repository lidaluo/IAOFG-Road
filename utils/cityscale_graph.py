"""
City-Scale 官方 pickle 图转 NetworkX，并写入边长 weight=length 供 APLS 使用。

与数据集根目录下 `generate_labels.py` 的读图方式一致：
- 输入：`20cities/region_{idx}_refine_gt_graph.p`（dict：邻接表）；
- 节点在 pickle 中为 (row, col)，OpenCV / 本仓库 pos 使用 (x, y) = (col, row)，
  故 `x0, y0 = int(n[1]), int(n[0])` 与官方 `graph.add_edge((int(n[1]), int(n[0])), ...)` 对齐。
"""

from __future__ import annotations

import os
import pickle
from typing import Any, Dict, Tuple

import networkx as nx
import numpy as np


def load_cityscale_gt_graph(pickle_path: str) -> nx.Graph:
    """
    与 generate_labels.py 一致：pickle 为 dict[node_tuple] -> list[neighbor_tuple]，
    node 存储为 (row, col)，图节点 pos 使用 (x, y) 像素坐标。
    """
    if not os.path.isfile(pickle_path):
        hint = (
            "官方数据布局为：<City-Scale 根目录>/20cities/region_<idx>_refine_gt_graph.p "
            "（子目录名为 20cities，不是 ale20cities）。"
            " 若路径中有 cityscdataset，常见正确名为 cityscaledataset。"
        )
        raise FileNotFoundError(f"找不到 GT pickle：{pickle_path}\n{hint}")

    with open(pickle_path, "rb") as f:
        gt_graph: Dict[Any, Any] = pickle.load(f)
    G = nx.Graph()
    for n, neis in gt_graph.items():
        # n 为 (row, col) 风格整数对
        x0, y0 = int(n[1]), int(n[0])
        G.add_node((x0, y0), pos=(float(x0), float(y0)))
        for nei in neis:
            x1, y1 = int(nei[1]), int(nei[0])
            G.add_node((x1, y1), pos=(float(x1), float(y1)))
    for n, neis in gt_graph.items():
        x0, y0 = int(n[1]), int(n[0])
        a = (x0, y0)
        for nei in neis:
            x1, y1 = int(nei[1]), int(nei[0])
            b = (x1, y1)
            if a == b:
                continue
            length = float(np.hypot(x1 - x0, y1 - y0))
            if G.has_edge(a, b):
                if length < G[a][b].get("length", length):
                    G[a][b]["length"] = length
            else:
                G.add_edge(a, b, length=length)
    return G


def relabel_graph_continuous(G: nx.Graph) -> nx.Graph:
    """将节点键改为连续 int，保留 pos / length。"""
    H = nx.Graph()
    mapping = {}
    for i, n in enumerate(G.nodes()):
        mapping[n] = i
        H.add_node(i, pos=tuple(G.nodes[n]["pos"]))
    for u, v, data in G.edges(data=True):
        H.add_edge(mapping[u], mapping[v], **data)
    return H


def ensure_edge_lengths(G: nx.Graph) -> nx.Graph:
    """若边缺少 length，用端点欧氏距离或 polyline 折线长补齐。"""
    for u, v, data in G.edges(data=True):
        if "length" in data and data["length"] is not None:
            continue
        pl = data.get("polyline")
        if pl is not None and len(pl) >= 2:
            L = 0.0
            for i in range(len(pl) - 1):
                x0, y0 = pl[i]
                x1, y1 = pl[i + 1]
                L += float(np.hypot(x1 - x0, y1 - y0))
            data["length"] = L
        else:
            pu = G.nodes[u]["pos"]
            pv = G.nodes[v]["pos"]
            data["length"] = float(np.hypot(pu[0] - pv[0], pu[1] - pv[1]))
    return G


def default_gt_pickle_path(cityscale_root: str, region_index: int) -> str:
    """同官方 `generate_labels.py`：`./20cities/region_{tile_index}_refine_gt_graph.p`。"""
    return os.path.join(cityscale_root, "20cities", f"region_{region_index}_refine_gt_graph.p")


def save_nx_graph_pickle(graph: nx.Graph, path: str) -> None:
    """保存 NetworkX 图（节点 int + pos、边含 length / polyline），供后续 Python 加载可视化。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(graph, f, protocol=pickle.HIGHEST_PROTOCOL)


def save_nx_graph_geojson(graph: nx.Graph, path: str) -> None:
    """
    将路网边导出为 GeoJSON LineString（坐标为像素 x,y，可在 QGIS 等中叠加影像）。
    """
    import json

    features: list[dict[str, Any]] = []
    for u, v, data in graph.edges(data=True):
        pl = data.get("polyline")
        if pl is not None and len(pl) >= 2:
            coords = [[float(px), float(py)] for px, py in pl]
        else:
            pu = graph.nodes[u]["pos"]
            pv = graph.nodes[v]["pos"]
            coords = [[float(pu[0]), float(pu[1])], [float(pv[0]), float(pv[1])]]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {"u": int(u), "v": int(v)},
            }
        )
    fc = {"type": "FeatureCollection", "features": features}
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fc, f)
