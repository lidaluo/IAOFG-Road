import csv
import json
import os
import sys
import argparse

import geopandas as gpd
import networkx as nx
import numpy as np
import rasterio
import torch
import yaml
from shapely.geometry import LineString, MultiLineString
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset_factory import build_datasets, get_all_sample_ids
from metrics.metrics import MetricsCalculator
from models.road_extraction_model import RoadExtractionModel
from postprocess.postprocessor import PostProcessor
from scripts.train import split_ids


def sample_id_to_paths(aoi_dir, sample_id):
    img_num = sample_id.replace("img", "")
    rgb_dir = os.path.join(aoi_dir, "PS-RGB")
    geo_dir = os.path.join(aoi_dir, "geojson_roads")
    tif_name = f"SN3_roads_train_AOI_3_Paris_PS-RGB_img{img_num}.tif"
    geo_name = f"SN3_roads_train_AOI_3_Paris_geojson_roads_img{img_num}.geojson"
    return os.path.join(rgb_dir, tif_name), os.path.join(geo_dir, geo_name)


def build_gt_graph_from_geojson(aoi_dir, sample_id):
    tif_path, geo_path = sample_id_to_paths(aoi_dir, sample_id)
    G = nx.Graph()
    if not os.path.exists(tif_path) or not os.path.exists(geo_path):
        return G

    with rasterio.open(tif_path) as src:
        transform = src.transform

    gdf = gpd.read_file(geo_path)
    node_map = {}
    next_id = 0

    def get_node_id(px):
        nonlocal next_id
        key = (int(px[0]), int(px[1]))
        if key not in node_map:
            node_map[key] = next_id
            G.add_node(next_id, pos=key)
            next_id += 1
        return node_map[key]

    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        lines = []
        if isinstance(geom, LineString):
            lines = [geom]
        elif isinstance(geom, MultiLineString):
            lines = [g for g in geom.geoms if isinstance(g, LineString)]

        for line in lines:
            coords = list(line.coords)
            if len(coords) < 2:
                continue
            pixel_path = []
            for x, y in coords:
                row, col = rasterio.transform.rowcol(transform, x, y)
                pixel_path.append((int(col), int(row)))

            for p0, p1 in zip(pixel_path[:-1], pixel_path[1:]):
                if p0 == p1:
                    continue
                u = get_node_id(p0)
                v = get_node_id(p1)
                length = float(np.sqrt((p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2))
                if G.has_edge(u, v):
                    if length < G[u][v].get("length", length):
                        G[u][v]["length"] = length
                else:
                    G.add_edge(u, v, length=length, path=[p0, p1])

    return G


def graph_to_geojson(graph, sample_id, source="pred"):
    features = []
    for node_id, attr in graph.nodes(data=True):
        x, y = attr.get("pos", (0, 0))
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [float(x), float(y)]},
                "properties": {"sample_id": sample_id, "source": source, "node_id": int(node_id)},
            }
        )
    for u, v, attr in graph.edges(data=True):
        path = attr.get("path")
        if path and len(path) >= 2:
            coords = [[float(px), float(py)] for (px, py) in path]
        else:
            ux, uy = graph.nodes[u].get("pos", (0, 0))
            vx, vy = graph.nodes[v].get("pos", (0, 0))
            coords = [[float(ux), float(uy)], [float(vx), float(vy)]]
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "sample_id": sample_id,
                    "source": source,
                    "u": int(u),
                    "v": int(v),
                    "length": float(attr.get("length", len(coords))),
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def to_postprocess_input_from_pred(outputs, idx):
    seg_logit = outputs["segmentation"][idx : idx + 1]
    seg_prob = torch.sigmoid(seg_logit)
    inter = torch.sigmoid(outputs["intersection"][idx : idx + 1])
    orient = torch.tanh(outputs["orientation"][idx : idx + 1, 0:2])
    # 评估时直接传一通道道路概率图，后处理内部按阈值生成掩码
    return {"segmentation": seg_prob.cpu(), "intersection": inter.cpu(), "direction_field": orient.cpu()}


def to_postprocess_input_from_gt(batch, idx):
    mask = batch["mask"][idx : idx + 1]  # [1,1,H,W]
    seg_2c = torch.cat([1.0 - mask, mask], dim=1)
    inter = batch["intersection"][idx : idx + 1]
    orient = batch["orientation"][idx : idx + 1, 0:2]
    return {"segmentation": seg_2c.cpu(), "intersection": inter.cpu(), "direction_field": orient.cpu()}


def main():
    parser = argparse.ArgumentParser(description="Evaluate topology metrics.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to yaml config")
    parser.add_argument("--post-threshold", type=float, default=None, help="Override PostProcessor threshold.")
    parser.add_argument("--post-nms-size", type=int, default=None, help="Override PostProcessor NMS size.")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    image_size = tuple(cfg["data"].get("image_size", [224, 224]))
    all_ids = get_all_sample_ids(cfg)
    train_ids, val_ids = split_ids(
        all_ids,
        val_ratio=cfg["data"].get("val_ratio", 0.2),
        seed=cfg["data"].get("split_seed", 42),
    )

    _, dataset = build_datasets(cfg, train_ids, val_ids, image_size)
    loader = DataLoader(dataset, batch_size=cfg["training"].get("batch_size", 2), shuffle=False, num_workers=0)

    ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "model_best_val_iou.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "model_latest.pth")

    model = RoadExtractionModel(encoder=cfg["model"]["encoder"], num_classes=1, input_size=image_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    eval_cfg = cfg.get("evaluation", {})
    post = PostProcessor(
        threshold=cfg["inference"].get("threshold", 0.5),
        nms_size=cfg["inference"].get("nms_size", 3),
        min_path_len=int(eval_cfg.get("min_path_len", 10)),
        endpoint_dist=float(eval_cfg.get("endpoint_dist", 10.0)),
        dir_stop_eps=float(eval_cfg.get("dir_stop_eps", 0.1)),
        max_steps=int(eval_cfg.get("max_steps", 500)),
        angle_step=int(eval_cfg.get("angle_step", 45)),
        step_size=float(eval_cfg.get("step_size", 1.0)),
        min_intersections=int(eval_cfg.get("min_intersections", 2)),
    )
    if "post_threshold" in eval_cfg:
        post.threshold = float(eval_cfg["post_threshold"])
    if "post_nms_size" in eval_cfg:
        post.nms_size = int(eval_cfg["post_nms_size"])
    if args.post_threshold is not None:
        post.threshold = float(args.post_threshold)
    if args.post_nms_size is not None:
        post.nms_size = int(args.post_nms_size)
    meter = MetricsCalculator()

    all_metrics = []
    per_sample_rows = []
    vector_dir = os.path.join(cfg["training"]["log_dir"], "eval", "vectors_geojson")
    os.makedirs(vector_dir, exist_ok=True)
    save_vectors = cfg.get("evaluation", {}).get("save_vectors_geojson", True)
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            out = model(img)
            for i in range(img.shape[0]):
                sample_id = batch["sample_id"][i]
                pred_res = post.postprocess(to_postprocess_input_from_pred(out, i))
                gt_res = post.postprocess(to_postprocess_input_from_gt(batch, i))
                # 有 GeoJSON 时可用矢量真值图；仅栅格标签时用后处理得到的 GT 图
                use_geojson = cfg["data"].get("gt_graph_source", "geojson") == "geojson"
                if use_geojson:
                    aoi_dir = cfg["data"].get("aoi_dir")
                    if aoi_dir:
                        gt_graph_geo = build_gt_graph_from_geojson(aoi_dir, sample_id)
                        if len(gt_graph_geo.nodes()) >= 2 and len(gt_graph_geo.edges()) > 0:
                            gt_res["graph"] = gt_graph_geo

                m = meter.calculate_all_metrics(pred_res, gt_res)
                all_metrics.append(m)
                pred_n = int(len(pred_res["graph"].nodes()))
                pred_e = int(len(pred_res["graph"].edges()))
                gt_n = int(len(gt_res["graph"].nodes()))
                gt_e = int(len(gt_res["graph"].edges()))
                strict_apls = m["topology_level"]["apls"] if (pred_n >= 2 and pred_e > 0 and gt_n >= 2 and gt_e > 0) else np.nan
                row = {
                    "sample_id": sample_id,
                    "pixel_iou": m["pixel_level"]["iou"],
                    "pixel_f1": m["pixel_level"]["f1_score"],
                    "apls": m["topology_level"]["apls"],
                    "strict_apls": strict_apls,
                    "topo_iou": m["topology_level"]["topo_iou"],
                    "inter_precision": m["intersection_level"]["precision"],
                    "inter_recall": m["intersection_level"]["recall"],
                    "inter_f1": m["intersection_level"]["f1_score"],
                    "pred_num_nodes": pred_n,
                    "pred_num_edges": pred_e,
                    "gt_num_nodes": gt_n,
                    "gt_num_edges": gt_e,
                }
                per_sample_rows.append(row)

                if save_vectors:
                    pred_geo = graph_to_geojson(pred_res["graph"], sample_id=sample_id, source="pred")
                    gt_geo = graph_to_geojson(gt_res["graph"], sample_id=sample_id, source="gt")
                    with open(os.path.join(vector_dir, f"{sample_id}_pred.geojson"), "w", encoding="utf-8") as f:
                        json.dump(pred_geo, f, ensure_ascii=False)
                    with open(os.path.join(vector_dir, f"{sample_id}_gt.geojson"), "w", encoding="utf-8") as f:
                        json.dump(gt_geo, f, ensure_ascii=False)

    def avg(path):
        vals = []
        for m in all_metrics:
            cur = m
            for k in path:
                cur = cur[k]
            vals.append(cur)
        return float(np.mean(vals)) if vals else 0.0

    summary = {
        "pixel_iou": avg(["pixel_level", "iou"]),
        "pixel_f1": avg(["pixel_level", "f1_score"]),
        "topology_apls": avg(["topology_level", "apls"]),
        "topology_topoiou": avg(["topology_level", "topo_iou"]),
        "intersection_precision": avg(["intersection_level", "precision"]),
        "intersection_recall": avg(["intersection_level", "recall"]),
        "intersection_f1": avg(["intersection_level", "f1_score"]),
        "intersection_type_acc": avg(["intersection_level", "type_accuracy"]),
        "num_samples": len(all_metrics),
        "checkpoint": ckpt,
    }
    strict_apls_values = [r["strict_apls"] for r in per_sample_rows if np.isfinite(r["strict_apls"])]
    summary["topology_apls_strict"] = float(np.mean(strict_apls_values)) if strict_apls_values else float("nan")
    summary["strict_apls_valid_samples"] = int(len(strict_apls_values))
    summary["strict_apls_invalid_samples"] = int(len(per_sample_rows) - len(strict_apls_values))

    out_dir = os.path.join(cfg["training"]["log_dir"], "eval")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "topology_eval.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out_dir, "topology_eval.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in summary.items():
            w.writerow([k, v])
    per_sample_csv = os.path.join(out_dir, "topology_eval_per_sample.csv")
    with open(per_sample_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id",
            "pixel_iou",
            "pixel_f1",
            "apls",
            "strict_apls",
            "topo_iou",
            "inter_precision",
            "inter_recall",
            "inter_f1",
            "pred_num_nodes",
            "pred_num_edges",
            "gt_num_nodes",
            "gt_num_edges",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(per_sample_rows)

    print("[Eval] Topology summary saved:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"[Eval] Per-sample APLS saved: {per_sample_csv}")
    if save_vectors:
        print(f"[Eval] Vector GeoJSON saved dir: {vector_dir}")


if __name__ == "__main__":
    main()

