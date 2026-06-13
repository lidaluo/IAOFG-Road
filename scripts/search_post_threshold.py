import csv
import json
import os
import sys
import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset_factory import build_datasets, get_all_sample_ids
from metrics.metrics import MetricsCalculator
from models.road_extraction_model import RoadExtractionModel
from postprocess.postprocessor import PostProcessor
from scripts.eval_topology import build_gt_graph_from_geojson
from scripts.train import split_ids


def to_pred_input(outputs, idx):
    seg_prob = torch.sigmoid(outputs["segmentation"][idx : idx + 1])
    inter = torch.sigmoid(outputs["intersection"][idx : idx + 1])
    orient = torch.tanh(outputs["orientation"][idx : idx + 1, 0:2])
    return {"segmentation": seg_prob.cpu(), "intersection": inter.cpu(), "direction_field": orient.cpu()}


def to_gt_input(batch, idx):
    mask = batch["mask"][idx : idx + 1]
    seg_2c = torch.cat([1.0 - mask, mask], dim=1)
    inter = batch["intersection"][idx : idx + 1]
    orient = batch["orientation"][idx : idx + 1, 0:2]
    return {"segmentation": seg_2c.cpu(), "intersection": inter.cpu(), "direction_field": orient.cpu()}


def evaluate_threshold(cfg, model, device, val_loader, val_ids, threshold, nms_size):
    eval_cfg = cfg.get("evaluation", {})
    post = PostProcessor(
        threshold=threshold,
        nms_size=nms_size,
        min_path_len=int(eval_cfg.get("min_path_len", 10)),
        endpoint_dist=float(eval_cfg.get("endpoint_dist", 10.0)),
        dir_stop_eps=float(eval_cfg.get("dir_stop_eps", 0.1)),
        max_steps=int(eval_cfg.get("max_steps", 500)),
        angle_step=int(eval_cfg.get("angle_step", 45)),
        step_size=float(eval_cfg.get("step_size", 1.0)),
        min_intersections=int(eval_cfg.get("min_intersections", 2)),
    )
    meter = MetricsCalculator()
    per_sample = []
    sid_ptr = 0
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            out = model(img)
            for i in range(img.shape[0]):
                sample_id = batch["sample_id"][i]
                pred_res = post.postprocess(to_pred_input(out, i))
                gt_res = post.postprocess(to_gt_input(batch, i))
                use_geojson = cfg["data"].get("gt_graph_source", "geojson") == "geojson"
                if use_geojson:
                    aoi_dir = cfg["data"].get("aoi_dir")
                    if aoi_dir:
                        gt_graph_geo = build_gt_graph_from_geojson(aoi_dir, sample_id)
                        if len(gt_graph_geo.nodes()) >= 2 and len(gt_graph_geo.edges()) > 0:
                            gt_res["graph"] = gt_graph_geo
                m = meter.calculate_all_metrics(pred_res, gt_res)
                pred_n, pred_e = len(pred_res["graph"].nodes()), len(pred_res["graph"].edges())
                gt_n, gt_e = len(gt_res["graph"].nodes()), len(gt_res["graph"].edges())
                strict_apls = m["topology_level"]["apls"] if (pred_n >= 2 and pred_e > 0 and gt_n >= 2 and gt_e > 0) else np.nan
                per_sample.append(
                    {
                        "sample_id": sample_id,
                        "strict_apls": strict_apls,
                        "topo_iou": m["topology_level"]["topo_iou"],
                        "inter_f1": m["intersection_level"]["f1_score"],
                    }
                )
                sid_ptr += 1

    strict_vals = [x["strict_apls"] for x in per_sample if np.isfinite(x["strict_apls"])]
    topo_vals = [x["topo_iou"] for x in per_sample]
    inter_vals = [x["inter_f1"] for x in per_sample]
    return {
        "threshold": threshold,
        "nms_size": nms_size,
        "strict_apls": float(np.mean(strict_vals)) if strict_vals else float("nan"),
        "strict_apls_valid_samples": len(strict_vals),
        "strict_apls_invalid_samples": len(per_sample) - len(strict_vals),
        "topo_iou": float(np.mean(topo_vals)) if topo_vals else 0.0,
        "intersection_f1": float(np.mean(inter_vals)) if inter_vals else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description="Search best post-process threshold.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to yaml config")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    eval_cfg = cfg.get("evaluation", {})
    thresholds = eval_cfg.get("threshold_candidates", [0.10, 0.15, 0.20, 0.25, 0.30, 0.35])
    nms_size = int(eval_cfg.get("post_nms_size", 3))

    all_ids = get_all_sample_ids(cfg)
    train_ids, val_ids = split_ids(
        all_ids,
        val_ratio=cfg["data"].get("val_ratio", 0.2),
        seed=cfg["data"].get("split_seed", 42),
    )
    _, dataset = build_datasets(
        cfg,
        train_ids,
        val_ids,
        tuple(cfg["data"].get("image_size", [224, 224])),
    )
    loader = DataLoader(dataset, batch_size=cfg["training"].get("batch_size", 2), shuffle=False, num_workers=0)

    ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "model_best_val_iou.pth")
    if not os.path.exists(ckpt):
        ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "model_latest.pth")
    model = RoadExtractionModel(encoder=cfg["model"]["encoder"], num_classes=1, input_size=tuple(cfg["data"].get("image_size", [224, 224])))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()

    rows = []
    for t in thresholds:
        row = evaluate_threshold(cfg, model, device, loader, val_ids, float(t), nms_size)
        rows.append(row)
        print(f"[Search] thr={t:.2f} strict_apls={row['strict_apls']:.4f} topo_iou={row['topo_iou']:.4f} inter_f1={row['intersection_f1']:.4f} valid={row['strict_apls_valid_samples']}")

    out_dir = os.path.join(cfg["training"]["log_dir"], "eval")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "threshold_search.csv")
    out_json = os.path.join(out_dir, "threshold_search.json")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "threshold",
                "nms_size",
                "strict_apls",
                "strict_apls_valid_samples",
                "strict_apls_invalid_samples",
                "topo_iou",
                "intersection_f1",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"[Search] Saved: {out_csv}")
    print(f"[Search] Saved: {out_json}")


if __name__ == "__main__":
    main()

