"""
后处理参数优化（thick / 任意 checkpoint）

该仓库的 `postprocess.PostProcessor` 目前只暴露两个可调参数：
  - threshold: 分割/热图阈值
  - nms_size: 非极大值抑制窗口大小

因此本脚本做 grid search：
  strict AP(L)S（仅在 pred/gt 都满足 pred_n>=2 且 pred_e>0 且 gt_n>=2 且 gt_e>0 时计入）
在阈值与 nms_size 的组合空间里寻找最优点。

输出：
  - {output_dir}/optimization_results.csv
  - {output_dir}/best_params.json
  - {output_dir}/optimization_log.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset_factory import build_datasets, get_all_sample_ids
from metrics.metrics import MetricsCalculator
from models.road_extraction_model import RoadExtractionModel
from postprocess.postprocessor import PostProcessor
from scripts.eval_topology import build_gt_graph_from_geojson
from scripts.train import split_ids


def _to_pred_input(outputs: dict[str, torch.Tensor], idx: int) -> dict[str, torch.Tensor]:
    seg_prob = torch.sigmoid(outputs["segmentation"][idx : idx + 1])
    inter = torch.sigmoid(outputs["intersection"][idx : idx + 1])
    orient = torch.tanh(outputs["orientation"][idx : idx + 1, 0:2])
    return {"segmentation": seg_prob.cpu(), "intersection": inter.cpu(), "direction_field": orient.cpu()}


def _to_gt_input(batch: dict[str, Any], idx: int) -> dict[str, torch.Tensor]:
    mask = batch["mask"][idx : idx + 1]  # [1,1,H,W]
    seg_2c = torch.cat([1.0 - mask, mask], dim=1)
    inter = batch["intersection"][idx : idx + 1]
    orient = batch["orientation"][idx : idx + 1, 0:2]
    return {"segmentation": seg_2c.cpu(), "intersection": inter.cpu(), "direction_field": orient.cpu()}


@dataclass
class EvalRow:
    post_threshold: float
    post_nms_size: int
    strict_apls: float
    strict_valid_samples: int
    strict_invalid_samples: int
    topo_iou: float
    intersection_f1: float


def evaluate_threshold(
    cfg: dict[str, Any],
    model: RoadExtractionModel,
    device: torch.device,
    val_loader: DataLoader,
    threshold: float,
    nms_size: int,
    min_path_len: int,
    endpoint_dist: float,
    dir_stop_eps: float,
    angle_step: int,
    step_size: float,
    num_eval_samples: int,
) -> dict[str, Any]:
    post = PostProcessor(
        threshold=threshold,
        nms_size=nms_size,
        min_path_len=min_path_len,
        endpoint_dist=endpoint_dist,
        dir_stop_eps=dir_stop_eps,
        angle_step=angle_step,
        step_size=step_size,
    )
    meter = MetricsCalculator()

    per_sample: list[dict[str, float]] = []
    with torch.no_grad():
        for batch in val_loader:
            img = batch["image"].to(device)
            out = model(img)
            for i in range(img.shape[0]):
                sample_id = batch["sample_id"][i]
                pred_res = post.postprocess(_to_pred_input(out, i))
                gt_res = post.postprocess(_to_gt_input(batch, i))

                use_geojson = cfg["data"].get("gt_graph_source", "geojson") == "geojson"
                if use_geojson:
                    aoi_dir = cfg["data"].get("aoi_dir")
                    if aoi_dir:
                        gt_graph_geo = build_gt_graph_from_geojson(aoi_dir, sample_id)
                        if len(gt_graph_geo.nodes()) >= 2 and len(gt_graph_geo.edges()) > 0:
                            gt_res["graph"] = gt_graph_geo

                m = meter.calculate_all_metrics(pred_res, gt_res)

                pred_n = int(len(pred_res["graph"].nodes()))
                pred_e = int(len(pred_res["graph"].edges()))
                gt_n = int(len(gt_res["graph"].nodes()))
                gt_e = int(len(gt_res["graph"].edges()))

                strict_apls = (
                    m["topology_level"]["apls"]
                    if (pred_n >= 2 and pred_e > 0 and gt_n >= 2 and gt_e > 0)
                    else np.nan
                )
                per_sample.append(
                    {
                        "strict_apls": strict_apls,
                        "topo_iou": float(m["topology_level"]["topo_iou"]),
                        "intersection_f1": float(m["intersection_level"]["f1_score"]),
                    }
                )
                if num_eval_samples and len(per_sample) >= num_eval_samples:
                    break
            if num_eval_samples and len(per_sample) >= num_eval_samples:
                break

    strict_vals = [x["strict_apls"] for x in per_sample if np.isfinite(x["strict_apls"])]
    topo_vals = [x["topo_iou"] for x in per_sample]
    inter_vals = [x["intersection_f1"] for x in per_sample]

    return {
        "strict_apls": float(np.mean(strict_vals)) if strict_vals else float("nan"),
        "strict_valid_samples": int(len(strict_vals)),
        "strict_invalid_samples": int(len(per_sample) - len(strict_vals)),
        "topo_iou": float(np.mean(topo_vals)) if topo_vals else 0.0,
        "intersection_f1": float(np.mean(inter_vals)) if inter_vals else 0.0,
        "total_samples": int(len(per_sample)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Optimize post-processing for strict APLS")
    ap.add_argument("--config", type=str, required=True, help="yaml config (data/model path)")
    ap.add_argument("--checkpoint", type=str, required=True, help="model checkpoint (.pth)")
    ap.add_argument("--output_dir", type=str, default="eval_results/optimization_postproc")
    ap.add_argument("--threshold_candidates", type=str, default="0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4")
    ap.add_argument("--nms_size_candidates", type=str, default="1,3,5")
    ap.add_argument("--min_path_len_candidates", type=str, default="6,8,10")
    ap.add_argument("--endpoint_dist_candidates", type=str, default="8,10,12")
    ap.add_argument("--dir_stop_eps_candidates", type=str, default="0.05,0.1")
    ap.add_argument("--angle_step_candidates", type=str, default="45")
    ap.add_argument("--step_size_candidates", type=str, default="1.0")
    ap.add_argument("--min_valid_ratio", type=float, default=0.8, help="要求 strict_valid_samples/total >= 此阈值")
    ap.add_argument("--num_eval_samples", type=int, default=0, help="0=全量；否则只评估前 N 个样本（加速粗搜）")
    ap.add_argument("--max_combinations", type=int, default=24)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / "optimization_log.txt"
    results_csv = out_dir / "optimization_results.csv"
    best_json = out_dir / "best_params.json"

    thresholds = [float(x) for x in args.threshold_candidates.split(",") if str(x).strip()]
    nms_sizes = [int(x) for x in args.nms_size_candidates.split(",") if str(x).strip()]
    min_path_lens = [int(x) for x in args.min_path_len_candidates.split(",") if str(x).strip()]
    endpoint_dists = [float(x) for x in args.endpoint_dist_candidates.split(",") if str(x).strip()]
    dir_stop_eps = [float(x) for x in args.dir_stop_eps_candidates.split(",") if str(x).strip()]
    angle_steps = [int(x) for x in args.angle_step_candidates.split(",") if str(x).strip()]
    step_sizes = [float(x) for x in args.step_size_candidates.split(",") if str(x).strip()]

    # dataset / loader
    image_size = tuple(cfg["data"].get("image_size", [224, 224]))
    all_ids = get_all_sample_ids(cfg)
    train_ids, val_ids = split_ids(
        all_ids,
        val_ratio=cfg["data"].get("val_ratio", 0.2),
        seed=cfg["data"].get("split_seed", 42),
    )
    _, dataset = build_datasets(cfg, train_ids, val_ids, image_size)
    val_loader = DataLoader(dataset, batch_size=cfg["training"].get("batch_size", 2), shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadExtractionModel(encoder=cfg["model"]["encoder"], num_classes=1, input_size=image_size)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    combos: list[tuple[float, int, int, float, float, int, float]] = []
    for t in thresholds:
        for n in nms_sizes:
            for mpl in min_path_lens:
                for ed in endpoint_dists:
                    for dse in dir_stop_eps:
                        for ang in angle_steps:
                            for ss in step_sizes:
                                combos.append((t, n, mpl, ed, dse, ang, ss))

    # 只跑前 max_combinations 组（候选顺序由用户提供决定：建议把“最可能”的放前面）
    combos = combos[: max(1, args.max_combinations)]

    total_combos = len(combos)
    rows: list[dict[str, Any]] = []
    best_score = -1.0
    best: dict[str, Any] | None = None

    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"checkpoint={ckpt_path}\nconfig={cfg_path}\n")
        lf.write(f"total val samples (expect)={len(dataset)}\n")
        lf.write(f"min_valid_ratio={args.min_valid_ratio}\n")
        lf.write(f"combos={total_combos}\n")

        for idx, (thr, nms, mpl, ed, dse, ang, ss) in enumerate(combos, start=1):
            t0 = time.time()
            metrics = evaluate_threshold(
                cfg,
                model,
                device,
                val_loader,
                threshold=thr,
                nms_size=nms,
                min_path_len=mpl,
                endpoint_dist=ed,
                dir_stop_eps=dse,
                angle_step=ang,
                step_size=ss,
                num_eval_samples=args.num_eval_samples,
            )
            total = metrics["total_samples"]
            valid_ratio = metrics["strict_valid_samples"] / max(total, 1)

            strict_apls = metrics["strict_apls"]
            strict_valid = metrics["strict_valid_samples"]
            strict_invalid = metrics["strict_invalid_samples"]
            topo_iou = metrics["topo_iou"]
            inter_f1 = metrics["intersection_f1"]

            score = float(strict_apls) if np.isfinite(strict_apls) and valid_ratio >= args.min_valid_ratio else -1.0

            row = {
                "post_threshold": thr,
                "post_nms_size": nms,
                "min_path_len": mpl,
                "endpoint_dist": ed,
                "dir_stop_eps": dse,
                "angle_step": ang,
                "step_size": ss,
                "strict_apls": float(strict_apls) if np.isfinite(strict_apls) else float("nan"),
                "strict_valid_samples": int(strict_valid),
                "strict_invalid_samples": int(strict_invalid),
                "topo_iou": float(topo_iou),
                "intersection_f1": float(inter_f1),
                "valid_ratio": valid_ratio,
                "score": score,
            }
            rows.append(row)

            msg = (
                f"[{idx}/{total_combos}] thr={thr:.3f} nms={nms} "
                f"min_path_len={mpl} endpoint_dist={ed} dir_stop_eps={dse} "
                f"strict_apls={('nan' if not np.isfinite(strict_apls) else f'{strict_apls:.4f}')} "
                f"valid={strict_valid}/{total} ({valid_ratio:.2f}) topo_iou={topo_iou:.4f} inter_f1={inter_f1:.4f} "
                f"time={time.time()-t0:.1f}s\n"
            )
            print(msg, end="")
            lf.write(msg)
            lf.flush()

            if score > best_score:
                best_score = score
                best = {
                    "post_threshold": thr,
                    "post_nms_size": nms,
                    "min_path_len": mpl,
                    "endpoint_dist": ed,
                    "dir_stop_eps": dse,
                    "angle_step": ang,
                    "step_size": ss,
                    "strict_apls": strict_apls,
                    "strict_valid_samples": strict_valid,
                    "strict_invalid_samples": strict_invalid,
                    "topo_iou": topo_iou,
                    "intersection_f1": inter_f1,
                    "valid_ratio": valid_ratio,
                    "checkpoint": str(ckpt_path),
                }

    # save csv
    with open(results_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "post_threshold",
                "post_nms_size",
                "min_path_len",
                "endpoint_dist",
                "dir_stop_eps",
                "angle_step",
                "step_size",
                "strict_apls",
                "strict_valid_samples",
                "strict_invalid_samples",
                "topo_iou",
                "intersection_f1",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["post_threshold"],
                    r["post_nms_size"],
                    r["min_path_len"],
                    r["endpoint_dist"],
                    r["dir_stop_eps"],
                    r["angle_step"],
                    r["step_size"],
                    r["strict_apls"],
                    r["strict_valid_samples"],
                    r["strict_invalid_samples"],
                    r["topo_iou"],
                    r["intersection_f1"],
                ]
            )

    if best is not None:
        best_json.write_text(json.dumps(best, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[OK] Best saved: {best_json}")
    else:
        print(f"\n[Warn] No valid combination meets min_valid_ratio={args.min_valid_ratio}. best_json not written.")

    print(f"[OK] CSV saved: {results_csv}")


if __name__ == "__main__":
    main()

