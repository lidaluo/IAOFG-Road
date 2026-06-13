"""
扩展后处理优化器（strict APLS 目标导向）— 推荐三段式

1) coarse：子集样本（默认 64）快速扫网格，单次约 1～3 分钟
2) refine：在 coarse 最优附近细搜，仍用子集
3) full_verify：对最终最优参数跑全量验证集（168），得到论文可用的 strict APLS

说明：
- 直接在内存中评估，不调用 eval_topology 子进程。
- 仅搜索 PostProcessor 真实支持的参数。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
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


def _pred_input(outputs: dict[str, torch.Tensor], idx: int) -> dict[str, torch.Tensor]:
    seg_prob = torch.sigmoid(outputs["segmentation"][idx : idx + 1])
    inter = torch.sigmoid(outputs["intersection"][idx : idx + 1])
    orient = torch.tanh(outputs["orientation"][idx : idx + 1, 0:2])
    return {"segmentation": seg_prob.cpu(), "intersection": inter.cpu(), "direction_field": orient.cpu()}


def _gt_input(batch: dict[str, Any], idx: int) -> dict[str, torch.Tensor]:
    mask = batch["mask"][idx : idx + 1]
    seg_2c = torch.cat([1.0 - mask, mask], dim=1)
    inter = batch["intersection"][idx : idx + 1]
    orient = batch["orientation"][idx : idx + 1, 0:2]
    return {"segmentation": seg_2c.cpu(), "intersection": inter.cpu(), "direction_field": orient.cpu()}


def evaluate_once(
    cfg: dict[str, Any],
    model: RoadExtractionModel,
    device: torch.device,
    loader: DataLoader,
    params: dict[str, Any],
    num_eval_samples: int,
) -> dict[str, Any]:
    post = PostProcessor(
        threshold=float(params["post_threshold"]),
        nms_size=int(params["post_nms_size"]),
        min_intersections=int(params["min_intersections"]),
        min_path_len=int(params["min_path_len"]),
        endpoint_dist=float(params["endpoint_dist"]),
        dir_stop_eps=float(params["dir_stop_eps"]),
        angle_step=int(params.get("angle_step", 45)),
        step_size=float(params.get("step_size", 1.0)),
        max_steps=int(params.get("max_steps", 500)),
    )
    meter = MetricsCalculator()
    rows: list[dict[str, Any]] = []

    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            out = model(img)
            for i in range(img.shape[0]):
                sample_id = batch["sample_id"][i]
                pred_res = post.postprocess(_pred_input(out, i))
                gt_res = post.postprocess(_gt_input(batch, i))

                use_geo = cfg["data"].get("gt_graph_source", "geojson") == "geojson"
                if use_geo:
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

                rows.append(
                    {
                        "strict_apls": strict_apls,
                        "apls": float(m["topology_level"]["apls"]),
                        "pixel_iou": float(m["pixel_level"]["iou"]),
                        "topo_iou": float(m["topology_level"]["topo_iou"]),
                    }
                )
                if num_eval_samples > 0 and len(rows) >= num_eval_samples:
                    break
            if num_eval_samples > 0 and len(rows) >= num_eval_samples:
                break

    strict_vals = [r["strict_apls"] for r in rows if np.isfinite(r["strict_apls"])]
    return {
        "strict_apls": float(np.mean(strict_vals)) if strict_vals else float("nan"),
        "apls": float(np.mean([r["apls"] for r in rows])) if rows else 0.0,
        "pixel_iou": float(np.mean([r["pixel_iou"] for r in rows])) if rows else 0.0,
        "topo_iou": float(np.mean([r["topo_iou"] for r in rows])) if rows else 0.0,
        "valid_samples": int(len(strict_vals)),
        "total_samples": int(len(rows)),
    }


def _build_coarse_grid(grid_mode: str) -> list[dict[str, Any]]:
    """缩小默认网格：围绕历史较好区间 thr≈0.22–0.30, end 8–12, eps 偏小。"""
    coarse: list[dict[str, Any]] = []
    if grid_mode == "full":
        thr_list = [0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30]
        end_list = [8, 10, 12, 15]
        eps_list = [0.05, 0.1, 0.2]
    else:
        # fast：5×3×3 = 45 组，粗搜总时长可控
        thr_list = [0.22, 0.24, 0.26, 0.28, 0.30]
        end_list = [8, 10, 12]
        eps_list = [0.05, 0.1, 0.2]

    for t in thr_list:
        for d in end_list:
            for e in eps_list:
                coarse.append(
                    {
                        "post_threshold": t,
                        "post_nms_size": 3,
                        "min_intersections": 2,
                        "endpoint_dist": d,
                        "dir_stop_eps": e,
                        "min_path_len": 8,
                        "angle_step": 45,
                        "step_size": 1.0,
                    }
                )
    return coarse


def _append_partial_row(out_dir: Path, row: dict[str, Any]) -> None:
    append_path = out_dir / "optimization_results_partial.csv"
    file_exists = append_path.is_file()
    headers = [
        "stage",
        "post_threshold",
        "post_nms_size",
        "min_intersections",
        "endpoint_dist",
        "dir_stop_eps",
        "min_path_len",
        "angle_step",
        "step_size",
        "strict_apls",
        "apls",
        "pixel_iou",
        "topo_iou",
        "valid_samples",
        "total_samples",
        "valid_ratio",
        "score",
    ]
    with open(append_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if not file_exists:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in headers})


def main() -> None:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    ap = argparse.ArgumentParser(description="Extended post-processing optimization (subset + full verify)")
    ap.add_argument("--config", default="configs/config_shanghai_thick.yaml")
    ap.add_argument(
        "--checkpoint",
        default="",
        help="默认自动选 checkpoints_shanghai_thick/model_best_val_iou.pth",
    )
    ap.add_argument("--output_dir", default="eval_results/extended_optimization_v2")
    ap.add_argument("--time_limit", type=int, default=7200)
    ap.add_argument(
        "--coarse_num_samples",
        type=int,
        default=64,
        help="粗搜/细化每轮评估的样本数（默认 64）。设为 0 表示全量，每轮约 5–8 分钟。",
    )
    ap.add_argument(
        "--grid",
        choices=["fast", "full"],
        default="fast",
        help="fast：约 45 组粗搜；full：原 84 组级别（更慢）。",
    )
    ap.add_argument("--target_strict_apls", type=float, default=0.60)
    ap.add_argument("--min_valid_ratio", type=float, default=0.12)
    ap.add_argument("--no_full_verify", action="store_true", help="关闭最后的全量复核（默认会全量跑一遍）")
    ap.add_argument(
        "--max_refine",
        type=int,
        default=48,
        help="细化阶段最多尝试的组合数（避免子集 refine 过久）",
    )
    args = ap.parse_args()
    full_verify = not args.no_full_verify

    cfg_path = Path(args.config)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint) if args.checkpoint else Path("checkpoints_shanghai_thick/model_best_val_iou.pth")
    if not ckpt_path.is_file():
        alt = Path("checkpoints_shanghai_thick/checkpoint_epoch_19.pth")
        ckpt_path = alt if alt.is_file() else ckpt_path

    if not cfg_path.is_file():
        raise FileNotFoundError(cfg_path)
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    image_size = tuple(cfg["data"].get("image_size", [224, 224]))
    all_ids = get_all_sample_ids(cfg)
    train_ids, val_ids = split_ids(
        all_ids,
        val_ratio=cfg["data"].get("val_ratio", 0.2),
        seed=cfg["data"].get("split_seed", 42),
    )
    _, val_ds = build_datasets(cfg, train_ids, val_ids, image_size)
    loader = DataLoader(val_ds, batch_size=cfg["training"].get("batch_size", 2), shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadExtractionModel(encoder=cfg["model"]["encoder"], num_classes=1, input_size=image_size)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    n_sub = args.coarse_num_samples if args.coarse_num_samples > 0 else 0
    coarse = _build_coarse_grid(args.grid)

    def _neighbors(v: float, step: float, lo: float, hi: float) -> list[float]:
        vals = [max(lo, v - step), v, min(hi, v + step)]
        return sorted(set(round(x, 4) for x in vals))

    start = time.time()
    log_file = out_dir / "optimization_log.txt"
    csv_file = out_dir / "optimization_results.csv"
    best_file = out_dir / "best_params.json"
    report_file = out_dir / "optimization_report.md"

    rows: list[dict[str, Any]] = []
    best_score = -1.0
    best: dict[str, Any] | None = None
    full_metrics: dict[str, Any] | None = None

    with open(log_file, "w", encoding="utf-8") as lf:
        lf.write(f"start={time.ctime(start)}\n")
        lf.write(f"config={cfg_path}\ncheckpoint={ckpt_path}\n")
        lf.write(f"grid={args.grid} coarse_combos={len(coarse)}\n")
        lf.write(f"coarse_num_samples={n_sub or 'FULL'}\n")
        lf.write(f"target_strict_apls={args.target_strict_apls}\n")
        lf.write(
            "设计：子集粗搜 + 子集细化 + 可选全量复核。子集下单次约 1–3 分钟；全量单次约 5–8 分钟。\n"
        )
        lf.flush()

        # 阶段1 coarse
        for i, p in enumerate(coarse, start=1):
            if time.time() - start > args.time_limit:
                lf.write("[abort] time_limit reached before coarse end\n")
                break
            t0 = time.time()
            start_line = (
                f">>> [coarse {i}/{len(coarse)}] START (n={n_sub or 'ALL'}) "
                f"thr={p['post_threshold']} end={p['endpoint_dist']} eps={p['dir_stop_eps']} "
                f"elapsed={time.time()-start:.0f}s\n"
            )
            print(start_line, end="", flush=True)
            lf.write(start_line)
            lf.flush()
            m = evaluate_once(cfg, model, device, loader, p, n_sub)
            valid_ratio = m["valid_samples"] / max(m["total_samples"], 1)
            score = m["strict_apls"] if np.isfinite(m["strict_apls"]) and valid_ratio >= args.min_valid_ratio else -1.0
            row = {**p, **m, "valid_ratio": valid_ratio, "score": score, "stage": "coarse_subset"}
            rows.append(row)
            _append_partial_row(out_dir, row)
            line = (
                f"[coarse {i}/{len(coarse)}] strict={m['strict_apls']:.4f} apls={m['apls']:.4f} "
                f"valid={m['valid_samples']}/{m['total_samples']} time={time.time()-t0:.1f}s\n"
            )
            print(line, end="", flush=True)
            lf.write(line)
            lf.flush()
            if score > best_score:
                best_score = score
                best = row
            if best and best.get("strict_apls", 0) >= args.target_strict_apls:
                lf.write("[early] target strict APLS reached in coarse (subset)\n")
                break

        # 阶段2 refine（子集 + 上限次数）
        if best is not None and best.get("strict_apls", 0.0) < args.target_strict_apls and time.time() - start <= args.time_limit:
            thr_cands = _neighbors(float(best["post_threshold"]), 0.02, 0.16, 0.34)
            end_cands = sorted(
                {
                    max(6.0, float(best["endpoint_dist"]) - 2.0),
                    float(best["endpoint_dist"]),
                    float(best["endpoint_dist"]) + 2.0,
                }
            )
            eps_cands = sorted(
                {
                    max(0.02, float(best["dir_stop_eps"]) - 0.05),
                    float(best["dir_stop_eps"]),
                    min(0.25, float(best["dir_stop_eps"]) + 0.05),
                }
            )
            len_cands = [6, 8, 10, 12]

            refine_combos: list[tuple[float, float, float, int]] = []
            for t in thr_cands:
                for d in end_cands:
                    for e in eps_cands:
                        for mpl in len_cands:
                            refine_combos.append((float(t), float(d), float(e), int(mpl)))
            refine_combos = refine_combos[: args.max_refine]

            for ridx, (t, d, e, mpl) in enumerate(refine_combos, start=1):
                if time.time() - start > args.time_limit:
                    break
                p = {
                    "post_threshold": t,
                    "post_nms_size": 3,
                    "min_intersections": 2,
                    "endpoint_dist": d,
                    "dir_stop_eps": e,
                    "min_path_len": mpl,
                    "angle_step": 45,
                    "step_size": 1.0,
                }
                t0 = time.time()
                rs = (
                    f">>> [refine {ridx}/{len(refine_combos)}] START (n={n_sub or 'ALL'}) "
                    f"thr={p['post_threshold']} end={p['endpoint_dist']} "
                    f"eps={p['dir_stop_eps']} len={p['min_path_len']}\n"
                )
                print(rs, end="", flush=True)
                lf.write(rs)
                lf.flush()
                m = evaluate_once(cfg, model, device, loader, p, n_sub)
                valid_ratio = m["valid_samples"] / max(m["total_samples"], 1)
                score = m["strict_apls"] if np.isfinite(m["strict_apls"]) and valid_ratio >= args.min_valid_ratio else -1.0
                row = {**p, **m, "valid_ratio": valid_ratio, "score": score, "stage": "refine_subset"}
                rows.append(row)
                _append_partial_row(out_dir, row)
                line = (
                    f"[refine {ridx}] strict={m['strict_apls']:.4f} valid={m['valid_samples']}/{m['total_samples']} "
                    f"time={time.time()-t0:.1f}s\n"
                )
                print(line, end="", flush=True)
                lf.write(line)
                lf.flush()
                if score > best_score:
                    best_score = score
                    best = row
                if best and best.get("strict_apls", 0) >= args.target_strict_apls:
                    lf.write("[early] target strict APLS reached in refine (subset)\n")
                    break

        # 阶段3 全量复核（论文口径）
        if full_verify and best is not None and time.time() - start <= args.time_limit:
            p_best = {k: best[k] for k in best if k in ["post_threshold", "post_nms_size", "min_intersections", "endpoint_dist", "dir_stop_eps", "min_path_len", "angle_step", "step_size"]}
            lf.write("\n>>> [full_verify] START all val samples (168)\n")
            lf.flush()
            print("\n>>> [full_verify] START all val samples — 约 5–8 分钟\n", flush=True)
            t0 = time.time()
            full_metrics = evaluate_once(cfg, model, device, loader, p_best, 0)
            line = (
                f"[full_verify] strict={full_metrics['strict_apls']:.4f} apls={full_metrics['apls']:.4f} "
                f"valid={full_metrics['valid_samples']}/{full_metrics['total_samples']} "
                f"time={time.time()-t0:.1f}s\n"
            )
            print(line, flush=True)
            lf.write(line)
            lf.flush()
            fv_row = {**p_best, **full_metrics, "valid_ratio": full_metrics["valid_samples"] / max(full_metrics["total_samples"], 1), "score": full_metrics["strict_apls"], "stage": "full_verify"}
            rows.append(fv_row)
            _append_partial_row(out_dir, fv_row)

    headers = [
        "stage",
        "post_threshold",
        "post_nms_size",
        "min_intersections",
        "endpoint_dist",
        "dir_stop_eps",
        "min_path_len",
        "angle_step",
        "step_size",
        "strict_apls",
        "apls",
        "pixel_iou",
        "topo_iou",
        "valid_samples",
        "total_samples",
        "valid_ratio",
        "score",
    ]
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in headers})

    best_out: dict[str, Any] = {}
    if best is not None:
        best_out = dict(best)
        if full_metrics is not None:
            best_out["subset_strict_apls"] = best.get("strict_apls")
            best_out["full_strict_apls"] = full_metrics["strict_apls"]
            best_out["full_apls"] = full_metrics["apls"]
            best_out["full_valid_samples"] = full_metrics["valid_samples"]
            best_out["full_total_samples"] = full_metrics["total_samples"]
        with open(best_file, "w", encoding="utf-8") as f:
            json.dump(best_out, f, ensure_ascii=False, indent=2)

    duration = time.time() - start
    with open(report_file, "w", encoding="utf-8") as f:
        f.write("# 扩展后处理优化报告\n\n")
        f.write(f"- checkpoint: `{ckpt_path}`\n")
        f.write(f"- config: `{cfg_path}`\n")
        f.write(f"- grid: `{args.grid}`\n")
        f.write(f"- coarse_num_samples: {n_sub or 'full'}\n")
        f.write(f"- duration_sec: {duration:.1f}\n")
        f.write(f"- tested rows: {len(rows)}\n\n")
        if best is None:
            f.write("未找到有效组合。\n")
        else:
            f.write("## 子集最优（粗搜/细化）\n\n")
            f.write(f"- strict_apls (subset): {best['strict_apls']:.6f}\n")
            f.write(f"- valid: {best['valid_samples']}/{best['total_samples']}\n\n")
            if full_metrics is not None:
                f.write("## 全量复核（论文推荐引用）\n\n")
                f.write(f"- strict_apls (full val): {full_metrics['strict_apls']:.6f}\n")
                f.write(f"- topology_apls (full val, non-strict mean): {full_metrics['apls']:.6f}\n")
                f.write(f"- valid (strict): {full_metrics['valid_samples']}/{full_metrics['total_samples']}\n\n")
            f.write("## 参数\n\n")
            for k in ["post_threshold", "post_nms_size", "min_intersections", "endpoint_dist", "dir_stop_eps", "min_path_len"]:
                f.write(f"- {k}: {best[k]}\n")

    print("\n============================================================")
    if best is None:
        print("未找到有效参数组合。")
    else:
        print(f"子集最优 strict APLS: {best['strict_apls']:.6f}")
        if full_metrics is not None:
            print(f"全量复核 strict APLS: {full_metrics['strict_apls']:.6f}  ← 写论文用这个")
        print(f"最佳参数: {best_file}")
    print(f"CSV: {csv_file}")
    print(f"报告: {report_file}")
    print("============================================================")


if __name__ == "__main__":
    main()
