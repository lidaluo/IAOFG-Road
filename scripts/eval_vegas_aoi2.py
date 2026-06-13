"""
在 SpaceNet AOI2 Vegas（扁平 images/ + masks/）上，用上海训练权重推理并评估。
像素 IoU/F1 + 与训练口径一致的后处理拓扑指标（APLS 等）。
GT 无官方 heatmap/orient 时，由掩膜合成近似 heatmap / 方向场（见 data/vegas_rgb_mask_dataset.py）。
"""
from __future__ import annotations

import argparse
import math
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.shanghai_filtered_dataset import list_image_stems
from data.vegas_rgb_mask_dataset import VegasRGBMaskDataset, mask_road_fraction
from metrics.metrics import MetricsCalculator
from models.road_extraction_model import RoadExtractionModel
from postprocess.postprocessor import PostProcessor
def to_postprocess_input_from_pred(outputs, idx):
    seg_logit = outputs["segmentation"][idx : idx + 1]
    seg_prob = torch.sigmoid(seg_logit)
    inter = torch.sigmoid(outputs["intersection"][idx : idx + 1])
    orient = torch.tanh(outputs["orientation"][idx : idx + 1, 0:2])
    return {"segmentation": seg_prob.cpu(), "intersection": inter.cpu(), "direction_field": orient.cpu()}


def to_postprocess_input_from_gt(batch, idx):
    mask = batch["mask"][idx : idx + 1]
    seg_2c = torch.cat([1.0 - mask, mask], dim=1)
    inter = batch["intersection"][idx : idx + 1]
    orient = batch["orientation"][idx : idx + 1, 0:2]
    return {"segmentation": seg_2c.cpu(), "intersection": inter.cpu(), "direction_field": orient.cpu()}


def _pixel_iou_f1(pred_prob: np.ndarray, gt_mask: np.ndarray, thresh: float) -> Tuple[float, float]:
    p = (pred_prob > thresh).astype(np.float32)
    g = (gt_mask > 0.5).astype(np.float32)
    inter = float((p * g).sum())
    union = float(p.sum() + g.sum() - inter)
    iou = inter / (union + 1e-6)
    prec = inter / (float(p.sum()) + 1e-6)
    rec = inter / (float(g.sum()) + 1e-6)
    f1 = 2.0 * prec * rec / (prec + rec + 1e-6)
    return iou, f1


def _select_stems(
    masks_dir: str,
    stems: List[str],
    min_road_frac: float,
    top_n: int,
    progress_every: int = 100,
) -> List[str]:
    scored: List[Tuple[str, float]] = []
    n_total = len(stems)
    for i, s in enumerate(stems):
        mp = None
        for ext in (".png", ".jpg", ".tif", ".tiff"):
            p = os.path.join(masks_dir, s + ext)
            if os.path.isfile(p):
                mp = p
                break
        if mp is None:
            continue
        m = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
        r = mask_road_fraction(m)
        if r >= min_road_frac:
            scored.append((s, r))
        if progress_every > 0 and (i + 1) % progress_every == 0:
            print(f"[Vegas] 筛选掩膜进度: {i + 1}/{n_total} …", flush=True)
    scored.sort(key=lambda x: -x[1])
    if top_n > 0:
        scored = scored[:top_n]
    return [s for s, _ in scored]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config_vegas_aoi2_eval.yaml")
    ap.add_argument("--min-road-frac", type=float, default=0.05, help="掩膜道路占比下限，0 表示不过滤")
    ap.add_argument("--top-n", type=int, default=0, help="按道路占比取前 N，0 表示不截断")
    ap.add_argument("--max-samples", type=int, default=0, help="最终评估样本数上限，0 表示全部")
    ap.add_argument("--pred-dir", type=str, default=None, help="保存预测概率图目录，默认 <Vegas_root>/predictions")
    ap.add_argument("--shanghai-ref", type=str, default="logs_shanghai_thick_optimized_final/eval/topology_eval.json")
    ap.add_argument(
        "--apls-max-gt-nodes",
        type=int,
        default=64,
        help="APLS 计算时 GT 图节点数上限（子采样），避免交叉口过多时 O(n²) 卡死；0 表示不限制（很慢）",
    )
    ap.add_argument(
        "--vegas-root",
        type=str,
        default=None,
        help="覆盖 config 中 data.root（Vegas 数据根目录）",
    )
    ap.add_argument(
        "--masks-subdir",
        type=str,
        default=None,
        help="覆盖 config 中 data.masks_subdir（如 masks 或 masks_thick）",
    )
    ap.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="覆盖 config 中 training.log_dir（区分原始/厚评估输出）",
    )
    args = ap.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.vegas_root:
        cfg["data"]["root"] = args.vegas_root
    if args.masks_subdir:
        cfg["data"]["masks_subdir"] = args.masks_subdir
    if args.log_dir:
        cfg["training"]["log_dir"] = args.log_dir

    data = cfg["data"]
    root = data["root"]
    images_dir = os.path.join(root, data["images_subdir"])
    masks_dir = os.path.join(root, data["masks_subdir"])
    image_size = tuple(data.get("image_size", [224, 224]))

    all_stems = list_image_stems(images_dir)
    print(f"[Vegas] 图像列表: {len(all_stems)} 个 stem，开始筛选…", flush=True)
    if args.min_road_frac > 0 or args.top_n > 0:
        stems = _select_stems(masks_dir, all_stems, args.min_road_frac, args.top_n)
    else:
        stems = all_stems
    if args.max_samples > 0:
        stems = stems[: args.max_samples]

    if not stems:
        raise RuntimeError("没有可选样本，请检查路径与筛选条件")

    pred_dir = args.pred_dir
    if not pred_dir:
        sub = data.get("masks_subdir", "masks")
        pred_dir = (
            os.path.join(root, "predictions_thick")
            if "thick" in str(sub).lower()
            else os.path.join(root, "predictions")
        )
    pred_dir = pred_dir if os.path.isabs(pred_dir) else os.path.join(PROJECT_ROOT, pred_dir)
    os.makedirs(pred_dir, exist_ok=True)
    print(f"[Vegas] 选中 {len(stems)} 个样本，预测输出目录: {pred_dir}", flush=True)

    selection_path = os.path.join(pred_dir, "vegas_selection.json")
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "root": root,
                "num_selected": len(stems),
                "min_road_frac": args.min_road_frac,
                "top_n": args.top_n,
                "max_samples": args.max_samples,
                "masks_subdir": data.get("masks_subdir", "masks"),
                "config_path": cfg_path,
                "sample_ids": stems,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    ds = VegasRGBMaskDataset(
        root=root,
        images_subdir=data["images_subdir"],
        masks_subdir=data["masks_subdir"],
        sample_ids=stems,
        image_size=image_size,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadExtractionModel(
        encoder=cfg["model"]["encoder"], num_classes=1, input_size=image_size
    )
    ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "model_best_val_iou.pth")
    if not os.path.isfile(ckpt):
        ckpt = os.path.join(cfg["training"]["checkpoint_dir"], "model_latest.pth")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"找不到权重: {ckpt}")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.to(device)
    model.eval()
    print(f"[Vegas] 设备: {device}，权重已加载，开始逐张推理（共 {len(stems)} 张）…", flush=True)

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

    seg_thresh = float(eval_cfg.get("post_threshold", post.threshold))
    apls_cap = args.apls_max_gt_nodes
    meter = MetricsCalculator(
        max_gt_nodes_for_apls=apls_cap if apls_cap > 0 else 0,
    )
    if apls_cap > 0:
        print(f"[Vegas] APLS 子采样: GT 节点数 > {apls_cap} 时随机保留 {apls_cap} 个（防卡死）", flush=True)
    all_metrics: List[Dict[str, Any]] = []
    per_sample_rows: List[Dict[str, Any]] = []
    pixel_ious: List[float] = []
    pixel_f1s: List[float] = []

    log_dir = cfg["training"].get("log_dir", "logs_vegas_aoi2_eval")
    log_dir = log_dir if os.path.isabs(log_dir) else os.path.join(PROJECT_ROOT, log_dir)
    eval_out = os.path.join(log_dir, "eval")
    os.makedirs(eval_out, exist_ok=True)

    n_done = 0
    n_all = len(stems)
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].to(device)
            sid = batch["sample_id"][0]
            n_done += 1
            print(f"[Vegas] [{n_done}/{n_all}] {sid} …", flush=True)
            out = model(img)
            seg_prob = torch.sigmoid(out["segmentation"][0, 0]).cpu().numpy()
            gt_hw = batch["mask"][0, 0].cpu().numpy()

            iou, f1 = _pixel_iou_f1(seg_prob, gt_hw, seg_thresh)
            pixel_ious.append(iou)
            pixel_f1s.append(f1)

            prob_png = os.path.join(pred_dir, f"{sid}_seg_prob.png")
            cv2.imwrite(prob_png, (np.clip(seg_prob, 0, 1) * 255).astype(np.uint8))
            bin_png = os.path.join(pred_dir, f"{sid}_seg_bin.png")
            cv2.imwrite(bin_png, ((seg_prob > seg_thresh) * 255).astype(np.uint8))

            pred_res = post.postprocess(to_postprocess_input_from_pred(out, 0))
            gt_res = post.postprocess(to_postprocess_input_from_gt(batch, 0))
            m = meter.calculate_all_metrics(pred_res, gt_res)
            print(
                f"      pixel_iou={m['pixel_level']['iou']:.3f} apls={m['topology_level']['apls']:.3f}",
                flush=True,
            )
            all_metrics.append(m)

            pred_n = int(len(pred_res["graph"].nodes()))
            pred_e = int(len(pred_res["graph"].edges()))
            gt_n = int(len(gt_res["graph"].nodes()))
            gt_e = int(len(gt_res["graph"].edges()))
            strict_apls = (
                m["topology_level"]["apls"]
                if (pred_n >= 2 and pred_e > 0 and gt_n >= 2 and gt_e > 0)
                else np.nan
            )
            per_sample_rows.append(
                {
                    "sample_id": sid,
                    "pixel_iou": m["pixel_level"]["iou"],
                    "pixel_f1": m["pixel_level"]["f1_score"],
                    "apls": m["topology_level"]["apls"],
                    "strict_apls": strict_apls,
                    "topo_iou": m["topology_level"]["topo_iou"],
                    "pred_num_nodes": pred_n,
                    "pred_num_edges": pred_e,
                    "gt_num_nodes": gt_n,
                    "gt_num_edges": gt_e,
                }
            )

    def avg(path: List[str]) -> float:
        vals = []
        for mm in all_metrics:
            cur = mm
            for k in path:
                cur = cur[k]
            vals.append(cur)
        return float(np.mean(vals)) if vals else 0.0

    summary: Dict[str, Any] = {
        "pixel_iou": avg(["pixel_level", "iou"]),
        "pixel_f1": avg(["pixel_level", "f1_score"]),
        "topology_apls": avg(["topology_level", "apls"]),
        "topology_topoiou": avg(["topology_level", "topo_iou"]),
        "num_samples": len(all_metrics),
        "checkpoint": ckpt,
        "vegas_root": root,
        "selection_file": selection_path,
        "pred_dir": pred_dir,
        "pixel_iou_thresh_match": float(np.mean(pixel_ious)) if pixel_ious else 0.0,
        "pixel_f1_thresh_match": float(np.mean(pixel_f1s)) if pixel_f1s else 0.0,
        "apls_max_gt_nodes": apls_cap if apls_cap > 0 else None,
        "masks_subdir": data.get("masks_subdir", "masks"),
    }
    strict_vals = [r["strict_apls"] for r in per_sample_rows if np.isfinite(r["strict_apls"])]
    summary["topology_apls_strict"] = float(np.mean(strict_vals)) if strict_vals else float("nan")
    summary["strict_apls_valid_samples"] = int(len(strict_vals))
    summary["strict_apls_invalid_samples"] = int(len(per_sample_rows) - len(strict_vals))

    strict_sa = summary.get("topology_apls_strict")
    strict_report = (
        f"{strict_sa:.4f}"
        if isinstance(strict_sa, float) and not math.isnan(strict_sa)
        else "N/A"
    )

    json_path = os.path.join(eval_out, "vegas_topology_eval.json")
    summary_json = dict(summary)
    if isinstance(summary_json.get("topology_apls_strict"), float) and math.isnan(
        summary_json["topology_apls_strict"]
    ):
        summary_json["topology_apls_strict"] = None
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(eval_out, "vegas_topology_eval_per_sample.csv")
    if per_sample_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(per_sample_rows[0].keys()))
            w.writeheader()
            w.writerows(per_sample_rows)
    else:
        csv_path = "(无逐样本数据)"

    shanghai_ref = args.shanghai_ref
    if not os.path.isabs(shanghai_ref):
        shanghai_ref = os.path.join(PROJECT_ROOT, shanghai_ref)
    shanghai_block = ""
    if os.path.isfile(shanghai_ref):
        with open(shanghai_ref, "r", encoding="utf-8") as f:
            sj = json.load(f)
        shanghai_block = f"""
- 参考文件: `{shanghai_ref}`
- Topology APLS (mean): {sj.get('topology_apls', 0):.4f}
- Strict APLS: {sj.get('topology_apls_strict', 0):.4f}
- Pixel IoU: {sj.get('pixel_iou', 0):.4f}
- Strict 有效样本: {sj.get('strict_apls_valid_samples', 0)}/{sj.get('num_samples', 0)}
"""

    _msub = data.get("masks_subdir", "masks")
    _thick_tag = "（厚掩膜 `" + str(_msub) + "`）" if "thick" in str(_msub).lower() else ""

    report_path = os.path.join(eval_out, "VEGAS_AOI2_TEST_REPORT.md")
    report = f"""## 拉斯维加斯数据集测试结果报告{_thick_tag}

### 1. 数据集概况

- 数据路径: `{root}`
- 掩膜子目录: `{_msub}`
- 测试图像数量: **{len(stems)}** 张（筛选后）
- 道路占比下限: {args.min_road_frac}；Top-N: {args.top_n or "未使用"}；max_samples: {args.max_samples or "全部"}
- 数据特点: 网格化、结构化强、道路占比相对较高（Vegas AOI2）
- 样本列表与参数: `{selection_path}`

### 2. 模型与设置

- 权重: `{ckpt}`（上海 thick 训练 `model_best_val_iou.pth`）
- 输入尺寸: {image_size}
- 后处理（与上海 optimized final 对齐）: `post_threshold={eval_cfg.get("post_threshold")}`, `post_nms_size={eval_cfg.get("post_nms_size")}`, `min_path_len={eval_cfg.get("min_path_len")}`, `endpoint_dist={eval_cfg.get("endpoint_dist")}`, `dir_stop_eps={eval_cfg.get("dir_stop_eps")}`
- APLS 计算: GT 节点数超过 **{apls_cap if apls_cap > 0 else "未启用"}** 时子采样（`--apls-max-gt-nodes`，避免 O(n²) 卡死）

### 3. 模型表现

**像素（MetricsCalculator 与掩膜）**

- Pixel IoU: **{summary['pixel_iou']:.4f}**
- Pixel F1: **{summary['pixel_f1']:.4f}**

**与后处理阈值一致的二值掩膜 IoU/F1（便于与分割论文对比）**

- Pixel IoU @ thresh={seg_thresh}: **{summary['pixel_iou_thresh_match']:.4f}**
- Pixel F1 @ thresh={seg_thresh}: **{summary['pixel_f1_thresh_match']:.4f}**

**拓扑（基于后处理图；GT 由掩膜合成 heatmap/方向场，非官方矢量，仅作跨城相对参考）**

- Topology APLS (mean): **{summary['topology_apls']:.4f}**
- Strict APLS: **{strict_report}**（有效样本 {summary['strict_apls_valid_samples']}/{summary['num_samples']}）
- Topology TopoIoU: **{summary['topology_topoiou']:.4f}**

- 逐样本 CSV: `{csv_path}`
- JSON 汇总: `{json_path}`
- 预测概率图目录: `{pred_dir}`

### 4. 与上海数据对比

{shanghai_block if shanghai_block else "- 未找到上海参考 JSON，请将 `--shanghai-ref` 指向 `logs_shanghai_thick_optimized_final/eval/topology_eval.json`"}

**差异说明（阅读时注意）**

- 上海结果为 **spacenet_filtered_thick** 上训练/验证口径；Vegas 为 **零样本泛化**，且 GT 拓扑由 **掩膜近似生成**，Strict APLS 与上海 **不可直接等同**，宜作为相对趋势参考。
- 若需与官方 SpaceNet 矢量真值严格对齐的 APLS，需增加 GeoJSON→像素的配准（本数据集 PNG 未在仓库内提供 worldfile 时未启用）。

### 5. 结论与建议

- 已在 Vegas AOI2 上完成批量推理与指标汇总；预测掩膜见 `{pred_dir}`（默认：`predictions/` 或厚掩膜时 `predictions_thick/`）。
- 建议将 Vegas 作为「结构化较强」对照城，与上海 thick 结果并列报告，并注明 GT 拓扑近似带来的 APLS 解释边界。

---
*报告由 `scripts/eval_vegas_aoi2.py` 自动生成*
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("[Vegas] summary:", json.dumps(summary_json, indent=2, ensure_ascii=False))
    print("[Vegas] report:", report_path)


if __name__ == "__main__":
    main()
