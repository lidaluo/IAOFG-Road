"""
根据 config、training_log.json、eval 结果生成完整实验报告与图表，输出到 log_dir/report/。
用法: python scripts/generate_experiment_report.py --config configs/config_shanghai.yaml
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


def _series(logs, key):
    return [e[key] for e in logs if e.get(key) is not None]


def _epochs_for_key(logs, key):
    return [e["epoch"] for e in logs if e.get(key) is not None]


def plot_training_curves(logs, fig_dir: Path):
    epochs = [e["epoch"] for e in logs]
    fig_dir.mkdir(parents=True, exist_ok=True)

    # 1 total loss
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(epochs, _series(logs, "train_loss"), label="train total", lw=1.8)
    ve = _epochs_for_key(logs, "val_loss")
    vl = _series(logs, "val_loss")
    if vl:
        ax.plot(ve, vl, label="val total", lw=1.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Total loss (train / val)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig01_loss_total.png", dpi=200)
    plt.close(fig)

    # 2 IoU + F1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(epochs, _series(logs, "train_iou"), label="train IoU@0.5", lw=1.8)
    vi = _series(logs, "val_iou")
    if vi:
        ax1.plot(_epochs_for_key(logs, "val_iou"), vi, label="val IoU@0.5", lw=1.8)
    best = max(logs, key=lambda x: x.get("val_iou") or -1)
    ax1.axvline(best["epoch"], color="gray", ls="--", alpha=0.7, label=f'best val IoU ep{best["epoch"]}')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("IoU")
    ax1.set_title("IoU @ threshold 0.5")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, _series(logs, "train_f1"), label="train F1@0.5", lw=1.8)
    vf = _series(logs, "val_f1")
    if vf:
        ax2.plot(_epochs_for_key(logs, "val_f1"), vf, label="val F1@0.5", lw=1.8)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("F1")
    ax2.set_title("F1 @ threshold 0.5")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig02_iou_f1.png", dpi=200)
    plt.close(fig)

    # 3 LR
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, _series(logs, "learning_rate"), color="darkgreen", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning rate (ReduceLROnPlateau on val loss)")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig03_learning_rate.png", dpi=200)
    plt.close(fig)

    # 4 loss breakdown train
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, _series(logs, "train_seg_loss"), label="train seg", lw=1.5)
    ax.plot(epochs, _series(logs, "train_seg_bce"), label="train seg BCE", alpha=0.85, lw=1.2)
    ax.plot(epochs, _series(logs, "train_seg_dice"), label="train seg Dice", alpha=0.85, lw=1.2)
    ax.plot(epochs, _series(logs, "train_inter_loss"), label="train inter", lw=1.5)
    ax.plot(epochs, _series(logs, "train_orient_loss"), label="train orient", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss breakdown")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig04_loss_breakdown_train.png", dpi=200)
    plt.close(fig)

    # 5 val seg BCE vs val IoU
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ve = _epochs_for_key(logs, "val_seg_bce")
    vb = _series(logs, "val_seg_bce")
    vi = _series(logs, "val_iou")
    ax1.plot(ve, vb, color="C1", lw=1.8, label="val seg BCE")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Val seg BCE", color="C1")
    ax1.tick_params(axis="y", labelcolor="C1")
    ax2 = ax1.twinx()
    ax2.plot(_epochs_for_key(logs, "val_iou"), vi, color="C0", lw=1.8, label="val IoU")
    ax2.set_ylabel("Val IoU@0.5", color="C0")
    ax2.tick_params(axis="y", labelcolor="C0")
    ax1.set_title("Val segmentation BCE vs IoU (overfitting signal)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig05_val_bce_vs_iou.png", dpi=200)
    plt.close(fig)

    # 6 effective lambda inter / orient
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs, _series(logs, "lambda_inter_effective"), label="lambda_inter eff.", lw=2)
    ax.plot(epochs, _series(logs, "lambda_orient_effective"), label="lambda_orient eff.", lw=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Effective task weight")
    ax.set_title("Warmup + multitask ramp (effective aux weights)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig06_lambda_effective.png", dpi=200)
    plt.close(fig)

    # 7 train - val IoU gap
    gap = [t - v for t, v in zip(_series(logs, "train_iou"), _series(logs, "val_iou"))]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(epochs, gap, color="purple", lw=2)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("train IoU - val IoU")
    ax.set_title("Generalization gap (higher => more overfitting to train)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig07_train_val_gap.png", dpi=200)
    plt.close(fig)

    # 8 IoU @0.3 from logs (stored as train_iou_t01 in json - actually 0.3 per train.py print)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(epochs, _series(logs, "train_iou_t01"), label="train IoU@0.3", lw=1.5)
    ax.plot(_epochs_for_key(logs, "val_iou_t01"), _series(logs, "val_iou_t01"), label="val IoU@0.3", lw=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("IoU")
    ax.set_title("IoU @ threshold 0.3 (from training logs)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig08_iou_t03.png", dpi=200)
    plt.close(fig)


def plot_threshold_search(csv_path: Path, fig_dir: Path):
    if not csv_path.is_file():
        return
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    if not rows:
        return
    thr = [float(r["threshold"]) for r in rows]
    apls = []
    for r in rows:
        v = r.get("strict_apls", "")
        try:
            apls.append(float(v) if v and v.lower() != "nan" else np.nan)
        except ValueError:
            apls.append(np.nan)
    valid = [int(r["strict_apls_valid_samples"]) for r in rows]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(thr, apls, "o-", lw=2, ms=8, label="mean strict APLS")
    ax1.set_xlabel("Post-process threshold")
    ax1.set_ylabel("Mean strict APLS")
    ax1.set_title("Threshold search: strict APLS vs threshold")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.bar([t + 0.01 for t in thr], valid, width=0.02, alpha=0.35, color="gray", label="valid samples")
    ax2.set_ylabel("Strict-APLS valid sample count")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig09_threshold_apls.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([str(t) for t in thr], valid, color="steelblue")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Valid samples for strict APLS")
    ax.set_title("Valid sample count per threshold (low => unreliable mean APLS)")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig10_threshold_valid_counts.png", dpi=200)
    plt.close(fig)


def plot_per_sample_apls(per_sample_csv: Path, fig_dir: Path):
    if not per_sample_csv.is_file():
        return
    vals = []
    topo = []
    piou = []
    with open(per_sample_csv, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            try:
                s = r.get("strict_apls", "")
                if not s or s.lower() == "nan":
                    continue
                v_apls = float(s)
            except ValueError:
                continue
            try:
                p = float(r.get("pixel_iou", 0))
            except ValueError:
                continue
            try:
                t = float(r.get("topo_iou", 0))
            except ValueError:
                t = 0.0
            vals.append(v_apls)
            piou.append(p)
            topo.append(t)
    if not vals:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].hist(vals, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("strict APLS distribution (val split)")
    axes[0].set_xlabel("strict APLS")
    axes[0].set_ylabel("count")
    axes[1].hist(topo, bins=20, color="coral", edgecolor="white")
    axes[1].set_title("TopoIoU per sample (see report caveat)")
    axes[1].set_xlabel("TopoIoU")
    axes[2].hist(piou, bins=20, color="seagreen", edgecolor="white")
    axes[2].set_title("Pixel IoU per sample")
    axes[2].set_xlabel("pixel IoU")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig11_per_sample_hists.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(piou, vals, alpha=0.35, s=12)
    ax.set_xlabel("Pixel IoU")
    ax.set_ylabel("strict APLS")
    ax.set_title("Pixel IoU vs strict APLS (per sample)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig12_pixel_iou_vs_apls.png", dpi=200)
    plt.close(fig)


def copy_qualitative(infer_vis: Path, fig_dir: Path, max_n: int = 8):
    if not infer_vis.is_dir():
        return []
    pngs = sorted(infer_vis.glob("*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    sub = fig_dir / "qualitative"
    sub.mkdir(parents=True, exist_ok=True)
    for i, p in enumerate(pngs[:max_n]):
        dst = sub / f"qual_{i:02d}_{p.name}"
        shutil.copy2(p, dst)
        out.append(dst.relative_to(fig_dir.parent))
    return out


def flatten_yaml(d, prefix=""):
    rows = []
    if isinstance(d, dict):
        for k, v in d.items():
            p = f"{prefix}.{k}" if prefix else k
            rows.extend(flatten_yaml(v, p))
    else:
        rows.append((prefix, d))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config_shanghai.yaml")
    args = ap.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg_path = Path(args.config)
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log_dir = PROJECT_ROOT / cfg["training"]["log_dir"]
    report_dir = log_dir / "report"
    fig_dir = report_dir / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    log_json = log_dir / "training_log.json"
    if not log_json.is_file():
        print(f"[Error] Missing {log_json}")
        sys.exit(1)
    with open(log_json, "r", encoding="utf-8") as f:
        logs = json.load(f)

    plot_training_curves(logs, fig_dir)
    eval_dir = log_dir / "eval"
    plot_threshold_search(eval_dir / "threshold_search.csv", fig_dir)
    plot_per_sample_apls(eval_dir / "topology_eval_per_sample.csv", fig_dir)

    snap_path = report_dir / "config_snapshot.yaml"
    with open(snap_path, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, allow_unicode=True, default_flow_style=False, sort_keys=False)

    infer_vis = log_dir / "infer_vis"
    qual_rel = copy_qualitative(infer_vis, fig_dir)

    best = max(logs, key=lambda x: x.get("val_iou") or -1)
    last = logs[-1]
    n_train = int(len(logs) * 0.8) if cfg["data"].get("max_samples", 0) in (0, None) else None
    all_ids_n = "见 get_all_sample_ids"
    try:
        from data.dataset_factory import get_all_sample_ids
        from scripts.train import split_ids

        ids = get_all_sample_ids(cfg)
        ms = cfg["data"].get("max_samples", 0)
        if ms and ms > 0:
            ids = ids[:ms]
        tr, va = split_ids(ids, val_ratio=cfg["data"].get("val_ratio", 0.2), seed=cfg["data"].get("split_seed", 42))
        n_train, n_val = len(tr), len(va)
    except Exception:
        n_train, n_val = "—", "—"

    topo_path = eval_dir / "topology_eval.json"
    topo = {}
    if topo_path.is_file():
        with open(topo_path, "r", encoding="utf-8") as f:
            topo = json.load(f)

    flat_cfg = flatten_yaml(cfg)
    cfg_md_rows = "\n".join(f"| `{k}` | {repr(v)} |" for k, v in flat_cfg)

    qual_md = ""
    for rel in qual_rel:
        qual_md += f"\n![qual]({rel.as_posix()})\n\n*文件: `{rel}`*\n"

    if not qual_md.strip():
        qual_md = (
            "_当前无 `infer_vis/*.png`。请运行：_\n\n"
            "`python scripts/infer_visualize.py --config configs/config_shanghai.yaml --num_samples 8`\n\n"
            "然后重新运行本脚本。\n"
        )

    report_md = f"""# 上海 IAOF 实验报告（自动生成）

> 生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
> 配置文件：`{cfg_path}`  
> 配置快照：`report/config_snapshot.yaml`

---

## 目录

1. [概述与结论摘要](#1-概述与结论摘要)
2. [原理与方法](#2-原理与方法)
3. [实验设置](#3-实验设置)
4. [训练曲线与动态分析](#4-训练曲线与动态分析)
5. [离线评估与拓扑指标](#5-离线评估与拓扑指标)
6. [后处理阈值搜索](#6-后处理阈值搜索)
7. [定性可视化](#7-定性可视化)
8. [问题诊断与改进清单](#8-问题诊断与改进清单)
9. [附录：超参数表](#9-附录超参数表)

---

## 1. 概述与结论摘要

本实验在 **上海 filtered 数据集**（栅格 RGB / mask / 交叉口 heatmap / 方向场 npy）上训练 **交叉口锚定方向场（IAOF）** 多任务道路提取模型。编码器为 **Swin-Tiny + FPN**，三头输出道路分割、交叉口热力图、方向场（dx,dy + confidence）。

| 项目 | 结果 |
|------|------|
| 训练轮数 | {len(logs)} epochs |
| 最优验证 IoU@0.5 | **{best.get("val_iou", float("nan")):.4f}**（第 **{best.get("epoch")}** 轮） |
| 最后一轮 train/val IoU@0.5 | {last.get("train_iou", 0):.4f} / {last.get("val_iou", 0):.4f} |
| 评估权重 | `{topo.get("checkpoint", "未运行 eval_topology")}` |
| 评估集像素 IoU / F1 | {topo.get("pixel_iou", float("nan")):.4f} / {topo.get("pixel_f1", float("nan")):.4f} |
| 拓扑 APLS（样本均值） | {topo.get("topology_apls", float("nan")):.4f} |
| strict APLS（有效子集） | {topo.get("topology_apls_strict", float("nan")):.4f}（有效 {topo.get("strict_apls_valid_samples", "—")} / {topo.get("num_samples", "—")}） |
| TopoIoU（样本均值） | {topo.get("topology_topoiou", float("nan")):.4f} |
| 交叉口 P/R/F1 | {topo.get("intersection_precision", 0):.4f} / {topo.get("intersection_recall", 0):.4f} / {topo.get("intersection_f1", 0):.4f} |

**简要结论**：分割分支有可见学习效果；**交叉口精度极低、召回偏高**（假阳性多）；**APLS 中等**；**TopoIoU 接近 0** 需结合实现方式解读（见第 5 节 caveat）。整体适合作为 **baseline**，若要冲 SCI 需按第 8 节逐项加强。

---

## 2. 原理与方法

### 2.1 任务定义

同时预测：

1. **道路二值分割**（主任务，BCE + Dice，带动态正类权重缓解类别不平衡）。
2. **交叉口热力图**（BCE，固定/动态正类权重）。
3. **方向场**：前两通道经 `tanh` 约束为方向向量，第三通道 logits 经 BCE 监督置信度；在 GT 高置信道路区域用 **masked cosine** 约束方向与真值一致。
4. **锚定一致性损失**（`lambda_anchor`）：交叉口概率图的拉普拉斯与方向场散度在道路区域上互补，强化「交叉口–方向」几何一致。

### 2.2 网络结构

- **Backbone**：`timm` 的 `swin_tiny_patch4_window7_224`，`features_only=True`，多尺度特征转 **BCHW**。
- **Neck/Decoder**：FPN 融合多尺度通道至 256 维。
- **三个 1×1/3×3 卷积头**：分别输出 1 通道分割 logits、1 通道交叉口 logits、3 通道方向 logits。

### 2.3 训练策略

- **Warmup**：前 `seg_warmup_epochs` 轮主要练分割，`lambda_inter`/`lambda_orient` 用较小辅助权重 `warmup_aux_*`。
- **Ramp**：随后 `multitask_ramp_epochs` 轮线性升至完整 `lambda_inter`/`lambda_orient`。
- **优化器**：Adam；**ReduceLROnPlateau** 监控验证损失，按因子衰减学习率，直至 `min_lr`。

### 2.4 评估与拓扑指标

- **后处理**：`PostProcessor` 对分割/交叉口/方向场阈值化与 NMS，提取 **图结构**（节点、边，边长为像素距离）。
- **上海数据** `gt_graph_source: postprocess`：真值图由 **与预测相同流程** 从 GT 栅格生成，便于公平对比，但与 SpaceNet **官方矢量真值 + 官方 APLS 脚本** 不等价；论文中需单独说明。
- **APLS（实现于 `metrics/metrics.py`）**：对 GT 图所有可连通节点对计算最短路长度比 `min/max`，预测侧通过 **坐标最近邻（距离阈值 50px）** 将 GT 节点映射到预测节点后再算最短路；取平均。
- **strict APLS**：仅当 pred/gt 图均满足「节点数≥2 且边数>0」的样本才计入均值，避免空图拉偏。
- **TopoIoU caveat**：当前实现为 **pred 与 gt 图各自节点 ID 下的边集合** 的 |E∩E|/|E∪E|。Pred/GT **节点编号体系不同**，即使几何相近，边集在 ID 空间也难重叠，**均值接近 0 不完全等价于「无拓扑相似」**。改进方向：先对节点做几何匹配再映射边，或改用仅基于几何的边匹配指标。

---

## 3. 实验设置

### 3.1 数据

| 项 | 设置 |
|----|------|
| 布局 | `{cfg["data"].get("layout")}` |
| 根目录 | `{cfg["data"].get("root")}` |
| 子目录 | images / masks / heatmap / orientations |
| 方向场 | `{cfg["data"].get("orientation_ext", ".npy")}` |
| 输入尺寸 | `{cfg["data"].get("image_size")}` |
| 划分 | val_ratio={cfg["data"].get("val_ratio")}, seed={cfg["data"].get("split_seed")} |
| 训练 / 验证样本数 | {n_train} / {n_val} |

### 3.2 实现环境

- 见本机 Conda 环境 `road_extraction`；PyTorch + timm + opencv + networkx 等。

---

## 4. 训练曲线与动态分析

下列图像由 `training_log.json` 绘制，路径相对于本报告所在目录 `report/`。

![fig01](figures/fig01_loss_total.png)

**图 4-1** 总损失：验证损失中后期上升常与 **验证集 BCE 上升** 同步，而 IoU 横盘，提示 **过拟合或难例权重放大**。

![fig02](figures/fig02_iou_f1.png)

**图 4-2** IoU / F1：竖虚线为 **验证 IoU 最优 epoch**。若最优明显早于最后一轮，推理应优先 `model_best_val_iou.pth`。

![fig03](figures/fig03_learning_rate.png)

**图 4-3** 学习率衰减轨迹。

![fig04](figures/fig04_loss_breakdown_train.png)

**图 4-4** 训练各子损失：观察 inter / orient 是否在 ramp 后稳定在合理量级（相对 seg）。

![fig05](figures/fig05_val_bce_vs_iou.png)

**图 4-5** 验证 BCE 与 IoU 双轴：**BCE 上升而 IoU 不涨** 是典型的校准/过拟合信号，可尝试标签平滑、更强增广、或早停。

![fig06](figures/fig06_lambda_effective.png)

**图 4-6** Warmup + Ramp 有效任务权重。

![fig07](figures/fig07_train_val_gap.png)

**图 4-7** Train−Val IoU 差距：差距扩大说明 **泛化瓶颈**。

![fig08](figures/fig08_iou_t03.png)

**图 4-8** 阈值 0.3 下的 IoU（与 0.5 对比可写进消融）。

---

## 5. 离线评估与拓扑指标

评估脚本：`python scripts/eval_topology.py --config configs/config_shanghai.yaml`

若已生成 `eval/topology_eval.json`，主指标见第 1 节表格。

![fig11](figures/fig11_per_sample_hists.png)

**图 5-1** 验证集 per-sample：strict APLS、TopoIoU、像素 IoU 分布。

![fig12](figures/fig12_pixel_iou_vs_apls.png)

**图 5-2** 像素 IoU 与 strict APLS 散点：检查 **像素好但拓扑差** 的离群样本（后处理/图抽取问题）。

---

## 6. 后处理阈值搜索

脚本：`python scripts/search_post_threshold.py --config configs/config_shanghai.yaml`

![fig09](figures/fig09_threshold_apls.png)

**图 6-1** strict APLS 与有效样本数（右轴）：**有效样本过少时，均值 APLS 不可靠**。

![fig10](figures/fig10_threshold_valid_counts.png)

**图 6-2** 各阈值下 strict APLS 有效样本数柱状图。

---

## 7. 定性可视化

由 `scripts/infer_visualize.py` 生成，默认含 **Pred / GT 两行** 对比。

{qual_md}

---

## 8. 问题诊断与改进清单

按优先级（结合你当前日志与评估）：

1. **交叉口头**：precision 极低、recall 高 → 减少假阳性：提高 `inter_pos_weight` 上限策略、对 heatmap 加 **spatial NMS**、或增大 `lambda_inter` 与 focal 化；检查 **heatmap 标签是否与 224 缩放对齐**。
2. **像素 IoU 上限**：尝试 **更大输入**（若显存允许需改 Swin 实现或换 CNN backbone）、**Copy-Paste / 颜色抖动**、**MixUp 慎用**；或 **TTA** 推理。
3. **验证 BCE 上升**：**早停**（以 val IoU 为准）、**weight decay**、**Dropout**（解码器）、**标签平滑**。
4. **拓扑与后处理**：调 `post_threshold` / `post_nms_size`；骨架化、断边修复；图太碎会导致 APLS 节点匹配失败 → 看 `vectors_geojson`。
5. **TopoIoU 解读与改进**：实现 **匹配后的边 IoU** 或 **图编辑距离** 类指标，避免误导性 0。
6. **论文可比性**：若对比 SpaceNet 文献，需说明 **分辨率、阈值、是否官方 APLS**；必要时单独跑官方评测流水线。

---

## 9. 附录：超参数表

| 键 | 值 |
|----|-----|
{cfg_md_rows}

---

*报告由 `scripts/generate_experiment_report.py` 生成，修改后重新运行即可更新图表与表格。*
"""

    out_md = report_dir / "EXPERIMENT_REPORT.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[OK] Report: {out_md}")
    print(f"[OK] Figures: {fig_dir}")


if __name__ == "__main__":
    main()
