"""
生成论文缺失图表（尽量使用真实实验日志/结果，不使用纯模拟数据）。
输出目录：paper_materials/figures
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = ROOT / "paper_materials" / "figures"


def _load_json(path: Path):
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _to_float(v: str, default: float = np.nan) -> float:
    try:
        if v is None:
            return default
        vv = str(v).strip().lower()
        if vv in {"nan", "none", ""}:
            return default
        return float(v)
    except Exception:
        return default


def create_network_architecture_figure() -> Path:
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        rect = patches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=color,
            edgecolor="black",
            linewidth=1.2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    box(0.03, 0.62, 0.14, 0.16, "Input RGB\n224x224x3", "#cfe8ff")
    box(0.22, 0.62, 0.18, 0.16, "Swin-Tiny\n(features_only)", "#d5f5d1")
    box(0.45, 0.62, 0.18, 0.16, "FPN Fusion\n4-level", "#fff0bf")

    box(0.70, 0.77, 0.22, 0.12, "Seg Head\nConv3x3-ReLU-Conv1x1", "#ffd1d1")
    box(0.70, 0.60, 0.22, 0.12, "Inter Head\nConv3x3-ReLU-Conv1x1", "#d8ffd8")
    box(0.70, 0.43, 0.22, 0.12, "Orient Head\nConv3x3-ReLU-Conv1x1", "#d7e3ff")

    box(0.08, 0.25, 0.22, 0.12, "L_seg = BCE + Dice", "#ffe5e5")
    box(0.34, 0.25, 0.22, 0.12, "L_inter = BCE", "#e6ffe6")
    box(0.60, 0.25, 0.22, 0.12, "L_orient = L_cos + L_conf", "#e8edff")
    box(0.34, 0.06, 0.26, 0.12, "L_anchor = mean(M * (ΔH + div F)^2)", "#f6e6ff")

    def arr(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))

    arr(0.17, 0.70, 0.22, 0.70)
    arr(0.40, 0.70, 0.45, 0.70)
    arr(0.63, 0.70, 0.70, 0.83)
    arr(0.63, 0.70, 0.70, 0.66)
    arr(0.63, 0.70, 0.70, 0.49)
    arr(0.81, 0.77, 0.19, 0.37)
    arr(0.81, 0.60, 0.45, 0.37)
    arr(0.81, 0.43, 0.71, 0.37)
    arr(0.81, 0.60, 0.47, 0.12)
    arr(0.81, 0.43, 0.47, 0.12)

    ax.set_title("Figure 1. IAOF Network Architecture and Loss Flow", fontsize=14, fontweight="bold")
    out = FIG_DIR / "network_architecture.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def create_loss_weight_sensitivity_figure() -> Path:
    logs = _load_json(ROOT / "logs_shanghai_thick" / "training_log.json") or []
    epochs, li, lo, la, val_iou = [], [], [], [], []
    for row in logs:
        if row.get("val_iou") is None:
            continue
        epochs.append(int(row["epoch"]))
        li.append(float(row.get("lambda_inter_effective", np.nan)))
        lo.append(float(row.get("lambda_orient_effective", np.nan)))
        la.append(float(row.get("train_anchor_loss", np.nan)))
        val_iou.append(float(row["val_iou"]))

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(epochs, li, label="lambda_inter_effective", color="#cc0000", lw=2)
    ax1.plot(epochs, lo, label="lambda_orient_effective", color="#008800", lw=2)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Effective loss weight")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, val_iou, label="val IoU@0.5", color="#0044cc", lw=2, linestyle="--")
    ax2.set_ylabel("Validation IoU")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="lower right", fontsize=9)
    plt.title("Figure 2. Loss Weights (effective) and Validation IoU")
    out = FIG_DIR / "loss_weight_sensitivity.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def create_threshold_sensitivity_figure() -> Path:
    rows = _load_csv(ROOT / "eval_results" / "extended_optimization_v2" / "optimization_results.csv")
    # 聚合到阈值：strict_apls 均值（忽略 nan）
    agg: Dict[float, List[float]] = {}
    for r in rows:
        thr = _to_float(r.get("post_threshold"))
        sa = _to_float(r.get("strict_apls"), np.nan)
        if np.isnan(thr) or np.isnan(sa):
            continue
        agg.setdefault(thr, []).append(sa)
    thrs = sorted(agg.keys())
    vals = [float(np.mean(agg[t])) for t in thrs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thrs, vals, marker="o", lw=2, color="#1f77b4")
    if vals:
        i = int(np.argmax(vals))
        ax.scatter([thrs[i]], [vals[i]], color="red", zorder=3)
        ax.text(thrs[i], vals[i], f" best={thrs[i]:.2f}", fontsize=9, va="bottom")
    ax.set_xlabel("post_threshold")
    ax.set_ylabel("mean strict APLS (subset optimization)")
    ax.set_title("Figure 3. Threshold Sensitivity from Optimization Log")
    ax.grid(alpha=0.3)
    out = FIG_DIR / "threshold_sensitivity_enhanced.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def create_success_cases_figure() -> Path:
    # 直接拼接已有 infer_vis 结果图，避免空占位。
    candidates = [
        ROOT / "logs_shanghai" / "infer_vis" / "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1138_infer.png",
        ROOT / "logs_shanghai" / "infer_vis" / "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1606_infer.png",
        ROOT / "logs_shanghai" / "infer_vis" / "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1607_infer.png",
        ROOT / "logs_shanghai" / "infer_vis" / "SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1942_infer.png",
    ]
    imgs = [p for p in candidates if p.exists()]
    if not imgs:
        # 空图兜底
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.text(0.5, 0.5, "No infer_vis examples found", ha="center", va="center")
        ax.axis("off")
        out = FIG_DIR / "success_cases.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out

    n = len(imgs)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, imgs):
        im = plt.imread(str(p))
        ax.imshow(im)
        ax.set_title(p.stem.replace("_infer", ""))
        ax.axis("off")
    plt.suptitle("Figure 4. Qualitative Success Cases", fontsize=14, fontweight="bold")
    out = FIG_DIR / "success_cases.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def create_failure_analysis_figure() -> Path:
    rows = _load_csv(ROOT / "logs_shanghai_thick_optimized_final" / "eval" / "topology_eval_per_sample.csv")
    # 失败定义：strict_apls 最低的有效样本 + invalid 样本统计
    valid = []
    invalid_count = 0
    for r in rows:
        sa = _to_float(r.get("strict_apls"), np.nan)
        if np.isnan(sa):
            invalid_count += 1
            continue
        valid.append((r.get("sample_id", ""), sa, _to_float(r.get("pred_num_edges"), 0.0), _to_float(r.get("gt_num_edges"), 0.0)))
    valid_sorted = sorted(valid, key=lambda x: x[1])[:8]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    if valid_sorted:
        labels = [v[0].split("_img")[-1] for v in valid_sorted]
        vals = [v[1] for v in valid_sorted]
        ax1.barh(range(len(vals)), vals, color="#ff7f7f")
        ax1.set_yticks(range(len(vals)))
        ax1.set_yticklabels(labels, fontsize=8)
        ax1.invert_yaxis()
        ax1.set_xlabel("strict APLS")
        ax1.set_title("Worst valid samples (strict APLS)")
    else:
        ax1.text(0.5, 0.5, "No valid strict samples", ha="center", va="center")
        ax1.axis("off")

    total = len(rows)
    ax2.pie(
        [invalid_count, max(total - invalid_count, 0)],
        labels=["strict invalid", "strict valid"],
        autopct="%1.1f%%",
        colors=["#ffb3b3", "#b3e6b3"],
    )
    ax2.set_title("Strict APLS computability")

    plt.suptitle("Figure 5. Failure Analysis", fontsize=14, fontweight="bold")
    out = FIG_DIR / "failure_analysis.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def create_training_convergence_figure() -> Path:
    thick_logs = _load_json(ROOT / "logs_shanghai_thick" / "training_log.json") or []
    thin_logs = _load_json(ROOT / "logs_shanghai" / "training_log.json") or []

    def ext(logs):
        ep = [int(x["epoch"]) for x in logs if x.get("val_iou") is not None]
        tr = [float(x.get("train_iou", np.nan)) for x in logs if x.get("val_iou") is not None]
        va = [float(x.get("val_iou", np.nan)) for x in logs if x.get("val_iou") is not None]
        return ep, tr, va

    ep_tk, tr_tk, va_tk = ext(thick_logs)
    ep_tn, tr_tn, va_tn = ext(thin_logs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(ep_tk, tr_tk, label="train IoU", color="#1f77b4")
    axes[0].plot(ep_tk, va_tk, label="val IoU", color="#d62728")
    axes[0].set_title("Thick-mask training")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("IoU")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(ep_tn, tr_tn, label="train IoU", color="#1f77b4", linestyle="--")
    axes[1].plot(ep_tn, va_tn, label="val IoU", color="#d62728", linestyle="--")
    axes[1].set_title("Thin-mask training")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("IoU")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.suptitle("Figure 6. Convergence Comparison: Thick vs Thin", fontsize=14, fontweight="bold")
    out = FIG_DIR / "training_convergence.png"
    fig.tight_layout()
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        create_network_architecture_figure(),
        create_loss_weight_sensitivity_figure(),
        create_threshold_sensitivity_figure(),
        create_success_cases_figure(),
        create_failure_analysis_figure(),
        create_training_convergence_figure(),
    ]
    # 将历史图（若存在）统一重存为 300 DPI，满足论文提交要求。
    for legacy in [
        "performance_summary.png",
        "topology_comparison_SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1332.png",
        "topology_comparison_SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1651.png",
        "topology_comparison_SN3_roads_train_AOI_4_Shanghai_PS-RGB_img1813.png",
        "topology_comparison_SN3_roads_train_AOI_4_Shanghai_PS-RGB_img437.png",
    ]:
        p = FIG_DIR / legacy
        if p.exists():
            im = Image.open(p)
            im.save(p, dpi=(300, 300))
            outputs.append(p)
    print("Generated figures:")
    for p in outputs:
        print(f"- {p}")


if __name__ == "__main__":
    main()
