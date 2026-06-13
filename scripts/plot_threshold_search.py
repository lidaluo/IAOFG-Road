import csv
import os

import argparse
import matplotlib.pyplot as plt
import numpy as np


def _to_float(v):
    try:
        return float(v)
    except Exception:
        return np.nan


def main():
    parser = argparse.ArgumentParser(description="Plot threshold sensitivity curves.")
    parser.add_argument(
        "--eval-dir",
        type=str,
        default=os.path.join("logs", "eval"),
        help="Directory containing threshold_search.csv (and where plots will be saved).",
    )
    args = parser.parse_args()

    csv_path = os.path.join(args.eval_dir, "threshold_search.csv")
    out_dir = args.eval_dir
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"threshold search csv not found: {csv_path}")

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                {
                    "threshold": _to_float(r.get("threshold")),
                    "strict_apls": _to_float(r.get("strict_apls")),
                    "valid": _to_float(r.get("strict_apls_valid_samples")),
                    "topo_iou": _to_float(r.get("topo_iou")),
                    "inter_f1": _to_float(r.get("intersection_f1")),
                }
            )

    rows = sorted(rows, key=lambda x: x["threshold"])
    thr = np.array([x["threshold"] for x in rows], dtype=float)
    apls = np.array([x["strict_apls"] for x in rows], dtype=float)
    valid = np.array([x["valid"] for x in rows], dtype=float)
    topo = np.array([x["topo_iou"] for x in rows], dtype=float)
    inter = np.array([x["inter_f1"] for x in rows], dtype=float)

    # 图1：Strict APLS + 有效样本数（双轴）
    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(thr, apls, marker="o", color="#1f77b4", label="Strict APLS")
    ax1.set_xlabel("Post Threshold")
    ax1.set_ylabel("Strict APLS", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(thr, valid, marker="s", linestyle="--", color="#d62728", label="Valid Samples")
    ax2.set_ylabel("Valid Samples", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")

    fig.suptitle("Threshold Sensitivity: Strict APLS vs Valid Samples")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "threshold_apls_valid.png"), dpi=220)
    plt.close(fig)

    # 图2：TopoIoU + Intersection-F1
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(thr, topo, marker="o", label="TopoIoU", color="#2ca02c")
    ax.plot(thr, inter, marker="^", label="Intersection F1", color="#9467bd")
    ax.set_xlabel("Post Threshold")
    ax.set_ylabel("Metric")
    ax.set_title("Threshold Sensitivity: TopoIoU and Intersection F1")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "threshold_topoiou_interf1.png"), dpi=220)
    plt.close(fig)

    print("[Plot] Saved:")
    print(f"  {os.path.join(out_dir, 'threshold_apls_valid.png')}")
    print(f"  {os.path.join(out_dir, 'threshold_topoiou_interf1.png')}")


if __name__ == "__main__":
    main()

