from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as patches


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "paper_materials" / "figures"


def _box(ax, x, y, w, h, txt, color="#eaf2ff", fs=10):
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
    ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs)


def _arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="->", lw=1.2))


def make_pipeline_flowchart():
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.02, 0.35, 0.15, 0.28, "Input Image\n(RGB)")
    _box(ax, 0.21, 0.35, 0.15, 0.28, "Multi-task Network\n(Swin+FPN)")
    _box(ax, 0.40, 0.55, 0.18, 0.18, "Road Mask\nPrediction", "#ffe9e9")
    _box(ax, 0.40, 0.32, 0.18, 0.18, "Intersection\nHeatmap", "#e9ffe9")
    _box(ax, 0.40, 0.09, 0.18, 0.18, "Direction Field\n(dx, dy, conf)", "#e9edff")
    _box(ax, 0.62, 0.35, 0.15, 0.28, "Post-processing\nNMS + Tracing", "#fff2d9")
    _box(ax, 0.81, 0.35, 0.17, 0.28, "Road Graph\n(nodes + edges)", "#f3e9ff")

    _arrow(ax, 0.17, 0.49, 0.21, 0.49)
    _arrow(ax, 0.36, 0.49, 0.40, 0.64)
    _arrow(ax, 0.36, 0.49, 0.40, 0.41)
    _arrow(ax, 0.36, 0.49, 0.40, 0.18)
    _arrow(ax, 0.58, 0.64, 0.62, 0.49)
    _arrow(ax, 0.58, 0.41, 0.62, 0.49)
    _arrow(ax, 0.58, 0.18, 0.62, 0.49)
    _arrow(ax, 0.77, 0.49, 0.81, 0.49)

    ax.set_title("Pipeline of Intersection-Anchored Direction Field Road Extraction", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "pipeline_flowchart_en.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_network_architecture():
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    _box(ax, 0.03, 0.62, 0.14, 0.16, "Input\n224x224x3", "#cfe8ff")
    _box(ax, 0.22, 0.62, 0.18, 0.16, "Swin-Tiny\nBackbone", "#d5f5d1")
    _box(ax, 0.45, 0.62, 0.18, 0.16, "FPN\nFusion", "#fff0bf")

    _box(ax, 0.70, 0.78, 0.23, 0.10, "Segmentation Head", "#ffd1d1")
    _box(ax, 0.70, 0.62, 0.23, 0.10, "Intersection Head", "#d8ffd8")
    _box(ax, 0.70, 0.46, 0.23, 0.10, "Orientation Head", "#d7e3ff")

    _box(ax, 0.06, 0.28, 0.22, 0.10, "L_seg = BCE + Dice", "#ffe5e5")
    _box(ax, 0.34, 0.28, 0.22, 0.10, "L_inter = BCE", "#e6ffe6")
    _box(ax, 0.62, 0.28, 0.22, 0.10, "L_orient = L_vec + L_conf", "#e8edff")
    _box(ax, 0.33, 0.10, 0.34, 0.10, "L_anchor = mean(M * (ΔH + div F)^2)", "#f6e6ff")

    _arrow(ax, 0.17, 0.70, 0.22, 0.70)
    _arrow(ax, 0.40, 0.70, 0.45, 0.70)
    _arrow(ax, 0.63, 0.70, 0.70, 0.83)
    _arrow(ax, 0.63, 0.70, 0.70, 0.67)
    _arrow(ax, 0.63, 0.70, 0.70, 0.51)
    _arrow(ax, 0.81, 0.78, 0.17, 0.38)
    _arrow(ax, 0.81, 0.62, 0.45, 0.38)
    _arrow(ax, 0.81, 0.46, 0.73, 0.38)
    _arrow(ax, 0.81, 0.62, 0.50, 0.20)
    _arrow(ax, 0.81, 0.46, 0.50, 0.20)

    ax.set_title("Network Architecture and Multi-task Losses", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "network_architecture_en.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    make_pipeline_flowchart()
    make_network_architecture()
    print("Saved:", OUT / "pipeline_flowchart_en.png")
    print("Saved:", OUT / "network_architecture_en.png")


if __name__ == "__main__":
    main()
