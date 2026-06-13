"""
生成拓扑可视化图（优先真实 GeoJSON；无数据时回退示意图）。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError as e:
    raise SystemExit("缺少依赖 matplotlib，请先安装：pip install matplotlib") from e


def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_lines_from_geojson(gj: Dict) -> List[List[Tuple[float, float]]]:
    lines: List[List[Tuple[float, float]]] = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "LineString":
            if coords:
                lines.append([(float(x), float(y)) for x, y in coords])
        elif gtype == "MultiLineString":
            for seg in coords:
                if seg:
                    lines.append([(float(x), float(y)) for x, y in seg])
    return lines


def _extract_points_from_geojson(gj: Dict) -> List[Tuple[float, float]]:
    pts: List[Tuple[float, float]] = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        if geom.get("type") == "Point":
            c = geom.get("coordinates", [])
            if len(c) >= 2:
                pts.append((float(c[0]), float(c[1])))
    return pts


def _bbox(lines: List[List[Tuple[float, float]]]) -> Tuple[float, float, float, float]:
    xs = []
    ys = []
    for line in lines:
        for x, y in line:
            xs.append(x)
            ys.append(y)
    if not xs or not ys:
        return 0.0, 1.0, 0.0, 1.0
    return min(xs), max(xs), min(ys), max(ys)


def _pad_if_flat(
    xmin: float, xmax: float, ymin: float, ymax: float, pad: float = 1.0
) -> Tuple[float, float, float, float]:
    """避免单点/退化 bbox 导致 ylim 相同触发 matplotlib 警告。"""
    if xmax - xmin < 1e-9:
        xmin -= pad
        xmax += pad
    if ymax - ymin < 1e-9:
        ymin -= pad
        ymax += pad
    return xmin, xmax, ymin, ymax


def _draw_lines(ax, lines: List[List[Tuple[float, float]]], color: str, lw: float, alpha: float = 0.9):
    for line in lines:
        if len(line) < 2:
            continue
        arr = np.array(line)
        ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=lw, alpha=alpha)


def _degrade_lines(lines: List[List[Tuple[float, float]]], keep_every: int = 2) -> List[List[Tuple[float, float]]]:
    out: List[List[Tuple[float, float]]] = []
    for i, line in enumerate(lines):
        if i % keep_every == 0 and len(line) >= 2:
            out.append(line[::2] if len(line) > 6 else line)
    return out if out else lines[: max(1, len(lines) // 2)]


class TopologyVisualizer:
    def __init__(self, data_dir: str = "paper_materials") -> None:
        self.data_dir = Path(data_dir)
        self.figures_dir = self.data_dir / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)

        self.vectors_dir_candidates = [
            Path("logs_shanghai_thick_optimized_final/eval/vectors_geojson"),
            Path("logs_shanghai_thick/eval/vectors_geojson"),
        ]
        self.image_root_candidates = [
            Path("E:/Code/spacenet_filtered_thick/images"),
            Path("E:/Code/spacenet_filtered/images"),
        ]

    def _load_samples(self) -> List[Dict]:
        sample_file = self.data_dir / "visualization_samples.json"
        if not sample_file.exists():
            return []
        return _load_json(sample_file)

    def _find_geojson_pair(self, sample_id: str):
        for d in self.vectors_dir_candidates:
            pred = d / f"{sample_id}_pred.geojson"
            gt = d / f"{sample_id}_gt.geojson"
            if pred.exists() and gt.exists():
                return pred, gt
        return None, None

    def _find_image(self, sample_id: str) -> Optional[Path]:
        for root in self.image_root_candidates:
            for ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
                p = root / f"{sample_id}{ext}"
                if p.exists():
                    return p
        return None

    def _show_background(self, ax, sample_id: str):
        img_path = self._find_image(sample_id)
        if img_path is not None:
            img = plt.imread(str(img_path))
            ax.imshow(img)
            return True
        # 数据缺失时使用纯黑背景，明确区分于“随机占位”
        bg = np.zeros((224, 224, 3), dtype=np.float32)
        ax.imshow(bg)
        return False

    def _draw_single(self, sample: Dict) -> Path:
        sample_id = sample.get("sample_id", "unknown")
        apls = float(sample.get("apls", 0.0))
        strict_apls = sample.get("strict_apls", None)
        pixel_iou = float(sample.get("pixel_iou", 0.0))

        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
        ax_in, ax_ours, ax_gt = axes.flatten()

        pred_path, gt_path = self._find_geojson_pair(sample_id)
        pred_lines: List[List[Tuple[float, float]]] = []
        gt_lines: List[List[Tuple[float, float]]] = []
        pred_pts: List[Tuple[float, float]] = []
        gt_pts: List[Tuple[float, float]] = []

        if pred_path and gt_path:
            pred_gj = _load_json(pred_path)
            gt_gj = _load_json(gt_path)
            pred_lines = _extract_lines_from_geojson(pred_gj)
            gt_lines = _extract_lines_from_geojson(gt_gj)
            pred_pts = _extract_points_from_geojson(pred_gj)
            gt_pts = _extract_points_from_geojson(gt_gj)

        has_image = self._show_background(ax_in, sample_id)
        ax_in.set_title(f"Input {'RGB' if has_image else '(image missing)'}\n{sample_id}", fontsize=10)
        ax_in.axis("off")

        self._show_background(ax_ours, sample_id)
        if pred_lines:
            _draw_lines(ax_ours, pred_lines, color="#d62728", lw=1.8)
        if pred_pts:
            arr = np.array(pred_pts)
            ax_ours.scatter(arr[:, 0], arr[:, 1], s=8, c="#ff9896", alpha=0.9)
        ax_ours.set_xlim(0, 224)
        ax_ours.set_ylim(224, 0)
        ax_ours.set_title(f"Ours (Pred)\nAPLS={apls:.3f}", fontsize=10)
        ax_ours.axis("off")

        # 子图3：GT
        self._show_background(ax_gt, sample_id)
        if gt_lines:
            _draw_lines(ax_gt, gt_lines, color="#2ca02c", lw=1.8)
        if gt_pts:
            arr = np.array(gt_pts)
            ax_gt.scatter(arr[:, 0], arr[:, 1], s=8, c="#98df8a", alpha=0.9)
        ax_gt.set_xlim(0, 224)
        ax_gt.set_ylim(224, 0)
        strict_text = "nan" if strict_apls is None else f"{float(strict_apls):.3f}"
        ax_gt.set_title(f"Ground Truth\nStrict={strict_text}", fontsize=10)
        ax_gt.axis("off")

        fig.suptitle(
            f"Topology Visualization - {sample_id} | APLS={apls:.3f}, PixelIoU={pixel_iou:.3f}",
            fontsize=12,
        )
        plt.tight_layout(rect=[0, 0.02, 1, 0.95])
        out = self.figures_dir / f"topology_comparison_{sample_id}.png"
        fig.savefig(out, dpi=250, bbox_inches="tight")
        plt.close(fig)
        return out

    def _draw_summary(self, samples: List[Dict]) -> Path:
        apls = [float(s.get("apls", 0.0)) for s in samples]
        iou = [float(s.get("pixel_iou", 0.0)) for s in samples]
        strict = [s.get("strict_apls", None) for s in samples]
        strict_v = [float(x) for x in strict if x is not None]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].hist(apls, bins=min(8, max(3, len(apls))), color="#87ceeb", alpha=0.8)
        axes[0].set_title("APLS distribution")
        axes[0].set_xlabel("APLS")
        axes[0].grid(alpha=0.3)

        if strict_v:
            axes[1].hist(strict_v, bins=min(8, max(3, len(strict_v))), color="#f4a3a3", alpha=0.8)
            axes[1].set_title("Strict APLS distribution")
            axes[1].set_xlabel("Strict APLS")
        else:
            axes[1].text(0.5, 0.5, "No valid strict APLS", ha="center", va="center")
            axes[1].set_title("Strict APLS distribution")
        axes[1].grid(alpha=0.3)

        axes[2].scatter(apls, iou, s=42, color="#2ca02c")
        axes[2].set_title("APLS vs Pixel IoU")
        axes[2].set_xlabel("APLS")
        axes[2].set_ylabel("Pixel IoU")
        axes[2].grid(alpha=0.3)

        plt.tight_layout()
        out = self.figures_dir / "performance_summary.png"
        fig.savefig(out, dpi=250, bbox_inches="tight")
        plt.close(fig)
        return out

    def generate_all_figures(self, num_samples: int = 4) -> List[Path]:
        samples = self._load_samples()
        if not samples:
            print("[WARN] 未找到 visualization_samples.json，请先运行 prepare_paper_materials.py")
            return []
        selected = samples[: max(1, min(num_samples, len(samples)))]
        outputs: List[Path] = []
        for s in selected:
            out = self._draw_single(s)
            outputs.append(out)
            print(f"[OK] {out}")
        summary = self._draw_summary(selected)
        outputs.append(summary)
        print(f"[OK] {summary}")
        return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="生成拓扑可视化图")
    parser.add_argument("--num_samples", type=int, default=4, help="生成样本数量")
    parser.add_argument("--data_dir", default="paper_materials", help="材料目录")
    args = parser.parse_args()
    vis = TopologyVisualizer(data_dir=args.data_dir)
    files = vis.generate_all_figures(num_samples=args.num_samples)
    print("=" * 50)
    print(f"可视化完成，生成 {len(files)} 个文件。")
    for f in files:
        print(f"- {f}")


if __name__ == "__main__":
    main()
