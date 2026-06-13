"""
在评估输出目录下，对 **RL 修复前 / 后** 的骨架与 **GT 骨架** 做基于采样的拓扑指标：

- 在骨架前景上随机采样；
- 半径 ``r`` 内匹配：**Topo-Precision**（预测点命中 GT）、**Topo-Recall**（GT 点命中预测）、**Topo-F1**。

GT：优先读 ``region_{id}_gt_skeleton.png``；否则从 ``--gt_root`` 查找 ``region_{id}_gt.png`` 等，经 ``skeletonize`` 生成并缓存到同目录。

用法::

    python calculate_topo_metrics.py --eval_dir eval_results --gt_root E:\\datasets\\cityscaledataset\\cityscale\\20cities --radius 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import numpy as np
from scipy.spatial import cKDTree
from skimage.morphology import skeletonize

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Topo-Precision / Recall / F1（图采样）")
    p.add_argument("--eval_dir", type=str, default="eval_results")
    p.add_argument(
        "--gt_root",
        type=str,
        default=r"E:\datasets\cityscaledataset\cityscale\20cities",
    )
    p.add_argument("--radius", type=float, default=10.0)
    p.add_argument("--n_samples", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _find_gt_png(gt_root: Path, region_id: int) -> Optional[Path]:
    rid = int(region_id)
    for p in (
        gt_root / f"region_{rid}" / f"region_{rid}_gt.png",
        gt_root / f"region_{rid}_gt.png",
        gt_root / f"{rid}_gt.png",
    ):
        if p.is_file():
            return p
    for sub in gt_root.glob(f"region_{rid}*"):
        if sub.is_dir():
            for name in (f"region_{rid}_gt.png", "gt.png"):
                q = sub / name
                if q.is_file():
                    return q
    return None


def _gt_skeleton_from_png(gt_png: Path, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = target_hw
    m = cv2.imread(str(gt_png), cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(str(gt_png))
    if m.shape != (th, tw):
        m = cv2.resize(m, (tw, th), interpolation=cv2.INTER_NEAREST)
    binm = m > 127
    if not np.any(binm):
        return np.zeros((th, tw), dtype=np.uint8)
    sk = skeletonize(binm)
    return (sk.astype(np.uint8) * 255)


def sample_points_from_skeleton(skel_u8: np.ndarray, n_max: int, rng: np.random.Generator) -> np.ndarray:
    ys, xs = np.where(skel_u8 > 127)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float64)
    n = min(int(n_max), len(xs))
    pick = rng.choice(len(xs), size=n, replace=False)
    return np.stack([xs[pick], ys[pick]], axis=1).astype(np.float64)


def topo_precision_recall_f1(pred_pts: np.ndarray, gt_pts: np.ndarray, radius: float) -> tuple[float, float, float]:
    if pred_pts.shape[0] == 0 or gt_pts.shape[0] == 0:
        return float("nan"), float("nan"), float("nan")
    d_p, _ = cKDTree(gt_pts).query(pred_pts, k=1)
    prec = float(np.mean(d_p <= radius))
    d_g, _ = cKDTree(pred_pts).query(gt_pts, k=1)
    rec = float(np.mean(d_g <= radius))
    if prec + rec < 1e-12:
        f1 = 0.0
    else:
        f1 = float(2.0 * prec * rec / (prec + rec))
    return prec, rec, f1


def _list_region_ids(eval_dir: Path) -> list[int]:
    ids: list[int] = []
    for p in sorted(eval_dir.glob("region_*_skeleton_final.png")):
        m = re.search(r"region_(\d+)_skeleton_final\.png", p.name)
        if m:
            ids.append(int(m.group(1)))
    return ids


def _ensure_gt_skeleton(
    eval_dir: Path,
    rid: int,
    gt_root: Path,
    hw: Tuple[int, int],
) -> Optional[np.ndarray]:
    p_cached = eval_dir / f"region_{rid}_gt_skeleton.png"
    if p_cached.is_file():
        g = cv2.imread(str(p_cached), cv2.IMREAD_GRAYSCALE)
        if g is not None and g.shape[:2] == hw:
            return g
    gt_png = _find_gt_png(gt_root, rid)
    if gt_png is None:
        return None
    g = _gt_skeleton_from_png(gt_png, hw)
    cv2.imwrite(str(p_cached), g)
    return g


def main() -> None:
    args = parse_args()
    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_absolute():
        eval_dir = (_ROOT / eval_dir).resolve()
    gt_root = Path(args.gt_root)
    rng = np.random.default_rng(int(args.seed))
    r = float(args.radius)
    n_s = int(args.n_samples)

    ids = _list_region_ids(eval_dir)
    if not ids:
        print(f"在 {eval_dir} 未找到 region_*_skeleton_final.png", file=sys.stderr)
        sys.exit(1)

    rows: list[dict[str, Any]] = []
    hdr = f"{'region':>8} | {'P_init':>7} {'R_init':>7} {'F1_init':>7} | {'P_ref':>7} {'R_ref':>7} {'F1_ref':>7} | {'dF1':>7}"
    print(hdr)
    print("-" * len(hdr))
    for rid in ids:
        p_fin = eval_dir / f"region_{rid}_skeleton_final.png"
        p_ini = eval_dir / f"region_{rid}_skeleton_initial.png"
        sk_ref = cv2.imread(str(p_fin), cv2.IMREAD_GRAYSCALE)
        if sk_ref is None:
            continue
        h, w = sk_ref.shape[:2]
        if p_ini.is_file():
            sk_ini = cv2.imread(str(p_ini), cv2.IMREAD_GRAYSCALE)
            if sk_ini is None or sk_ini.shape != (h, w):
                sk_ini = sk_ref.copy()
        else:
            sk_ini = sk_ref.copy()

        gt_sk = _ensure_gt_skeleton(eval_dir, rid, gt_root, (h, w))
        if gt_sk is None:
            print(f"[skip] region={rid} 无 GT", file=sys.stderr)
            continue

        gt_pts = sample_points_from_skeleton(gt_sk, n_s, rng)
        ini_pts = sample_points_from_skeleton(sk_ini, n_s, rng)
        ref_pts = sample_points_from_skeleton(sk_ref, n_s, rng)

        pi, ri, fi = topo_precision_recall_f1(ini_pts, gt_pts, r)
        pr, rr, fr = topo_precision_recall_f1(ref_pts, gt_pts, r)
        df = (fr - fi) if not (np.isnan(fr) or np.isnan(fi)) else float("nan")
        rows.append(
            {
                "region_id": rid,
                "radius_px": r,
                "initial_topo_precision": pi,
                "initial_topo_recall": ri,
                "initial_topo_f1": fi,
                "refined_topo_precision": pr,
                "refined_topo_recall": rr,
                "refined_topo_f1": fr,
                "delta_topo_f1": df,
            }
        )
        print(f"{rid:8d} | {pi:7.3f} {ri:7.3f} {fi:7.3f} | {pr:7.3f} {rr:7.3f} {fr:7.3f} | {df:7.3f}")

    out_json = eval_dir / "topo_metrics_report.json"
    out_json.write_text(json.dumps({"radius": r, "n_samples": n_s, "regions": rows}, indent=2), encoding="utf-8")
    print("-" * len(hdr))
    print(f"已写入: {out_json.resolve()}")


if __name__ == "__main__":
    main()
