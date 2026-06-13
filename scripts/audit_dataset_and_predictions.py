"""
数据与预测分布审计（输入是否正常、GT mask 是否合理、分割概率是否塌缩到 0）。

检查项：
  1) 输入 RGB：是否接近全黑/全白（按通道 min/max/mean 判定）
  2) GT mask：道路像素占比；是否全 0 或全 1
  3) 若提供 --checkpoint：sigmoid(分割) 的全局分布；是否「不敢预测」（大量像素接近 0）

输出：CSV + 若干 PNG 到 log_dir/audit_qc/

用法:
  python scripts/audit_dataset_and_predictions.py --config configs/config_shanghai.yaml
  python scripts/audit_dataset_and_predictions.py --config configs/config_shanghai.yaml \\
      --checkpoint checkpoints_shanghai/model_best_val_iou.pth --num_samples 168
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset_factory import build_datasets, get_all_sample_ids
from models.road_extraction_model import RoadExtractionModel
from scripts.train import split_ids

plt.rcParams["font.family"] = "DejaVu Sans"


def classify_image(img_chw: np.ndarray) -> tuple[str, dict]:
    """img [3,H,W] float 0~1"""
    x = np.clip(img_chw, 0, 1)
    ch_mean = x.mean(axis=(1, 2))
    ch_min = x.min(axis=(1, 2))
    ch_max = x.max(axis=(1, 2))
    gmean = float(x.mean())
    gstd = float(x.std())
    tag = "ok"
    if float(ch_max.max()) < 0.08:
        tag = "near_black"
    elif float(ch_min.min()) > 0.92:
        tag = "near_white"
    return tag, {
        "img_mean_r": ch_mean[0],
        "img_mean_g": ch_mean[1],
        "img_mean_b": ch_mean[2],
        "img_min": float(ch_min.min()),
        "img_max": float(ch_max.max()),
        "img_global_mean": gmean,
        "img_global_std": gstd,
        "img_tag": tag,
    }


def classify_mask(mask_hw: np.ndarray) -> tuple[str, dict]:
    """mask [H,W] float 0~1"""
    m = np.clip(mask_hw.astype(np.float64), 0, 1)
    road_ratio = float(m.mean())
    tag = "ok"
    if road_ratio < 1e-6:
        tag = "mask_all_zero"
    elif road_ratio > 1.0 - 1e-6:
        tag = "mask_all_one"
    elif road_ratio < 0.001:
        tag = "mask_very_sparse"
    elif road_ratio > 0.95:
        tag = "mask_very_dense"
    return tag, {"mask_road_ratio": road_ratio, "mask_tag": tag}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/config_shanghai.yaml")
    ap.add_argument("--checkpoint", type=str, default=None, help="提供则统计分割预测分布")
    ap.add_argument("--split", choices=["train", "val", "all"], default="val")
    ap.add_argument("--num_samples", type=int, default=0, help="0=该划分全部样本")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_dir", type=str, default=None)
    args = ap.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    log_dir = cfg["training"].get("log_dir", "logs")
    out_dir = args.out_dir or os.path.join(PROJECT_ROOT, log_dir, "audit_qc")
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(PROJECT_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    image_size = tuple(cfg["data"].get("image_size", [224, 224]))
    if cfg["model"]["encoder"] == "swin_tiny":
        image_size = (224, 224)

    all_ids = get_all_sample_ids(cfg)
    ms = cfg["data"].get("max_samples", 0)
    if ms and ms > 0:
        all_ids = all_ids[:ms]
    train_ids, val_ids = split_ids(
        all_ids,
        val_ratio=cfg["data"].get("val_ratio", 0.2),
        seed=cfg["data"].get("split_seed", 42),
    )
    if args.split == "train":
        pool_ids = train_ids
    elif args.split == "val":
        pool_ids = val_ids
    else:
        pool_ids = all_ids

    train_ds, val_ds = build_datasets(cfg, train_ids, val_ids, image_size)
    if args.split == "train":
        ds = train_ds
    elif args.split == "val":
        ds = val_ds
    else:
        from data.shanghai_filtered_dataset import ShanghaiFilteredDataset

        if cfg["data"].get("layout") == "shanghai_flat":
            root = cfg["data"]["root"]
            d = cfg["data"]

            def sub(name):
                return os.path.join(root, d[name])

            ds = ShanghaiFilteredDataset(
                images_dir=sub("images_subdir"),
                masks_dir=sub("masks_subdir"),
                heatmap_dir=sub("heatmap_subdir"),
                orientations_dir=sub("orientations_subdir"),
                sample_ids=all_ids,
                image_size=image_size,
                orientation_ext=d.get("orientation_ext", ".npy"),
            )
        else:
            from data.spacenet_dataset import SpaceNetRoadDataset

            ds = SpaceNetRoadDataset(
                aoi_dir=cfg["data"]["aoi_dir"],
                labels_dir=cfg["data"]["labels_dir"],
                sample_ids=all_ids,
                image_size=image_size,
            )

    n_total = len(ds)
    rng = np.random.default_rng(args.seed)
    if args.num_samples and args.num_samples > 0:
        idxs = rng.choice(n_total, size=min(args.num_samples, n_total), replace=False)
    else:
        idxs = np.arange(n_total)

    rows = []
    all_seg_probs: list[np.ndarray] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    if args.checkpoint:
        ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(PROJECT_ROOT, args.checkpoint)
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(ckpt)
        model = RoadExtractionModel(
            encoder=cfg["model"]["encoder"],
            num_classes=1,
            input_size=image_size,
        )
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.to(device)
        model.eval()

    loader = DataLoader(
        Subset(ds, idxs.tolist()),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
    )

    with torch.no_grad():
        for batch in loader:
            img = batch["image"]
            mask = batch["mask"]
            ids = batch["sample_id"]
            b = img.shape[0]
            if model is not None:
                out = model(img.to(device))
                seg_prob = torch.sigmoid(out["segmentation"][:, 0]).cpu().numpy()
            else:
                seg_prob = None

            for i in range(b):
                sid = ids[i]
                im = img[i].numpy()
                mk = mask[i, 0].numpy()
                itag, istats = classify_image(im)
                mtag, mstats = classify_mask(mk)
                row = {"sample_id": sid, **istats, **mstats}
                if seg_prob is not None:
                    sp = seg_prob[i].ravel()
                    all_seg_probs.append(sp)
                    row["pred_mean"] = float(sp.mean())
                    row["pred_std"] = float(sp.std())
                    row["pred_frac_lt_0.05"] = float((sp < 0.05).mean())
                    row["pred_frac_lt_0.1"] = float((sp < 0.1).mean())
                    row["pred_frac_gt_0.5"] = float((sp > 0.5).mean())
                rows.append(row)

    csv_path = os.path.join(out_dir, "audit_per_sample.csv")
    fieldnames = list(rows[0].keys()) if rows else []
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    # 汇总图
    road_ratios = [r["mask_road_ratio"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist(road_ratios, bins=40, color="steelblue", edgecolor="white")
    axes[0].set_title("GT mask road pixel ratio")
    axes[0].set_xlabel("mean(mask)")
    axes[0].set_ylabel("count")

    img_means = [r["img_global_mean"] for r in rows]
    axes[1].hist(img_means, bins=40, color="seagreen", edgecolor="white")
    axes[1].set_title("Input image global mean (RGB)")
    axes[1].set_xlabel("mean pixel")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "fig_mask_and_image_stats.png"), dpi=180)
    plt.close(fig)

    if all_seg_probs:
        flat = np.concatenate(all_seg_probs)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(flat, bins=80, range=(0, 1), color="coral", edgecolor="white", alpha=0.9)
        ax.axvline(0.5, color="k", ls="--", lw=1)
        ax.set_title("Seg sigmoid distribution (all audited pixels pooled)")
        ax.set_xlabel("P(road)")
        ax.set_ylabel("count")
        frac_low = float((flat < 0.05).mean())
        frac_mid = float(((flat >= 0.05) & (flat < 0.5)).mean())
        frac_hi = float((flat >= 0.5).mean())
        ax.text(
            0.02,
            0.98,
            f"mean={flat.mean():.4f} std={flat.std():.4f}\n"
            f"P<0.05: {100*frac_low:.1f}%  |  0.05~0.5: {100*frac_mid:.1f}%  |  >=0.5: {100*frac_hi:.1f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=10,
            family="monospace",
        )
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "fig_seg_probability_hist.png"), dpi=180)
        plt.close(fig)

    # 文本摘要
    n_img_bad = sum(1 for r in rows if r.get("img_tag") != "ok")
    n_mask_bad = sum(1 for r in rows if r.get("mask_tag") not in ("ok", "mask_very_sparse", "mask_very_dense"))
    lines = [
        "# Audit summary",
        "",
        f"- split: {args.split}, samples: {len(rows)} / ds_size={n_total}",
        f"- input abnormal (near_black/near_white): {n_img_bad}",
        f"- mask abnormal (all_zero/all_one): {sum(1 for r in rows if r.get('mask_tag') in ('mask_all_zero', 'mask_all_one'))}",
        f"- mask very sparse (<0.1% road): {sum(1 for r in rows if r.get('mask_tag') == 'mask_very_sparse')}",
        "",
    ]
    if all_seg_probs:
        flat = np.concatenate(all_seg_probs)
        lines.extend(
            [
                "## Segmentation prediction (sigmoid)",
                "",
                f"- global mean={flat.mean():.6f}, std={flat.std():.6f}",
                f"- fraction P<0.05: {100*(flat<0.05).mean():.2f}%  (high => model timid / collapsed low)",
                f"- fraction P>=0.5: {100*(flat>=0.5).mean():.2f}%",
                "",
            ]
        )
    lines.append(f"CSV: {csv_path}")

    summary_path = os.path.join(out_dir, "AUDIT_SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    print(f"\n[OK] Figures and CSV under: {out_dir}")


if __name__ == "__main__":
    main()
