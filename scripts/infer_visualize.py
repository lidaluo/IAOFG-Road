"""用 checkpoint 在图像上推理并保存可视化（分割 / 交叉口 / 方向场 + 可选 GT）。"""
import argparse
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.dataset_factory import build_datasets, get_all_sample_ids
from data.shanghai_filtered_dataset import ShanghaiFilteredDataset
from data.spacenet_dataset import SpaceNetRoadDataset
from models.road_extraction_model import RoadExtractionModel
from scripts.train import split_ids


def load_checkpoint(model, ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)


def _to_hw2(t):
    """[1,1,H,W] or [1,H,W] or [H,W] -> numpy H,W"""
    x = t[0, 0].cpu().numpy() if t.dim() == 4 else t.cpu().numpy()
    if x.ndim == 3:
        x = x[0]
    return x


def visualize_one(
    img_tensor,
    outputs,
    out_path,
    title_prefix="",
    gt_mask=None,
    gt_inter=None,
    gt_orientation=None,
):
    """img_tensor [1,3,H,W] on cpu；gt_* 任一为 None 则只画单行（无 GT 对比行）。"""
    img_np = img_tensor[0].cpu().permute(1, 2, 0).numpy()
    seg_prob = torch.sigmoid(outputs["segmentation"][0, 0]).cpu().numpy()
    inter_prob = torch.sigmoid(outputs["intersection"][0, 0]).cpu().numpy()
    orient = torch.tanh(outputs["orientation"][0, 0:2]).cpu().numpy()
    orient_mag = np.sqrt(orient[0] ** 2 + orient[1] ** 2)

    seg_bin = (seg_prob > 0.5).astype(np.float32)
    overlay_pred = np.clip(img_np.copy(), 0, 1)
    red = np.zeros_like(overlay_pred)
    red[:, :, 0] = seg_bin
    overlay_pred = np.clip(overlay_pred * 0.55 + red * 0.45, 0, 1)

    has_gt = gt_mask is not None and gt_inter is not None
    nrows = 2 if has_gt else 1
    ncols = 5
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.5 * nrows))
    if title_prefix:
        fig.suptitle(title_prefix, fontsize=11)

    if nrows == 1:
        axes = np.array([axes])

    def show(ax, arr, cmap=None, t=""):
        if cmap:
            ax.imshow(arr, cmap=cmap)
        else:
            ax.imshow(np.clip(arr, 0, 1))
        ax.set_title(t, fontsize=9)
        ax.axis("off")

    # 上行：预测
    show(axes[0, 0], img_np, None, "RGB")
    show(axes[0, 1], seg_prob, "gray", "Pred seg")
    show(axes[0, 2], inter_prob, "magma", "Pred inter")
    show(axes[0, 3], orient_mag, "viridis", "Pred |orient|")
    show(axes[0, 4], overlay_pred, None, "Pred overlay (seg>0.5)")

    if has_gt:
        gm = _to_hw2(gt_mask)
        gi = _to_hw2(gt_inter)
        overlay_gt = np.clip(img_np.copy(), 0, 1)
        green = np.zeros_like(overlay_gt)
        green[:, :, 1] = (gm > 0.5).astype(np.float32)
        overlay_gt = np.clip(overlay_gt * 0.55 + green * 0.45, 0, 1)

        gt_omag = np.zeros_like(gm, dtype=np.float32)
        if gt_orientation is not None:
            o = gt_orientation
            if o.dim() == 3:
                dx = o[0].cpu().numpy()
                dy = o[1].cpu().numpy()
                conf = o[2].cpu().numpy()
            else:
                o = o[0]
                dx = o[0].cpu().numpy()
                dy = o[1].cpu().numpy()
                conf = o[2].cpu().numpy()
            gt_omag = np.sqrt(dx * dx + dy * dy) * np.clip(conf, 0, 1)

        show(axes[1, 0], img_np, None, "RGB")
        show(axes[1, 1], gm, "gray", "GT mask")
        show(axes[1, 2], gi, "magma", "GT inter")
        show(axes[1, 3], gt_omag, "viridis", "GT |orient|×conf")
        show(axes[1, 4], overlay_gt, None, "GT overlay (mask>0.5)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_on_image_paths(paths, model, device, image_size, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    h, w = int(image_size[0]), int(image_size[1])
    model.eval()
    with torch.no_grad():
        for path in paths:
            bgr = cv2.imread(path, cv2.IMREAD_COLOR)
            if bgr is None:
                print(f"[Skip] 无法读取: {path}")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            t = torch.from_numpy(rgb.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
            out = model(t)
            stem = os.path.splitext(os.path.basename(path))[0]
            out_path = os.path.join(out_dir, f"{stem}_infer.png")
            visualize_one(t.cpu(), out, out_path, title_prefix=stem)
            print(f"[OK] {out_path}")


def main():
    p = argparse.ArgumentParser(description="推理并保存可视化 PNG")
    p.add_argument("--config", type=str, default="configs/config_shanghai.yaml")
    p.add_argument("--checkpoint", type=str, default=None, help="默认先试 best 再 latest")
    p.add_argument(
        "--split",
        type=str,
        choices=["val", "train", "all"],
        default="val",
        help="从划分里抽样（all=全数据）",
    )
    p.add_argument("--num_samples", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no_gt",
        action="store_true",
        help="不画 GT 对比行（默认：从数据集推理时画上下两行 Pred vs GT）",
    )
    p.add_argument("--output_dir", type=str, default=None, help="默认 log_dir/infer_vis")
    p.add_argument(
        "--image",
        type=str,
        nargs="*",
        default=None,
        help="指定一张或多张 RGB 图像路径（与 --split 二选一；仅推理无 GT）",
    )
    args = p.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    image_size = tuple(cfg["data"].get("image_size", [224, 224]))
    enc = cfg["model"]["encoder"]
    if enc == "swin_tiny" and image_size != (224, 224):
        image_size = (224, 224)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RoadExtractionModel(encoder=enc, num_classes=1, input_size=image_size)
    ckpt_dir = cfg["training"]["checkpoint_dir"]
    if args.checkpoint:
        ckpt = args.checkpoint if os.path.isabs(args.checkpoint) else os.path.join(PROJECT_ROOT, args.checkpoint)
    else:
        best = os.path.join(ckpt_dir, "model_best_val_iou.pth")
        latest = os.path.join(ckpt_dir, "model_latest.pth")
        ckpt = best if os.path.isfile(best) else latest
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"找不到权重: {ckpt}")
    load_checkpoint(model, ckpt, device)
    model.to(device)
    print(f"[Model] {ckpt}")

    log_dir = cfg["training"].get("log_dir", "logs")
    out_dir = args.output_dir or os.path.join(log_dir, "infer_vis")
    out_dir = out_dir if os.path.isabs(out_dir) else os.path.join(PROJECT_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.image:
        run_on_image_paths(args.image, model, device, image_size, out_dir)
        return

    all_ids = get_all_sample_ids(cfg)
    max_s = cfg["data"].get("max_samples", 0)
    if max_s and max_s > 0:
        all_ids = all_ids[:max_s]
    train_ids, val_ids = split_ids(
        all_ids,
        val_ratio=cfg["data"].get("val_ratio", 0.2),
        seed=cfg["data"].get("split_seed", 42),
    )
    if args.split == "val":
        pick_ids = val_ids
    elif args.split == "train":
        pick_ids = train_ids
    else:
        pick_ids = all_ids

    train_ds, val_ds = build_datasets(cfg, train_ids, val_ids, image_size)
    if args.split == "val":
        ds = val_ds
    elif args.split == "train":
        ds = train_ds
    else:
        data = cfg["data"]
        if data.get("layout") == "shanghai_flat":
            root = data["root"]

            def sub(name):
                return os.path.join(root, data[name])

            ds = ShanghaiFilteredDataset(
                images_dir=sub("images_subdir"),
                masks_dir=sub("masks_subdir"),
                heatmap_dir=sub("heatmap_subdir"),
                orientations_dir=sub("orientations_subdir"),
                sample_ids=all_ids,
                image_size=image_size,
                orientation_ext=data.get("orientation_ext", ".npy"),
            )
        else:
            ds = SpaceNetRoadDataset(
                aoi_dir=data["aoi_dir"],
                labels_dir=data["labels_dir"],
                sample_ids=all_ids,
                image_size=image_size,
            )
        pick_ids = all_ids

    id_to_idx = {ds.sample_ids[i]: i for i in range(len(ds))}
    rng = np.random.default_rng(args.seed)
    candidates = [sid for sid in pick_ids if sid in id_to_idx]
    if not candidates:
        raise RuntimeError("划分与数据集无交集，请检查 config")
    n = min(args.num_samples, len(candidates))
    chosen = list(rng.choice(candidates, size=n, replace=False))

    model.eval()
    with torch.no_grad():
        for sid in chosen:
            idx = id_to_idx[sid]
            batch = ds[idx]
            image = batch["image"].unsqueeze(0).to(device)
            out = model(image)
            out_path = os.path.join(out_dir, f"{sid}_infer.png")
            use_gt = not args.no_gt
            gt_m = batch["mask"].unsqueeze(0) if use_gt else None
            gt_i = batch["intersection"].unsqueeze(0) if use_gt else None
            gt_o = batch["orientation"] if use_gt else None
            visualize_one(
                image.cpu(),
                out,
                out_path,
                title_prefix=sid,
                gt_mask=gt_m,
                gt_inter=gt_i,
                gt_orientation=gt_o,
            )
            print(f"[OK] {out_path}")

    print(f"[Done] 输出目录: {out_dir}")


if __name__ == "__main__":
    main()
