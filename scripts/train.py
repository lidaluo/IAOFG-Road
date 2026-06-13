from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from collections import defaultdict

# Windows 下常见的 OpenMP 运行时冲突兜底（libomp.dll vs libiomp5md.dll）
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import yaml
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# 允许直接使用 `python scripts/train.py` 运行
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.cityscale_dataset import load_cityscale_split
from data.dataset_factory import build_datasets, get_all_sample_ids
from losses.multi_task_loss import MultiTaskLoss
from models.road_extraction_model import RoadExtractionModel


def split_ids(sample_ids, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.array(sample_ids)
    rng.shuffle(ids)
    split_idx = int(len(ids) * (1 - val_ratio))
    train_ids = ids[:split_idx].tolist()
    val_ids = ids[split_idx:].tolist()
    return train_ids, val_ids


def make_loss_fn_for_epoch(config, epoch_idx):
    """epoch_idx 从 0 开始。warmup 内以分割为主；inter/orient 若全为 0 则两路头几乎无梯度，建议用 warmup_aux_*。"""
    tr = config["training"]
    warmup = int(tr.get("seg_warmup_epochs", 0))
    lambda_seg = float(tr.get("lambda_seg", 1.0))
    lambda_inter = float(tr.get("lambda_inter", 1.0))
    lambda_orient = float(tr.get("lambda_orient", 1.0))
    aux_i = float(tr.get("warmup_aux_lambda_inter", 0.0))
    aux_o = float(tr.get("warmup_aux_lambda_orient", 0.0))
    ramp = int(tr.get("multitask_ramp_epochs", 0))

    if warmup > 0 and epoch_idx < warmup:
        eff_inter = aux_i
        eff_orient = aux_o
        phase = "warmup_seg_focus" if (aux_i > 0 or aux_o > 0) else "warmup_seg_only"
    elif ramp > 0 and warmup > 0 and epoch_idx < warmup + ramp:
        t = (epoch_idx - warmup + 1) / float(ramp)
        t = min(max(t, 0.0), 1.0)
        eff_inter = aux_i + t * (lambda_inter - aux_i)
        eff_orient = aux_o + t * (lambda_orient - aux_o)
        phase = "multitask_ramp"
    else:
        eff_inter = lambda_inter
        eff_orient = lambda_orient
        phase = "full_multitask"

    mo = config.get("model", {})
    loss_fn = MultiTaskLoss(
        lambda_seg=lambda_seg,
        lambda_inter=eff_inter,
        lambda_orient=eff_orient,
        lambda_anchor=float(tr.get("lambda_anchor", 0.0)),
        lambda_topo=float(tr.get("lambda_topo", 0.0)),
        seg_pos_weight=float(tr.get("seg_pos_weight", 1.0)),
        inter_pos_weight=float(tr.get("inter_pos_weight", 1.0)),
        use_dynamic_pos_weight=bool(tr.get("use_dynamic_pos_weight", True)),
        use_dynamic_seg_pos_weight=bool(tr.get("use_dynamic_seg_pos_weight", True)),
        use_dynamic_inter_pos_weight=bool(tr.get("use_dynamic_inter_pos_weight", False)),
        min_pos_weight=float(tr.get("min_pos_weight", 5.0)),
        max_pos_weight=float(tr.get("max_pos_weight", 400.0)),
        orient_num_bins=int(mo.get("orient_num_bins", 0)),
        orient_focal_gamma=float(tr.get("orient_focal_gamma", 2.0)),
        lambda_orient_smooth=float(tr.get("lambda_orient_smooth", 0.1)),
    )
    return loss_fn, eff_inter, eff_orient, phase


def compute_binary_metrics(seg_logits, seg_gt, threshold=0.5):
    pred = (torch.sigmoid(seg_logits) > threshold).float()
    gt = (seg_gt > 0.5).float()
    intersection = (pred * gt).sum(dim=(1, 2, 3))
    union = (pred + gt - pred * gt).sum(dim=(1, 2, 3))
    pred_sum = pred.sum(dim=(1, 2, 3))
    gt_sum = gt.sum(dim=(1, 2, 3))
    iou = ((intersection + 1e-6) / (union + 1e-6)).mean().item()
    f1 = ((2 * intersection + 1e-6) / (pred_sum + gt_sum + 1e-6)).mean().item()
    return iou, f1


def run_one_epoch(
    model,
    loader,
    loss_fn,
    device,
    optimizer=None,
    use_amp: bool = False,
    scaler: GradScaler | None = None,
    progress_every: int = 0,
    progress_prefix: str = "",
):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    amp_ok = bool(use_amp and device.type == "cuda" and scaler is not None)

    running = defaultdict(float)
    iou_list_t05 = []
    f1_list_t05 = []
    iou_list_t01 = []
    f1_list_t01 = []
    pred_prob_list = []
    n_batches = 0

    for batch in loader:
        image = batch["image"].to(device)
        targets = {
            "mask": batch["mask"].to(device),
            "intersection": batch["intersection"].to(device),
            "orientation": batch["orientation"].to(device),
        }

        with torch.set_grad_enabled(train_mode):
            if amp_ok and train_mode:
                with autocast():
                    outputs = model(image)
                    loss_dict = loss_fn(outputs, targets)
                    total_loss = loss_dict["total_loss"]
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(image)
                loss_dict = loss_fn(outputs, targets)
                total_loss = loss_dict["total_loss"]
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                    total_loss.backward()
                    optimizer.step()

        for key in (
            "total_loss",
            "segmentation_loss",
            "seg_bce",
            "seg_dice",
            "intersection_loss",
            "orientation_loss",
            "orientation_vec_loss",
            "orientation_conf_loss",
            "anchor_loss",
            "seg_pos_weight_used",
            "inter_pos_weight_used",
        ):
            if key in loss_dict and isinstance(loss_dict[key], torch.Tensor):
                running[key] += float(loss_dict[key].item())

        n_batches += 1
        iou05, f105 = compute_binary_metrics(outputs["segmentation"], targets["mask"], threshold=0.5)
        iou01, f101 = compute_binary_metrics(outputs["segmentation"], targets["mask"], threshold=0.3)
        iou_list_t05.append(iou05)
        f1_list_t05.append(f105)
        iou_list_t01.append(iou01)
        f1_list_t01.append(f101)
        pred_prob_list.append(float(torch.sigmoid(outputs["segmentation"]).mean().item()))

        if (
            train_mode
            and progress_every > 0
            and n_batches % progress_every == 0
        ):
            lt = float(loss_dict["total_loss"].item())
            print(
                f"{progress_prefix}  batch {n_batches}/{len(loader)}  loss={lt:.4f}",
                flush=True,
            )

    denom = max(n_batches, 1)
    avg_breakdown = {k: v / denom for k, v in running.items()}
    avg_loss = avg_breakdown.get("total_loss", 0.0)
    avg_iou_t05 = float(np.mean(iou_list_t05)) if iou_list_t05 else 0.0
    avg_f1_t05 = float(np.mean(f1_list_t05)) if f1_list_t05 else 0.0
    avg_iou_t01 = float(np.mean(iou_list_t01)) if iou_list_t01 else 0.0
    avg_f1_t01 = float(np.mean(f1_list_t01)) if f1_list_t01 else 0.0
    avg_pred_prob = float(np.mean(pred_prob_list)) if pred_prob_list else 0.0
    return avg_loss, avg_iou_t05, avg_f1_t05, avg_iou_t01, avg_f1_t01, avg_breakdown, avg_pred_prob


def _series(logs, key):
    return [entry[key] for entry in logs if entry.get(key) is not None]


def save_training_plots(logs, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    epochs = [entry["epoch"] for entry in logs]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, _series(logs, "train_loss"), label="train_loss")
    val_loss = _series(logs, "val_loss")
    if len(val_loss) > 0:
        val_epochs = [entry["epoch"] for entry in logs if entry.get("val_loss") is not None]
        plt.plot(val_epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training/Validation Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "loss_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, _series(logs, "train_iou"), label="train_iou")
    val_iou = _series(logs, "val_iou")
    if len(val_iou) > 0:
        val_epochs = [entry["epoch"] for entry in logs if entry.get("val_iou") is not None]
        plt.plot(val_epochs, val_iou, label="val_iou")
    plt.xlabel("Epoch")
    plt.ylabel("IoU")
    plt.title("IoU Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "iou_curve.png"), dpi=200)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, _series(logs, "train_f1"), label="train_f1")
    val_f1 = _series(logs, "val_f1")
    if len(val_f1) > 0:
        val_epochs = [entry["epoch"] for entry in logs if entry.get("val_f1") is not None]
        plt.plot(val_epochs, val_f1, label="val_f1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("F1 Curve")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(log_dir, "f1_curve.png"), dpi=200)
    plt.close()


def save_paper_caption_template(logs, log_dir):
    if not logs:
        return
    latest = logs[-1]
    best = max(logs, key=lambda x: (x.get("val_iou") if x.get("val_iou") is not None else -1.0))
    md_path = os.path.join(log_dir, "paper_caption_template.md")
    text = f"""# 论文图注模板（自动生成）

> 使用说明：将下述模板中的 `Fig. X` / `Table X` 替换为论文最终编号。
> 本文件由训练脚本自动更新，当前最新 epoch 为 {latest.get("epoch")}。

## Figure Caption Templates

### 1) Training Curves (`logs/loss_curve.png`, `logs/iou_curve.png`, `logs/f1_curve.png`)
**Fig. X.** Training dynamics of the proposed IAOF framework on the validation split.  
From epoch {latest.get("epoch")}, the model reaches `train_loss={latest.get("train_loss"):.4f}`, `train_iou={latest.get("train_iou"):.4f}`, `train_f1={latest.get("train_f1"):.4f}`.
The best validation IoU appears at epoch {best.get("epoch")} with `val_iou={best.get("val_iou") if best.get("val_iou") is not None else float('nan'):.4f}`.

### 2) Validation Visualizations (`logs/val_visuals/*.png`)
**Fig. X.** Qualitative visualization of predicted road segmentation, intersection heatmap, and orientation magnitude.  
The model captures major road structures and intersection-centric orientation patterns, while residual topological discontinuities remain in sparse/occluded regions.

## Table Caption Templates

### 1) Training Summary Table
**Table X.** Training summary of the proposed model.  
At epoch {latest.get("epoch")}, the model obtains `train_loss={latest.get("train_loss"):.4f}`, `train_iou={latest.get("train_iou"):.4f}`, `train_f1={latest.get("train_f1"):.4f}`, and
`val_iou={latest.get("val_iou") if latest.get("val_iou") is not None else float('nan'):.4f}`.

### 2) Ablation Table (模板)
**Table X.** Ablation study on intersection anchoring and topology-aware supervision.  
We compare variants: (i) segmentation-only baseline, (ii) + intersection branch, (iii) + orientation branch, (iv) full IAOF with anchor consistency loss.
The full model improves both pixel-level and topology-level metrics, validating the effectiveness of intersection-anchored direction modeling.

---

## 中文图注模板（可选）

**图X** 训练过程曲线图。  
在第 {latest.get("epoch")} 轮时，模型达到 `train_loss={latest.get("train_loss"):.4f}`、`train_iou={latest.get("train_iou"):.4f}`、`train_f1={latest.get("train_f1"):.4f}`；
验证集最优 IoU 出现在第 {best.get("epoch")} 轮（`val_iou={best.get("val_iou") if best.get("val_iou") is not None else float('nan'):.4f}`）。

**表X** 消融实验结果。  
完整 IAOF 模型在像素级与拓扑级指标上均优于分割基线与非锚定方向场变体，证明交叉口锚定设计的有效性。
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(text)


def save_val_visuals(model, val_loader, device, epoch, out_dir, max_samples=2):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    with torch.no_grad():
        batch = next(iter(val_loader), None)
        if batch is None:
            return
        image = batch["image"].to(device)
        outputs = model(image)
        n = min(max_samples, image.shape[0])
        for i in range(n):
            img_np = image[i].cpu().permute(1, 2, 0).numpy()
            seg_prob = torch.sigmoid(outputs["segmentation"][i, 0]).cpu().numpy()
            inter_prob = torch.sigmoid(outputs["intersection"][i, 0]).cpu().numpy()
            o_ch = outputs["orientation"].shape[1]
            if o_ch > 3:
                from utils.orient_bins import logits_to_expected_dxdy

                ex = logits_to_expected_dxdy(outputs["orientation"][i : i + 1])[0].cpu().numpy()
                orient_mag = np.sqrt(ex[0] ** 2 + ex[1] ** 2)
            else:
                orient = torch.tanh(outputs["orientation"][i, 0:2]).cpu().numpy()
                orient_mag = np.sqrt(orient[0] ** 2 + orient[1] ** 2)

            fig, axes = plt.subplots(1, 4, figsize=(16, 4))
            axes[0].imshow(np.clip(img_np, 0, 1))
            axes[0].set_title("RGB")
            axes[1].imshow(seg_prob, cmap="gray")
            axes[1].set_title("Seg Prob")
            axes[2].imshow(inter_prob, cmap="magma")
            axes[2].set_title("Intersection Prob")
            axes[3].imshow(orient_mag, cmap="viridis")
            axes[3].set_title("Orientation Magnitude")
            for ax in axes:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, f"epoch_{epoch:03d}_sample_{i}.png"), dpi=160)
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train IAOF road extraction model.")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to yaml config")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖配置中的 batch_size")
    parser.add_argument("--use_amp", action="store_true", help="启用 CUDA 自动混合精度（也可在 yaml training.use_amp: true）")
    parser.add_argument("--gpu_id", type=int, default=None, help="设置 CUDA_VISIBLE_DEVICES，例如 0")
    parser.add_argument(
        "--resume",
        nargs="?",
        const="latest",
        default=None,
        help="断点续训：可指定 .pth；仅写 --resume 则用配置里 checkpoint_dir/model_latest.pth",
    )
    args = parser.parse_args()

    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    os.chdir(PROJECT_ROOT)
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.batch_size is not None:
        config["training"]["batch_size"] = int(args.batch_size)

    checkpoint_dir = config["training"]["checkpoint_dir"]
    log_dir = config["training"]["log_dir"]
    log_file = os.path.join(log_dir, "training_log.json")

    num_workers = config["data"].get("num_workers", 4)
    image_size = tuple(config["data"].get("image_size", [512, 512]))
    encoder_name = config["model"]["encoder"]
    if encoder_name == "swin_tiny" and config["model"].get("force_swin_224", True):
        # 默认兼容 timm 预训练；City-Scale 可在配置中关闭并配合 encoder_input_size
        if image_size != (224, 224):
            print(f"[Info] Override image_size from {image_size} to (224, 224) for pretrained Swin-Tiny.")
            image_size = (224, 224)

    if (
        config["data"].get("layout") == "cityscale_patches"
        and config["data"].get("use_official_split", True)
    ):
        root = config["data"]["root"]
        split_path = config["data"].get("split_json") or os.path.join(root, "data_split.json")
        sp = load_cityscale_split(split_path)
        full_sz = int(config["data"].get("full_image_size", 2048))
        patch_sz = int(config["data"].get("patch_size", 512))
        grid_n = full_sz // patch_sz

        def _expand_regions(regions):
            out = []
            for t in regions:
                for gi in range(grid_n):
                    for gj in range(grid_n):
                        out.append(f"{t}_{gi}_{gj}")
            return out

        train_ids = _expand_regions(sp.get("train", []))
        val_ids = _expand_regions(sp.get("valid", []))
        max_samples = config["data"].get("max_samples", 0)
        if max_samples and max_samples > 0:
            train_ids = train_ids[:max_samples]
    else:
        all_ids = get_all_sample_ids(config)
        max_samples = config["data"].get("max_samples", 0)
        if max_samples and max_samples > 0:
            all_ids = all_ids[:max_samples]

        train_ids, val_ids = split_ids(
            all_ids,
            val_ratio=config["data"].get("val_ratio", 0.2),
            seed=config["data"].get("split_seed", 42),
        )
    print(f"[Data] layout={config['data'].get('layout', 'spacenet_aoi')} train N={len(train_ids)}, val N={len(val_ids)}")
    print(f"[Data] train_ids (first 10): {train_ids[:10]}")
    print(f"[Data] val_ids (first 10): {val_ids[:10]}")

    train_dataset, val_dataset = build_datasets(config, train_ids, val_ids, image_size)

    batch_size = config["training"].get("batch_size", 2)
    _cuda_for_loader = torch.cuda.is_available()
    _dl_common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=_cuda_for_loader,
        persistent_workers=bool(num_workers > 0),
    )
    if num_workers > 0:
        _dl_common["prefetch_factor"] = int(config["data"].get("prefetch_factor", 2))
    train_loader = DataLoader(train_dataset, shuffle=True, **_dl_common)
    val_loader = DataLoader(val_dataset, shuffle=False, **_dl_common)

    swin_img = config["model"].get("swin_img_size")
    if swin_img is not None:
        swin_img = int(swin_img)
    orient_bins = int(config.get("model", {}).get("orient_num_bins", 0))
    model = RoadExtractionModel(
        encoder=encoder_name,
        num_classes=1,
        input_size=image_size,
        swin_img_size=swin_img,
        orient_num_bins=orient_bins,
    )
    cuda_ok = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_ok else "cpu")
    print(
        f"[Device] torch.cuda.is_available()={cuda_ok}  -> 使用 {device}"
        + (f"  ({torch.cuda.get_device_name(0)})" if cuda_ok else ""),
        flush=True,
    )
    if not cuda_ok:
        print(
            "[Device][WARN] 当前为 CPU 训练，City-Scale + Swin 会非常慢（数秒/batch 属正常）。"
            "请检查本环境 PyTorch 是否为 CUDA 版：pip show torch 看版本，或运行 "
            "`python -c \"import torch; print(torch.__version__, torch.version.cuda)\"`。"
            "CPU-only 的 wheel 需按 https://pytorch.org 重装带 cu124/cu121 的包。",
            flush=True,
        )
    model.to(device)
    if cuda_ok:
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.empty_cache()
        print(
            f"[Device] 模型已载入 GPU，显存约 {torch.cuda.memory_allocated(0) / 1024**2:.0f} MB（随 batch 会升）",
            flush=True,
        )
        print(
            "[Device] cudnn.benchmark=True, matmul TF32=high（固定 224 输入下通常更快）",
            flush=True,
        )
    use_amp = bool(args.use_amp or config["training"].get("use_amp", False))
    scaler = GradScaler() if (use_amp and device.type == "cuda") else None
    if scaler is not None:
        print("[AMP] 已启用 CUDA autocast + GradScaler")
    elif use_amp and not cuda_ok:
        print("[AMP][WARN] 已请求 --use_amp 但无 CUDA，已退回 FP32 CPU。", flush=True)

    start_epoch = 0
    logs = []
    best_val_iou = -1.0
    resume_lr = None
    if args.resume is not None:
        if args.resume == "latest":
            ckpt_path = os.path.join(checkpoint_dir, "model_latest.pth")
        else:
            ckpt_path = args.resume if os.path.isabs(args.resume) else os.path.join(PROJECT_ROOT, args.resume)
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"--resume 权重不存在: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        if os.path.isfile(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                logs = json.load(f)
            if not isinstance(logs, list) or len(logs) == 0:
                raise ValueError(f"日志文件无效或为空: {log_file}")
            start_epoch = len(logs)
            for row in logs:
                vi = row.get("val_iou")
                if vi is not None and vi > best_val_iou:
                    best_val_iou = vi
            resume_lr = float(logs[-1]["learning_rate"])
            print(
                f"[Resume] 已加载 {ckpt_path}，从 Epoch {start_epoch + 1}/{config['training']['epochs']} 继续 "
                f"(已完成 {start_epoch} 个 epoch，best_val_iou={best_val_iou:.4f}，lr={resume_lr})"
            )
        else:
            raise FileNotFoundError(f"续训需要已有训练日志: {log_file}")

    warmup_epochs = int(config["training"].get("seg_warmup_epochs", 0))
    ramp_epochs = int(config["training"].get("multitask_ramp_epochs", 0))
    aux_i = float(config["training"].get("warmup_aux_lambda_inter", 0.0))
    aux_o = float(config["training"].get("warmup_aux_lambda_orient", 0.0))
    if warmup_epochs > 0:
        print(
            f"[Warmup] 前 {warmup_epochs} 个 epoch：以分割为主，inter/orient 使用辅助权重 "
            f"({aux_i}, {aux_o})；若为 0 则两路头几乎无梯度。"
        )
        if ramp_epochs > 0:
            print(
                f"[Ramp] 随后 {ramp_epochs} 个 epoch：inter/orient 从辅助权重线性升到 "
                f"lambda_inter/orient。"
            )
    print(
        f"[Loss] lambda_seg={config['training'].get('lambda_seg', 1.0)}, "
        f"lambda_inter={config['training'].get('lambda_inter', 1.0)}, "
        f"lambda_orient={config['training'].get('lambda_orient', 1.0)}, "
        f"lambda_anchor={config['training'].get('lambda_anchor', 0.0)}, "
        f"seg_warmup_epochs={warmup_epochs}, multitask_ramp_epochs={ramp_epochs}, "
        f"warmup_aux=({aux_i},{aux_o}), "
        f"seg_pos_weight={config['training'].get('seg_pos_weight', 1.0)}, "
        f"inter_pos_weight={config['training'].get('inter_pos_weight', 1.0)}"
    )
    init_lr = resume_lr if resume_lr is not None else float(config["training"]["learning_rate"])
    optimizer = optim.Adam(
        model.parameters(),
        lr=init_lr,
        weight_decay=config["training"]["weight_decay"],
    )
    _sched_kw = dict(
        mode="min",
        factor=float(config["training"].get("lr_scheduler_factor", 0.5)),
        patience=int(config["training"].get("lr_scheduler_patience", 3)),
        min_lr=float(config["training"].get("min_lr", 1e-6)),
    )
    if "verbose" in inspect.signature(ReduceLROnPlateau.__init__).parameters:
        _sched_kw["verbose"] = True
    scheduler = ReduceLROnPlateau(optimizer, **_sched_kw)

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    epochs = config["training"]["epochs"]
    save_every = config["training"].get("save_every", 10)
    validate_every = config["training"].get("validate_every", 1)

    if start_epoch >= epochs:
        print(f"[Resume] 已完成全部 {epochs} 个 epoch，无需继续训练。")
        return

    prog_every = int(config["training"].get("progress_print_batches", 0))
    if prog_every <= 0 and len(train_loader) > 400:
        prog_every = 100
    if prog_every > 0:
        print(
            f"[Info] 每个 train epoch 共 {len(train_loader)} 个 batch；每 {prog_every} 个 batch 打印一次 loss。"
            f"（yaml: training.progress_print_batches，0 则仅小数据集自动关闭）",
            flush=True,
        )

    for epoch in range(start_epoch, epochs):
        loss_fn, eff_li, eff_lo, phase = make_loss_fn_for_epoch(config, epoch)
        if warmup_epochs > 0 and epoch == warmup_epochs:
            print(
                f"[Warmup] 阶段结束，本 epoch 起 eff_li={eff_li:.4f}, eff_lo={eff_lo:.4f} "
                f"（ramp 或全量多任务）"
            )

        train_loss, train_iou, train_f1, train_iou_t01, train_f1_t01, train_bd, train_pred_p = run_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            device=device,
            optimizer=optimizer,
            use_amp=use_amp,
            scaler=scaler,
            progress_every=prog_every,
            progress_prefix=f"Epoch {epoch + 1} train",
        )

        val_loss = None
        val_iou = None
        val_f1 = None
        val_iou_t01 = None
        val_f1_t01 = None
        val_bd = {}
        val_pred_p = None
        if len(val_dataset) > 0 and (epoch + 1) % validate_every == 0:
            val_loss_fn, _, _, _ = make_loss_fn_for_epoch(config, epoch)
            with torch.no_grad():
                val_loss, val_iou, val_f1, val_iou_t01, val_f1_t01, val_bd, val_pred_p = run_one_epoch(
                    model=model,
                    loader=val_loader,
                    loss_fn=val_loss_fn,
                    device=device,
                    optimizer=None,
                    use_amp=False,
                    scaler=None,
                )

        def fmt_bd(prefix, bd):
            if not bd:
                return ""
            return (
                f" | {prefix} seg={bd.get('segmentation_loss', 0):.4f} "
                f"(bce={bd.get('seg_bce', 0):.4f} dice={bd.get('seg_dice', 0):.4f}) "
                f"inter={bd.get('intersection_loss', 0):.4f} "
                f"orient={bd.get('orientation_loss', 0):.4f} "
                f"(vec={bd.get('orientation_vec_loss', 0):.4f} conf={bd.get('orientation_conf_loss', 0):.4f}) "
                f"anchor={bd.get('anchor_loss', 0):.6f} "
                f"pw(seg/inter)=({bd.get('seg_pos_weight_used', 0):.1f}/{bd.get('inter_pos_weight_used', 0):.1f})"
            )

        print(
            f"Epoch {epoch + 1}/{epochs} | phase={phase} eff_li={eff_li} eff_lo={eff_lo} | "
            f"train_loss={train_loss:.4f}, train_iou@0.5={train_iou:.4f}, train_f1@0.5={train_f1:.4f}, "
            f"train_iou@0.3={train_iou_t01:.4f}, train_f1@0.3={train_f1_t01:.4f}, "
            f"pred_p(mean sigmoid seg)={train_pred_p:.4f}"
            + fmt_bd("train", train_bd)
            + (
                f" | val_loss={val_loss:.4f}, val_iou@0.5={val_iou:.4f}, val_f1@0.5={val_f1:.4f}, "
                f"val_iou@0.3={val_iou_t01:.4f}, val_f1@0.3={val_f1_t01:.4f}, "
                f"val_pred_p={val_pred_p:.4f}"
                + fmt_bd("val", val_bd)
                if val_loss is not None
                else ""
            )
        )
        if val_loss is not None:
            scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)
        current_lr = float(optimizer.param_groups[0]["lr"])
        print(f"[LR] current_lr={current_lr:.8f}")

        if (
            config["training"].get("dynamic_loss_adjust", False)
            and val_bd
            and (epoch + 1) % int(config["training"].get("dynamic_loss_every", 5)) == 0
        ):
            s = float(val_bd.get("segmentation_loss", 1.0))
            o = float(val_bd.get("orientation_loss", 1.0)) + 1e-6
            if s / o > float(config["training"].get("dynamic_loss_ratio_thresh", 3.0)):
                lo = float(config["training"].get("lambda_orient", 0.35))
                cap = float(config["training"].get("lambda_orient_cap", 0.85))
                config["training"]["lambda_orient"] = min(lo * 1.08, cap)
                print(f"[DynamicLoss] val seg/orient 失衡，lambda_orient -> {config['training']['lambda_orient']:.4f}")

        log_entry = {
            "epoch": epoch + 1,
            "warmup_phase": phase,
            "lambda_inter_effective": eff_li,
            "lambda_orient_effective": eff_lo,
            "train_loss": train_loss,
            "train_iou": train_iou,
            "train_f1": train_f1,
            "train_iou_t01": train_iou_t01,
            "train_f1_t01": train_f1_t01,
            "train_iou_t03": train_iou_t01,
            "train_f1_t03": train_f1_t01,
            "train_pred_prob_mean": train_pred_p,
            "train_seg_loss": train_bd.get("segmentation_loss"),
            "train_seg_bce": train_bd.get("seg_bce"),
            "train_seg_dice": train_bd.get("seg_dice"),
            "train_inter_loss": train_bd.get("intersection_loss"),
            "train_orient_loss": train_bd.get("orientation_loss"),
            "train_orient_vec": train_bd.get("orientation_vec_loss"),
            "train_orient_conf": train_bd.get("orientation_conf_loss"),
            "train_anchor_loss": train_bd.get("anchor_loss"),
            "train_seg_pos_weight_used": train_bd.get("seg_pos_weight_used"),
            "train_inter_pos_weight_used": train_bd.get("inter_pos_weight_used"),
            "val_loss": val_loss,
            "val_iou": val_iou,
            "val_f1": val_f1,
            "val_iou_t01": val_iou_t01,
            "val_f1_t01": val_f1_t01,
            "val_iou_t03": val_iou_t01,
            "val_f1_t03": val_f1_t01,
            "val_pred_prob_mean": val_pred_p,
            "val_seg_loss": val_bd.get("segmentation_loss") if val_bd else None,
            "val_seg_bce": val_bd.get("seg_bce") if val_bd else None,
            "val_seg_dice": val_bd.get("seg_dice") if val_bd else None,
            "val_inter_loss": val_bd.get("intersection_loss") if val_bd else None,
            "val_orient_loss": val_bd.get("orientation_loss") if val_bd else None,
            "val_orient_vec": val_bd.get("orientation_vec_loss") if val_bd else None,
            "val_orient_conf": val_bd.get("orientation_conf_loss") if val_bd else None,
            "val_anchor_loss": val_bd.get("anchor_loss") if val_bd else None,
            "val_seg_pos_weight_used": val_bd.get("seg_pos_weight_used") if val_bd else None,
            "val_inter_pos_weight_used": val_bd.get("inter_pos_weight_used") if val_bd else None,
            "learning_rate": current_lr,
        }
        logs.append(log_entry)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(logs, f, indent=2, ensure_ascii=False)
        save_training_plots(logs, log_dir)
        save_paper_caption_template(logs, log_dir)
        if val_loss is not None and (epoch + 1) % config["training"].get("save_visual_every", 1) == 0:
            save_val_visuals(model, val_loader, device, epoch + 1, os.path.join(log_dir, "val_visuals"))

        if (epoch + 1) % save_every == 0:
            ckpt = os.path.join(checkpoint_dir, f"model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt)
        torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_latest.pth"))
        if val_iou is not None and val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_best_val_iou.pth"))


if __name__ == "__main__":
    main()
