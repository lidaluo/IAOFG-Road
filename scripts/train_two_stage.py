"""
IAOF-Graph 两阶段训练入口（Global = IAOF 主干，Local = LocalQueryDecoder）。
阶段一：调用 scripts/train.py 与 City-Scale 配置训练 RoadExtractionModel。
阶段二：冻结主干（不加载也可），仅优化局部解码器；样本来自 data/local_query_dataset.py。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from data.cityscale_dataset import load_cityscale_split
from data.local_query_dataset import LocalQueryCityScaleDataset
from models.local_query_decoder import LocalQueryDecoder


def train_local_stage(config: dict, device: torch.device) -> None:
    lcfg = config.get("local_query", {})
    root = config["data"]["root"]
    split_path = config["data"].get("split_json") or os.path.join(root, "data_split.json")
    sp = load_cityscale_split(split_path)
    train_regions = sp.get("train", [])

    d = config["data"]
    ds = LocalQueryCityScaleDataset(
        cityscale_root=root,
        region_indices=train_regions,
        patch_size=int(lcfg.get("patch_size", 128)),
        max_samples_per_tile=int(lcfg.get("max_samples_per_tile", 350)),
        seed=42,
        pred_mask_dir=lcfg.get("pred_mask_dir"),
        processed_subdir=d.get("processed_subdir", "processed"),
        images_subdir=d.get("images_subdir", "images"),
        sat_subdir=d.get("sat_images_subdir", "20cities"),
    )
    if len(ds) == 0:
        print("[Local] 无有效局部样本（检查 processed 掩膜与骨架）。")
        return

    loader = DataLoader(
        ds,
        batch_size=int(lcfg.get("batch_size", 16)),
        shuffle=True,
        num_workers=int(config["data"].get("num_workers", 0)),
    )

    net = LocalQueryDecoder(in_channels=4, base=32).to(device)
    opt = Adam(net.parameters(), lr=float(lcfg.get("learning_rate", 1e-3)))
    loss_fn = nn.SmoothL1Loss()
    epochs = int(lcfg.get("epochs", 25))

    out_path = lcfg.get("checkpoint_path", os.path.join(config["training"]["checkpoint_dir"], "local_query_decoder.pth"))
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    for ep in range(epochs):
        net.train()
        running = 0.0
        n = 0
        for batch in loader:
            x = batch["input"].to(device)
            y = batch["target"].to(device)
            opt.zero_grad()
            pred = net(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            running += float(loss.item())
            n += 1
        print(f"[Local] epoch {ep+1}/{epochs} loss={running / max(n,1):.5f}")

    torch.save(net.state_dict(), out_path)
    print(f"[Local] 已保存 {out_path}")


def main():
    parser = argparse.ArgumentParser(description="IAOF-Graph two-stage training.")
    parser.add_argument("--config", type=str, default="configs/config_cityscale_iaof_graph.yaml")
    parser.add_argument("--skip-global", action="store_true", help="跳过阶段一（已有主干权重）")
    parser.add_argument("--skip-local", action="store_true", help="仅运行阶段一")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    cfg_path = args.config if os.path.isabs(args.config) else os.path.join(PROJECT_ROOT, args.config)
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not args.skip_global:
        train_py = os.path.join(PROJECT_ROOT, "scripts", "train.py")
        cmd = [sys.executable, "-u", train_py, "--config", cfg_path]
        print("[Global] 运行:", " ".join(cmd))
        r = subprocess.call(cmd)
        if r != 0:
            sys.exit(r)

    if args.skip_local:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_local_stage(config, device)


if __name__ == "__main__":
    main()
