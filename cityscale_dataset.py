"""
City-Scale 数据：2048×2048 按 GLD-Road 论文裁成 512×512 块训练。
标签来自 processed/road_mask_*.png 与 keypoint_mask_*.png；方向场由 utils.orientation_from_mask 生成。

**卫星图（官方布局）**：默认在 `20cities/region_{idx}_sat.png`（如 `region_0_sat.png`）。
若不存在，再尝试 `images/` 下的 `region_{idx}.*`；仍无则用道路掩膜复制为三通道占位。
"""

from __future__ import annotations

import glob
import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from utils.orientation_from_mask import orientation_from_road_mask


def _list_rgb_path_under_images_dir(images_dir: str, region_index: int) -> Optional[str]:
    """可选目录 `images/region_{idx}.*`，作为无 _sat 影像时的回退。"""
    if not images_dir or not os.path.isdir(images_dir):
        return None
    stem = f"region_{region_index}"
    for ext in (".png", ".jpg", ".tif", ".tiff", ".jp2"):
        p = os.path.join(images_dir, stem + ext)
        if os.path.isfile(p):
            return p
    cands = glob.glob(os.path.join(images_dir, f"{stem}.*"))
    return cands[0] if cands else None


def resolve_cityscale_rgb_path(
    cityscale_root: str,
    region_index: int,
    *,
    sat_subdir: str = "20cities",
    images_subdir: str = "images",
) -> Optional[str]:
    """
    City-Scale 卫星 RGB 路径解析。
    优先：`{root}/{sat_subdir}/region_{idx}_sat.png`（及同 stem 的 .jpg/.tif 等）；
    否则：`{root}/{images_subdir}/region_{idx}.*`。
    """
    stem_sat = f"region_{int(region_index)}_sat"
    sat_dir = os.path.join(cityscale_root, sat_subdir)
    if os.path.isdir(sat_dir):
        for ext in (".png", ".jpg", ".tif", ".tiff", ".jp2"):
            p = os.path.join(sat_dir, stem_sat + ext)
            if os.path.isfile(p):
                return p
    images_dir = os.path.join(cityscale_root, images_subdir)
    return _list_rgb_path_under_images_dir(images_dir, region_index)


# 兼容旧调用：仅查 images/
def _list_rgb_path(images_dir: str, region_index: int) -> Optional[str]:
    return _list_rgb_path_under_images_dir(images_dir, region_index)


class CityScalePatchDataset(Dataset):
    """
    每个样本 = 某 tile 的某 512×512 切块；sample_id 为 "{tile}_{row}_{col}"，row,col in {0,1,2,3}。
    可通过 patch_ids 显式指定列表；否则由 region_indices 枚举全部块。
    """

    def __init__(
        self,
        cityscale_root: str,
        region_indices: Optional[List[int]] = None,
        patch_ids: Optional[List[str]] = None,
        patch_size: int = 512,
        full_size: int = 2048,
        images_subdir: str = "images",
        sat_subdir: str = "20cities",
        processed_subdir: str = "processed",
        encoder_input_size: Optional[Tuple[int, int]] = None,
    ):
        self.root = cityscale_root
        self.patch = int(patch_size)
        self.full = int(full_size)
        self.images_subdir = images_subdir
        self.sat_subdir = sat_subdir
        self.images_dir = os.path.join(cityscale_root, images_subdir)
        self.proc_dir = os.path.join(cityscale_root, processed_subdir)
        self.encoder_input_size = encoder_input_size

        assert self.full % self.patch == 0
        self.grid = self.full // self.patch

        self.index: List[Tuple[int, int, int]] = []
        if patch_ids is not None:
            for pid in patch_ids:
                self.index.append(self.parse_patch_id(pid))
        else:
            regions = region_indices if region_indices is not None else []
            for t in regions:
                for gi in range(self.grid):
                    for gj in range(self.grid):
                        self.index.append((t, gi, gj))

    def __len__(self):
        return len(self.index)

    @staticmethod
    def parse_patch_id(patch_id: str) -> Tuple[int, int, int]:
        parts = patch_id.split("_")
        return int(parts[0]), int(parts[1]), int(parts[2])

    def __getitem__(self, idx: int):
        tile, gi, gj = self.index[idx]
        y0, x0 = gi * self.patch, gj * self.patch
        y1, x1 = y0 + self.patch, x0 + self.patch

        road_path = os.path.join(self.proc_dir, f"road_mask_{tile}.png")
        kp_path = os.path.join(self.proc_dir, f"keypoint_mask_{tile}.png")
        road = cv2.imread(road_path, cv2.IMREAD_GRAYSCALE)
        kp = cv2.imread(kp_path, cv2.IMREAD_GRAYSCALE)
        if road is None or kp is None:
            raise FileNotFoundError(f"缺少 processed 掩膜: {road_path} / {kp_path}")

        rgb_path = resolve_cityscale_rgb_path(
            self.root, tile, sat_subdir=self.sat_subdir, images_subdir=self.images_subdir
        )
        if rgb_path:
            image_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"无法读取 RGB: {rgb_path}")
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            # 占位：三通道均为归一化道路掩膜（用户应提供真实影像）
            r = (road > 0).astype(np.float32)
            image = np.stack([r, r, r], axis=-1)

        image = image[y0:y1, x0:x1]
        road = road[y0:y1, x0:x1]
        kp = kp[y0:y1, x0:x1]

        mask = (road.astype(np.float32) / 255.0)[None, :, :]
        inter = (kp.astype(np.float32) / 255.0)[None, :, :]

        orient_np = orientation_from_road_mask(road)
        orientation = np.stack(
            [orient_np[:, :, 0], orient_np[:, :, 1], orient_np[:, :, 2]], axis=0
        ).astype(np.float32)

        if self.encoder_input_size is not None:
            eh, ew = int(self.encoder_input_size[0]), int(self.encoder_input_size[1])
            image = cv2.resize(image, (ew, eh), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask[0], (ew, eh), interpolation=cv2.INTER_NEAREST)[None, :, :]
            inter = cv2.resize(inter[0], (ew, eh), interpolation=cv2.INTER_LINEAR)[None, :, :]
            o0 = cv2.resize(orientation[0], (ew, eh), interpolation=cv2.INTER_LINEAR)
            o1 = cv2.resize(orientation[1], (ew, eh), interpolation=cv2.INTER_LINEAR)
            o2 = cv2.resize(orientation[2], (ew, eh), interpolation=cv2.INTER_LINEAR)
            orientation = np.stack([o0, o1, o2], axis=0).astype(np.float32)

        image_t = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask_t = torch.from_numpy(mask).float()
        inter_t = torch.from_numpy(inter).float()
        orient_t = torch.from_numpy(orientation).float()
        patch_id = f"{tile}_{gi}_{gj}"
        return {
            "image": image_t,
            "mask": mask_t,
            "intersection": inter_t,
            "orientation": orient_t,
            "patch_id": patch_id,
            "tile_index": tile,
        }


def load_cityscale_split(split_json_path: str) -> dict:
    """
    读取官方 `data_split.json`。
    约定键名：**train**、**valid**、**test**（官方验证集字段为 `valid`，不是 `val`）。
    """
    with open(split_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "valid" not in data and "val" in data:
        data["valid"] = data["val"]
    return data
