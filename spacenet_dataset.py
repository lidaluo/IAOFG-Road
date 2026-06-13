import os
from typing import List, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class SpaceNetRoadDataset(Dataset):
    def __init__(
        self,
        aoi_dir: str,
        labels_dir: str,
        sample_ids: Optional[List[str]] = None,
        image_size: Optional[tuple] = (512, 512),
    ):
        self.rgb_dir = os.path.join(aoi_dir, "PS-RGB")
        self.labels_dir = labels_dir
        self.image_size = image_size

        if sample_ids is None:
            sample_ids = sorted(
                [d for d in os.listdir(labels_dir) if os.path.isdir(os.path.join(labels_dir, d))]
            )
        self.sample_ids = sample_ids

    def __len__(self):
        return len(self.sample_ids)

    def _read_rgb(self, sample_id: str):
        img_num = sample_id.replace("img", "")
        tif_name = f"SN3_roads_train_AOI_3_Paris_PS-RGB_img{img_num}.tif"
        img_path = os.path.join(self.rgb_dir, tif_name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"RGB image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        return image

    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        sample_dir = os.path.join(self.labels_dir, sample_id)

        image = self._read_rgb(sample_id)

        mask = cv2.imread(os.path.join(sample_dir, "mask.png"), cv2.IMREAD_GRAYSCALE)
        inter = cv2.imread(os.path.join(sample_dir, "intersection.png"), cv2.IMREAD_GRAYSCALE)
        orient = cv2.imread(os.path.join(sample_dir, "orientation.png"), cv2.IMREAD_COLOR)
        if mask is None or inter is None or orient is None:
            raise FileNotFoundError(f"Missing label files under: {sample_dir}")

        if self.image_size is not None:
            w, h = int(self.image_size[1]), int(self.image_size[0])
            image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            inter = cv2.resize(inter, (w, h), interpolation=cv2.INTER_LINEAR)
            orient = cv2.resize(orient, (w, h), interpolation=cv2.INTER_LINEAR)

        mask = (mask.astype(np.float32) / 255.0)[None, :, :]
        inter = (inter.astype(np.float32) / 255.0)[None, :, :]

        orient = orient.astype(np.float32)
        dx = orient[:, :, 0] / 127.5 - 1.0
        dy = orient[:, :, 1] / 127.5 - 1.0
        conf = orient[:, :, 2] / 255.0

        # resize 后方向向量长度会被插值破坏，这里在道路区域重归一化
        road = conf > 0.1  # 降低置信度阈值，确保更多方向场被正确归一化
        norm = np.sqrt(dx * dx + dy * dy) + 1e-6
        dx = np.where(road, dx / norm, 0.0)
        dy = np.where(road, dy / norm, 0.0)

        orientation = np.stack([dx, dy, conf], axis=0).astype(np.float32)

        image = torch.from_numpy(np.transpose(image, (2, 0, 1))).float()
        mask = torch.from_numpy(mask).float()
        inter = torch.from_numpy(inter).float()
        orientation = torch.from_numpy(orientation).float()

        return {
            "image": image,
            "mask": mask,
            "intersection": inter,
            "orientation": orientation,
            "sample_id": sample_id,
        }

