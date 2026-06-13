"""
IAOF-Graph — Local Query Decoder（对应 GLD-Road 论文 §3.4 Local Query Decoder 思想）

论文要点：在全局阶段得到的初始图 G_init 往往存在断裂；以道路端点为中心取局部窗口，
联合「原始影像 + 当前粗道路栅格」做局部查询，预测缺失连接的走向 / 下一节点位置。

本实现说明（论文未给出逐层结构时采用轻量 CNN + 回归头）：
- 输入：4 通道 — RGB（或占位）与粗道路掩膜（G_init 栅格化），空间尺寸 patch_size（默认 128）。
- 输出：2 维 — 在 patch 像素坐标系下，相对 patch 几何中心的偏移 (dx, dy)，
  训练时用 tanh 压缩到 (-1,1) 再反归一化，对应「沿道路延伸方向上的下一锚点」。
  该设计与 Algorithm 1 中「在局部窗口内搜索/补全拓扑」一致，便于用 SmoothL1 监督。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LocalQueryDecoder(nn.Module):
    def __init__(self, in_channels: int = 4, base: int = 32):
        super().__init__()
        self.in_channels = in_channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base, 3, padding=1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 4, base * 4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base * 4, base * 2),
            nn.ReLU(inplace=True),
            nn.Linear(base * 2, 2),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 4, H, W) — RGB + road
        返回 (B, 2)，每维在 (-1,1)，需乘以 scale 得到像素偏移。
        """
        z = self.encoder(x)
        return self.head(z)

    @staticmethod
    def offsets_from_output(out: torch.Tensor, patch_hw: tuple[int, int]) -> torch.Tensor:
        """将 (-1,1) 预测转为像素偏移 (B,2)，以 patch 半宽/半高为尺度。"""
        h, w = patch_hw
        sx = float(w) * 0.5
        sy = float(h) * 0.5
        dx = out[:, 0:1] * sx
        dy = out[:, 1:2] * sy
        return torch.cat([dx, dy], dim=1)
