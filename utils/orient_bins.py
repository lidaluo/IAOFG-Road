"""IAOF++：将连续切向 (dx,dy) 映射为 K 个角度 bin（每 bin 10° 当 K=36）。"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


def bin_centers_unit_vectors(num_bins: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """形状 (K,2)：第 k 个 bin 中心方向单位向量 (cos θ, sin θ)，θ = (k+0.5)/K * 2π。"""
    k = torch.arange(num_bins, device=device, dtype=dtype)
    theta = (k + 0.5) * (2 * math.pi / num_bins)
    return torch.stack([torch.cos(theta), torch.sin(theta)], dim=1)


def dxdy_to_bin_labels(
    dx: torch.Tensor,
    dy: torch.Tensor,
    conf: torch.Tensor,
    num_bins: int,
    conf_thr: float = 0.5,
    norm_thr: float = 0.15,
) -> torch.Tensor:
    """
    dx,dy,conf: (B,1,H,W) 或 (B,H,W)；返回 (B,H,W) long，无效像素为 -1。
    """
    if dx.dim() == 4:
        dx = dx[:, 0]
        dy = dy[:, 0]
        conf = conf[:, 0]
    ang = torch.atan2(dy, dx)
    b = ((ang + math.pi) / (2 * math.pi) * num_bins).long().clamp(0, num_bins - 1)
    norm2 = dx * dx + dy * dy
    valid = (conf > conf_thr) & (norm2 > norm_thr * norm_thr)
    out = torch.full_like(b, -1, dtype=torch.long)
    out = torch.where(valid, b, out)
    return out


def logits_to_expected_dxdy(orient_logits: torch.Tensor) -> torch.Tensor:
    """orient_logits (B,K,H,W) -> 期望方向 (B,2,H,W)（softmax 加权 bin 中心单位向量）。"""
    p = torch.softmax(orient_logits, dim=1)
    uv = bin_centers_unit_vectors(p.shape[1], device=p.device, dtype=p.dtype)
    ex = torch.einsum("bkhw,k->bhw", p, uv[:, 0]).unsqueeze(1)
    ey = torch.einsum("bkhw,k->bhw", p, uv[:, 1]).unsqueeze(1)
    return torch.cat([ex, ey], dim=1)
