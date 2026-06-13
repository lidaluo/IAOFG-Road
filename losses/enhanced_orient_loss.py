"""
IAOF++ 方向场增强监督：多 bin 分类 Focal Loss + 基于期望方向场的拉普拉斯平滑（道路区域内）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from utils.orient_bins import dxdy_to_bin_labels, logits_to_expected_dxdy


def focal_ce_loss_masked(
    logits: torch.Tensor,
    targets: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    logits: (B,C,H,W), targets: (B,H,W) long (-1 忽略), valid_mask: (B,1,H,W) float
    """
    B, C, H, W = logits.shape
    t = targets.clone()
    t[targets < 0] = 0
    ce = F.cross_entropy(logits, t, reduction="none")
    p = F.softmax(logits, dim=1)
    pt = p.gather(1, t.unsqueeze(1)).squeeze(1)
    focal = (1.0 - pt).pow(gamma) * ce
    vm = valid_mask.squeeze(1)
    valid = (targets >= 0).float() * vm
    denom = valid.sum() + eps
    return (focal * valid).sum() / denom


def laplacian_vec_smoothness(vec_xy: torch.Tensor, weight_mask: torch.Tensor) -> torch.Tensor:
    """
    vec_xy: (B,2,H,W) 期望方向分量；weight_mask: (B,1,H,W) 在道路内惩罚拉普拉斯能量 ||Lap(v)||^2。
    """
    lap_k = torch.tensor(
        [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
        dtype=vec_xy.dtype,
        device=vec_xy.device,
    ).view(1, 1, 3, 3)
    w = weight_mask
    lx = F.conv2d(vec_xy[:, 0:1], lap_k, padding=1)
    ly = F.conv2d(vec_xy[:, 1:2], lap_k, padding=1)
    return torch.mean(w * (lx * lx + ly * ly))


def enhanced_orient_losses(
    orient_logits: torch.Tensor,
    orient_gt: torch.Tensor,
    num_bins: int,
    focal_gamma: float = 2.0,
    lambda_smooth: float = 0.1,
    conf_thr: float = 0.5,
    norm_thr: float = 0.15,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    orient_gt: (B,3,H,W) dx,dy,conf（与原版 IAOF 一致）。
    返回 (focal_loss, smooth_loss, combined_orient_loss_for_logging)。
    """
    gt_dx = orient_gt[:, 0:1]
    gt_dy = orient_gt[:, 1:2]
    gt_conf = orient_gt[:, 2:3]

    targets = dxdy_to_bin_labels(gt_dx, gt_dy, gt_conf, num_bins, conf_thr, norm_thr)
    valid = (targets >= 0).float().unsqueeze(1)
    focal = focal_ce_loss_masked(orient_logits, targets, valid, gamma=focal_gamma)

    exp_xy = logits_to_expected_dxdy(orient_logits)
    smooth = laplacian_vec_smoothness(exp_xy, gt_conf.clamp(0, 1))
    total = focal + lambda_smooth * smooth
    return focal, smooth, total
