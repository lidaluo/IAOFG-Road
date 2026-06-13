"""
IAOF++ 轻量级图边打分器（参数量极小，<100K）。
对候选节点对 (i,j) 根据拼接特征预测「应连接」logit，供后处理过滤假边 / 补边。
离线训练时：正样本为 GT 图边，负样本为同图内距离在范围内的非边对。
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LightGraphEdgeScorer(nn.Module):
    """
    输入每条边的向量：[xi,yi,xj,yj, dx,dy, dist, cos_align] 等（由调用方构造）。
    默认 in_dim=10，hidden=48。
    """

    def __init__(self, in_dim: int = 10, hidden: int = 48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, edge_feats: torch.Tensor) -> torch.Tensor:
        """
        edge_feats: (N, in_dim) -> (N,) logits
        """
        return self.net(edge_feats).squeeze(-1)
