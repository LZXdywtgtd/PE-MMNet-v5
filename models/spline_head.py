"""
Spline 输出头（v5-α 占位）

> 待 V5A2 任务实现。本文件作为 v5-α 占位，提醒后续开发者这里需要：
>   1. 从融合特征回归 bbox（4 维）
>   2. 从融合特征回归 8–16 动态关键点（变长）
>   3. 用 B 样条拟合关键点 → 平滑曲线
>   4. 关键点 K 自适应选择策略（端点 + 等弧长采样）

详见：
- docs/v5_架构设计.md §2.4
- tasks/team_v5_alpha.json 中 V5A2
"""

import torch
import torch.nn as nn


class SplineHead(nn.Module):
    """Spline 输出头（v5-α 占位）"""

    def __init__(self, in_dim: int, hidden_dim: int = 256, max_kpts: int = 16):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.max_kpts = max_kpts

        # TODO(V5A2): 实现 bbox 回归 + 关键点回归 + B 样条拟合
        raise NotImplementedError(
            "SplineHead 待 V5A2 实现。详见 docs/v5_架构设计.md §2.4"
        )

    def forward(self, feat: torch.Tensor):
        """
        Args:
            feat: (B, in_dim) — 融合特征
        Returns:
            bbox: (B, 4) — 归一化角点
            keypoints: (B, K, 2), K ∈ [8, max_kpts] — 有序关键点
        """
        raise NotImplementedError