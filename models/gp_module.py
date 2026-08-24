"""
GP 嵌入模块（v5-α 占位）

> 待 V5A2 任务实现。本文件作为 v5-α 占位，提醒后续开发者这里需要：
>   1. 实现高斯过程先验（RBF 核默认，可换 Matern）
>   2. 提供 log_marginal_likelihood() 用于正则项
>   3. 调制方式：output = input + GP_residual(input)
>   4. 长度尺度 ℓ 与方差 σ² 设为可学习参数

详见：
- docs/v5_架构设计.md §2.3
- tasks/team_v5_alpha.json 中 V5A2
"""

import torch
import torch.nn as nn


class GPModule(nn.Module):
    """GP 嵌入模块（v5-α 占位）"""

    def __init__(self, in_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # TODO(V5A2): 实现 GP 调制
        #   - RBF 核：k(x, x') = σ² exp(-||x - x'||² / (2ℓ²))
        #   - 可学习参数：log_sigma, log_ell
        #   - log_marginal_likelihood() 用于正则
        raise NotImplementedError(
            "GPModule 待 V5A2 实现。详见 docs/v5_架构设计.md §2.3"
        )

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, in_dim) — 融合特征
        Returns:
            output: (B, in_dim) — 经过 GP 调制后的特征
        """
        raise NotImplementedError

    def log_marginal_likelihood(self) -> torch.Tensor:
        """GP 超参数正则项（loss 中使用）"""
        raise NotImplementedError