"""
覆盖距离 loss（B 样条 + Hausdorff）

> 待 V5A3 任务实现。本文件作为 v5-α 占位，提醒后续开发者这里需要：
>   1. 用预测关键点 (B, K, 2) 拟合三次 B 样条曲线（每样本一条）
>   2. 在曲线上密集采样 M 个点
>   3. 真值裂纹像素 (B, N, 2) 到曲线的最小距离 → Hausdorff 距离作为 loss
>   4. 配套：bbox Smooth L1 loss、GP 模块正则项、泊松先验项

详见：
- docs/v5_架构设计.md §3.1
- tasks/team_v5_alpha.json 中 V5A3
"""

import torch
import torch.nn as nn


def coverage_loss(
    pred_kpts: torch.Tensor,
    true_crack_pixels: torch.Tensor,
    M: int = 200,
) -> torch.Tensor:
    """
    B 样条 + Hausdorff 覆盖距离 loss

    Args:
        pred_kpts: (B, K, 2) — 预测的有序关键点，K ∈ [8, 16]
        true_crack_pixels: (B, N, 2) — 真值裂纹像素（来自 crack_mask）
        M: 在曲线上密集采样的点数
    Returns:
        loss: (B,) — 每个样本的 Hausdorff 距离
    """
    # TODO(V5A3): 实现 B 样条拟合 + 距离计算
    #   1. fit_cubic_splines(pred_kpts) — scipy.interpolate 或自实现
    #   2. sample_along_splines(splines, M) — 沿参数 t 等间隔采样
    #   3. cdist(true_crack_pixels, spline_samples).min(dim=-1).values — 真值到曲线最近距离
    #   4. .max(dim=-1).values — Hausdorff 距离
    raise NotImplementedError(
        "coverage_loss 待 V5A3 实现。详见 docs/v5_架构设计.md §3.1"
    )


class OrderedKeypointLoss(nn.Module):
    """组合 loss：覆盖距离 + bbox + GP 正则 + 泊松先验"""

    def __init__(
        self,
        lambda_coverage: float = 1.0,
        lambda_bbox: float = 1.0,
        lambda_gp: float = 0.1,
        lambda_poisson: float = 0.05,
    ):
        super().__init__()
        self.lambda_coverage = lambda_coverage
        self.lambda_bbox = lambda_bbox
        self.lambda_gp = lambda_gp
        self.lambda_poisson = lambda_poisson

    def forward(
        self,
        pred_bbox: torch.Tensor,       # (B, 4)
        pred_kpts: torch.Tensor,        # (B, K, 2)
        true_bbox: torch.Tensor,        # (B, 4)
        true_crack_pixels: torch.Tensor,  # (B, N, 2)
        gp_module=None,                 # GPModule 实例（可选）
    ) -> dict:
        # TODO(V5A3): 实现总 loss 组合
        #   total = lambda_coverage * coverage_loss(...)
        #         + lambda_bbox * smooth_l1(pred_bbox, true_bbox)
        #         + lambda_gp * (-gp_module.log_marginal_likelihood())
        #         + lambda_poisson * poisson_prior(pred_kpts)
        raise NotImplementedError(
            "OrderedKeypointLoss 待 V5A3 实现。详见 docs/v5_架构设计.md §3.5"
        )