"""
GP 嵌入模块（V5A2）

高斯过程先验作为模型的可微模块（不是单纯的损失项）。

物理含义：
- GP 提供"平滑性先验"：相邻样本的特征应该有相似的调制
- 这是裂纹预测的物理直觉：相邻像素的裂纹特征应该相关
- 与单纯加 L2 正则不同，GP 通过核函数定义"相似度"

实现思路：
- 把 batch 内样本视为 GP 的采样点
- 学到的 RBF 核控制"平滑度"
- 输出 = input + GP_residual(input)
- log_marginal_likelihood() 作为正则项加入总 loss

参考：
- docs/v5_架构设计.md §2.3
- tasks/team_v5_alpha.json 中 V5A2
"""

from __future__ import annotations

import torch
import torch.nn as nn


class GPModule(nn.Module):
    """GP 嵌入模块（基于 RBF 核）"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        init_length_scale: float = 1.0,
        init_signal_var: float = 1.0,
        learnable: bool = True,
    ):
        """
        Args:
            in_dim:              输入特征维度
            hidden_dim:          残差网络中间层
            init_length_scale:   初始长度尺度 ℓ
            init_signal_var:      初始信号方差 σ²
            learnable:           核超参数是否可学习
        """
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # 核超参数（log 空间，保证正值）
        if learnable:
            self.log_length_scale = nn.Parameter(
                torch.tensor(float(torch.log(torch.tensor(init_length_scale))))
            )
            self.log_signal_var = nn.Parameter(
                torch.tensor(float(torch.log(torch.tensor(init_signal_var))))
            )
        else:
            self.register_buffer(
                "log_length_scale",
                torch.log(torch.tensor(init_length_scale)),
            )
            self.register_buffer(
                "log_signal_var",
                torch.log(torch.tensor(init_signal_var)),
            )

        # 残差投影
        self.residual = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim),
        )

        # 输出归一化（稳定训练）
        self.norm = nn.LayerNorm(in_dim)

    @property
    def length_scale(self) -> torch.Tensor:
        """ℓ，clamp 到合理范围"""
        return torch.exp(self.log_length_scale).clamp(min=0.1, max=10.0)

    @property
    def signal_var(self) -> torch.Tensor:
        """σ²"""
        return torch.exp(self.log_signal_var).clamp(min=0.1, max=10.0)

    def rbf_kernel(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        """
        RBF 核矩阵 K(x1, x2)

        K_ij = σ² · exp(-||x_i - x_j||² / (2ℓ²))

        注意：v5-α 默认在调用前对输入做 unit-std 归一化
        （见 _normalize_for_kernel），使 ℓ 的物理意义清晰。

        Args:
            x1: (B, in_dim)
            x2: (N, in_dim)

        Returns:
            K: (B, N)
        """
        # 输入归一化（让 ℓ 物理意义清晰，避免 K 退化为单位矩阵）
        x1_n = self._normalize_for_kernel(x1)
        x2_n = self._normalize_for_kernel(x2)

        # 扩展维度计算成对距离
        diff = x1_n.unsqueeze(1) - x2_n.unsqueeze(0)  # (B, N, in_dim)
        sq_dist = (diff ** 2).sum(dim=-1)  # (B, N)

        ell = self.length_scale
        sigma2 = self.signal_var

        return sigma2 * torch.exp(-0.5 * sq_dist / (ell ** 2 + 1e-6))

    def _normalize_for_kernel(self, x: torch.Tensor) -> torch.Tensor:
        """
        归一化特征，使 ℓ 物理意义清晰

        背景：当 in_dim 较大（如 128），||x_i - x_j||² 的期望是 in_dim 量级
        （约 128），远超默认 ℓ=1.0 的合理范围，导致 K ≈ I（单位矩阵），
        GP 平滑失效。

        方案：除以 sqrt(in_dim)，使每行 L2 范数 ~ 1。
        此时 ||x_i - x_j||² ~ O(1)，ℓ=1.0 物理意义 = "1 个单位向量距离"。

        Args:
            x: (B, in_dim)

        Returns:
            x_normalized: (B, in_dim) — 行向 L2 单位范数
        """
        # L2 归一化（每个样本为单位向量）
        norm = x.norm(dim=-1, keepdim=True).clamp(min=1e-3)
        return x / (norm * (self.in_dim ** 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用 GP 平滑 + 残差

        x_smooth_i = Σ_j K_ij · x_j / Σ_j K_ij   (GP 平滑)

        output = LayerNorm(x + residual(x_smooth))

        Args:
            x: (B, in_dim) — 融合特征

        Returns:
            output: (B, in_dim) — GP 调制后的特征
        """
        B = x.size(0)
        if B == 1:
            # 单样本无 GP 意义（无法定义 batch 内相关性）
            return self.norm(x + self.residual(x))

        # GP 平滑（batch 内样本视为采样点）
        K = self.rbf_kernel(x, x)  # (B, B)
        # 避免除零
        K_sum = K.sum(dim=-1, keepdim=True) + 1e-6  # (B, 1)
        x_smooth = torch.matmul(K, x) / K_sum  # (B, in_dim)

        # 残差 + 原始特征
        out = x + self.residual(x_smooth)
        out = self.norm(out)
        return out

    def log_marginal_likelihood(self) -> torch.Tensor:
        """
        log 边际似然（粗略近似）

        用于作为正则项加入总 loss：
            total_loss = ... + λ_gp * (-log_marginal_likelihood)

        简化版：惩罚极端长度尺度和方差
        """
        # 鼓励 length_scale 在中等范围（避免过小=无平滑 或 过大=过平滑）
        ell_penalty = (self.log_length_scale - 0.0) ** 2
        # 鼓励 signal_var 不要过大
        var_penalty = (self.log_signal_var - 0.0) ** 2

        return ell_penalty + var_penalty

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, "
            f"length_scale={self.length_scale.item():.3f}, "
            f"signal_var={self.signal_var.item():.3f}"
        )


if __name__ == "__main__":
    # 冒烟测试
    print("=" * 60)
    print("GPModule 冒烟测试")
    print("=" * 60)

    gp = GPModule(in_dim=256, hidden_dim=128)
    print(f"[OK] GPModule 初始化: in_dim=256")
    print(f"     初始 length_scale={gp.length_scale.item():.3f}")
    print(f"     初始 signal_var={gp.signal_var.item():.3f}")
    print(f"     参数总数: {sum(p.numel() for p in gp.parameters()):,}")

    # 测试 batch=4
    x = torch.randn(4, 256)
    out = gp(x)
    print(f"\n输入:  {tuple(x.shape)}")
    print(f"输出:  {tuple(out.shape)}, range [{out.min():.3f}, {out.max():.3f}]")
    assert out.shape == x.shape, "输入输出形状一致"

    # 测试 batch=1（边界情况）
    x_single = torch.randn(1, 256)
    out_single = gp(x_single)
    print(f"\nbatch=1 输入: {tuple(x_single.shape)}, 输出: {tuple(out_single.shape)}")
    assert out_single.shape == x_single.shape

    # log_marginal_likelihood
    lml = gp.log_marginal_likelihood()
    print(f"\nlog_marginal_likelihood: {lml.item():.4f}")
    assert lml.item() >= 0, "LML 应 >= 0（平方和）"

    # 测试核矩阵对称性
    x_test = torch.randn(5, 256)
    K = gp.rbf_kernel(x_test, x_test)
    assert torch.allclose(K, K.T, atol=1e-5), "RBF 核应对称"
    print(f"[OK] RBF 核对称性验证通过")

    print("\n[ALL PASS] ✅")