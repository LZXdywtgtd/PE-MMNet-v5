"""
PE-MMNet v4: 门控多模态融合模块

门控多模态融合是一种将温度和应力特征分离处理后通过门控机制动态融合的方法。

原理：
1. 将2D特征按通道分为"温度子空间"和"应力子空间"
2. 1D时序特征分别与两个子空间进行交叉注意力
3. 门控网络根据1D特征决定如何融合两个分支的输出

优势：
- 温度和应力特征独立建模，避免相互干扰
- 门控机制自适应选择融合权重
- 更符合物理直觉（温度和应力是不同物理量）

使用方法：
```python
from models.pe_tsnet_fusion import GatedMultimodalFusion

# 创建融合模块
fusion = GatedMultimodalFusion(dim_2d=512, dim_1d=64, split_ratio=0.5)

# 前向传播
feat_2d = torch.randn(2, 512)  # 2D特征（温度+应力）
feat_1d = torch.randn(2, 64)   # 1D时序特征
fused = fusion(feat_2d, feat_1d)  # (2, 64)
```
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttentionBranch(nn.Module):
    """
    交叉注意力分支

    将1D特征作为Query，2D特征的子空间作为Key和Value进行交叉注意力计算。
    """

    def __init__(self, dim_2d, dim_1d, num_heads=4, dropout=0.1):
        """
        Args:
            dim_2d: 2D特征子空间维度
            dim_1d: 1D特征维度
            num_heads: 注意力头数
            dropout: Dropout比例
        """
        super().__init__()
        self.dim_2d = dim_2d
        self.dim_1d = dim_1d
        self.num_heads = num_heads
        self.head_dim = dim_1d // num_heads

        assert dim_1d % num_heads == 0, "dim_1d must be divisible by num_heads"

        # 1D特征作为Query
        self.q_linear = nn.Linear(dim_1d, dim_1d)
        # 2D特征子空间作为Key和Value
        self.kv_linear = nn.Linear(dim_2d, dim_1d * 2)

        self.out_linear = nn.Linear(dim_1d, dim_1d)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, feat_2d_subspace, feat_1d):
        """
        Args:
            feat_2d_subspace: 2D特征的子空间 (B, dim_2d)
            feat_1d: 1D特征 (B, dim_1d)

        Returns:
            (B, dim_1d) - 交叉注意力增强的1D特征
        """
        batch_size = feat_2d_subspace.size(0)

        # Q: 1D特征, KV: 2D特征子空间
        q = self.q_linear(feat_1d)  # (B, dim_1d)
        kv = self.kv_linear(feat_2d_subspace)  # (B, dim_1d * 2)
        k, v = kv.chunk(2, dim=-1)  # 各 (B, dim_1d)

        # Reshape for multi-head attention
        q = q.view(batch_size, self.num_heads, self.head_dim)
        k = k.view(batch_size, self.num_heads, self.head_dim)
        v = v.view(batch_size, self.num_heads, self.head_dim)

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, head_dim, head_dim)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = attn @ v  # (B, num_heads, head_dim)
        out = out.view(batch_size, self.dim_1d)
        out = self.out_linear(out)

        return out


class GatedMultimodalFusion(nn.Module):
    """
    门控多模态融合模块

    将2D特征按通道分为温度子空间和应力子空间，
    分别与1D时序特征做交叉注意力，
    最后通过门控机制动态融合两个分支的输出。

    设计原理：
    - 裂纹的形成与温度变化（热应力）和机械应力都有关
    - 但温度和应力对裂纹的影响机制不同
    - 分别建模可以更好地捕获各自的特征
    - 门控机制可以自适应决定当前样本更依赖哪个分支
    """

    def __init__(self, dim_2d=512, dim_1d=64, split_ratio=0.5,
                 num_heads=4, dropout=0.1):
        """
        Args:
            dim_2d: 2D特征总维度（默认512）
            dim_1d: 1D特征维度（默认64）
            split_ratio: 温度子空间占比（默认0.5，即各占一半）
            num_heads: 交叉注意力的头数
            dropout: Dropout比例
        """
        super().__init__()

        self.dim_2d = dim_2d
        self.dim_1d = dim_1d
        self.split_ratio = split_ratio

        # 通道分割
        self.temp_channels = int(dim_2d * split_ratio)
        self.stress_channels = dim_2d - self.temp_channels

        # 温度分支交叉注意力
        self.temp_attn = CrossAttentionBranch(
            dim_2d=self.temp_channels,
            dim_1d=dim_1d,
            num_heads=num_heads,
            dropout=dropout
        )

        # 应力分支交叉注意力
        self.stress_attn = CrossAttentionBranch(
            dim_2d=self.stress_channels,
            dim_1d=dim_1d,
            num_heads=num_heads,
            dropout=dropout
        )

        # 门控网络：根据1D特征生成融合权重
        self.gate = nn.Sequential(
            nn.Linear(dim_1d, dim_1d // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_1d // 2, 2),
            nn.Softmax(dim=-1)  # 输出两个概率值，和为1
        )

    def forward(self, feat_2d, feat_1d):
        """
        前向传播

        Args:
            feat_2d: 2D特征 (B, dim_2d) - 包含温度和应力信息
            feat_1d: 1D时序特征 (B, dim_1d)

        Returns:
            (B, dim_2d + dim_1d) - 融合后的特征，拼接原始2D特征与门控融合结果
        """
        # 分割2D特征为温度子空间和应力子空间
        temp_feat = feat_2d[:, :self.temp_channels]
        stress_feat = feat_2d[:, self.temp_channels:]

        # 分别做交叉注意力
        temp_out = self.temp_attn(temp_feat, feat_1d)  # (B, dim_1d)
        stress_out = self.stress_attn(stress_feat, feat_1d)  # (B, dim_1d)

        # 生成门控权重
        gate_weight = self.gate(feat_1d)  # (B, 2)，两个值和为1

        # 门控融合
        gate_temp = gate_weight[:, 0:1]  # (B, 1)
        gate_stress = gate_weight[:, 1:2]  # (B, 1)
        gated = gate_temp * temp_out + gate_stress * stress_out  # (B, dim_1d)

        # 拼接原始2D特征与门控融合结果，保持与 CrossAttentionFusion 输出维度一致
        return torch.cat([feat_2d, gated], dim=-1)  # (B, dim_2d + dim_1d)


class AdaptiveFusion(nn.Module):
    """
    自适应融合模块（简化版）

    与GatedMultimodalFusion不同，这个版本直接用1D特征作为门控信号，
    通过加权求和融合2D特征的不同通道组。
    """

    def __init__(self, dim_2d=512, dim_1d=64, num_groups=4, dropout=0.1):
        """
        Args:
            dim_2d: 2D特征维度
            dim_1d: 1D特征维度
            num_groups: 分组数量
            dropout: Dropout比例
        """
        super().__init__()

        self.dim_2d = dim_2d
        self.num_groups = num_groups
        self.group_size = dim_2d // num_groups

        # 通道权重生成器
        self.channel_weights = nn.Sequential(
            nn.Linear(dim_1d, dim_1d // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_1d // 2, dim_2d),
            nn.Sigmoid()
        )

        # 2D特征投影
        self.proj_2d = nn.Linear(dim_2d, dim_1d)

        # 1D特征处理
        self.proj_1d = nn.Linear(dim_1d, dim_1d)

    def forward(self, feat_2d, feat_1d):
        """
        Args:
            feat_2d: (B, dim_2d)
            feat_1d: (B, dim_1d)

        Returns:
            (B, dim_1d)
        """
        # 生成通道权重
        weights = self.channel_weights(feat_1d)  # (B, dim_2d)

        # 加权融合2D特征
        weighted_2d = feat_2d * weights  # (B, dim_2d)

        # 投影到统一维度
        fused_2d = self.proj_2d(weighted_2d)  # (B, dim_1d)

        # 与1D特征融合
        proj_1d = self.proj_1d(feat_1d)  # (B, dim_1d)

        return fused_2d + proj_1d


def test_gated_fusion():
    """测试门控融合模块"""
    print("Testing GatedMultimodalFusion...")

    fusion = GatedMultimodalFusion(dim_2d=512, dim_1d=64, split_ratio=0.5)

    feat_2d = torch.randn(2, 512)
    feat_1d = torch.randn(2, 64)

    out = fusion(feat_2d, feat_1d)
    assert out.shape == (2, 64), f"Expected (2, 64), got {out.shape}"
    print(f"  GatedMultimodalFusion: ({feat_2d.shape}, {feat_1d.shape}) → {out.shape} OK")

    # 测试门控权重
    gate_weight = fusion.gate(feat_1d)
    assert gate_weight.shape == (2, 2), f"Expected (2, 2), got {gate_weight.shape}"
    assert torch.allclose(gate_weight.sum(dim=-1), torch.ones(2), atol=1e-5), \
        "Gate weights should sum to 1"
    print(f"  Gate weights sum: {gate_weight.sum(dim=-1).tolist()} OK")

    # 测试AdaptiveFusion
    print("Testing AdaptiveFusion...")
    adaptive = AdaptiveFusion(dim_2d=512, dim_1d=64)
    out = adaptive(feat_2d, feat_1d)
    assert out.shape == (2, 64), f"Expected (2, 64), got {out.shape}"
    print(f"  AdaptiveFusion: ({feat_2d.shape}, {feat_1d.shape}) → {out.shape} OK")

    print("All fusion tests passed!")


if __name__ == "__main__":
    test_gated_fusion()
