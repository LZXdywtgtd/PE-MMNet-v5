"""
PE-MMNet v4: PatchTST 时序骨干网络

PatchTST (Patch Time Series Transformer) 是一种专为时序预测设计的Transformer架构，
通过将时间序列分块（patch）来捕获局部和全局时序模式。

相比传统的1D-CNN，PatchTST具有以下优势：
1. 更强的局部模式捕获能力（通过patch）
2. 全局感受野（Transformer attention）
3. 更好的时序依赖建模

参考论文：PatchTST: A Vision Transformer for Time Series Forecasting (ICLR 2023)

使用方法：
```python
from models.pe_tsnet_patchtst import PatchTST1D

# 创建模型
patchtst = PatchTST1D(seq_len=300, d_model=64)

# 前向传播
x = torch.randn(2, 300)  # (batch, seq_len)
feat = patchtst(x)  # (batch, 64)
```
"""

import torch
import torch.nn as nn
import math


class PatchTST1D(nn.Module):
    """
    PatchTST 风格的时序编码器

    架构：
    1. Patch Embedding: 将序列分块并投影到d_model维
    2. Positional Encoding: 可学习的patch位置编码
    3. Transformer Encoder: 捕获patch间依赖关系
    4. 输出投影: 将d_model投影到目标维度

    Args:
        seq_len: 输入序列长度（默认300）
        patch_size: 分块大小（默认10，即每10个时间步为一个patch）
        d_model: Transformer维度（默认64）
        nhead: 注意力头数（默认4）
        num_layers: Transformer层数（默认2）
        dropout: Dropout比例（默认0.1）
        output_dim: 输出维度（默认64）
    """

    def __init__(self, seq_len=300, patch_size=10, d_model=64,
                 nhead=4, num_layers=2, dropout=0.1, output_dim=64):
        super().__init__()

        self.seq_len = seq_len
        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        # 如果序列长度不能被 patch_size 整除，截断到最近的整数倍
        if seq_len % patch_size != 0:
            self.adjusted_seq_len = (seq_len // patch_size) * patch_size
        else:
            self.adjusted_seq_len = seq_len

        # Patch Embedding: (B, patch_size) → (B, d_model)
        self.patch_embed = nn.Linear(patch_size, d_model)

        # 可学习的位置编码
        self.pos_embed = nn.Parameter(
            torch.randn(1, self.num_patches, d_model) * 0.02
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Layer Normalization
        self.norm = nn.LayerNorm(d_model)

        # 输出投影: d_model → output_dim
        self.output_proj = nn.Linear(d_model, output_dim)

        # 初始化
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.pos_embed, mean=0, std=0.02)
        nn.init.xavier_uniform_(self.patch_embed.weight)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入序列 (B, seq_len) 或 (B, num_channels, seq_len) for triple_channel

        Returns:
            (B, output_dim) - 时序特征
        """
        batch_size = x.size(0)

        # 支持 triple_channel: (B, 3, seq_len) → 对通道维度取均值 → (B, seq_len)
        if x.dim() == 3:
            x = x.mean(dim=1)  # (B, seq_len)

        # 截断到 adjusted_seq_len（如需要）
        if self.adjusted_seq_len < self.seq_len:
            x = x[:, :self.adjusted_seq_len]

        # 重塑为patches: (B, num_patches, patch_size)
        x = x.view(batch_size, self.num_patches, self.patch_size)

        # Patch embedding: (B, num_patches, patch_size) → (B, num_patches, d_model)
        x = self.patch_embed(x)

        # 添加位置编码
        x = x + self.pos_embed
        x = self.dropout(x)

        # Transformer编码
        x = self.transformer(x)

        # Layer Normalization
        x = self.norm(x)

        # 全局池化：取所有patch的均值
        x = x.mean(dim=1)  # (B, d_model)

        # 输出投影
        x = self.output_proj(x)  # (B, output_dim)

        return x


class PatchTSTWithRate(nn.Module):
    """
    PatchTST 增强版：同时处理原始温度和温度变化率

    输入为三通道：
    - 通道0: 初始温度
    - 通道1: 当前温度
    - 通道2: 温度变化率

    这种设计可以更好地捕获温度的时序动态特征。
    """

    def __init__(self, seq_len=300, patch_size=10, d_model=64,
                 nhead=4, num_layers=2, dropout=0.1, output_dim=64):
        super().__init__()

        # 温度序列编码器（处理拼接后的温度序列，seq_len * 2）
        self.temp_encoder = PatchTST1D(
            seq_len=seq_len * 2,  # 初始温度 + 当前温度拼接后长度翻倍
            patch_size=patch_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=d_model
        )

        # 变化率编码器（处理通道2）
        self.rate_encoder = PatchTST1D(
            seq_len=seq_len,
            patch_size=patch_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=d_model
        )

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x):
        """
        前向传播

        Args:
            x: 三通道输入 (B, 3, seq_len)
                - x[:, 0, :]: 初始温度
                - x[:, 1, :]: 当前温度
                - x[:, 2, :]: 温度变化率

        Returns:
            (B, output_dim) - 融合后的时序特征
        """
        # 分离通道
        init_temp = x[:, 0, :]  # (B, seq_len)
        curr_temp = x[:, 1, :]  # (B, seq_len)
        temp_rate = x[:, 2, :]  # (B, seq_len)

        # 编码温度序列（初始+当前拼接）
        temp_feat = self.temp_encoder(
            torch.cat([init_temp, curr_temp], dim=1)
        )  # (B, d_model)

        # 编码变化率
        rate_feat = self.rate_encoder(temp_rate)  # (B, d_model)

        # 融合
        fused = torch.cat([temp_feat, rate_feat], dim=1)  # (B, d_model*2)
        out = self.fusion(fused)  # (B, output_dim)

        return out


def test_patchtst():
    """测试 PatchTST 模块"""
    print("Testing PatchTST1D...")

    # 基本测试
    patchtst = PatchTST1D(seq_len=300, d_model=64)
    x = torch.randn(2, 300)
    out = patchtst(x)
    assert out.shape == (2, 64), f"Expected (2, 64), got {out.shape}"
    print(f"  PatchTST1D: {x.shape} → {out.shape} OK")

    # 测试不同配置
    patchtst_large = PatchTST1D(seq_len=300, d_model=128, nhead=8, num_layers=4)
    out = patchtst_large(x)
    assert out.shape == (2, 64), f"Expected (2, 64), got {out.shape}"
    print(f"  PatchTST1D(large): {x.shape} → {out.shape} OK")

    # 测试 PatchTSTWithRate
    print("Testing PatchTSTWithRate...")
    patchtst_rate = PatchTSTWithRate(seq_len=300)
    x_triple = torch.randn(2, 3, 300)  # (B, 3, seq_len)
    out = patchtst_rate(x_triple)
    assert out.shape == (2, 64), f"Expected (2, 64), got {out.shape}"
    print(f"  PatchTSTWithRate: {x_triple.shape} → {out.shape} OK")

    print("All PatchTST tests passed!")


if __name__ == "__main__":
    test_patchtst()
