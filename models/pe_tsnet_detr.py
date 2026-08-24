"""
PE-MMNet v4: DETR 风格变体模型

DETR (DEtection TRansformer) 架构

架构特点：
- 2D骨干：ResNet-18
- 位置编码：2D 正弦位置编码
- Transformer：Encoder-Decoder 架构
- Object Queries：100 个可学习查询
- Hungarian Matching：预测与标签的最优匹配

关键设计：
- 使用 self.training 属性区分训练/推理模式
- 训练模式返回完整 query 预测
- 推理模式返回最高 conf 的预测
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .pe_tsnet_multimodal import (
    ResNet18Backbone2D,
    TemporalFeatureExtractor,
    MultiTaskHead,
    CrossAttentionFusion
)


# =============================================================================
# 2D 位置编码
# =============================================================================

class PositionalEncoding2D(nn.Module):
    """
    2D 正弦位置编码

    为 Transformer 提供空间位置信息

    Args:
        d_model: 编码维度
        height: 特征图高度
        width: 特征图宽度
    """

    def __init__(self, d_model, height=16, width=16):
        super().__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(d_model, height, width)

        # 分割维度：前半部分给高度，后半部分给宽度
        d_model_h = d_model // 2
        d_model_w = d_model - d_model_h

        # 位置编码计算
        # 使用正弦和余弦函数创建周期性编码
        div_term_h = torch.exp(
            torch.arange(0, d_model_h, 2).float() * (-math.log(10000.0) / d_model_h)
        )
        div_term_w = torch.exp(
            torch.arange(0, d_model_w, 2).float() * (-math.log(10000.0) / d_model_w)
        )

        # 高度维度
        for i in range(d_model_h // 2):
            pe[2 * i, :, :] = torch.sin(div_term_h[i] * torch.arange(0, height).float().view(-1, 1))
            pe[2 * i + 1, :, :] = torch.cos(div_term_h[i] * torch.arange(0, height).float().view(-1, 1))

        # 宽度维度
        for i in range(d_model_w // 2):
            pe[d_model_h + 2 * i, :, :] = torch.sin(div_term_w[i] * torch.arange(0, width).float().view(1, -1))
            pe[d_model_h + 2 * i + 1, :, :] = torch.cos(div_term_w[i] * torch.arange(0, width).float().view(1, -1))

        # 注册为 buffer（不参与梯度计算，但会随模型移动）
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, d_model, H, W)

        self.height = height
        self.width = width

    def forward(self, x):
        """
        Args:
            x: (B, d_model, H, W) 输入特征

        Returns:
            x + positional_encoding: 添加位置编码后的特征
        """
        # 确保尺寸匹配
        H, W = x.shape[-2:]
        if H != self.height or W != self.width:
            # 动态调整（理论上不应该发生）
            pe = F.interpolate(
                self.pe,
                size=(H, W),
                mode='bilinear',
                align_corners=False
            )
        else:
            pe = self.pe

        return x + pe


# =============================================================================
# DETR Transformer Decoder
# =============================================================================

class DETRDecoder(nn.Module):
    """
    DETR Transformer 解码器

    使用可学习的 Object Queries 通过交叉注意力从图像特征中查询目标

    Args:
        d_model: 模型维度
        nhead: 注意力头数
        num_decoder_layers: 解码器层数
        num_queries: Object Queries 数量
        dropout: Dropout 比例
    """

    def __init__(self, d_model=512, nhead=8, num_decoder_layers=6,
                 num_queries=100, dropout=0.1):
        super().__init__()

        self.num_queries = num_queries

        # 可学习的 Object Queries
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Transformer 解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN 稳定性更好
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )

    def forward(self, memory):
        """
        Args:
            memory: (B, seq_len, d_model) 来自编码器的图像特征

        Returns:
            hs: (B, num_queries, d_model) 解码器输出
        """
        B = memory.size(0)

        # 可学习的 queries
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)  # (B, num_queries, d_model)

        # 解码器：queries 关注 memory
        hs = self.transformer_decoder(query_embed, memory)  # (B, num_queries, d_model)

        return hs


# =============================================================================
# DETR 预测头
# =============================================================================

class DETRHead(nn.Module):
    """
    DETR 预测头

    将 decoder 输出转换为检测预测

    Args:
        d_model: 输入维度
    """

    def __init__(self, d_model=512):
        super().__init__()

        # 边界框预测：归一化坐标 [0, 1]
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, 4)  # [x, y, l, w]
        )

        # 置信度预测
        self.conf_embed = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1)
        )

        # 密度预测
        self.density_embed = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, hs):
        """
        Args:
            hs: (B, num_queries, d_model) decoder 输出

        Returns:
            pred: (B, num_queries, 6) 预测 [x, y, l, w, conf, density]
        """
        # 边界框：使用 Sigmoid 归一化到 [0, 1]
        boxes = torch.sigmoid(self.bbox_embed(hs))  # (B, num_queries, 4)

        # 置信度：使用 Sigmoid
        conf = torch.sigmoid(self.conf_embed(hs))  # (B, num_queries, 1)

        # 密度：使用 Sigmoid
        density = torch.sigmoid(self.density_embed(hs))  # (B, num_queries, 1)

        # 拼接
        pred = torch.cat([boxes, conf, density], dim=-1)  # (B, num_queries, 6)

        return pred


# =============================================================================
# DETR 风格模型
# =============================================================================

class DETRStyle(nn.Module):
    """
    DETR 风格检测模型

    架构：
    1. ResNet-18 2D 骨干
    2. 2D 位置编码
    3. Transformer Encoder (6层)
    4. Transformer Decoder (6层) + 100 Object Queries
    5. DETR Head
    6. CrossAttentionFusion + 1D 分支
    7. MultiTaskHead

    输出：
    - 训练模式 (model.train()): 返回完整 query 预测
    - 推理模式 (model.eval()): 返回最高 conf 的预测

    Args:
        seq_len: 1D 序列长度（默认 300）
        image_channels: 2D 图像通道数（默认 2）
        image_size: 输入图像尺寸（默认 256）
        pretrained_2d: 2D 骨干是否使用预训练权重
        dropout: Dropout 比例
        d_model: Transformer 维度（默认 512）
        num_queries: Object Queries 数量（默认 100）
        encoder_layers: 编码器层数
        decoder_layers: 解码器层数
    """

    def __init__(self, seq_len=300, image_channels=2, image_size=256,
                 pretrained_2d=True, dropout=0.2, d_model=512,
                 num_queries=100, encoder_layers=6, decoder_layers=6):
        super().__init__()

        self.seq_len = seq_len
        self.image_channels = image_channels
        self.image_size = image_size
        self.d_model = d_model
        self.num_queries = num_queries

        # 特征图尺寸（ResNet-18 stride=32：image_size / 32）
        self.feat_size = image_size // 32

        # 2D 骨干：ResNet-18（启用空间特征输出）
        self.backbone_2d = ResNet18Backbone2D(
            in_channels=image_channels,
            pretrained=pretrained_2d
        )
        self.backbone_2d.set_spatial_output(True)  # 输出 (B, 512, 8, 8) 而非 (B, 512)

        # 投影层：512 -> d_model
        self.input_proj = nn.Conv2d(512, d_model, kernel_size=1)

        # 2D 位置编码
        self.pos_encoder = PositionalEncoding2D(d_model, self.feat_size, self.feat_size)

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            enable_nested_tensor=False  # norm_first=True 时嵌套张量优化不兼容
        )

        # Transformer 解码器
        self.decoder = DETRDecoder(
            d_model=d_model,
            nhead=8,
            num_decoder_layers=decoder_layers,
            num_queries=num_queries,
            dropout=dropout
        )

        # DETR 预测头
        self.detr_head = DETRHead(d_model=d_model)

        # 1D 分支
        self.backbone_1d = TemporalFeatureExtractor(
            input_dim=1,
            hidden_dim=32,
            num_heads=4,
            dropout=dropout
        )
        feat_dim_1d = 64

        # query 投影：将 6 维 query 投影到 128 维再融合
        self.query_proj = nn.Linear(6, 128)

        # 融合层
        self.fusion = CrossAttentionFusion(dim_2d=128, dim_1d=feat_dim_1d)

        # 输出头
        fused_dim = 128 + feat_dim_1d  # 128 + 64 = 192
        self.output_head = MultiTaskHead(
            input_dim=fused_dim,
            hidden_dim=128,
            dropout=dropout
        )

    def forward(self, x_1d, x_2d):
        """
        前向传播

        Args:
            x_1d: (B, seq_len) 温度时序
            x_2d: (B, 2, H, W) 温度场+应力场图像

        Returns:
            output: 检测结果
                - 训练模式: (B, num_queries, 6) 完整 query 预测
                - 推理模式: (B, 6) 最高 conf 的预测
            global_density: (B, 1) 全局密度（用于损失计算）
        """
        # ========== 2D 分支 ==========
        # ResNet 特征提取
        feat_2d = self.backbone_2d(x_2d)  # (B, 512, 16, 16)

        # 投影到 d_model
        feat_2d = self.input_proj(feat_2d)  # (B, d_model, 16, 16)

        # 添加位置编码
        feat_2d = self.pos_encoder(feat_2d)  # (B, d_model, 16, 16)

        # 展平为序列
        memory = feat_2d.flatten(2).transpose(1, 2)  # (B, 256, d_model)

        # Transformer 编码器
        memory = self.transformer_encoder(memory)  # (B, 256, d_model)

        # ========== DETR 解码器 ==========
        decoder_out = self.decoder(memory)  # (B, num_queries, d_model)

        # DETR 预测
        detr_pred = self.detr_head(decoder_out)  # (B, num_queries, 6)

        # 全局密度：取所有 query 预测密度的最大值（两个模式都用）
        global_density = detr_pred[..., 5:6].max(dim=1, keepdim=True)[0]  # (B, 1)

        # ========== 模式判断 ==========
        # ✅ 使用 self.training 判断模式
        if self.training:
            # 训练模式：返回完整 detr_pred 供损失计算（不做融合）
            return detr_pred, global_density

        # ========== 推理模式 ==========
        # 取最高 conf 的预测
        conf = detr_pred[..., 4:5]  # (B, num_queries, 1)
        best_idx = conf.squeeze(-1).argmax(dim=1)  # (B,)
        B_d = detr_pred.size(0)
        best_query = detr_pred[torch.arange(B_d, device=detr_pred.device), best_idx]  # (B, 6)

        # ========== 1D 分支 ==========
        feat_1d = self.backbone_1d(x_1d)  # (B, 64)

        # ========== 融合 ==========
        # 推理时：query_proj → fusion → output_head（与训练链路一致）
        query_feat = self.query_proj(best_query)  # (B, 6) → (B, 128)
        fused = self.fusion(query_feat, feat_1d)  # (B, 192)

        # ========== 输出 ==========
        output = self.output_head(fused)

        return output, global_density

    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
