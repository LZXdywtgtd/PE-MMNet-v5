"""
PE-MMNet v4: 多模态融合网络
用于日用陶瓷热震裂纹实时预测

架构：非对称双分支融合网络
  - 分支 1（2D）：ResNet-18 视觉骨干 → 512 维特征向量
  - 分支 2（1D）：Micro + Macro 1D-CNN + 自注意力 → 128 维特征向量
  - 融合层：Cross-Attention（交叉注意力）融合双分支特征
  - 输出层：MLP 多任务输出 [x, y, l, w, confidence, density]

输入：
  - 1D 分支：(batch, 300) 温度时序
  - 2D 分支：(batch, 2, 256, 256) 温度场+应力场图像

输出：
  - (batch, 6): [x, y, l, w, confidence, density]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


# =============================================================================
# SE/CoordAttention 模块
# =============================================================================

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation 模块
    通过通道注意力机制增强特征表示

    原理：
    - Squeeze: 全局平均池化，将空间信息压缩为通道描述符
    - Excitation: 两层全连接学习通道依赖关系
    - Scale: 用学习到的注意力权重重新校准通道
    """

    def __init__(self, channels, reduction=16):
        """
        Args:
            channels: 输入通道数
            reduction: 压缩比（默认16，如512→32）
        """
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, C, H, W) - 通道注意力加权后的特征
        """
        b, c, _, _ = x.size()
        # Squeeze: (B, C, H, W) → (B, C)
        y = self.squeeze(x).view(b, c)
        # Excitation: (B, C) → (B, C) → (B, C, 1, 1)
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: 通道注意力加权
        return x * y.expand_as(x)


class CoordAtt(nn.Module):
    """
    Coordinate Attention（坐标注意力）

    通过将通道注意力分解为两个1D特征编码来实现，
    分别沿水平和垂直方向聚合特征，从而捕获长程依赖和精确位置信息。

    适用于需要位置感知的视觉任务（如裂纹检测）。
    """

    def __init__(self, inp, oup, reduction=32):
        """
        Args:
            inp: 输入通道数
            oup: 输出通道数（通常与输入相同）
            reduction: 中间层通道压缩比
        """
        super().__init__()
        mip = max(8, inp // reduction)

        # 高度方向池化：保留宽度信息
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        # 宽度方向池化：保留高度信息
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # 共享卷积：降低通道维度
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)

        # 分离的方向卷积
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, C, H, W) - 坐标注意力加权后的特征
        """
        identity = x
        n, c, h, w = x.size()

        # 高度方向编码
        x_h = self.pool_h(x)  # (B, C, H, 1)
        # 宽度方向编码
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, 1, W) → (B, C, W, 1)

        # 拼接并通过共享卷积
        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+1, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        # 分离高度和宽度特征
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        # 生成注意力权重
        a_h = self.conv_h(x_h).sigmoid()  # (B, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (B, C, 1, W)

        # 坐标注意力加权
        return identity * a_w * a_h


class BackboneWithAttention(nn.Module):
    """
    骨干网络包装器，自动添加注意力模块

    用法：
    ```python
    backbone = ResNet18Backbone2D(in_channels=2)
    enhanced_backbone = BackboneWithAttention(backbone, attention_type='se')
    # 或
    enhanced_backbone = BackboneWithAttention(backbone, attention_type='coord')
    ```
    """

    def __init__(self, backbone, attention_type='se', reduction=16):
        """
        Args:
            backbone: 骨干网络（如 ResNet18Backbone2D）
            attention_type: 注意力类型，'se' 或 'coord' 或 None
            reduction: 注意力模块的压缩比
        """
        super().__init__()
        self.backbone = backbone

        # 安全获取通道数（兼容不同骨干网络）
        if hasattr(backbone, 'out_channels'):
            channels = backbone.out_channels
        elif hasattr(backbone, 'feature_dim'):
            channels = backbone.feature_dim
        else:
            # 如果都没有，尝试动态推导
            import torch
            with torch.no_grad():
                try:
                    dummy = torch.randn(1, 2, 256, 256)
                    channels = backbone(dummy).size(1)
                    print(f"[提示] 未找到 out_channels/feature_dim，动态推导通道数: {channels}")
                except Exception as e:
                    channels = 512  # 回退到默认值
                    print(f"[警告] 无法推导通道数，使用默认值: {channels} ({e})")

        out_channels = channels

        if attention_type == 'se':
            self.attention = SEBlock(channels=out_channels, reduction=reduction)
        elif attention_type == 'coord':
            self.attention = CoordAtt(inp=out_channels, oup=out_channels, reduction=reduction)
        else:
            self.attention = nn.Identity()

        # 保持与原始骨干网络相同的接口
        self.feature_dim = out_channels

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)

        Returns:
            (B, feature_dim) - 注意力增强的特征
        """
        feat = self.backbone(x)

        # 处理多尺度特征（骨干返回列表，如 Swin 多层输出）
        if isinstance(feat, (list, tuple)):
            # 取最后一层作为主特征（最高语义级别）
            feat = feat[-1]

        # 如果骨干返回的是2D特征图，先应用注意力
        if feat.dim() == 4:
            feat = self.attention(feat)
            feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
        return feat


# =============================================================================
# 分支 1：2D 视觉特征提取（ResNet-18 骨干）
# =============================================================================

class ResNet18Backbone2D(nn.Module):
    """
    2D 图像特征提取分支

    使用预训练的 ResNet-18 作为视觉骨干网络：
    - 输入：单通道或三通道图像（修改 conv1 以适配单通道输入）
    - 结构：ResNet-18 的 conv1 ~ layer4（去除最后的 fc 层）
    - 输出：全局平均池化后的特征向量

    设计选择：
    - 使用 ResNet-18 而非 MobileNet-V3：更稳定的特征提取能力
    - 全局平均池化：将 spatial feature map 聚合成固定长度向量
    """

    def __init__(self, in_channels=2, pretrained=True):
        """
        Args:
            in_channels: 输入通道数（默认 2：温度场+应力场）
            pretrained: 是否使用 ImageNet 预训练权重
        """
        super().__init__()

        # 加载预训练的 ResNet-18
        self.resnet = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)

        # -------------------------------------------------------------------------
        # 修改第一层卷积以适配单通道输入
        # -------------------------------------------------------------------------
        # 原始 ResNet-18 的 conv1：输入 3 通道，输出 64 通道，kernel=7, stride=2
        # 如果 in_channels == 1 或 2，我们需要修改输入层
        if in_channels != 3:
            # 保存原始 conv1 的配置
            original_conv1 = self.resnet.conv1

            # 创建新的卷积层：保持输出通道不变，但修改输入通道
            new_conv1 = nn.Conv2d(
                in_channels=in_channels,
                out_channels=64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

            # 初始化新卷积层的权重
            # 如果是单通道：用原始权重的均值复制到 3 通道
            # 如果是双通道：取前两通道的均值
            with torch.no_grad():
                if in_channels == 1:
                    # 单通道：取 RGB 权重的均值
                    new_conv1.weight[:, 0, :, :] = original_conv1.weight[:, 0, :, :].clone()
                    new_conv1.weight[:, 0, :, :] += original_conv1.weight[:, 1, :, :].clone()
                    new_conv1.weight[:, 0, :, :] += original_conv1.weight[:, 2, :, :].clone()
                    new_conv1.weight[:, 0, :, :] /= 3.0
                elif in_channels == 2:
                    # 双通道：取 RG 权重的均值
                    new_conv1.weight[:, 0, :, :] = original_conv1.weight[:, 0, :, :].clone()
                    new_conv1.weight[:, 1, :, :] = original_conv1.weight[:, 1, :, :].clone()
                else:
                    # 扩展到更多通道（不太可能用到）
                    for i in range(min(in_channels, 3)):
                        new_conv1.weight[:, i, :, :] = original_conv1.weight[:, i, :, :].clone()

            # 替换 conv1
            self.resnet.conv1 = new_conv1

        # -------------------------------------------------------------------------
        # 移除原始的全连接层（我们只需要特征）
        # -------------------------------------------------------------------------
        self.resnet.fc = nn.Identity()

        # 特征维度：ResNet-18 的 layer4 输出通道数为 512
        self.feature_dim = 512
        self.output_spatial = False  # 是否输出空间特征图

    def set_spatial_output(self, enabled=True):
        """切换是否输出空间特征图（供 DETR 等模型使用）"""
        self.output_spatial = enabled
        return self

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像张量，形状 (batch, in_channels, H, W)

        Returns:
            feat_2d: 2D 特征向量，形状 (batch, 512) 或 (batch, 512, H', W')
        """
        # 通过 ResNet 的各层
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        if self.output_spatial:
            # 输出空间特征图（供 DETR 使用）
            return x  # (B, 512, H', W')

        # 全局平均池化：将 (batch, 512, H, W) → (batch, 512, 1, 1) → (batch, 512)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        feat_2d = x.view(x.size(0), -1)

        return feat_2d


# =============================================================================
# 2D 骨干网络选项：ViT-Small
# =============================================================================

class ViTBackbone2D(nn.Module):
    """
    Vision Transformer Small for 2D 图像编码

    使用 timm 加载预训练的 ViT-Small，并适配 2 通道输入

    Args:
        image_size: 输入图像尺寸（默认256）
        in_channels: 输入通道数（默认2：温度+应力）
        pretrained: 是否使用 ImageNet 预训练权重
        embed_dim: ViT 嵌入维度（默认384 for ViT-Small）
    """

    def __init__(self, image_size=256, in_channels=2, pretrained=True, embed_dim=384):
        super().__init__()

        self.embed_dim = embed_dim
        self.image_size = image_size

        # 使用 timm 加载 ViT-Small
        try:
            import timm
            # 根据输入尺寸选择合适的模型
            if image_size == 224:
                model_name = 'vit_small_patch16_224'
            else:
                # 对于非标准尺寸，使用不检查尺寸的配置
                model_name = 'vit_small_patch16_224'

            self.vit = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=0,  # 移除分类头
                img_size=image_size,  # 指定图像尺寸
            )
        except ImportError:
            raise ImportError("timm 未安装，请运行: pip install timm")

        # 修改 patch embedding 以支持 2 通道输入
        # 方法：取预训练权重的均值复制到新通道
        with torch.no_grad():
            old_conv = self.vit.patch_embed.proj

            if in_channels == 2:
                # 取前两个通道的均值
                new_weight = old_conv.weight[:, :2, :, :].mean(dim=1, keepdim=True)
                new_weight = new_weight.repeat(1, 2, 1, 1) / 2
            elif in_channels == 1:
                # 取三个通道的均值复制到单通道
                new_weight = old_conv.weight.mean(dim=1, keepdim=True)
            else:
                new_weight = old_conv.weight[:, :in_channels, :, :]

            self.vit.patch_embed.proj = nn.Conv2d(
                in_channels, embed_dim,
                kernel_size=16, stride=16
            )
            self.vit.patch_embed.proj.weight.data = new_weight

        self.feature_dim = embed_dim

    def forward(self, x):
        """
        Args:
            x: (B, 2, 256, 256) 或 (B, in_channels, H, W)

        Returns:
            cls_token: (B, embed_dim)，取 [CLS] token
        """
        features = self.vit.forward_features(x)  # (B, N+1, embed_dim)
        cls_token = features[:, 0]  # 取 [CLS] token
        return cls_token


# =============================================================================
# 1D 骨干网络选项：DLinear
# =============================================================================

class DLinear1D(nn.Module):
    """
    DLinear: 简单的线性时序模型（轻量级基线）

    将序列分解为趋势和季节性分量，分别用线性层处理

    来自论文: "Are Transformers Effective for Time Series Forecasting?"
    (AAAI 2023)

    Args:
        seq_len: 序列长度
        feature_dim: 特征维度（默认1：单变量温度）
        output_dim: 输出特征维度
    """

    def __init__(self, seq_len=300, feature_dim=1, output_dim=64):
        super().__init__()
        self.seq_len = seq_len
        self.output_dim = output_dim

        # 趋势分解（较大 kernel 捕捉长程趋势）
        self.trend_conv = nn.Conv1d(feature_dim, 32, kernel_size=25, padding=12)
        # 季节性分解（较小 kernel 捕捉周期性）
        self.seasonal_conv = nn.Conv1d(feature_dim, 32, kernel_size=11, padding=5)

        self.fusion = nn.Sequential(
            nn.Linear(64, output_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: (B, seq_len) 或 (B, seq_len, 1)

        Returns:
            (B, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, seq_len, 1)

        x = x.transpose(1, 2)  # (B, 1, seq_len)

        trend = self.trend_conv(x)      # (B, 32, seq_len)
        seasonal = self.seasonal_conv(x)  # (B, 32, seq_len)

        # 全局池化
        trend = trend.mean(dim=-1)        # (B, 32)
        seasonal = seasonal.mean(dim=-1)  # (B, 32)

        fused = torch.cat([trend, seasonal], dim=-1)  # (B, 64)
        return self.fusion(fused)  # (B, output_dim)


# =============================================================================
# 1D 骨干网络选项：Transformer Encoder
# =============================================================================

class Transformer1D(nn.Module):
    """
    纯 Transformer 编码器用于 1D 时序

    使用标准 Transformer Encoder Layer 处理时序依赖

    Args:
        seq_len: 序列长度
        feature_dim: 输入特征维度
        d_model: Transformer 维度
        nhead: 注意力头数
        num_layers: Transformer 层数
        dropout: Dropout 比例
        output_dim: 输出特征维度
    """

    def __init__(self, seq_len=300, feature_dim=1, d_model=64, nhead=4,
                 num_layers=2, dropout=0.1, output_dim=64):
        super().__init__()

        self.input_proj = nn.Linear(feature_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, seq_len)

        Returns:
            (B, output_dim)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (B, seq_len, 1)

        x = self.input_proj(x)           # (B, seq_len, d_model)
        x = self.transformer(x)          # (B, seq_len, d_model)

        # 取所有 token 的平均
        x = x.mean(dim=1)                # (B, d_model)
        return self.output_proj(x)       # (B, output_dim)


# =============================================================================
# 融合策略选项：Adaptive Fusion
# =============================================================================

class AdaptiveFusion(nn.Module):
    """
    自适应加权融合：让模型学习 CNN 和 Transformer 特征的重要性

    使用可学习的权重对两个模态进行加权融合

    Args:
        dim_2d: 2D 特征维度
        dim_1d: 1D 特征维度
    """

    def __init__(self, dim_2d=512, dim_1d=64):
        super().__init__()

        # 可学习的权重
        self.weight_2d = nn.Parameter(torch.ones(1) * 0.5)
        self.weight_1d = nn.Parameter(torch.ones(1) * 0.5)

        # 投影层统一 1D 特征到 2D 维度
        self.proj_1d = nn.Linear(dim_1d, dim_2d)

        # 融合后的处理
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim_2d, dim_2d),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, feat_2d, feat_1d):
        """
        Args:
            feat_2d: (B, dim_2d)
            feat_1d: (B, dim_1d)

        Returns:
            (B, dim_2d)
        """
        feat_1d = self.proj_1d(feat_1d)  # (B, dim_2d)

        # 归一化权重（使用 sigmoid 确保在 0-1 之间）
        w_2d = torch.sigmoid(self.weight_2d)
        w_1d = torch.sigmoid(self.weight_1d)
        total = w_2d + w_1d + 1e-8
        w_2d, w_1d = w_2d / total, w_1d / total

        # 加权融合
        fused = w_2d * feat_2d + w_1d * feat_1d
        return self.fusion_layer(fused)  # (B, dim_2d)


# =============================================================================
# 分支 2：1D 时序特征提取（Micro + Macro 1D-CNN + 自注意力）
# =============================================================================

class Micro1DCNN(nn.Module):
    """
    微观 1D-CNN 分支

    负责捕捉瞬态温度跳变（高频变化）
    - kernel_size=3, dilation=1：局部感受野，捕捉细微变化
    - 两层卷积提取浅层局部特征
    """

    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()

        self.conv = nn.Sequential(
            # 第一层：kernel=3, dilation=1, padding=1（保持长度）
            nn.Conv1d(in_channels, out_channels, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),

            # 第二层：继续提取局部特征
            nn.Conv1d(out_channels, out_channels, kernel_size=3, dilation=1, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, seq_len)

        Returns:
            (batch, out_channels, seq_len)
        """
        return self.conv(x)


class Macro1DCNN(nn.Module):
    """
    宏观 1D-CNN 分支

    负责提取长程热耗散趋势
    - kernel_size=7, dilation=3, padding=9：更大感受野，捕捉长距离依赖
    - 两层卷积提取长程特征
    """

    def __init__(self, in_channels=1, out_channels=32):
        super().__init__()

        self.conv = nn.Sequential(
            # 第一层：kernel=7, dilation=3, padding=9（保持长度）
            nn.Conv1d(in_channels, out_channels, kernel_size=7, dilation=3, padding=9),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.1),

            # 第二层：继续提取长程特征
            nn.Conv1d(out_channels, out_channels, kernel_size=7, dilation=3, padding=9),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        """
        Args:
            x: (batch, in_channels, seq_len)

        Returns:
            (batch, out_channels, seq_len)
        """
        return self.conv(x)


class TemporalFeatureExtractor(nn.Module):
    """
    1D 时序特征提取器

    架构：Micro 1D-CNN + Macro 1D-CNN → Concat → LayerNorm → MultiHead Self-Attention

    设计参考了 v2 版本的 PETSNet 非对称双分支结构
    """

    def __init__(self, input_dim=1, hidden_dim=32, num_heads=4, dropout=0.1):
        """
        Args:
            input_dim: 输入通道数（默认 1：单变量温度序列）
            hidden_dim: 每个分支的输出通道数
            num_heads: 多头自注意力的头数
            dropout: Dropout 比例
        """
        super().__init__()

        self.hidden_dim = hidden_dim

        # 微观分支：捕捉瞬态跳变
        self.micro_branch = Micro1DCNN(in_channels=input_dim, out_channels=hidden_dim)

        # 宏观分支：提取长程趋势
        self.macro_branch = Macro1DCNN(in_channels=input_dim, out_channels=hidden_dim)

        # 特征融合层：双分支拼接后 (64 通道) → 64 通道
        self.fusion = nn.Sequential(
            nn.Conv1d(hidden_dim * 2, hidden_dim * 2, kernel_size=1),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # LayerNorm：沿着时间维度做归一化
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # 多头自注意力：捕捉时间步之间的依赖关系
        self.self_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # FFN（前馈网络）：Transformer 风格
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.Dropout(dropout),
        )

        # 最终 LayerNorm
        self.norm2 = nn.LayerNorm(hidden_dim * 2)

        # 全局池化：时间维度 → 1
        self.global_pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len) 或 (batch, seq_len, 1)

        Returns:
            feat_1d: 1D 特征向量，形状 (batch, hidden_dim * 2) = (batch, 64)
        """
        # 输入调整：(batch, seq_len) → (batch, 1, seq_len)
        if x.dim() == 2:
            x = x.unsqueeze(1)

        # 双分支并行卷积
        micro_feat = self.micro_branch(x)    # (batch, hidden_dim, seq_len)
        macro_feat = self.macro_branch(x)    # (batch, hidden_dim, seq_len)

        # 拼接双分支特征： (batch, hidden_dim*2, seq_len)
        fused = torch.cat([micro_feat, macro_feat], dim=1)

        # 特征融合
        fused = self.fusion(fused)           # (batch, hidden_dim*2, seq_len)

        # 准备自注意力输入：(batch, hidden_dim*2, seq_len) → (batch, seq_len, hidden_dim*2)
        fused_transposed = fused.transpose(1, 2)

        # LayerNorm
        fused_transposed = self.layer_norm(fused_transposed)

        # 多头自注意力（残差连接）
        attn_output, _ = self.self_attention(
            fused_transposed, fused_transposed, fused_transposed
        )
        fused = fused_transposed + attn_output  # 残差连接

        # FFN（残差连接）
        ffn_output = self.ffn(fused)
        fused = self.norm2(fused + ffn_output)

        # 全局池化：(batch, seq_len, hidden_dim*2) → (batch, hidden_dim*2)
        fused = fused.transpose(1, 2)        # → (batch, hidden_dim*2, seq_len)
        feat_1d = self.global_pool(fused).squeeze(-1)  # → (batch, hidden_dim*2)

        return feat_1d


# =============================================================================
# 融合层：Cross-Attention（交叉注意力）
# =============================================================================

class CrossAttentionFusion(nn.Module):
    """
    交叉注意力融合模块

    作用：让 2D 图像特征和 1D 时序特征互相"查询"，实现跨模态信息融合

    原理：
    - Q（查询）：一个模态的特征，用于"提问"
    - K（键）：另一个模态的特征，用于"匹配"
    - V（值）：另一个模态的特征，用于"传递信息"

    实现：双向交叉注意力
    - 图像特征查询温度时序
    - 温度时序查询图像特征
    """

    def __init__(self, dim_2d=512, dim_1d=64, num_heads=4, dropout=0.1):
        """
        Args:
            dim_2d: 2D 分支特征维度（默认 512）
            dim_1d: 1D 分支特征维度（默认 64）
            num_heads: 注意力头数
            dropout: Dropout 比例
        """
        super().__init__()

        self.dim_2d = dim_2d
        self.dim_1d = dim_1d
        self.num_heads = num_heads

        # 动态计算每个头的维度，确保能被 num_heads 整除
        # 如果 dim_2d 不能被 num_heads 整除，扩展到最近的倍数
        self.head_dim_2d = max(1, dim_2d // num_heads)
        self.head_dim_1d = max(1, dim_1d // num_heads)

        # 记录实际的投影维度（用于扩展/压缩）
        self.proj_dim_2d = self.head_dim_2d * num_heads
        self.proj_dim_1d = self.head_dim_1d * num_heads

        # 投影层（当维度不能整除头数时使用）
        if self.proj_dim_2d != dim_2d:
            self.proj_2d = nn.Linear(self.proj_dim_2d, dim_2d)
        else:
            self.proj_2d = nn.Identity()
        if self.proj_dim_1d != dim_1d:
            self.proj_1d = nn.Linear(self.proj_dim_1d, dim_1d)
        else:
            self.proj_1d = nn.Identity()

        # -------------------------------------------------------------------------
        # 图像 → 时序 的交叉注意力
        # -------------------------------------------------------------------------
        # Q: 2D 特征（图像），K/V: 1D 特征（时序）
        # 输出维度使用 proj_dim（适配多头注意力）
        self.q_linear_2d = nn.Linear(dim_2d, self.proj_dim_2d)
        self.k_linear_1d = nn.Linear(dim_1d, self.proj_dim_2d)
        self.v_linear_1d = nn.Linear(dim_1d, self.proj_dim_2d)

        # -------------------------------------------------------------------------
        # 时序 → 图像 的交叉注意力
        # -------------------------------------------------------------------------
        # Q: 1D 特征（时序），K/V: 2D 特征（图像）
        self.q_linear_1d = nn.Linear(dim_1d, self.proj_dim_1d)
        self.k_linear_2d = nn.Linear(dim_2d, self.proj_dim_1d)
        self.v_linear_2d = nn.Linear(dim_2d, self.proj_dim_1d)

        # -------------------------------------------------------------------------
        # 输出归一化
        # -------------------------------------------------------------------------
        self.norm_2d = nn.LayerNorm(dim_2d)
        self.norm_1d = nn.LayerNorm(dim_1d)

        # -------------------------------------------------------------------------
        # 融合层：将两个方向的输出拼接
        # -------------------------------------------------------------------------
        self.fusion = nn.Linear(dim_2d + dim_1d, dim_2d + dim_1d)

    def forward(self, feat_2d, feat_1d):
        """
        前向传播：双向交叉注意力融合

        Args:
            feat_2d: 2D 图像特征，(batch, dim_2d)
            feat_1d: 1D 时序特征，(batch, dim_1d)

        Returns:
            fused: 融合后的特征，(batch, dim_2d + dim_1d)
        """
        batch_size = feat_2d.size(0)

        # -------------------------------------------------------------------------
        # 方向 1：图像查询时序
        # feat_2d 作为 Q，feat_1d 作为 K 和 V
        # -------------------------------------------------------------------------
        q2 = self.q_linear_2d(feat_2d)                           # (batch, proj_dim_2d)
        k1 = self.k_linear_1d(feat_1d)                           # (batch, proj_dim_2d)
        v1 = self.v_linear_1d(feat_1d)                           # (batch, proj_dim_2d)

        # 多头注意力：reshape 为 (batch, num_heads, head_dim_2d)
        q2 = q2.view(batch_size, self.num_heads, self.head_dim_2d)
        k1 = k1.view(batch_size, self.num_heads, self.head_dim_2d)
        v1 = v1.view(batch_size, self.num_heads, self.head_dim_2d)

        # 注意力分数：Q · K^T / sqrt(d_k)
        attn_scores_2d_to_1d = torch.einsum('bhd,bhd->bh', q2, k1) / math.sqrt(self.head_dim_2d)
        attn_weights_2d_to_1d = F.softmax(attn_scores_2d_to_1d, dim=-1)

        # 加权求和：attn_weights · V
        attn_output_2d_to_1d = torch.einsum('bh,bhd->bhd', attn_weights_2d_to_1d, v1)
        attn_output_2d_to_1d = attn_output_2d_to_1d.contiguous().view(batch_size, self.proj_dim_2d)

        # 残差连接 + LayerNorm（投影到 dim_2d）
        attn_output_2d_to_1d = self.proj_2d(attn_output_2d_to_1d)
        fused_2d = self.norm_2d(feat_2d + attn_output_2d_to_1d)

        # -------------------------------------------------------------------------
        # 方向 2：时序查询图像
        # feat_1d 作为 Q，feat_2d 作为 K 和 V
        # -------------------------------------------------------------------------
        q1 = self.q_linear_1d(feat_1d)                           # (batch, proj_dim_1d)
        k2 = self.k_linear_2d(feat_2d)                           # (batch, proj_dim_1d)
        v2 = self.v_linear_2d(feat_2d)                           # (batch, proj_dim_1d)

        # 多头注意力
        q1 = q1.view(batch_size, self.num_heads, self.head_dim_1d)
        k2 = k2.view(batch_size, self.num_heads, self.head_dim_1d)
        v2 = v2.view(batch_size, self.num_heads, self.head_dim_1d)

        attn_scores_1d_to_2d = torch.einsum('bhd,bhd->bh', q1, k2) / math.sqrt(self.head_dim_1d)
        attn_weights_1d_to_2d = F.softmax(attn_scores_1d_to_2d, dim=-1)

        attn_output_1d_to_2d = torch.einsum('bh,bhd->bhd', attn_weights_1d_to_2d, v2)
        attn_output_1d_to_2d = attn_output_1d_to_2d.contiguous().view(batch_size, self.proj_dim_1d)

        # 残差连接 + LayerNorm（投影到 dim_1d）
        attn_output_1d_to_2d = self.proj_1d(attn_output_1d_to_2d)
        fused_1d = self.norm_1d(feat_1d + attn_output_1d_to_2d)

        # -------------------------------------------------------------------------
        # 融合两个方向的输出
        # -------------------------------------------------------------------------
        fused = torch.cat([fused_2d, fused_1d], dim=-1)
        fused = self.fusion(fused)

        return fused


# =============================================================================
# 门控多模态融合
# =============================================================================

class GatedMultimodalFusion(nn.Module):
    """门控多模态融合 - 温度/应力分治策略

    将2D特征按比例拆分为温度和应力两个分支，通过注意力加权后由时序特征门控融合

    ⚠️ 注意：此实现假设2D特征是按"温度、应力"顺序排列的。
    对于非标准骨干网络，建议使用 channel_split 参数指定具体拆分位置。
    """
    def __init__(self, dim_2d=512, dim_1d=64, split_ratio=0.5, channel_split=None):
        super().__init__()
        # 允许用户指定具体的拆分位置（优先级高于 split_ratio）
        if channel_split is not None:
            self.temp_channels = channel_split
        else:
            self.temp_channels = int(dim_2d * split_ratio)
        self.stress_channels = dim_2d - self.temp_channels

        # 温度分支注意力
        self.temp_attn = nn.Sequential(
            nn.Linear(self.temp_channels, max(1, self.temp_channels // 4)),
            nn.ReLU(),
            nn.Linear(max(1, self.temp_channels // 4), self.temp_channels),
            nn.Sigmoid()
        )

        # 应力分支注意力
        self.stress_attn = nn.Sequential(
            nn.Linear(self.stress_channels, max(1, self.stress_channels // 4)),
            nn.ReLU(),
            nn.Linear(max(1, self.stress_channels // 4), self.stress_channels),
            nn.Sigmoid()
        )

        # 门控网络
        self.gate = nn.Sequential(
            nn.Linear(dim_1d, max(1, dim_1d // 2)),
            nn.ReLU(),
            nn.Linear(max(1, dim_1d // 2), 2),
            nn.Softmax(dim=-1)
        )

        # 输出投影（支持拼接 1D 特征后的 dim_2d + dim_1d = 576 维）
        self.output_proj = nn.Linear(dim_2d + dim_1d, dim_2d + dim_1d)

    def forward(self, feat_2d, feat_1d):
        """
        Args:
            feat_2d: 2D图像特征 (B, dim_2d)
            feat_1d: 1D时序特征 (B, dim_1d)
        Returns:
            融合后的特征 (B, dim_2d + dim_1d) = (B, 576)
        """
        # 拆分温度和应力特征
        temp_feat = feat_2d[:, :self.temp_channels]
        stress_feat = feat_2d[:, self.temp_channels:]

        # 计算注意力权重
        temp_weight = self.temp_attn(temp_feat)
        stress_weight = self.stress_attn(stress_feat)

        # 门控权重
        gate = self.gate(feat_1d)  # (B, 2)

        # 加权融合
        temp_out = temp_feat * temp_weight * gate[:, 0:1]
        stress_out = stress_feat * stress_weight * gate[:, 1:2]

        # 融合温度/应力特征后拼接 1D 特征，再通过输出投影
        fused = torch.cat([temp_out, stress_out, feat_1d], dim=-1)  # (B, 512+64=576)
        output = self.output_proj(fused)  # (B, 576)

        return output


# =============================================================================
# 多任务输出头
# =============================================================================

class MultiTaskHead(nn.Module):
    """
    多任务输出头

    将融合特征解码为 6 个目标值：
    - [x, y, l, w]：裂纹边界框（位置和尺寸）
    - [confidence]：裂纹存在置信度 (0~1)
    - [density]：裂纹密度 (0~1)

    设计选择：
    - 位置 (x, y) 和尺寸 (l, w) 用 ReLU 确保非负
    - confidence 和 density 用 Sigmoid 激活到 [0,1]
    - 几何先验：l, w 通过 ReLU 确保不为负
    """

    def __init__(self, input_dim=576, hidden_dim=256, dropout=0.2):
        """
        Args:
            input_dim: 输入特征维度（512 + 64 = 576）
            hidden_dim: 隐藏层维度
            dropout: Dropout 比例
        """
        super().__init__()

        # 共享特征提取层
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # -------------------------------------------------------------------------
        # 独立的输出头
        # -------------------------------------------------------------------------
        # 位置预测头 (x, y)：无激活（ReLU 在 forward 中应用确保非负）
        self.head_xy = nn.Linear(hidden_dim // 2, 2)

        # 尺寸预测头 (l, w)：ReLU 确保非负
        self.head_lw = nn.Linear(hidden_dim // 2, 2)

        # 置信度预测头：Sigmoid 激活
        self.head_confidence = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        # 密度预测头：Sigmoid 激活
        self.head_density = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: 融合特征，(batch, input_dim)

        Returns:
            output: (batch, 6) -> [x, y, l, w, confidence, density]
        """
        # 共享特征提取
        shared_feat = self.shared(x)

        # 预测各分量
        xy = self.head_xy(shared_feat)                    # (batch, 2)
        lw = self.head_lw(shared_feat)                    # (batch, 2)
        confidence = self.head_confidence(shared_feat)    # (batch, 1)
        density = self.head_density(shared_feat)          # (batch, 1)

        # 确保几何量非负
        xy = torch.relu(xy)   # x, y 非负
        lw = torch.relu(lw)   # l, w 非负（裂纹尺寸不能为负）

        # 拼接所有输出
        output = torch.cat([xy, lw, confidence, density], dim=-1)  # (batch, 6)

        return output


class MaskDecoder(nn.Module):
    """
    掩膜解码器：将融合特征上采样回指定尺寸的二值掩膜

    架构：逐步转置卷积上采样 + 可选插值
    - 输入：(batch, 576) 融合特征
    - 内部固定上采样到 256×256
    - 如果 target_size > 256，使用插值resize到目标尺寸

    输出：(batch, 1, target_size, target_size) 二值掩膜
    """

    def __init__(self, in_channels=576, hidden_dim=128, target_size=256):
        super().__init__()

        self.target_size = target_size
        self.base_size = 256  # 内部固定输出256

        # 将特征reshape为 (batch, 576, 1, 1)
        self.reshape = lambda x: x.view(x.size(0), -1, 1, 1)

        # 逐步上采样：1→4→8→16→32→64→128→256
        self.up1 = nn.ConvTranspose2d(in_channels, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)

        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)

        self.up5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(16)

        self.up6 = nn.ConvTranspose2d(16, 8, kernel_size=4, stride=2, padding=1)
        self.bn6 = nn.BatchNorm2d(8)

        self.up7 = nn.ConvTranspose2d(8, 4, kernel_size=4, stride=2, padding=1)
        self.bn7 = nn.BatchNorm2d(4)

        self.up8 = nn.ConvTranspose2d(4, 1, kernel_size=4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (batch, 576) 融合特征

        Returns:
            mask: (batch, 1, target_size, target_size) 二值掩膜，值域 [0, 1]
        """
        x = self.reshape(x)  # (batch, 576, 1, 1)

        x = F.relu(self.bn1(self.up1(x)))   # (batch, 256, 2, 2)
        x = F.relu(self.bn2(self.up2(x)))   # (batch, 128, 4, 4)
        x = F.relu(self.bn3(self.up3(x)))   # (batch, 64, 8, 8)
        x = F.relu(self.bn4(self.up4(x)))   # (batch, 32, 16, 16)
        x = F.relu(self.bn5(self.up5(x)))   # (batch, 16, 32, 32)
        x = F.relu(self.bn6(self.up6(x)))   # (batch, 8, 64, 64)
        x = F.relu(self.bn7(self.up7(x)))   # (batch, 4, 128, 128)
        x = self.up8(x)                      # (batch, 1, 256, 256)

        mask = self.sigmoid(x)

        # 如果目标尺寸大于256，使用插值resize
        if self.target_size > self.base_size:
            mask = F.interpolate(
                mask,
                size=(self.target_size, self.target_size),
                mode='bilinear',
                align_corners=False
            )

        return mask


# =============================================================================
# 完整的多模态融合网络
# =============================================================================

class PETSNetMultimodal(nn.Module):
    """
    PE-MMNet v4: 多模态融合网络

    非对称双分支架构：
    1. 2D 分支：ResNet-18 视觉骨干 → 512 维特征
    2. 1D 分支：Micro + Macro 1D-CNN + 自注意力 → 64 维特征（不是 128！）

    融合：Cross-Attention → 576 维融合特征 → 多任务输出头

    支持三种任务模式：
    - detection: 输出 (batch, 6) [x, y, l, w, confidence, density]
    - segmentation: 输出 (batch, 1, image_size, image_size) 掩膜
    - multitask: 输出 (检测标签, 掩膜) 元组

    设计选择：
    - 1D 分支输出 64 维而非 128 维（平衡计算量和特征表达能力）
    - Cross-Attention 使用双向注意力权重
    - 输出层对几何量应用 ReLU，对概率量应用 Sigmoid
    """

    def __init__(self,
                 seq_len=300,
                 image_channels=2,
                 image_size=256,
                 pretrained_2d=True,
                 dropout=0.2,
                 task='detection',
                 fusion='cross_attn',
                 seq_channels=1):
        """
        Args:
            seq_len: 1D 序列长度（默认 300）
            image_channels: 2D 图像通道数（默认 2：温度场+应力场）
            image_size: 输出图像尺寸（默认 256，支持 512, 1024 等更高分辨率）
            pretrained_2d: 2D 分支是否使用预训练权重
            dropout: Dropout 比例
            task: 任务模式，'detection' | 'segmentation' | 'multitask'
            fusion: 融合策略，'cross_attn' | 'gated'
        """
        super().__init__()

        self.seq_len = seq_len
        self.task = task
        self.image_size = image_size
        self.fusion_type = fusion
        self.seq_channels = seq_channels

        # -------------------------------------------------------------------------
        # 分支 1：2D 视觉特征提取（512 维）
        # -------------------------------------------------------------------------
        self.branch_2d = ResNet18Backbone2D(
            in_channels=image_channels,
            pretrained=pretrained_2d
        )
        feat_dim_2d = 512

        # -------------------------------------------------------------------------
        # 分支 2：1D 时序特征提取（64 维）
        # -------------------------------------------------------------------------
        # hidden_dim=32 → 双分支 concat 后 64 维 → 全局池化后仍是 64 维
        self.branch_1d = TemporalFeatureExtractor(
            input_dim=seq_channels,  # 支持 triple_channel (3) 或单通道 (1)
            hidden_dim=32,           # 每个分支 32 通道
            num_heads=4,
            dropout=dropout
        )
        feat_dim_1d = 64

        # -------------------------------------------------------------------------
        # 融合层：根据 fusion 参数选择
        # -------------------------------------------------------------------------
        if fusion == 'gated':
            self.fusion = GatedMultimodalFusion(
                dim_2d=feat_dim_2d,
                dim_1d=feat_dim_1d,
                split_ratio=0.5
            )
        else:
            # 默认使用 Cross-Attention
            self.fusion = CrossAttentionFusion(
                dim_2d=feat_dim_2d,
                dim_1d=feat_dim_1d,
                num_heads=4,
                dropout=dropout
            )
        fused_dim = feat_dim_2d + feat_dim_1d  # 512 + 64 = 576

        # -------------------------------------------------------------------------
        # 检测输出头
        # -------------------------------------------------------------------------
        self.output_head = MultiTaskHead(
            input_dim=fused_dim,
            hidden_dim=256,
            dropout=dropout
        )

        # -------------------------------------------------------------------------
        # 分割输出头（按需创建）
        # -------------------------------------------------------------------------
        if task in ['segmentation', 'multitask']:
            self.mask_decoder = MaskDecoder(in_channels=fused_dim, target_size=image_size)
        else:
            self.mask_decoder = None

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化（跳过预训练层）"""
        for m in self.modules():
            # 跳过预训练的 ResNet 层（已从 ImageNet 加载权重）
            # 这些层通过 weights='IMAGENET1K_V1' 加载，无需重新初始化
            if hasattr(m, 'weight') and m.weight is not None:
                # 检查是否是 ResNet 的预训练层（3通道输入的原始 conv1）
                if isinstance(m, nn.Conv2d) and m.weight.shape[1] == 3:
                    continue  # 跳过 ImageNet 预训练的原始 conv1
                # 检查是否是 BatchNorm 预训练层
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                    # 如果已经有非默认初始化，跳过
                    if torch.all(m.weight.data != 1.0):
                        continue
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                    if hasattr(m, 'running_mean'):
                        nn.init.zeros_(m.running_mean)
                        nn.init.ones_(m.running_var)
                    continue

            if isinstance(m, nn.Conv2d):
                # 仅初始化新添加的层（单/双通道输入的层）
                if m.weight.shape[1] != 3:  # 不是原始 ImageNet conv1
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_1d, x_2d):
        """
        前向传播

        Args:
            x_1d: 1D 温度时序，(batch, 300)
            x_2d: 2D 图像，(batch, 2, 256, 256)

        Returns:
            detection: (batch, 6) -> [x, y, l, w, confidence, density]
            segmentation: (batch, 1, 256, 256) 掩膜
            multitask: (mask, detection) 元组
        """
        # 分支 1：2D 图像特征
        feat_2d = self.branch_2d(x_2d)          # (batch, 512)

        # 分支 2：1D 时序特征
        feat_1d = self.branch_1d(x_1d)          # (batch, 64)

        # 融合：根据 fusion 类型选择
        fused = self.fusion(feat_2d, feat_1d)  # (batch, 576)

        if self.task == 'detection':
            # 检测任务：输出 6 维向量
            output = self.output_head(fused)    # (batch, 6)
            return output

        elif self.task == 'segmentation':
            # 分割任务：输出掩膜
            mask = self.mask_decoder(fused)     # (batch, 1, 256, 256)
            return mask

        else:  # multitask
            # 多任务：输出 (掩膜, 检测标签) 元组
            detection = self.output_head(fused)  # (batch, 6)
            mask = self.mask_decoder(fused)      # (batch, 1, 256, 256)
            return mask, detection

    def extract_features(self, x_1d, x_2d):
        """
        提取中间特征（用于分析可视化）

        Args:
            x_1d: 1D 温度时序
            x_2d: 2D 图像

        Returns:
            dict: 包含各层特征的字典
        """
        feat_2d = self.branch_2d(x_2d)
        feat_1d = self.branch_1d(x_1d)
        fused = self.fusion(feat_2d, feat_1d)

        return {
            'feat_2d': feat_2d,
            'feat_1d': feat_1d,
            'fused': fused
        }

    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 轻量版（用于边缘部署）
# =============================================================================

class PETSNetMultimodalSmall(PETSNetMultimodal):
    """
    轻量版 PE-MMNet v4

    用于边缘设备部署，减少参数量：
    - 2D 分支：使用更小的模型（减少特征维度）
    - 1D 分支：减少隐藏层维度
    - 输出头：减少隐藏层大小
    """

    def __init__(self, seq_len=300, dropout=0.2):
        super().__init__(
            seq_len=seq_len,
            image_channels=2,
            pretrained_2d=False,  # 轻量版不使用预训练
            dropout=dropout
        )

        # 替换为更小的 2D 骨干（简化版 ResNet）
        self.branch_2d = Lightweight2DEncoder()

        # 替换为更小的 1D 特征提取器
        self.branch_1d = Lightweight1DEncoder(seq_len=seq_len)


class Lightweight2DEncoder(nn.Module):
    """
    轻量级 2D 编码器

    简化版：3 层卷积 + 全局池化，替代 ResNet-18
    输出 256 维特征
    """

    def __init__(self, in_channels=2):
        super().__init__()

        self.conv = nn.Sequential(
            # Layer 1: (batch, 2, 256, 256) → (batch, 32, 128, 128)
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 2: → (batch, 64, 64, 64)
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            # Layer 3: → (batch, 128, 32, 32)
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            # Layer 4: → (batch, 256, 16, 16)
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 256

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class Lightweight1DEncoder(nn.Module):
    """
    轻量级 1D 编码器

    简化版：单层 1D-CNN + 全局池化，替代复杂的双分支结构
    输出 64 维特征
    """

    def __init__(self, seq_len=300):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.feature_dim = 64

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.pool(x)
        return x.squeeze(-1)


# =============================================================================
# 可配置多模态网络
# =============================================================================

class ConfigurableMultimodal(nn.Module):
    """
    可配置的多模态融合网络

    支持不同的 2D 骨干、1D 骨干和融合策略的组合

    Args:
        backbone_2d: 2D 骨干网络模块
        backbone_1d: 1D 骨干网络模块
        fusion: 融合层模块（concat 模式为 None）
        fusion_type: 融合类型，'cross_attn' | 'concat' | 'adaptive'
        dim_2d: 2D 特征维度
        dim_1d: 1D 特征维度
        fused_dim: 融合后特征维度
        dropout: Dropout 比例
    """

    def __init__(self, backbone_2d, backbone_1d, fusion, fusion_type,
                 dim_2d, dim_1d, fused_dim, dropout):
        super().__init__()

        self.backbone_2d = backbone_2d
        self.backbone_1d = backbone_1d
        self.fusion = fusion
        self.fusion_type = fusion_type

        self.output_head = MultiTaskHead(
            input_dim=fused_dim,
            hidden_dim=256,
            dropout=dropout
        )

        # 权重初始化
        self._init_weights()

    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_1d, x_2d):
        """
        Args:
            x_1d: (batch, seq_len)
            x_2d: (batch, 2, 256, 256)

        Returns:
            output: (batch, 6)
        """
        feat_2d = self.backbone_2d(x_2d)
        feat_1d = self.backbone_1d(x_1d)

        if self.fusion_type == 'concat':
            fused = torch.cat([feat_2d, feat_1d], dim=-1)
        else:
            fused = self.fusion(feat_2d, feat_1d)

        return self.output_head(fused)

    def extract_features(self, x_1d, x_2d):
        """提取中间特征"""
        feat_2d = self.backbone_2d(x_2d)
        feat_1d = self.backbone_1d(x_1d)

        if self.fusion_type == 'concat':
            fused = torch.cat([feat_2d, feat_1d], dim=-1)
        else:
            fused = self.fusion(feat_2d, feat_1d)

        return {
            'feat_2d': feat_2d,
            'feat_1d': feat_1d,
            'fused': fused
        }

    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# 模型工厂函数
# =============================================================================

def create_model(
    backbone_2d='resnet18',      # resnet18 | vit_small
    backbone_1d='cnn_attn',       # cnn_attn | transformer | dlinear
    fusion='cross_attn',           # cross_attn | concat | adaptive
    seq_len=300,
    dropout=0.2,
    pretrained_2d=True,
):
    """
    统一的模型工厂函数

    支持的配置组合：
    - backbone_2d: 'resnet18', 'vit_small'
    - backbone_1d: 'cnn_attn', 'transformer', 'dlinear'
    - fusion: 'cross_attn', 'concat', 'adaptive'

    Args:
        backbone_2d: 2D 骨干网络类型
        backbone_1d: 1D 骨干网络类型
        fusion: 融合策略
        seq_len: 序列长度
        dropout: Dropout 比例
        pretrained_2d: 2D 骨干是否使用预训练权重

    Returns:
        nn.Module: 配置好的模型
    """
    # -------------------------------------------------------------------------
    # 1. 创建 2D 骨干
    # -------------------------------------------------------------------------
    # 默认图像尺寸
    image_size = 256

    if backbone_2d == 'resnet18':
        backbone_2d_module = ResNet18Backbone2D(in_channels=2, pretrained=pretrained_2d)
        dim_2d = 512
    elif backbone_2d == 'vit_small':
        backbone_2d_module = ViTBackbone2D(
            image_size=image_size, in_channels=2,
            pretrained=pretrained_2d, embed_dim=384
        )
        dim_2d = 384
    else:
        raise ValueError(f"Unknown 2D backbone: {backbone_2d}. "
                        f"Supported: resnet18, vit_small")

    # -------------------------------------------------------------------------
    # 2. 创建 1D 骨干
    # -------------------------------------------------------------------------
    if backbone_1d == 'cnn_attn':
        backbone_1d_module = TemporalFeatureExtractor(
            input_dim=1, hidden_dim=32, num_heads=4, dropout=dropout
        )
        dim_1d = 64
    elif backbone_1d == 'transformer':
        backbone_1d_module = Transformer1D(
            seq_len=seq_len, feature_dim=1, d_model=64, nhead=4,
            num_layers=2, dropout=dropout, output_dim=64
        )
        dim_1d = 64
    elif backbone_1d == 'dlinear':
        backbone_1d_module = DLinear1D(
            seq_len=seq_len, feature_dim=1, output_dim=64
        )
        dim_1d = 64
    else:
        raise ValueError(f"Unknown 1D backbone: {backbone_1d}. "
                        f"Supported: cnn_attn, transformer, dlinear")

    # -------------------------------------------------------------------------
    # 3. 创建融合层
    # -------------------------------------------------------------------------
    if fusion == 'cross_attn':
        fusion_module = CrossAttentionFusion(
            dim_2d=dim_2d, dim_1d=dim_1d, num_heads=4, dropout=dropout
        )
        fused_dim = dim_2d + dim_1d  # 拼接后维度
    elif fusion == 'concat':
        fusion_module = None  # 简单拼接在 forward 中处理
        fused_dim = dim_2d + dim_1d
    elif fusion == 'adaptive':
        fusion_module = AdaptiveFusion(dim_2d=dim_2d, dim_1d=dim_1d)
        fused_dim = dim_2d  # 自适应融合后维度
    else:
        raise ValueError(f"Unknown fusion: {fusion}. "
                        f"Supported: cross_attn, concat, adaptive")

    # -------------------------------------------------------------------------
    # 4. 创建主模型
    # -------------------------------------------------------------------------
    model = ConfigurableMultimodal(
        backbone_2d=backbone_2d_module,
        backbone_1d=backbone_1d_module,
        fusion=fusion_module,
        fusion_type=fusion,
        dim_2d=dim_2d,
        dim_1d=dim_1d,
        fused_dim=fused_dim,
        dropout=dropout,
    )

    return model


def get_arch_specific_config(backbone_2d, backbone_1d, args_lr=None, args_dropout=None):
    """
    根据架构自动配置学习率和 dropout

    Args:
        backbone_2d: 2D 骨干类型
        backbone_1d: 1D 骨干类型
        args_lr: 命令行传入的学习率（可选）
        args_dropout: 命令行传入的 dropout（可选）

    Returns:
        dict: 包含 lr 和 dropout 的配置
    """
    # 默认值
    lr = args_lr if args_lr is not None else 1e-3
    dropout = args_dropout if args_dropout is not None else 0.2

    # Transformer 类架构需要更小的学习率和更高的 dropout
    if backbone_2d == 'vit_small':
        lr = min(lr, 1e-4)  # ViT 需要较小的学习率
        dropout = max(dropout, 0.2)

    if backbone_1d in ['transformer', 'dlinear']:
        lr = min(lr, 1e-4)  # Transformer 类统一用较小学习率
        dropout = max(dropout, 0.2)

    # Transformer + Transformer 组合使用更高的 dropout
    if backbone_2d == 'vit_small' and backbone_1d == 'transformer':
        dropout = max(dropout, 0.3)

    return {'lr': lr, 'dropout': dropout}
