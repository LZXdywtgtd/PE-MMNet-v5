"""
PE-MMNet v5-α: DETR 风格多模态网络（V5A4）

v5-α 改动：
- 输出头：DETRHead (网格 6 维) + MultiTaskHead → SplineHead
- 物理约束：可选 GP 嵌入模块（默认启用）
- 输入通道：v4 (2 通道) → v5-α (3 通道)

DETR 风格的关键设计：
- Transformer decoder 的 (B, num_queries, d_model) 输出需要聚合到 (B, d_model)
- 策略：取最高 confidence 的 query 作为代表（沿用 v4 推理逻辑）
- v5-α 训练时也使用同一 query（保持 train/eval 一致）

详见：
- docs/v5_架构设计.md §2.4
- tasks/team_v5_alpha.json 中 V5A4
"""

from __future__ import annotations

import torch
import torch.nn as nn

# 复用 v4 的 DETR 模块和工具
from models.pe_tsnet_multimodal import (
    ResNet18Backbone2D,
    TemporalFeatureExtractor,
    CrossAttentionFusion,
)
from models.pe_tsnet_detr import (
    DETRDecoder,
    DETRHead,
    PositionalEncoding2D,
)
from models.spline_head import SplineHead
from models.gp_module import GPModule


class DETRStyleV5(nn.Module):
    """
    DETR + SplineHead v5-α 模型

    架构：
      ResNet-18 → proj → pos_enc → Transformer encoder → DETR decoder
      → DETRHead (num_queries × 6) → 取 top-1 → concat 1D → GP → SplineHead

    输入：
      - x_1d: (B, seq_len) 温度时序
      - x_2d: (B, image_channels, H, W) 多模态图像（v5-α 默认 3 通道）

    输出：dict {bbox, keypoints, validity, K, gp_lml?}
    """

    def __init__(
        self,
        seq_len: int = 300,
        image_channels: int = 3,
        image_size: int = 256,
        pretrained_2d: bool = True,
        dropout: float = 0.2,
        d_model: int = 512,
        num_queries: int = 100,
        encoder_layers: int = 6,
        decoder_layers: int = 6,
        feat_dim_1d: int = 64,
        max_kpts: int = 16,
        min_kpts: int = 8,
        use_gp: bool = True,
        gp_hidden_dim: int = 128,
    ):
        super().__init__()
        if image_channels not in (1, 2, 3, 4):
            raise ValueError(
                f"image_channels 应 ∈ {{1,2,3,4}}，实际 {image_channels}"
            )

        self.seq_len = seq_len
        self.image_channels = image_channels
        self.image_size = image_size
        self.d_model = d_model
        self.num_queries = num_queries
        self.max_kpts = max_kpts
        self.min_kpts = min_kpts
        self.use_gp = use_gp

        # 特征图尺寸（ResNet-18 stride=32）
        self.feat_size = image_size // 32

        # 2D 骨干：ResNet-18（v4 沿用，支持灵活 in_channels）
        self.backbone_2d = ResNet18Backbone2D(
            in_channels=image_channels,
            pretrained=pretrained_2d,
        )
        self.backbone_2d.set_spatial_output(True)  # 输出 (B, 512, 8, 8)

        # 投影
        self.input_proj = nn.Conv2d(512, d_model, kernel_size=1)

        # 2D 位置编码
        self.pos_encoder = PositionalEncoding2D(
            d_model, self.feat_size, self.feat_size
        )

        # Transformer 编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=encoder_layers,
            enable_nested_tensor=False,
        )

        # Transformer 解码器
        self.decoder = DETRDecoder(
            d_model=d_model,
            nhead=8,
            num_decoder_layers=decoder_layers,
            num_queries=num_queries,
            dropout=dropout,
        )

        # DETR 预测头（v4 沿用，输出 (B, num_queries, 6)）
        self.detr_head = DETRHead(d_model=d_model)

        # 1D 骨干
        self.backbone_1d = TemporalFeatureExtractor(
            input_dim=1,
            hidden_dim=32,
            num_heads=4,
            dropout=dropout,
        )

        # Query 投影 + 拼接（与 v4 推理链路对齐）
        # query 6维 → 128维 → 与 1D (64) 拼接 → fused
        self.query_proj = nn.Linear(6, 128)
        fused_dim = 128 + feat_dim_1d  # 192

        # GP 嵌入模块
        if use_gp:
            self.gp_module = GPModule(
                in_dim=fused_dim,
                hidden_dim=gp_hidden_dim,
                learnable=True,
            )

        # 输出头：SplineHead
        self.output_head = SplineHead(
            in_dim=fused_dim,
            hidden_dim=256,
            max_kpts=max_kpts,
            min_kpts=min_kpts,
            dropout=dropout,
        )

    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor) -> dict:
        # 2D 分支
        feat_2d = self.backbone_2d(x_2d)        # (B, 512, H', W')
        feat_2d = self.input_proj(feat_2d)       # (B, d_model, H', W')
        feat_2d = self.pos_encoder(feat_2d)      # (B, d_model, H', W')

        # 展平 + Transformer
        memory = feat_2d.flatten(2).transpose(1, 2)  # (B, H'*W', d_model)
        memory = self.transformer_encoder(memory)    # (B, H'*W', d_model)

        # DETR 解码 + 头
        decoder_out = self.decoder(memory)            # (B, num_queries, d_model)
        detr_pred = self.detr_head(decoder_out)       # (B, num_queries, 6)

        # 取最高 conf 的 query（与 v4 推理链路一致）
        conf = detr_pred[..., 4:5]  # (B, num_queries, 1)
        best_idx = conf.squeeze(-1).argmax(dim=1)  # (B,)
        B = detr_pred.size(0)
        best_query = detr_pred[torch.arange(B, device=detr_pred.device), best_idx]  # (B, 6)

        # 1D 分支
        feat_1d = self.backbone_1d(x_1d)  # (B, 64)

        # query 投影 + 拼接
        query_feat = self.query_proj(best_query)  # (B, 128)
        fused = torch.cat([query_feat, feat_1d], dim=-1)  # (B, 192)

        # GP 调制
        if self.gp_module is not None:
            fused = self.gp_module(fused)

        # 输出头
        out = self.output_head(fused)

        if self.gp_module is not None:
            out["gp_lml"] = self.gp_module.log_marginal_likelihood()

        return out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
#  冒烟测试
# =============================================================================

def _smoke_test():
    print("=" * 60)
    print("DETRStyleV5 冒烟测试")
    print("=" * 60)
    torch.manual_seed(0)
    model = DETRStyleV5(
        seq_len=300,
        image_channels=3,
        pretrained_2d=False,
        use_gp=True,
    )
    print(f"[OK] 初始化: {model.count_parameters():,} 参数")

    B = 2
    x_1d = torch.randn(B, 300)
    x_2d = torch.randn(B, 3, 256, 256)

    out = model(x_1d, x_2d)
    print(f"  bbox={tuple(out['bbox'].shape)}, "
          f"kpts={tuple(out['keypoints'].shape)}, "
          f"K={out['K'].tolist()}")
    assert out["bbox"].shape == (B, 4)
    assert out["keypoints"].shape == (B, model.max_kpts, 2)

    # 反向传播
    out["bbox"].sum().backward()
    print(f"  [PASS] ✅")


if __name__ == "__main__":
    _smoke_test()