"""
PE-MMNet v5-α: YOLO 风格多模态网络（V5A4）

包含：
- SwinYOLOFPNV5 — Swin-Tiny + YOLO-FPN + SplineHead
- ViTYOLOFPNV5  — ViT-Small + YOLO-FPN + SplineHead

v5-α 改动：
- 输出头：YOLOFPNHead (网格 6 维) + MultiTaskHead (6 维) → SplineHead (bbox + 8-16 关键点)
- 物理约束：可选 GP 嵌入模块（默认启用）
- 输入通道：v4 (2 通道) → v5-α (3 通道，温度场+应力场+热力图)

详见：
- docs/v5_架构设计.md §2.4
- tasks/team_v5_alpha.json 中 V5A4
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用 v4 的骨干和工具
from models.pe_tsnet_multimodal import (
    TemporalFeatureExtractor,
    CrossAttentionFusion,
)
from models.pe_tsnet_yolo import (
    SwinBackbone2D,
    YOLOFPNHead,
)
from models.spline_head import SplineHead
from models.gp_module import GPModule


# =============================================================================
#  ViT-YOLO 骨干的 v5 适配版（支持灵活输入通道）
# =============================================================================

class ViTYOLOBackbone2DV5(nn.Module):
    """
    ViT-Small + YOLO 骨干（v5-α 自适应输入通道）

    与 v4 的 ViTYOLOBackbone2D 区别：
    - v4 硬编码 `Conv2d(2, 3)` 输入适配
    - v5-α 支持 1/2/3/4 通道输入（通过自适应 1×1 conv 适配到 3 通道）

    输出：(B, 256, output_size, output_size)
    """

    def __init__(
        self,
        image_channels: int = 3,
        pretrained: bool = True,
        output_size: int = 16,
        input_size: int = 224,
    ):
        super().__init__()
        if image_channels not in (1, 2, 3, 4):
            raise ValueError(
                f"image_channels 应 ∈ {{1,2,3,4}}，实际 {image_channels}"
            )

        try:
            import timm
        except ImportError:
            raise ImportError("timm 未安装，请运行: pip install timm>=0.9.0")

        self.output_size = output_size
        self.input_size = input_size

        # 自适应输入通道适配（与 SwinBackbone2D 同款）
        self.input_resize = nn.Sequential(
            nn.Conv2d(image_channels, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )

        # ViT-Small 骨干
        self.vit = timm.create_model(
            'vit_small_patch16_224',
            pretrained=pretrained,
            features_only=True,
            img_size=input_size,
        )
        self.out_channels = self.vit.feature_info.channels()[-1]  # 384

        # 投影到统一通道数
        self.proj = nn.Sequential(
            nn.Conv2d(self.out_channels, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, image_channels, H, W)

        Returns:
            feat: (B, 256, output_size, output_size)
        """
        # resize 到 ViT 期望尺寸
        B, C, H, W = x.shape
        if H != self.input_size or W != self.input_size:
            x = F.interpolate(
                x, size=(self.input_size, self.input_size),
                mode='bilinear', align_corners=False,
            )

        # 自适应通道适配
        x = self.input_resize(x)  # (B, 3, input_size, input_size)

        # ViT 特征
        features = self.vit(x)
        feat = features[-1]

        # 上/下采样到目标尺寸
        feat = F.interpolate(
            feat, size=(self.output_size, self.output_size),
            mode='bilinear', align_corners=False,
        )

        # 投影
        feat = self.proj(feat)
        return feat


# =============================================================================
#  Swin-YOLO-FPN v5-α
# =============================================================================

class SwinYOLOFPNV5(nn.Module):
    """
    Swin-Tiny + YOLO-FPN + SplineHead v5-α 模型

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
        grid_size: int = 16,
        feat_dim_1d: int = 64,
        max_kpts: int = 16,
        min_kpts: int = 8,
        use_gp: bool = True,
        gp_hidden_dim: int = 128,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.image_channels = image_channels
        self.image_size = image_size
        self.grid_size = grid_size
        self.max_kpts = max_kpts
        self.min_kpts = min_kpts
        self.use_gp = use_gp

        # 2D 骨干：Swin-Tiny（v4 沿用，原生支持灵活 in_channels）
        self.backbone_2d = SwinBackbone2D(
            pretrained=pretrained_2d,
            img_size=image_size,
            in_channels=image_channels,
        )
        # Swin 最后一层输出 768 通道
        self.yolo_proj = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.yolo_head = YOLOFPNHead(in_channels=256, grid_size=grid_size)

        # 1D 骨干（v4 沿用）
        self.backbone_1d = TemporalFeatureExtractor(
            input_dim=1,
            hidden_dim=32,
            num_heads=4,
            dropout=dropout,
        )

        # 全局池化 YOLO 输出到 fused_dim（替换 v4 的 fusion+output_head）
        # YOLO 输出 (B, num_grids, 6) → 全局平均池化 → (B, 6)
        # 然后与 1D 特征拼接 → GP → SplineHead
        fused_dim = 6 + feat_dim_1d  # 6 + 64 = 70

        # GP 嵌入模块（v5 新增）
        if use_gp:
            self.gp_module = GPModule(
                in_dim=fused_dim,
                hidden_dim=gp_hidden_dim,
                init_length_scale=1.0,
                init_signal_var=1.0,
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
        multi_scale_features = self.backbone_2d(x_2d)
        feat = multi_scale_features[-1]  # (B, 768, 16, 16)
        feat = self.yolo_proj(feat)      # (B, 256, 16, 16)
        grid_pred = self.yolo_head([feat])  # (B, num_grids, 6)

        # 全局池化 YOLO 输出
        grid_feat = grid_pred.mean(dim=1)  # (B, 6)

        # 1D 分支
        feat_1d = self.backbone_1d(x_1d)  # (B, 64)

        # 拼接
        fused = torch.cat([grid_feat, feat_1d], dim=-1)  # (B, 70)

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
#  ViT-YOLO-FPN v5-α
# =============================================================================

class ViTYOLOFPNV5(nn.Module):
    """
    ViT-Small + YOLO-FPN + SplineHead v5-α 模型
    """

    def __init__(
        self,
        seq_len: int = 300,
        image_channels: int = 3,
        image_size: int = 256,
        pretrained_2d: bool = True,
        dropout: float = 0.2,
        grid_size: int = 16,
        feat_dim_1d: int = 64,
        max_kpts: int = 16,
        min_kpts: int = 8,
        use_gp: bool = True,
        gp_hidden_dim: int = 128,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.image_channels = image_channels
        self.image_size = image_size
        self.grid_size = grid_size
        self.max_kpts = max_kpts
        self.min_kpts = min_kpts
        self.use_gp = use_gp

        # 动态计算网格尺寸
        computed_grid = image_size // 32
        self.actual_grid_size = max(16, (computed_grid // 4) * 4)

        # 2D 骨干：ViT-Small（v5-α 自适应版）
        self.backbone_2d = ViTYOLOBackbone2DV5(
            image_channels=image_channels,
            pretrained=pretrained_2d,
            output_size=self.actual_grid_size,
        )

        # 简化 FPN
        self.fpn = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.yolo_head = YOLOFPNHead(in_channels=256, grid_size=grid_size)

        # 1D 骨干
        self.backbone_1d = TemporalFeatureExtractor(
            input_dim=1,
            hidden_dim=32,
            num_heads=4,
            dropout=dropout,
        )

        # 拼接
        fused_dim = 6 + feat_dim_1d  # 70

        # GP 嵌入模块
        if use_gp:
            self.gp_module = GPModule(
                in_dim=fused_dim,
                hidden_dim=gp_hidden_dim,
                learnable=True,
            )

        # 输出头
        self.output_head = SplineHead(
            in_dim=fused_dim,
            hidden_dim=256,
            max_kpts=max_kpts,
            min_kpts=min_kpts,
            dropout=dropout,
        )

    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor) -> dict:
        # 2D 分支
        feat_2d = self.backbone_2d(x_2d)        # (B, 256, grid_size, grid_size)
        fpn_feat = self.fpn(feat_2d)            # (B, 256, grid_size, grid_size)
        grid_pred = self.yolo_head([fpn_feat])   # (B, num_grids, 6)

        # 全局池化
        grid_feat = grid_pred.mean(dim=1)        # (B, 6)

        # 1D 分支
        feat_1d = self.backbone_1d(x_1d)         # (B, 64)

        # 拼接
        fused = torch.cat([grid_feat, feat_1d], dim=-1)  # (B, 70)

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

def _smoke_test_swin():
    print("=" * 60)
    print("SwinYOLOFPNV5 冒烟测试")
    print("=" * 60)
    torch.manual_seed(0)
    model = SwinYOLOFPNV5(
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
    print(f"  [PASS] ✅\n")


def _smoke_test_vit():
    print("=" * 60)
    print("ViTYOLOFPNV5 冒烟测试")
    print("=" * 60)
    torch.manual_seed(0)
    model = ViTYOLOFPNV5(
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
    print(f"  [PASS] ✅\n")


if __name__ == "__main__":
    _smoke_test_swin()
    _smoke_test_vit()