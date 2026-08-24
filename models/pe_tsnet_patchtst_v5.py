"""
PE-MMNet v5-α: Swin-YOLO + PatchTST 1D 骨干网络（V5A4）

与 SwinYOLOFPNV5 的区别：
- 1D 骨干使用 PatchTST1D（v4 沿用）而非 TemporalFeatureExtractor
- 架构：Swin-Tiny + YOLO-FPN + PatchTST-1D + GP + SplineHead

v5-α 改动：
- 输出头：MultiTaskHead → SplineHead
- 物理约束：可选 GP 嵌入模块
- 输入通道：v4 (2 通道) → v5-α (3 通道)

详见：
- docs/v5_架构设计.md §2.4
- tasks/team_v5_alpha.json 中 V5A4
"""

from __future__ import annotations

import torch
import torch.nn as nn

# 复用 v4 模块
from models.pe_tsnet_yolo import SwinBackbone2D, YOLOFPNHead
from models.pe_tsnet_patchtst import PatchTST1D
from models.spline_head import SplineHead
from models.gp_module import GPModule


# 检测 PatchTST 是否可用（实际 v4 中总是可用，但保持与 v4 一致）
PATCHTST_AVAILABLE = True
try:
    from models.pe_tsnet_patchtst import PatchTST1D as _PatchTST1D_check  # noqa: F401
except ImportError:
    PATCHTST_AVAILABLE = False


class SwinYOLOFPNWithPatchTSTV5(nn.Module):
    """
    Swin-YOLO-FPN + PatchTST 1D 骨干 + SplineHead v5-α

    输入：
      - x_1d: (B, seq_len) 温度时序
      - x_2d: (B, image_channels, H, W) 多模态图像

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
        patch_size: int = 10,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
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
        self.grid_size = grid_size
        self.max_kpts = max_kpts
        self.min_kpts = min_kpts
        self.use_gp = use_gp

        # 2D 骨干：Swin-Tiny
        self.backbone_2d = SwinBackbone2D(
            pretrained=pretrained_2d,
            img_size=image_size,
            in_channels=image_channels,
        )
        self.yolo_proj = nn.Sequential(
            nn.Conv2d(768, 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.yolo_head = YOLOFPNHead(in_channels=256, grid_size=grid_size)

        # 1D 骨干：PatchTST
        self.backbone_1d = PatchTST1D(
            seq_len=seq_len,
            patch_size=patch_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            output_dim=feat_dim_1d,
        )

        # 拼接维度
        fused_dim = 6 + feat_dim_1d

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
        multi_scale_features = self.backbone_2d(x_2d)
        feat = multi_scale_features[-1]  # (B, 768, ...)
        feat = self.yolo_proj(feat)      # (B, 256, ...)
        grid_pred = self.yolo_head([feat])  # (B, num_grids, 6)

        # 全局池化
        grid_feat = grid_pred.mean(dim=1)  # (B, 6)

        # 1D 分支
        feat_1d = self.backbone_1d(x_1d)  # (B, feat_dim_1d)

        # 拼接
        fused = torch.cat([grid_feat, feat_1d], dim=-1)

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
    print("SwinYOLOFPNWithPatchTSTV5 冒烟测试")
    print("=" * 60)
    torch.manual_seed(0)
    model = SwinYOLOFPNWithPatchTSTV5(
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