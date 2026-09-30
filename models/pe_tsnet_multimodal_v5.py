"""
PE-MMNet v5-α: ResNet-18 多模态融合网络（V5A4）

v5-α 改动的核心：
- 输出头：MultiTaskHead (6 维) → SplineHead (bbox + 8-16 动态关键点)
- 物理约束：可选 GP 嵌入模块（默认启用）
- 输入通道：2 (温度+应力) → 3 (温度场+应力场+热力图)，符合 v5-α 设计

继承策略：
- **复用** v4 的 `ResNet18Backbone2D`、`TemporalFeatureExtractor`、`CrossAttentionFusion`
- **替换** v4 的 `MultiTaskHead` 为 v5 的 `SplineHead` + 可选 `GPModule`

输入：
  - x_1d: (B, 300) 温度时序（v4 沿用）
  - x_2d: (B, 3, 256, 256) — 温度场 + 应力场 + 热力图（v5-α 3 模态）

输出：
  - dict:
      - bbox: (B, 4) — [x1, y1, x2, y2]
      - keypoints: (B, max_kpts, 2) — (y, x)
      - validity: (B, max_kpts) — 每点有效性
      - K: (B,) — 实际关键点数
      - gp_lml: 标量 — GP 模块 log_marginal_likelihood（供损失）

详见：
- docs/v5_架构设计.md §2.4
- tasks/team_v5_alpha.json 中 V5A4
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.pe_tsnet_multimodal import (
    ResNet18Backbone2D,
    TemporalFeatureExtractor,
    CrossAttentionFusion,
)
from models.spline_head import SplineHead
from models.gp_module import GPModule


class PETSNetMultimodalV5(nn.Module):
    """
    ResNet-18 + SplineHead v5-α 模型

    Args:
        seq_len: 1D 序列长度
        image_channels: 2D 输入通道数（v5-α 默认 3）
        image_size: 输入图像尺寸（默认 256）
        pretrained_2d: 是否使用 ImageNet 预训练权重
        dropout: Dropout 比例
        feat_dim_2d: 2D 特征维度（默认 512，ResNet-18 输出）
        feat_dim_1d: 1D 特征维度（默认 64）
        max_kpts: 关键点上界（默认 16）
        min_kpts: 关键点下界（默认 8）
        use_gp: 是否启用 GP 嵌入模块（默认 True）
        gp_hidden_dim: GP 模块中间层维度
    """

    def __init__(
        self,
        seq_len: int = 300,
        image_channels: int = 3,
        image_size: int = 256,
        pretrained_2d: bool = True,
        dropout: float = 0.2,
        feat_dim_2d: int = 512,
        feat_dim_1d: int = 64,
        max_kpts: int = 16,
        min_kpts: int = 8,
        use_gp: bool = True,
        gp_hidden_dim: int = 128,
        spatial_head: bool = False,
    ):
        super().__init__()
        if image_channels not in (1, 2, 3, 4):
            raise ValueError(
                f"image_channels 应 ∈ {{1,2,3,4}}，实际 {image_channels}"
            )
        if min_kpts < 2:
            raise ValueError(f"min_kpts 必须 >= 2，实际 {min_kpts}")
        if max_kpts < min_kpts:
            raise ValueError(
                f"max_kpts ({max_kpts}) 必须 >= min_kpts ({min_kpts})"
            )

        self.seq_len = seq_len
        self.image_channels = image_channels
        self.image_size = image_size
        self.max_kpts = max_kpts
        self.min_kpts = min_kpts
        self.use_gp = use_gp

        # 2D 骨干：ResNet-18（直接用 v4 的实现，支持任意 in_channels）
        self.backbone_2d = ResNet18Backbone2D(
            in_channels=image_channels,
            pretrained=pretrained_2d,
        )

        # 1D 骨干：v4 的 TemporalFeatureExtractor（沿用）
        self.backbone_1d = TemporalFeatureExtractor(
            input_dim=1,
            hidden_dim=32,
            num_heads=4,
            dropout=dropout,
        )

        # 融合层：CrossAttentionFusion（沿用 v4）
        self.fusion = CrossAttentionFusion(
            dim_2d=feat_dim_2d,
            dim_1d=feat_dim_1d,
            num_heads=4,
            dropout=dropout,
        )
        fused_dim = feat_dim_2d + feat_dim_1d  # 512 + 64 = 576

        # GP 嵌入模块（v5 新增，可选）
        if use_gp:
            self.gp_module = GPModule(
                in_dim=fused_dim,
                hidden_dim=gp_hidden_dim,
                init_length_scale=1.0,
                init_signal_var=1.0,
                learnable=True,
            )
            head_in_dim = fused_dim
        else:
            self.gp_module = None
            head_in_dim = fused_dim

        # 输出头：SplineHead（v5 新增，替换 v4 MultiTaskHead）
        self.output_head = SplineHead(
            in_dim=head_in_dim,
            hidden_dim=256,
            max_kpts=max_kpts,
            min_kpts=min_kpts,
            dropout=dropout,
        )

        # 空间关键点头（V5-030 第三轮最小架构实验，默认关）
        # 全局池化→Linear 头对细线状裂纹定位有结构上限（~40-70px vs 5px），
        # 本头从 backbone 空间特征图直接定位。
        self.use_spatial_head = bool(spatial_head)
        if self.use_spatial_head:
            from models.spatial_kpt_head import SpatialKptHead
            self.backbone_2d.set_spatial_output(True)
            self.spatial_head = SpatialKptHead(
                in_channels=512, d_model=128, num_queries=max_kpts,
            )

    def forward(self, x_1d: torch.Tensor, x_2d: torch.Tensor) -> dict:
        """
        前向传播

        Args:
            x_1d: (B, seq_len) 温度时序
            x_2d: (B, image_channels, H, W) 多模态图像

        Returns:
            dict: 见 SplineHead.forward
        """
        # 2D 分支
        feat_2d = self.backbone_2d(x_2d)  # (B, 512)；spatial_head 时 (B, 512, H', W')

        # 空间关键点头（默认关；开启时 kpts/validity 从空间图直接定位）
        spatial_out = None
        if self.use_spatial_head:
            spatial_out = self.spatial_head(feat_2d)
            feat_2d_vec = feat_2d.mean(dim=(2, 3))  # 空间图 → 池化向量照旧融合
        else:
            feat_2d_vec = feat_2d

        # 1D 分支
        feat_1d = self.backbone_1d(x_1d)  # (B, 64)

        # 融合
        fused = self.fusion(feat_2d_vec, feat_1d)  # (B, 576)

        # GP 调制（可选）
        if self.gp_module is not None:
            fused = self.gp_module(fused)  # (B, 576)

        # 输出头
        out = self.output_head(fused)

        # 空间头输出覆盖 kpts/validity（bbox 仍来自融合分支）
        if spatial_out is not None:
            out["keypoints"] = spatial_out["keypoints"]
            out["validity"] = spatial_out["validity"]
            out["heatmap"] = spatial_out["heatmap"]

        # 附 GP LML（供 OrderedKeypointLoss 用）
        if self.gp_module is not None:
            out["gp_lml"] = self.gp_module.log_marginal_likelihood()

        return out

    def count_parameters(self) -> int:
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
#  冒烟测试
# =============================================================================

def _smoke_test():
    """快速冒烟测试"""
    print("=" * 60)
    print("PETSNetMultimodalV5 冒烟测试")
    print("=" * 60)

    torch.manual_seed(0)
    model = PETSNetMultimodalV5(
        seq_len=300,
        image_channels=3,  # v5-α 3 模态
        image_size=256,
        pretrained_2d=False,  # 测试时不下载预训练
        use_gp=True,
    )
    print(f"[OK] 模型初始化: {model.count_parameters():,} 参数")
    print(f"     max_kpts={model.max_kpts}, min_kpts={model.min_kpts}")
    print(f"     use_gp={model.use_gp}")

    # 构造测试输入
    B = 4
    x_1d = torch.randn(B, 300)
    x_2d = torch.randn(B, 3, 256, 256)

    out = model(x_1d, x_2d)

    print(f"\n输出形状：")
    print(f"  bbox:      {tuple(out['bbox'].shape)}")
    print(f"  keypoints: {tuple(out['keypoints'].shape)}")
    print(f"  validity:  {tuple(out['validity'].shape)}")
    print(f"  K:         {tuple(out['K'].shape)}, values={out['K'].tolist()}")
    print(f"  gp_lml:    {out['gp_lml'].item():.4f}")

    # 验证值域
    assert (out["bbox"] >= 0).all() and (out["bbox"] <= 1).all()
    assert (out["keypoints"] >= 0).all() and (out["keypoints"] <= 1).all()
    assert (out["validity"] >= 0).all() and (out["validity"] <= 1).all()
    assert (out["K"] >= model.min_kpts).all() and (out["K"] <= model.max_kpts).all()
    assert (out["bbox"][:, 0] <= out["bbox"][:, 2]).all()  # x1 <= x2
    assert (out["bbox"][:, 1] <= out["bbox"][:, 3]).all()  # y1 <= y2

    # 反向传播
    fake_bbox_loss = out["bbox"].sum()
    fake_kpts_loss = out["keypoints"].sum()
    (fake_bbox_loss + fake_kpts_loss).backward()

    print(f"\n[ALL PASS] ✅")


if __name__ == "__main__":
    _smoke_test()