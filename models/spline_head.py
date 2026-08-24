"""
Spline 输出头（V5A2）

v5-α 模型最后一段：把 backbone 输出的融合特征回归为
bbox + 8-16 动态关键点（变长有序轨迹）。

设计：
- 输入：(B, hidden_dim) 融合特征
- 输出：
  * bbox: (B, 4) 归一化角点 (x1, y1, x2, y2)
  * keypoints: (B, max_kpts, 2) 归一化 (y, x) 坐标
  * K: (B,) 动态关键点数（[min_kpts, max_kpts]）
  * validity: (B, max_kpts) 每个关键点的有效性概率

训练/推理流程：
- 训练：覆盖距离 loss 用全部 max_kpts + validity 权重
- 推理：取 validity 最高的 K 个点，按 x 坐标排序后拟合 B 样条

详见：
- docs/v5_架构设计.md §2.4
- tasks/team_v5_alpha.json 中 V5A2
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SplineHead(nn.Module):
    """Spline 输出头"""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        max_kpts: int = 16,
        min_kpts: int = 8,
        dropout: float = 0.1,
    ):
        """
        Args:
            in_dim:    输入特征维度（backbone 输出）
            hidden_dim: 中间层维度
            max_kpts:  关键点上界
            min_kpts:  关键点下界
            dropout:   Dropout 概率
        """
        super().__init__()
        if min_kpts < 2:
            raise ValueError(f"min_kpts 必须 >= 2，当前 {min_kpts}")
        if max_kpts < min_kpts:
            raise ValueError(f"max_kpts ({max_kpts}) 必须 >= min_kpts ({min_kpts})")

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.max_kpts = max_kpts
        self.min_kpts = min_kpts
        self.kpt_range = max_kpts - min_kpts

        # 共享 backbone
        self.shared = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
        )

        # bbox 头：4 维 (x1, y1, x2, y2)，sigmoid 到 [0, 1]
        self.bbox_head = nn.Linear(hidden_dim // 2, 4)

        # 关键点头：max_kpts * 2 维 (y, x)，sigmoid 到 [0, 1]
        self.kpt_head = nn.Linear(hidden_dim // 2, max_kpts * 2)

        # 有效性头：max_kpts 维，每点一个 sigmoid 概率
        self.validity_head = nn.Linear(hidden_dim // 2, max_kpts)

        # K 预测头：1 维，sigmoid → [min_kpts, max_kpts]
        self.K_head = nn.Linear(hidden_dim // 2, 1)

        # 初始化（让 bbox 初始偏向中心）
        nn.init.constant_(self.bbox_head.bias, 0.0)
        nn.init.normal_(self.bbox_head.weight, std=0.01)

    def forward(self, feat: torch.Tensor) -> dict:
        """
        Args:
            feat: (B, in_dim) — 融合特征

        Returns:
            dict:
                bbox:      (B, 4)            归一化角点
                keypoints: (B, max_kpts, 2)  关键点 (y, x)
                validity:  (B, max_kpts)    关键点有效性
                K:         (B,)             实际关键点数
        """
        if feat.dim() != 2 or feat.size(1) != self.in_dim:
            raise ValueError(
                f"feat 应为 (B, {self.in_dim})，实际 {tuple(feat.shape)}"
            )

        h = self.shared(feat)  # (B, hidden_dim // 2)

        # bbox
        bbox_raw = self.bbox_head(h)  # (B, 4)
        bbox = torch.sigmoid(bbox_raw)
        # 确保 x1 <= x2, y1 <= y2（重新参数化）
        x1 = bbox[:, 0]
        y1 = bbox[:, 1]
        x2 = bbox[:, 2]
        y2 = bbox[:, 3]
        x_min = torch.minimum(x1, x2)
        x_max = torch.maximum(x1, x2)
        y_min = torch.minimum(y1, y2)
        y_max = torch.maximum(y1, y2)
        bbox = torch.stack([x_min, y_min, x_max, y_max], dim=1)  # (B, 4)

        # 关键点
        kpts_raw = self.kpt_head(h)  # (B, max_kpts * 2)
        keypoints = torch.sigmoid(kpts_raw).view(-1, self.max_kpts, 2)  # (B, K, 2)

        # 有效性
        validity = torch.sigmoid(self.validity_head(h))  # (B, max_kpts)

        # K（关键点总数）
        K_raw = torch.sigmoid(self.K_head(h)).squeeze(-1)  # (B,)
        K = (K_raw * self.kpt_range + self.min_kpts).round().long()  # (B,) in [min, max]
        K = torch.clamp(K, min=self.min_kpts, max=self.max_kpts)

        return {
            "bbox": bbox,
            "keypoints": keypoints,
            "validity": validity,
            "K": K,
        }

    def predict_kpts_only(
        self, feat: torch.Tensor, sort_by: str = "x"
    ) -> torch.Tensor:
        """
        推理便捷接口：返回按 validity 排序并截断到 K 的关键点

        Args:
            feat: (B, in_dim)
            sort_by: "x"（按 x 坐标排序）/ "validity"（按有效性排序）

        Returns:
            ordered_kpts: (B, max_kpts, 2) 排序后的关键点
        """
        out = self.forward(feat)
        kpts = out["keypoints"]  # (B, max_kpts, 2)
        validity = out["validity"]  # (B, max_kpts)

        if sort_by == "x":
            # 按 x 坐标排序（确保有序）
            sorted_kpts, _ = torch.sort(kpts[:, :, 1], dim=1)
            # 重建 (y, x) 形式
            x_sorted = sorted_kpts  # (B, max_kpts)
            # 用插值找回对应的 y
            y_sorted = torch.gather(
                kpts[:, :, 0], 1,
                torch.argsort(kpts[:, :, 1], dim=1)
            )
            ordered_kpts = torch.stack([y_sorted, x_sorted], dim=-1)
        elif sort_by == "validity":
            # 按 validity 降序排
            sorted_kpts = []
            for b in range(kpts.size(0)):
                _, idx = torch.sort(validity[b], descending=True)
                sorted_kpts.append(kpts[b][idx])
            ordered_kpts = torch.stack(sorted_kpts, dim=0)
        else:
            raise ValueError(f"sort_by 应为 'x' 或 'validity'，实际 {sort_by}")

        return ordered_kpts


if __name__ == "__main__":
    # 冒烟测试
    print("=" * 60)
    print("SplineHead 冒烟测试")
    print("=" * 60)

    head = SplineHead(in_dim=512, hidden_dim=256, max_kpts=16, min_kpts=8)
    print(f"[OK] SplineHead 初始化: in_dim=512, max_kpts=16")
    print(f"     参数总数: {sum(p.numel() for p in head.parameters()):,}")

    feat = torch.randn(4, 512)
    out = head(feat)

    print(f"\n输出形状：")
    print(f"  bbox:      {tuple(out['bbox'].shape)}, "
          f"range [{out['bbox'].min():.3f}, {out['bbox'].max():.3f}]")
    print(f"  keypoints: {tuple(out['keypoints'].shape)}, "
          f"range [{out['keypoints'].min():.3f}, {out['keypoints'].max():.3f}]")
    print(f"  validity:  {tuple(out['validity'].shape)}, "
          f"range [{out['validity'].min():.3f}, {out['validity'].max():.3f}]")
    print(f"  K:         {tuple(out['K'].shape)}, values={out['K'].tolist()}")

    # 关键点应该在 [0, 1] 内
    assert (out["keypoints"] >= 0).all() and (out["keypoints"] <= 1).all()
    # bbox 应满足 x1 <= x2, y1 <= y2
    bbox = out["bbox"]
    assert (bbox[:, 0] <= bbox[:, 2]).all(), "x1 <= x2"
    assert (bbox[:, 1] <= bbox[:, 3]).all(), "y1 <= y2"
    # K 应在 [8, 16]
    assert (out["K"] >= 8).all() and (out["K"] <= 16).all()

    print("\n[ALL PASS] ✅")