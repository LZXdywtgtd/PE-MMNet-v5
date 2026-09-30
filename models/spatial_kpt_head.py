# -*- coding: utf-8 -*-
"""空间关键点头 + heatmap 辅助监督（V5-030 第三轮最小架构实验）

背景（结构性上限证据链，2026-09-30）：
全局池化特征 → Linear 头直接回归 16 点，所有变体（kd0.5/kd2/ordered
v1/v2）pred→裂纹距离全部停在 ~40-70px，无一贴上细线状裂纹；判据 4
上界 0.029 要求 ~5px 定位。overfit-32 排除容量（train 0.077 可记忆
目标）——瓶颈是全局池化丢失空间位置信息，回归头无从定位。

本模块（用户提出的最小实验，~50 行级改动）：
1. SpatialKptHead：backbone 空间特征图 (B,512,8,8)（复用 DETR 变体
   已验证的 set_spatial_output(True) 接口）→ 16 个可学习 query 对
   64 个空间位置做 cross-attention → 16 点 (y,x) + validity。
   空间位置信息保留，定位有据可依。
2. heatmap 辅助头：1×1 conv → (B,1,8,8) sigmoid，与 GT mask 下采样
   的 8×8 heatmap 做逐像素 BCE——给空间图一个直接的"裂纹在哪"监督，
   不经样条/回归的间接路径。

两者均默认关闭（model_kwargs 显式开启），不碰 106 回归默认行为。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialKptHead(nn.Module):
    """空间特征图 → 16 关键点（query-based cross-attention）"""

    def __init__(
        self,
        in_channels: int = 512,
        d_model: int = 128,
        num_queries: int = 16,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.d_model = d_model
        self.queries = nn.Parameter(torch.randn(1, num_queries, d_model) * 0.02)
        self.queries_yx = nn.Parameter(torch.randn(1, num_queries, 2) * 0.02)

        self.input_proj = nn.Conv2d(in_channels, d_model, kernel_size=1)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=dropout, batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=num_layers)
        self.coord_head = nn.Linear(d_model, 2)
        self.validity_head = nn.Linear(d_model, 1)
        # heatmap 辅助头：空间图 → 单通道显著图
        self.heatmap_head = nn.Conv2d(in_channels, 1, kernel_size=1)

    def _pos_embed(self, H: int, W: int, device, dtype) -> torch.Tensor:
        """2D 正弦位置编码（y/x 各半通道），(1, H*W, d_model)"""
        d = self.d_model
        assert d % 4 == 0, "d_model 应为 4 的倍数"
        pos = torch.zeros(H, W, d, device=device, dtype=dtype)
        y = torch.arange(H, device=device, dtype=dtype).view(H, 1)   # (H, 1)
        x = torch.arange(W, device=device, dtype=dtype).view(W, 1)   # (W, 1)
        div = torch.exp(
            torch.arange(0, d // 4, device=device, dtype=dtype)
            * (-torch.log(torch.tensor(10000.0)) / (d // 4))
        )                                                            # (d/4,)
        q = d // 4
        pos[:, :, :q] = y.sin() * div          # (H,1)*(q,) → (H,q) 广播到 W
        pos[:, :, q:2 * q] = (y * div).cos()
        pos[:, :, 2 * q:3 * q] = (x * div).sin()   # (W,q) 广播到 H
        pos[:, :, 3 * q:] = (x * div).cos()
        return pos.view(1, H * W, d)

    def forward(self, feat_map: torch.Tensor) -> dict:
        """
        Args:
            feat_map: (B, C, H, W) backbone 空间特征图（如 512×8×8）

        Returns:
            dict:
                keypoints: (B, Q, 2) 归一化 (y, x) ∈ [0,1]
                validity:  (B, Q) sigmoid
                heatmap:   (B, 1, H, W) sigmoid 显著图
        """
        B, C, H, W = feat_map.shape
        memory = self.input_proj(feat_map).flatten(2).transpose(1, 2)  # (B, HW, d)
        memory = memory + self._pos_embed(H, W, memory.device, memory.dtype)

        q = self.queries.expand(B, -1, -1)
        # 坐标 = 可学习参考点 + 学到的偏移（参考点给定位先验）
        ref = self.queries_yx.expand(B, -1, -1)
        decoded = self.decoder(q, memory)  # (B, Q, d)

        offsets = torch.sigmoid(self.coord_head(decoded))
        kpts = torch.clamp(ref + 0.5 * (offsets - 0.5), 0.0, 1.0)

        validity = torch.sigmoid(self.validity_head(decoded)).squeeze(-1)  # (B, Q)
        heatmap = torch.sigmoid(self.heatmap_head(feat_map))  # (B, 1, H, W)
        return {"keypoints": kpts, "validity": validity, "heatmap": heatmap}


def gt_heatmap_target(
    pixel_mask: torch.Tensor,
    true_crack_pixels: torch.Tensor,
    feat_h: int,
    feat_w: int,
) -> torch.Tensor:
    """
    GT mask → 下采样 heatmap 目标 (B, 1, feat_h, feat_w)

    pixel_mask: (B, N) bool；true_crack_pixels: (B, N, 2) 归一化 (y, x)
    负样本（无裂纹）目标全 0。
    """
    B = pixel_mask.size(0)
    device = pixel_mask.device
    tgt = torch.zeros(B, 1, feat_h, feat_w, device=device)
    for i in range(B):
        n = int(pixel_mask[i].sum().item())
        if n == 0:
            continue
        px = true_crack_pixels[i, :n]  # (n, 2) (y, x) 归一化
        gy = (px[:, 0] * feat_h).long().clamp(0, feat_h - 1)
        gx = (px[:, 1] * feat_w).long().clamp(0, feat_w - 1)
        tgt[i, 0, gy, gx] = 1.0
    return tgt


def heatmap_loss(
    pred_heatmap: torch.Tensor,
    pixel_mask: torch.Tensor,
    true_crack_pixels: torch.Tensor,
) -> torch.Tensor:
    """逐像素 BCE（正类稀疏，pos_weight 用负/正比，上限 50 防爆）"""
    B, _, H, W = pred_heatmap.shape
    tgt = gt_heatmap_target(pixel_mask, true_crack_pixels, H, W)
    n_pos = tgt.sum()
    n_neg = tgt.numel() - n_pos
    pos_weight = torch.clamp(n_neg / n_pos.clamp(min=1.0), max=50.0)
    return F.binary_cross_entropy_with_logits(
        pred_heatmap, tgt, pos_weight=pos_weight
    )
