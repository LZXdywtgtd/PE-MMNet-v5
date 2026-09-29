"""
覆盖距离 loss（V5A3）— B 样条 + Hausdorff + 组合损失

v5-α 主损失函数：

1. **覆盖距离 (Coverage loss)**：
   - 用预测关键点拟合 Catmull-Rom 三次样条
   - 在样条上密集采样 M 个点
   - 真值裂纹像素到样条的最小距离 → chamfer / Hausdorff

2. **bbox 损失**：Smooth L1（对 (x1,y1,x2,y2) 角点回归）

3. **GP 正则项**：GPModule.log_marginal_likelihood()（鼓励 ℓ/σ² 留在默认附近）

4. **泊松先验**：鼓励关键点等弧长分布（段长方差最小化）

5. **[v5-β 占位] 位置 6D loss**：Smooth L1（pose=[tx,ty,tz,rx,ry,rz] → 0）
   - v5-α 默认权重 0.0（占位，不贡献梯度）
   - 真值位姿缺失时返回 0（zero surrogate）

6. **[v5-β 占位] UV 投影 loss**：MSE（3D→2D 重投影误差）
   - v5-α 默认权重 0.0
   - 真值 UV 缺失时返回 0

总损失（架构设计 §3.5）：
    total = 1.0  * coverage
          + 1.0  * bbox
          + 0.1  * gp
          + 0.05 * poisson
          + λ_pos6d * position_6d   (v5-β)
          + λ_uv    * uv_projection (v5-β)

样条选择说明
-------------
为什么用 Catmull-Rom 而不是 scipy CubicSpline？

- scipy.interpolate.CubicSpline **不可微**（基于 numpy，无 autograd 通路）
- Catmull-Rom 是 Hermite 三次样条（C¹ 连续），完全在 PyTorch 内实现
- 对每个 segment (P_{i-1}, P_i, P_{i+1}, P_{i+2}) 做 Hermite 插值
- 端点用镜像延拓 ghost points（保持端点切线连续）
- 标准"三次 B 样条"的近似；用户文档中的"三次 B 样条"即指此

详见：
- docs/v5_架构设计.md §3.1, §3.5
- tasks/team_v5_alpha.json 中 V5A3
- docs/v5_范围说明.md §11 v5-β 8 任务
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ============================================================
#  B 样条拟合（纯 PyTorch，可微）
# ============================================================

def catmull_rom_spline_torch(
    points: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """
    均匀 Catmull-Rom 三次样条拟合

    公式（每个 segment）：
        P(t) = 0.5 * (
            (2*P1) +
            (-P0 + P2) * t +
            (2*P0 - 5*P1 + 4*P2 - P3) * t² +
            (-P0 + 3*P1 - 3*P2 + P3) * t³
        )
        其中 t ∈ [0, 1]

    Args:
        points: (B, K, 2) — 有序关键点，K >= 2
        num_samples: M — 在样条上密集采样的点数

    Returns:
        samples: (B, M, 2) — 沿样条均匀采样的 M 个点

    Note:
        - K=2 时退化为线性插值（无足够控制点做 Catmull-Rom）
        - 端点用镜像延拓：P_{-1} = 2*P_0 - P_1, P_K = 2*P_{K-1} - P_{K-2}
        - 所有操作在 PyTorch 内，可对 points 求梯度
    """
    if points.dim() != 3 or points.size(-1) != 2:
        raise ValueError(
            f"points 应为 (B, K, 2)，实际 {tuple(points.shape)}"
        )
    B, K, _ = points.shape

    if K < 2:
        raise ValueError(f"K 必须 >= 2，当前 {K}")

    if num_samples < 2:
        raise ValueError(f"num_samples 必须 >= 2，当前 {num_samples}")

    device = points.device
    dtype = points.dtype

    # K=2 退化为线性插值
    if K == 2:
        t = torch.linspace(0.0, 1.0, num_samples, device=device, dtype=dtype)
        # (B, 1, 2) * (1, M, 1) + (B, 1, 2) * (1, M, 1)
        return (
            points[:, 0:1, :] * (1.0 - t.view(1, -1, 1))
            + points[:, 1:2, :] * t.view(1, -1, 1)
        )

    # 端点 ghost points（镜像延拓）
    pad_start = 2.0 * points[:, 0:1, :] - points[:, 1:2, :]
    pad_end = 2.0 * points[:, -1:, :] - points[:, -2:-1, :]
    padded = torch.cat([pad_start, points, pad_end], dim=1)  # (B, K+2, 2)

    # K-1 个 segment，分配 num_samples 个采样点
    n_per_seg = num_samples // (K - 1)
    remainder = num_samples - n_per_seg * (K - 1)

    seg_samples = []

    for seg_idx in range(K - 1):
        # 当前 segment 的四个控制点
        P0 = padded[:, seg_idx, :]      # (B, 2)
        P1 = padded[:, seg_idx + 1, :]
        P2 = padded[:, seg_idx + 2, :]
        P3 = padded[:, seg_idx + 3, :]

        # 决定当前 segment 的采样数
        n_here = n_per_seg + (1 if seg_idx < remainder else 0)

        if seg_idx == 0:
            # 第一个 segment: t ∈ [0, 1] 均匀
            t = torch.linspace(0.0, 1.0, n_here, device=device, dtype=dtype)
        else:
            # 后续 segment: 跳过 t=0，避免与前一段重复
            if n_here == 1:
                t = torch.tensor([1.0], device=device, dtype=dtype)
            else:
                t = torch.linspace(1.0 / n_here, 1.0, n_here,
                                   device=device, dtype=dtype)

        # Catmull-Rom 公式
        t2 = t * t
        t3 = t2 * t
        t_view = t.view(1, -1, 1)  # (1, n_here, 1) for broadcast over batch

        sample = 0.5 * (
            2.0 * P1.unsqueeze(1)
            + (-P0 + P2).unsqueeze(1) * t_view
            + (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3).unsqueeze(1) * (t2.view(1, -1, 1))
            + (-P0 + 3.0 * P1 - 3.0 * P2 + P3).unsqueeze(1) * (t3.view(1, -1, 1))
        )  # (B, n_here, 2)

        seg_samples.append(sample)

    return torch.cat(seg_samples, dim=1)  # (B, num_samples, 2)


# ============================================================
#  覆盖距离 loss（核心）
# ============================================================

def coverage_loss(
    pred_kpts: torch.Tensor,
    true_crack_pixels: torch.Tensor,
    M: int = 200,
    mode: str = "chamfer",
    pixel_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    B 样条 + 双向覆盖距离 loss（2026-09-29 修复版）

    修复两处（v5a6 平台解根因，定量证据见 _diag_negpad.py）：
    1. padding 掩蔽：pixel_mask=None 时 batch 中负样本的 -1 padding
       坐标会全部进入 chamfer（2 正 6 负 batch 中 9 万个假像素），
       监督目标被污染。现在 mask=False 的像素不参与距离。
    2. 双向 chamfer：原单向（真像素→样条）允许"摊开覆盖"平凡解——
       边界角点 0.0148 < 真关键点 0.0925。现加"样条→真像素"反向项，
       样条上的点也必须落在裂纹附近。

    步骤：
        1. 用预测关键点 (B, K, 2) 拟合 Catmull-Rom 样条
        2. 在样条上密集采样 M 个点
        3. 正向：真像素 → 最近样条点距离（mean over 真像素）
        4. 反向：样条点 → 最近真像素距离（mean over 样条点）
        5. chamfer = (正向 + 反向) / 2；hausdorff 用两方向 max 的较大者

    Args:
        pred_kpts: (B, K, 2) — 预测的有序关键点，K ∈ [8, 16]
        true_crack_pixels: (B, N, 2) — 真值裂纹像素（-1 = padding）
        M: 在曲线上密集采样的点数（默认 200）
        mode: "chamfer"（默认）或 "hausdorff"
        pixel_mask: (B, N) bool — True=真实像素。None 时全按真实处理
            （仅整个 batch 都是正样本且无 padding 时安全）。

    Returns:
        loss: 标量（正样本子集上的均值；全负样本 batch 返回 0）
    """
    if mode not in ("chamfer", "hausdorff"):
        raise ValueError(f"mode 应为 'chamfer' 或 'hausdorff'，实际 {mode!r}")

    if pred_kpts.size(0) != true_crack_pixels.size(0):
        raise ValueError(
            f"pred_kpts batch={pred_kpts.size(0)} 与 "
            "true_crack_pixels batch={true_crack_pixels.size(0)} 不一致"
        )

    B = pred_kpts.size(0)

    # 每样本真实像素数（mask=None 时按原 N 处理）
    if pixel_mask is not None:
        n_real = pixel_mask.sum(dim=-1)  # (B,)
    else:
        n_real = torch.full(
            (B,), true_crack_pixels.size(1),
            device=true_crack_pixels.device,
        )

    has_crack = n_real > 0  # (B,)
    if not has_crack.any():
        # 全负样本 batch：无监督信号，返回 0（保留图以维持 autograd 图）
        return (pred_kpts.sum() * 0.0)

    spline_samples = catmull_rom_spline_torch(pred_kpts, M)  # (B, M, 2)

    per_sample = []
    for i in range(B):
        if not has_crack[i]:
            continue
        n = int(n_real[i])
        px_i = true_crack_pixels[i, :n]        # (n, 2) 真实像素
        sp_i = spline_samples[i]               # (M, 2)

        # 正向：真像素 → 最近样条点
        d_t2s = torch.cdist(px_i, sp_i)        # (n, M)
        fwd = d_t2s.min(dim=-1).values         # (n,)
        # 反向：样条点 → 最近真像素
        bwd = d_t2s.min(dim=0).values          # (M,)  ← 复用同一距离矩阵

        if mode == "chamfer":
            s = fwd.mean() + bwd.mean()
        else:  # hausdorff：两方向最坏距离的较大者
            s = torch.maximum(fwd.max(), bwd.max())
        per_sample.append(s)

    return torch.stack(per_sample).mean()


def kpt_direct_loss(
    pred_kpts: torch.Tensor,
    true_crack_pixels: torch.Tensor,
    pixel_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    关键点直接监督（2026-09-29 加入）

    pred_kpts → 最近真像素的 L2，不经过样条。修复原设计中
    kpts 只有"样条→chamfer"双重间接监督（梯度穿两层、样条在
    饱和点梯度病态）导致的塌缩。仅正样本参与。

    Returns:
        loss: 标量（正样本子集均值；全负样本 batch 返回 0）
    """
    B, K, _ = pred_kpts.shape
    if pixel_mask is not None:
        n_real = pixel_mask.sum(dim=-1)
    else:
        n_real = torch.full(
            (B,), true_crack_pixels.size(1),
            device=true_crack_pixels.device,
        )
    has_crack = n_real > 0
    if not has_crack.any():
        return pred_kpts.sum() * 0.0

    per_sample = []
    for i in range(B):
        if not has_crack[i]:
            continue
        n = int(n_real[i])
        d = torch.cdist(pred_kpts[i], true_crack_pixels[i, :n])  # (K, n)
        per_sample.append(d.min(dim=-1).values.mean())  # (K,) → 标量

    return torch.stack(per_sample).mean()


# ============================================================
#  泊松先验（鼓励等弧长）
# ============================================================

def poisson_prior(pred_kpts: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    泊松先验：鼓励关键点等弧长分布

    物理含义：
    - 泊松过程产生的关键点应该是"随机但均匀"的
    - 相邻关键点间距方差应最小

    实现：
        seg_lens = ||p_{i+1} - p_i||  for i = 0..K-2
        loss = var(seg_lens)

    Args:
        pred_kpts: (B, K, 2)

    Returns:
        loss: 标量（段长方差）
    """
    if pred_kpts.size(-2) < 2:
        return torch.tensor(0.0, device=pred_kpts.device, dtype=pred_kpts.dtype)

    # 相邻段长度
    diffs = pred_kpts[:, 1:, :] - pred_kpts[:, :-1, :]  # (B, K-1, 2)
    seg_lens = diffs.norm(dim=-1)  # (B, K-1)

    # 方差（per-sample mean）
    seg_mean = seg_lens.mean(dim=-1, keepdim=True) + eps  # (B, 1)
    seg_var = ((seg_lens - seg_mean) ** 2).mean(dim=-1)  # (B,)

    return seg_var.mean()


# ============================================================
#  v5-β 占位 loss
# ============================================================

def position_6d_loss(
    pred_position_6d: torch.Tensor | None,
    true_position_6d: torch.Tensor | None,
) -> torch.Tensor:
    """
    位置 6D loss（v5-β 占位）

    真值位姿缺失（None）→ 返回 0（占位）。
    v5-α 当前默认权重 0.0，本函数实际不贡献梯度。

    Args:
        pred_position_6d: (B, 6) — [tx,ty,tz,rx,ry,rz] 或 None
        true_position_6d: (B, 6) 或 None

    Returns:
        loss: 标量（Smooth L1），缺失则返回 0
    """
    if pred_position_6d is None or true_position_6d is None:
        device = (
            pred_position_6d.device
            if pred_position_6d is not None
            else torch.device("cpu")
        )
        return torch.tensor(0.0, device=device)
    if pred_position_6d.shape != true_position_6d.shape:
        raise ValueError(
            f"pred_position_6d {tuple(pred_position_6d.shape)} 与 "
            f"true_position_6d {tuple(true_position_6d.shape)} 形状不匹配"
        )
    return torch.nn.functional.smooth_l1_loss(pred_position_6d, true_position_6d)


def uv_projection_loss(
    pred_kpts: torch.Tensor,
    pred_position_6d: torch.Tensor | None,
    true_position_6d: torch.Tensor | None,
    K_matrix: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    UV 投影 loss（v5-β 占位）

    思路（v5-β 启动后实现）：
        1. 把预测关键点提升到陶瓷局部 3D（基于 UV 网格）
        2. 用 pred_position_6d 做 6D 变换
        3. 投影到相机平面（K_matrix 相机内参）
        4. 与真值 2D 像素坐标比对 → MSE

    v5-α 当前默认权重 0.0，本函数占位返回 0。

    Args:
        pred_kpts: (B, K, 2) — 预测的 2D 关键点
        pred_position_6d: (B, 6) — 预测位姿
        true_position_6d: (B, 6) — 真值位姿
        K_matrix: (B, 3, 3) — 相机内参（None → 返回 0）

    Returns:
        loss: 标量，缺失则返回 0
    """
    if (
        pred_position_6d is None
        or true_position_6d is None
        or K_matrix is None
    ):
        device = pred_kpts.device
        return torch.tensor(0.0, device=device)
    # v5-β 实现占位：返回 L2 距离（让接口可调用）
    # 真值实现见 docs/v5_架构设计.md §8.4
    diff = pred_position_6d - true_position_6d
    return (diff * diff).mean()


# ============================================================
#  组合 loss（OrderedKeypointLoss）
# ============================================================

class OrderedKeypointLoss(nn.Module):
    """
    组合 loss：覆盖距离 + bbox + GP 正则 + 泊松先验 + （v5-β 占位）

    总损失（架构设计 §3.5）：
        total = λ_cov * coverage
              + λ_bbox * smooth_l1_bbox
              + λ_gp * (-gp_module.log_marginal_likelihood)
              + λ_poisson * poisson_prior
              + λ_pos6d * position_6d_loss   (v5-β)
              + λ_uv    * uv_projection_loss (v5-β)

    默认权重（v5-α）：λ_cov=1.0, λ_bbox=1.0, λ_gp=0.1, λ_poisson=0.05
    v5-β 默认权重：λ_pos6d=0.1, λ_uv=0.05（占位，暂未启用）
    """

    def __init__(
        self,
        lambda_coverage: float = 1.0,
        lambda_bbox: float = 1.0,
        lambda_gp: float = 0.1,
        lambda_poisson: float = 0.05,
        coverage_mode: str = "chamfer",
        M: int = 200,
        bbox_beta: float = 1.0,
        lambda_kpt_direct: float = 0.5,
        # ---- v5-β 占位权重 ----
        lambda_position_6d: float = 0.0,
        lambda_uv_projection: float = 0.0,
    ):
        """
        Args:
            lambda_coverage: 覆盖距离权重
            lambda_bbox: bbox Smooth L1 权重
            lambda_gp: GP 模块 LML 权重
            lambda_poisson: 泊松先验权重
            coverage_mode: "chamfer" / "hausdorff"
            M: 样条采样点数
            bbox_beta: Smooth L1 的 β 参数
            lambda_kpt_direct: 关键点直接监督权重（2026-09-29 加入，
                修复样条双重间接监督的梯度病态）
            lambda_position_6d: 位置 6D loss 权重（v5-β 占位）
            lambda_uv_projection: UV 投影 loss 权重（v5-β 占位）
        """
        super().__init__()
        if coverage_mode not in ("chamfer", "hausdorff"):
            raise ValueError(
                f"coverage_mode 应为 'chamfer' 或 'hausdorff'，实际 {coverage_mode!r}"
            )
        self.lambda_kpt_direct = lambda_kpt_direct
        self.lambda_coverage = lambda_coverage
        self.lambda_bbox = lambda_bbox
        self.lambda_gp = lambda_gp
        self.lambda_poisson = lambda_poisson
        self.lambda_position_6d = lambda_position_6d
        self.lambda_uv_projection = lambda_uv_projection
        self.coverage_mode = coverage_mode
        self.M = M

        self.bbox_loss = nn.SmoothL1Loss(beta=bbox_beta, reduction="mean")

    def forward(
        self,
        pred_bbox: torch.Tensor,         # (B, 4)
        pred_kpts: torch.Tensor,         # (B, K, 2)
        true_bbox: torch.Tensor,         # (B, 4)
        true_crack_pixels: torch.Tensor, # (B, N, 2)
        gp_module=None,                  # GPModule 实例（可选）
        pixel_mask: torch.Tensor | None = None,  # (B, N) True=真实像素
        # ---- v5-β 占位 ----
        true_position_6d: torch.Tensor | None = None,
        pred_position_6d: torch.Tensor | None = None,
    ) -> dict:
        """
        计算总损失

        Args:
            pred_bbox: (B, 4) — 预测 bbox [x1, y1, x2, y2]，值域 [0, 1]
            pred_kpts: (B, K, 2) — 预测关键点 (y, x)，值域 [0, 1]
            true_bbox: (B, 4) — 真值 bbox
            true_crack_pixels: (B, N, 2) — 真值裂纹像素 (y, x)
            gp_module: 可选 GPModule 实例
            pixel_mask: (B, N) 真实像素标记。2026-09-29 必传：负样本的
                -1 padding 坐标若进入 chamfer，会成为 9 万个假像素级
                的错误监督（v5a6 平台解根因，见 _diag_negpad.py）。
                None 时按旧行为（仅单样本无 padding 场景安全）。
            true_position_6d: (B, 6) 真值位姿（v5-β），None → 占位
            pred_position_6d: (B, 6) 预测位姿（v5-β），None → 占位

        Returns:
            dict:
                total: 总损失（标量，可反向传播）
                coverage: 覆盖距离分量（detached）
                bbox: bbox Smooth L1 分量（detached）
                gp: GP LML 分量（detached），gp_module=None 时为 0
                poisson: 泊松先验分量（detached）
                kpt_direct: 关键点直接监督分量（detached）
                position_6d: 位置 6D 分量（detached，v5-β）
                uv_projection: UV 投影分量（detached，v5-β）
        """
        # 1. 覆盖距离（双向 chamfer + padding 掩蔽 + 负样本跳过）
        loss_cov = coverage_loss(
            pred_kpts, true_crack_pixels, M=self.M, mode=self.coverage_mode,
            pixel_mask=pixel_mask,
        )

        # 1b. 关键点直接监督：pred_kpts → 最近真像素 L2（不经样条）
        loss_kpt_direct = kpt_direct_loss(
            pred_kpts, true_crack_pixels, pixel_mask=pixel_mask,
        )

        # 2. bbox Smooth L1
        loss_bbox = self.bbox_loss(pred_bbox, true_bbox)

        # 3. GP LML（可选）
        if gp_module is not None:
            loss_gp = gp_module.log_marginal_likelihood()
        else:
            loss_gp = torch.tensor(0.0, device=pred_bbox.device)

        # 4. 泊松先验
        loss_poisson = poisson_prior(pred_kpts)

        # 5. [v5-β] 位置 6D loss
        loss_pos6d = position_6d_loss(pred_position_6d, true_position_6d)

        # 6. [v5-β] UV 投影 loss（K_matrix 暂为 None → 返回 0）
        loss_uv = uv_projection_loss(
            pred_kpts, pred_position_6d, true_position_6d, K_matrix=None,
        )

        # 总损失
        total = (
            self.lambda_coverage * loss_cov
            + self.lambda_kpt_direct * loss_kpt_direct
            + self.lambda_bbox * loss_bbox
            + self.lambda_gp * loss_gp
            + self.lambda_poisson * loss_poisson
            + self.lambda_position_6d * loss_pos6d
            + self.lambda_uv_projection * loss_uv
        )

        return {
            "total": total,
            "coverage": loss_cov.detach(),
            "kpt_direct": loss_kpt_direct.detach(),
            "bbox": loss_bbox.detach(),
            "gp": loss_gp.detach(),
            "poisson": loss_poisson.detach(),
            "position_6d": loss_pos6d.detach(),
            "uv_projection": loss_uv.detach(),
        }


# ============================================================
#  冒烟测试
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("OrderedKeypointLoss 冒烟测试")
    print("=" * 60)

    B, K = 4, 12
    torch.manual_seed(42)

    # 构造测试数据
    pred_bbox = torch.rand(B, 4)
    pred_kpts = torch.rand(B, K, 2)
    true_bbox = torch.rand(B, 4)
    true_crack_pixels = torch.rand(B, 50, 2)

    # 无 GP
    loss_fn = OrderedKeypointLoss()
    losses = loss_fn(pred_bbox, pred_kpts, true_bbox, true_crack_pixels)
    print(f"  coverage = {losses['coverage'].item():.4f}")
    print(f"  bbox     = {losses['bbox'].item():.4f}")
    print(f"  gp       = {losses['gp'].item():.4f}")
    print(f"  poisson  = {losses['poisson'].item():.4f}")
    print(f"  total    = {losses['total'].item():.4f}")
    assert losses["total"].item() > 0
    assert losses["coverage"].item() > 0

    # 验证反向传播
    pred_kpts.requires_grad_(True)
    losses = loss_fn(pred_bbox, pred_kpts, true_bbox, true_crack_pixels)
    losses["total"].backward()
    assert pred_kpts.grad is not None and pred_kpts.grad.abs().sum() > 0
    print(f"\n  [OK] 反向传播：pred_kpts.grad 非零")

    # 带 GP
    from models.gp_module import GPModule
    gp = GPModule(in_dim=128, hidden_dim=64)
    pred_bbox2 = torch.rand(B, 4)
    pred_kpts2 = torch.rand(B, K, 2)
    losses = loss_fn(pred_bbox2, pred_kpts2, true_bbox, true_crack_pixels, gp_module=gp)
    print(f"\n  带 GP: total = {losses['total'].item():.4f}")
    print(f"        gp   = {losses['gp'].item():.4f}（应 > 0）")
    assert losses["gp"].item() > 0

    print("\n[ALL PASS] ✅")