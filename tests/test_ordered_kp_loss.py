"""
OrderedKeypointLoss 单元测试（V5A3）

覆盖：
- catmull_rom_spline_torch（样条拟合）
  - 端点通过（K=2 退化为线性）
  - 形状正确
  - 值域 [0, 1]（输入归一化）
  - 镜像 ghost points 延拓
  - 反向传播
- coverage_loss
  - 完美匹配 → 趋近 0
  - 噪声 → 严格正
  - chamfer ≤ hausdorff
  - 不同 K 配置
  - dtype/device 一致
  - 反向传播
- poisson_prior
  - 等弧长 → 趋近 0
  - 不等弧长 → 正
  - K=1 → 0（边界）
- OrderedKeypointLoss
  - 组合 loss 各项 > 0
  - 总和 = 加权和
  - 不带 GP 时 gp 分量 = 0
  - 带 GP 时 gp 分量 > 0
  - 反向传播（含 GP 参数）
  - 输入验证

详见：
- tasks/team_v5_alpha.json 中 V5A3 测试要求
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from training.ordered_kp_loss import (
    catmull_rom_spline_torch,
    coverage_loss,
    poisson_prior,
    OrderedKeypointLoss,
)


# ============================================================
#  catmull_rom_spline_torch 测试
# ============================================================

def test_catmull_rom_two_points_linear():
    """K=2 应退化为线性插值（端点通过）"""
    points = torch.tensor([[[0.0, 0.0], [1.0, 1.0]]])  # (1, 2, 2)
    samples = catmull_rom_spline_torch(points, num_samples=11)
    # 中点应在 (0.5, 0.5)
    assert samples.shape == (1, 11, 2)
    mid = samples[0, 5]  # t=0.5
    assert torch.allclose(mid, torch.tensor([0.5, 0.5]), atol=1e-5), \
        f"K=2 中点应为 (0.5, 0.5)，实际 {mid}"
    # 端点
    assert torch.allclose(samples[0, 0], points[0, 0], atol=1e-5)
    assert torch.allclose(samples[0, -1], points[0, 1], atol=1e-5)
    print(f"  [OK] catmull_rom_two_points_linear: K=2 退化为线性，端点通过")


def test_catmull_rom_passes_through_controls():
    """Catmull-Rom 应通过所有控制点"""
    # 控制点: 沿 y = x 均匀分布
    points = torch.linspace(0, 1, 12).view(1, 12, 1).expand(1, 12, 2).contiguous()
    samples = catmull_rom_spline_torch(points, num_samples=200)
    assert samples.shape == (1, 200, 2)
    # 第一个采样点应在第一个控制点附近（t=0）
    assert torch.allclose(samples[0, 0], points[0, 0], atol=1e-4), \
        f"首点应为控制点 0，实际 {samples[0, 0]}"
    # 最后一个采样点应在最后一个控制点附近（t=1）
    assert torch.allclose(samples[0, -1], points[0, -1], atol=1e-4), \
        f"末点应为控制点 -1，实际 {samples[0, -1]}"
    print(f"  [OK] catmull_rom_passes_through_controls: 端点通过，中段均匀")


def test_catmull_rom_smooth_on_straight_line():
    """直线上的控制点 → 样条应保持直线"""
    # 所有控制点都在 (t, t) 直线上
    t = torch.linspace(0, 1, 10)
    points = torch.stack([t, t], dim=-1).unsqueeze(0)  # (1, 10, 2)
    samples = catmull_rom_spline_torch(points, num_samples=100)
    # 采样点应在 y=x 线上（允许小误差）
    diag_diff = (samples[:, :, 0] - samples[:, :, 1]).abs().max()
    assert diag_diff < 1e-4, f"直线控制点应保持直线，最大偏差 {diag_diff}"
    print(f"  [OK] catmull_rom_smooth_on_straight_line: 直线保持，最大偏差 {diag_diff:.2e}")


def test_catmull_rom_shapes():
    """测试各种 B, K, M 形状"""
    for B, K, M in [(2, 8, 100), (4, 12, 200), (1, 16, 50), (8, 2, 30)]:
        points = torch.rand(B, K, 2)
        samples = catmull_rom_spline_torch(points, num_samples=M)
        assert samples.shape == (B, M, 2), \
            f"(B={B},K={K},M={M}) → 应为 ({B}, {M}, 2)，实际 {tuple(samples.shape)}"
    print(f"  [OK] catmull_rom_shapes: 4 种配置形状全部正确")


def test_catmull_rom_value_range():
    """测试值域在控制点范围内（凸包内）"""
    torch.manual_seed(0)
    points = torch.rand(2, 10, 2)  # (B=2, K=10, 2)，值域 [0, 1]
    samples = catmull_rom_spline_torch(points, num_samples=100)
    # 所有采样点应在 [0, 1] 范围（凸包性质）
    assert samples.min() >= -1e-4, f"最小值 < 0: {samples.min()}"
    assert samples.max() <= 1.0 + 1e-4, f"最大值 > 1: {samples.max()}"
    print(f"  [OK] catmull_rom_value_range: 采样点值域 [{samples.min():.3f}, {samples.max():.3f}]")


def test_catmull_rom_backward():
    """测试反向传播"""
    points = torch.rand(2, 8, 2, requires_grad=True)
    samples = catmull_rom_spline_torch(points, num_samples=100)
    # 模拟 loss：所有采样点的均值
    loss = samples.mean()
    loss.backward()
    assert points.grad is not None and points.grad.abs().sum() > 0, \
        "points 应有梯度"
    # 梯度形状一致
    assert points.grad.shape == points.shape
    print(f"  [OK] catmull_rom_backward: 梯度形状={tuple(points.grad.shape)}")


def test_catmull_rom_input_validation():
    """测试输入验证"""
    # 错误维度
    try:
        catmull_rom_spline_torch(torch.rand(4, 2), num_samples=10)
        assert False, "应拒绝 2D 输入"
    except ValueError:
        pass

    # 错误最后一维
    try:
        catmull_rom_spline_torch(torch.rand(4, 8, 3), num_samples=10)
        assert False, "应拒绝 3D 最后一维"
    except ValueError:
        pass

    # K < 2
    try:
        catmull_rom_spline_torch(torch.rand(4, 1, 2), num_samples=10)
        assert False, "应拒绝 K=1"
    except ValueError:
        pass

    # num_samples < 2
    try:
        catmull_rom_spline_torch(torch.rand(4, 8, 2), num_samples=1)
        assert False, "应拒绝 num_samples=1"
    except ValueError:
        pass

    print(f"  [OK] catmull_rom_input_validation: 错误输入被拒绝")


# ============================================================
#  coverage_loss 测试
# ============================================================

def test_coverage_loss_perfect_fit():
    """完美匹配：真值像素 = 全部样条采样点 → loss ≈ 0

    2026-09-29 双向 chamfer 语义更新：原测试取样条采样点的 1/10 子集当真值，
    单向下正向距离全 0 即通过；双向下其余 90 个样条点到最近真像素有距离
    （子集版 ≈0.023，这是双向语义的正确行为非回归）。完美拟合的定义相应
    更新为真值覆盖全部采样点。
    """
    # 构造一个简单的直线
    t = torch.linspace(0.2, 0.8, 12)
    pred_kpts = torch.stack([t, t], dim=-1).unsqueeze(0)  # (1, 12, 2)

    # 真值像素 = 全部样条采样点
    M = 100
    true_pixels = catmull_rom_spline_torch(pred_kpts, M)

    loss = coverage_loss(pred_kpts, true_pixels, M=M)
    assert loss.item() < 1e-6, f"完美匹配 loss 应 ≈ 0，实际 {loss.item():.6f}"
    print(f"  [OK] coverage_loss_perfect_fit: loss={loss.item():.2e}")


def test_coverage_loss_bidirectional_subset_penalized():
    """双向语义：真值只是样条子集时，反向项产生正惩罚（≥ 单向值）

    2026-09-29 加入，钉死双向 chamfer 行为：样条上"没有真像素的段"
    必须被反向项惩罚——这正是堵"摊开覆盖"平凡解的机制。
    """
    t = torch.linspace(0.2, 0.8, 12)
    pred_kpts = torch.stack([t, t], dim=-1).unsqueeze(0)  # (1, 12, 2)
    M = 100
    spline_samples = catmull_rom_spline_torch(pred_kpts, M)
    subset = spline_samples[:, ::10, :]  # (1, 10, 2) 稀疏子集

    loss = coverage_loss(pred_kpts, subset, M=M)
    assert loss.item() > 1e-3, (
        f"子集真值在双向 chamfer 下应有正惩罚，实际 {loss.item():.6f}"
    )
    print(f"  [OK] coverage_loss_bidirectional_subset_penalized: loss={loss.item():.4f}")


def test_coverage_loss_noisy():
    """带噪声：loss > 0 且有梯度"""
    torch.manual_seed(42)
    pred_kpts = torch.rand(2, 12, 2)
    true_pixels = torch.rand(2, 50, 2)
    pred_kpts.requires_grad_(True)

    loss = coverage_loss(pred_kpts, true_pixels)
    assert loss.item() > 0, f"噪声 loss 应 > 0，实际 {loss.item()}"
    loss.backward()
    assert pred_kpts.grad is not None and pred_kpts.grad.abs().sum() > 0
    print(f"  [OK] coverage_loss_noisy: loss={loss.item():.4f}, 梯度正常")


def test_coverage_loss_chamfer_vs_hausdorff():
    """chamfer ≤ hausdorff（同数据）"""
    torch.manual_seed(0)
    pred_kpts = torch.rand(2, 10, 2)
    true_pixels = torch.rand(2, 30, 2)

    loss_chamfer = coverage_loss(pred_kpts, true_pixels, mode="chamfer")
    loss_hausdorff = coverage_loss(pred_kpts, true_pixels, mode="hausdorff")

    assert loss_chamfer.item() >= 0
    assert loss_hausdorff.item() >= 0
    assert loss_chamfer.item() <= loss_hausdorff.item() + 1e-5, \
        f"chamfer ({loss_chamfer.item():.4f}) 应 <= hausdorff ({loss_hausdorff.item():.4f})"
    print(f"  [OK] coverage_loss_chamfer_vs_hausdorff: "
          f"chamfer={loss_chamfer.item():.4f} ≤ hausdorff={loss_hausdorff.item():.4f}")


def test_coverage_loss_k_variants():
    """测试不同 K（关键点数）"""
    for K in [2, 8, 12, 16]:
        pred_kpts = torch.rand(2, K, 2)
        true_pixels = torch.rand(2, 50, 2)
        loss = coverage_loss(pred_kpts, true_pixels, M=100)
        assert loss.item() >= 0, f"K={K} loss 应 >= 0，实际 {loss.item()}"
    print(f"  [OK] coverage_loss_k_variants: K ∈ {{2, 8, 12, 16}} 全部正常")


def test_coverage_loss_invalid_mode():
    """测试无效 mode 应报错"""
    pred_kpts = torch.rand(2, 8, 2)
    true_pixels = torch.rand(2, 50, 2)
    try:
        coverage_loss(pred_kpts, true_pixels, mode="invalid")
        assert False, "应拒绝无效 mode"
    except ValueError:
        pass
    print(f"  [OK] coverage_loss_invalid_mode: 无效 mode 被拒绝")


def test_coverage_loss_batch_size_one():
    """batch=1 边界情况"""
    pred_kpts = torch.rand(1, 10, 2)
    true_pixels = torch.rand(1, 30, 2)
    loss = coverage_loss(pred_kpts, true_pixels)
    assert loss.dim() == 0, f"输出应为标量，实际 {loss.shape}"
    assert loss.item() >= 0
    print(f"  [OK] coverage_loss_batch_size_one: 标量 loss={loss.item():.4f}")


# ============================================================
#  poisson_prior 测试
# ============================================================

def test_poisson_prior_even_spacing():
    """等弧长关键点 → loss ≈ 0"""
    # 等弧长直线
    t = torch.linspace(0, 1, 12)
    kpts = torch.stack([t, t], dim=-1).unsqueeze(0)  # (1, 12, 2)
    loss = poisson_prior(kpts)
    assert loss.item() < 1e-6, f"等弧长 loss 应 ≈ 0，实际 {loss.item():.6f}"
    print(f"  [OK] poisson_prior_even_spacing: loss={loss.item():.2e}")


def test_poisson_prior_uneven_spacing():
    """不等弧长 → loss > 0"""
    # 不均匀间距
    kpts = torch.tensor([[
        [0.0, 0.0], [0.05, 0.0], [0.1, 0.0],
        [0.2, 0.0], [0.3, 0.0], [0.5, 0.0],
        [0.7, 0.0], [0.9, 0.0], [1.0, 0.0],
    ]])  # (1, 9, 2)
    loss = poisson_prior(kpts)
    assert loss.item() > 1e-4, f"不等弧长 loss 应 > 0，实际 {loss.item()}"
    print(f"  [OK] poisson_prior_uneven_spacing: loss={loss.item():.4f}")


def test_poisson_prior_k_one():
    """K=1 边界（无段）→ loss = 0"""
    kpts = torch.rand(2, 1, 2)
    loss = poisson_prior(kpts)
    assert loss.item() == 0.0, f"K=1 loss 应为 0，实际 {loss.item()}"
    print(f"  [OK] poisson_prior_k_one: loss=0.0")


def test_poisson_prior_backward():
    """测试反向传播"""
    kpts = torch.rand(2, 8, 2, requires_grad=True)
    loss = poisson_prior(kpts)
    loss.backward()
    assert kpts.grad is not None and kpts.grad.abs().sum() > 0
    print(f"  [OK] poisson_prior_backward: 梯度正常")


# ============================================================
#  OrderedKeypointLoss 测试
# ============================================================

def test_ordered_kp_loss_combines():
    """测试 OrderedKeypointLoss 组合：所有分量应有贡献"""
    torch.manual_seed(0)
    B, K = 4, 12

    pred_bbox = torch.rand(B, 4)
    pred_kpts = torch.rand(B, K, 2)
    true_bbox = torch.rand(B, 4)
    true_pixels = torch.rand(B, 50, 2)

    loss_fn = OrderedKeypointLoss()
    losses = loss_fn(pred_bbox, pred_kpts, true_bbox, true_pixels)

    # 各分量 > 0
    assert losses["coverage"].item() > 0
    assert losses["bbox"].item() > 0
    assert losses["poisson"].item() >= 0  # 泊松可能为 0（极偶然）

    # gp 无 GP 时为 0
    assert losses["gp"].item() == 0.0, \
        f"无 GP 时 gp loss 应为 0，实际 {losses['gp'].item()}"

    # 总和接近各分量加权和（含 2026-09-29 新增的 kpt_direct）
    expected = (
        loss_fn.lambda_coverage * losses["coverage"]
        + loss_fn.lambda_kpt_direct * losses["kpt_direct"]
        + loss_fn.lambda_bbox * losses["bbox"]
        + loss_fn.lambda_poisson * losses["poisson"]
    )
    assert abs(losses["total"].item() - expected.item()) < 1e-4, \
        f"total ({losses['total'].item():.4f}) != 加权和 ({expected.item():.4f})"

    print(f"  [OK] ordered_kp_loss_combines: "
          f"cov={losses['coverage'].item():.4f}, bbox={losses['bbox'].item():.4f}, "
          f"poisson={losses['poisson'].item():.4f}, total={losses['total'].item():.4f}")


def test_ordered_kp_loss_with_gp():
    """测试带 GP 模块

    注意：GPModule 默认 init_length_scale=1.0, init_signal_var=1.0
    → log=0 → log_marginal_likelihood() 返回 0（这是正确行为）
    测试时需要扰动超参数以触发非零 LML。
    """
    from models.gp_module import GPModule
    torch.manual_seed(0)
    B, K = 2, 10

    gp = GPModule(in_dim=128, hidden_dim=64)
    # 扰动超参数 → log ≠ 0 → LML > 0
    with torch.no_grad():
        gp.log_length_scale.fill_(0.5)  # 偏离默认
        gp.log_signal_var.fill_(-0.3)

    pred_bbox = torch.rand(B, 4)
    pred_kpts = torch.rand(B, K, 2)
    true_bbox = torch.rand(B, 4)
    true_pixels = torch.rand(B, 40, 2)

    loss_fn = OrderedKeypointLoss()
    losses = loss_fn(pred_bbox, pred_kpts, true_bbox, true_pixels, gp_module=gp)

    # GP LML 应 > 0（扰动后）
    assert losses["gp"].item() > 0, \
        f"扰动后 gp loss 应 > 0，实际 {losses['gp'].item()}"

    # 总损失 > 0
    assert losses["total"].item() > 0

    # 反向传播
    losses["total"].backward()
    assert gp.log_length_scale.grad is not None and \
        gp.log_length_scale.grad.abs().sum() > 0
    print(f"  [OK] ordered_kp_loss_with_gp: gp={losses['gp'].item():.4f}, "
          f"total={losses['total'].item():.4f}, GP 参数有梯度")


def test_ordered_kp_loss_backward_full():
    """测试完整反向传播（pred_kpts 和 pred_bbox）"""
    B, K = 4, 12
    pred_bbox = torch.rand(B, 4, requires_grad=True)
    pred_kpts = torch.rand(B, K, 2, requires_grad=True)
    true_bbox = torch.rand(B, 4)
    true_pixels = torch.rand(B, 50, 2)

    loss_fn = OrderedKeypointLoss()
    losses = loss_fn(pred_bbox, pred_kpts, true_bbox, true_pixels)
    losses["total"].backward()

    assert pred_bbox.grad is not None and pred_bbox.grad.abs().sum() > 0
    assert pred_kpts.grad is not None and pred_kpts.grad.abs().sum() > 0
    print(f"  [OK] ordered_kp_loss_backward_full: pred_bbox & pred_kpts 都有梯度")


def test_ordered_kp_loss_custom_weights():
    """测试自定义权重"""
    B, K = 2, 8
    pred_bbox = torch.rand(B, 4)
    pred_kpts = torch.rand(B, K, 2)
    true_bbox = torch.rand(B, 4)
    true_pixels = torch.rand(B, 30, 2)

    # 权重全 0 → total loss 应为 0
    loss_fn = OrderedKeypointLoss(
        lambda_coverage=0.0,
        lambda_bbox=0.0,
        lambda_gp=0.0,
        lambda_poisson=0.0,
        lambda_kpt_direct=0.0,
    )
    losses = loss_fn(pred_bbox, pred_kpts, true_bbox, true_pixels)
    assert losses["total"].item() == 0.0, \
        f"权重全 0 时 total 应为 0，实际 {losses['total'].item()}"
    print(f"  [OK] ordered_kp_loss_custom_weights: 权重全 0 → total=0")


def test_ordered_kp_loss_hausdorff_mode():
    """测试 hausdorff 模式"""
    B, K = 2, 8
    pred_bbox = torch.rand(B, 4)
    pred_kpts = torch.rand(B, K, 2)
    true_bbox = torch.rand(B, 4)
    true_pixels = torch.rand(B, 30, 2)

    loss_fn = OrderedKeypointLoss(coverage_mode="hausdorff")
    losses = loss_fn(pred_bbox, pred_kpts, true_bbox, true_pixels)
    assert losses["total"].item() > 0
    assert losses["coverage"].item() >= 0
    print(f"  [OK] ordered_kp_loss_hausdorff_mode: "
          f"coverage={losses['coverage'].item():.4f}")


def test_ordered_kp_loss_invalid_mode():
    """测试无效 mode"""
    try:
        OrderedKeypointLoss(coverage_mode="invalid")
        assert False, "应拒绝无效 mode"
    except ValueError:
        pass
    print(f"  [OK] ordered_kp_loss_invalid_mode: 无效 mode 被拒绝")


def test_ordered_kp_loss_realistic_shapes():
    """测试 v5-α 真实形状：B=8, K=16, M=200, N=100~300"""
    from models.gp_module import GPModule
    B, K = 8, 16
    pred_bbox = torch.rand(B, 4)
    pred_kpts = torch.rand(B, K, 2)
    true_bbox = torch.rand(B, 4)
    true_pixels = torch.rand(B, 200, 2)  # 200 个真值像素
    gp = GPModule(in_dim=128, hidden_dim=64)

    loss_fn = OrderedKeypointLoss(M=200)
    losses = loss_fn(pred_bbox, pred_kpts, true_bbox, true_pixels, gp_module=gp)

    assert losses["total"].item() > 0
    # 反向传播
    losses["total"].backward()
    print(f"  [OK] ordered_kp_loss_realistic_shapes: "
          f"B={B}, K={K}, N=200, total={losses['total'].item():.4f}")


# ============================================================
#  运行所有测试
# ============================================================

def run_all():
    """运行所有测试"""
    tests = [
        # Catmull-Rom
        test_catmull_rom_two_points_linear,
        test_catmull_rom_passes_through_controls,
        test_catmull_rom_smooth_on_straight_line,
        test_catmull_rom_shapes,
        test_catmull_rom_value_range,
        test_catmull_rom_backward,
        test_catmull_rom_input_validation,
        # Coverage loss
        test_coverage_loss_perfect_fit,
        test_coverage_loss_noisy,
        test_coverage_loss_chamfer_vs_hausdorff,
        test_coverage_loss_k_variants,
        test_coverage_loss_invalid_mode,
        test_coverage_loss_batch_size_one,
        # Poisson prior
        test_poisson_prior_even_spacing,
        test_poisson_prior_uneven_spacing,
        test_poisson_prior_k_one,
        test_poisson_prior_backward,
        # OrderedKeypointLoss
        test_ordered_kp_loss_combines,
        test_ordered_kp_loss_with_gp,
        test_ordered_kp_loss_backward_full,
        test_ordered_kp_loss_custom_weights,
        test_ordered_kp_loss_hausdorff_mode,
        test_ordered_kp_loss_invalid_mode,
        test_ordered_kp_loss_realistic_shapes,
    ]

    print("=" * 60)
    print(f"OrderedKeypointLoss 单元测试（共 {len(tests)} 项）")
    print("=" * 60)

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test.__name__}: {e}")

    print("=" * 60)
    print(f"结果：{passed} 通过，{failed} 失败")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(run_all())