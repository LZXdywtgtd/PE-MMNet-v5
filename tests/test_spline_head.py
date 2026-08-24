"""
SplineHead 单元测试（V5A2）

测试 SplineHead 的：
- 形状正确性
- bbox 约束（x1 <= x2, y1 <= y2, 值域 [0, 1]）
- 关键点值域 [0, 1]
- K 值域 [min_kpts, max_kpts]
- 推理便捷接口
- 反向传播

详见：
- tasks/team_v5_alpha.json 中 V5A2 测试要求
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.spline_head import SplineHead


# =========================================================
#  SplineHead 测试
# =========================================================

def test_spline_head_init():
    """测试初始化与参数验证"""
    head = SplineHead(in_dim=512, hidden_dim=256, max_kpts=16, min_kpts=8)
    n_params = sum(p.numel() for p in head.parameters())
    assert n_params > 0
    print(f"  [OK] spline_head_init: {n_params:,} 参数")

    # 测试无效参数
    try:
        SplineHead(in_dim=512, min_kpts=1)
        assert False, "应拒绝 min_kpts < 2"
    except ValueError:
        pass

    try:
        SplineHead(in_dim=512, max_kpts=8, min_kpts=16)
        assert False, "应拒绝 max_kpts < min_kpts"
    except ValueError:
        pass
    print(f"  [OK] spline_head_init_validation: 参数边界检查通过")


def test_spline_head_forward_shapes():
    """测试 forward 输出形状"""
    head = SplineHead(in_dim=256, hidden_dim=128, max_kpts=16)
    feat = torch.randn(8, 256)
    out = head(feat)

    assert out["bbox"].shape == (8, 4), f"bbox 应 (8, 4)，实际 {out['bbox'].shape}"
    assert out["keypoints"].shape == (8, 16, 2), \
        f"keypoints 应 (8, 16, 2)，实际 {out['keypoints'].shape}"
    assert out["validity"].shape == (8, 16), \
        f"validity 应 (8, 16)，实际 {out['validity'].shape}"
    assert out["K"].shape == (8,), f"K 应 (8,)，实际 {out['K'].shape}"
    print(f"  [OK] spline_head_forward_shapes: bbox/kpts/validity/K 形状正确")


def test_spline_head_value_ranges():
    """测试输出值域约束"""
    head = SplineHead(in_dim=256, hidden_dim=128, max_kpts=16)
    feat = torch.randn(16, 256)
    out = head(feat)

    # bbox 在 [0, 1]
    bbox = out["bbox"]
    assert bbox.min() >= 0.0 and bbox.max() <= 1.0, \
        f"bbox 应在 [0, 1]，实际 [{bbox.min()}, {bbox.max()}]"
    # 关键点在 [0, 1]
    kpts = out["keypoints"]
    assert kpts.min() >= 0.0 and kpts.max() <= 1.0, \
        f"keypoints 应在 [0, 1]，实际 [{kpts.min()}, {kpts.max()}]"
    # validity 在 [0, 1]
    validity = out["validity"]
    assert validity.min() >= 0.0 and validity.max() <= 1.0, \
        f"validity 应在 [0, 1]，实际 [{validity.min()}, {validity.max()}]"
    # K 在 [min_kpts, max_kpts]
    K = out["K"]
    assert K.min() >= 8 and K.max() <= 16, \
        f"K 应在 [8, 16]，实际 [{K.min()}, {K.max()}]"
    print(f"  [OK] spline_head_value_ranges: bbox ∈[0,1], kpts ∈[0,1], K ∈[8,16]")


def test_spline_head_bbox_ordering():
    """测试 bbox 顺序约束（x1 <= x2, y1 <= y2）"""
    head = SplineHead(in_dim=256, hidden_dim=128)
    feat = torch.randn(32, 256)
    out = head(feat)
    bbox = out["bbox"]

    assert (bbox[:, 0] <= bbox[:, 2]).all(), "x1 <= x2 应成立"
    assert (bbox[:, 1] <= bbox[:, 3]).all(), "y1 <= y2 应成立"
    print(f"  [OK] spline_head_bbox_ordering: x1≤x2 ∧ y1≤y2 全部成立")


def test_spline_head_backward():
    """测试反向传播"""
    head = SplineHead(in_dim=128, hidden_dim=64, max_kpts=16)
    feat = torch.randn(4, 128, requires_grad=True)
    out = head(feat)

    # 模拟 loss：bbox MSE + 关键点与目标的距离
    target_bbox = torch.tensor([[0.2, 0.2, 0.8, 0.8]] * 4)
    target_kpts = torch.rand(4, 16, 2)

    loss_bbox = torch.nn.functional.mse_loss(out["bbox"], target_bbox)
    loss_kpts = torch.nn.functional.mse_loss(out["keypoints"], target_kpts)
    loss = loss_bbox + loss_kpts

    loss.backward()

    # 至少有一些参数梯度非零
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in head.parameters())
    assert has_grad, "至少一个参数应有梯度"
    assert feat.grad is not None and feat.grad.abs().sum() > 0, \
        "输入 feat 应该有梯度"
    print(f"  [OK] spline_head_backward: loss={loss.item():.4f}, 梯度正常")


def test_spline_head_different_min_max():
    """测试不同 min/max_kpts 配置"""
    configs = [
        (8, 12),
        (4, 20),
        (10, 10),  # min == max
    ]
    for min_k, max_k in configs:
        head = SplineHead(in_dim=64, hidden_dim=32, max_kpts=max_k, min_kpts=min_k)
        feat = torch.randn(2, 64)
        out = head(feat)
        assert out["keypoints"].shape == (2, max_k, 2)
        assert out["K"].min() >= min_k
        assert out["K"].max() <= max_k
    print(f"  [OK] spline_head_different_min_max: {len(configs)} 配置全部通过")


def test_spline_head_predict_kpts_only():
    """测试 predict_kpts_only 推理接口"""
    head = SplineHead(in_dim=128, hidden_dim=64, max_kpts=12)
    feat = torch.randn(3, 128)

    # 按 x 排序
    kpts_x = head.predict_kpts_only(feat, sort_by="x")
    assert kpts_x.shape == (3, 12, 2)

    # 按 validity 排序
    kpts_v = head.predict_kpts_only(feat, sort_by="validity")
    assert kpts_v.shape == (3, 12, 2)

    # 无效 sort_by 应报错
    try:
        head.predict_kpts_only(feat, sort_by="invalid")
        assert False, "应拒绝无效的 sort_by"
    except ValueError:
        pass
    print(f"  [OK] spline_head_predict_kpts_only: x/validity 两种排序可用")


def test_spline_head_input_validation():
    """测试输入验证"""
    head = SplineHead(in_dim=128, hidden_dim=64)

    # 错误维度
    try:
        head(torch.randn(4, 64))  # in_dim 应该是 128
        assert False, "应拒绝错误的 in_dim"
    except ValueError:
        pass

    # 错误 batch 维度
    try:
        head(torch.randn(4, 4, 128))  # 应该是 2D
        assert False, "应拒绝非 2D 输入"
    except ValueError:
        pass
    print(f"  [OK] spline_head_input_validation: 错误输入被拒绝")


def run_all():
    """运行所有测试"""
    tests = [
        test_spline_head_init,
        test_spline_head_forward_shapes,
        test_spline_head_value_ranges,
        test_spline_head_bbox_ordering,
        test_spline_head_backward,
        test_spline_head_different_min_max,
        test_spline_head_predict_kpts_only,
        test_spline_head_input_validation,
    ]

    print("=" * 60)
    print(f"SplineHead 单元测试（共 {len(tests)} 项）")
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