"""
GPModule 单元测试（V5A2）

测试 GPModule 的：
- 形状正确性（输入输出同形）
- batch=1 边界情况
- RBF 核矩阵对称性
- 长度尺度/方差超参数 clamp
- log_marginal_likelihood 行为
- 反向传播（含超参数）
- 训练态 vs eval 态一致性

详见：
- tasks/team_v5_alpha.json 中 V5A2 测试要求
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.gp_module import GPModule


# =========================================================
#  GPModule 测试
# =========================================================

def test_gp_module_init():
    """测试初始化"""
    gp = GPModule(in_dim=256, hidden_dim=128, init_length_scale=2.0,
                  init_signal_var=0.5)
    n_params = sum(p.numel() for p in gp.parameters())
    assert n_params > 0
    assert abs(gp.length_scale.item() - 2.0) < 1e-5
    assert abs(gp.signal_var.item() - 0.5) < 1e-5
    print(f"  [OK] gp_module_init: {n_params:,} 参数, ℓ=2.0, σ²=0.5")


def test_gp_module_forward_shapes():
    """测试 forward 形状"""
    gp = GPModule(in_dim=256, hidden_dim=128)
    x = torch.randn(8, 256)
    out = gp(x)
    assert out.shape == x.shape, f"输出应保持输入形状 {x.shape}"
    print(f"  [OK] gp_module_forward_shapes: (8, 256) → {tuple(out.shape)}")


def test_gp_module_batch_one():
    """测试 batch=1 边界情况（GP 需要 batch 内多点）"""
    gp = GPModule(in_dim=256, hidden_dim=128)
    x = torch.randn(1, 256)
    out = gp(x)
    assert out.shape == x.shape, "batch=1 应也能 forward"
    # 输出应不是 NaN
    assert not torch.isnan(out).any(), "batch=1 输出不应有 NaN"
    print(f"  [OK] gp_module_batch_one: (1, 256) → {tuple(out.shape)}")


def test_gp_module_rbf_kernel_symmetric():
    """测试 RBF 核矩阵对称性"""
    gp = GPModule(in_dim=128, hidden_dim=64)
    x = torch.randn(10, 128)
    K = gp.rbf_kernel(x, x)
    assert K.shape == (10, 10)
    assert torch.allclose(K, K.T, atol=1e-5), "RBF 核应对称 K[i,j] = K[j,i]"
    print(f"  [OK] gp_module_rbf_kernel_symmetric: K[10x10] 对称性正确")


def test_gp_module_rbf_kernel_values():
    """测试 RBF 核值域（对角线最大，正定）"""
    gp = GPModule(in_dim=64, hidden_dim=32, init_length_scale=1.0,
                  init_signal_var=1.0)
    x = torch.randn(5, 64)
    K = gp.rbf_kernel(x, x)

    # 对角线 = σ² · exp(0) = σ² = 1.0
    diag = K.diag()
    assert torch.allclose(diag, torch.ones(5), atol=1e-4), \
        f"对角线应为 σ²，实际 {diag}"

    # 非对角线 <= 对角线（exp 项 <= 1）
    off_diag = K - torch.diag(K.diag())
    assert (off_diag <= diag.unsqueeze(1) + 1e-6).all(), \
        "非对角线值应 <= 对角线"
    print(f"  [OK] gp_module_rbf_kernel_values: 对角线={diag[0]:.3f}, 正定")


def test_gp_module_length_scale_clamp():
    """测试长度尺度 clamp 到合理范围"""
    gp = GPModule(in_dim=64, hidden_dim=32, init_length_scale=1.0)
    # 故意把 log_length_scale 设为极端值
    with torch.no_grad():
        gp.log_length_scale.fill_(20.0)  # 极大
    ell = gp.length_scale.item()
    assert ell <= 10.0, f"ℓ 应 clamp 到 <= 10，实际 {ell}"

    with torch.no_grad():
        gp.log_length_scale.fill_(-20.0)  # 极小
    ell = gp.length_scale.item()
    assert ell >= 0.1, f"ℓ 应 clamp 到 >= 0.1，实际 {ell}"
    print(f"  [OK] gp_module_length_scale_clamp: ℓ ∈ [0.1, 10.0]")


def test_gp_module_log_marginal_likelihood():
    """测试 log_marginal_likelihood 行为"""
    gp = GPModule(in_dim=64, hidden_dim=32)
    lml = gp.log_marginal_likelihood()
    assert lml.item() >= 0, f"LML 应 >= 0，实际 {lml.item()}"
    # LML 对超参数敏感
    with torch.no_grad():
        gp.log_length_scale.fill_(1.0)  # 偏离 0
        gp.log_signal_var.fill_(1.0)
    lml2 = gp.log_marginal_likelihood()
    assert lml2.item() > lml.item(), "偏离 0 应增大 LML（鼓励保持默认）"
    print(f"  [OK] gp_module_log_marginal_likelihood: 偏离默认 → LML 增大")


def test_gp_module_backward():
    """测试反向传播（含超参数梯度）"""
    gp = GPModule(in_dim=128, hidden_dim=64)
    x = torch.randn(8, 128, requires_grad=True)
    out = gp(x)
    target = torch.randn(8, 128)
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()

    # 输入梯度
    assert x.grad is not None and x.grad.abs().sum() > 0, "输入应有梯度"
    # 超参数梯度
    assert gp.log_length_scale.grad is not None and \
           gp.log_length_scale.grad.abs().sum() > 0, "ℓ 超参数应有梯度"
    assert gp.log_signal_var.grad is not None and \
           gp.log_signal_var.grad.abs().sum() > 0, "σ² 超参数应有梯度"
    print(f"  [OK] gp_module_backward: ℓ/σ² 都有梯度，loss={loss.item():.4f}")


def test_gp_module_train_eval_consistency():
    """测试 train/eval 态输出稳定性"""
    gp = GPModule(in_dim=64, hidden_dim=32)
    x = torch.randn(8, 64)

    gp.train()
    out_train = gp(x)

    gp.eval()
    with torch.no_grad():
        out_eval = gp(x)

    # GP 无 dropout/batchnorm，train/eval 应等价
    assert torch.allclose(out_train, out_eval, atol=1e-5), \
        "GP 无随机性，train/eval 输出应一致"
    print(f"  [OK] gp_module_train_eval_consistency: 输出一致")


def test_gp_module_non_learnable():
    """测试超参数不可学习模式"""
    gp = GPModule(in_dim=64, hidden_dim=32, learnable=False)
    # log_length_scale 应是 buffer 不是 parameter
    param_names = [n for n, _ in gp.named_parameters()]
    buffer_names = [n for n, _ in gp.named_buffers()]
    assert "log_length_scale" not in param_names
    assert "log_length_scale" in buffer_names
    print(f"  [OK] gp_module_non_learnable: 超参数为 buffer")


def run_all():
    """运行所有测试"""
    tests = [
        test_gp_module_init,
        test_gp_module_forward_shapes,
        test_gp_module_batch_one,
        test_gp_module_rbf_kernel_symmetric,
        test_gp_module_rbf_kernel_values,
        test_gp_module_length_scale_clamp,
        test_gp_module_log_marginal_likelihood,
        test_gp_module_backward,
        test_gp_module_train_eval_consistency,
        test_gp_module_non_learnable,
    ]

    print("=" * 60)
    print(f"GPModule 单元测试（共 {len(tests)} 项）")
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