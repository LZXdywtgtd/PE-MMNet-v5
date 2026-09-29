# -*- coding: utf-8 -*-
"""loss 语义回归测试：满面 GT vs 线状 GT 下，修复后 loss 能否区分好解/坏解

外部 AI 提出的独立验证（2026-09-29）：更新后的单测只验证数值正确性，
未验证语义正确性——"在真实场景下 loss 能否区分'真关键点'和'随机点'"。

判定标准（OrderedKeypointLoss 修复目标）：
- 线状 GT：真关键点 loss < 随机点 loss（排序正确 = 修复生效）
- 满面 GT：这是关键独立验证——满面退化下任何 loss 都可能给出病态排序；
  若修复后满面 GT 下真关键点仍优于随机点，说明 loss 对满面有鲁棒性；
  若反转，则证明"loss 修复不解决满面问题"的结论（V5-021）从 loss 侧独立成立。

满面 GT 构造：真像素 = [0,1]² 均匀网格 40×40（全覆盖），
对应"GT 要求覆盖全场"的退化场景。
"""
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import functools

    print = functools.partial(print, flush=True)
    import numpy as np
    import torch

    from training.ordered_kp_loss import OrderedKeypointLoss

    torch.manual_seed(0)
    loss_fn = OrderedKeypointLoss()  # 默认权重（含 kpt_direct=0.5）

    B, K = 4, 12

    def candidates(B, K, rng):
        corner = torch.tensor(
            [[1., 1.], [0., 0.], [0., 1.], [1., 0.]] * 4
        ).unsqueeze(0).repeat(B, 1, 1)  # 模型现状（角点塌缩）
        rand = torch.from_numpy(
            rng.uniform(0, 1, (B, K, 2)).astype(np.float32))
        return corner, rand

    def pixel_grid(n_side):
        g = np.linspace(0.02, 0.98, n_side)
        yy, xx = np.meshgrid(g, g)
        pts = np.stack([yy.ravel(), xx.ravel()], axis=-1)
        return torch.from_numpy(pts.astype(np.float32)).unsqueeze(0).repeat(B, 1, 1)

    rng = np.random.default_rng(0)

    scenarios = {
        "线状GT（修复目标场景）": pixel_grid(3),       # 3×3 稀疏网格≈线状
        "团块GT（区域级）": pixel_grid(8),
        "满面GT（退化场景）": pixel_grid(40),          # 1600 像素全覆盖
    }

    print(f"{'场景':<24}{'角点(模型现状)':>14}{'随机点':>10}{'正确排序?':>10}")
    for name, gt in scenarios.items():
        corner, rand = candidates(B, K, rng)
        l_corner = loss_fn(
            torch.rand(B, 4), corner, torch.rand(B, 4), gt)["total"].item()
        l_rand = loss_fn(
            torch.rand(B, 4), rand, torch.rand(B, 4), gt)["total"].item()
        ok = "是" if l_corner > l_rand else "否——loss 奖励塌缩解"
        print(f"{name:<24}{l_corner:>14.4f}{l_rand:>10.4f}{ok:>12}")

    print("\n判读：")
    print("- 线状 GT 排序正确 → loss 修复对目标场景生效")
    print("- 满面 GT 若排序错误 → 独立证明 loss 无法在满面 GT 上工作，")
    print("  V5-021 的'必须修 GT 生成'从 loss 侧成立")
