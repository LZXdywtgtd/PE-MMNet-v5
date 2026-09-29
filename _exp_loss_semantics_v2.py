# -*- coding: utf-8 -*-
"""loss 语义测试 v2：用真实形态的线状 GT（连续线状像素带）而非稀疏网格

v1 的 3×3 稀疏网格不是线状——9 个孤立点时样条（连续曲线）天然无法
同时贴近所有点，双向 chamfer 惩罚样条全长，角点+中心折衷解可能更优。
这测的是"样条模型 vs 离散点集"的失配，不是 loss 语义错误。

v2 用真实裂纹形态：连续像素带（宽 3px 的折线 + 高斯扰动），
与 _diag_negpad.py 的真实裂纹子集一致。
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
    loss_fn = OrderedKeypointLoss()
    B, K = 4, 12

    def line_pixels(B, n_px=300, width=0.02):
        """对角连续线带 + 正弦弯曲（像素级连续，不是稀疏点）"""
        t = np.linspace(0.05, 0.95, n_px)
        y = t + 0.08 * np.sin(2 * np.pi * t)
        x = t
        y = y + np.random.default_rng(3).normal(0, width, n_px)
        pts = np.stack([y, x], axis=-1).astype(np.float32)
        return torch.from_numpy(pts).unsqueeze(0).repeat(B, 1, 1)

    def blob_pixels(B, n_px=300, radius=0.18):
        """内部圆斑（区域级 GT）"""
        rng = np.random.default_rng(5)
        r = radius * np.sqrt(rng.uniform(0, 1, n_px))
        a = rng.uniform(0, 2 * np.pi, n_px)
        y = 0.5 + r * np.cos(a)
        x = 0.5 + r * np.sin(a)
        pts = np.stack([y, x], axis=-1).astype(np.float32)
        return torch.from_numpy(pts).unsqueeze(0).repeat(B, 1, 1)

    def full_pixels(B, n_px=1600):
        g = np.linspace(0.02, 0.98, 40)
        yy, xx = np.meshgrid(g, g)
        pts = np.stack([yy.ravel(), xx.ravel()], axis=-1).astype(np.float32)
        return torch.from_numpy(pts).unsqueeze(0).repeat(B, 1, 1)

    def true_like_kpts(B):
        """真关键点：沿线取 12 点（模型理想输出）"""
        t = np.linspace(0.1, 0.9, K)
        y = t + 0.08 * np.sin(2 * np.pi * t)
        pts = np.stack([y, t], axis=-1).astype(np.float32)
        return torch.from_numpy(pts).unsqueeze(0).repeat(B, 1, 1)

    rng = np.random.default_rng(0)
    corner = torch.tensor(
        [[1., 1.], [0., 0.], [0., 1.], [1., 0.]] * 4
    ).unsqueeze(0).repeat(B, 1, 1)
    rand = torch.from_numpy(rng.uniform(0, 1, (B, K, 2)).astype(np.float32))
    true_k = true_like_kpts(B)

    scenarios = {
        "线状GT(连续带300px)": line_pixels(B),
        "团块GT(圆斑300px)": blob_pixels(B),
        "满面GT(1600px全覆盖)": full_pixels(B),
    }

    print(f"{'场景':<22}{'真kpts(理想)':>12}{'随机点':>9}{'角点塌缩':>9}  排序(理想<随机?)")
    for name, gt in scenarios.items():
        l_true = loss_fn(torch.rand(B, 4), true_k, torch.rand(B, 4), gt)["total"].item()
        l_rand = loss_fn(torch.rand(B, 4), rand, torch.rand(B, 4), gt)["total"].item()
        l_corner = loss_fn(torch.rand(B, 4), corner, torch.rand(B, 4), gt)["total"].item()
        ok = "是" if l_true < l_rand else "否"
        print(f"{name:<22}{l_true:>12.4f}{l_rand:>9.4f}{l_corner:>9.4f}   {ok}")
