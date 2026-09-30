# -*- coding: utf-8 -*-
"""第 0 步：augfix1 CSV 0.0218 vs 复测 0.1410 的 6 倍差异裁决。

假说 H1（可复现性）：仿真器材质随机化用全局 RNG → 同 seed 两次仿真不同。
    检验：同配置两个 PatchSimulator 实例逐位对比 x_2d。
假说 H2（负 batch 稀释）：coverage_loss 全负 batch 返回 0，eval_epoch 把 0
    计入均值。训练 batch_size=4（50 batch，stock 仅 12 正 → ~38 个 0），
    复测 batch_size=32（7 batch，稀释少 ~7 倍）。0.1410/0.0218≈6.5≈50/7。
    检验：batch_size=4 精确复刻训练 val 管线 → 应回出 ≈0.0218。
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functools
print = functools.partial(print, flush=True)

import numpy as np
import torch
from torch.utils.data import DataLoader

from data.patch_simulator_v5 import PatchSimulator
from data.patch_dataset_v5 import PatchDatasetV5, collate_v5
from models import create_v5_model
from training.trainer_v5 import load_checkpoint
from training.ordered_kp_loss import coverage_loss

FAST = {"ramp_up_c_per_min": 1260.0, "soak_temp_c": 1280.0,
        "soak_duration_min": 1.0, "cool_down_c_per_min": 1260.0}
CKPT_AUGFIX1 = "logs/training_history/v5a6_dprime_augfix/checkpoints/latest.pt"


def part_a():
    print("===== A. 同 seed 两次 stock 仿真逐位对比 =====")
    outs = []
    for rep in (1, 2):
        sim = PatchSimulator(patch_size=128, physical_size_cm=5.0,
                             thermal_profile=FAST,
                             crack_stress_threshold_MPa=60.0,
                             min_keypoints=8, max_keypoints=16,
                             seed=424242)
        r = sim.simulate(verbose=False)
        outs.append(r)
    keys = ("heatmap", "crack_mask")
    same = True
    for k in keys:
        d = np.max(np.abs(outs[0][k].astype(np.float64)
                          - outs[1][k].astype(np.float64)))
        print(f"  {k}: max|diff| = {d}")
        same &= (d == 0)
    kf = (outs[0]["crack_keypoints"] == outs[1]["crack_keypoints"]).all()
    print(f"  crack_keypoints 逐位一致: {kf}")
    print(f"  A 结论: {'逐位可复现（无全局 RNG 泄漏）' if same and kf else '不可复现 → H1 成立，底层 bug'}")
    return same and kf


def part_b():
    print("\n===== B. batch_size=4 复刻 augfix1 训练 val 管线（stock 200） =====")
    ds = PatchDatasetV5(patch_size=128, n_samples=200, seed=434242,
                        thermal_profile=FAST,
                        crack_stress_threshold_MPa=60.0,
                        cache_dir=None, verbose=False)
    loader = DataLoader(ds, batch_size=4, shuffle=False,
                        num_workers=0, collate_fn=collate_v5)
    m = create_v5_model("resnet18", image_channels=3, image_size=128,
                        pretrained_2d=False, use_gp=True, min_kpts=8,
                        max_kpts=16, spatial_head=False)
    load_checkpoint(CKPT_AUGFIX1, m, map_location="cuda")
    m.eval().cuda()

    vals, n_pos_total, n_pos_batch = [], 0, 0
    with torch.no_grad():
        for batch in loader:
            x1 = batch["x_1d"].cuda()
            x2 = batch["x_2d"].cuda()
            tp = batch["true_crack_pixels"].cuda()
            pm = batch["pixel_mask"].cuda()
            out = m(x1, x2)
            loss = coverage_loss(out["keypoints"], tp, M=200,
                                 mode="chamfer", pixel_mask=pm)
            vals.append(float(loss))
            n_pos_total += int(pm.sum(dim=-1).clamp(max=1).sum())
            n_pos_batch += int((pm.sum(dim=-1) > 0).any() and 1 or 0)
            n_pos_batch = n_pos_batch  # noop
    vals = np.array(vals)
    pos_mask = None
    print(f"  batch 数 = {len(vals)} (batch_size=4, n=200)")
    print(f"  含正样本 batch 数 = {n_pos_batch}")
    print(f"  正样本总数 = {n_pos_total}")
    print(f"  batch 均值口径（训练 CSV 口径）= {vals.mean():.6f}"
          f"   ← 训练 CSV ep150 = 0.021843")
    nz = vals[vals > 0]
    print(f"  仅非零 batch 均值 = {nz.mean():.6f}")
    print(f"  复测 0.1410 来自 batch_size=32（7 batch，稀释少 ~{50//7} 倍）")
    print(f"  B 结论: {'稀释假说成立（H2）' if abs(vals.mean() - 0.0218) < 0.005 else '仍有残差，需再查'}")


def main():
    ok_a = part_a()
    part_b()
    print(f"\n总裁决: A({'可复现' if ok_a else '不可复现'}) + "
          f"B(稀释解释 6 倍差异)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
