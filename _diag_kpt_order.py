# -*- coding: utf-8 -*-
"""V5-030 第 2 步：pred 控制点顺序 vs GT 关键点顺序诊断（用户 5 分钟诊断）。

对 10 个 val 正样本：
  1. GT 关键点序列（沿裂纹端到端顺序，来自缓存 true_keypoints）
  2. pred 16 控制点（模型输出顺序）
  3. 相邻 pred 点连线 vs 相邻 GT 点连线的平均角度差（按配对序逐段比，
     带方向反转容差——pred 沿裂纹反向遍历不算乱序）

判定规则（用户口径）：
  角度差大（顺序乱）          → 候选 1（有序监督）对症
  角度差小但距裂纹远（位置偏）→ 候选 1 不够，需加 validity 监督

用法：python _diag_kpt_order.py [ckpt_path]
"""
import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

import functools

print = functools.partial(print, flush=True)

import numpy as np
import torch

from models import create_v5_model
from training.trainer_v5 import load_checkpoint

CKPT = sys.argv[1] if len(sys.argv) > 1 else \
    "logs/training_history/v5a6_dprime_resnet18/checkpoints/latest.pt"
VAL_CACHE = "logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0.npz"
N_SAMPLES = 10


def seg_angle_diff(pred: np.ndarray, gt: np.ndarray) -> float:
    """相邻连线角度差（度），带方向反转容差。逐配对段 i→i+1。"""
    n_seg = min(len(pred), len(gt)) - 1
    if n_seg < 1:
        return float("nan")
    dp = pred[1:n_seg + 1] - pred[:n_seg]
    dg = gt[1:n_seg + 1] - gt[:n_seg]
    np_len = np.linalg.norm(dp, axis=1)
    ng_len = np.linalg.norm(dg, axis=1)
    ok = (np_len > 1e-8) & (ng_len > 1e-8)
    if not ok.any():
        return float("nan")
    dp = dp[ok] / np_len[ok, None]
    dg = dg[ok] / ng_len[ok, None]
    dot = np.clip((dp * dg).sum(axis=1), -1.0, 1.0)
    ang = np.degrees(np.arccos(dot))
    ang = np.minimum(ang, 180.0 - ang)  # 反向遍历容差
    return float(ang.mean())


def main():
    ckpt = torch.load(CKPT, map_location="cuda", weights_only=False)
    print(f"ckpt epoch: {ckpt.get('epoch')}")
    model = create_v5_model(
        "resnet18", image_channels=3, image_size=128,
        pretrained_2d=False, use_gp=True, min_kpts=8, max_kpts=16,
    )
    load_checkpoint(CKPT, model, map_location="cuda")
    model.eval().cuda()

    d = np.load(VAL_CACHE)
    px_mask = d["pixel_mask"]
    pos_idx = np.where(px_mask.sum(axis=1) > 0)[0]
    sel = pos_idx[:N_SAMPLES]
    print(f"样本（val 正样本前 {N_SAMPLES} 个）: {sel.tolist()}\n")

    x2 = torch.from_numpy(d["x_2d"][sel]).float().cuda()
    x1 = torch.from_numpy(d["x_1d"][sel]).float().cuda()
    with torch.no_grad():
        out = model(x1, x2)
    preds = out["keypoints"].cpu().numpy()  # (N, 16, 2)

    print(f"{'样本':>4} {'K_gt':>4} {'角度差°':>8} {'pred→裂纹px':>11} "
          f"{'pred→最近GT点':>12}")
    angles, dists_px, dists_kpt = [], [], []
    for j, i in enumerate(sel):
        gt = d["true_keypoints"][i][d["keypoint_mask"][i]]      # (K,2) 有序
        px = d["true_crack_pixels"][i][d["pixel_mask"][i]]       # (P,2)
        pr = preds[j]                                            # (16,2)
        K = len(gt)
        ang = seg_angle_diff(pr, gt)
        # pred 点→最近真像素距离（128px 口径，×128 转 px）
        d_px = np.linalg.norm(pr[:, None, :] - px[None, :, :], axis=2).min(axis=1) * 128
        # pred 点→最近 GT 关键点距离
        d_kp = np.linalg.norm(pr[:, None, :] - gt[None, :, :], axis=2).min(axis=1) * 128
        angles.append(ang)
        dists_px.append(d_px.mean())
        dists_kpt.append(d_kp.mean())
        print(f"{i:>4} {K:>4} {ang:>8.1f} {d_px.mean():>11.1f} {d_kp.mean():>12.1f}")

    print(f"\n=== 汇总（{N_SAMPLES} 样本） ===")
    print(f"平均角度差: {np.nanmean(angles):.1f}°  "
          f"(顺序对≈小角度；乱序≈接近随机 45~90°)")
    med_list = []
    for j, i in enumerate(sel):
        pxi = d["true_crack_pixels"][i][d["pixel_mask"][i]]
        dmin = np.linalg.norm(preds[j][:, None, :] - pxi[None, :, :], axis=2).min(axis=1)
        med_list.append(np.median(dmin) * 128)
    print(f"pred→裂纹 px 距离(逐样本中位数的均值): {np.mean(med_list):.1f}px")
    print(f"pred→最近GT点平均: {np.mean(dists_kpt):.1f}px")
    d.close()


if __name__ == "__main__":
    main()
