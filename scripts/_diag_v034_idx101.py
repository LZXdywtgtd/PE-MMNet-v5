# -*- coding: utf-8 -*-
"""V5-034 核实：idx=101 GT 骨架崩塌的成因定性（P2 结案用，一次性）。

已知问题口径：GT kpts 崩塌在 2×7px 角落（行0-1/列116-122）而 mask
~2300px 蔓延 101×29px。本脚本回答三个问题：
  Q1 mask 是否贴边（说明骨架提取在边界处截断）
  Q2 mask 连通分量形态（主分量的 bbox / 细长度）
  Q3 kpts 采样是否只落在 mask 的一个角落子区
"""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import functools
print = functools.partial(print, flush=True)
import numpy as np
from scipy import ndimage

VAL_CACHE = "logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz"
IDX = 101
PX = 128

d = np.load(VAL_CACHE)
mask = d["pixel_mask"][IDX]
xy = d["true_crack_pixels"][IDX][mask][:, ::-1]   # (N,2) yx 归一化 → xy
xy_px = xy * PX
kpts = d["true_keypoints"][IDX][d["keypoint_mask"][IDX]]  # (K,2) 归一化 yx

print(f"idx={IDX}  mask 像素={mask.sum()}  kpts 数={len(kpts)}")
ys, xs = xy_px[:, 1], xy_px[:, 0]
print(f"mask bbox: 行[{ys.min():.0f},{ys.max():.0f}] 列[{xs.min():.0f},{xs.max():.0f}]  "
      f"(扩展 {ys.max()-ys.min()+1:.0f}×{xs.max()-xs.min()+1:.0f})")

# Q1 贴边检查
on_edge = int(((ys < 1) | (ys > PX - 2) | (xs < 1) | (xs > PX - 2)).sum())
print(f"Q1 边缘 1px 带内像素: {on_edge}/{len(ys)} ({on_edge/len(ys)*100:.1f}%)")

# Q2 连通分量（栅格化）
grid = np.zeros((PX, PX), dtype=bool)
grid[np.clip(ys.astype(int), 0, PX-1), np.clip(xs.astype(int), 0, PX-1)] = True
lab, ncomp = ndimage.label(grid)
print(f"Q2 连通分量: {ncomp}")
for c in range(1, min(ncomp, 5) + 1):
    cys, cxs = np.where(lab == c)
    print(f"   分量{c}: {len(cys)}px  行[{cys.min()},{cys.max()}] "
          f"列[{cxs.min()},{cxs.max()}]")

# Q3 kpts 分布
ky, kx = kpts[:, 0] * PX, kpts[:, 1] * PX
print(f"Q3 kpts 像素范围: 行[{ky.min():.1f},{ky.max():.1f}] "
      f"列[{kx.min():.1f},{kx.max():.1f}]")
row_cov = (ky.max() - ky.min() + 1) / (ys.max() - ys.min() + 1) * 100
col_cov = (kx.max() - kx.min() + 1) / (xs.max() - xs.min() + 1) * 100
print(f"   kpts 覆盖 mask bbox: 行 {row_cov:.0f}% × 列 {col_cov:.0f}%")
d.close()
