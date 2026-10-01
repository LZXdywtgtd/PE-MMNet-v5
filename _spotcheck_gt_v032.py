# -*- coding: utf-8 -*-
"""V5-032 阶段 2 抽检：重生成后 mask 无边框污染。

① 指定抽检 20 样本逐个打印边缘覆盖；
② 全量扫描：正样本中"双邻边 L 形"必须为 0（旧缓存 val 11/train 37）；
③ GT 自洽下界抽查：修复后正样本 GT kpts→GT 像素 chamfer 应贴线上界。
"""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import functools
print = functools.partial(print, flush=True)
import numpy as np
import torch

PX = 128
REQ = 0.0441

def edges_of(px_flat):
    gy = np.rint(px_flat[:, 0] * (PX - 1)).astype(int)
    gx = np.rint(px_flat[:, 1] * (PX - 1)).astype(int)
    t = int((gy <= 1).sum()); b = int((gy >= PX - 2).sum())
    l = int((gx <= 1).sum()); r = int((gx >= PX - 2).sum())
    return t, b, l, r

def scan(path, tag, spot_ids):
    d = np.load(path)
    n = d["x_2d"].shape[0]
    pos = lshape = 0
    worst_edge = 0
    floors = []
    for i in range(n):
        if d["pixel_mask"][i].sum() < 2:
            continue
        pos += 1
        px = d["true_crack_pixels"][i][d["pixel_mask"][i]]
        t, b, l, r = edges_of(px)
        nhit = sum(1 for v in (t, b, l, r) if v >= 100)
        worst_edge = max(worst_edge, nhit)
        if nhit >= 2:
            lshape += 1
            print(f"  [污染] {tag} idx={i}: T{t} B{b} L{l} R{r}")
        if i in spot_ids:
            gy = np.rint(px[:, 0] * (PX-1)).astype(int)
            gx = np.rint(px[:, 1] * (PX-1)).astype(int)
            grid = np.zeros((PX, PX), bool); grid[gy, gx] = True
            rows = [int(grid[r_].sum()) for r_ in (0, 1, 126, 127)]
            cols = [int(grid[:, c].sum()) for c in (0, 1, 126, 127)]
            print(f"  [抽检] idx={i}: n_px={int(grid.sum())} "
                  f"行0/1/126/127={rows} 列0/1/126/127={cols}")
    print(f"[{tag}] 正样本 {pos}，双邻边 L 形污染 {lshape}，"
          f"单样本最多命中边数 {worst_edge}")
    d.close()
    return pos, lshape

print("== 抽检 20（旧污染 11 个 + 干净 9 个）==")
old_polluted = {2, 19, 32, 59, 99, 106, 118, 144, 151, 168, 194}
spots = set(sorted(old_polluted)[:11]) | {0, 1, 3, 4, 5, 6, 10, 20, 30}
spots = set(sorted(spots)[:20])

p1, l1 = scan("logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz",
              "val", spots)
p2, l2 = scan("logs/sim_cache/sim_cache_p128_n1000_s424242_t6eb66e1b_th60.0_dp.npz",
              "train", set())

print()
if l1 > 0 or l2 > 0:
    print(f"[FAIL] 仍有 L 形污染 val={l1} train={l2}——停在阶段 2")
    sys.exit(1)

# ③ GT 自洽下界（val 前 60 正样本快速口径）
from _diag_v5b_direction2 import cham_of
from training.ordered_kp_loss import catmull_rom_spline_torch
d = np.load("logs/sim_cache/sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz")
fl = []
for i in range(d["x_2d"].shape[0]):
    if d["pixel_mask"][i].sum() < 2:
        continue
    px = d["true_crack_pixels"][i][d["pixel_mask"][i]].astype(np.float64)
    kt = d["true_keypoints"][i][d["keypoint_mask"][i]].astype(np.float64)
    fl.append(cham_of(kt, px))
    if len(fl) >= 60:
        break
fl = np.array(fl)
print(f"GT 自洽下界（val 前 {len(fl)} 正样本）：p50={np.median(fl):.4f} "
      f"max={fl.max():.4f} 超阈率={float((fl > REQ).mean())*100:.1f}%"
      f"（旧缓存同口径 p50≈0.0286 超阈率 18%——修复后期望超阈率大降）")
d.close()
print("[阶段2 抽检 PASS]")
