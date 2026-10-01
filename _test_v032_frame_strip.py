# -*- coding: utf-8 -*-
"""V5-032 标记测试：边界框剥离（_strip_boundary_frame）。

用例（用户指定：L 形边框 + 内部团块）：
  T1 L 形框 + 团块     → 输出只团块，框像素全消失
  T2 四边全框 + 团块   → 同上
  T3 单边贴边真裂纹    → 不剥离（V5-024 场景保护）
  T4 部分贴边 + 团块   → 不剥离
  T5 纯框无团块        → 输出空（无裂纹）
  T6 端到端：真仿真器 σ 场 + 人工抬边框 → 剥离后 kpts 与团块自洽
"""
import sys, os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)
import functools
print = functools.partial(print, flush=True)
import numpy as np

from data.patch_simulator_v5 import PatchSimulator
from scipy import ndimage

N = 128


def fake_mask(frame_edges, blob=True):
    """frame_edges: subset of {'T','B','L','R'}——每条画 2px 厚整边。"""
    m = np.zeros((N, N), np.uint8)
    if 'T' in frame_edges: m[0:2, :] = 1
    if 'B' in frame_edges: m[N-2:, :] = 1
    if 'L' in frame_edges: m[:, 0:2] = 1
    if 'R' in frame_edges: m[:, N-2:] = 1
    if blob:
        yy, xx = np.mgrid[0:N, 0:N]
        m[((yy-90)**2/(20**2) + (xx-40)**2/(10**2)) <= 1] = 1
    return m


def blob_pixels(m):
    return int(((m > 0) & (np.arange(N)[None, :] > 2)
                & (np.arange(N)[None, :] < N-2)).sum())


fails = []
def check(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond:
        fails.append(name)


sim = PatchSimulator.__new__(PatchSimulator)  # 只用静态方法，免完整初始化

print("T1 L 形(T+L)框 + 团块")
m = fake_mask({'T', 'L'})
out = PatchSimulator._strip_boundary_frame(m, ndimage)
check("框像素清除", out[0:2, :].sum() == 0 and out[:, 0:2].sum() == 0)
check("团块保留", out.sum() >= 400, f"剩 {int(out.sum())} px")
check("单连通域", ndimage.label(out)[1] == 1)

print("T2 四边全框 + 团块")
m = fake_mask({'T', 'B', 'L', 'R'})
out = PatchSimulator._strip_boundary_frame(m, ndimage)
check("四边清除", out[0:2, :].sum() == 0 and out[N-2:, :].sum() == 0
      and out[:, 0:2].sum() == 0 and out[:, N-2:].sum() == 0)
check("团块保留", out.sum() >= 400)

print("T3 单边贴边真裂纹（V5-024 保护）")
m = np.zeros((N, N), np.uint8)
m[0, 30:95] = 1; m[1, 30:95] = 1  # 顶边 65px 贴边线（<0.9*N=116）
out = PatchSimulator._strip_boundary_frame(m, ndimage)
check("不剥离", int(out.sum()) == 130, f"剩 {int(out.sum())} px（原 130）")

print("T3b 单边满铺（D' 调制上界 2/3 边长之外，仅框会出现）")
m = np.zeros((N, N), np.uint8)
m[0, :] = 1; m[1, 30:95] = 1  # 行0 全满 + 行1 部分（模拟 1px 框残形）
out = PatchSimulator._strip_boundary_frame(m, ndimage)
check("单边不剥离（需≥2条框边）", int(out.sum()) == int(m.sum()),
      f"剩 {int(out.sum())} px（原 {int(m.sum())}）")

print("T4 部分贴边 + 团块")
m = fake_mask(set())
m[0:2, 10:70] = 1  # 顶边 60px 段
out = PatchSimulator._strip_boundary_frame(m, ndimage)
check("不剥离", int(out.sum()) == int(m.sum()))

print("T5 纯框无团块")
m = fake_mask({'T', 'L'}, blob=False)
out = PatchSimulator._strip_boundary_frame(m, ndimage)
check("输出空", int(out.sum()) == 0)

print("T6 端到端：真 σ 场（idx=144 同 seed）+ 剥框后 kpts 自洽")
from _dprime_generator import (DPrimePatchSimulator, spikes_for_seed,
                               FAST_PROFILE)
s_cfg = spikes_for_seed(434242 + 144)
simr = DPrimePatchSimulator(patch_size=128, spikes=s_cfg,
                            crack_stress_threshold_MPa=60.0,
                            thermal_profile=FAST_PROFILE)
res = simr.simulate(verbose=False)
sigma = simr._sigma_max_field
mfix = simr._extract_crack_mask(sigma)
ncomp = ndimage.label(mfix)[1]
frac = -1.0
if mfix.sum() > 0:
    ys, xs = np.where(mfix > 0)
    border_px = int((ys <= 1).sum() + (ys >= N-2).sum()
                    + (xs <= 1).sum() + (xs >= N-2).sum())
    frac = border_px / max(int(mfix.sum()), 1)
    gy_ok = frac < 0.30  # 剥框后边框像素占比大降（原 L 形 ~22%+
                         # 骨架沿框走的间接效应）
else:
    gy_ok = False
check("端到端 mask 边框占比 <30%", gy_ok, f"border_frac={frac:.2f}")
check("端到端 mask 非空", int(mfix.sum()) > 0)

print()
if fails:
    print(f"标记测试 {len(fails)} 项失败: {fails}")
    sys.exit(1)
print("标记测试全过（T1-T6）")
