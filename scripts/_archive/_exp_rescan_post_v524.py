# -*- coding: utf-8 -*-
"""V5-024 修后口径重扫（border_value=1，2026-09-29）

外部 AI 指出的逻辑闭环缺口：th=50 统计在未修口径下算出，全量生成用修后
口径 → 统计与数据分布不一致。修复顺序改为：先修 V5-024 → 免费重扫 →
统计与生成同口径。

σ 场缓存（_exp_sigma_fields/seed_*.npy，100 seeds）不变——binary_closing
只影响 mask 提取，不影响 σ 场。本脚本用修后 _extract_crack_mask 离线重扫
th∈{40..60}，并与修前数字并排对比；判据写死：
  **在正样本率 ≥95% 前提下，线状数最大；并列取 th 最小**
"""
import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import functools

    print = functools.partial(print, flush=True)
    import numpy as np
    from scipy import ndimage

    from data.patch_simulator_v5 import PatchSimulator

    PATCH = 128
    BASE_SEED = 4200
    N_TOTAL = 100
    THS = [40.0, 45.0, 50.0, 55.0, 60.0]
    FIELD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "_exp_sigma_fields")

    FAST_PROFILE = {
        "ramp_up_c_per_min": 1260.0,
        "soak_temp_c": 1280.0,
        "soak_duration_min": 1.0,
        "cool_down_c_per_min": 1260.0,
    }

    # 修前数字（_exp_field_cache_sweep.py 输出，border_value=0 口径）
    BEFORE = {40: (100, 79), 45: (99, 86), 50: (98, 87),
              55: (93, 84), 60: (92, 88)}

    def gt_stats(mask):
        n_px = int((mask > 0).sum())
        if n_px == 0:
            return {"n_px": 0, "area_frac": 0.0, "n_comp": 0, "elong": 0.0,
                    "line_like": False}
        area_frac = n_px / mask.size
        labeled, n_comp = ndimage.label(mask > 0)
        sizes = ndimage.sum(mask > 0, labeled, range(1, n_comp + 1))
        largest = mask * (labeled == (int(np.argmax(sizes)) + 1))
        ys, xs = np.nonzero(largest)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        elong = max(h, w) / max(1, min(h, w))
        return {"n_px": n_px, "area_frac": float(area_frac),
                "n_comp": int(n_comp), "elong": float(elong),
                "line_like": bool(area_frac < 0.20 and elong > 3.0)}

    dummy = PatchSimulator(patch_size=PATCH, thermal_profile=FAST_PROFILE,
                           crack_stress_threshold_MPa=10.0, seed=0)

    print("===== V5-024 修后口径重扫（border_value=1，n=100，离线免费）=====")
    print(f"{'th':>5}{'正样本':>8}{'线状数':>8}{'线状率':>8}"
          f"{'修前(正,线)':>13}{'面积max':>9}")
    table = {}
    for th in THS:
        dummy.crack_threshold_Pa = th * 1e6
        n_pos = n_line = 0
        areas = []
        for seed in range(BASE_SEED, BASE_SEED + N_TOTAL):
            fld = np.load(os.path.join(FIELD_DIR, f"seed_{seed}.npy"))
            st = gt_stats(dummy._extract_crack_mask(fld))
            n_pos += st["n_px"] > 0
            n_line += st["line_like"]
            if st["n_px"] > 0:
                areas.append(st["area_frac"])
        b = BEFORE[int(th)]
        table[int(th)] = (n_pos, n_line)
        print(f"{int(th):>5}{n_pos:>8}{n_line:>8}"
              f"{n_line/max(1,n_pos)*100:>7.0f}%{str(b):>13}"
              f"{max(areas)*100 if areas else 0:>8.1f}%")

    # 判据：正样本率 ≥95% 前提下线状数最大，并列取 th 最小
    cands = [(th, n_pos, n_line) for th, (n_pos, n_line) in table.items()
             if n_pos >= 95]
    if cands:
        best = max(cands, key=lambda t: (t[2], -t[0]))
        print(f"\n判据（写死）：正样本率 ≥95% 前提下线状数最大，并列取 th 最小")
        print(f"→ 产线 th = {best[0]}  正样本 {best[1]}/100  线状 {best[2]}/100"
              f"（外推 1000 样本 ≈ {best[2]*10}）")
    else:
        print("\n无档满足正样本率 ≥95%，需扩扫 th 范围")
