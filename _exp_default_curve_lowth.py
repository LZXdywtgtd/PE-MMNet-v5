# -*- coding: utf-8 -*-
"""默认曲线低 th 补测（2026-09-29，零成本离线扫描）

v2 spot check 显示真默认曲线（5/30/3, 11.7h）下 σ 峰值 p50=12 MPa，
th 40~240 全 0 正样本。本脚本扫低 th 档（2~20 MPa）+ 背景应力分布，
回答："降 th 能否救默认曲线"——若背景应力与调制峰值同量级，
低 th 必然满面（对比度不足），该路死。
σ 场直接读 _exp_default_curve_fields_v2/，无仿真成本。
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
    FIELD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "_exp_default_curve_fields_v2")

    def gt_stats(mask):
        n_px = int((mask > 0).sum())
        if n_px == 0:
            return {"n_px": 0, "area_frac": 0.0, "elong": 0.0, "line_like": False}
        area_frac = n_px / mask.size
        labeled, n_comp = ndimage.label(mask > 0)
        sizes = ndimage.sum(mask > 0, labeled, range(1, n_comp + 1))
        largest = mask * (labeled == (int(np.argmax(sizes)) + 1))
        ys, xs = np.nonzero(largest)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        elong = max(h, w) / max(1, min(h, w))
        return {"n_px": n_px, "area_frac": float(area_frac), "elong": float(elong),
                "line_like": bool(area_frac < 0.20 and elong > 3.0)}

    dummy = PatchSimulator(patch_size=PATCH, thermal_profile={
        "ramp_up_c_per_min": 5.0, "soak_temp_c": 1280.0,
        "soak_duration_min": 30, "cool_down_c_per_min": 3.0},
        crack_stress_threshold_MPa=10.0, seed=0)

    seeds = [4500, 4501, 4502, 4503, 4504, 4600, 4601]

    print("===== 背景应力分布（7 样本，σ_max 场）=====")
    print(f"{'seed':>6} | {'p50':>5} {'p90':>5} {'p99':>5} {'max':>6} (MPa)")
    fields = {}
    for s in seeds:
        fld = np.load(os.path.join(FIELD_DIR, f"seed_{s}.npy"))
        fields[s] = fld
        f = fld / 1e6
        print(f"{s:>6} | {np.percentile(f,50):5.1f} {np.percentile(f,90):5.1f} "
              f"{np.percentile(f,99):5.1f} {f.max():6.1f}")

    THS = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 15.0, 20.0]
    print("\n===== 低 th 扫描（产线口径 _extract_crack_mask）=====")
    print(f"{'th':>5} | {'随机段(n=5) 正/线/满面':>22} | {'短段(n=2) 正/线/满面':>20}")
    for th in THS:
        dummy.crack_threshold_Pa = th * 1e6
        r = [0, 0, 0]  # pos, line, full
        s = [0, 0, 0]
        for i, sd in enumerate(seeds):
            st = gt_stats(dummy._extract_crack_mask(fields[sd]))
            bucket = r if i < 5 else s
            bucket[0] += st["n_px"] > 0
            bucket[1] += st["line_like"]
            bucket[2] += st["n_px"] > 0 and st["area_frac"] >= 0.20
        print(f"{th:>5.0f} | {r[0]}正 {r[1]}线 {r[2]}满面{'':<6} | "
              f"{s[0]}正 {s[1]}线 {s[2]}满面")
