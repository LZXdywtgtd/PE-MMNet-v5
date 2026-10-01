# -*- coding: utf-8 -*-
"""路线 C v2：窄深高斯强度通道 → 线状 GT？（2026-09-29）

v1 失败教训：s ∈ [0.3,0.5] 的宽条纹打不碎"边界高"应力结构——最大连通域
贴边呈环（elong=1.0）。v2 改窄深通道：s = 1 - delta·exp(-(d/w)²)，
delta=0.85~0.95、w=1~3px，通道内有效阈值骤降一个量级，弱化通道主导几何。
对照组合：通道位置（中心/偏置）、th 相对峰值的三档。
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
    from data.thermal_profile import generate_thermal_profile

    PATCH = 128
    FAST_PROFILE = {
        "ramp_up_c_per_min": 1260.0,
        "soak_temp_c": 1280.0,
        "soak_duration_min": 1.0,
        "cool_down_c_per_min": 1260.0,
    }

    def gt_stats(mask: np.ndarray) -> dict:
        n_px = int((mask > 0).sum())
        if n_px == 0:
            return {"area_frac": 0.0, "n_comp": 0, "elong": 0.0, "line_like": False}
        area_frac = n_px / mask.size
        labeled, n_comp = ndimage.label(mask > 0)
        sizes = ndimage.sum(mask > 0, labeled, range(1, n_comp + 1))
        largest = mask * (labeled == (int(np.argmax(sizes)) + 1))
        ys, xs = np.nonzero(largest)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        elong = max(h, w) / max(1, min(h, w))
        return {
            "area_frac": area_frac, "n_comp": n_comp, "elong": float(elong),
            "line_like": area_frac < 0.20 and elong > 3.0,
        }

    # baseline 应力场
    sim = PatchSimulator(
        patch_size=PATCH, thermal_profile=FAST_PROFILE,
        crack_stress_threshold_MPa=10.0, seed=42,
    )
    H = W = PATCH
    sim.T_field = np.full((H, W), 20.0, dtype=np.float64)
    if getattr(sim, "_init_temp_noise_std_c", 0) > 0:
        sim.T_field += sim._init_rng.normal(0.0, sim._init_temp_noise_std_c, (H, W))
    sim.sigma_field = np.zeros((H, W))
    sim._sigma_max_field = np.zeros((H, W))
    times_s, temps = generate_thermal_profile(**FAST_PROFILE, dt_s=sim.dt)
    sim._T_prev_for_stress = sim.T_field.copy()
    for step in range(len(times_s)):
        sim._step_heat(sim.T_field, temps[step])
        if step > 0:
            dT = sim.T_field - sim._T_prev_for_stress
            sim.sigma_field = sim.stress_solver.update_step(
                sim.sigma_field, dT, sim.dt)
        sim._T_prev_for_stress = sim.T_field.copy()
        np.maximum(sim._sigma_max_field, sim.sigma_field, out=sim._sigma_max_field)

    peak = sim._sigma_max_field.max() / 1e6
    # 应力剖面：中心行 vs 边界行的峰值（决定通道内能否越阈）
    center_row = sim._sigma_max_field[H // 2, :] / 1e6
    print(f"峰值={peak:.1f} MPa  中心行σ范围=[{center_row.min():.1f}, "
          f"{center_row.max():.1f}]  中心像素σ={center_row[W//2]:.1f}\n")

    best = []
    for frac_r in (0.5, 0.35):        # 通道行位置（中心 / 偏置）
        for width_px in (1.5, 3.0):
            for delta in (0.85, 0.95):
                rng = np.random.default_rng(7)
                row = int(H * frac_r)
                d2 = (np.arange(H)[:, None] - row) ** 2 + np.zeros((1, W))
                s = 1.0 - delta * np.exp(-d2 / (2 * width_px ** 2))
                # 应力中心低：th 取"中心σ/delta 与边界峰之间"的档位
                for th in (peak * 0.25, peak * 0.35, peak * 0.5):
                    eff = sim._sigma_max_field > th * 1e6 * s
                    st = gt_stats(eff.astype(np.uint8) * 255)
                    tag = ("LINE✓" if st["line_like"] else
                           ("FULL✗" if st["area_frac"] > 0.5 else "other"))
                    if st["line_like"] or tag == "other":
                        print(f"  row@{frac_r} w={width_px} d={delta} "
                              f"th={th:4.1f} 面积={st['area_frac']*100:5.2f}% "
                              f"域={st['n_comp']:3d} elong={st['elong']:5.2f} [{tag}]")
                    if st["line_like"]:
                        best.append((frac_r, width_px, delta, th,
                                     st["area_frac"], st["elong"]))

    print("\n===== 结论 =====")
    if best:
        print(f"路线 C v2 可行：{len(best)} 个组合出线状 GT，示例：")
        for r, w, d, th, a, e in best[:8]:
            print(f"  row@{r} w={w} delta={d} th={th:.1f} "
                  f"面积={a*100:.2f}% elong={e:.1f}")
    else:
        print("v2 仍无线状——通道阈值设计不足以让线状结构在最大连通域中存活")
