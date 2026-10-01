# -*- coding: utf-8 -*-
"""路线 C v3：强度弱化通道 + 边缘 margin 排除 → 线状 GT？（2026-09-29）

v2 失败根因：应力"边界高中心低"（中心行 σ≈1.1 MPa，边界峰值 45.8），
任何 th 下边界框都是最大连通域（512px 环 vs 256px 线），线状永远选不中。

v3 物理修正：patch 是大坯体的观察窗，四边 Robin 散热是人工边界条件的
伪影——真实窑内 patch 四周连续、无自由边。故 mask 提取在内部裁剪区
（margin=12px）内做，边界框被物理排除。

组合：窄深通道（w=2, delta∈{0.90,0.96}）× th∈[8..30] × margin∈{8,12,16}。
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

    def gt_stats_interior(mask: np.ndarray, margin: int) -> dict:
        """在内部裁剪区内统计（边界框被物理排除）"""
        inner = mask[margin:-margin, margin:-margin]
        n_px = int((inner > 0).sum())
        total = inner.size
        if n_px == 0:
            return {"area_frac": 0.0, "n_comp": 0, "elong": 0.0, "line_like": False}
        area_frac = n_px / total
        labeled, n_comp = ndimage.label(inner > 0)
        sizes = ndimage.sum(inner > 0, labeled, range(1, n_comp + 1))
        largest = inner * (labeled == (int(np.argmax(sizes)) + 1))
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

    # 内部应力水平（margin 后）决定 th 的可行带
    m = 12
    inner_sigma = sim._sigma_max_field[m:-m, m:-m] / 1e6
    print(f"内部σ（margin={m}）: max={inner_sigma.max():.1f} "
          f"p90={np.percentile(inner_sigma, 90):.1f} "
          f"p50={np.percentile(inner_sigma, 50):.1f} MPa\n")

    best = []
    for margin in (8, 12, 16):
        for width_px in (2.0,):
            for delta in (0.90, 0.96):
                rng = np.random.default_rng(7)
                # 三条随机方向通道（更接近真实多裂纹）：水平/对角/竖直各取弱化最深者
                row = int(H * 0.45)
                d_row2 = (np.arange(H)[:, None] - row) ** 2 + np.zeros((1, W))
                col = int(W * 0.6)
                d_col2 = np.zeros((H, 1)) + (np.arange(W)[None, :] - col) ** 2
                s_row = 1.0 - delta * np.exp(-d_row2 / (2 * width_px ** 2))
                s_col = 1.0 - delta * np.exp(-d_col2 / (2 * width_px ** 2))
                s = np.minimum(s_row, s_col)  # 两条通道都弱化
                for th in (8.0, 12.0, 16.0, 20.0, 24.0):
                    eff = sim._sigma_max_field > th * 1e6 * s
                    st = gt_stats_interior(eff.astype(np.uint8) * 255, margin)
                    tag = ("LINE✓" if st["line_like"] else
                           ("FULL✗" if st["area_frac"] > 0.5 else
                            ("blob" if st["area_frac"] > 0.01 else "empty")))
                    show = st["line_like"] or tag in ("blob",)
                    if show:
                        print(f"  m={margin:2d} d={delta} th={th:4.1f} "
                              f"面积={st['area_frac']*100:5.2f}% "
                              f"域={st['n_comp']:3d} elong={st['elong']:5.2f} [{tag}]")
                    if st["line_like"]:
                        best.append((margin, delta, th, st["area_frac"], st["elong"]))

    print("\n===== 结论 =====")
    if best:
        print(f"路线 C v3 可行：{len(best)} 个组合出线状 GT，示例：")
        for mg, d, th, a, e in best[:10]:
            print(f"  margin={mg} delta={d} th={th:.1f} "
                  f"面积={a*100:.2f}% elong={e:.1f}")
    else:
        print("v3 仍无线状：内部应力太平（中心 1 MPa vs 内区峰值 "
              f"{inner_sigma.max():.1f}），通道开裂阈值带不存在")
