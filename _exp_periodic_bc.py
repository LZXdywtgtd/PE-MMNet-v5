# -*- coding: utf-8 -*-
"""周期边界对照实验：观察窗假设的"解决问题"版 vs "接受问题"版（2026-09-29）

外部 AI 质疑：margin 排除是"接受边界伪影"，周期边界才是"解决边界伪影"
（patch 是大坯体观察窗 → 热流应跨边界连续）。

实现：温度步进改为全周期拉普拉斯（np.roll 环绕差分），跳过 Robin 边界；
环境加热通过体积源项施加（整场朝 T_target 松弛，替代边界流入）——
保持"窑内环境加热"的物理，同时消除自由边。

源项形式：dT_source = dt/tau_env * (T_target - T)，tau_env 标定使整体
升温速率与原边界模型同量级（原模型 1280℃ 约 60s 内跟上 fast 曲线）。

对照组：
  A. baseline（Robin 边界）——已知"边界框"
  B. 周期边界 + 体积源项（无自由边）
  C. 周期边界 + 体积源项 + 弱化通道（看 margin 是否还需要）
每组报：应力剖面（中心 vs 全场）、th 扫描 GT 形态（纪律：报面积占比）。
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

    def run(thermal_cfg, seed=42, periodic=False, tau_env=25.0):
        sim = PatchSimulator(
            patch_size=PATCH, thermal_profile=thermal_cfg,
            crack_stress_threshold_MPa=10.0, seed=seed,
        )
        H = W = PATCH
        sim.T_field = np.full((H, W), 20.0, dtype=np.float64)
        if getattr(sim, "_init_temp_noise_std_c", 0) > 0:
            sim.T_field += sim._init_rng.normal(0.0, sim._init_temp_noise_std_c, (H, W))
        sim.sigma_field = np.zeros((H, W))
        sim._sigma_max_field = np.zeros((H, W))
        times_s, temps = generate_thermal_profile(**thermal_cfg, dt_s=sim.dt)
        sim._T_prev_for_stress = sim.T_field.copy()
        r = sim.alpha_thermal * sim.dt / sim.dx ** 2  # 稳定要求 ≤ 0.25（CFL）
        for step in range(len(times_s)):
            if periodic:
                # 全周期拉普拉斯（np.roll 环绕）
                lap = (
                    np.roll(sim.T_field, 1, 0) + np.roll(sim.T_field, -1, 0)
                    + np.roll(sim.T_field, 1, 1) + np.roll(sim.T_field, -1, 1)
                    - 4 * sim.T_field
                )
                T_src = sim.T_field + r * lap
                # 体积源项：整场朝 T_target 松弛（替代边界流入）
                T_src += (sim.dt / tau_env) * (temps[step] - sim.T_field)
                sim.T_field = np.clip(T_src, 0, None)
            else:
                sim._step_heat(sim.T_field, temps[step])
            if step > 0:
                dT = sim.T_field - sim._T_prev_for_stress
                sim.sigma_field = sim.stress_solver.update_step(
                    sim.sigma_field, dT, sim.dt)
            sim._T_prev_for_stress = sim.T_field.copy()
            np.maximum(sim._sigma_max_field, sim.sigma_field,
                       out=sim._sigma_max_field)
        return sim

    ths = [20.0, 30.0, 40.0, 50.0]

    # A. baseline
    H = W = PATCH
    simA = run(FAST_PROFILE)
    sA = simA._sigma_max_field / 1e6
    m = 12
    print(f"== A. baseline(Robin) == 峰值={sA.max():.1f} "
          f"中心12px内σ p50={np.percentile(sA[m:-m, m:-m], 50):.1f} "
          f"p90={np.percentile(sA[m:-m, m:-m], 90):.1f}")

    # B. 周期边界 + 体积源项（无弱化、无 margin）
    simB = run(FAST_PROFILE, periodic=True)
    sB = simB._sigma_max_field / 1e6
    print(f"== B. 周期边界+体积源项 == 峰值={sB.max():.1f} "
          f"全场σ p50={np.percentile(sB, 50):.1f} "
          f"p90={np.percentile(sB, 90):.1f} p99={np.percentile(sB, 99):.1f}")
    print(f"   温度终值: 中心={simB.T_field[H//2, W//2]:.0f}℃ "
          f"角落={simB.T_field[0, 0]:.0f}℃（周期下应接近）")
    for th in ths:
        simB.crack_threshold_Pa = th * 1e6
        st = gt_stats(simB._extract_crack_mask(simB._sigma_max_field))
        tag = ("LINE✓" if st["line_like"] else
               ("FULL✗" if st["area_frac"] > 0.5 else
                ("blob" if st["area_frac"] > 0 else "empty")))
        print(f"   th={th:5.1f} 面积={st['area_frac']*100:6.2f}% "
              f"域={st['n_comp']:3d} elong={st['elong']:5.2f} [{tag}]")

    # C. 周期边界 + 弱化通道（不需要 margin 的验证）
    rng = np.random.default_rng(7)
    H = W = PATCH
    row = int(H * 0.45)
    d2 = (np.arange(H)[:, None] - row) ** 2 + np.zeros((1, W))
    s_field = 1.0 - 0.93 * np.exp(-d2 / (2 * 2.0 ** 2))
    print(f"\n== C. 周期边界 + 弱化通道(row@0.45, w=2, δ=0.93) ==")
    for th in (10.0, 15.0, 20.0, 25.0):
        eff = (simB._sigma_max_field > th * 1e6 * s_field).astype(np.uint8) * 255
        st = gt_stats(eff)
        tag = ("LINE✓" if st["line_like"] else
               ("FULL✗" if st["area_frac"] > 0.5 else
                ("blob" if st["area_frac"] > 0 else "empty")))
        print(f"   th={th:5.1f} 面积={st['area_frac']*100:6.2f}% "
              f"域={st['n_comp']:3d} elong={st['elong']:5.2f} [{tag}]")

    # 应力场空间结构可视化（文本热图，32×32 下采样）
    def txt_heat(a, label):
        small = a[::4, ::4]
        lo, hi = np.percentile(small, 2), np.percentile(small, 98)
        chars = " .:-=+*#%@"
        print(f"\n[{label}] 2%~98% 分位拉伸:")
        for r in small[::4]:
            print("".join(chars[min(9, int((v - lo) / (hi - lo + 1e-9) * 9))] for v in r))

    txt_heat(sA, "A baseline 应力")
    txt_heat(sB, "B 周期边界 应力")
