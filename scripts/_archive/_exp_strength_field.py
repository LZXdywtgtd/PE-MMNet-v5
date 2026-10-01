# -*- coding: utf-8 -*-
"""路线 C 验证：空间强度弱化场 → 线状 GT？（2026-09-29）

背景：路线 A（初温扰动）已证伪（_exp_hotspot_linearity.py）——逐像素
标量应力模型 + 热扩散抹平 → 阈值化平滑场只能出团块。

路线 C（外部 AI 提议）：GT 判据从"σ > th 全场统一阈值"改为
    crack = (σ_max > th × s(x,y))
其中 s(x,y) ∈ [s_min, 1] 是空间相关的材料强度场（条纹 + 平滑噪声）。
物理含义：陶瓷局部强度不均匀，应力超过**局部**强度才开裂。
不依赖空间耦合应力模型，改动只在 mask 提取层。

实验设计：
- 复用路线 A 实验的 baseline 应力场（fast 128px，峰值 45.8 MPa，
  空间分布"边界高中心低"——这正是需要的空间变化）
- s(x,y) = 1 - delta × pattern，pattern ∈ [0,1]：
  pattern = 0.5·|sin(2π·k·y/H + φ)|（沿 y 方向条纹，k 条）+ 0.5·平滑高斯噪声
  条纹方向沿 x（水平线状），k ∈ {4, 8}，delta ∈ {0.5, 0.7}
- th 扫描 [20, 30, 40, 50]：有效局部阈值 = th·s ∈ [th·s_min, th]
- 判据（与 V5-021 验收标准一致）：面积占比 < 20% 且 elongation > 3
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
    FAST_PROFILE = {
        "ramp_up_c_per_min": 1260.0,
        "soak_temp_c": 1280.0,
        "soak_duration_min": 1.0,
        "cool_down_c_per_min": 1260.0,
    }
    THS = [20.0, 30.0, 40.0, 50.0]

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
            "area_frac": area_frac,
            "n_comp": n_comp,
            "elong": float(elong),
            "line_like": area_frac < 0.20 and elong > 3.0,
        }

    def strength_field(H, W, rng, n_stripes: int, delta: float) -> np.ndarray:
        """条纹 + 平滑噪声 ∈ [1-delta, 1]，条纹沿 x（水平线）"""
        y = np.arange(H)[:, None] * np.ones((1, W))
        phase = rng.uniform(0, 2 * np.pi)
        stripe = np.abs(np.sin(2 * np.pi * n_stripes * y / H + phase))
        noise = rng.normal(0, 1, (H, W))
        noise = ndimage.gaussian_filter(noise, sigma=6.0)
        noise = (noise - noise.min()) / (np.ptp(noise) + 1e-9)
        pattern = 0.5 * stripe + 0.5 * noise
        return 1.0 - delta * pattern

    # 1. baseline 应力场（与路线 A 实验同配置：fast/128/seed42）
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
    from data.thermal_profile import generate_thermal_profile
    times_s, temps = generate_thermal_profile(**FAST_PROFILE, dt_s=sim.dt)
    sim._T_prev_for_stress = sim.T_field.copy()
    for step in range(len(times_s)):
        sim._step_heat(sim.T_field, temps[step])
        if step > 0:
            dT = sim.T_field - sim._T_prev_for_stress
            sim.sigma_field = sim.stress_solver.update_step(sim.sigma_field, dT, sim.dt)
        sim._T_prev_for_stress = sim.T_field.copy()
        np.maximum(sim._sigma_max_field, sim.sigma_field, out=sim._sigma_max_field)
    print(f"baseline 应力场峰值 = {sim._sigma_max_field.max()/1e6:.1f} MPa  "
          f"(min {sim._sigma_max_field.min()/1e6:.1f})\n")

    # 2. 路线 C 扫描
    best = []
    for n_stripes in (4, 8):
        for delta in (0.5, 0.7):
            rng = np.random.default_rng(7)
            s = strength_field(H, W, rng, n_stripes, delta)
            print(f"== 条纹 k={n_stripes}  弱化深度 delta={delta}  "
                  f"s ∈ [{s.min():.2f}, {s.max():.2f}] ==")
            for th in THS:
                eff = sim._sigma_max_field > th * 1e6 * s
                mask = sim._extract_crack_mask((eff * 1e6).astype(np.float64) * 1e6)
                # 直接用二值结果统计（绕过 _extract_crack_mask 内部阈值）
                st = gt_stats(eff.astype(np.uint8) * 255)
                tag = "LINE✓" if st["line_like"] else (
                    "FULL✗" if st["area_frac"] > 0.5 else "other")
                print(f"  th={th:5.1f}  有效阈值带=[{th*s.min():5.1f},{th:5.1f}]MPa  "
                      f"面积={st['area_frac']*100:6.2f}%  连通域={st['n_comp']:3d}  "
                      f"elong={st['elong']:5.2f}  [{tag}]")
                if st["line_like"]:
                    best.append((n_stripes, delta, th, st["area_frac"], st["elong"]))
            print()

    print("===== 结论 =====")
    if best:
        print(f"路线 C 可行：{len(best)} 个 (k, delta, th) 组合出线状 GT")
        for k, d, th, a, e in best[:6]:
            print(f"  k={k} delta={d} th={th}  面积={a*100:.1f}%  elong={e:.1f}")
    else:
        print("路线 C 在本扫描范围内未出线状 GT——需要调 pattern 结构或收窄扫描")
