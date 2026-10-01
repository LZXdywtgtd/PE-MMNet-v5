# -*- coding: utf-8 -*-
"""路线 D 验证：边界空间调制（局部强散热区）→ 线状 GT？（2026-09-29）

外部 AI 提议：周期边界实验证明空间结构唯一来自边界 → 在边界上做文章。
h/ε 从标量改为空间分布：某边界一小段 h 增强（出风口/窑车接缝/观察孔的
物理对应），产生局部持续低温区 → 应力场沿线有结构 → 阈值化出线状？

实现：重写边界更新循环（与 _apply_robin_boundary 同逻辑），h 逐单元取
h_map（边界段高斯凸包）。对照组：
  D1 顶部边缘单窄凸包（σ=4px, boost 3×）
  D2 对边双凸包（top x=0.3, bottom x=0.7）
  D3 整条顶边 h×4（线性增强边）
  D4 单凸包极端 boost ×10（最大化对比度）
判据同前：面积 < 20% 且 elongation > 3。
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

    def boundary_hmaps(sim, spikes):
        """每条边的 h 倍率数组。spikes = [(edge, frac, sigma_px, boost), ...]

        edge ∈ {"top","bottom","left","right"}
        """
        H = W = sim.patch_size
        maps = {}
        x = np.arange(W) / W
        y = np.arange(H) / H
        base = np.ones(W)
        base_y = np.ones(H)
        for edge, frac, sig, boost in spikes:
            if edge in ("top", "bottom"):
                base = base + boost * np.exp(-((x - frac) ** 2) / (2 * (sig / W) ** 2))
            else:
                base_y = base_y + boost * np.exp(-((y - frac) ** 2) / (2 * (sig / H) ** 2))
        maps["top"] = base
        maps["bottom"] = base
        maps["left"] = base_y
        maps["right"] = base_y
        return maps

    def run(thermal_cfg, seed=42, spikes=()):
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
        hmaps = boundary_hmaps(sim, spikes)
        times_s, temps = generate_thermal_profile(**thermal_cfg, dt_s=sim.dt)
        sim._T_prev_for_stress = sim.T_field.copy()

        dt = sim.dt
        rho_cp = sim.rho * sim.cp
        L = sim.dx / 2.0
        edges = [("top", (0, slice(None)), hmaps["top"]),
                 ("bottom", (-1, slice(None)), hmaps["bottom"]),
                 ("left", (slice(None), 0), hmaps["left"]),
                 ("right", (slice(None), -1), hmaps["right"])]
        for step in range(len(times_s)):
            T_target = temps[step]
            # 内部拉普拉斯（与 _step_heat 一致）
            T_new = sim.T_field.copy()
            a = sim.alpha_thermal * dt / sim.dx ** 2
            T_new[1:-1, 1:-1] = sim.T_field[1:-1, 1:-1] + a * (
                sim.T_field[2:, 1:-1] - 2 * sim.T_field[1:-1, 1:-1] + sim.T_field[:-2, 1:-1]
                + sim.T_field[1:-1, 2:] - 2 * sim.T_field[1:-1, 1:-1] + sim.T_field[1:-1, :-2])
            # 边界 Robin（h 空间调制）
            T_env_c = T_target
            T_env_K = sim.radiation.celsius_to_kelvin(T_env_c)
            for _, sl, hm in edges:
                T_edge = sim.T_field[sl]
                T_edge_K = sim.radiation.celsius_to_kelvin(T_edge)
                q = sim.h_conv * hm * (T_env_c - T_edge) \
                    + sim.radiation.heat_flux(T_edge_K, T_env_K)
                T_new[sl] = T_edge + dt * q / (rho_cp * L)
            sim.T_field = np.clip(T_new, 0, None)
            if step > 0:
                dT = sim.T_field - sim._T_prev_for_stress
                sim.sigma_field = sim.stress_solver.update_step(
                    sim.sigma_field, dT, sim.dt)
            sim._T_prev_for_stress = sim.T_field.copy()
            np.maximum(sim._sigma_max_field, sim.sigma_field,
                       out=sim._sigma_max_field)
        return sim

    def txt_heat(a, label):
        small = a[::4, ::4]
        lo, hi = np.percentile(small, 2), np.percentile(small, 98)
        chars = " .:-=+*#%@"
        print(f"\n[{label}] 2%~98% 拉伸:")
        for r in small:
            print("".join(chars[min(9, max(0, int((v - lo) / (hi - lo + 1e-9) * 9)))] for v in r))

    configs = {
        "D1 顶边单凸包(σ=4,×3)": [("top", 0.5, 4.0, 3.0)],
        "D2 对边双凸包": [("top", 0.3, 4.0, 3.0), ("bottom", 0.7, 4.0, 3.0)],
        "D3 整条顶边×4": [("top", 0.5, 64.0, 3.0)],
        "D4 单凸包极端×10": [("top", 0.5, 4.0, 10.0)],
    }
    ths = [20.0, 30.0, 40.0, 50.0]
    fields = {}
    for name, spikes in configs.items():
        sim = run(FAST_PROFILE, spikes=spikes)
        fields[name] = sim._sigma_max_field / 1e6
        s = fields[name]
        print(f"\n===== {name} ===== 峰值={s.max():.1f} MPa  "
              f"p50={np.percentile(s, 50):.1f} p95={np.percentile(s, 95):.1f}")
        any_line = False
        for th in ths:
            sim.crack_threshold_Pa = th * 1e6
            st = gt_stats(sim._extract_crack_mask(sim._sigma_max_field))
            tag = ("LINE✓" if st["line_like"] else
                   ("FULL✗" if st["area_frac"] > 0.5 else
                    ("blob" if st["area_frac"] > 0 else "empty")))
            any_line |= st["line_like"]
            print(f"  th={th:5.1f} 面积={st['area_frac']*100:6.2f}% "
                  f"域={st['n_comp']:3d} elong={st['elong']:5.2f} [{tag}]")
        txt_heat(s, name + " 应力")
        print(f"  → {name}: {'出线状' if any_line else '无线状'}")
