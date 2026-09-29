# -*- coding: utf-8 -*-
"""corner 调制对角线状验证（2026-09-29）

H/V 限制的解法②（外部 AI 中成本方案）：在角部同时调制两条边的近角段，
看能否产生对角方向的线状 GT。物理直觉：角部双向强化散热 → 温度场沿
对角线对称 → 应力等值线垂直于对角 → 线状沿对角？

配置（fast 曲线 128px，修后 V5-024 口径）：
  E1 角部双边段：top 近左角 1/3 段 ×3 + left 近顶角 1/3 段 ×3
  E2 角部双边强：同位置 ×4
  E3 三边近角段：top+left+（right 近顶角）各 1/3 ×3（L 形包围）
判据不变：面积 <20% 且 elong >3；额外报对角度（mask 主轴与 x 轴夹角）。
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

    def seg_hmap(axis_vals, c, seg_len, boost, ramp=0.04):
        u = np.clip(axis_vals, 0.0, 1.0)
        aa, bb = c - seg_len / 2, c + seg_len / 2
        s1 = np.clip((u - aa) / ramp, 0, 1); s1 = s1 * s1 * (3 - 2 * s1)
        s2 = np.clip((u - bb) / ramp, 0, 1); s2 = s2 * s2 * (3 - 2 * s2)
        return 1.0 + (boost - 1.0) * s1 * (1 - s2)

    def gt_stats(mask):
        n_px = int((mask > 0).sum())
        if n_px == 0:
            return {"area_frac": 0.0, "elong": 0.0, "angle_deg": None,
                    "line_like": False}
        area_frac = n_px / mask.size
        labeled, _ = ndimage.label(mask > 0)
        sizes = ndimage.sum(mask > 0, labeled, range(1, int(labeled.max()) + 1))
        largest = mask * (labeled == (int(np.argmax(sizes)) + 1))
        ys, xs = np.nonzero(largest)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        elong = max(h, w) / max(1, min(h, w))
        # 主轴角度：二阶中心矩
        ym, xm = ys - ys.mean(), xs - xs.mean()
        cov = np.cov(np.stack([xm, ym]))
        evals, evecs = np.linalg.eigh(cov)
        v = evecs[:, -1]
        ang = float(np.degrees(np.arctan2(abs(v[0]), abs(v[1]))))
        if ang > 90:
            ang = 180 - ang
        return {"area_frac": float(area_frac), "elong": float(elong),
                "angle_deg": ang,
                "line_like": bool(area_frac < 0.20 and elong > 3.0)}

    def run(spikes, seed=42):
        sim = PatchSimulator(
            patch_size=PATCH, thermal_profile=FAST_PROFILE,
            crack_stress_threshold_MPa=10.0, seed=seed,
        )
        H = W = PATCH
        sim.T_field = np.full((H, W), 20.0, dtype=np.float64)
        if getattr(sim, "_init_temp_noise_std_c", 0) > 0:
            sim.T_field += sim._init_rng.normal(0.0, sim._init_temp_noise_std_c, (H, W))
        sim.sigma_field = np.zeros((H, W))
        sim._sigma_max_field = np.zeros((H, W))
        tcoord = np.arange(W) / W
        ycoord = np.arange(H) / H
        hmaps = {"top": np.ones(W), "bottom": np.ones(W),
                 "left": np.ones(H), "right": np.ones(H)}
        for edge, c, seg, boost in spikes:
            if edge in ("top", "bottom"):
                hmaps[edge] = seg_hmap(tcoord, c, seg, boost)
            else:
                hmaps[edge] = seg_hmap(ycoord, c, seg, boost)
        times_s, temps = generate_thermal_profile(**FAST_PROFILE, dt_s=sim.dt)
        sim._T_prev_for_stress = sim.T_field.copy()
        dt = sim.dt
        rho_cp = sim.rho * sim.cp
        L = sim.dx / 2.0
        edges = [("top", (0, slice(None))), ("bottom", (-1, slice(None))),
                 ("left", (slice(None), 0)), ("right", (slice(None), -1))]
        for step in range(len(times_s)):
            T_target = temps[step]
            T_new = sim.T_field.copy()
            a = sim.alpha_thermal * dt / sim.dx ** 2
            T_new[1:-1, 1:-1] = sim.T_field[1:-1, 1:-1] + a * (
                sim.T_field[2:, 1:-1] - 2 * sim.T_field[1:-1, 1:-1] + sim.T_field[:-2, 1:-1]
                + sim.T_field[1:-1, 2:] - 2 * sim.T_field[1:-1, 1:-1] + sim.T_field[1:-1, :-2])
            T_env_c = T_target
            T_env_K = sim.radiation.celsius_to_kelvin(T_env_c)
            for name, sl in edges:
                T_edge = sim.T_field[sl]
                T_edge_K = sim.radiation.celsius_to_kelvin(T_edge)
                q = sim.h_conv * hmaps[name] * (T_env_c - T_edge) \
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

    configs = {
        "E1 角部双边段×3": [("top", 1/6, 1/3, 3.0), ("left", 1/6, 1/3, 3.0)],
        "E2 角部双边段×4": [("top", 1/6, 1/3, 4.0), ("left", 1/6, 1/3, 4.0)],
        "E3 三边近角段×3": [("top", 1/6, 1/3, 3.0), ("left", 1/6, 1/3, 3.0),
                            ("right", 1/6, 1/3, 3.0)],
    }
    ths = [40.0, 50.0, 60.0, 80.0, 100.0]
    for name, spikes in configs.items():
        sim = run(spikes)
        s = sim._sigma_max_field / 1e6
        print(f"\n===== {name} ===== 峰值={s.max():.1f}")
        any_line = False
        for th in ths:
            sim.crack_threshold_Pa = th * 1e6
            st = gt_stats(sim._extract_crack_mask(sim._sigma_max_field))
            tag = ("LINE✓" if st["line_like"] else
                   ("FULL✗" if st["area_frac"] > 0.5 else
                    ("blob" if st["area_frac"] > 0 else "empty")))
            any_line |= st["line_like"]
            ang = f"{st['angle_deg']:.0f}°" if st["angle_deg"] is not None else "-"
            print(f"  th={th:6.1f} 面积={st['area_frac']*100:6.2f}% "
                  f"elong={st['elong']:5.2f} 主轴角={ang:>5} [{tag}]")
        print(f"  → {name}: {'出线状' if any_line else '无线状'}")
