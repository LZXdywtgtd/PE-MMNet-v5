# -*- coding: utf-8 -*-
"""默认曲线扩大 spot check（2026-09-29）

外部 AI 质疑 3：全量生成前的 spot check 不能只查"调制存活"——
默认曲线（11.17h）下每个环节都可能失效。三项全查：
  ① 边界调制是否存活（10 个随机 D' 配置，段长覆盖 1/3~2/3 全分布）
  ② 段长下限（3 个短段对照 seg=1/6）
  ③ th 是否重选（σ 峰值分布 + 宽范围 th 扫描 20~200）

默认曲线 128px 仿真一步时长约 fast 的 60 倍（11.17h vs 11min），
10+3 样本预计 30-60 分钟。σ 场落盘断点续跑。
"""
import os
import sys
import json

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import functools

    print = functools.partial(print, flush=True)
    import numpy as np
    from scipy import ndimage

    from data.patch_simulator_v5 import PatchSimulator
    from data.thermal_profile import generate_thermal_profile

    PATCH = 128
    BASE_SEED = 4300
    FIELD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "_exp_default_curve_fields")
    os.makedirs(FIELD_DIR, exist_ok=True)

    # 默认烧成曲线（与 assets 副本 th=11 缓存同口径）
    DEFAULT_PROFILE = {
        "ramp_up_c_per_min": 180.0,
        "soak_temp_c": 1280.0,
        "soak_duration_min": 60.0,
        "cool_down_c_per_min": 150.0,
    }

    def seg_hmap(axis_vals, c, seg_len, boost, ramp=0.04):
        u = np.clip(axis_vals, 0.0, 1.0)
        aa, bb = c - seg_len / 2, c + seg_len / 2
        s1 = np.clip((u - aa) / ramp, 0, 1); s1 = s1 * s1 * (3 - 2 * s1)
        s2 = np.clip((u - bb) / ramp, 0, 1); s2 = s2 * s2 * (3 - 2 * s2)
        return 1.0 + (boost - 1.0) * s1 * (1 - s2)

    def spikes_for(seed, short=False):
        rng = np.random.default_rng(seed)
        edge = str(rng.choice(["top", "bottom", "left", "right"]))
        c = float(rng.uniform(0.3, 0.7))
        if short:
            seg = 1 / 6
        else:
            seg = float(rng.uniform(1 / 3, 2 / 3))
        boost = float(rng.choice([2.0, 3.0, 4.0]))
        return [(edge, c, seg, boost)]

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

    def run_sample(seed, spikes):
        sim = PatchSimulator(
            patch_size=PATCH, thermal_profile=DEFAULT_PROFILE,
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
        times_s, temps = generate_thermal_profile(**DEFAULT_PROFILE, dt_s=sim.dt)
        sim._T_prev_for_stress = sim.T_field.copy()
        dt = sim.dt
        rho_cp = sim.rho * sim.cp
        L = sim.dx / 2.0
        edges = [("top", (0, slice(None))), ("bottom", (-1, slice(None))),
                 ("left", (slice(None), 0)), ("right", (slice(None), -1))]
        n_steps = len(times_s)
        for step in range(n_steps):
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
            if step % 200000 == 0 and step > 0:
                print(f"    seed={seed} step {step}/{n_steps} "
                      f"峰值={sim._sigma_max_field.max()/1e6:.0f} MPa", flush=True)
        return sim

    # 阶段 1：10 随机 D' + 3 短段，σ 场落盘
    plan = [(BASE_SEED + i, spikes_for(BASE_SEED + i)) for i in range(10)]
    plan += [(BASE_SEED + 100 + i, spikes_for(BASE_SEED + 100 + i, short=True))
             for i in range(3)]
    for seed, spikes in plan:
        out = os.path.join(FIELD_DIR, f"seed_{seed}.npy")
        if os.path.exists(out):
            print(f"seed={seed} 已有缓存，跳过")
            continue
        print(f"仿真 seed={seed} {spikes} ...", flush=True)
        sim = run_sample(seed, spikes)
        np.save(out, sim._sigma_max_field.astype(np.float32))
        print(f"  → 峰值={sim._sigma_max_field.max()/1e6:.1f} MPa")

    # 阶段 2：宽范围 th 扫描（含 ③）
    dummy = PatchSimulator(patch_size=PATCH, thermal_profile=DEFAULT_PROFILE,
                           crack_stress_threshold_MPa=10.0, seed=0)
    THS = [20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0, 200.0]
    print("\n===== 默认曲线 D' spot check（n=10 随机段长 + 3 短段对照）=====")
    print(f"{'th':>6} | {'随机段(n=10): 正/线':>20} | {'短段(n=3): 正/线':>18}")
    for th in THS:
        dummy.crack_threshold_Pa = th * 1e6
        r_pos = r_line = 0
        s_pos = s_line = 0
        for i in range(10):
            fld = np.load(os.path.join(FIELD_DIR, f"seed_{BASE_SEED + i}.npy"))
            st = gt_stats(dummy._extract_crack_mask(fld))
            r_pos += st["n_px"] > 0
            r_line += st["line_like"]
        for i in range(3):
            fld = np.load(os.path.join(FIELD_DIR, f"seed_{BASE_SEED + 100 + i}.npy"))
            st = gt_stats(dummy._extract_crack_mask(fld))
            s_pos += st["n_px"] > 0
            s_line += st["line_like"]
        print(f"{th:>6.0f} | {r_pos}正 {r_line}线{'':<10} | {s_pos}正 {s_line}线")

    # 峰值分布
    peaks = []
    for i in range(10):
        fld = np.load(os.path.join(FIELD_DIR, f"seed_{BASE_SEED + i}.npy"))
        peaks.append(float(fld.max()) / 1e6)
    print(f"\n随机段 σ 峰值分布：p50={np.percentile(peaks,50):.0f} "
          f"min={min(peaks):.0f} max={max(peaks):.0f} MPa"
          f"（fast 曲线同配置 p50≈234，默认曲线若远高于此则 th 需重选——"
          f"上表选线状绝对数最大档）")
