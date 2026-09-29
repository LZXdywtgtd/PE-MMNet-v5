# -*- coding: utf-8 -*-
"""默认曲线扩大 spot check v2——修正 profile（2026-09-29）

v1 事故：_exp_default_curve_spot.py 用了自造曲线（180/60/150，总长 1.26h），
而产线 config.json 默认曲线是 5.0/30/3.0（总长 ≈11.7h）。外部 AI 质疑 3
的核心就是"默认曲线长时热扩散抹平调制"——1.26h 曲线扩散长度短 ~3 倍，
v1 结论不能外推，本脚本用真实曲线重跑。

三项不变：①调制存活（5 随机 D' 配置）②段长下限（2 个 seg=1/6 对照）
③th 重选（峰值分布 + th 40~240 宽扫描，v1 显示默认口径下 th 大幅上移）。
11.7h 曲线 128px ≈ 108 万步/样本，单样本 ~25-30 分钟，7 样本 ~3 小时。
σ 场落盘断点续跑。
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
    BASE_SEED = 4500
    FIELD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "_exp_default_curve_fields_v2")
    os.makedirs(FIELD_DIR, exist_ok=True)

    # 产线默认曲线（config.json thermal_profile 原文）
    DEFAULT_PROFILE = {
        "ramp_up_c_per_min": 5.0,
        "soak_temp_c": 1280.0,
        "soak_duration_min": 30,
        "cool_down_c_per_min": 3.0,
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
        print(f"  seed={seed} 总步数={n_steps}（dt={dt:.4f}s，"
              f"总时长={times_s[-1]/3600:.2f}h）", flush=True)
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

    plan = [(BASE_SEED + i, spikes_for(BASE_SEED + i)) for i in range(5)]
    plan += [(BASE_SEED + 100 + i, spikes_for(BASE_SEED + 100 + i, short=True))
             for i in range(2)]
    for seed, spikes in plan:
        out = os.path.join(FIELD_DIR, f"seed_{seed}.npy")
        if os.path.exists(out):
            print(f"seed={seed} 已有缓存，跳过", flush=True)
            continue
        print(f"仿真 seed={seed} {spikes} ...", flush=True)
        sim = run_sample(seed, spikes)
        np.save(out, sim._sigma_max_field.astype(np.float32))
        print(f"  → 峰值={sim._sigma_max_field.max()/1e6:.1f} MPa", flush=True)

    dummy = PatchSimulator(patch_size=PATCH, thermal_profile=DEFAULT_PROFILE,
                           crack_stress_threshold_MPa=10.0, seed=0)
    THS = [40.0, 60.0, 80.0, 120.0, 160.0, 200.0, 240.0]
    print("\n===== 真默认曲线(5/30/3, 11.7h) D' spot check（5 随机段 + 2 短段）=====")
    print(f"{'th':>6} | {'随机段(n=5): 正/线':>20} | {'短段(n=2): 正/线':>18}")
    for th in THS:
        dummy.crack_threshold_Pa = th * 1e6
        r_pos = r_line = 0
        s_pos = s_line = 0
        for i in range(5):
            fld = np.load(os.path.join(FIELD_DIR, f"seed_{BASE_SEED + i}.npy"))
            st = gt_stats(dummy._extract_crack_mask(fld))
            r_pos += st["n_px"] > 0
            r_line += st["line_like"]
        for i in range(2):
            fld = np.load(os.path.join(FIELD_DIR, f"seed_{BASE_SEED + 100 + i}.npy"))
            st = gt_stats(dummy._extract_crack_mask(fld))
            s_pos += st["n_px"] > 0
            s_line += st["line_like"]
        print(f"{th:>6.0f} | {r_pos}正 {r_line}线{'':<10} | {s_pos}正 {s_line}线")

    peaks = []
    for i in range(5):
        fld = np.load(os.path.join(FIELD_DIR, f"seed_{BASE_SEED + i}.npy"))
        peaks.append(float(fld.max()) / 1e6)
    print(f"\n随机段 σ 峰值分布（真默认曲线）：p50={np.percentile(peaks,50):.0f} "
          f"min={min(peaks):.0f} max={max(peaks):.0f} MPa"
          f"（v1 自造 1.26h 曲线 p50=170；fast 曲线 p50≈234）")
