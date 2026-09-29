# -*- coding: utf-8 -*-
"""σ 场缓存 + 产线口径 th 细扫（2026-09-29）

前序脚本两个口径问题（外部 AI 质疑 1 的延伸）：
1. _exp_th_sweep.py 用裸阈值 (s>th)，而产线 GT 用 _extract_crack_mask
   （阈值→最大连通域→闭运算 iter=2）——两口径数字不可比
   （th40: 裸 59% vs 产线 78%）
2. scipy binary_closing 默认 border_value=0 会腐蚀贴边像素：D' 线状沿边
   分布，峰值勉强过阈的细线会被闭运算整体删除（seed 4215 peak=61.7>50
   却 n_px=0 的候选解释，本脚本实证）

做法：100 seeds 各仿真一次，σ_max 场逐样本落盘 .npy（断点续跑；
以后任何 th 扫描离线免费）。然后产线口径扫 th∈{40,45,50,55,60}：
  - 线状绝对数（判据 1）
  - 4215 逐口径解剖（裸阈值像素数/位置 → 闭运算后像素数）
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
    N_TOTAL = 100
    BASE_SEED = 4200
    THS = [40.0, 45.0, 50.0, 55.0, 60.0]
    FIELD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "_exp_sigma_fields")
    os.makedirs(FIELD_DIR, exist_ok=True)

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

    def spikes_for(seed, dprime):
        if seed in dprime:
            return dprime[seed]
        rng = np.random.default_rng(seed)
        edge = str(rng.choice(["top", "bottom", "left", "right"]))
        c = float(rng.uniform(0.3, 0.7))
        seg = float(rng.uniform(1 / 3, 2 / 3))
        boost = float(rng.choice([2.0, 3.0, 4.0]))
        return [(edge, c, seg, boost)]

    def run_sample(seed, spikes):
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

    # 读 D' 50 seeds 配置
    dprime = {}
    dp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_exp_boundary_dprime_results.jsonl")
    with open(dp_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            dprime[r["seed"]] = r["spikes"]

    # 阶段 1：σ 场落盘（断点：跳过已有 .npy）
    todo = [s for s in range(BASE_SEED, BASE_SEED + N_TOTAL)
            if not os.path.exists(os.path.join(FIELD_DIR, f"seed_{s}.npy"))]
    print(f"σ 场缓存：需仿真 {len(todo)}/{N_TOTAL}（已有 "
          f"{N_TOTAL - len(todo)}）...")
    ref_sim = None
    for i, seed in enumerate(todo):
        sim = run_sample(seed, spikes_for(seed, dprime))
        np.save(os.path.join(FIELD_DIR, f"seed_{seed}.npy"),
                sim._sigma_max_field.astype(np.float32))
        ref_sim = sim
        print(f"  [{i+1}/{len(todo)}] seed={seed} "
              f"峰值={sim._sigma_max_field.max()/1e6:.1f} MPa")

    # 阶段 2：产线口径离线扫描（免仿真）
    dummy = PatchSimulator(patch_size=PATCH, thermal_profile=FAST_PROFILE,
                           crack_stress_threshold_MPa=10.0, seed=0)

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

    print("\n===== 产线口径 th 细扫（_extract_crack_mask 全流程，n=100）=====")
    print(f"{'th':>5}{'正样本':>8}{'线状数':>8}{'线状率':>8}{'外推1000':>10}"
          f"{'面积max':>9}{'elong_p50':>10}")
    table = {}
    for th in THS:
        dummy.crack_threshold_Pa = th * 1e6
        n_pos = n_line = 0
        areas, elongs = [], []
        for seed in range(BASE_SEED, BASE_SEED + N_TOTAL):
            fld = np.load(os.path.join(FIELD_DIR, f"seed_{seed}.npy"))
            st = gt_stats(dummy._extract_crack_mask(fld))
            n_pos += st["n_px"] > 0
            n_line += st["line_like"]
            if st["n_px"] > 0:
                areas.append(st["area_frac"])
            if st["line_like"]:
                elongs.append(st["elong"])
        table[th] = (n_pos, n_line)
        print(f"{int(th):>5}{n_pos:>8}{n_line:>8}"
              f"{n_line/max(1,n_pos)*100:>7.0f}%{n_line*10:>10}"
              f"{max(areas)*100 if areas else 0:>8.1f}%"
              f"{np.percentile(elongs,50) if elongs else 0:>10.2f}")

    # 阶段 3：4215 解剖（闭运算贴边腐蚀实证）
    print("\n===== seed 4215 解剖（闭运算 border_value=0 贴边腐蚀假设）=====")
    fld = np.load(os.path.join(FIELD_DIR, "seed_4215.npy"))
    for th in (40.0, 50.0):
        raw = (fld > th * 1e6)
        n_raw = int(raw.sum())
        ys, xs = np.nonzero(raw)
        loc = f"y∈[{ys.min()},{ys.max()}] x∈[{xs.min()},{xs.max()}]" \
            if n_raw else "-"
        dummy.crack_threshold_Pa = th * 1e6
        proc = dummy._extract_crack_mask(fld)
        n_proc = int((proc > 0).sum())
        print(f"  th={int(th)}: 裸阈值像素={n_raw} 位置[{loc}] → "
              f"闭运算后像素={n_proc} "
              f"（删光={n_proc==0}；贴边行占比="
              f"{(ys==0).mean()*100 if n_raw else 0:.0f}% y=0 / "
              f"{(ys==PATCH-1).mean()*100 if n_raw else 0:.0f}% y=127 / "
              f"{(xs==0).mean()*100 if n_raw else 0:.0f}% x=0 / "
              f"{(xs==PATCH-1).mean()*100 if n_raw else 0:.0f}% x=127）")

    # 全体统计：各 th 下"裸阈值有像素但闭运算删光"的样本数
    print("\n===== 闭运算删光效应统计（裸>0 → 产线=0）=====")
    for th in THS:
        dummy.crack_threshold_Pa = th * 1e6
        killed = 0
        for seed in range(BASE_SEED, BASE_SEED + N_TOTAL):
            fld = np.load(os.path.join(FIELD_DIR, f"seed_{seed}.npy"))
            if (fld > th * 1e6).any():
                proc = dummy._extract_crack_mask(fld)
                killed += (proc > 0).sum() == 0
        print(f"  th={int(th)}: {killed}/{N_TOTAL} 被删光（这些在裸口径下是正样本）")
