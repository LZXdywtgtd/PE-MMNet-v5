# -*- coding: utf-8 -*-
"""th 细扫：100 样本 × th∈{35,40,45,50,55}，选线状绝对数最大的档（2026-09-29）

外部 AI 质疑 1：th=40"甜点"可能只是 50 样本噪声 + "第一个满员档位"；
50→49 差 1 个样本在噪声范围内。改为 100 样本细扫五档，直接统计
线状绝对数（判据 1 的真实约束是绝对数 ≥300，不是比率）。

复用 _exp_boundary_dprime_results.jsonl 的 50 seeds 配置 + 扩展 50 个新
seeds（4250-4299），D' 参数同口径：随机边/段中心[0.3,0.7]/段长[1/3,2/3]/
强度{2,3,4}。每样本只仿真一次，五个 th 从同一 σ_max 场提取（增量成本≈0）。
JSONL 断点续跑。
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
    THS = [35.0, 40.0, 45.0, 50.0, 55.0]
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_exp_th_sweep_results.jsonl")

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
        return {
            "n_px": n_px, "area_frac": float(area_frac), "n_comp": int(n_comp),
            "elong": float(elong),
            "line_like": bool(area_frac < 0.20 and elong > 3.0),
        }

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
        return sim._sigma_max_field / 1e6

    # 100 个样本配置（前 50 复用 D' JSONL，后 50 扩展 seeds）
    spikes_list = []
    dprime = {}
    dp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "_exp_boundary_dprime_results.jsonl")
    with open(dp_path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            dprime[r["seed"]] = r["spikes"]
    for i in range(N_TOTAL):
        seed = BASE_SEED + i
        if seed in dprime:
            spikes_list.append((seed, dprime[seed]))
        else:
            rng = np.random.default_rng(seed)
            edge = str(rng.choice(["top", "bottom", "left", "right"]))
            c = float(rng.uniform(0.3, 0.7))
            seg = float(rng.uniform(1 / 3, 2 / 3))
            boost = float(rng.choice([2.0, 3.0, 4.0]))
            spikes_list.append((seed, [(edge, c, seg, boost)]))

    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["seed"])
                except Exception:
                    pass
    print(f"已完成 {len(done)}/{N_TOTAL}，继续...")

    with open(OUT, "a", encoding="utf-8") as fout:
        for i, (seed, spikes) in enumerate(spikes_list):
            if seed in done:
                continue
            s = run_sample(seed, spikes)
            per_th = {}
            for th in THS:
                mask = (s > th).astype(np.uint8) * 255
                per_th[f"th{int(th)}"] = gt_stats(mask)
            rec = {"seed": seed, "peak_MPa": float(s.max()), "spikes": spikes,
                   "per_th": per_th}
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            line = {k: v["line_like"] for k, v in per_th.items()}
            print(f"[{i+1}/{N_TOTAL}] seed={seed} 峰值={s.max():.1f} {line}")

    recs = []
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    print(f"\n===== th 细扫汇总（n={len(recs)}，判据1=线状绝对数≥300@1000样本）=====")
    print(f"{'th':>5}{'正样本':>8}{'线状数':>8}{'线状率':>8}{'外推1000':>10}"
          f"{'面积max':>9}{'elong_p50':>10}")
    for th in THS:
        key = f"th{int(th)}"
        n_pos = sum(1 for r in recs if r["per_th"][key]["n_px"] > 0)
        n_line = sum(1 for r in recs if r["per_th"][key]["line_like"])
        areas = [r["per_th"][key]["area_frac"] for r in recs
                 if r["per_th"][key]["n_px"] > 0]
        elongs = [r["per_th"][key]["elong"] for r in recs
                  if r["per_th"][key]["line_like"]]
        print(f"{int(th):>5}{n_pos:>8}{n_line:>8}"
              f"{n_line/max(1,n_pos)*100:>7.0f}%{n_line*10:>10}"
              f"{max(areas)*100 if areas else 0:>8.1f}%"
              f"{np.percentile(elongs,50) if elongs else 0:>10.2f}")
