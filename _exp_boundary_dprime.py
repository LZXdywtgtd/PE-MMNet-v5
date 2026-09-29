# -*- coding: utf-8 -*-
"""D' 50 样本统计：随机边 + 随机段 + 随机强度（2026-09-29）

外部 AI 复核意见：D3 整边 h×4 单配置单样本，bbox 位置多样性必不达标
（判据 2 跨象限 ≥3/4），面积 17.94% 贴门槛。D' 方案：每次采样
  - 从 4 条边随机选 1 条
  - 在该边上随机选一段（seg_len ∈ [1/3, 2/3] 边长），中心 c ∈ [0.3, 0.7]
  - 强化倍数 boost ∈ {2, 3, 4}
  - 平滑窗（smoothstep，r=0.04 过渡）避免间断

50 seeds，th ∈ {30,40,50,60} 逐样本提取统计（GT 生产用固定 th，
此处多 th 报告是为了选产线 th）。指标：
  - 正样本率 / 线状率（判据 1：正样本分母 ≥60% 且绝对数 ≥300）
  - bbox 中心象限分布（判据 2：跨象限 ≥3/4）
  - 关键点提取质量：端点数、骨架点数、K
  - 判据 4 上界基线：bidir chamfer(GT kpts, GT 像素) 分布
JSONL 断点续跑（每样本落一行，重跑跳过已有 seed）。
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
    N_SAMPLES = 50
    BASE_SEED = 4200
    THS = [30.0, 40.0, 50.0, 60.0]
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_exp_boundary_dprime_results.jsonl")

    FAST_PROFILE = {
        "ramp_up_c_per_min": 1260.0,
        "soak_temp_c": 1280.0,
        "soak_duration_min": 1.0,
        "cool_down_c_per_min": 1260.0,
    }

    def smoothstep(u):
        u = np.clip(u, 0.0, 1.0)
        return u * u * (3 - 2 * u)

    def seg_hmap(axis_vals, c, seg_len, boost, ramp=0.04):
        """1+（boost-1)·窗：[c-seg/2, c+seg/2] 平台，smoothstep 过渡"""
        a, b = c - seg_len / 2, c + seg_len / 2
        w = smoothstep((axis_vals - a) / ramp) * (1 - smoothstep((axis_vals - b) / ramp))
        return 1.0 + (boost - 1.0) * w

    def gt_stats(mask):
        n_px = int((mask > 0).sum())
        if n_px == 0:
            return {"n_px": 0, "area_frac": 0.0, "n_comp": 0, "elong": 0.0,
                    "bbox_c": None, "line_like": False}
        area_frac = n_px / mask.size
        labeled, n_comp = ndimage.label(mask > 0)
        sizes = ndimage.sum(mask > 0, labeled, range(1, n_comp + 1))
        largest = mask * (labeled == (int(np.argmax(sizes)) + 1))
        ys, xs = np.nonzero(largest)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        elong = max(h, w) / max(1, min(h, w))
        y1, y2, x1, x2 = ys.min(), ys.max(), xs.min(), xs.max()
        bbox_c = ((x1 + x2) / 2 / PATCH, (y1 + y2) / 2 / PATCH)
        return {
            "n_px": n_px, "area_frac": float(area_frac), "n_comp": int(n_comp),
            "elong": float(elong), "bbox_c": bbox_c,
            "line_like": bool(area_frac < 0.20 and elong > 3.0),
        }

    def chamfer_stats(kpts, mask):
        """判据 4 上界基线：bidir chamfer(GT kpts, GT 像素)，归一化 patch 边长"""
        ys, xs = np.nonzero(mask > 0)
        if len(ys) == 0 or len(kpts) == 0:
            return None
        px = np.stack([ys, xs], 1).astype(np.int32)
        # kpts → 像素集
        occ_px = np.zeros((PATCH, PATCH), bool)
        occ_px[ys, xs] = True
        edt_px = ndimage.distance_transform_edt(~occ_px)
        d1 = float(np.mean(edt_px[kpts[:, 0].astype(int), kpts[:, 1].astype(int)]))
        # 像素 → kpts 集
        occ_kp = np.zeros((PATCH, PATCH), bool)
        occ_kp[kpts[:, 0].astype(int), kpts[:, 1].astype(int)] = True
        edt_kp = ndimage.distance_transform_edt(~occ_kp)
        if len(px) > 4000:
            px_s = px[np.random.default_rng(0).choice(len(px), 4000, replace=False)]
        else:
            px_s = px
        d2 = float(np.mean(edt_kp[px_s[:, 0], px_s[:, 1]]))
        return (d1 + d2) / 2 / PATCH

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
        hmaps = {"top": None, "bottom": None, "left": None, "right": None}
        for edge, c, seg, boost in spikes:
            if edge in ("top", "bottom"):
                hmaps[edge] = seg_hmap(tcoord, c, seg, boost)
            else:
                hmaps[edge] = seg_hmap(ycoord, c, seg, boost)
        for k in hmaps:
            if hmaps[k] is None:
                hmaps[k] = np.ones(W if k in ("top", "bottom") else H)

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
                hm = hmaps[name]
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

        s = sim._sigma_max_field / 1e6
        per_th = {}
        for th in THS:
            sim.crack_threshold_Pa = th * 1e6
            mask = sim._extract_crack_mask(sim._sigma_max_field)
            st = gt_stats(mask)
            entry = dict(st)
            if st["n_px"] > 0:
                kpts, bbox = sim._extract_keypoints(mask)
                entry["kpts_K"] = int(len(kpts))
                entry["chamfer_ub"] = chamfer_stats(kpts, mask)
            else:
                entry["kpts_K"] = 0
                entry["chamfer_ub"] = None
            per_th[f"th{int(th)}"] = entry
        return {
            "seed": seed, "peak_MPa": float(s.max()),
            "p99_MPa": float(np.percentile(s, 99)),
            "spikes": spikes, "per_th": per_th,
        }

    # 断点续跑：跳过已有 seed
    done = set()
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            for line in f:
                try:
                    done.add(json.loads(line)["seed"])
                except Exception:
                    pass
    print(f"已完成 {len(done)}/{N_SAMPLES}，继续...")

    with open(OUT, "a", encoding="utf-8") as fout:
        for i in range(N_SAMPLES):
            seed = BASE_SEED + i
            if seed in done:
                continue
            rng = np.random.default_rng(seed)
            edge = str(rng.choice(["top", "bottom", "left", "right"]))
            c = float(rng.uniform(0.3, 0.7))
            seg = float(rng.uniform(1 / 3, 2 / 3))
            boost = float(rng.choice([2.0, 3.0, 4.0]))
            spikes = [(edge, c, seg, boost)]
            rec = run_sample(seed, spikes)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            line = {k: rec["per_th"][k]["line_like"] for k in rec["per_th"]}
            print(f"[{i+1}/{N_SAMPLES}] seed={seed} {edge} c={c:.2f} "
                  f"seg={seg:.2f} ×{boost:.0f} 峰值={rec['peak_MPa']:.1f} {line}")

    # 汇总
    recs = []
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            recs.append(json.loads(line))
    print(f"\n===== D' 汇总（n={len(recs)}）=====")
    for th in THS:
        key = f"th{int(th)}"
        n_pos = sum(1 for r in recs if r["per_th"][key]["n_px"] > 0)
        n_line = sum(1 for r in recs if r["per_th"][key]["line_like"])
        areas = [r["per_th"][key]["area_frac"] for r in recs if r["per_th"][key]["n_px"] > 0]
        quads = set()
        for r in recs:
            e = r["per_th"][key]
            if e["bbox_c"]:
                quads.add((e["bbox_c"][0] < 0.5, e["bbox_c"][1] < 0.5))
        cubs = [r["per_th"][key]["chamfer_ub"] for r in recs
                if r["per_th"][key]["chamfer_ub"] is not None]
        cub_s = (f"chamfer_ub p50={np.percentile(cubs,50):.4f} "
                 f"p90={np.percentile(cubs,90):.4f}") if cubs else "chamfer_ub=-"
        print(f"th={int(th)}: 正样本 {n_pos}/{len(recs)}  线状 {n_line} "
              f"({(n_line / max(1, n_pos) * 100):.0f}% of 正样本)  "
              f"面积 p50={np.percentile(areas, 50) * 100 if areas else 0:.1f}% "
              f"max={max(areas) * 100 if areas else 0:.1f}%  象限 {len(quads)}/4  {cub_s}")
