# -*- coding: utf-8 -*-
"""安全网验证：D' 线状 GT 在修复后 loss 下可学性（2026-09-29）

外部 AI 建议：合并阶段（直接 D'+th=40 全量生成）之前，先做零成本验证——
不用 1000 样本不用训练，只用 30 个 D' 样本测 loss 数值行为：
  1. 三形态排序：真kpts(理想输出) vs 随机点 vs 角点塌缩 → 真kpts 必须胜出
  2. 判据 4 上界：bidir chamfer(GT kpts, GT 像素) 逐样本分布
  3. 判据 3 下界参考：GT kpts 跨样本方差（GT 侧多样性，非模型侧）

样本配置从 _exp_boundary_dprime_results.jsonl 复查（同 seed 同参数，
保证与 50 样本统计一致）；th=40，只取 th40 线状样本。
"""
import os
import sys
import json

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import functools

    print = functools.partial(print, flush=True)
    import numpy as np
    import torch
    from scipy import ndimage

    from data.patch_simulator_v5 import PatchSimulator
    from data.thermal_profile import generate_thermal_profile
    from training.ordered_kp_loss import OrderedKeypointLoss

    PATCH = 128
    TH = 40.0
    N = 30
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "_exp_boundary_dprime_results.jsonl")

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
        sim.crack_threshold_Pa = TH * 1e6
        mask = sim._extract_crack_mask(sim._sigma_max_field)
        kpts, bbox = sim._extract_keypoints(mask)
        return mask, kpts, bbox

    # 读 D' 结果：取 th40 线状的前 N 个 seed
    recs = []
    with open(OUT, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["per_th"].get("th40", {}).get("line_like"):
                recs.append(r)
    recs = recs[:N]
    print(f"取 th40 线状样本 {len(recs)} 个")

    torch.manual_seed(0)
    loss_fn = OrderedKeypointLoss()
    K = 12
    loss_fn.lambda_kpt_direct = 0.5

    all_kpts = []
    ub_list = []
    rank_pass = 0
    print(f"\n{'seed':<6}{'真kpts':>9}{'随机':>9}{'角点':>9}{'排序':>6}{'chamfer_ub':>11}")
    for r in recs:
        seed = r["seed"]
        mask, kpts, bbox = run_sample(seed, [tuple(s) for s in r["spikes"]])
        ys, xs = np.nonzero(mask > 0)
        gt_px = torch.from_numpy(
            np.stack([ys / PATCH, xs / PATCH], 1).astype(np.float32)).unsqueeze(0)

        # 真关键点（理想输出）
        true_k = torch.from_numpy(
            np.stack([kpts[:, 0] / PATCH, kpts[:, 1] / PATCH], 1)
            .astype(np.float32)).unsqueeze(0).repeat(4, 1, 1)
        # 随机点
        rng = np.random.default_rng(seed)
        rand_k = torch.from_numpy(
            rng.uniform(0.02, 0.98, (4, K, 2)).astype(np.float32))
        # 角点塌缩（平台解形态）
        corner_k = torch.tensor([[0., 0.]] * K).unsqueeze(0).repeat(4, 1, 1)
        corner_k[1] = 0.0; corner_k[2] = 1.0

        px = gt_px.repeat(4, 1, 1)
        l_true = loss_fn(torch.rand(4, 4), true_k, torch.rand(4, 4), px)["total"].item()
        l_rand = loss_fn(torch.rand(4, 4), rand_k, torch.rand(4, 4), px)["total"].item()
        l_corner = loss_fn(torch.rand(4, 4), corner_k, torch.rand(4, 4), px)["total"].item()
        ok = l_true < l_rand and l_true < l_corner
        rank_pass += ok

        # 判据 4 上界（GT 侧，独立于 loss）
        occ = np.zeros((PATCH, PATCH), bool); occ[ys, xs] = True
        edt_px = ndimage.distance_transform_edt(~occ)
        d1 = float(np.mean(edt_px[kpts[:, 0].astype(int), kpts[:, 1].astype(int)]))
        occ_k = np.zeros((PATCH, PATCH), bool)
        occ_k[kpts[:, 0].astype(int), kpts[:, 1].astype(int)] = True
        edt_k = ndimage.distance_transform_edt(~occ_k)
        sel = np.random.default_rng(0).choice(len(ys), min(4000, len(ys)), replace=False)
        d2 = float(np.mean(edt_k[ys[sel], xs[sel]]))
        ub = (d1 + d2) / 2 / PATCH
        ub_list.append(ub)

        all_kpts.append(np.stack([kpts[:, 0] / PATCH, kpts[:, 1] / PATCH], 1))
        print(f"{seed:<6}{l_true:>9.4f}{l_rand:>9.4f}{l_corner:>9.4f}"
              f"{'✓' if ok else '✗':>6}{ub:>11.4f}")

    # 判据 3 下界参考：GT kpt 集合两两 chamfer（K 动态 8-16，不能 stack，
    # 用集合间距离度量"不同样本的 GT 关键点确实不同"）
    def set_chamfer(a, b):
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
        return (d.min(1).mean() + d.min(0).mean()) / 2

    n = len(all_kpts)
    pair_d = [set_chamfer(all_kpts[i], all_kpts[j])
              for i in range(n) for j in range(i + 1, n)]
    pair_d = np.array(pair_d)
    ub_arr = np.array(ub_list)
    print(f"\n===== 汇总（n={n}，th=40）=====")
    print(f"三形态排序通过率：{rank_pass}/{n}"
          f"（真kpts 同时优于随机+角点）")
    print(f"chamfer_ub：p50={np.percentile(ub_arr,50):.4f} "
          f"p90={np.percentile(ub_arr,90):.4f} max={ub_arr.max():.4f}")
    print(f"GT kpt 集合两两 chamfer（判据3 GT侧多样性参考）："
          f"p50={np.percentile(pair_d,50):.4f} p05={np.percentile(pair_d,5):.4f}"
          f"（若 GT 塌缩到角点此值≈0）")
    verdict = "PASS" if rank_pass == n else "FAIL"
    print(f"结论：{verdict}（排序全过 + chamfer_ub 紧凑 → 放行全量生成；"
          f"判据 3 严格版仍需训练后实测 pred_kpts 跨输入方差）")
