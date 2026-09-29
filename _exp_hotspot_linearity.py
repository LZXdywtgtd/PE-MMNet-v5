# -*- coding: utf-8 -*-
"""P0 验证实验：局部温度异常 → 线状裂纹 or 满面？（2026-09-29）

回答的问题（用户+外部 AI 会诊结论）：v5-α GT 满面退化是不是仿真器的必然？

物理机制分析（实验前预判）：
    MaxwellStress.update_step 是逐像素标量模型——σ(x,y) 只依赖该像素自身
    的 dT 历史，无空间耦合、无力学平衡方程。因此：
    - 热点（圆形凸斑）→ 应力场是平滑凸斑 → 阈值化出圆斑，非线状
    - 线状冷/热缺陷 → 应力场沿线状结构 → 阈值化可能出线状 mask
    若预判成立，"满面"不是 bug 而是标量应力模型的几何必然；
    路线 A（空间场扰动）必须用**线状**扰动而非圆形热点才能出线状 GT。

对照组设计：
    A. 基线（无扰动）——预期满面或空（已知）
    B. 圆形热点（中心 5×5 +200℃ 初温偏移）——用户提议的路线 A 原案
    C. 线状冷斑（对角线划痕 −150℃ 初温偏移，宽 2px）——预判的可行变体
    D. 线状热斑（水平线 +150℃）

每组跑 128px 默认曲线（~134s/样本 @8进程单样本≈独占1核），th 扫描
[8, 10, 12, 15]，报告每个 th 下的 GT 面积占比分布（纪律：不能只数正样本率）。
用 fast_thermal（1min 短周期）做首轮筛选——若线状可行再上默认曲线验证。
"""

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import functools

    print = functools.partial(print, flush=True)
    import numpy as np

    from data.patch_simulator_v5 import PatchSimulator
    PATCH = 128
    THS = [8.0, 10.0, 12.0, 15.0]
    FAST_PROFILE = {
        "ramp_up_c_per_min": 1260.0,
        "soak_temp_c": 1280.0,
        "soak_duration_min": 1.0,
        "cool_down_c_per_min": 1260.0,
    }

    def gt_stats(mask: np.ndarray) -> dict:
        """GT 形态统计：面积占比 + 连通域elongation（纪律：必须报形态）"""
        from scipy import ndimage

        total = mask.size
        n_px = int((mask > 0).sum())
        if n_px == 0:
            return {"area_frac": 0.0, "n_comp": 0, "elong": 0.0, "line_like": False}
        area_frac = n_px / total
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

    def perturb_factory(kind: str):
        """返回注入函数：在 simulate() 初始化 T_field 后被调用"""

        def inject(sim: PatchSimulator) -> None:
            H = W = sim.patch_size
            if kind == "baseline":
                return
            if kind == "hotspot":  # B: 圆形热点，中心 5×5 +200℃
                c = H // 2
                sim.T_field[c - 2:c + 3, c - 2:c + 3] += 200.0
            elif kind == "cold_line":  # C: 对角线冷斑，宽2px −150℃
                for i in range(H):
                    sim.T_field[i, max(0, i - 1):min(W, i + 2)] -= 150.0
            elif kind == "hot_line":  # D: 水平热线 +150℃
                r = H // 2
                sim.T_field[r - 1:r + 2, :] += 150.0
            # 注入后扰动会向周围扩散（热传导），线宽 2px 在 dt=CFL 下稳定

        return inject

    results = {}
    for kind in ["baseline", "hotspot", "cold_line", "hot_line"]:
        sim = PatchSimulator(
            patch_size=PATCH,
            thermal_profile=FAST_PROFILE,
            crack_stress_threshold_MPa=10.0,  # 阈值只影响 mask 提取，不影响应力场
            seed=42,
        )
        inject = perturb_factory(kind)
        # monkey-patch：在 simulate 初始化温度场后注入
        orig_simulate = sim.simulate

        # 手动重放 simulate 的初始化段 + 注入 + 完整步进（避免改生产代码）
        H = W = PATCH
        sim.T_field = np.full((H, W), 20.0, dtype=np.float64)
        if getattr(sim, "_init_temp_noise_std_c", 0) > 0:
            sim.T_field += sim._init_rng.normal(0.0, sim._init_temp_noise_std_c, size=(H, W))
        inject(sim)  # ← 实验注入点
        sim.sigma_field = np.zeros((H, W), dtype=np.float64)
        sim._sigma_max_field = np.zeros((H, W), dtype=np.float64)
        from collections import deque
        sim._T_hist_buf = deque(maxlen=sim.max_history_samples)
        sim._sigma_hist_buf = deque(maxlen=sim.max_history_samples)
        sim.t_history = []

        from data.thermal_profile import generate_thermal_profile
        times_s, target_temps_c = generate_thermal_profile(
            **FAST_PROFILE, dt_s=sim.dt
        )
        sim._target_temps_c = target_temps_c
        sim._times_s = times_s

        for step in range(len(times_s)):
            T_target = target_temps_c[step]
            sim._step_heat(sim.T_field, T_target)
            if step > 0:
                dT = sim.T_field - sim._T_prev_for_stress
                sim.sigma_field = sim.stress_solver.update_step(sim.sigma_field, dT, sim.dt)
            sim._T_prev_for_stress = sim.T_field.copy()
            np.maximum(sim._sigma_max_field, sim.sigma_field, out=sim._sigma_max_field)

        peak = sim._sigma_max_field.max() / 1e6
        per_th = {}
        for th in THS:
            sim.crack_threshold_Pa = th * 1e6
            mask = sim._extract_crack_mask(sim._sigma_max_field)
            per_th[th] = gt_stats(mask)
        results[kind] = {"peak_MPa": peak, "per_th": per_th}
        print(f"\n===== {kind} ===== 峰值拉应力 {peak:.2f} MPa")
        for th, st in per_th.items():
            tag = "LINE✓" if st["line_like"] else ("FULL✗" if st["area_frac"] > 0.5 else "other")
            print(f"  th={th:5.1f}  面积占比={st['area_frac']*100:6.2f}%  "
                  f"连通域={st['n_comp']:3d}  elong={st['elong']:5.2f}  [{tag}]")

    print("\n===== 结论要点 =====")
    for kind, r in results.items():
        line_any = any(st["line_like"] for st in r["per_th"].values())
        print(f"  {kind:10s}: 峰值={r['peak_MPa']:.1f}MPa  "
              f"任一 th 出线状: {'是' if line_any else '否'}")
