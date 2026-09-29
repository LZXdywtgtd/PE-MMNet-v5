# -*- coding: utf-8 -*-
"""V5-025 诊断：默认曲线下 D' 调制消失主因——热扩散 vs Maxwell 松弛（2026-09-29）

用户要求：不修改代码，只分析已落盘的 σ 场数据。10 分钟内出结论。

两种机制的**可区分指纹**：
- 热扩散 = 空间低通：把调制段的空间图案抹平 → 沿边 bump 对比度崩塌，
  调制段足迹与对边的 σ 差消失。扩散长度 L=√(αt)：fast 11min→7px @128、
  默认 11.7h→52px（物理口径见脚本内常数）。
- Maxwell 松弛 = 逐像素时间低通（σ̇+σ/τ=E·α_T·Ṫ，准稳态 σ∝τ·|Ṫ|）：
  只缩幅、**保形**——空间图案不变，对比度（bump/中位数）应与 fast 同量级。

关键点：σ_max 是全程逐像素取 max——若任一时刻存在沿边对比，它会留在
σ_max 场里。因此对比度消失只能说明"全程都无空间对比"，而非"被后期抹掉"。

数据：_exp_sigma_fields/seed_4200..4299.npy（fast，D' 100 seeds）
     _exp_default_curve_fields_v2/seed_4500..4504,4600,4601.npy（默认曲线 7 seeds）
两批同一 D' 抽取逻辑（default_rng(seed)），σ 场同口径（手动循环复刻）。
"""
import os
import sys
import json

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import functools

    print = functools.partial(print, flush=True)
    import numpy as np

    PATCH = 128
    ROOT = os.path.dirname(os.path.abspath(__file__))

    # ── 物理常数（patch_simulator_v5 默认口径）──
    alpha_th = 1.5 / (2400.0 * 800.0)      # 7.8e-7 m²/s
    L_cm = 5.0
    dx_m = (L_cm / 100.0) / PATCH          # 3.9e-4 m/px
    E_GPa, a_T, tau_s = 200.0, 8e-6, 60.0  # 名义值（实际有 ±20/30% 随机化）

    def edge_profile_stats(fld, spikes):
        """沿调制边的 σ 分布 + 调制段 vs 段外 vs 对边的对比度。"""
        f = fld / 1e6  # MPa
        edge, c, seg, boost = spikes[0]
        axis = np.arange(PATCH) / PATCH
        inside = np.abs(axis - c) <= seg / 2
        if edge in ("top", "bottom"):
            band = f[0, :] if edge == "top" else f[-1, :]
            opp = f[-1, :] if edge == "top" else f[0, :]
        else:
            band = f[:, 0] if edge == "left" else f[:, -1]
            opp = f[:, -1] if edge == "left" else f[:, 0]
        on = band[inside]
        off = band[~inside]
        interior = f[16:-16, 16:-16].ravel()
        med = float(np.median(f))
        return {
            "on_med": float(np.median(on)), "off_med": float(np.median(off)),
            "opp_med": float(np.median(opp)),
            "interior_p99": float(np.percentile(interior, 99)),
            "contrast_on_off": float(np.median(on) - np.median(off)),
            "contrast_on_opp": float(np.median(on) - np.median(opp)),
            "field_med": med,
        }

    def load_batch(d, seeds):
        out = {}
        for s in seeds:
            p = os.path.join(ROOT, d, f"seed_{s}.npy")
            if os.path.exists(p):
                out[s] = np.load(p)
        return out

    def spikes_of(seed):
        rng = np.random.default_rng(seed)
        edge = str(rng.choice(["top", "bottom", "left", "right"]))
        c = float(rng.uniform(0.3, 0.7))
        seg = float(rng.uniform(1 / 3, 2 / 3))
        boost = float(rng.choice([2.0, 3.0, 4.0]))
        return [(edge, c, seg, boost)]

    # ══════ A. 空间对比度对比（裁决性判据）══════
    print("===== A. 沿边 σ 空间对比度：fast(n=100) vs 默认曲线(n=7) =====")
    print(f"{'batch':<10} | {'on-off 对比度':>14} {'on-opp 对比度':>14} "
          f"| {'场中位':>7} {'on/off 比':>9}")
    for name, d, seeds in [
        ("fast", "_exp_sigma_fields", range(4200, 4300)),
        ("default", "_exp_default_curve_fields_v2", [4500, 4501, 4502, 4503, 4504, 4600, 4601]),
    ]:
        flds = load_batch(d, seeds)
        co, cp, fm, ratio = [], [], [], []
        for s, fld in flds.items():
            st = edge_profile_stats(fld, spikes_of(s))
            co.append(st["contrast_on_off"])
            cp.append(st["contrast_on_opp"])
            fm.append(st["field_med"])
            if abs(st["off_med"]) > 1e-9:
                ratio.append(st["on_med"] / st["off_med"])
        print(f"{name:<10} | {np.median(co):>10.2f} MPa {np.median(cp):>10.2f} MPa "
              f"| {np.median(fm):>5.1f} MPa {np.median(ratio):>7.2f}x")

    # ══════ B. 量级：准稳态 Maxwell 预测 vs 实测峰值 ══════
    print("\n===== B. 量级校核：σ_∞≈τ·E·α_T·|dT/dt| =====")
    # fast: 1260 C/min = 21 K/s → σ∞ ≈ 60*200e9*8e-6*21 = 2.0e9?? 超大——
    # 但实测 fast 峰值 234 MPa。检查：h_conv 限制边界降温速率（边界不能瞬时
    # 跟上环境温度），实际 dT/dt 由 h·(T_env-T_edge)/(ρc·L) 决定。
    h, rho, cp_, Lm = 15.0, 2400.0, 800.0, dx_m / 2
    for name, rate_Ks in [("fast", 21.0), ("default", 5.0 / 60.0)]:
        # 边界像素最大冷却速率（一阶集总近似）
        dTdt_bound = h * 1260.0 / (rho * cp_ * Lm)  # T_env-T_edge 最大 ~1260K
        sigma_inf = tau_s * E_GPa * 1e9 * a_T * dTdt_bound
        print(f"  {name}: 边界集总 dT/dt 上限={dTdt_bound:.1f} K/s → "
              f"σ_∞≈{sigma_inf/1e6:.0f} MPa（集总上限，实测峰值见 A 表）")

    # ══════ C. 时间轴证据：σ_max 是不是早期瞬态留下的（保形 vs 抹平）══════
    print("\n===== C. 每像素 σ_max 的空间纯度：径向衰减（斑点指纹）vs 沿边带状 =====")
    print("    调制边行/列 σ 与到调制边距离 d 的剖面（n=100 fast vs n=7 default 各自中位）")
    for name, d, seeds in [
        ("fast", "_exp_sigma_fields", range(4200, 4300)),
        ("default", "_exp_default_curve_fields_v2", [4500, 4501, 4502, 4503, 4504, 4600, 4601]),
    ]:
        flds = load_batch(d, seeds)
        prof = np.zeros((5,))  # d=0,1,2,4,8 行的中位 σ
        for s, fld in flds.items():
            f = fld / 1e6
            edge = spikes_of(s)[0][0]
            for j, dd in enumerate([0, 1, 2, 4, 8]):
                row = f[dd, :] if edge == "top" else (
                    f[-1-dd, :] if edge == "bottom" else
                    (f[:, dd] if edge == "left" else f[:, -1-dd]))
                prof[j] += np.median(row)
        prof /= len(flds)
        print(f"  {name:<8}: " + "  ".join(f"d={dd}:{v:6.1f}" for dd, v in zip([0,1,2,4,8], prof)))

    # ══════ D. τ 敏感性反事实（解析，零仿真）══════
    print("\n===== D. 反事实校核（解析）：若主因是 Maxwell，τ∞ 时默认曲线应存活 =====")
    print("  默认曲线总长 42120s = 702 个 τ。每像素 dT/dt 在 ramp 阶段恒定，")
    print("  准稳态解 σ(t)→τ·E·α_T·rate（保形）；降温阶段反向弹性项把 σ 拉回负值，")
    print("  σ_max 记录升温段峰值。对比度（A 表 on/off 比）才是裁决量——")
    print("  Maxwell 保形 → 默认曲线 on/off 比应 ≈ fast；扩散抹平 → ≈1。")
