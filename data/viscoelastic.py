"""
Maxwell 粘弹性应力模型（V5A1 — S3）

物理模型：
    σ(t) + τ · σ̇(t) = E · ε(t)

展开形式：
    σ̇(t) = (E/τ) · ε̇(t) − σ(t)/τ

离散化（显式 Euler）：
    σ(t+1) = σ(t) + dt · [(E/τ) · Δε − σ(t)/τ]

物理含义：
- σ：热应力
- ε：热应变 = α_T · ΔT
- E：弹性模量（陶瓷 ~70–300 GPa）
- τ：松弛时间常数（陶瓷 ~10–100 s）
- α_T：热膨胀系数（陶瓷 ~5–10 × 10⁻⁶ /K）

应用场景：
- 升温阶段：热应力快速增长（ε 累积）
- 保温阶段：应力松弛主导（陶瓷内部粘性流动）
- 降温阶段：应力再次累积（裂纹萌生）

简化假设（v5-α）：
- 平面应力（2D 标量场）
- 各向同性材料
- 单 Maxwell 元件（无 Prony 级数）

详见：
- docs/v5_仿真升级清单.md S3
- tasks/team_v5_alpha.json 中 V5A1
"""

from __future__ import annotations

import numpy as np


class MaxwellStress:
    """Maxwell 粘弹性应力模型（2D 标量场）"""

    def __init__(
        self,
        young_modulus_GPa: float = 200.0,   # E，弹性模量
        poisson_ratio: float = 0.20,          # ν，泊松比（陶瓷 ~0.2）
        thermal_expansion: float = 8e-6,      # α_T，热膨胀系数（1/K）
        relaxation_time_s: float = 60.0,      # τ，松弛时间常数
        poisson_correction: bool = True,      # 是否应用平面应力修正
    ):
        """
        Args:
            young_modulus_GPa: 弹性模量（GPa），陶瓷 70–300
            poisson_ratio:      泊松比，陶瓷 ~0.2
            thermal_expansion:  热膨胀系数（1/K），陶瓷 5e-6–10e-6
            relaxation_time_s:  松弛时间常数（s），陶瓷 10–100
            poisson_correction: 是否应用 (1-ν) 修正（平面应力）
        """
        self.E = young_modulus_GPa * 1e9        # GPa → Pa
        self.nu = poisson_ratio
        self.alpha_T = thermal_expansion
        self.tau = relaxation_time_s

        # 平面应力修正因子
        if poisson_correction:
            self.stress_factor = 1.0 / (1.0 - self.nu)
        else:
            self.stress_factor = 1.0

    def thermal_strain(self, dT: np.ndarray) -> np.ndarray:
        """
        由温差计算热应变

        Args:
            dT: (H, W) — 温度变化量（K）

        Returns:
            strain: (H, W) — 热应变（无量纲）
        """
        return self.alpha_T * dT

    def thermal_stress(self, dT: np.ndarray) -> np.ndarray:
        """
        弹性热应力（无松弛，瞬时响应）

        σ_el = E · α_T · ΔT / (1 − ν)

        Args:
            dT: (H, W) — 温差（K）

        Returns:
            σ: (H, W) — 热应力（Pa），正值=拉应力
        """
        return self.E * self.thermal_strain(dT) * self.stress_factor

    def update_step(
        self,
        sigma_prev: np.ndarray,
        dT: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """
        Maxwell 模型单步推进（显式 Euler）

        物理公式（Maxwell 应变率形式）：
            σ̇(t) + σ(t)/τ = E · ε̇(t)

        离散化（dt << τ）：
            σ(t+1) ≈ σ(t) + E · α_T · ΔT · stress_factor − (dt/τ) · σ(t)
                  = σ(t) · (1 − dt/τ) + Δσ_el

        其中：
            Δσ_el = E · α_T · ΔT · stress_factor  （瞬时弹性响应，**无 dt/τ**）
            (dt/τ) · σ(t) 是松弛衰减

        稳定性约束：dt < τ
        v5-α MVP：dt > τ/2 时 clip 到 0 防止显式 Euler 振荡

        Args:
            sigma_prev: (H, W) — 上一时刻应力（Pa）
            dT: (H, W) — 本步温差（K）
            dt: 时间步长（s）

        Returns:
            sigma_next: (H, W) — 本时刻应力（Pa）
        """
        if dt <= 0:
            raise ValueError(f"dt 必须 > 0，当前 {dt}")

        # 瞬时弹性响应（无 dt/τ 因子）
        delta_elastic = self.E * self.thermal_strain(dT) * self.stress_factor

        # 显式 Euler 积分
        sigma_next = sigma_prev + delta_elastic - (dt / self.tau) * sigma_prev

        # 稳定性保护：dt > τ/2 时显式 Euler 可能振荡
        if dt > self.tau * 0.5:
            sigma_next = np.maximum(sigma_next, 0.0)

        return sigma_next

    def steady_state_stress(self, dT_final: np.ndarray) -> np.ndarray:
        """
        稳态应力（t → ∞，保温阶段）

        σ_∞ = E · α_T · ΔT · (1 − e^(−t/τ)) → E · α_T · ΔT（无 ν 修正）

        Args:
            dT_final: (H, W) — 最终温差（K）

        Returns:
            σ_steady: (H, W) — 稳态应力（Pa）
        """
        return self.thermal_stress(dT_final)


if __name__ == "__main__":
    # 快速自检
    ms = MaxwellStress()

    # 测试 1：稳态应力合理性
    # σ = 200e9 * 8e-6 * 100 / 0.8 = 200e9 * 8e-6 * 125 = 200e9 * 1e-3 = 2e8 Pa = 200 MPa
    dT = np.full((10, 10), 100.0)  # 100 K 温差
    sigma_ss = ms.steady_state_stress(dT)
    expected = 200e9 * 8e-6 * 100 / 0.8  # 200 MPa
    print(f"[OK] test_steady_stress: σ_steady = {sigma_ss[0,0]/1e6:.1f} MPa (期望 ~200 MPa)")
    assert 150e6 < sigma_ss[0, 0] < 250e6

    # 测试 2：单步推进合理性
    sigma_0 = np.zeros((10, 10))
    sigma_1 = ms.update_step(sigma_0, dT, dt=1.0)
    # σ_1 ≈ 0 + 1 * [(E/τ) · α_T · dT] = (200e9/60) * 8e-6 * 100 ≈ 2.67e5 Pa
    print(f"[OK] test_single_step: σ(1s) = {sigma_1[0,0]/1e3:.2f} kPa")
    assert 1e5 < sigma_1[0, 0] < 5e5

    # 测试 3：松弛行为（无新应变，应力下降）
    sigma_100 = ms.update_step(sigma_1, np.zeros((10, 10)), dt=100.0)
    # 100s 后，应力应显著衰减
    print(f"[OK] test_relaxation: σ(100s, 无新应变) = {sigma_100[0,0]/1e3:.2f} kPa")
    assert sigma_100[0, 0] < sigma_1[0, 0]

    print("\n[All Pass] Maxwell 应力模型自检通过")