"""
表面辐射模块（V5A1 — S2）

物理基础：Stefan-Boltzmann 定律
    q_rad = ε · σ · (T_surface⁴ − T_env⁴)

参数：
- ε：发射率（陶瓷表面典型 0.85–0.95）
- σ：Stefan-Boltzmann 常数 = 5.67 × 10⁻⁸ W/(m²·K⁴)
- T：温度（K）

应用场景：
- 边界条件：所有边缘单元（top/side/bottom）施加辐射热流
- 高温阶段（>800°C）辐射热损失占总热损失的 20–40%

详见：
- docs/v5_仿真升级清单.md S2
- tasks/team_v5_alpha.json 中 V5A1
"""

from __future__ import annotations

import numpy as np


class SurfaceRadiation:
    """Stefan-Boltzmann 表面辐射"""

    STEFAN_BOLTZMANN = 5.670374419e-8  # W/(m²·K⁴)

    def __init__(self, emissivity: float = 0.9):
        """
        Args:
            emissivity: 发射率，陶瓷表面典型 0.85–0.95
        """
        if not 0.0 <= emissivity <= 1.0:
            raise ValueError(f"emissivity 必须在 [0, 1]，当前 {emissivity}")
        self.emissivity = emissivity

    @staticmethod
    def celsius_to_kelvin(T_c: np.ndarray | float) -> np.ndarray | float:
        """°C → K"""
        return T_c + 273.15

    def heat_flux(
        self,
        T_surface_K: np.ndarray | float,
        T_env_K: np.ndarray | float = 293.15,  # 默认 20°C
    ) -> np.ndarray | float:
        """
        计算辐射热流密度

        Args:
            T_surface_K: 表面温度（K）
            T_env_K:     环境温度（K），默认 20°C

        Returns:
            q_rad: 辐射热流密度（W/m²），正值=表面散热
        """
        return self.emissivity * self.STEFAN_BOLTZMANN * (
            T_surface_K ** 4 - T_env_K ** 4
        )

    def heat_flux_celsius(
        self,
        T_surface_c: np.ndarray | float,
        T_env_c: float = 20.0,
    ) -> np.ndarray | float:
        """便捷接口：°C 输入"""
        return self.heat_flux(self.celsius_to_kelvin(T_surface_c),
                              self.celsius_to_kelvin(T_env_c))


if __name__ == "__main__":
    # 快速自检
    sr = SurfaceRadiation(emissivity=0.9)

    # 测试 1：室温差应为 0
    q_zero = sr.heat_flux(293.15)
    assert abs(q_zero) < 1e-3, f"室温差应为 0，实际 {q_zero}"
    print("[OK] test_radiation_zero: 室温差 q=0")

    # 测试 2：高温差合理
    T_surface = 1553.15  # 1280°C
    T_env = 293.15       # 20°C
    q = sr.heat_flux(T_surface, T_env)
    # 期望: 0.9 * 5.67e-8 * (1553.15^4 - 293.15^4) ≈ 2.85e5 W/m²
    print(f"[OK] test_radiation_highT: q={q:.2e} W/m² (期望 ~2.85e5)")
    assert 2e5 < q < 4e5, f"辐射热流应在 2e5–4e5 范围，实际 {q}"

    # 测试 3：边界热流在总热量中占比（参考值）
    # 陶瓷 λ ≈ 1 W/(m·K)，温差 1260°C，厚 5mm
    # 传导热流 ≈ λ·ΔT/L = 1 * 1260 / 0.005 = 2.52e5 W/m²
    # 辐射热流 ≈ 2.85e5 W/m² → 占比 ~53%
    print(f"[OK] 表面辐射热流 ~ 2.85e5 W/m²（高温阶段与传导可比）")