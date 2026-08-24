"""
三段升降温曲线生成器（V5A1 — S4）

会议 5 决策：三段式升降温曲线（升温 / 保温 / 降温）。

物理含义：
- 升温：从室温升至烧成温度（陶瓷致密化）
- 保温：维持最高温度（晶相稳定化）
- 降温：从最高温度冷却（裂纹萌生主因）

输出：时间数组（秒）+ 目标温度数组（°C），供 PatchSimulator 调用。

详见：
- docs/v5_仿真升级清单.md S4
- tasks/team_v5_alpha.json 中 V5A1
"""

from __future__ import annotations

import numpy as np


def generate_thermal_profile(
    ramp_up_c_per_min: float = 5.0,
    soak_temp_c: float = 1280.0,
    soak_duration_min: float = 30.0,
    cool_down_c_per_min: float = 3.0,
    room_temp_c: float = 20.0,
    dt_s: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    生成三段升降温曲线

    Args:
        ramp_up_c_per_min:    升温速率（°C/min）
        soak_temp_c:          保温温度（°C，陶瓷烧成温度）
        soak_duration_min:    保温时长（min）
        cool_down_c_per_min:  降温速率（°C/min）
        room_temp_c:          室温（°C）
        dt_s:                 时间步长（秒）

    Returns:
        times: (T,) — 时间数组（秒）
        temps: (T,) — 目标温度数组（°C）

    Examples:
        >>> times, temps = generate_thermal_profile()
        >>> len(times) == len(temps)
        True
        >>> temps.max() == 1280.0
        True
        >>> times[-1] ≈ (240 + 30 + 400) * 60  # 总时长约 11h
        True
    """
    # 升温段时间
    delta_T_up = soak_temp_c - room_temp_c
    duration_up_min = delta_T_up / ramp_up_c_per_min  # 240 min
    duration_soak_min = soak_duration_min               # 30 min
    duration_cool_min = delta_T_up / cool_down_c_per_min  # 400 min

    # 各段步数
    n_up = max(1, int(duration_up_min * 60 / dt_s))
    n_soak = max(1, int(duration_soak_min * 60 / dt_s))
    n_cool = max(1, int(duration_cool_min * 60 / dt_s))

    # 升温段（线性升温）
    t_up = np.linspace(0, duration_up_min * 60, n_up, endpoint=False)
    T_up = room_temp_c + np.linspace(0, delta_T_up, n_up, endpoint=False)

    # 保温段（恒温）
    t_soak = t_up[-1] + dt_s + np.arange(n_soak) * dt_s
    T_soak = np.full(n_soak, soak_temp_c)

    # 降温段（线性降温）
    t_cool = t_soak[-1] + dt_s + np.arange(n_cool) * dt_s
    T_cool = soak_temp_c - np.linspace(0, delta_T_up, n_cool, endpoint=False)

    times = np.concatenate([t_up, t_soak, t_cool])
    temps = np.concatenate([T_up, T_soak, T_cool])

    return times, temps


def total_duration_s(
    ramp_up_c_per_min: float = 5.0,
    soak_temp_c: float = 1280.0,
    soak_duration_min: float = 30.0,
    cool_down_c_per_min: float = 3.0,
    room_temp_c: float = 20.0,
) -> float:
    """计算总烧制时长（秒）"""
    delta_T = soak_temp_c - room_temp_c
    duration_up_min = delta_T / ramp_up_c_per_min
    duration_cool_min = delta_T / cool_down_c_per_min
    return (duration_up_min + soak_duration_min + duration_cool_min) * 60.0


if __name__ == "__main__":
    # 快速自检
    times, temps = generate_thermal_profile()
    print(f"总时长: {times[-1]:.0f} s = {times[-1]/3600:.2f} h")
    print(f"步数: {len(times)}")
    print(f"温度范围: {temps.min():.1f} - {temps.max():.1f} °C")
    print(f"升温段末: T={temps[times <= 240*60].max():.1f} °C @ t={times[times <= 240*60].max():.0f} s")