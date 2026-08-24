"""
V5A1 单元测试（thermal_profile + surface_radiation + viscoelastic + PatchSimulator）

测试策略：
- 每个模块独立的 `test_xxx()` 函数
- 用 `assert` 而非 pytest（无外部依赖）
- 失败时给出有意义的错误信息

运行：
    python tests/test_patch_simulator_v5.py
    或：
    python -m unittest tests.test_patch_simulator_v5 -v
"""

from __future__ import annotations

import sys
from pathlib import Path

# 添加项目根到 sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from data.thermal_profile import generate_thermal_profile, total_duration_s
from data.surface_radiation import SurfaceRadiation
from data.viscoelastic import MaxwellStress
from data.patch_simulator_v5 import PatchSimulator


# ============================================================
#  thermal_profile 测试
# ============================================================

def test_thermal_profile_basic():
    """测试三段曲线基本形态"""
    times, temps = generate_thermal_profile()

    assert len(times) == len(temps), "时间和温度数组长度应一致"
    assert times[0] == 0.0, "起始时间应为 0"
    assert abs(temps[0] - 20.0) < 0.1, "起始温度应为 20°C"
    assert abs(temps.max() - 1280.0) < 0.1, "峰值温度应为 1280°C"
    assert temps[-1] < 100, "终止温度应回到接近室温"
    print(f"  [OK] thermal_profile_basic: T_max={temps.max():.1f}°C, "
          f"总时长={times[-1]/3600:.2f}h")


def test_thermal_profile_duration():
    """测试总时长合理性"""
    total_s = total_duration_s()
    # 默认配置：升温 252 min + 保温 30 min + 降温 420 min = 702 min = 11.7 h
    expected_h = (252 + 30 + 420) / 60
    assert abs(total_s / 3600 - expected_h) < 0.1, \
        f"总时长应为 {expected_h:.2f}h，实际 {total_s/3600:.2f}h"
    print(f"  [OK] thermal_profile_duration: {total_s/3600:.2f}h")


def test_thermal_profile_monotonic_up_and_down():
    """测试升降温段单调性"""
    times, temps = generate_thermal_profile()
    # 升温段（0–4h）
    n_up = int(4 * 3600)  # 240 min * 60 s/min = 14400 s
    up_segment = temps[:n_up]
    assert np.all(np.diff(up_segment) >= 0), "升温段应单调递增"

    # 降温段（>4.5h）
    n_soak_end = int(4.5 * 3600)
    down_segment = temps[n_soak_end:]
    assert np.all(np.diff(down_segment) <= 0), "降温段应单调递减"
    print(f"  [OK] thermal_profile_monotonic: 升/降温段单调性正确")


# ============================================================
#  surface_radiation 测试
# ============================================================

def test_radiation_zero_diff():
    """测试无温差时辐射热流为 0"""
    sr = SurfaceRadiation()
    q = sr.heat_flux(293.15, 293.15)
    assert abs(q) < 1e-3, f"无温差时 q 应为 0，实际 {q}"
    print(f"  [OK] radiation_zero_diff: q={q:.3e}")


def test_radiation_high_temperature():
    """测试高温辐射热流合理性"""
    sr = SurfaceRadiation(emissivity=0.9)
    # 1280°C vs 20°C
    q = sr.heat_flux(1553.15, 293.15)
    expected = 0.9 * 5.670374419e-8 * (1553.15**4 - 293.15**4)
    assert abs(q - expected) / expected < 1e-9, "辐射公式精度"
    assert 2e5 < q < 4e5, f"1280°C 辐射热流应在 2e5–4e5，实际 {q:.2e}"
    print(f"  [OK] radiation_high_temperature: q={q:.2e} W/m²")


def test_radiation_emissivity_bounds():
    """测试发射率边界"""
    try:
        SurfaceRadiation(emissivity=1.5)
        assert False, "应拒绝超出范围的发射率"
    except ValueError:
        pass
    print(f"  [OK] radiation_emissivity_bounds: 边界检查通过")


# ============================================================
#  viscoelastic 测试
# ============================================================

def test_viscoelastic_steady_state():
    """测试 Maxwell 稳态应力"""
    ms = MaxwellStress()
    dT = np.full((10, 10), 100.0)  # 100 K
    sigma_ss = ms.steady_state_stress(dT)
    expected = 200e9 * 8e-6 * 100 / 0.8  # 200 MPa
    assert 150e6 < sigma_ss[0, 0] < 250e6, \
        f"稳态应力应 ~200 MPa，实际 {sigma_ss[0,0]/1e6:.1f} MPa"
    print(f"  [OK] viscoelastic_steady_state: {sigma_ss[0,0]/1e6:.1f} MPa")


def test_viscoelastic_single_step():
    """测试单步推进（考虑 (1-ν) 修正）"""
    ms = MaxwellStress()
    sigma_0 = np.zeros((10, 10))
    dT = np.full((10, 10), 100.0)
    sigma_1 = ms.update_step(sigma_0, dT, dt=1.0)
    # 期望（修正后，无 dt/τ 因子）: E · α_T · ΔT · stress_factor
    #     = 200e9 * 8e-6 * 100 * 1.25 = 2.0e8 Pa = 200 MPa
    expected = 200e9 * 8e-6 * 100 * (1.0 / (1.0 - 0.2))
    assert 1e8 < sigma_1[0, 0] < 1e9, \
        f"σ 应在 [1e8, 1e9]，实际 {sigma_1[0,0]:.2e}"
    assert abs(sigma_1[0, 0] - expected) / expected < 0.05, "单步公式精度 5%"
    print(f"  [OK] viscoelastic_single_step: {sigma_1[0,0]/1e6:.1f} MPa (期望 {expected/1e6:.1f})")


def test_viscoelastic_relaxation():
    """测试松弛行为（无新应变时应力下降）"""
    ms = MaxwellStress()
    sigma_1 = np.full((10, 10), 1e6)  # 初始 1 MPa
    sigma_after = ms.update_step(sigma_1, np.zeros((10, 10)), dt=100.0)
    # 100s 后应衰减（dt > τ/2 时 clip 到 0）
    assert sigma_after[0, 0] <= sigma_1[0, 0], "无新应变时应力应下降或保持"
    assert sigma_after[0, 0] >= 0, "应力不能为负（物理合理性）"
    print(f"  [OK] viscoelastic_relaxation: 1MPa → {sigma_after[0,0]/1e3:.1f} kPa (100s)")


# ============================================================
#  PatchSimulator 集成测试
# ============================================================

def test_simulator_end_to_end():
    """端到端：小尺寸 + 短流程仿真"""
    sim = PatchSimulator(
        patch_size=64,                  # 小尺寸加速
        physical_size_cm=2.0,
        thermal_profile={
            "ramp_up_c_per_min": 50.0,  # 加速升温
            "soak_temp_c": 200.0,       # 降低温度
            "soak_duration_min": 1.0,
            "cool_down_c_per_min": 30.0,
        },
        crack_stress_threshold_MPa=5.0,  # 低阈值确保有裂纹
        seed=42,
    )
    result = sim.simulate(sample_interval_steps=200, verbose=False)

    # 输出格式检查
    assert result["temperature_field"].ndim == 3, "温度场应为 3D (T, H, W)"
    assert result["stress_field"].ndim == 3
    assert result["heatmap"].shape == (64, 64)
    assert result["crack_mask"].shape == (64, 64)
    assert result["crack_keypoints"].ndim == 2 and result["crack_keypoints"].shape[1] == 2

    # 关键点数量
    K = len(result["crack_keypoints"])
    assert 8 <= K <= 16, f"关键点数量应在 [8, 16]，实际 {K}"

    # bbox 在 [0, 1] 范围
    bbox = result["crack_bbox"]
    assert 0 <= bbox[0] <= bbox[2] <= 1, "bbox x 范围"
    assert 0 <= bbox[1] <= bbox[3] <= 1, "bbox y 范围"

    # 峰值应力
    assert result["metadata"]["peak_stress_MPa"] > 0
    print(f"  [OK] simulator_end_to_end: K={K}, "
          f"peak_σ={result['metadata']['peak_stress_MPa']:.1f} MPa, "
          f"crack_px={result['crack_mask'].sum()/255:.0f}")


def test_simulator_no_crack_when_low_stress():
    """当应力低于阈值时，无裂纹（边界情况）"""
    sim = PatchSimulator(
        patch_size=32,
        physical_size_cm=2.0,
        thermal_profile={
            "ramp_up_c_per_min": 10.0,
            "soak_temp_c": 50.0,    # 低温
            "soak_duration_min": 1.0,
            "cool_down_c_per_min": 10.0,
        },
        crack_stress_threshold_MPa=1000.0,  # 超高阈值
    )
    result = sim.simulate(verbose=False)
    assert result["crack_mask"].sum() == 0, "低应力应无裂纹"
    K = len(result["crack_keypoints"])
    assert K == sim.min_keypoints, "无裂纹时 K 应等于下界"
    print(f"  [OK] simulator_no_crack_when_low_stress: 空 mask + K={K}")


def test_simulator_reproducibility():
    """同种子应产生相同结果（可重复性）"""
    kwargs = dict(
        patch_size=32,
        physical_size_cm=2.0,
        thermal_profile={
            "ramp_up_c_per_min": 30.0,
            "soak_temp_c": 150.0,
            "soak_duration_min": 1.0,
            "cool_down_c_per_min": 20.0,
        },
        crack_stress_threshold_MPa=5.0,
    )

    result_a = PatchSimulator(seed=42, **kwargs).simulate(verbose=False)
    result_b = PatchSimulator(seed=42, **kwargs).simulate(verbose=False)

    assert np.allclose(result_a["crack_mask"], result_b["crack_mask"]), \
        "同种子应产生相同 crack mask"
    assert np.allclose(result_a["crack_keypoints"], result_b["crack_keypoints"]), \
        "同种子应产生相同关键点"
    print(f"  [OK] simulator_reproducibility: 种子可重复")


# ============================================================
#  测试入口
# ============================================================

def run_all():
    """运行所有测试"""
    tests = [
        # thermal_profile
        test_thermal_profile_basic,
        test_thermal_profile_duration,
        test_thermal_profile_monotonic_up_and_down,
        # surface_radiation
        test_radiation_zero_diff,
        test_radiation_high_temperature,
        test_radiation_emissivity_bounds,
        # viscoelastic
        test_viscoelastic_steady_state,
        test_viscoelastic_single_step,
        test_viscoelastic_relaxation,
        # PatchSimulator
        test_simulator_end_to_end,
        test_simulator_no_crack_when_low_stress,
        test_simulator_reproducibility,
    ]

    print("=" * 60)
    print(f"V5A1 单元测试（共 {len(tests)} 项）")
    print("=" * 60)

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {test.__name__}: {e}")

    print("=" * 60)
    print(f"结果：{passed} 通过，{failed} 失败")
    if failed == 0:
        print("[ALL PASS] ✅")
        return 0
    else:
        print("[SOME FAILED] ❌")
        return 1


if __name__ == "__main__":
    sys.exit(run_all())