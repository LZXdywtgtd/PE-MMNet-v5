# -*- coding: utf-8 -*-
"""V5-027 回归测试：_extract_keypoints 碎片化骨架退化分支修复

历史 bug：骨架碎片化（多连通域）时 _find_endpoints 的两个端点可能落在
不同连通域 → _trace_path 跨域 BFS 返回 [] → len(path)<2 分支返回全部
骨架点（D' 全量实测 74~182 点，违反模型 max_kpts=16 契约）。

修复：len(path)<2 时改在最大骨架连通域内重追路径，统一走等弧长采样
（K ∈ [min_kpts, max_kpts]）。
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from data.patch_simulator_v5 import PatchSimulator

FAST_PROFILE = {
    "ramp_up_c_per_min": 1260.0,
    "soak_temp_c": 1280.0,
    "soak_duration_min": 1.0,
    "cool_down_c_per_min": 1260.0,
}


def _make_sim(seed=424242, patch_size=128):
    return PatchSimulator(
        patch_size=patch_size,
        thermal_profile=FAST_PROFILE,
        crack_stress_threshold_MPa=60.0,
        seed=seed,
    )


def test_v5027_fragmented_skeleton_bounded_kpts():
    """碎片化骨架（跨域端点）下 K ≤ max_kpts，且关键点在骨架上"""
    sim = _make_sim()
    skeleton = np.zeros((32, 32), dtype=bool)
    skeleton[5, 2:10] = True          # 域 1：横线（2 个端点）
    skeleton[20:27, 25] = True        # 域 2：竖线（2 个端点）
    keypoints, bbox = sim._extract_keypoints(skeleton.astype(np.uint8) * 255)
    assert sim.min_keypoints <= len(keypoints) <= sim.max_keypoints, (
        f"K={len(keypoints)} 超出 [{sim.min_keypoints},{sim.max_keypoints}]"
    )
    for y, x in keypoints:
        yi, xi = int(round(y)), int(round(x))
        assert 0 <= yi < 32 and 0 <= xi < 32
        assert skeleton[yi, xi], f"kpt ({yi},{xi}) 不在骨架上"


def test_v5027_closed_loop_domain_all_points_fallback():
    """闭环骨架（无端点）：最大域取全部点后截断保 K 契约"""
    sim = _make_sim()
    skeleton = np.zeros((32, 32), dtype=bool)
    # 圆环：无邻居数=1 的端点 → _find_endpoints 返回空 → 原分支
    ys, xs = [], []
    for t in np.linspace(0, 2 * np.pi, 60)[:-1]:
        ys.append(int(round(16 + 8 * np.sin(t))))
        xs.append(int(round(16 + 8 * np.cos(t))))
    skeleton[ys, xs] = True
    keypoints, _ = sim._extract_keypoints(skeleton.astype(np.uint8) * 255)
    assert 1 <= len(keypoints) <= sim.max_keypoints, f"K={len(keypoints)}"


def test_v5027_good_path_unchanged():
    """好样本路径（端点同域、BFS 成功）不进回退分支——K 与路径形态正常"""
    sim = _make_sim(seed=424250)
    skeleton = np.zeros((64, 64), dtype=bool)
    skeleton[30, 5:50] = True         # 单连通域直线，2 端点
    keypoints, _ = sim._extract_keypoints(skeleton.astype(np.uint8) * 255)
    assert sim.min_keypoints <= len(keypoints) <= sim.max_keypoints
    # 直线路径采样点应全部落在该直线上
    assert all(y == 30 for y, _ in keypoints), "采样点偏离直线路径"


def test_v5027_real_bad_seeds_bounded():
    """D' 全量实测坏样本（74~182 点）修复后重仿真 K ≤ 16"""
    bad = [424249, 424255, 424266, 424284]  # idx 7,13,24,42 → seed=424242+idx
    for seed in bad:
        sim = _make_sim(seed=seed)
        result = sim.simulate(verbose=False)
        kpts = result["crack_keypoints"]
        assert len(kpts) <= sim.max_keypoints, (
            f"seed={seed}: K={len(kpts)} 仍超上限"
        )


if __name__ == "__main__":
    import functools

    _print = functools.partial(print, flush=True)
    tests = [
        test_v5027_fragmented_skeleton_bounded_kpts,
        test_v5027_closed_loop_domain_all_points_fallback,
        test_v5027_good_path_unchanged,
        test_v5027_real_bad_seeds_bounded,
    ]
    n_pass = 0
    for t in tests:
        try:
            t()
            n_pass += 1
            _print(f"[PASS] {t.__name__}")
        except AssertionError as e:
            _print(f"[FAIL] {t.__name__}: {e}")
    _print(f"\n{n_pass}/{len(tests)} passed")
    sys.exit(0 if n_pass == len(tests) else 1)
