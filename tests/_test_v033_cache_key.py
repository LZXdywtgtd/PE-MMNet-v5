# -*- coding: utf-8 -*-
"""V5-033 P0 验证：缓存键数据变体标记（dp/sd + gs/og）。

验证项：
T1 新键命中——dprime 数据集加载 dpgs 键缓存，GT 与旧 _dp 文件逐位一致
T2 dprime 键 miss → FileNotFoundError（拒绝 stock 顶替）
T3 dprime + 无 cache_dir → ValueError
T4 stock 路径正常（键带 sdgs 标记，临时目录现算 + 二次命中）
T5 og 标记（gt_frame_strip=False）生成独立键
"""
import sys, os, functools
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
print = functools.partial(print, flush=True)
import shutil
import tempfile
import numpy as np

from data.patch_dataset_v5 import (
    PatchDatasetV5, _cache_filename, _thermal_profile_hash,
)

FAST = {
    "ramp_up_c_per_min": 1260.0, "soak_temp_c": 1280.0,
    "soak_duration_min": 1.0, "cool_down_c_per_min": 1260.0,
}
CACHE = os.path.join(PROJECT_ROOT, "logs", "sim_cache")
th_hash = _thermal_profile_hash(FAST)

# ── T1: 新键命中 + 与旧 _dp 文件逐位一致 ──
exp_train = _cache_filename(128, 1000, 424242, th_hash, 60.0,
                            data_variant="dp", gt_frame_strip=True)
exp_val = _cache_filename(128, 200, 434242, th_hash, 60.0,
                          data_variant="dp", gt_frame_strip=True)
print(f"T1 期望训练键: {exp_train}")
print(f"T1 期望验证键: {exp_val}")
assert os.path.exists(os.path.join(CACHE, exp_train)), "新训练键文件缺失"
assert os.path.exists(os.path.join(CACHE, exp_val)), "新验证键文件缺失"

old_train = os.path.join(CACHE, "sim_cache_p128_n1000_s424242_t6eb66e1b_th60.0_dp.npz")
old_val = os.path.join(CACHE, "sim_cache_p128_n200_s434242_t6eb66e1b_th60.0_dp.npz")

ds = PatchDatasetV5(patch_size=128, n_samples=200, seed=434242,
                    thermal_profile=FAST, crack_stress_threshold_MPa=60.0,
                    cache_dir=CACHE, data_variant="dprime", precompute=True,
                    verbose=True)
d_new = np.load(os.path.join(CACHE, exp_val))
assert np.array_equal(ds._cache[0]["x_2d"].numpy(), d_new["x_2d"][0])
d_old = np.load(old_val)
assert np.array_equal(d_new["x_2d"][0], d_old["x_2d"][0]), "新键与旧 _dp 内容不一致"
d_new.close(); d_old.close()
# 逐位对比训练键（大文件，抽 3 个样本）
d_new = np.load(os.path.join(CACHE, exp_train))
d_old = np.load(old_train)
for i in [0, 500, 999]:
    assert np.array_equal(d_new["x_2d"][i], d_old["x_2d"][i]), f"train idx{i} 不一致"
d_new.close(); d_old.close()
print("T1 PASS：新 dpgs 键命中，内容与旧 _dp 逐位一致")

# ── T2: dprime miss → FileNotFoundError ──
tmp = tempfile.mkdtemp(prefix="_v033_t2_")
try:
    try:
        PatchDatasetV5(patch_size=64, n_samples=2, seed=42,
                       thermal_profile=FAST, cache_dir=tmp,
                       data_variant="dprime", precompute=False)
        raise AssertionError("T2 应报 FileNotFoundError")
    except FileNotFoundError as e:
        print(f"T2 PASS：miss 报错 → {str(e)[:80]}...")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ── T3: dprime + 无 cache_dir → ValueError ──
try:
    PatchDatasetV5(patch_size=64, n_samples=2, seed=42,
                   thermal_profile=FAST, data_variant="dprime",
                   precompute=False)
    raise AssertionError("T3 应报 ValueError")
except ValueError as e:
    print(f"T3 PASS：无缓存目录报错 → {str(e)[:60]}...")

# ── T4: stock 现算 + 键带 sdgs + 二次命中 ──
tmp = tempfile.mkdtemp(prefix="_v033_t4_")
try:
    ds1 = PatchDatasetV5(patch_size=64, n_samples=1, seed=42,
                         thermal_profile=FAST, cache_dir=tmp,
                         precompute=True, verbose=True)
    files = os.listdir(tmp)
    assert len(files) == 1 and "sdgs" in files[0], f"stock 键应含 sdgs: {files}"
    ds2 = PatchDatasetV5(patch_size=64, n_samples=1, seed=42,
                         thermal_profile=FAST, cache_dir=tmp,
                         precompute=True)
    assert np.array_equal(ds1[0]["x_2d"].numpy(), ds2[0]["x_2d"].numpy())
    print(f"T4 PASS：stock 键 {files[0]} 现算+命中一致")
finally:
    shutil.rmtree(tmp, ignore_errors=True)

# ── T5: og 标记独立键 ──
f_gs = _cache_filename(64, 2, 42, th_hash, 5.0, data_variant="sd",
                       gt_frame_strip=True)
f_og = _cache_filename(64, 2, 42, th_hash, 5.0, data_variant="sd",
                       gt_frame_strip=False)
assert f_gs != f_og and "sdgs" in f_gs and "sdog" in f_og
print(f"T5 PASS：gs/og 独立键（{f_gs} vs {f_og}）")

print("\n[ALL PASS] V5-033 P0 缓存键验证 5/5")
