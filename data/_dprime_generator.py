# -*- coding: utf-8 -*-
"""fast+D' 全量生成适配层（2026-09-29，用户拍板 fast 口径管线验证）

V5-025 诊断结论（_exp_v5025_cause.py）：默认曲线下 D' 调制消失主因是
热扩散（空间低通，11.7h 扩散长度 52px ≥ 调制段宽 21-42px，对比度
9.21x → 1.12x），非 Maxwell 松弛（逐像素时间滤波只缩幅保形）。
fast 曲线（11min，扩散 7px ≪ 段宽）是 D' 唯一路径——本模块为
generate_cache_v5.py 提供 D' 调制仿真器。

设计（零改动原库，规避回归风险）：
- DPrimePatchSimulator 子类化 PatchSimulator，仅重写 _apply_robin_boundary：
  对流项乘每边 h 空间调制（辐射不动），与 _exp_boundary_dprime.py 已验证
  的手动循环逐字等价（同一 smoothstep seg_hmap、同一四边循环顺序）。
- simulate() 主循环/历史采样/GT 提取/元数据全部继承，样本打包逻辑
  复刻 PatchDatasetV5._generate_sample（含 seq_len=300 截尾、归一化、
  kpts/像素归一化）。
- D' 抽取与全部实验脚本一致：default_rng(seed) → edge,c∈U(0.3,0.7),
  seg∈U(1/3,2/3), boost∈{2,3,4}。

位级安全网（--selftest）：
① 抽样比对：本模块 sample(idx)（带调制）的 x_1d/x_2d 张量 shape/dtype
   与 stock 一致；GT 非空率统计（th=60 口径下预期 ≥95%）
② seed 4200-4204 的 σ_max 场 vs _exp_sigma_fields 缓存逐位一致
   （同 seed 同 spike → 物理必须复现）③ seed 4205 无调制模式 vs
   stock simulate 逐位一致（子类不调制时必须退化为原行为）。
"""
from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from data.patch_simulator_v5 import PatchSimulator

FAST_PROFILE = {
    "ramp_up_c_per_min": 1260.0,
    "soak_temp_c": 1280.0,
    "soak_duration_min": 1.0,
    "cool_down_c_per_min": 1260.0,
}


def spikes_for_seed(seed: int):
    """D' 配置抽取——与 _exp_boundary_dprime.py 逐字一致。"""
    rng = np.random.default_rng(seed)
    edge = str(rng.choice(["top", "bottom", "left", "right"]))
    c = float(rng.uniform(0.3, 0.7))
    seg = float(rng.uniform(1 / 3, 2 / 3))
    boost = float(rng.choice([2.0, 3.0, 4.0]))
    return [(edge, c, seg, boost)]


def seg_hmap(axis_vals, c, seg_len, boost, ramp=0.04):
    """1+(boost-1)·窗：[c-seg/2, c+seg/2] 平台，smoothstep 过渡（实验同款）"""
    u = np.clip(axis_vals, 0.0, 1.0)
    aa, bb = c - seg_len / 2, c + seg_len / 2
    s1 = np.clip((u - aa) / ramp, 0, 1); s1 = s1 * s1 * (3 - 2 * s1)
    s2 = np.clip((u - bb) / ramp, 0, 1); s2 = s2 * s2 * (3 - 2 * s2)
    return 1.0 + (boost - 1.0) * s1 * (1 - s2)


class DPrimePatchSimulator(PatchSimulator):
    """边界 h 空间调制的 patch 仿真器（D' 路线，fast 口径）。

    spikes=None 时完全不调制（逐位退化为 PatchSimulator）。
    """

    def __init__(self, *args, spikes=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dprime_spikes = spikes or []

    def _hmaps(self):
        H = W = self.patch_size
        tcoord = np.arange(W) / W
        ycoord = np.arange(H) / H
        hmaps = {"top": None, "bottom": None, "left": None, "right": None}
        for edge, c, seg, boost in self.dprime_spikes:
            if edge in ("top", "bottom"):
                hmaps[edge] = seg_hmap(tcoord, c, seg, boost)
            else:
                hmaps[edge] = seg_hmap(ycoord, c, seg, boost)
        for k in hmaps:
            if hmaps[k] is None:
                hmaps[k] = np.ones(W if k in ("top", "bottom") else H)
        return hmaps

    def _apply_robin_boundary(self, T_new, T_old, T_target):
        if not self.dprime_spikes:
            return super()._apply_robin_boundary(T_new, T_old, T_target)
        dt = self.dt
        rho_cp = self.rho * self.cp
        L = self.dx / 2.0
        T_env_c = T_target
        T_env_K = self.radiation.celsius_to_kelvin(T_env_c)
        hmaps = self._hmaps()
        for name, edge_slice in [
            ("top", (0, slice(None))),
            ("bottom", (-1, slice(None))),
            ("left", (slice(None), 0)),
            ("right", (slice(None), -1)),
        ]:
            T_edge = T_old[edge_slice]
            T_edge_K = self.radiation.celsius_to_kelvin(T_edge)
            q = self.h_conv * hmaps[name] * (T_env_c - T_edge) \
                + self.radiation.heat_flux(T_edge_K, T_env_K)
            T_new[edge_slice] = T_edge + dt * q / (rho_cp * L)
        return T_new


def generate_dprime_sample(
    idx: int, seed_base: int, patch_size: int = 128,
    threshold_MPa: float = 60.0, thermal_profile: dict | None = None,
    seq_len: int = 300, modulate: bool = True,
):
    """生成单个 D' 样本，返回与 PatchDatasetV5._generate_sample 同构的 dict。

    modulate=False 时关闭调制（安全网③用）。
    """
    import torch

    from data.patch_dataset_v5 import _normalize_field

    sim_seed = seed_base + idx
    spikes = spikes_for_seed(sim_seed) if modulate else []
    sim = DPrimePatchSimulator(
        patch_size=patch_size,
        thermal_profile=thermal_profile or FAST_PROFILE,
        crack_stress_threshold_MPa=threshold_MPa,
        seed=sim_seed,
        spikes=spikes,
    )
    result = sim.simulate(verbose=False)

    H = W = patch_size
    T_hist = result["temperature_field"]
    if len(T_hist) >= seq_len:
        T_seq = T_hist[-seq_len:].mean(axis=(1, 2))
    else:
        T_seq = T_hist.mean(axis=(1, 2))
        T_seq = np.pad(T_seq, (seq_len - len(T_seq), 0), mode="edge")
    T_seq = _normalize_field(T_seq)

    T_final = T_hist[-1] if len(T_hist) > 0 else result["heatmap"]
    T_field_norm = _normalize_field(T_final)
    S_hist = result["stress_field"]
    S_final = S_hist[-1] if len(S_hist) > 0 else np.zeros((H, W))
    S_field_norm = _normalize_field(S_final)
    H_map = result["heatmap"].astype(np.float32)
    x_2d = np.stack([T_field_norm, S_field_norm, H_map], axis=0)

    bbox_xyxy = result["crack_bbox"].astype(np.float32).copy()
    kpts = result["crack_keypoints"].astype(np.float32).copy()
    kpts[:, 0] /= H
    kpts[:, 1] /= W
    crack_mask = result["crack_mask"]
    ys, xs = np.where(crack_mask > 0)
    if len(ys) > 0:
        true_pixels = np.stack(
            [ys.astype(np.float32) / H, xs.astype(np.float32) / W], axis=1)
    else:
        true_pixels = np.zeros((0, 2), dtype=np.float32)

    return {
        "x_1d": torch.from_numpy(T_seq),
        "x_2d": torch.from_numpy(x_2d),
        "true_bbox": torch.from_numpy(bbox_xyxy),
        "true_keypoints": torch.from_numpy(kpts),
        "true_crack_pixels": torch.from_numpy(true_pixels),
        "metadata": result["metadata"],
        "sigma_max_field": sim._sigma_max_field,
        "spikes": spikes,
    }


def selftest(n_bits: int = 5) -> bool:
    """三道位级安全网。全部通过返回 True。"""
    import functools
    _print = functools.partial(print, flush=True)
    ok = True

    # ① D' 样本 GT 非空率 + 张量形状/dtype
    _print("[selftest①] D' 样本 10 个（th=60, fast）：GT 非空 + 形状核对")
    n_pos = 0
    for i in range(10):
        s = generate_dprime_sample(i, seed_base=4200, threshold_MPa=60.0)
        k = s["true_keypoints"].shape[0]
        n_px = s["true_crack_pixels"].shape[0]
        n_pos += n_px > 0
        assert s["x_1d"].shape == (300,) and str(s["x_1d"].dtype) == "torch.float32", \
            f"x_1d {s['x_1d'].shape} {s['x_1d'].dtype}"
        assert s["x_2d"].shape == (3, 128, 128), s["x_2d"].shape
        assert 0 <= s["true_bbox"].min() and s["true_bbox"].max() <= 1
        if i < 3:
            _print(f"  idx{i} seed{4200+i} {s['spikes'][0][0]} "
                  f"peak={s['metadata']['peak_stress_MPa']:.1f}MPa "
                  f"kpts={k} px={n_px}")
    _print(f"  GT 非空率 {n_pos}/10（100 样本口径预期 ≥95%）")

    # ② σ 场逐位复现：与 _exp_sigma_fields 缓存对比
    _print("[selftest②] seed 4200-4204 σ_max 场 vs 实验缓存逐位对比")
    fld_dir = os.path.join(PROJECT_ROOT, "output", "exp_fields", "_exp_sigma_fields")
    for i in range(n_bits):
        seed = 4200 + i
        ref_path = os.path.join(fld_dir, f"seed_{seed}.npy")
        if not os.path.exists(ref_path):
            _print(f"  [SKIP] 缓存缺 seed_{seed}（不在本机）")
            continue
        s = generate_dprime_sample(i, seed_base=4200, threshold_MPa=60.0)
        ref = np.load(ref_path)
        same = np.array_equal(s["sigma_max_field"].astype(np.float32), ref)
        _print(f"  seed={seed}: {'[OK] 逐位一致' if same else '[FAIL] 不一致'}")
        ok &= same

    # ③ 无调制退化为 stock 行为
    _print("[selftest③] spikes=[] vs stock PatchSimulator.simulate 逐位对比")
    from data.patch_simulator_v5 import PatchSimulator as Stock
    for i in range(2):
        seed = 9000 + i
        d = DPrimePatchSimulator(
            patch_size=64, thermal_profile=FAST_PROFILE,
            crack_stress_threshold_MPa=60.0, seed=seed, spikes=[])
        r1 = d.simulate()
        st = Stock(patch_size=64, thermal_profile=FAST_PROFILE,
                   crack_stress_threshold_MPa=60.0, seed=seed)
        r2 = st.simulate()
        same = (np.array_equal(r1["crack_mask"], r2["crack_mask"])
                and np.array_equal(r1["heatmap"], r2["heatmap"])
                and np.array_equal(r1["crack_keypoints"], r2["crack_keypoints"]))
        _print(f"  seed={seed}: {'[OK] 逐位一致' if same else '[FAIL]'}")
        ok &= same

    return ok


if __name__ == "__main__":
    import functools
    print = functools.partial(print, flush=True)
    all_ok = selftest()
    print("\nselftest:", "ALL PASS" if all_ok else "FAILED")
    sys.exit(0 if all_ok else 1)
