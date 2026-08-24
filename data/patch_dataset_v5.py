"""
v5-α patch 数据集（V5A5）

将 PatchSimulator 包装为 PyTorch Dataset。

设计：
- 每个 __getitem__ 调用 PatchSimulator.simulate() 生成一个样本
- 样本包含：
  - x_1d: 温度时序（取 crack 时刻附近的 1D 温度均值）
  - x_2d: 3 通道图像（温度场 + 应力场 + 热力图）
  - true_bbox: (4,) 真值 bbox（[y1, x1, y2, x2] 归一化到 [0,1]）
  - true_keypoints: (K, 2) 真值关键点（K ∈ [min, max]）
  - true_crack_pixels: (N, 2) 真值裂纹像素（来自 crack_mask）

注：
- v5-α 不需要真实 3D 数据；仿真数据已足够训练算法骨架
- v5-β 启动后接入真实数据时，只需替换本 Dataset 即可，trainer 接口不变

详见：
- docs/v5_范围说明.md
- tasks/team_v5_alpha.json 中 V5A5
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from data.patch_simulator_v5 import PatchSimulator


def _normalize_field(field: np.ndarray) -> np.ndarray:
    """归一化字段到 [0, 1]"""
    f_min, f_max = field.min(), field.max()
    if f_max - f_min < 1e-9:
        return np.zeros_like(field, dtype=np.float32)
    return ((field - f_min) / (f_max - f_min)).astype(np.float32)


class PatchDatasetV5(Dataset):
    """
    v5-α patch 数据集

    每个样本由 PatchSimulator.simulate() 生成。
    默认在 __init__ 时**预计算全部样本**到内存（v5-α MVP 简化）。
    真实数据接入后（v5-β）应改为懒加载 + 磁盘缓存。

    Args:
        patch_size:  patch 像素尺寸（默认 256）
        n_samples:   数据集样本数（默认 100）
        min_kpts:    关键点下界
        max_kpts:    关键点上界
        seq_len:     1D 序列长度（取 crack 时刻附近窗口均值）
        crack_stress_threshold_MPa: 裂纹阈值
        seed:        基础随机种子（每个样本用 seed + idx）
        thermal_profile: 三段曲线配置
        material:    材料参数
        return_pixels: 是否返回 crack 像素（用于覆盖距离 loss）
        precompute:   是否在 __init__ 时预计算（默认 True）
        verbose:      是否打印预计算进度
    """

    def __init__(
        self,
        patch_size: int = 256,
        n_samples: int = 100,
        min_kpts: int = 8,
        max_kpts: int = 16,
        seq_len: int = 300,
        crack_stress_threshold_MPa: float = 50.0,
        seed: int = 42,
        thermal_profile: dict | None = None,
        material: dict | None = None,
        return_pixels: bool = True,
        precompute: bool = True,
        verbose: bool = False,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.n_samples = n_samples
        self.min_kpts = min_kpts
        self.max_kpts = max_kpts
        self.seq_len = seq_len
        self.crack_threshold_MPa = crack_stress_threshold_MPa
        self.seed = seed
        self.thermal_profile = thermal_profile or {}
        self.material = material or {}
        self.return_pixels = return_pixels
        self.verbose = verbose

        # 物量单位（米）
        self.physical_size_cm = 5.0
        self.physical_size_m = self.physical_size_cm / 100.0

        # 预计算缓存
        self._cache: list[dict] | None = None
        if precompute:
            self._precompute()

    def _precompute(self) -> None:
        """预计算所有样本并缓存"""
        self._cache = []
        for idx in range(self.n_samples):
            if self.verbose:
                print(f"  [PatchDatasetV5] 预计算样本 {idx + 1}/{self.n_samples}")
            self._cache.append(self._generate_sample(idx))

    def __len__(self) -> int:
        return self.n_samples

    def _generate_sample(self, idx: int) -> dict:
        """生成单个样本（不依赖缓存）"""
        sim_seed = self.seed + idx
        sim = PatchSimulator(
            patch_size=self.patch_size,
            physical_size_cm=self.physical_size_cm,
            thermal_profile=self.thermal_profile,
            material=self.material,
            crack_stress_threshold_MPa=self.crack_threshold_MPa,
            min_keypoints=self.min_kpts,
            max_keypoints=self.max_kpts,
            seed=sim_seed,
        )
        result = sim.simulate(verbose=False)

        H = W = self.patch_size

        # 1D 时序：取 T_history 的最后 seq_len 个点的全局平均
        T_hist = result["temperature_field"]  # (T_samples, H, W)
        if len(T_hist) >= self.seq_len:
            T_seq = T_hist[-self.seq_len:].mean(axis=(1, 2))
        else:
            T_seq = T_hist.mean(axis=(1, 2))
            T_seq = np.pad(
                T_seq,
                (self.seq_len - len(T_seq), 0),
                mode="edge",
            )

        T_seq = _normalize_field(T_seq)

        # 2D 图像：3 通道
        T_final = T_hist[-1] if len(T_hist) > 0 else result["heatmap"]
        T_field_norm = _normalize_field(T_final)

        S_hist = result["stress_field"]
        S_final = S_hist[-1] if len(S_hist) > 0 else np.zeros((H, W))
        S_field_norm = _normalize_field(S_final)

        H_map = result["heatmap"].astype(np.float32)

        x_2d = np.stack([T_field_norm, S_field_norm, H_map], axis=0)

        # bbox：[x1, y1, x2, y2] 归一化
        bbox_yxyx = result["crack_bbox"].astype(np.float32).copy()
        bbox_yxyx[0::2] /= H
        bbox_yxyx[1::2] /= W
        bbox_xyxy = np.array([
            bbox_yxyx[1], bbox_yxyx[0], bbox_yxyx[3], bbox_yxyx[2],
        ], dtype=np.float32)

        # 关键点：(y, x) 归一化
        kpts = result["crack_keypoints"].astype(np.float32).copy()
        kpts[:, 0] /= H
        kpts[:, 1] /= W

        # 裂纹像素
        crack_mask = result["crack_mask"]
        if self.return_pixels:
            ys, xs = np.where(crack_mask > 0)
            if len(ys) > 0:
                true_pixels = np.stack([
                    ys.astype(np.float32) / H,
                    xs.astype(np.float32) / W,
                ], axis=1)
            else:
                true_pixels = np.zeros((0, 2), dtype=np.float32)
        else:
            true_pixels = np.zeros((0, 2), dtype=np.float32)

        return {
            "x_1d": torch.from_numpy(T_seq),
            "x_2d": torch.from_numpy(x_2d),
            "true_bbox": torch.from_numpy(bbox_xyxy),
            "true_keypoints": torch.from_numpy(kpts),
            "true_crack_pixels": torch.from_numpy(true_pixels),
            "metadata": result["metadata"],
        }

    def __getitem__(self, idx: int) -> dict:
        if self._cache is not None:
            return self._cache[idx]
        return self._generate_sample(idx)


def collate_v5(batch: list[dict]) -> dict:
    """
    v5-α 自定义 collate 函数

    处理变长数据（true_keypoints, true_crack_pixels 不同长度）：
    - 用 padding（-1 标记填充）保留 batch 维度
    - 用 mask 标记有效位置

    Returns:
        dict:
            x_1d:        (B, seq_len)
            x_2d:        (B, 3, H, W)
            true_bbox:   (B, 4)
            true_keypoints: (B, max_K, 2) — padding 到最大 K
            keypoint_mask: (B, max_K) — True = 有效
            true_crack_pixels: (B, max_N, 2) — padding 到最大 N
            pixel_mask:    (B, max_N) — True = 有效
    """
    B = len(batch)

    x_1d = torch.stack([b["x_1d"] for b in batch], dim=0)
    x_2d = torch.stack([b["x_2d"] for b in batch], dim=0)
    true_bbox = torch.stack([b["true_bbox"] for b in batch], dim=0)

    # 变长关键点 — padding
    Ks = [b["true_keypoints"].size(0) for b in batch]
    max_K = max(Ks)
    if max_K > 0:
        true_kpts = torch.full((B, max_K, 2), -1.0, dtype=torch.float32)
        kpt_mask = torch.zeros(B, max_K, dtype=torch.bool)
        for i, b in enumerate(batch):
            k = b["true_keypoints"].size(0)
            if k > 0:
                true_kpts[i, :k] = b["true_keypoints"]
                kpt_mask[i, :k] = True
    else:
        true_kpts = torch.zeros(B, 0, 2)
        kpt_mask = torch.zeros(B, 0, dtype=torch.bool)

    # 变长裂纹像素 — padding
    Ns = [b["true_crack_pixels"].size(0) for b in batch]
    max_N = max(Ns) if Ns else 0
    if max_N > 0:
        true_pixels = torch.full((B, max_N, 2), -1.0, dtype=torch.float32)
        pixel_mask = torch.zeros(B, max_N, dtype=torch.bool)
        for i, b in enumerate(batch):
            n = b["true_crack_pixels"].size(0)
            if n > 0:
                true_pixels[i, :n] = b["true_crack_pixels"]
                pixel_mask[i, :n] = True
    else:
        true_pixels = torch.zeros(B, 0, 2)
        pixel_mask = torch.zeros(B, 0, dtype=torch.bool)

    return {
        "x_1d": x_1d,
        "x_2d": x_2d,
        "true_bbox": true_bbox,
        "true_keypoints": true_kpts,
        "keypoint_mask": kpt_mask,
        "true_crack_pixels": true_pixels,
        "pixel_mask": pixel_mask,
    }


# ============================================================
#  冒烟测试
# ============================================================

def _smoke_test():
    print("=" * 60)
    print("PatchDatasetV5 冒烟测试（小尺寸 + 短周期加速）")
    print("=" * 60)

    # 用小 patch + 短热曲线，让 1 次仿真 < 5s
    # 默认：11.7h 周期 + 256x256 → 几十分钟
    # 测试用：1 min 周期 + 64x64 patch → < 1s
    fast_profile = {
        "ramp_up_c_per_min": 1260.0,    # 1 min ramp
        "soak_temp_c": 1280.0,
        "soak_duration_min": 1.0,
        "cool_down_c_per_min": 1260.0,  # 1 min cool
    }
    ds = PatchDatasetV5(
        patch_size=64,
        n_samples=2,
        seed=42,
        thermal_profile=fast_profile,
        verbose=True,
    )
    print(f"\n[OK] 数据集大小: {len(ds)}")

    sample = ds[0]
    print(f"  x_1d:        {tuple(sample['x_1d'].shape)}")
    print(f"  x_2d:        {tuple(sample['x_2d'].shape)}")
    print(f"  true_bbox:   {tuple(sample['true_bbox'].shape)}, values={sample['true_bbox'].tolist()}")
    print(f"  true_keypoints: {tuple(sample['true_keypoints'].shape)}")
    print(f"  true_crack_pixels: {tuple(sample['true_crack_pixels'].shape)}")
    print(f"  metadata:    peak={sample['metadata']['peak_stress_MPa']:.2f} MPa, "
          f"K={sample['metadata']['n_keypoints']}")

    # collate 测试
    batch = collate_v5([ds[0], ds[1]])
    print(f"\n[Collate]")
    print(f"  x_1d:        {tuple(batch['x_1d'].shape)}")
    print(f"  x_2d:        {tuple(batch['x_2d'].shape)}")
    print(f"  true_keypoints: {tuple(batch['true_keypoints'].shape)}, mask sum={batch['keypoint_mask'].sum().item()}")
    print(f"  true_crack_pixels: {tuple(batch['true_crack_pixels'].shape)}, mask sum={batch['pixel_mask'].sum().item()}")

    print("\n[ALL PASS] ✅")


if __name__ == "__main__":
    _smoke_test()