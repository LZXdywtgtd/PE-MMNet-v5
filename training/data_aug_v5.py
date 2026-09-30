"""
v5-α 数据增强（V5A5）

轻量数据增强，物理安全（不破坏物理场）：
- 随机水平/垂直翻转
- 随机 90° 旋转
- 高斯噪声（仅用于训练正则化）
- bbox / 关键点 / 像素坐标同步变换

设计原则：
- 温度场 + 应力场 + 热力图 必须保持物理一致性
- bbox 和 mask 坐标必须随图像同步变换

详见：
- tasks/team_v5_alpha.json 中 V5A5
"""

from __future__ import annotations

import random
import numpy as np
import torch


def _mirror_y(kp: np.ndarray) -> np.ndarray:
    """y 坐标翻转（0 -> 1）"""
    out = kp.copy()
    if out.shape[0] > 0:
        out[:, 0] = 1.0 - out[:, 0]
    return out


def _mirror_x(kp: np.ndarray) -> np.ndarray:
    """x 坐标翻转"""
    out = kp.copy()
    if out.shape[0] > 0:
        out[:, 1] = 1.0 - out[:, 1]
    return out


class PatchAugmentorV5:
    """
    v5-α patch 数据增强器

    Args:
        hflip_prob:  水平翻转概率（默认 0.5）
        vflip_prob:  垂直翻转概率（默认 0.5）
        rot90_prob:  90° 旋转概率（默认 0.5）
        noise_prob:  高斯噪声概率（默认 0.3）
        noise_std:   高斯噪声标准差（默认 0.02）
    """

    def __init__(
        self,
        hflip_prob: float = 0.5,
        vflip_prob: float = 0.5,
        rot90_prob: float = 0.5,
        noise_prob: float = 0.3,
        noise_std: float = 0.02,
    ):
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.rot90_prob = rot90_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std

    def __call__(
        self,
        x_1d: torch.Tensor,
        x_2d: torch.Tensor,
        true_bbox: torch.Tensor,
        true_keypoints: torch.Tensor,
        true_crack_pixels: torch.Tensor,
    ) -> tuple:
        """
        应用随机增强

        Args:
            x_1d: (seq_len,)
            x_2d: (3, H, W)
            true_bbox: (4,) — [x1, y1, x2, y2]
            true_keypoints: (K, 2) — (y, x)
            true_crack_pixels: (N, 2) — (y, x)

        Returns:
            同形状的 5 个张量
        """
        kpts = true_keypoints.numpy()
        pixels = true_crack_pixels.numpy()
        bbox = true_bbox.numpy().copy()

        # 随机水平翻转
        if random.random() < self.hflip_prob:
            x_2d = torch.flip(x_2d, dims=[-1])  # 沿 W 翻转
            kpts = _mirror_x(kpts)
            pixels = _mirror_x(pixels)
            # bbox: x1 ↔ x2
            bbox = bbox.copy()
            bbox[0], bbox[2] = 1.0 - bbox[2], 1.0 - bbox[0]

        # 随机垂直翻转
        if random.random() < self.vflip_prob:
            x_2d = torch.flip(x_2d, dims=[-2])  # 沿 H 翻转
            kpts = _mirror_y(kpts)
            pixels = _mirror_y(pixels)
            # bbox: y1 ↔ y2
            bbox[1], bbox[3] = 1.0 - bbox[3], 1.0 - bbox[1]

        # 随机 90° 旋转（顺时针）
        if random.random() < self.rot90_prob:
            x_2d = torch.rot90(x_2d, k=-1, dims=[-2, -1])
            # 图像顺时针 90°（实测标记点）：(y, x) → (x, 1-y)
            if kpts.shape[0] > 0:
                kpts_new = np.stack([kpts[:, 1], 1.0 - kpts[:, 0]], axis=1)
                kpts = kpts_new
            if pixels.shape[0] > 0:
                pixels_new = np.stack([pixels[:, 1], 1.0 - pixels[:, 0]], axis=1)
                pixels = pixels_new
            # bbox 旋转: [x1, y1, x2, y2] → [1-y2, x1, 1-y1, x2]
            x1, y1, x2, y2 = bbox
            new_bbox = np.array([1.0 - y2, x1, 1.0 - y1, x2], dtype=np.float32)
            bbox = new_bbox
            # 排序：确保 x1 ≤ x2, y1 ≤ y2
            bbox[0], bbox[2] = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
            bbox[1], bbox[3] = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])

        # 高斯噪声（仅 2D 图像）
        if random.random() < self.noise_prob:
            noise = torch.randn_like(x_2d) * self.noise_std
            x_2d = (x_2d + noise).clamp(0.0, 1.0)

        true_bbox = torch.from_numpy(bbox.copy())
        true_keypoints = torch.from_numpy(kpts.copy())
        true_crack_pixels = torch.from_numpy(pixels.copy())

        return x_1d, x_2d, true_bbox, true_keypoints, true_crack_pixels


# ============================================================
#  冒烟测试
# ============================================================

def _smoke_test():
    print("=" * 60)
    print("PatchAugmentorV5 冒烟测试")
    print("=" * 60)

    aug = PatchAugmentorV5(seed=42) if False else PatchAugmentorV5()
    random.seed(42)

    # 构造测试数据
    x_1d = torch.linspace(0, 1, 300)
    x_2d = torch.rand(3, 64, 64)
    bbox = torch.tensor([0.2, 0.3, 0.7, 0.8])  # [x1, y1, x2, y2]
    kpts = torch.tensor([
        [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8],
    ])
    pixels = torch.tensor([
        [0.1, 0.15], [0.2, 0.25], [0.3, 0.35],
    ])

    # 增强 100 次，验证形状 + 物理约束
    for i in range(100):
        x_1d_a, x_2d_a, bbox_a, kpts_a, pixels_a = aug(
            x_1d.clone(), x_2d.clone(),
            bbox.clone(), kpts.clone(), pixels.clone(),
        )
        assert x_1d_a.shape == (300,)
        assert x_2d_a.shape == (3, 64, 64)
        assert bbox_a.shape == (4,)
        assert kpts_a.shape == (4, 2)
        assert pixels_a.shape == (3, 2)
        # bbox 约束
        assert bbox_a[0] <= bbox_a[2] + 1e-5, f"iter {i}: x1={bbox_a[0]} > x2={bbox_a[2]}"
        assert bbox_a[1] <= bbox_a[3] + 1e-5, f"iter {i}: y1={bbox_a[1]} > y2={bbox_a[3]}"
        # 值域
        assert (kpts_a >= 0).all() and (kpts_a <= 1).all(), \
            f"iter {i}: kpts out of [0,1]"
        assert (pixels_a >= 0).all() and (pixels_a <= 1).all(), \
            f"iter {i}: pixels out of [0,1]"
        assert (x_2d_a >= 0).all() and (x_2d_a <= 1.0 + 1e-5).all(), \
            f"iter {i}: x_2d out of [0,1]"

    print(f"  [OK] 100 次增强：形状正确，bbox 顺序约束成立，值域 [0,1]")
    print(f"\n[ALL PASS] ✅")


if __name__ == "__main__":
    _smoke_test()