"""
PE-MMNet v4: 数据增强模块

提供多种数据增强方法，包括：
1. ThermalCutMix: 物理安全的CutMix增强（仅作用于温度通道）
2. 其他基础增强方法

注意：
- ThermalCutMix 默认关闭（prob=0.0），用户需显式启用
- 物理安全版仅混合温度通道，应力通道保持不变
- 这符合热弹性力学规律，避免破坏应力场的物理边界条件

使用方法：
```python
from training.augmentation import ThermalCutMix

# 创建增强器（默认关闭）
cutmix = ThermalCutMix(alpha=1.0, prob=0.0)

# 在训练循环中使用
img1, img2 = batch1[1], batch2[1]  # 2D图像 (2, H, W)
labels1, labels2 = batch1[2], batch2[2]

# 以50%概率启用CutMix
cutmix.prob = 0.5
img_aug, _, labels_aug, _ = cutmix(img1, img2, labels1, labels2)
```
"""

import random
import torch
import torch.nn.functional as F


class ThermalCutMix:
    """
    Thermal CutMix 增强（物理安全版）

    与标准CutMix不同，本实现仅在温度通道上进行混合，
    应力通道保持不变，以符合热弹性力学规律。

    原理：
    - 裂纹形成与温度变化密切相关（热应力）
    - 温度场的混合是物理合理的（热传导可以叠加）
    - 应力场混合会破坏物理边界条件（如支撑点、约束）

    适用场景：
    - 训练数据有限时的数据增强
    - 提高模型对温度变化的鲁棒性
    """

    def __init__(self, alpha=1.0, prob=0.0):
        """
        Args:
            alpha: Beta分布参数，用于生成混合比例lambda
            prob: 启用概率（默认0.0，即默认关闭）
        """
        self.alpha = alpha
        self.prob = prob

    def __call__(self, img1, img2, labels1, labels2):
        """
        执行CutMix增强

        Args:
            img1: 图像1 (2, H, W) 或 (H, W) - 通道0=温度，通道1=应力
            img2: 图像2 (2, H, W) 或 (H, W)
            labels1: 标签1 (6,) - [x, y, l, w, conf, density]
            labels2: 标签2 (6,)

        Returns:
            img_mixed: 混合后的图像
            img2: 原始图像2（保持不变）
            labels_mixed: 插值后的标签
            labels2: 原始标签2（保持不变）
        """
        # 决定是否执行CutMix
        if random.random() > self.prob:
            return img1, img2, labels1, labels2

        # 生成混合比例lambda（从Beta分布采样）
        lambda_ = random.betavariate(self.alpha, self.alpha)

        # 转换为Tensor（如果是numpy）
        if not isinstance(img1, torch.Tensor):
            img1 = torch.from_numpy(img1).float()
        if not isinstance(img2, torch.Tensor):
            img2 = torch.from_numpy(img2).float()

        # 确保是2D图像 (2, H, W)
        if img1.dim() == 2:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        # 仅在温度通道上混合（物理安全）
        # 温度通道：img[0] 或 img[..., 0, :, :]
        img_mixed = img1.clone()

        if img1.size(0) >= 1:
            # 温度通道混合
            temp_mask = self._generate_mask(img1[0])
            img_mixed[0] = img1[0] * temp_mask + img2[0] * (1 - temp_mask)

        # 应力通道保持不变（物理安全约束）

        # 标签插值
        labels_mixed = lambda_ * labels1 + (1 - lambda_) * labels2

        return img_mixed, img2, labels_mixed, labels2

    def _generate_mask(self, temp_map):
        """
        根据温度图生成混合掩码

        使用高温区域作为mask，在高温区域保留img1，
        在低温区域使用img2。这样可以保留高温区域的
        裂纹特征，同时引入低温区域的变化。

        Args:
            temp_map: 温度图 (H, W)

        Returns:
            mask: (H, W) 0-1掩码
        """
        # 计算阈值：均值 + 标准差
        threshold = temp_map.mean() + temp_map.std()

        # 生成二值掩码
        mask = (temp_map > threshold).float()

        # 如果掩码全0或全1，使用随机掩码
        if mask.sum() == 0 or mask.sum() == mask.numel():
            # BUG FIX: 原来引用了未定义的 lambda_，改为使用随机数
            mask = (torch.rand_like(temp_map) > 0.5).float()

        return mask


class RandomNoise:
    """
    随机噪声增强

    添加高斯噪声或椒盐噪声来提高模型鲁棒性。
    """

    def __init__(self, noise_level=0.01, prob=0.5):
        """
        Args:
            noise_level: 噪声水平（相对于[0,1]范围的百分比）
            prob: 启用概率
        """
        self.noise_level = noise_level
        self.prob = prob

    def __call__(self, img):
        """
        Args:
            img: (C, H, W) 或 (H, W)

        Returns:
            加噪后的图像
        """
        if random.random() > self.prob:
            return img

        noise = torch.randn_like(img) * self.noise_level
        return torch.clamp(img + noise, 0, 1)


class RandomFlip:
    """
    随机翻转增强

    支持水平翻转和垂直翻转。
    """

    def __init__(self, h_prob=0.5, v_prob=0.0):
        """
        Args:
            h_prob: 水平翻转概率
            v_prob: 垂直翻转概率
        """
        self.h_prob = h_prob
        self.v_prob = v_prob

    def __call__(self, img, labels=None):
        """
        Args:
            img: (C, H, W)
            labels: 可选，标签 (6,)

        Returns:
            翻转后的图像和标签
        """
        if random.random() < self.h_prob:
            img = torch.flip(img, dims=[-1])  # 水平翻转
            if labels is not None:
                labels[0] = 1 - labels[0]  # x坐标翻转

        if random.random() < self.v_prob:
            img = torch.flip(img, dims=[-2])  # 垂直翻转
            if labels is not None:
                labels[1] = 1 - labels[1]  # y坐标翻转

        return img, labels


class Compose:
    """组合多个增强操作"""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, labels=None):
        for t in self.transforms:
            if labels is not None:
                img, labels = t(img, labels)
            else:
                img = t(img) if callable(getattr(t, '__call____')) else img
        return img, labels


def test_thermal_cutmix():
    """测试ThermalCutMix"""
    print("Testing ThermalCutMix...")

    # 创建增强器（默认关闭）
    cutmix = ThermalCutMix(alpha=1.0, prob=0.0)
    img1 = torch.rand(2, 64, 64)
    img2 = torch.rand(2, 64, 64)
    labels1 = torch.tensor([0.5, 0.5, 0.1, 0.1, 0.9, 0.3])
    labels2 = torch.tensor([0.6, 0.6, 0.2, 0.2, 0.8, 0.4])

    # prob=0 时不执行混合
    img_out, _, labels_out, _ = cutmix(img1, img2, labels1, labels2)
    assert torch.allclose(img_out, img1), "prob=0时应该不改变图像"
    assert torch.allclose(labels_out, labels1), "prob=0时应该不改变标签"
    print("  prob=0: No change OK")

    # prob=1 时执行混合
    cutmix.prob = 1.0
    img_out, _, labels_out, _ = cutmix(img1.clone(), img2.clone(), labels1.clone(), labels2.clone())

    # 检查温度通道是否被混合
    assert not torch.allclose(img_out[0], img1[0]), "温度通道应该被混合"
    # 检查应力通道是否保持不变
    assert torch.allclose(img_out[1], img1[1]), "应力通道应该保持不变"
    print("  prob=1: Temperature mixed, stress preserved OK")

    # 检查标签是否插值
    assert not torch.allclose(labels_out, labels1), "标签应该被插值"
    assert not torch.allclose(labels_out, labels2), "标签应该被插值"
    print("  Labels interpolated OK")

    print("All augmentation tests passed!")


if __name__ == "__main__":
    test_thermal_cutmix()
