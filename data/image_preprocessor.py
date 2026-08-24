"""
PE-MMNet v4 图像预处理模块

提供图像预处理工具，包括：
- ImagePreprocessor: 标准图像预处理（灰度、归一化、裁剪、resize）
- ContourRemover: 等值线去除器（用于参数化扫描4）

用法:
    from data.image_preprocessor import ImagePreprocessor, ContourRemover

    # 标准预处理
    preprocessor = ImagePreprocessor(image_size=256, crop_ratio=0.7)
    tensor = preprocessor(image_pil)

    # 等值线去除（仅用于参数化扫描4）
    contour_remover = ContourRemover.create_for_batch4()
    processed_image = contour_remover(image_rgb)
"""

import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T


class ImagePreprocessor:
    """
    图像预处理器
    核心功能：
    1. 灰度转换
    2. Max-Min归一化（解决跨批次色标差异）
    3. Center Crop（聚焦物理区域）
    4. Resize到目标尺寸
    """

    def __init__(self, image_size=256, crop_ratio=0.70):
        """
        Args:
            image_size: 目标图像尺寸
            crop_ratio: 中心裁剪比例
        """
        self.image_size = image_size
        self.crop_ratio = crop_ratio

    def __call__(self, img):
        """
        预处理图像

        Args:
            img: PIL.Image对象

        Returns:
            torch.Tensor: (1, H, W)，值域[0, 1]
        """
        # 1. 转为灰度图
        img = img.convert('L')

        # 2. 转为numpy数组
        img_array = np.array(img, dtype=np.float32)

        # 3. Max-Min归一化（关键：解决跨批次色标差异）
        img_min = img_array.min()
        img_max = img_array.max()
        if img_max - img_min > 1e-8:
            img_norm = (img_array - img_min) / (img_max - img_min)
        else:
            img_norm = np.zeros_like(img_array)

        # 4. Center Crop（聚焦物理区域，去除边缘留白）
        h, w = img_norm.shape
        crop_h = int(h * self.crop_ratio)
        crop_w = int(w * self.crop_ratio)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        img_cropped = img_norm[start_h:start_h+crop_h, start_w:start_w+crop_w]

        # 5. Resize到目标尺寸
        img_pil = Image.fromarray((img_cropped * 255).astype(np.uint8))
        img_resized = img_pil.resize((self.image_size, self.image_size), Image.BILINEAR)

        # 6. 转为Tensor [0, 1]
        tensor = T.ToTensor()(img_resized)

        return tensor


class ContourRemover:
    """
    等值线去除器（用于参数化扫描4）

    基于参考图片分析：
    - 需裁剪掉左侧负坐标、顶部标题、右侧图例
    - 保留中心物理区域 (0~10mm × 0~10mm)
    - 使用 OpenCV Inpainting 修复等值线覆盖区域

    处理流程（关键顺序！）：
    Step 1: 精确裁剪（去除负坐标、标题、图例）
    Step 2: 检测并修复等值线和数值标签
    Step 3: 调整到标准尺寸

    参考图片分析结果：
    | 干扰类型 | 位置 | 严重程度 |
    |---------|------|---------|
    | 左侧负坐标空白 | X < 0 | 完全无用 |
    | 底部空白 | Y < 0 | 完全无用 |
    | 顶部标题文字 | Y ≈ 上方15% | 完全无用 |
    | 右侧双图例栏 | X ≈ 右方25% | 完全无用 |
    | 黑色等值线 | 覆盖裂纹区域 | 需修复 |
    | 数值标签 | 裂纹峰值区 | 需修复 |
    """

    def __init__(self,
                 crop_left=0.08,      # 左侧裁剪比例（去除负坐标+标题）
                 crop_right=0.75,     # 右侧裁剪比例（保留75%，去除图例）
                 crop_top=0.12,       # 顶部裁剪比例（去除标题）
                 crop_bottom=0.0,     # 底部裁剪（通常无需裁剪）
                 black_threshold=30,  # 黑色像素阈值
                 inpaint_radius=3,
                 inpaint_method='telea'):
        """
        Args:
            crop_left: 左侧裁剪起点（0~1，相对于图像宽度）
            crop_right: 右侧裁剪终点（0~1）
            crop_top: 顶部裁剪起点
            crop_bottom: 底部裁剪终点
            black_threshold: 黑色像素检测阈值（0-255）
            inpaint_radius: inpainting 修复半径
            inpaint_method: 'telea' 或 'ns'（Navier-Stokes）
        """
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom
        self.black_threshold = black_threshold
        self.inpaint_radius = inpaint_radius
        self.inpaint_method = (cv2.INPAINT_TELEA if inpaint_method == 'telea'
                               else cv2.INPAINT_NS)

    def __call__(self, image):
        """
        处理单张图像

        Args:
            image: numpy array, RGB 格式 (H, W, 3)

        Returns:
            处理后的图像，RGB 格式
        """
        h, w = image.shape[:2]

        # Step 1: 精确裁剪（基于参考图片分析）
        # - 左侧 8%：去除负坐标区域和Y轴刻度
        # - 右侧 25%：去除双图例栏
        # - 顶部 12%：去除标题文字
        x1 = int(w * self.crop_left)
        x2 = int(w * self.crop_right)
        y1 = int(h * self.crop_top)
        y2 = int(h * (1 - self.crop_bottom))

        cropped = image[y1:y2, x1:x2]

        # Step 2: 检测并修复等值线
        repaired = self._inpaint_contours(cropped)

        # Step 3: 调整到标准尺寸（256x256）
        resized = cv2.resize(repaired, (256, 256), interpolation=cv2.INTER_LINEAR)

        return resized

    def _inpaint_contours(self, image):
        """
        检测并修复黑色等值线和数值标签

        Args:
            image: numpy array, RGB 格式

        Returns:
            修复后的图像，RGB 格式
        """
        # 转换为灰度用于掩膜生成
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # 检测黑色/极暗像素（等值线和文字都是黑色）
        # 阈值 30：覆盖大部分黑色等值线（RGB ≈ [0,0,0]）
        _, black_mask = cv2.threshold(
            gray, self.black_threshold, 255, cv2.THRESH_BINARY_INV
        )

        # 轻微膨胀，确保细线被完全覆盖
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(black_mask, kernel, iterations=1)

        # 形态学闭操作：连接断裂的等值线，避免修复不连续
        closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        # 排除图像边缘的纯黑区域（可能是裁剪后的边界）
        h, w = closed.shape
        border_mask = np.zeros_like(closed)
        border_mask[5:h-5, 5:w-5] = 255  # 留 5px 边界
        final_mask = cv2.bitwise_and(closed, border_mask)

        # Inpainting 修复
        # 方法：TELEA 基于周围像素的加权平均，速度快质量好
        # 注意：cv2.inpaint 需要 BGR 格式
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        repaired_bgr = cv2.inpaint(
            bgr,
            final_mask,
            inpaintRadius=self.inpaint_radius,
            flags=self.inpaint_method
        )
        repaired = cv2.cvtColor(repaired_bgr, cv2.COLOR_BGR2RGB)

        return repaired

    @staticmethod
    def create_for_batch4():
        """
        工厂方法：创建针对参数化扫描4的默认配置

        Returns:
            ContourRemover: 配置好的实例
        """
        return ContourRemover(
            crop_left=0.08,
            crop_right=0.75,
            crop_top=0.12,
            crop_bottom=0.0,
            black_threshold=30,
            inpaint_radius=3,
            inpaint_method='telea'
        )


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    print("测试 ContourRemover...")

    # 创建测试图像（模拟有等值线的图像）
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image.fill(200)  # 灰色背景

    # 添加一些黑色线条（模拟等值线）
    cv2.line(test_image, (100, 50), (100, 400), (0, 0, 0), 2)
    cv2.line(test_image, (200, 50), (200, 400), (0, 0, 0), 2)
    cv2.line(test_image, (300, 50), (300, 400), (0, 0, 0), 2)

    # 测试 ContourRemover
    remover = ContourRemover.create_for_batch4()
    processed = remover(test_image)

    print(f"原始尺寸: {test_image.shape}")
    print(f"处理后尺寸: {processed.shape}")
    print("✓ ContourRemover 测试通过")
