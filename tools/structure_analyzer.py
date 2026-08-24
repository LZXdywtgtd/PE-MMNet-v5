"""
PE-MMNet v4 结构分析工具
用于分离和标注气孔（小圆孔）与裂纹（蜿蜒暗带）

核心算法：
1. 多级阈值分割：裂纹用低阈值+反向，气孔用高阈值
2. 连通域分析：根据面积和长宽比分离裂纹和气孔
3. 可视化：用不同颜色标注两种结构
"""

import numpy as np
from scipy import ndimage
import cv2
from dataclasses import dataclass
from typing import Tuple, List, Optional


@dataclass
class StructureStats:
    """结构统计信息"""
    # 裂纹统计
    crack_count: int
    crack_total_area: float
    crack_avg_area: float
    crack_max_length: float

    # 气孔统计
    pore_count: int
    pore_total_area: float
    pore_avg_area: float
    pore_avg_diameter: float


def separate_crack_pore(img: np.ndarray,
                        crack_threshold: float = 0.2,
                        pore_threshold: float = 0.7,
                        crack_min_area: int = 50,
                        crack_min_aspect_ratio: float = 2.0,
                        merge_distance: int = 10,
                        crack_mode: str = 'skeleton',
                        edge_exclusion: int = 25) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StructureStats]:
    """
    分离裂纹和气孔

    Args:
        img: 归一化后的 d_hist 图像 (H, W), 值域 [0, 1]
        crack_threshold: 裂纹阈值（低于此值为裂纹）
        pore_threshold: 气孔阈值（高于此值为气孔）
        crack_min_area: 裂纹最小面积（像素）
        crack_min_aspect_ratio: 裂纹最小长宽比
        merge_distance: 相邻裂纹碎片合并距离阈值（像素）
        crack_mode: 裂纹显示模式 - 'skeleton' 或 'solid'
        edge_exclusion: 边界排除区宽度（像素），接触此区域的连通域将被丢弃

    Returns:
        crack_mask: 裂纹掩膜 (H, W), 值域 [0, 1]
        pore_mask: 气孔掩膜 (H, W), 值域 [0, 1]
        combined_overlay: 叠加视图 (H, W, 3), RGB格式
        stats: 统计信息
    """
    h, w = img.shape

    # ============ 创建边界排除掩膜 ============
    # 边缘区域标记为 True（需要排除）
    edge_mask = np.zeros((h, w), dtype=bool)
    if edge_exclusion > 0:
        edge_mask[:edge_exclusion, :] = True       # 上边
        edge_mask[-edge_exclusion:, :] = True      # 下边
        edge_mask[:, :edge_exclusion] = True      # 左边
        edge_mask[:, -edge_exclusion:] = True      # 右边

    def touches_boundary(region_mask: np.ndarray) -> bool:
        """检查连通域是否接触边界"""
        # 膨胀连通域边界检查
        dilated = ndimage.binary_dilation(region_mask, iterations=edge_exclusion)
        return np.any(dilated & edge_mask)

    def is_boundary_artifact(region_mask: np.ndarray) -> bool:
        """检查是否是边界伪影（太靠近边缘且扁平）"""
        if not touches_boundary(region_mask):
            return False

        # 检查是否是水平或垂直的长条（边界伪影特征）
        y_coords, x_coords = np.where(region_mask)
        if len(x_coords) < 10:
            return True

        x_range = x_coords.max() - x_coords.min()
        y_range = y_coords.max() - y_coords.min()

        # 如果水平范围远大于垂直范围，很可能是边界伪影
        if y_range > 0 and x_range / y_range > 5:
            return True
        if x_range > 0 and y_range / x_range > 5:
            return True

        return False

    # ============ 裂纹提取 ============
    # 裂纹在 d_hist 中是低灰度区域（暗带），需要反向阈值
    crack_binary = (img < crack_threshold).astype(np.uint8)

    # 第一步：形态学膨胀 + 合并相邻碎片
    if merge_distance > 0:
        # 使用较小的核进行膨胀（避免过度合并）
        kernel_size = max(3, merge_distance // 3)
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        crack_dilated = cv2.dilate(crack_binary, kernel, iterations=1)
        crack_merged = crack_dilated
    else:
        crack_merged = crack_binary

    # 连通域标记（在合并后的图像上）
    crack_labeled, crack_num = ndimage.label(crack_merged)

    # ============ 过滤边界伪影并统计有效裂纹 ============
    valid_crack_count = 0
    valid_crack_total_area = 0
    valid_crack_max_length = 0
    valid_crack_mask = np.zeros_like(img)

    for i in range(1, crack_num + 1):
        region = crack_labeled == i

        # 跳过边界伪影
        if is_boundary_artifact(region):
            continue

        area = region.sum()
        if area < crack_min_area:
            continue

        # 计算长宽比
        y_coords, x_coords = np.where(region)
        if len(x_coords) < 2:
            continue

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        width = x_max - x_min + 1
        height = y_max - y_min + 1
        aspect_ratio = max(width, height) / (min(width, height) + 1e-6)

        if aspect_ratio < crack_min_aspect_ratio:
            continue

        valid_crack_count += 1
        valid_crack_total_area += area
        valid_crack_max_length = max(valid_crack_max_length, max(width, height))

        # 始终填充 valid_crack_mask（过滤后的有效裂纹）
        valid_crack_mask[region] = 1.0

    # ============ 根据模式生成最终裂纹掩膜 ============
    crack_mask = valid_crack_mask
    crack_total_area = valid_crack_total_area
    crack_max_length = valid_crack_max_length

    # 更新裂纹数量为有效裂纹数
    crack_num = valid_crack_count

    # ============ 气孔提取（简化版）============
    # 气孔在 d_hist 中是高灰度区域（亮点）
    # 不再预设圆度/大小过滤，只用亮度阈值
    pore_binary = (img > pore_threshold).astype(np.uint8)

    # 连通域标记
    pore_labeled, pore_num = ndimage.label(pore_binary)

    pore_mask = np.zeros_like(img)
    pore_total_area = 0
    pore_count_valid = 0

    for i in range(1, pore_num + 1):
        region = pore_labeled == i
        area = region.sum()

        if area < 5:
            continue  # 太小了，可能是噪声

        # 简单判定：只要亮度高于阈值、面积足够大，就是气孔
        pore_mask[region] = 1.0
        pore_total_area += area
        pore_count_valid += 1

    # ============ 生成叠加视图 ============
    # RGB 格式: (H, W, 3), 值域 [0, 1]
    overlay = np.zeros((h, w, 3), dtype=np.float32)

    # 基础灰度背景（使用 viridis 风格的灰度）
    overlay[..., 0] = img * 0.35  # R
    overlay[..., 1] = img * 0.35  # G
    overlay[..., 2] = img * 0.35  # B

    # 裂纹叠加（红色半透明）
    crack_bool = crack_mask > 0.5
    overlay[crack_bool] = [1.0, 0.15, 0.15]  # 亮红色

    # 气孔叠加（蓝色）
    pore_bool = pore_mask > 0.5
    overlap = crack_bool & pore_bool
    crack_only = crack_bool & ~pore_bool
    pore_only = pore_bool & ~crack_bool

    overlay[pore_only] = [0.15, 0.15, 1.0]  # 蓝色
    overlay[overlap] = [1.0, 1.0, 0.2]  # 黄色（裂纹和气孔重叠）

    # ============ 统计信息 ============
    # crack_num 已经是合并后的连通域数量（物理裂纹数）
    stats = StructureStats(
        crack_count=int(crack_num),
        crack_total_area=float(crack_total_area),
        crack_avg_area=float(crack_total_area / max(crack_num, 1)),
        crack_max_length=float(crack_max_length),
        pore_count=int(pore_count_valid),
        pore_total_area=float(pore_total_area),
        pore_avg_area=float(pore_total_area / max(pore_count_valid, 1)),
        pore_avg_diameter=float(2 * np.sqrt(pore_total_area / max(pore_count_valid, 1) / np.pi)) if pore_count_valid > 0 else 0.0
    )

    return crack_mask, pore_mask, overlay, stats


def auto_optimize_params(img: np.ndarray,
                         min_crack_length: float = 30.0) -> dict:
    """
    自动寻优：搜索最佳参数组合以提取最长裂纹

    Args:
        img: 归一化后的 d_hist 图像 (H, W)
        min_crack_length: 最小裂纹长度阈值，低于此值判定为无裂纹

    Returns:
        dict: 最佳参数组合和结果统计
    """
    # 定义搜索空间（与手动滑块范围一致）
    crack_thresholds = np.arange(0.10, 0.42, 0.02)  # 0.10 ~ 0.40
    merge_distances = [5, 8, 10, 12, 15, 18, 20]  # 5 ~ 20（与滑块一致）
    min_areas = [10, 20, 30, 50, 80]  # 10 ~ 80（与滑块一致）
    edge_exclusions = [15, 20, 25, 30]  # 15 ~ 30（与滑块一致）

    best_params = None
    best_length = 0
    best_result = None
    total_combinations = (len(crack_thresholds) * len(merge_distances) *
                        len(min_areas) * len(edge_exclusions))

    count = 0
    for crack_thresh in crack_thresholds:
        for merge_dist in merge_distances:
            for min_area in min_areas:
                for edge_excl in edge_exclusions:
                    count += 1

                    # 调用分离函数
                    crack_mask, _, _, stats = separate_crack_pore(
                        img,
                        crack_threshold=crack_thresh,
                        pore_threshold=0.85,  # 固定气孔阈值
                        crack_min_area=min_area,
                        merge_distance=merge_dist,
                        crack_mode='skeleton',
                        edge_exclusion=edge_excl
                    )

                    # 获取最长裂纹长度
                    max_length = stats.crack_max_length

                    # 更新最优解
                    if max_length > best_length:
                        best_length = max_length
                        best_params = {
                            'crack_threshold': crack_thresh,
                            'merge_distance': merge_dist,
                            'crack_min_area': min_area,
                            'edge_exclusion': edge_excl,  # 最佳边界排除距离
                        }
                        best_result = stats

    # 判断是否找到有效裂纹
    if best_length < min_crack_length:
        return {
            'success': False,
            'message': f"未发现明显裂纹特征（最长长度: {best_length:.1f}px < {min_crack_length}px）",
            'best_params': None,
            'best_stats': None
        }

    return {
        'success': True,
        'message': f"找到最佳参数！最长裂纹: {best_length:.1f}px",
        'best_params': best_params,
        'best_stats': best_result,
        'total_combinations': total_combinations
    }


def analyze_structure(img: np.ndarray,
                     crack_threshold: float = 0.2,
                     pore_threshold: float = 0.7) -> dict:
    """
    快速结构分析（不生成可视化）

    Args:
        img: 归一化后的 d_hist 图像
        crack_threshold: 裂纹阈值
        pore_threshold: 气孔阈值

    Returns:
        dict: 包含各阈值下的像素比例
    """
    below_crack = (img < crack_threshold).mean() * 100
    above_pore = (img > pore_threshold).mean() * 100

    return {
        'crack_ratio': below_crack,
        'pore_ratio': above_pore,
        'neutral_ratio': 100 - below_crack - above_pore
    }


# ============ 示例代码 ============
if __name__ == '__main__':
    # 示例用法
    from data.dataset_multimodal import imread_unicode
    from PIL import Image

    # 读取图像
    img_path = r'D:\Desktop\team_project\simulation\参考输入\单次扫描\d_hist\d_hist01.png'
    img = imread_unicode(img_path)
    img_array = np.array(img.convert('L'), dtype=np.float32) / 255.0

    # 分离分析
    crack_mask, pore_mask, overlay, stats = separate_crack_pore(
        img_array,
        crack_threshold=0.2,
        pore_threshold=0.7
    )

    print("=" * 50)
    print("结构分析结果")
    print("=" * 50)
    print(f"裂纹数量: {stats.crack_count}")
    print(f"裂纹总面积: {stats.crack_total_area:.2f} 像素")
    print(f"裂纹平均面积: {stats.crack_avg_area:.2f} 像素")
    print(f"裂纹最大长度: {stats.crack_max_length:.2f} 像素")
    print("-" * 50)
    print(f"气孔数量: {stats.pore_count}")
    print(f"气孔总面积: {stats.pore_total_area:.2f} 像素")
    print(f"气孔平均面积: {stats.pore_avg_area:.2f} 像素")
    print(f"气孔平均直径: {stats.pore_avg_diameter:.2f} 像素")
    print("=" * 50)

    # 保存叠加图
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(img_array, cmap='gray')
    axes[0].set_title('Original d_hist')
    axes[0].axis('off')

    axes[1].imshow(crack_mask, cmap='Reds')
    axes[1].set_title(f'Cracks ({stats.crack_count})')
    axes[1].axis('off')

    axes[2].imshow(pore_mask, cmap='Blues')
    axes[2].set_title(f'Pores ({stats.pore_count})')
    axes[2].axis('off')

    axes[3].imshow(overlay)
    axes[3].set_title('Overlay (R=Crack, B=Pore)')
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig('output/structure_analysis.png', dpi=150)
    print("\n已保存到 output/structure_analysis.png")
