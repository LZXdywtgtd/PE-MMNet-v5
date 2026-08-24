"""
PE-MMNet v4 数据与模型可视化工具

功能：
1. 预处理检查（--check-preprocess）：验证 d_hist → 二值掩膜转换效果
2. 等值线去除效果（--show-contour-removal）：对比原始图、去等值线图、二值掩膜
3. 单样本对比（--compare）：对比温度场、应力场、真实掩膜、预测掩膜、叠层
4. 批量对比（--batch-compare）：批量生成多张样本的对比图
5. 模型推理：加载检查点进行预测可视化

用法:
    # 预处理检查
    python tools/visualization.py --check-preprocess --d_hist "单次扫描/d/d001.png" --threshold 0.1

    # 等值线去除效果
    python tools/visualization.py --check-preprocess \
        --d_hist "参数化扫描4/d/d001.png" --show-contour-removal

    # 单样本对比
    python tools/visualization.py --compare --batch "参数化扫描1" --index 5

    # 带预测的对比
    python tools/visualization.py --compare --batch "参数化扫描1" --index 5 \
        --checkpoint ./checkpoints/resnet18_cnn_attn_cross_attn_tasksegmentation_offset0_best.pt

    # 批量对比
    python tools/visualization.py --batch-compare --batch "参数化扫描1" --start 0 --end 20 \
        --output output/visualization/
"""

import argparse
import os
import sys
import torch
import matplotlib
matplotlib.use('Agg')  # 非交互式后端，避免 plt.show() 阻塞
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import warnings

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_multimodal import (
    MaskLabelProcessor, ImagePreprocessor, imread_unicode,
    load_csv_safely, parse_label_csv, parse_probe_csv,
    interpolate_seq, process_seq, compute_lw_from_density
)
from data.image_preprocessor import ContourRemover
from models.pe_tsnet_multimodal import PETSNetMultimodal
from utils.config import get_data_root, get_data_batches, ensure_config

# 常量
IMAGE_SIZE = 256
VIS_IMAGE_SIZE = 512  # 可视化专用高分辨率
SEQ_LEN = 300
CROP_RATIO = 0.70

# 配置 matplotlib 中文字体
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception:
    pass

# 忽略 matplotlib 警告
warnings.filterwarnings('ignore')

# 可用的色图选项
COLORMAPS = {
    'gray': 'Grayscale (Original)',
    'viridis': 'Viridis (Scientific)',
    'jet': 'Jet (Rainbow)',
    'hot': 'Hot (Thermal)',
    'inferno': 'Inferno (Dark)',
    'plasma': 'Plasma (Bright)',
    'magma': 'Magma (Red-based)',
    'coolwarm': 'Cool-Warm (Diverging)',
}

# =============================================================================
# 核心函数
# =============================================================================

def load_model_from_checkpoint(checkpoint_path, task_hint=None):
    """
    从检查点加载模型用于推理

    Args:
        checkpoint_path: 检查点文件路径
        task_hint: 任务类型提示（可选，从检查点文件名推断）

    Returns:
        tuple: (model, task) - 模型实例和实际任务类型
    """
    import torch

    # 1. 优先从检查点 metadata 读取 task
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if isinstance(checkpoint, dict):
        if 'task' in checkpoint:
            task = checkpoint['task']
        elif 'config' in checkpoint and 'task' in checkpoint['config']:
            task = checkpoint['config']['task']
        else:
            # 2. 从文件名推断
            task = task_hint or 'segmentation'
            if 'taskdetection' in checkpoint_path:
                task = 'detection'
            elif 'taskmultitask' in checkpoint_path:
                task = 'multitask'
            elif 'tasksegmentation' in checkpoint_path:
                task = 'segmentation'
    else:
        task = task_hint or 'segmentation'

    # 3. 创建模型
    model = PETSNetMultimodal(
        seq_len=SEQ_LEN,
        image_channels=2,
        pretrained_2d=True,
        dropout=0.2,
        task=task
    )

    # 4. 加载权重
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    model.eval()

    return model, task


def overlay_images(img1, img2, mode='red_transparent'):
    """
    叠层可视化

    Args:
        img1: 基础图像 (H, W) 或 (H, W, 3)
        img2: 叠加掩膜 (H, W)
        mode: 'red_transparent' 或 'red_blue'

    Returns:
        np.ndarray: 叠层图像 (H, W, 3)
    """
    # 确保 img1 是 RGB 格式
    if len(img1.shape) == 2:
        img1 = np.stack([img1] * 3, axis=-1)
    elif img1.shape[-1] == 1:
        img1 = np.concatenate([img1] * 3, axis=-1)

    overlay = img1.copy()

    if mode == 'red_transparent':
        # 透明红色覆盖
        mask_idx = img2 > 0.5
        overlay[mask_idx, 0] = np.minimum(overlay[mask_idx, 0] + 0.4, 1.0)  # R
        overlay[mask_idx, 1] = np.maximum(overlay[mask_idx, 1] - 0.2, 0.0)  # G
        overlay[mask_idx, 2] = np.maximum(overlay[mask_idx, 2] - 0.2, 0.0)  # B

    elif mode == 'red_blue':
        # 红蓝通道叠加（用于对比真实与预测掩膜）
        result = np.zeros_like(overlay)
        mask1 = img2[..., 0] > 0.5  # 真实掩膜
        mask2 = img2[..., 1] > 0.5  # 预测掩膜

        result[mask1 & ~mask2, 0] = 1.0   # 红色 = 仅真实
        result[mask1 & ~mask2, 1] = 0.0
        result[mask1 & ~mask2, 2] = 0.0

        result[~mask1 & mask2, 0] = 0.0   # 蓝色 = 仅预测
        result[~mask1 & mask2, 1] = 0.0
        result[~mask1 & mask2, 2] = 1.0

        result[mask1 & mask2, 0] = 0.0    # 绿色 = 交集
        result[mask1 & mask2, 1] = 1.0
        result[mask1 & mask2, 2] = 0.0

        overlay = result

    return overlay


def check_preprocess(d_hist_path, threshold=0.1, invert=True,
                     show_contour_removal=False, save_path=None,
                     high_res=True, cmap='viridis'):
    """
    预处理检查：显示原始 d_hist 与处理后的掩膜

    Args:
        d_hist_path: d_hist 图像路径
        threshold: 二值化阈值
        invert: 是否反转
        show_contour_removal: 是否显示等值线去除效果
        save_path: 保存路径（可选）
        high_res: 是否使用高分辨率 (512x512)
        cmap: 原始图像色图 ('gray', 'viridis', 'jet', 'hot', etc.)
    """
    print(f"[预处理检查] 图像: {d_hist_path}")
    print(f"  阈值: {threshold}, 反转: {invert}, 等值线去除: {show_contour_removal}")
    print(f"  分辨率: {'512x512 (高分辨率)' if high_res else '256x256'}, 色图: {cmap}")

    # 读取原始图像
    orig_img = imread_unicode(d_hist_path)
    if orig_img is None:
        print(f"  错误: 无法读取图像 {d_hist_path}")
        return

    # 根据分辨率决定处理方式
    if high_res:
        # 高分辨率处理：直接读取并 resize
        orig_array = np.array(orig_img.convert('L'), dtype=np.float32)

        # Center Crop 70%
        h, w = orig_array.shape
        crop_h = int(h * CROP_RATIO)
        crop_w = int(w * CROP_RATIO)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        cropped = orig_array[start_h:start_h+crop_h, start_w:start_w+crop_w]

        # Max-Min 归一化
        img_min = cropped.min()
        img_max = cropped.max()
        if img_max - img_min > 1e-8:
            cropped = (cropped - img_min) / (img_max - img_min)

        # Resize 到高分辨率
        orig_resized = Image.fromarray((cropped * 255).astype(np.uint8)).resize(
            (VIS_IMAGE_SIZE, VIS_IMAGE_SIZE), Image.BILINEAR)
        orig_array = np.array(orig_resized, dtype=np.float32) / 255.0

        # 高分辨率掩膜
        mask_highres = Image.fromarray((cropped * 255).astype(np.uint8)).resize(
            (VIS_IMAGE_SIZE, VIS_IMAGE_SIZE), Image.NEAREST)
        mask_np = np.array(mask_highres, dtype=np.float32) / 255.0
        if invert:
            mask_np = 1.0 - mask_np
        mask_array = (mask_np > threshold).astype(np.float32)
    else:
        # 原有 256x256 处理
        orig_array = np.array(orig_img.convert('L'), dtype=np.float32) / 255.0
        mask_proc = MaskLabelProcessor(binary_threshold=threshold, invert=invert)
        mask_tensor = mask_proc(d_hist_path)
        mask_array = mask_tensor.squeeze().numpy()

    # 2. 等值线去除（如果需要）
    contour_cleaned = None
    if show_contour_removal:
        print("  应用等值线去除...")
        contour_remover = ContourRemover.create_for_batch4()
        contour_img = np.array(orig_img)
        contour_cleaned = contour_remover(contour_img)
        # 转为灰度并归一化
        if len(contour_cleaned.shape) == 3:
            contour_cleaned = np.array(Image.fromarray(contour_cleaned).convert('L'))
        contour_cleaned = contour_cleaned.astype(np.float32) / 255.0

    # 绘制图像（使用高分辨率和彩色色图）
    fig_size = (18, 6) if show_contour_removal else (14, 6)

    if show_contour_removal:
        fig, axes = plt.subplots(1, 3, figsize=fig_size, dpi=100)

        # 原始图像（彩色色图）
        im0 = axes[0].imshow(orig_array, cmap=cmap, interpolation='bilinear')
        axes[0].set_title('Original d_hist', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], shrink=0.8, label='Intensity')

        # 等值线去除后
        im1 = axes[1].imshow(contour_cleaned, cmap=cmap, interpolation='bilinear')
        axes[1].set_title('After Contour Removal', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], shrink=0.8, label='Intensity')

        # 二值掩膜
        im2 = axes[2].imshow(mask_array, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title(f'Binary Mask (threshold={threshold})', fontsize=14, fontweight='bold')
        axes[2].axis('off')

        # 添加裂纹像素统计
        crack_ratio = mask_array.mean() * 100
        axes[2].text(0.5, -0.1, f'Crack ratio: {crack_ratio:.2f}%',
                     ha='center', transform=axes[2].transAxes, fontsize=11)
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)

        # 原始图像（彩色色图）
        im0 = axes[0].imshow(orig_array, cmap=cmap, interpolation='bilinear')
        axes[0].set_title('Original d_hist', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        plt.colorbar(im0, ax=axes[0], shrink=0.8, label='Intensity')

        # 二值掩膜
        im1 = axes[1].imshow(mask_array, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f'Binary Mask (threshold={threshold})', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        crack_ratio = mask_array.mean() * 100
        axes[1].text(0.5, -0.1, f'Crack ratio: {crack_ratio:.2f}%',
                     ha='center', transform=axes[1].transAxes, fontsize=11)

        # 叠层可视化（仅高分辨率模式下显示，避免尺寸不匹配）
        if high_res and orig_array.shape == mask_array.shape:
            overlay = np.zeros((*orig_array.shape, 3))
            gray = orig_array
            overlay[..., 0] = gray  # R
            overlay[..., 1] = gray * 0.8  # G
            overlay[..., 2] = gray * 0.6  # B

            # 掩膜区域标记为红色
            mask_bool = mask_array > 0.5
            overlay[mask_bool] = [1.0, 0.2, 0.2]

            im2 = axes[2].imshow(overlay)
            axes[2].set_title('Overlay (Red=Crack)', fontsize=14, fontweight='bold')
            axes[2].axis('off')
        else:
            axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  已保存到: {save_path}")
    else:
        output_path = 'output/preprocess.png'
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  已保存到: {output_path}")

    plt.close()


def get_batch_info(data_root):
    """
    获取批次的样本信息

    Args:
        data_root: 批次根目录

    Returns:
        dict: 包含批次信息的字典
    """
    batch_name = os.path.basename(os.path.normpath(data_root))
    table_dir = os.path.join(data_root, '表格')

    # 动态查找标签CSV
    label_candidates = [
        '参数化扫描-表面最大值.csv',
        'Table_表面最大值.csv',
        'Table_Crack.csv',
        'Table_Hist.csv',
    ]
    label_csv_path = None
    for name in label_candidates:
        path = os.path.join(table_dir, name)
        if os.path.exists(path):
            label_csv_path = path
            break

    if label_csv_path is None:
        return None

    # 加载标签
    label_df = load_csv_safely(label_csv_path)
    label_data = parse_label_csv(label_df)
    n_samples = len(label_data['time'])

    return {
        'name': batch_name,
        'n_samples': n_samples,
        'temp_dir': os.path.join(data_root, '温度'),
        'stress_dir': os.path.join(data_root, '应力'),
        'd_hist_dir': os.path.join(data_root, 'd_hist'),
        'has_d_hist': os.path.exists(os.path.join(data_root, 'd_hist')),
        'label_data': label_data
    }


def load_sample_images(batch_info, index):
    """
    加载样本的图像数据

    Args:
        batch_info: 批次信息字典
        index: 样本索引

    Returns:
        dict: 包含温度场、应力场、掩膜的字典
    """
    preprocessor = ImagePreprocessor(image_size=IMAGE_SIZE, crop_ratio=CROP_RATIO)
    mask_processor = MaskLabelProcessor(image_size=IMAGE_SIZE, crop_ratio=CROP_RATIO)

    # 获取文件列表
    temp_images = sorted([f for f in os.listdir(batch_info['temp_dir'])
                         if f.endswith('.png')], key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0])) or '0'))
    stress_images = sorted([f for f in os.listdir(batch_info['stress_dir'])
                            if f.endswith('.png')], key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0])) or '0'))

    if batch_info['has_d_hist']:
        d_hist_images = sorted([f for f in os.listdir(batch_info['d_hist_dir'])
                                if f.endswith('.png')], key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0])) or '0'))
    else:
        d_hist_images = []

    # 限制索引范围
    actual_n = min(len(temp_images), len(stress_images), len(d_hist_images) if d_hist_images else float('inf'),
                   batch_info['n_samples'])
    index = min(index, actual_n - 1)

    # 读取温度场
    temp_img = imread_unicode(os.path.join(batch_info['temp_dir'], temp_images[index]))
    temp_tensor = preprocessor(temp_img)
    temp_array = temp_tensor.squeeze().numpy()

    # 读取应力场
    stress_img = imread_unicode(os.path.join(batch_info['stress_dir'], stress_images[index]))
    stress_tensor = preprocessor(stress_img)
    stress_array = stress_tensor.squeeze().numpy()

    # 读取真实掩膜（如果有）
    true_mask = None
    if batch_info['has_d_hist'] and index < len(d_hist_images):
        mask_tensor = mask_processor(os.path.join(batch_info['d_hist_dir'], d_hist_images[index]))
        true_mask = mask_tensor.squeeze().numpy()

    return {
        'temp': temp_array,
        'stress': stress_array,
        'true_mask': true_mask,
        'index': index,
        'filename': temp_images[index] if index < len(temp_images) else f'未知_{index}'
    }


def predict_mask(model, batch_info, index):
    """
    使用模型预测掩膜

    Args:
        model: 已加载的模型
        batch_info: 批次信息
        index: 样本索引

    Returns:
        np.ndarray: 预测的掩膜 (H, W)
    """
    import torch

    # 加载探针数据用于 1D 序列
    table_dir = os.path.join(os.path.dirname(batch_info['temp_dir']), '表格')
    probe_candidates = [
        '参数化扫描-所有探针.csv',
        'Table_Temp.csv',
    ]
    probe_csv_path = None
    for name in probe_candidates:
        path = os.path.join(table_dir, name)
        if os.path.exists(path):
            probe_csv_path = path
            break

    if probe_csv_path is None:
        return None

    # 加载探针数据
    probe_df = load_csv_safely(probe_csv_path)
    probe_data = parse_probe_csv(probe_df)

    # 按 h_ceramic 和 Time 排序
    sort_idx = np.lexsort((probe_data['time'], probe_data['h_ceramic']))
    probe_h = probe_data['h_ceramic'][sort_idx]
    probe_time = probe_data['time'][sort_idx]
    probe_temp = probe_data['temp_values'][sort_idx]

    # 获取唯一 h 值
    unique_h = sorted(set(probe_h))

    # 获取样本对应的 h_ceramic
    label_data = batch_info['label_data']
    sort_idx2 = np.lexsort((label_data['time'], label_data['h_ceramic']))
    h_ceramic = label_data['h_ceramic'][sort_idx2]
    h_val = h_ceramic[index] if index < len(h_ceramic) else h_ceramic[0]

    # 建立插值函数
    if h_val not in unique_h:
        h_val = min(unique_h, key=lambda x: abs(x - h_val))

    mask = probe_h == h_val
    h_times = probe_time[mask]
    h_temps = probe_temp[mask]
    sort_idx3 = np.argsort(h_times)
    h_times = h_times[sort_idx3]
    h_temps = h_temps[sort_idx3]
    seq_300 = interpolate_seq(h_times, h_temps, SEQ_LEN)

    # 处理序列长度
    original_len = len(seq_300)
    if original_len > SEQ_LEN:
        seq_300 = process_seq(None, seq_300, SEQ_LEN, mode='pool')

    seq_1d = torch.from_numpy(seq_300).float().unsqueeze(0)  # (1, 300)

    # 加载 2D 图像
    preprocessor = ImagePreprocessor(image_size=IMAGE_SIZE, crop_ratio=CROP_RATIO)

    temp_images = sorted([f for f in os.listdir(batch_info['temp_dir'])
                         if f.endswith('.png')], key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0])) or '0'))
    stress_images = sorted([f for f in os.listdir(batch_info['stress_dir'])
                            if f.endswith('.png')], key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0])) or '0'))

    actual_n = min(len(temp_images), len(stress_images), batch_info['n_samples'])
    index = min(index, actual_n - 1)

    temp_img = imread_unicode(os.path.join(batch_info['temp_dir'], temp_images[index]))
    stress_img = imread_unicode(os.path.join(batch_info['stress_dir'], stress_images[index]))

    temp_tensor = preprocessor(temp_img)
    stress_tensor = preprocessor(stress_img)
    img_2d = torch.cat([temp_tensor, stress_tensor], dim=0).unsqueeze(0)  # (1, 2, 256, 256)

    # 推理
    with torch.no_grad():
        output = model(seq_1d, img_2d)

        if isinstance(output, tuple):
            # multitask: (mask, detection)
            pred_mask = output[0]
        else:
            # segmentation: mask
            pred_mask = output

        # 应用 sigmoid 并转为 numpy
        pred_mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()

    return pred_mask


def compare_sample(batch_name, index, checkpoint_path=None, task='segmentation', save_path=None):
    """
    单样本对比：温度场、应力场、真实掩膜、预测掩膜、叠层

    Args:
        batch_name: 批次名称
        index: 样本索引
        checkpoint_path: 检查点路径（可选）
        task: 任务类型
        save_path: 保存路径（可选）
    """
    print(f"[样本对比] 批次: {batch_name}, 索引: {index}")

    # 确保配置存在
    data_root = get_data_root()
    if not data_root:
        ensure_config()
        data_root = get_data_root()

    if not data_root:
        print("  错误: 无法获取数据根目录")
        return

    # 查找批次
    batch_path = os.path.join(data_root, batch_name)
    if not os.path.exists(batch_path):
        print(f"  错误: 批次不存在 {batch_path}")
        print(f"  可用批次: {os.listdir(data_root)}")
        return

    # 获取批次信息
    batch_info = get_batch_info(batch_path)
    if batch_info is None:
        print(f"  错误: 无法加载批次信息")
        return

    print(f"  样本数: {batch_info['n_samples']}")

    if index >= batch_info['n_samples']:
        print(f"  错误: 索引 {index} 超出范围 [0, {batch_info['n_samples']-1}]")
        return

    # 加载样本图像
    sample = load_sample_images(batch_info, index)
    print(f"  图像文件: {sample['filename']}")

    # 加载模型（如果提供检查点）
    model = None
    actual_task = task
    if checkpoint_path:
        print(f"  加载检查点: {checkpoint_path}")
        try:
            model, actual_task = load_model_from_checkpoint(checkpoint_path, task)
            print(f"  模型任务: {actual_task}")
        except Exception as e:
            print(f"  警告: 无法加载检查点: {e}")
            model = None

    # 预测掩膜（如果模型已加载）
    pred_mask = None
    if model is not None:
        print("  推理中...")
        pred_mask = predict_mask(model, batch_info, index)

    # 绘制对比图
    fig = None
    if actual_task == 'detection':
        # 检测模式：显示温度场、应力场、检测框
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))

        axes[0].imshow(sample['temp'], cmap='gray')
        axes[0].set_title('温度场')
        axes[0].axis('off')

        axes[1].imshow(sample['stress'], cmap='gray')
        axes[1].set_title('应力场')
        axes[1].axis('off')

    elif actual_task == 'segmentation' or (actual_task == 'detection' and sample['true_mask'] is None):
        # 分割模式：温度场、应力场、真实掩膜、预测掩膜、叠层
        n_rows = 2
        n_cols = 3

        if pred_mask is not None and sample['true_mask'] is not None:
            # 有真实掩膜和预测掩膜
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))

            axes[0, 0].imshow(sample['temp'], cmap='gray')
            axes[0, 0].set_title('温度场')
            axes[0, 0].axis('off')

            axes[0, 1].imshow(sample['stress'], cmap='gray')
            axes[0, 1].set_title('应力场')
            axes[0, 1].axis('off')

            axes[0, 2].imshow(sample['true_mask'], cmap='gray')
            axes[0, 2].set_title('真实掩膜')
            axes[0, 2].axis('off')

            # 第二行：预测掩膜和叠层
            axes[1, 0].imshow(pred_mask, cmap='gray')
            axes[1, 0].set_title('预测掩膜')
            axes[1, 0].axis('off')

            # 叠层1：温度场 + 预测掩膜
            overlay = overlay_images(sample['temp'], pred_mask, mode='red_transparent')
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title('温度场 + 预测掩膜')
            axes[1, 1].axis('off')

            # 叠层2：真实掩膜 + 预测掩膜（红蓝对比）
            combined = np.stack([sample['true_mask'], pred_mask], axis=-1)
            overlay2 = overlay_images(np.zeros_like(sample['true_mask']),
                                     combined, mode='red_blue')
            axes[1, 2].imshow(overlay2)
            axes[1, 2].set_title('真实(红) vs 预测(蓝)')
            axes[1, 2].axis('off')

        elif sample['true_mask'] is not None:
            # 只有真实掩膜
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(sample['temp'], cmap='gray')
            axes[0].set_title('温度场')
            axes[0].axis('off')

            axes[1].imshow(sample['stress'], cmap='gray')
            axes[1].set_title('应力场')
            axes[1].axis('off')

            axes[2].imshow(sample['true_mask'], cmap='gray')
            axes[2].set_title('真实掩膜')
            axes[2].axis('off')

        elif pred_mask is not None:
            # 只有预测掩膜
            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(sample['temp'], cmap='gray')
            axes[0].set_title('温度场')
            axes[0].axis('off')

            axes[1].imshow(sample['stress'], cmap='gray')
            axes[1].set_title('应力场')
            axes[1].axis('off')

            axes[2].imshow(pred_mask, cmap='gray')
            axes[2].set_title('预测掩膜')
            axes[2].axis('off')

        else:
            # 都没有
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))

            axes[0].imshow(sample['temp'], cmap='gray')
            axes[0].set_title('温度场')
            axes[0].axis('off')

            axes[1].imshow(sample['stress'], cmap='gray')
            axes[1].set_title('应力场')
            axes[1].axis('off')

    else:  # multitask
        # 多任务模式：与分割模式类似，但额外显示检测框
        n_rows = 3
        n_cols = 3

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 12))

        axes[0, 0].imshow(sample['temp'], cmap='gray')
        axes[0, 0].set_title('温度场')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(sample['stress'], cmap='gray')
        axes[0, 1].set_title('应力场')
        axes[0, 1].axis('off')

        if sample['true_mask'] is not None:
            axes[0, 2].imshow(sample['true_mask'], cmap='gray')
            axes[0, 2].set_title('真实掩膜')
            axes[0, 2].axis('off')
        else:
            axes[0, 2].axis('off')

        if pred_mask is not None:
            axes[1, 0].imshow(pred_mask, cmap='gray')
            axes[1, 0].set_title('预测掩膜')
            axes[1, 0].axis('off')

            # 叠层1：温度场 + 预测掩膜
            overlay = overlay_images(sample['temp'], pred_mask, mode='red_transparent')
            axes[1, 1].imshow(overlay)
            axes[1, 1].set_title('温度场 + 预测掩膜')
            axes[1, 1].axis('off')

            # 叠层2：真实 vs 预测
            if sample['true_mask'] is not None:
                combined = np.stack([sample['true_mask'], pred_mask], axis=-1)
                overlay2 = overlay_images(np.zeros_like(sample['true_mask']),
                                         combined, mode='red_blue')
                axes[1, 2].imshow(overlay2)
                axes[1, 2].set_title('真实(红) vs 预测(蓝)')
                axes[1, 2].axis('off')
            else:
                axes[1, 2].axis('off')
        else:
            axes[1, 0].axis('off')
            axes[1, 1].axis('off')
            axes[1, 2].axis('off')

        # 第三行：显示检测框信息（如果有）
        axes[2, 0].text(0.5, 0.5, f"批次: {batch_name}\n"
                        f"索引: {index}\n"
                        f"任务: multitask\n"
                        f"掩膜预测: {'是' if pred_mask is not None else '否'}",
                        ha='center', va='center', fontsize=12,
                        transform=axes[2, 0].transAxes)
        axes[2, 0].axis('off')
        axes[2, 1].axis('off')
        axes[2, 2].axis('off')

    plt.suptitle(f'{batch_name} - 样本 {index:04d}', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  已保存到: {save_path}")
    else:
        output_path = f'output/{batch_name}_{index:04d}_{task}_对比.png'
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  已保存到: {output_path}")

    plt.close()


def batch_compare(batch_name, start_idx, end_idx, checkpoint_path=None,
                  task='segmentation', output_dir='output/visualization'):
    """
    批量对比：批量生成多张样本的对比图

    Args:
        batch_name: 批次名称
        start_idx: 起始索引
        end_idx: 结束索引（不包含）
        checkpoint_path: 检查点路径（可选）
        task: 任务类型
        output_dir: 输出目录
    """
    print(f"[批量对比] 批次: {batch_name}, 范围: [{start_idx}, {end_idx})")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 确保配置存在
    data_root = get_data_root()
    if not data_root:
        ensure_config()
        data_root = get_data_root()

    if not data_root:
        print("  错误: 无法获取数据根目录")
        return

    # 查找批次
    batch_path = os.path.join(data_root, batch_name)
    if not os.path.exists(batch_path):
        print(f"  错误: 批次不存在 {batch_path}")
        return

    # 获取批次信息
    batch_info = get_batch_info(batch_path)
    if batch_info is None:
        print(f"  错误: 无法加载批次信息")
        return

    n_samples = batch_info['n_samples']

    # 加载模型（如果提供检查点）
    model = None
    actual_task = task
    if checkpoint_path:
        print(f"  加载检查点: {checkpoint_path}")
        try:
            model, actual_task = load_model_from_checkpoint(checkpoint_path, task)
            print(f"  模型任务: {actual_task}")
        except Exception as e:
            print(f"  警告: 无法加载检查点: {e}")
            model = None

    # 生成每张样本的对比图
    count = 0
    for i in range(start_idx, min(end_idx, n_samples)):
        save_path = os.path.join(output_dir, f'{batch_name}_{i:04d}_{task}_对比.png')
        print(f"  处理样本 {i}/{min(end_idx, n_samples) - 1}...")

        try:
            # 加载样本图像
            sample = load_sample_images(batch_info, i)

            # 预测掩膜（如果模型已加载）
            pred_mask = None
            if model is not None:
                pred_mask = predict_mask(model, batch_info, i)

            # 绘制对比图（简化版）
            if sample['true_mask'] is not None and pred_mask is not None:
                fig, axes = plt.subplots(2, 3, figsize=(12, 8))

                axes[0, 0].imshow(sample['temp'], cmap='gray')
                axes[0, 0].set_title('温度场')
                axes[0, 0].axis('off')

                axes[0, 1].imshow(sample['stress'], cmap='gray')
                axes[0, 1].set_title('应力场')
                axes[0, 1].axis('off')

                axes[0, 2].imshow(sample['true_mask'], cmap='gray')
                axes[0, 2].set_title('真实掩膜')
                axes[0, 2].axis('off')

                axes[1, 0].imshow(pred_mask, cmap='gray')
                axes[1, 0].set_title('预测掩膜')
                axes[1, 0].axis('off')

                overlay = overlay_images(sample['temp'], pred_mask, mode='red_transparent')
                axes[1, 1].imshow(overlay)
                axes[1, 1].set_title('温度场 + 预测掩膜')
                axes[1, 1].axis('off')

                combined = np.stack([sample['true_mask'], pred_mask], axis=-1)
                overlay2 = overlay_images(np.zeros_like(sample['true_mask']),
                                         combined, mode='red_blue')
                axes[1, 2].imshow(overlay2)
                axes[1, 2].set_title('真实(红) vs 预测(蓝)')
                axes[1, 2].axis('off')

            elif sample['true_mask'] is not None:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(sample['temp'], cmap='gray')
                axes[0].set_title('温度场')
                axes[0].axis('off')

                axes[1].imshow(sample['stress'], cmap='gray')
                axes[1].set_title('应力场')
                axes[1].axis('off')

                axes[2].imshow(sample['true_mask'], cmap='gray')
                axes[2].set_title('真实掩膜')
                axes[2].axis('off')

            elif pred_mask is not None:
                fig, axes = plt.subplots(1, 3, figsize=(12, 4))

                axes[0].imshow(sample['temp'], cmap='gray')
                axes[0].set_title('温度场')
                axes[0].axis('off')

                axes[1].imshow(sample['stress'], cmap='gray')
                axes[1].set_title('应力场')
                axes[1].axis('off')

                axes[2].imshow(pred_mask, cmap='gray')
                axes[2].set_title('预测掩膜')
                axes[2].axis('off')

            else:
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))

                axes[0].imshow(sample['temp'], cmap='gray')
                axes[0].set_title('温度场')
                axes[0].axis('off')

                axes[1].imshow(sample['stress'], cmap='gray')
                axes[1].set_title('应力场')
                axes[1].axis('off')

            plt.suptitle(f'{batch_name} - 样本 {i:04d}', fontsize=12)
            plt.tight_layout()
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close()
            count += 1

        except Exception as e:
            print(f"  警告: 处理样本 {i} 时出错: {e}")
            continue

    print(f"  完成！已生成 {count} 张对比图到: {output_dir}")


def list_batches():
    """列出所有可用的数据批次"""
    data_root = get_data_root()
    if not data_root:
        ensure_config()
        data_root = get_data_root()

    if not data_root:
        print("错误: 无法获取数据根目录")
        return []

    batches = get_data_batches()
    if not batches:
        print("未找到数据批次")
        return []

    print("=" * 60)
    print("可用数据批次:")
    print("=" * 60)
    for batch_path in batches:
        batch_name = os.path.basename(os.path.normpath(batch_path))
        info = get_batch_info(batch_path)
        if info:
            d_hist_status = "[OK] 有 d_hist" if info['has_d_hist'] else "[--] 无 d_hist"
            print(f"  {batch_name}: {info['n_samples']} 样本, {d_hist_status}")
        else:
            print(f"  {batch_name}: (无法加载)")
    print("=" * 60)

    return batches


# =============================================================================
# 主函数
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='PE-MMNet v4 可视化工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

  # 1. 预处理检查（简单）
  python tools/visualization.py --check-preprocess \\
      --d_hist "D:/数据/单次扫描/d/d001.png" --threshold 0.1

  # 2. 预处理检查（含等值线去除）
  python tools/visualization.py --check-preprocess \\
      --d_hist "D:/数据/参数化扫描4/d/d001.png" \\
      --show-contour-removal --save output/preprocess.png

  # 3. 列出可用批次
  python tools/visualization.py --list-batches

  # 4. 单样本对比（无模型）
  python tools/visualization.py --compare --batch "参数化扫描1" --index 0

  # 5. 单样本对比（带预测）
  python tools/visualization.py --compare \\
      --batch "参数化扫描1" --index 5 \\
      --checkpoint ./checkpoints/xxx_tasksegmentation_offset0_best.pt

  # 6. 多任务对比
  python tools/visualization.py --compare \\
      --batch "参数化扫描1" --index 5 --task multitask \\
      --checkpoint ./checkpoints/xxx_taskmultitask_offset0_best.pt

  # 7. 批量对比
  python tools/visualization.py --batch-compare \\
      --batch "参数化扫描1" --start 0 --end 20 \\
      --checkpoint ./checkpoints/xxx.pt \\
      --output output/visualization/
"""
    )

    # 操作模式
    parser.add_argument('--check-preprocess', action='store_true',
                        help='启用预处理检查模式')
    parser.add_argument('--compare', action='store_true',
                        help='启用单样本对比模式')
    parser.add_argument('--batch-compare', action='store_true',
                        help='启用批量对比模式')
    parser.add_argument('--list-batches', action='store_true',
                        help='列出所有可用的数据批次')

    # 预处理参数
    parser.add_argument('--d_hist', type=str,
                        help='d_hist 图像路径（用于预处理检查）')
    parser.add_argument('--threshold', type=float, default=0.1,
                        help='二值化阈值 (默认: 0.1)')
    parser.add_argument('--invert', action='store_true', default=True,
                        help='反转掩膜 (默认: True)')
    parser.add_argument('--no-invert', action='store_true',
                        help='不反转掩膜')
    parser.add_argument('--show-contour-removal', action='store_true',
                        help='显示等值线去除效果')
    parser.add_argument('--save', type=str,
                        help='保存输出图像的路径')
    parser.add_argument('--high-res', action='store_true', default=True,
                        help='高分辨率模式 512x512 (默认: True)')
    parser.add_argument('--low-res', action='store_true',
                        help='低分辨率模式 256x256')
    parser.add_argument('--cmap', type=str, default='viridis',
                        choices=list(COLORMAPS.keys()),
                        help=f'原始图像色图 (默认: viridis)')

    # 样本对比参数
    parser.add_argument('--batch', type=str,
                        help='批次名称')
    parser.add_argument('--index', type=int, default=0,
                        help='样本索引 (默认: 0)')
    parser.add_argument('--checkpoint', type=str,
                        help='模型检查点路径（可选）')
    parser.add_argument('--task', type=str, default='segmentation',
                        choices=['detection', 'segmentation', 'multitask'],
                        help='任务类型 (默认: segmentation)')

    # 批量对比参数
    parser.add_argument('--start', type=int, default=0,
                        help='起始索引 (默认: 0)')
    parser.add_argument('--end', type=int, default=10,
                        help='结束索引 (默认: 10)')
    parser.add_argument('--output', type=str, default='output/visualization',
                        help='输出目录 (默认: output/visualization)')

    args = parser.parse_args()

    # 处理反转参数
    invert = not args.no_invert

    # 根据模式执行对应功能
    if args.list_batches:
        list_batches()

    elif args.check_preprocess:
        if not args.d_hist:
            print("错误: --check-preprocess 需要 --d_hist 参数")
            print("示例: python tools/visualization.py --check-preprocess --d_hist \"单次扫描/d/d001.png\"")
            sys.exit(1)

        # 检查路径是否是绝对路径
        d_hist_path = args.d_hist
        # 统一路径分隔符（Windows 兼容）
        d_hist_path = d_hist_path.replace('/', os.sep).replace('\\', os.sep)
        if not os.path.isabs(d_hist_path):
            # 尝试从数据根目录构建路径
            data_root = get_data_root()
            if data_root:
                d_hist_path = os.path.join(data_root, d_hist_path)

        check_preprocess(d_hist_path, args.threshold, invert,
                        args.show_contour_removal, args.save,
                        high_res=not args.low_res, cmap=args.cmap)

    elif args.compare:
        if not args.batch:
            print("错误: --compare 需要 --batch 参数")
            print("示例: python tools/visualization.py --compare --batch \"参数化扫描1\" --index 0")
            sys.exit(1)

        compare_sample(args.batch, args.index, args.checkpoint,
                      args.task, args.save)

    elif args.batch_compare:
        if not args.batch:
            print("错误: --batch-compare 需要 --batch 参数")
            print("示例: python tools/visualization.py --batch-compare --batch \"参数化扫描1\" --start 0 --end 20")
            sys.exit(1)

        batch_compare(args.batch, args.start, args.end, args.checkpoint,
                      args.task, args.output)

    else:
        # 默认显示帮助
        parser.print_help()
        print("\n" + "=" * 60)
        print("提示: 使用 --list-batches 查看可用的数据批次")
        print("=" * 60)


if __name__ == '__main__':
    main()
