"""
PE-MMNet v4 多批次多模态裂纹数据集加载器

支持功能：
- 多批次数据合并加载（单次扫描、参数化扫描1、参数化扫描2）
- 跨批次图像归一化（Max-Min归一化解决色标差异）
- 坐标轴对齐（Center Crop聚焦物理区域）
- 统一训练/测试集划分
- 支持像素级裂纹掩膜分割任务（d_hist图像作为标签）
- 支持 --task 参数：detection（检测）/ segmentation（分割）/ multitask（多任务）

核心设计：
1. MultiBatchDataset：管理多个批次的数据加载
2. 每张图像执行：灰度转换 → Max-Min归一化 → Center Crop → Resize → Tensor
3. 跨批次统一随机80/20划分
4. 分割模式：读取 d_hist/*.png 作为二值掩膜标签
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import interp1d
from torch.utils.data import Dataset, ConcatDataset
import torchvision.transforms as T


# =============================================================================
# 全局配置
# =============================================================================

# 图像尺寸配置（可动态修改）
IMAGE_SIZE = 256  # 默认 256x256，支持 512, 1024 等更高分辨率
IMAGE_SIZE_OPTIONS = [256, 384, 512, 768, 1024]  # 常用分辨率选项


# =============================================================================
# 常量定义
# =============================================================================

# 数据批次配置（固定顺序）
DATA_BATCHES = [
    "单次扫描",
    "参数化扫描1",
    "参数化扫描2",
]

# 图像参数
IMAGE_SIZE = 256  # 最终resize尺寸
RAW_IMAGE_SIZE = (640, 480)  # 原始图像尺寸

# 物理区域（10mm×10mm）对应矩阵中心的裁剪比例
# 假设图像留白约15%，物理区域约占70%
CROP_RATIO = 0.70

# 1D序列参数
SEQ_LEN = 300


# =============================================================================
# 动态路径获取（从配置中心读取）
# =============================================================================

def _get_data_root():
    """获取数据根目录（从配置中心）"""
    try:
        # 尝试从配置中心获取
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.config import get_data_root

        root = get_data_root()
        if root:
            return root
    except Exception:
        pass

    # 如果配置中心未设置或失败，返回None
    return None


# 懒加载的 DATA_ROOT
class _LazyDataRoot:
    """懒加载的数据根目录"""
    _root = None

    def __str__(self):
        if self._root is None:
            self._root = _get_data_root()
        return self._root or ""

    def __repr__(self):
        return self.__str__()

    def __bool__(self):
        if self._root is None:
            self._root = _get_data_root()
        return bool(self._root)

    def __fspath__(self):
        return self.__str__()


DATA_ROOT = _LazyDataRoot()

# 图像参数
IMAGE_SIZE = 256  # 最终resize尺寸
RAW_IMAGE_SIZE = (640, 480)  # 原始图像尺寸

# 物理区域（10mm×10mm）对应矩阵中心的裁剪比例
# 假设图像留白约15%，物理区域约占70%
CROP_RATIO = 0.70

# 1D序列参数
SEQ_LEN = 300


# =============================================================================
# 工具函数
# =============================================================================

def natural_sort_key(s):
    """自然排序key函数（处理'001', '002'等）"""
    import re
    return [int(c) if c.isdigit() else c.lower() for c in re.split(r'(\d+)', s)]


def load_csv_safely(filepath):
    """
    安全加载CSV文件（跳过COMSOL注释行）

    Args:
        filepath: CSV文件路径

    Returns:
        pandas.DataFrame: 跳过注释行的数据框
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 找到数据起始位置
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip().startswith('%'):
            data_start = i + 1
        else:
            break

    # 提取数据行
    if data_start < len(lines):
        first_line = lines[data_start].strip()
        if first_line and (first_line[0].isdigit() or first_line[0] == '-'):
            data_lines = lines[data_start:]
        else:
            data_lines = lines[data_start + 1:]
    else:
        data_lines = []

    from io import StringIO
    data_str = ''.join(data_lines)
    df = pd.read_csv(StringIO(data_str), header=None)

    return df


def parse_label_csv(df):
    """
    解析标签CSV（支持多种格式）

    根据列索引和统计特征推断列含义：
    - 单次扫描（10列）：time(0), temp(1), x(2), y(3), density(4), hist_x(5), hist_y(6), hist_d(7)
    - 参数化扫描（11列）：h(0), time(1), temp(2), x(3), y(4), density(5), hist_x(6), hist_y(7), hist_d(8)

    Args:
        df: 原始DataFrame

    Returns:
        dict: 解析后的数据
    """
    n_cols = len(df.columns)

    # 根据列数决定格式
    if n_cols == 11 or (n_cols == 6 and df.iloc[:, 0].mean() > 100):
        # 参数化扫描格式：h, time, temp, x, y, density, ...
        result = {
            'h_ceramic': df.iloc[:, 0].values.astype(np.float64),
            'time': df.iloc[:, 1].values.astype(np.float64),
            'temp': df.iloc[:, 2].values.astype(np.float64),
            'x': df.iloc[:, 3].values.astype(np.float64),
            'y': df.iloc[:, 4].values.astype(np.float64),
            'density': df.iloc[:, 5].values.astype(np.float64),
        }
    elif n_cols == 10 or n_cols >= 9:
        # 单次扫描格式：time, temp, x, y, density, hist_x, hist_y, hist_d...
        result = {
            'time': df.iloc[:, 0].values.astype(np.float64),
            'temp': df.iloc[:, 1].values.astype(np.float64),
            'x': df.iloc[:, 2].values.astype(np.float64),
            'y': df.iloc[:, 3].values.astype(np.float64),
            'density': df.iloc[:, 4].values.astype(np.float64),
            'h_ceramic': np.zeros(len(df)),  # 单次扫描没有h参数
        }
    else:
        # 通用解析：基于统计特征
        cols = df.columns.tolist()
        result = {}

        # 分析每列的数据特征
        col_stats = []
        for c in cols:
            col_data = df[c].values.astype(np.float64)
            stats = {
                'col': c,
                'min': np.min(col_data),
                'max': np.max(col_data),
                'mean': np.mean(col_data),
                'std': np.std(col_data),
                'range': np.max(col_data) - np.min(col_data),
            }
            col_stats.append(stats)

        # 1. 时间列：起始值接近0，后续递增
        time_candidates = [s for s in col_stats if s['min'] >= 0 and s['std'] > 0 and s['min'] < 1]
        if time_candidates:
            time_candidates.sort(key=lambda x: x['min'])
            time_col = time_candidates[0]['col']
        else:
            time_col = cols[0]
        result['time'] = df[time_col].values.astype(np.float64)

        # 2. 温度列：均值 > 500
        temp_candidates = [s for s in col_stats if s['mean'] > 500 and s['max'] > 1000]
        if temp_candidates:
            temp_col = temp_candidates[0]['col']
        else:
            temp_col = cols[2] if n_cols >= 3 else cols[1]
        result['temp'] = df[temp_col].values.astype(np.float64)

        # 3. 坐标列：范围 < 0.1
        coord_candidates = [s for s in col_stats if s['max'] > 0 and s['max'] < 0.1 and s['mean'] > 0]
        if len(coord_candidates) >= 2:
            coord_candidates.sort(key=lambda x: x['max'])
            x_col = coord_candidates[0]['col']
            y_col = coord_candidates[1]['col']
        else:
            x_col = cols[2] if n_cols > 2 else cols[0]
            y_col = cols[3] if n_cols > 3 else cols[1]

        result['x'] = df[x_col].values.astype(np.float64)
        result['y'] = df[y_col].values.astype(np.float64)

        # 4. 密度列：范围 0-1
        density_candidates = [s for s in col_stats if 0 <= s['min'] and s['max'] <= 1.1 and s['range'] > 0.01]
        if density_candidates:
            density_candidates.sort(key=lambda x: x['range'], reverse=True)
            density_col = density_candidates[0]['col']
        else:
            density_col = cols[-1]
        result['density'] = df[density_col].values.astype(np.float64)

        # 5. h_ceramic
        h_candidates = [s for s in col_stats if s['mean'] > 100 and s['std'] > 10 and s['min'] >= 0]
        if h_candidates:
            h_col = h_candidates[0]['col']
        else:
            h_col = None
        result['h_ceramic'] = df[h_col].values.astype(np.float64) if h_col else np.zeros(len(df))

    return result

    return result


def parse_probe_csv(df):
    """
    解析参数化扫描探针CSV

    Args:
        df: 原始DataFrame

    Returns:
        dict: {'h_ceramic': array, 'time': array, 'temp_values': array}
    """
    result = {
        'h_ceramic': df.iloc[:, 0].values.astype(np.float64),
        'time': df.iloc[:, 1].values.astype(np.float64),
    }

    if len(df.columns) >= 3:
        temp_cols = df.iloc[:, 2:]
        result['temp_values'] = temp_cols.mean(axis=1).values.astype(np.float32)
    else:
        result['temp_values'] = np.zeros(len(df), dtype=np.float32)

    return result


def interpolate_seq(time_points, values, target_len=SEQ_LEN):
    """将序列插值到目标长度"""
    values = np.array(values)

    # 处理空数组
    if len(values) == 0:
        return np.zeros(target_len, dtype=np.float32)

    # 处理单元素
    if len(values) < 2:
        return np.full(target_len, values[0], dtype=np.float32)

    # 如果没有提供 time_points，使用索引
    if time_points is None:
        time_points = np.arange(len(values))
    else:
        time_points = np.array(time_points)

    sort_idx = np.argsort(time_points)
    time_points = time_points[sort_idx]
    values = values[sort_idx]

    interp_func = interp1d(
        time_points, values,
        kind='linear',
        fill_value='extrapolate',
        assume_sorted=True
    )

    t_min, t_max = time_points.min(), time_points.max()
    target_time = np.linspace(t_min, t_max, target_len)
    result = interp_func(target_time).astype(np.float32)
    result = np.ascontiguousarray(result)

    if np.any(np.isnan(result)):
        mask = np.isnan(result)
        result[mask] = np.interp(np.where(mask)[0], np.where(~mask)[0], result[~mask])

    return result


def create_triple_channel_seq(seq_1d, seq_len):
    """
    将单通道序列转换为三通道序列（初始温度 + 当前温度 + 温度变化率）

    Args:
        seq_1d: 单通道序列 (seq_len,)
        seq_len: 目标序列长度

    Returns:
        三通道序列 (3, seq_len)
    """
    seq = np.array(seq_1d, dtype=np.float32)

    # 通道0: 初始温度（第一个温度值）
    init_temp = np.full(seq_len, seq[0] if len(seq) > 0 else 0.0, dtype=np.float32)

    # 通道1: 当前温度
    curr_temp = seq.copy()

    # 通道2: 温度变化率（差分计算）
    # 使用前向差分，最后一个点使用前一个差分值
    if len(seq) >= 2:
        temp_rate = np.diff(seq, prepend=seq[0])
    else:
        temp_rate = np.zeros(seq_len, dtype=np.float32)

    # 归一化温度变化率到合理范围（-1, 1）
    rate_std = np.std(temp_rate) + 1e-6
    temp_rate = np.clip(temp_rate / (rate_std * 3), -1, 1).astype(np.float32)

    # 堆叠成三通道
    triple_seq = np.stack([init_temp, curr_temp, temp_rate], axis=0)

    return triple_seq


def process_seq(time_points, values, target_len=SEQ_LEN, mode='interpolate'):
    """
    处理序列到目标长度

    支持两种模式：
    - 'interpolate': 线性插值（默认，适合短序列）
    - 'pool': 自适应池化（适合长序列，减少高频信息损失）

    Args:
        time_points: 原始时间点（可选，pool模式可传None）
        values: 序列值
        target_len: 目标长度
        mode: 处理模式，'interpolate' 或 'pool'

    Returns:
        处理后的 numpy 数组
    """
    original_len = len(values)

    # 如果原始序列已经不长，使用插值
    if original_len <= target_len or mode == 'interpolate':
        return interpolate_seq(time_points, values, target_len)

    # 长序列模式：使用自适应池化减少高频信息损失
    import torch
    tensor = torch.tensor(values, dtype=torch.float32)
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len)
    pooled = torch.nn.functional.adaptive_avg_pool1d(tensor, target_len)
    result = pooled.squeeze().numpy().astype(np.float32)
    result = np.ascontiguousarray(result)

    return result


def compute_lw_from_density(density):
    """根据裂纹密度计算边界框尺寸"""
    L_SCALE, W_SCALE = 0.3, 0.2
    density = max(0.0, min(1.0, density))
    return float(np.sqrt(density) * L_SCALE), float(np.sqrt(density) * W_SCALE)


# =============================================================================
# 图像预处理
# =============================================================================

def imread_unicode(filepath):
    """解决 OpenCV/PIL 中文路径读取问题"""
    try:
        # 使用 np.fromfile + cv2.imdecode
        import cv2
        import numpy as np
        with open(filepath, 'rb') as f:
            data = f.read()
        img_array = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        if img is not None:
            return Image.fromarray(img)
    except Exception:
        pass
    # fallback: 直接使用 PIL
    try:
        return Image.open(filepath)
    except Exception:
        return None


class ImagePreprocessor:
    """
    图像预处理器
    核心功能：
    1. 灰度转换
    2. Max-Min归一化（解决跨批次色标差异）
    3. Center Crop（聚焦物理区域）
    4. Resize到目标尺寸
    """

    def __init__(self, image_size=IMAGE_SIZE, crop_ratio=CROP_RATIO):
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
        # 确保裁剪区域为正方形（使用 min(h,w) 保证 1:1 长宽比）
        h, w = img_norm.shape
        crop_size = int(min(h, w) * self.crop_ratio)  # 正方形边长
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        img_cropped = img_norm[start_h:start_h+crop_size, start_w:start_w+crop_size]

        # 5. Resize到目标尺寸
        img_pil = Image.fromarray((img_cropped * 255).astype(np.uint8))
        img_resized = img_pil.resize((self.image_size, self.image_size), Image.BILINEAR)

        # 6. 转为Tensor [0, 1]
        tensor = T.ToTensor()(img_resized)

        return tensor


class MaskLabelProcessor:
    """
    分割标签处理器
    将 d_hist/*.png 损伤场图像转换为二值掩膜
    """

    def __init__(self, image_size=IMAGE_SIZE, crop_ratio=CROP_RATIO,
                 binary_threshold=0.1, invert=True):
        """
        Args:
            image_size: 目标图像尺寸
            crop_ratio: 中心裁剪比例（需与 ImagePreprocessor 一致）
            binary_threshold: 二值化阈值
            invert: 是否反转（d_hist 裂纹区域为暗色）
        """
        self.image_size = image_size
        self.crop_ratio = crop_ratio
        self.binary_threshold = binary_threshold
        self.invert = invert

    def __call__(self, mask_path):
        """
        读取并处理掩膜图像

        Args:
            mask_path: d_hist 图像路径

        Returns:
            torch.Tensor: (1, H, W)，二值掩膜，值域[0, 1]
        """
        import cv2
        import numpy as np

        # 读取图像（支持中文路径）
        try:
            with open(mask_path, 'rb') as f:
                data = f.read()
            img_array = np.frombuffer(data, np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return torch.zeros(1, self.image_size, self.image_size)

        if img is None:
            return torch.zeros(1, self.image_size, self.image_size)

        # Center Crop
        h, w = img.shape
        crop_h = int(h * self.crop_ratio)
        crop_w = int(w * self.crop_ratio)
        start_h = (h - crop_h) // 2
        start_w = (w - crop_w) // 2
        img_cropped = img[start_h:start_h+crop_h, start_w:start_w+crop_w]

        # 归一化
        img_norm = img_cropped.astype(np.float32) / 255.0

        # 反转（如果需要）
        if self.invert:
            img_norm = 1.0 - img_norm

        # 二值化
        mask_binary = (img_norm > self.binary_threshold).astype(np.float32)

        # Resize
        mask_pil = Image.fromarray((mask_binary * 255).astype(np.uint8))
        mask_resized = mask_pil.resize((self.image_size, self.image_size), Image.NEAREST)

        # 转为 Tensor
        tensor = T.ToTensor()(mask_resized)

        return tensor


# =============================================================================
# 数据增强
# =============================================================================

class PhysicalSafeTransform1D:
    """1D序列的物理安全数据增强"""
    def __init__(self, mask_ratio=0.05, noise_std=0.005, enabled=True):
        self.mask_ratio = mask_ratio
        self.noise_std = noise_std
        self.enabled = enabled

    def __call__(self, seq):
        if not self.enabled:
            return seq
        seq = seq.copy()
        seq_len = seq.shape[-1]
        if np.random.random() < 0.3:
            mask_len = max(1, int(seq_len * self.mask_ratio))
            mask_start = np.random.randint(0, seq_len - mask_len)
            seq[..., mask_start:mask_start + mask_len] = 0
        seq += np.random.randn(seq_len).astype(np.float32) * self.noise_std
        return seq


class GaussianNoise:
    def __init__(self, std=0.01):
        self.std = std

    def __call__(self, img):
        noise = torch.randn_like(img) * self.std
        return (img + noise).clamp(0, 1)


class PhysicalSafeTransform2D:
    """2D图像的物理安全数据增强"""
    def __init__(self, noise_std=0.01, enabled=True):
        self.enabled = enabled
        if enabled:
            self.transform = T.Compose([GaussianNoise(noise_std)])
        else:
            self.transform = None

    def __call__(self, img_tensor):
        if not self.enabled or self.transform is None:
            return img_tensor
        return self.transform(img_tensor)


# =============================================================================
# 单批次数据集
# =============================================================================

class SingleBatchDataset(Dataset):
    """
    单批次数据集
    加载一个数据目录下的所有样本
    支持检测、分割、多任务三种模式
    """

    def __init__(self, data_root, seq_len=SEQ_LEN, image_size=IMAGE_SIZE,
                 augment=True, sample_indices=None, predict_offset=0,
                 seq_interp_mode='interpolate', remove_contours=False,
                 task='detection', triple_channel=False):
        """
        Args:
            data_root: 数据目录根路径
            seq_len: 1D序列长度
            image_size: 2D图像尺寸
            augment: 是否启用数据增强
            sample_indices: 指定使用的样本索引（用于划分）
            predict_offset: 时间偏移量（预测未来多少步的标签，默认0）
            seq_interp_mode: 序列处理模式，'interpolate' 或 'pool'
            remove_contours: 是否对参数化扫描4启用等值线去除
            task: 任务模式，'detection' / 'segmentation' / 'multitask'
            triple_channel: 是否启用三通道时序输入（初始温度 + 当前温度 + 温度变化率）
        """
        self.triple_channel = triple_channel
        self.data_root = data_root
        self.seq_len = seq_len
        self.image_size = image_size
        self.augment = augment
        self.sample_indices = sample_indices
        self.predict_offset = predict_offset
        self.seq_interp_mode = seq_interp_mode
        self.remove_contours = remove_contours
        self.task = task

        # 获取批次名称
        self.batch_name = os.path.basename(os.path.normpath(data_root))
        self.data_root = data_root
        self.seq_len = seq_len
        self.image_size = image_size
        self.augment = augment
        self.sample_indices = sample_indices
        self.predict_offset = predict_offset

        # 检查数据目录
        table_dir = os.path.join(data_root, '表格')
        has_table_dir = os.path.isdir(table_dir)

        # 动态查找标签CSV（标签CSV可选）
        label_candidates = [
            '参数化扫描-表面最大值.csv',
            'Table_表面最大值.csv',
            'Table_Crack.csv',
            'Table_Hist.csv',
        ]
        self.label_csv_path = None
        if has_table_dir:
            for name in label_candidates:
                path = os.path.join(table_dir, name)
                if os.path.exists(path):
                    self.label_csv_path = path
                    break

        # 动态查找探针CSV（探针CSV可选）
        probe_candidates = [
            '参数化扫描-所有探针.csv',
            'Table_Temp.csv',
        ]
        self.probe_csv_path = None
        if has_table_dir:
            for name in probe_candidates:
                path = os.path.join(table_dir, name)
                if os.path.exists(path):
                    self.probe_csv_path = path

        # 记录是否有CSV数据
        self.has_label_csv = self.label_csv_path is not None
        self.has_probe_csv = self.probe_csv_path is not None

        # 图像目录
        self.temp_image_dir = os.path.join(data_root, '温度')
        self.stress_image_dir = os.path.join(data_root, '应力')

        # d_hist 分割标签目录（可选）
        self.d_hist_dir = os.path.join(data_root, 'd_hist')
        self.has_d_hist = os.path.exists(self.d_hist_dir)
        if self.has_d_hist:
            print(f"    d_hist目录: 存在 ({self.d_hist_dir})")
        else:
            print(f"    d_hist目录: 不存在（分割任务需要）")

        # 自适应加载数据（支持无CSV情况）
        print(f"  加载: {os.path.basename(data_root)}")
        print(f"    标签CSV: {'存在' if self.has_label_csv else '不存在（将使用图片数量）'}")
        print(f"    探针CSV: {'存在' if self.has_probe_csv else '不存在（将使用默认温度序列）'}")

        # 获取图片数量作为样本数基准
        temp_image_dir = os.path.join(data_root, '温度')
        stress_image_dir = os.path.join(data_root, '应力')
        if os.path.isdir(temp_image_dir):
            n_images = len([f for f in os.listdir(temp_image_dir) if f.endswith('.png')])
        else:
            n_images = 0

        # 情况1: 有标签CSV
        if self.has_label_csv:
            label_df = load_csv_safely(self.label_csv_path)
            label_data = parse_label_csv(label_df)

            # 按h_ceramic和Time排序
            sort_idx = np.lexsort((label_data['time'], label_data['h_ceramic']))
            self.h_ceramic = label_data['h_ceramic'][sort_idx]
            self.times = label_data['time'][sort_idx]
            self.temp_values = label_data['temp'][sort_idx]
            self.x_labels = label_data['x'][sort_idx]
            self.y_labels = label_data['y'][sort_idx]
            self.density_labels = label_data['density'][sort_idx]

            n_samples = len(self.times)
            # 取CSV和图片数量的较小值
            n_samples = min(n_samples, n_images) if n_images > 0 else n_samples

        # 情况2: 无标签CSV但有探针CSV
        elif self.has_probe_csv:
            probe_df = load_csv_safely(self.probe_csv_path)
            probe_data = parse_probe_csv(probe_df)
            n_samples = len(probe_data['time'])
            n_samples = min(n_samples, n_images) if n_images > 0 else n_samples

            # 使用探针CSV的数据
            sort_idx = np.lexsort((probe_data['time'], probe_data['h_ceramic']))
            self.h_ceramic = probe_data['h_ceramic'][sort_idx][:n_samples]
            self.times = probe_data['time'][sort_idx][:n_samples]
            self.temp_values = probe_data['temp_values'][sort_idx][:n_samples]

            # 标签使用默认值
            self.x_labels = np.ones(n_samples) * 0.01
            self.y_labels = np.ones(n_samples) * 0.01
            self.density_labels = np.zeros(n_samples)

        # 情况3: 完全没有CSV
        else:
            n_samples = n_images
            if n_samples == 0:
                raise ValueError(f"数据目录中没有图片: {data_root}")

            # 使用图片数量作为样本数
            self.h_ceramic = np.zeros(n_samples)  # 默认h_ceramic
            self.times = np.arange(n_samples)  # 默认时间
            self.temp_values = np.ones(n_samples) * 1400  # 默认温度（陶瓷烧制温度）

            # 标签使用默认值
            self.x_labels = np.ones(n_samples) * 0.01
            self.y_labels = np.ones(n_samples) * 0.01
            self.density_labels = np.zeros(n_samples)

            print(f"    [完全自适应] 无CSV，使用默认标签和温度序列")

        print(f"    样本数: {n_samples}")

        # 建立时序插值函数
        self.unique_h = sorted(set(self.h_ceramic))
        self.seq_300_dict = {}

        # 情况A: 有探针CSV
        if self.has_probe_csv:
            probe_df = load_csv_safely(self.probe_csv_path)
            probe_data = parse_probe_csv(probe_df)
            sort_idx2 = np.lexsort((probe_data['time'], probe_data['h_ceramic']))
            self.probe_h = probe_data['h_ceramic'][sort_idx2]
            self.probe_time = probe_data['time'][sort_idx2]
            self.probe_temp = probe_data['temp_values'][sort_idx2]

            for h_val in self.unique_h:
                mask = self.probe_h == h_val
                h_times = self.probe_time[mask]
                h_temps = self.probe_temp[mask]
                sort_idx3 = np.argsort(h_times)
                h_times = h_times[sort_idx3]
                h_temps = h_temps[sort_idx3]
                self.seq_300_dict[h_val] = interpolate_seq(h_times, h_temps, SEQ_LEN)

        # 情况B: 无探针CSV但有标签CSV
        elif self.has_label_csv:
            for h_val in self.unique_h:
                mask = self.h_ceramic == h_val
                h_temps = self.temp_values[mask]
                h_times = np.arange(len(h_temps))
                self.seq_300_dict[h_val] = interpolate_seq(h_times, h_temps, SEQ_LEN)

        # 情况C: 完全没有CSV
        else:
            # 使用默认温度序列
            default_seq = np.linspace(1400, 1200, SEQ_LEN)  # 从高温降到中温
            for h_val in self.unique_h:
                self.seq_300_dict[h_val] = default_seq

        # 获取排序后的图片文件
        self.temp_images = sorted([f for f in os.listdir(self.temp_image_dir) if f.endswith('.png')], key=natural_sort_key)
        self.stress_images = sorted([f for f in os.listdir(self.stress_image_dir) if f.endswith('.png')], key=natural_sort_key)

        # 对齐样本数（考虑 d_hist）
        if self.has_d_hist:
            d_hist_images = sorted([f for f in os.listdir(self.d_hist_dir) if f.endswith('.png')], key=natural_sort_key)
            actual_n = min(len(self.temp_images), len(self.stress_images), len(d_hist_images), n_samples)
            self.d_hist_images = d_hist_images[:actual_n]
        else:
            actual_n = min(len(self.temp_images), len(self.stress_images), n_samples)
            self.d_hist_images = []

        self.temp_images = self.temp_images[:actual_n]
        self.stress_images = self.stress_images[:actual_n]
        self.h_ceramic = self.h_ceramic[:actual_n]
        self.times = self.times[:actual_n]
        self.x_labels = self.x_labels[:actual_n]
        self.y_labels = self.y_labels[:actual_n]
        self.density_labels = self.density_labels[:actual_n]

        # 如果没有指定sample_indices，使用全部索引
        if self.sample_indices is None:
            self.sample_indices = list(range(actual_n))

        # 预处理和增强
        self.preprocessor = ImagePreprocessor(image_size=image_size)
        self.mask_processor = MaskLabelProcessor(image_size=image_size)
        self.transform_1d = PhysicalSafeTransform1D(enabled=augment)
        self.transform_2d = PhysicalSafeTransform2D(enabled=augment)

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, idx):
        sample_idx = self.sample_indices[idx]

        # 1D序列
        h_val = self.h_ceramic[sample_idx]
        seq_1d = self.seq_300_dict[h_val].copy()

        # 根据原始序列长度和模式决定处理方式
        original_len = len(seq_1d)
        if self.seq_interp_mode == 'pool' and original_len > self.seq_len:
            seq_1d = process_seq(None, seq_1d, self.seq_len, mode='pool')

        seq_1d = self.transform_1d(seq_1d)

        # 三通道处理：初始温度 + 当前温度 + 温度变化率
        if self.triple_channel:
            seq_1d = create_triple_channel_seq(seq_1d, self.seq_len)

        seq_1d = torch.from_numpy(seq_1d).float()

        # 2D图像
        temp_img_path = os.path.join(self.temp_image_dir, self.temp_images[sample_idx])
        stress_img_path = os.path.join(self.stress_image_dir, self.stress_images[sample_idx])

        # 使用中文路径友好的读取方式
        temp_pil = imread_unicode(temp_img_path)
        stress_pil = imread_unicode(stress_img_path)

        temp_tensor = self.preprocessor(temp_pil)
        stress_tensor = self.preprocessor(stress_pil)

        img_2d = torch.cat([temp_tensor, stress_tensor], dim=0)
        img_2d = self.transform_2d(img_2d)

        # 根据任务模式返回不同标签
        if self.task in ['segmentation', 'multitask'] and self.has_d_hist:
            # 分割标签
            mask_path = os.path.join(self.d_hist_dir, self.d_hist_images[sample_idx])
            mask = self.mask_processor(mask_path)
        else:
            mask = torch.zeros(1, self.image_size, self.image_size)

        if self.task == 'segmentation':
            # 仅分割任务
            label = mask
        elif self.task == 'detection':
            # 仅检测任务
            target_idx = sample_idx + self.predict_offset
            x = self.x_labels[target_idx] / 0.02
            y = self.y_labels[target_idx] / 0.015
            density = self.density_labels[target_idx]
            l, w = compute_lw_from_density(density)
            confidence = 1.0 if density > 0.5 else 0.0
            label = torch.tensor([x, y, l, w, confidence, density], dtype=torch.float32)
        else:
            # multitask: 返回 (掩膜, 检测标签) 元组，顺序与模型输出一致
            target_idx = sample_idx + self.predict_offset
            x = self.x_labels[target_idx] / 0.02
            y = self.y_labels[target_idx] / 0.015
            density = self.density_labels[target_idx]
            l, w = compute_lw_from_density(density)
            confidence = 1.0 if density > 0.5 else 0.0
            detection_label = torch.tensor([x, y, l, w, confidence, density], dtype=torch.float32)
            # 统一顺序：(mask, detection) 与 PETSNetMultimodal.forward() 输出一致
            label = (mask, detection_label)

        return (seq_1d, img_2d), label


# =============================================================================
# 多批次数据集
# =============================================================================

class MultiBatchDataset(Dataset):
    """
    多批次合并数据集
    遍历多个数据目录，将所有样本合并为一个统一的数据集
    """

    def __init__(self, data_roots, train_ratio=0.8, seq_len=SEQ_LEN,
                 image_size=IMAGE_SIZE, augment=True, seed=42):
        """
        Args:
            data_roots: 数据目录列表
            train_ratio: 训练集比例
            seq_len: 1D序列长度
            image_size: 2D图像尺寸
            augment: 是否启用数据增强
            seed: 随机种子（保证可复现的划分）
        """
        self.data_roots = data_roots
        self.train_ratio = train_ratio
        self.seq_len = seq_len
        self.image_size = image_size

        # 收集所有数据集的样本信息
        print(f"\n{'='*60}")
        print(f"加载 {len(data_roots)} 个数据批次...")
        print(f"{'='*60}")

        all_sample_infos = []

        for root in data_roots:
            # 加载标签获取样本数
            label_csv = os.path.join(root, '表格', '参数化扫描-表面最大值.csv')
            if os.path.exists(label_csv):
                label_df = load_csv_safely(label_csv)
                label_data = parse_label_csv(label_df)
                n_samples = len(label_data['time'])
                print(f"  {os.path.basename(root)}: {n_samples} 样本")

                for i in range(n_samples):
                    all_sample_infos.append({
                        'root': root,
                        'index': i
                    })
            else:
                print(f"  [警告] {root} 缺少标签文件，跳过")

        print(f"\n总计: {len(all_sample_infos)} 样本")

        # 随机划分
        import random
        random.seed(seed)
        random.shuffle(all_sample_infos)

        train_count = int(len(all_sample_infos) * train_ratio)
        train_infos = all_sample_infos[:train_count]
        test_infos = all_sample_infos[train_count:]

        print(f"训练集: {len(train_infos)} 样本")
        print(f"测试集: {len(test_infos)} 样本")

        # 按root分组加载
        self.datasets = {}
        for root in data_roots:
            root_train = [info['index'] for info in train_infos if info['root'] == root]
            root_test = [info['index'] for info in test_infos if info['root'] == root]

            if root_train or root_test:
                ds = SingleBatchDataset(
                    root, seq_len, image_size, augment=False,
                    sample_indices=list(range(155))  # 暂时使用全部
                )
                self.datasets[root] = ds

        # 保存划分信息
        self.train_infos = train_infos
        self.test_infos = test_infos

        # 预处理
        self.preprocessor = ImagePreprocessor(image_size=image_size)
        self.transform_1d = PhysicalSafeTransform1D(enabled=augment)
        self.transform_2d = PhysicalSafeTransform2D(enabled=augment)

    def get_train_dataset(self):
        """获取训练数据集"""
        train_samples = []

        import random
        random.seed(42)
        random.shuffle(self.train_infos)

        for info in self.train_infos:
            root = info['root']
            idx = info['index']
            ds = self.datasets[root]
            sample = ds[idx]
            train_samples.append(sample)

        return train_samples

    def get_test_dataset(self):
        """获取测试数据集"""
        test_samples = []

        for info in self.test_infos:
            root = info['root']
            idx = info['index']
            ds = self.datasets[root]
            sample = ds[idx]
            test_samples.append(sample)

        return test_samples

    def __len__(self):
        return len(self.train_infos)


class MultiBatchCollateDataset(Dataset):
    """
    多批次数据集（使用ConcatDataset实现）
    支持检测、分割、多任务三种模式
    """

    def __init__(self, data_roots, split='train', train_ratio=0.8,
                 seq_len=SEQ_LEN, image_size=IMAGE_SIZE, augment=True,
                 seed=42, predict_offset=0,
                 seq_interp_mode='interpolate', remove_contours=False,
                 disabled_batches=None, task='detection',
                 triple_channel=False):
        """
        Args:
            data_roots: 数据目录列表
            split: 'train' 或 'test'
            train_ratio: 训练集比例
            seq_len: 序列长度
            image_size: 图像尺寸
            augment: 数据增强
            seed: 随机种子
            predict_offset: 时间偏移量（预测未来多少步的标签）
            seq_interp_mode: 序列处理模式，'interpolate' 或 'pool'
            remove_contours: 是否启用等值线去除
            disabled_batches: 禁用的批次列表
            task: 任务模式，'detection' / 'segmentation' / 'multitask'
            triple_channel: 是否启用三通道时序输入
        """
        self.split = split
        self.seq_len = seq_len
        self.image_size = image_size
        self.predict_offset = predict_offset
        self.seq_interp_mode = seq_interp_mode
        self.remove_contours = remove_contours
        self.disabled_batches = disabled_batches or []
        self.task = task
        self.triple_channel = triple_channel

        try:
            from utils.console import print_title, print_info, print_error, print_warning
        except ImportError:
            # 如果console模块不存在，使用原始print
            def print_title(*args, **kwargs): print(f"\n{'='*60}\n多批次数据集加载 ({split})\n{'='*60}")
            def print_info(*args, **kwargs): print(f"  {args[0]}")
            def print_error(*args, **kwargs): print(f"  [错误] {args[0]}")
            def print_warning(*args, **kwargs): print(f"  [跳过] {args[0]}")

        print_title(f"加载多批次数据集 ({split})")

        # 收集所有数据集信息
        all_sample_infos = []
        batch_sizes = {}

        for root in data_roots:
            # 查找标签CSV文件（支持多种命名方式）
            table_dir = os.path.join(root, '表格')
            label_csv = None

            # 尝试不同的文件名
            possible_names = [
                '参数化扫描-表面最大值.csv',  # 参数化扫描1/2
                'Table_表面最大值.csv',        # 单次扫描
                'Table_Crack.csv',             # 单次扫描备选
                'Table_Hist.csv',              # 备选
            ]

            for name in possible_names:
                candidate = os.path.join(table_dir, name)
                if os.path.exists(candidate):
                    label_csv = candidate
                    break

            if label_csv is None:
                print_warning(f"{os.path.basename(root)}: 未找到标签文件")
                continue

            try:
                label_df = load_csv_safely(label_csv)
                label_data = parse_label_csv(label_df)
                n_samples = len(label_data['time'])

                # 边界处理：过滤掉最后 predict_offset 个样本（避免标签越界）
                max_valid_idx = n_samples - self.predict_offset - 1
                if max_valid_idx < 0:
                    print_warning(f"{os.path.basename(root)}: 样本数不足（{n_samples}），需要至少 {self.predict_offset + 1} 个样本")
                    continue

                valid_count = max_valid_idx + 1
                batch_sizes[root] = valid_count
                print_info(f"{os.path.basename(root)}: {valid_count}/{n_samples} 样本（偏移{self.predict_offset}）")
                for i in range(valid_count):
                    all_sample_infos.append({'root': root, 'index': i})
            except Exception as e:
                print_error(f"{os.path.basename(root)}: {e}")
                continue

        # 随机划分
        import random
        random.seed(seed)
        random.shuffle(all_sample_infos)

        train_count = int(len(all_sample_infos) * train_ratio)
        if split == 'train':
            selected_infos = all_sample_infos[:train_count]
        else:
            selected_infos = all_sample_infos[train_count:]

        print_info(f"{split}集: {len(selected_infos)} 样本")

        # 按root分组
        root_to_indices = {}
        for info in selected_infos:
            root = info['root']
            if root not in root_to_indices:
                root_to_indices[root] = []
            root_to_indices[root].append(info['index'])

        # 创建子数据集
        self.sub_datasets = []
        for root, indices in root_to_indices.items():
            batch_name = os.path.basename(os.path.normpath(root))

            # 检查是否被禁用
            if batch_name in self.disabled_batches:
                print_info(f"跳过禁用的批次: {batch_name}")
                continue

            ds = SingleBatchDataset(
                root, seq_len, image_size, augment=augment,
                sample_indices=indices,
                predict_offset=self.predict_offset,
                seq_interp_mode=self.seq_interp_mode,
                remove_contours=self.remove_contours,
                task=self.task,
                triple_channel=self.triple_channel
            )
            self.sub_datasets.append(ds)

        # ConcatDataset
        self.concat_ds = ConcatDataset(self.sub_datasets)

    def __len__(self):
        return len(self.concat_ds)

    def __getitem__(self, idx):
        return self.concat_ds[idx]


# =============================================================================
# 工厂函数
# =============================================================================

def get_all_data_batches():
    """获取所有数据批次路径"""
    batches = []
    for batch_name in DATA_BATCHES:
        path = os.path.join(DATA_ROOT, batch_name)
        if os.path.exists(path):
            batches.append(path)
    return batches


def create_multibatch_dataloaders(data_roots=None, batch_size=16,
                                  seq_len=SEQ_LEN, image_size=IMAGE_SIZE,
                                  num_workers=0, train_ratio=0.8, augment=True,
                                  predict_offset=0,
                                  seq_interp_mode='interpolate',
                                  remove_contours=False,
                                  disabled_batches=None,
                                  task='detection',
                                  triple_channel=False,
                                  cutmix_prob=0.0):
    """
    创建多批次数据加载器

    Args:
        data_roots: 数据目录列表（None则自动扫描）
        batch_size: 批次大小
        seq_len: 序列长度
        image_size: 图像尺寸
        num_workers: 数据加载线程
        train_ratio: 训练集比例
        augment: 是否启用数据增强
        predict_offset: 时间偏移量
        seq_interp_mode: 序列处理模式，'interpolate' 或 'pool'
        remove_contours: 是否启用等值线去除
        disabled_batches: 禁用的批次列表
        task: 任务模式，'detection' / 'segmentation' / 'multitask'
        triple_channel: 是否启用三通道时序输入（初始温度 + 当前温度 + 温度变化率）
        cutmix_prob: ThermalCutMix 增强概率（默认0关闭）

    Returns:
        tuple: (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    import random

    # 自动获取数据目录
    if data_roots is None:
        data_roots = get_all_data_batches()

    try:
        from utils.console import print_title, print_info
    except ImportError:
        def print_title(*args, **kwargs): pass
        def print_info(*args, **kwargs): print(args[0])

    print_title("创建多批次数据加载器...")
    print_info(f"数据目录: {[os.path.basename(p) for p in data_roots]}")
    if triple_channel:
        print_info("启用三通道时序输入: [初始温度, 当前温度, 温度变化率]")
    if cutmix_prob > 0:
        print_info(f"启用 ThermalCutMix 增强: prob={cutmix_prob}")

    # 创建数据集
    train_dataset = MultiBatchCollateDataset(
        data_roots=data_roots,
        split='train',
        train_ratio=train_ratio,
        seq_len=seq_len,
        image_size=image_size,
        augment=augment,
        predict_offset=predict_offset,
        seq_interp_mode=seq_interp_mode,
        remove_contours=remove_contours,
        disabled_batches=disabled_batches,
        task=task,
        triple_channel=triple_channel
    )

    test_dataset = MultiBatchCollateDataset(
        data_roots=data_roots,
        split='test',
        train_ratio=train_ratio,
        seq_len=seq_len,
        image_size=image_size,
        augment=False,
        predict_offset=predict_offset,
        seq_interp_mode=seq_interp_mode,
        remove_contours=remove_contours,
        disabled_batches=disabled_batches,
        task=task,
        triple_channel=triple_channel
    )

    # 创建DataLoader
    def collate_fn_with_cutmix(batch):
        """collate_fn 支持 ThermalCutMix 增强"""
        # 数据集返回 ((seq_1d, img_2d), label)，一次 zip 拆解
        features_and_labels = [item[0] for item in batch]
        labels_list = [item[1] for item in batch]
        seq_1d_list, img_2d_list = zip(*features_and_labels)
        seq_1d_list, img_2d_list = list(seq_1d_list), list(img_2d_list)

        # ThermalCutMix: 批量级别混合（仅训练集，物理安全版）
        cutmix_applied = [False] * len(batch)
        if cutmix_prob > 0 and augment and len(batch) >= 2 and random.random() < cutmix_prob:
            # 随机选择一对样本
            idx1, idx2 = random.sample(range(len(batch)), 2)
            lambda_ = random.betavariate(1.0, 1.0)

            # 仅混合温度通道（物理安全）
            img1 = img_2d_list[idx1]
            img2 = img_2d_list[idx2]

            mixed_img = img1.clone()
            # 温度通道混合
            mixed_img[0] = img1[0] * lambda_ + img2[0] * (1 - lambda_)
            # 应力通道保持不变

            # 标签线性插值
            mixed_labels = lambda_ * labels_list[idx1] + (1 - lambda_) * labels_list[idx2]

            # 替换其中一个样本
            img_2d_list = list(img_2d_list)
            labels_list = list(labels_list)
            img_2d_list[idx1] = mixed_img
            labels_list[idx1] = mixed_labels
            cutmix_applied[idx1] = True

        # 堆叠
        seq_1d = torch.stack(seq_1d_list)
        img_2d = torch.stack(img_2d_list)

        # multitask 模式：label 是 (mask, detection) 元组，需要分别堆叠
        if isinstance(labels_list[0], tuple):
            mask_list, detection_list = zip(*labels_list)
            labels = (torch.stack(mask_list), torch.stack(detection_list))
        else:
            labels = torch.stack(labels_list)

        return (seq_1d, img_2d), labels

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        collate_fn=collate_fn_with_cutmix
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, test_loader


# =============================================================================
# 测试
# =============================================================================

if __name__ == "__main__":
    try:
        from utils.console import print_title, print_info
    except ImportError:
        def print_title(*args, **kwargs): print(f"\n{'='*60}\n测试多批次数据加载\n{'='*60}")
        def print_info(*args, **kwargs): print(f"  {args[0]}")

    print_title("测试多批次数据加载")

    # 获取所有数据目录
    batches = get_all_data_batches()
    print_info(f"找到 {len(batches)} 个数据批次:")
    for b in batches:
        print_info(f"  - {os.path.basename(b)}")

    # 创建数据加载器
    train_loader, test_loader = create_multibatch_dataloaders(
        data_roots=batches,
        batch_size=4
    )

    print(f"\n训练样本数: {len(train_loader.dataset)}")
    print(f"测试样本数: {len(test_loader.dataset)}")

    # 测试加载一个批次
    print("\n加载一个批次...")
    (seq_1d, img_2d), label = train_loader.dataset[0]

    print(f"1D序列形状: {seq_1d.shape}")
    print(f"2D图像形状: {img_2d.shape}")
    print(f"标签形状: {label.shape}")
    print(f"标签内容: {label.numpy()}")

    print("\n测试通过!")
