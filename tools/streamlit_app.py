"""
PE-MMNet v4 交互式可视化工具 (Streamlit)

功能：
- 实时调整二值化阈值
- 高分辨率预览
- 多种色图切换
- 气孔与裂纹分离提取
- 结构分解可视化

运行方式：
    streamlit run tools/streamlit_app.py

依赖安装：
    pip install streamlit opencv-python scipy
"""

import streamlit as st
import sys
import os
import numpy as np
from PIL import Image
import cv2
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.dataset_multimodal import (
    MaskLabelProcessor, ImagePreprocessor, imread_unicode,
    load_csv_safely, parse_label_csv
)
from utils.config import get_data_batches, get_data_root
from data.image_preprocessor import ContourRemover
from tools.structure_analyzer import separate_crack_pore, analyze_structure, auto_optimize_params

# 页面配置
st.set_page_config(
    page_title="PE-MMNet v4 裂纹分析工具",
    layout="wide",
    page_icon="🔍"
)

# 移除 Streamlit 默认列间距
st.markdown("""
<style>
[data-testid="stHorizontalBlock"] > div {
    padding: 0px !important;
}
[data-testid="column"] {
    padding: 0px !important;
}
.block-container {
    padding: 0px !important;
}
</style>
""", unsafe_allow_html=True)

# 标题
st.title("🔍 PE-MMNet v4 交互式裂纹分析器")
st.markdown("---")

# 侧边栏：参数控制
st.sidebar.header("⚙️ 参数设置")

# 获取数据批次
data_root = get_data_root()
batches = get_data_batches() if data_root else []

if not batches:
    st.error("No data found. Please configure data path first.")
    st.stop()

# 批次选择
batch_options = [os.path.basename(os.path.normpath(b)) for b in batches]
selected_batch = st.sidebar.selectbox("Select Batch", batch_options)

# 获取批次路径
batch_path = os.path.join(data_root, selected_batch)
batch_name = selected_batch

# 获取样本数
table_dir = os.path.join(batch_path, '表格')
label_csv_path = None
for name in ['参数化扫描-表面最大值.csv', 'Table_表面最大值.csv', 'Table_Crack.csv', 'Table_Hist.csv']:
    path = os.path.join(table_dir, name)
    if os.path.exists(path):
        label_csv_path = path
        break

if label_csv_path:
    label_df = load_csv_safely(label_csv_path)
    label_data = parse_label_csv(label_df)
    # 按 h_ceramic 和 Time 排序（与数据集一致）
    sort_idx = np.lexsort((label_data['time'], label_data['h_ceramic']))
    label_data['time'] = label_data['time'][sort_idx]
    label_data['h_ceramic'] = label_data['h_ceramic'][sort_idx]
    label_data['density'] = label_data['density'][sort_idx]
    n_samples = len(label_data['time'])
else:
    label_data = None
    n_samples = 0

# 样本索引
sample_index = st.sidebar.slider("样本索引", 0, max(0, n_samples-1), 0)

# 获取当前样本的物理时间
if label_data is not None and sample_index < len(label_data['time']):
    current_time = label_data['time'][sample_index]
    current_h = label_data['h_ceramic'][sample_index]
    current_density = label_data['density'][sample_index]
    time_display = f"t = {current_time:.4f} s | h = {current_h} | ρ = {current_density:.4f}"
else:
    current_time = None
    time_display = ""

# 可视化模式选择
st.sidebar.markdown("---")
st.sidebar.header("🔍 可视化模式")
view_mode = st.sidebar.radio(
    "选择模式",
    ["二值掩膜", "结构分解视图", "对比视图"],
    help="二值掩膜: 传统二值化 | 结构分解: 气孔与裂纹分离 | 对比视图: 两者叠加"
)

# 高分辨率选项（可调节）
target_size = st.sidebar.slider(
    "图像分辨率",
    min_value=256,
    max_value=2048,
    value=512,
    step=256,
    help="选择输出图像分辨率。更高分辨率=更清晰但处理更慢"
)

# 色图选择
st.sidebar.markdown("---")
st.sidebar.header("🎨 色图设置")
cmap_options = {
    'gray': '灰度 (Gray)',
    'viridis': 'Viridis (科学)',
    'jet': 'Jet (彩虹)',
    'hot': '热力图 (Hot)',
    'inferno': 'Inferno',
    'plasma': 'Plasma',
}
selected_cmap = st.sidebar.selectbox("原始图像色图", list(cmap_options.keys()),
                                      format_func=lambda x: cmap_options[x])

# 气孔/裂纹分离参数
st.sidebar.markdown("---")
st.sidebar.header("⚡ 结构分离参数")

# 裂纹阈值（低阈值，提取暗色区域）
crack_threshold = st.sidebar.slider(
    "裂纹阈值 (Crack Threshold)",
    min_value=0.05,
    max_value=0.50,
    value=0.20,
    step=0.01,
    help="低于此值的暗色区域被识别为裂纹"
)

# 气孔阈值（高阈值，提取亮色区域）
pore_threshold = st.sidebar.slider(
    "气孔阈值 (Pore Threshold)",
    min_value=0.50,
    max_value=0.95,
    value=0.70,
    step=0.01,
    help="高于此值的亮色区域被识别为气孔"
)

# 显示模式：骨架线 vs 实心掩膜
crack_mode = st.sidebar.radio(
    "裂纹显示模式",
    ["骨架线 (Skeleton)", "实心掩膜 (Solid)"],
    help="骨架线：细线轨迹 | 实心掩膜：宽带区域（适合分割任务）"
)
crack_mode = 'skeleton' if '骨架线' in crack_mode else 'solid'

# 高级参数折叠
with st.sidebar.expander("⚙️ 高级参数"):
    edge_exclusion = st.slider("忽略边界像素 (px)", 0, 40, 25, 1,
                             help="自动排除图像边缘区域的伪影，默认25像素")
    merge_distance = st.slider("裂纹碎片合并距离", 0, 20, 8, 1,
                              help="将距离小于此值的相邻碎片合并为同一条裂纹")
    crack_min_area = st.slider("裂纹最小面积", 10, 200, 30, 10)
    crack_min_aspect = st.slider("裂纹最小长宽比", 1.5, 5.0, 2.0, 0.1)

# 功能选项
st.sidebar.markdown("---")
show_contour = st.sidebar.checkbox("显示等值线去除")
save_image = st.sidebar.button("💾 保存当前视图")

# 裁剪可视化设置
st.sidebar.markdown("---")
st.sidebar.subheader("📐 裁剪设置")
show_crop_box = st.sidebar.checkbox("显示裁剪框", value=False,
    help="在原始图像上显示 Center Crop 70% 的裁剪区域")
crop_ratio = st.sidebar.slider("裁剪比例", 0.5, 1.0, 0.7, 0.05,
    help="Center Crop 的比例，使用 min(H,W) 计算正方形区域")

# 交互诊断模式
st.sidebar.markdown("---")
enable_click_diagnose = st.sidebar.checkbox("🔍 启用点击诊断模式", value=False,
    help="开启后，点击图像上的任意像素，可查看该点的详细诊断信息")

# 主画布
col1, col2, col3 = st.columns(3, gap="small")

# 智能分析区域
st.sidebar.markdown("---")
st.sidebar.header("🤖 智能分析")

if 'auto_optimized' not in st.session_state:
    st.session_state.auto_optimized = False

# 自动调参按钮（放在主区域上方）
auto_optimize = st.sidebar.button("⚡ 自动调参寻优", help="自动搜索最佳参数组合以提取最长裂纹")

def detect_substrate_bbox(img_gray):
    """自动检测基板边界

    方法：基于灰度分布找到基板（既不是纯白也不是纯黑的区域）
    """
    # 确保输入是 uint8
    if img_gray.dtype != np.uint8:
        img_gray = img_gray.astype(np.uint8)

    h, w = img_gray.shape

    # 1. 找到既不是纯白(>240)也不是纯黑(<15)的区域
    mask = (img_gray > 15) & (img_gray < 240)
    mask = mask.astype(np.uint8) * 255

    # 2. 形态学处理去除噪点，连接相邻区域
    kernel = np.ones((15, 15), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # 3. 找到最大轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # 4. 找最大轮廓（基板）
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)

    return (x, y, cw, ch)

def load_sample_data(batch_path, index):
    """加载样本数据"""
    # 图像目录
    temp_dir = os.path.join(batch_path, '温度')
    stress_dir = os.path.join(batch_path, '应力')
    d_hist_dir = os.path.join(batch_path, 'd_hist')

    # 获取文件列表
    temp_images = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')],
                         key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0])) or '0'))
    stress_images = sorted([f for f in os.listdir(stress_dir) if f.endswith('.png')],
                           key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0])) or '0'))

    has_d_hist = os.path.exists(d_hist_dir)
    d_hist_images = []
    if has_d_hist:
        d_hist_images = sorted([f for f in os.listdir(d_hist_dir) if f.endswith('.png')],
                              key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0])) or '0'))

    actual_n = min(len(temp_images), len(stress_images), n_samples)
    index = min(index, actual_n - 1)

    # 读取图像
    temp_path = os.path.join(temp_dir, temp_images[index])
    stress_path = os.path.join(stress_dir, stress_images[index])

    temp_img = imread_unicode(temp_path)
    stress_img = imread_unicode(stress_path)

    d_hist_img = None
    if has_d_hist and index < len(d_hist_images):
        d_hist_path = os.path.join(d_hist_dir, d_hist_images[index])
        d_hist_img = imread_unicode(d_hist_path)

    return temp_img, stress_img, d_hist_img, temp_images[index] if index < len(temp_images) else ""

def process_image(img, target_size=256, crop_ratio=0.7, center=None):
    """预处理图像

    Args:
        img: PIL Image
        target_size: 目标尺寸
        crop_ratio: 裁剪比例
        center: 可选，裁剪中心点 (cy, cx)，默认为图像中心
    """
    if img is None:
        return np.zeros((target_size, target_size))

    # 转为灰度
    img = img.convert('L')
    img_array = np.array(img, dtype=np.float32)

    # Max-Min 归一化
    img_min = img_array.min()
    img_max = img_array.max()
    if img_max - img_min > 1e-8:
        img_array = (img_array - img_min) / (img_max - img_min)

    # Center Crop（确保正方形，以指定中心为基准）
    h, w = img_array.shape
    crop_size = int(min(h, w) * crop_ratio)

    # 使用指定中心或图像中心
    if center is None:
        center_y, center_x = h // 2, w // 2
    else:
        center_y, center_x = center

    start_h = center_y - crop_size // 2
    start_w = center_x - crop_size // 2

    # 确保裁剪区域在图像范围内
    start_h = max(0, min(start_h, h - crop_size))
    start_w = max(0, min(start_w, w - crop_size))

    img_cropped = img_array[start_h:start_h+crop_size, start_w:start_w+crop_size]

    # Resize
    img_pil = Image.fromarray((img_cropped * 255).astype(np.uint8))
    img_resized = img_pil.resize((target_size, target_size), Image.BILINEAR)

    return np.array(img_resized, dtype=np.float32) / 255.0

def generate_mask(img_array, threshold=0.1, invert=True):
    """生成二值掩膜"""
    mask = img_array.copy()
    if invert:
        mask = 1.0 - mask
    mask_binary = (mask > threshold).astype(np.float32)
    return mask_binary

def display_image_mpl(ax, img_array, cmap='gray', title='', is_binary=False):
    """使用 matplotlib 在 axis 上显示图像

    Args:
        ax: matplotlib axis
        img_array: 图像数组
        cmap: 色图
        title: 标题
        is_binary: 是否为二值图像（使用最近邻插值）
    """
    interpolation = 'nearest' if is_binary else 'bilinear'
    ax.imshow(img_array, cmap=cmap, interpolation=interpolation)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    ax.set_aspect('equal')  # 保持宽高比


def create_plotly_image(img_array, cmap='gray', title='', zmin=None, zmax=None):
    """创建 Plotly 交互式图像（支持点击）

    Args:
        img_array: 图像数组 (H, W)
        cmap: 色图
        title: 标题
        zmin/zmax: 颜色范围

    Returns:
        plotly Figure
    """
    fig = go.Figure()
    fig.add_trace(go.Image(
        z=img_array,
        colorscale=cmap if cmap != 'gray' else 'Greys',
        zmin=zmin,
        zmax=zmax,
        hovertemplate='x: %{x}<br>y: %{y}<br>值: %{z:.4f}<extra></extra>'
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        dragmode='pan',
        margin=dict(l=0, r=0, t=30, b=0),
        height=350,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x'),
    )
    return fig


def diagnose_pixel(x, y, d_hist_array, crack_mask, pore_mask,
                  crack_threshold, pore_threshold, edge_exclusion, merge_distance):
    """诊断指定像素为何未被识别为裂纹/气孔

    Returns:
        dict: 诊断信息
    """
    h, w = d_hist_array.shape
    result = {
        'coords': (int(x), int(y)),
        'gray_value': float(d_hist_array[int(y), int(x)]) if 0 <= y < h and 0 <= x < w else None,
        'in_boundary': False,
        'crack_reason': '',
        'pore_reason': '',
        'connected_area': 0,
        'aspect_ratio': 0,
    }

    if result['gray_value'] is None:
        result['crack_reason'] = '❌ 坐标超出图像范围'
        return result

    gray = result['gray_value']

    # 检查是否在边界排除区
    if (x < edge_exclusion or x >= w - edge_exclusion or
        y < edge_exclusion or y >= h - edge_exclusion):
        result['in_boundary'] = True
        result['crack_reason'] = f'❌ 位于边界排除区 (< {edge_exclusion}px)'
        return result

    # 检查裂纹阈值（裂纹是暗色，灰度值低于阈值）
    if gray < crack_threshold:
        result['crack_reason'] = f'✅ 是裂纹（暗色 {gray:.3f} < {crack_threshold:.2f}）'
    else:
        result['crack_reason'] = f'❌ 不是裂纹（灰度 {gray:.3f} >= {crack_threshold:.2f}）'

    # 检查气孔阈值
    if gray > pore_threshold:
        result['pore_reason'] = f'✅ 高于气孔阈值 ({gray:.3f} > {pore_threshold:.2f})'
    else:
        result['pore_reason'] = f'❌ 低于气孔阈值 ({gray:.3f} <= {pore_threshold:.2f})'

    # 检查掩膜状态
    is_crack = crack_mask[int(y), int(x)] > 0.5 if 0 <= y < h and 0 <= x < w else False
    is_pore = pore_mask[int(y), int(x)] > 0.5 if 0 <= y < h and 0 <= x < w else False

    result['is_crack'] = is_crack
    result['is_pore'] = is_pore

    if is_crack:
        result['crack_reason'] = '✅ 该像素已被识别为裂纹'
    if is_pore:
        result['pore_reason'] = '✅ 该像素已被识别为气孔'

    return result

# 加载数据
with st.spinner("Loading sample..."):
    temp_img, stress_img, d_hist_img, filename = load_sample_data(batch_path, sample_index)

# 检测基板中心（用于裁剪）
substrate_center = None
if show_crop_box and temp_img is not None:
    # 检测基板边界
    img_gray = np.array(temp_img.convert('L'))
    bbox = detect_substrate_bbox(img_gray)
    if bbox is not None:
        bx, by, bw, bh = bbox
        # 基板中心
        substrate_center = (by + bh // 2, bx + bw // 2)
        # 更新裁剪比例滑块的提示
        st.session_state['substrate_center'] = substrate_center

# 处理图像（使用用户选择的分辨率和基板中心）
temp_array = process_image(temp_img, target_size, crop_ratio=crop_ratio, center=substrate_center)
stress_array = process_image(stress_img, target_size, crop_ratio=crop_ratio, center=substrate_center)

# 生成掩膜和结构分析
if d_hist_img is not None:
    d_hist_array = process_image(d_hist_img, target_size, crop_ratio=crop_ratio, center=substrate_center)
    mask_array = generate_mask(d_hist_array, 0.1, invert=True)

    # 使用优化后的参数（如果有）
    if st.session_state.get('auto_optimized') and 'optimized_params' in st.session_state:
        opt_p = st.session_state.optimized_params
        opt_crack_threshold = opt_p['crack_threshold']
        opt_merge_distance = opt_p['merge_distance']
        opt_crack_min_area = opt_p['crack_min_area']
        opt_edge_exclusion = opt_p.get('edge_exclusion', 25)
        param_source = "auto"  # 标记参数来源
    else:
        opt_crack_threshold = crack_threshold
        opt_merge_distance = merge_distance
        opt_crack_min_area = crack_min_area
        opt_edge_exclusion = edge_exclusion
        param_source = "manual"

    # 显示当前使用的参数（让用户清楚知道微调的基准）
    with st.expander("📊 当前处理参数", expanded=True):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.write(f"**裂纹阈值**: `{opt_crack_threshold:.2f}`")
            st.write(f"**合并距离**: `{opt_merge_distance}` px")
            st.write(f"**最小面积**: `{opt_crack_min_area}` px")
        with col_p2:
            st.write(f"**边界排除**: `{opt_edge_exclusion}` px")
            st.write(f"**参数来源**: {'🤖 自动优化' if param_source == 'auto' else '👆 手动滑块'}")
            if param_source == 'auto':
                st.write("_手动微调会覆盖优化结果_")

        # 重置按钮：恢复到自动优化参数
        if st.session_state.get('auto_optimized'):
            if st.button("🔄 恢复优化参数", key="reset_params"):
                # 保持 auto_optimized 状态，刷新页面
                st.rerun()

    # 结构分离分析（使用优化后的参数）
    crack_mask, pore_mask, overlay_combined, struct_stats = separate_crack_pore(
        d_hist_array,
        crack_threshold=opt_crack_threshold,
        pore_threshold=pore_threshold,
        crack_min_area=opt_crack_min_area,
        crack_min_aspect_ratio=crack_min_aspect,
        merge_distance=opt_merge_distance,
        crack_mode=crack_mode,
        edge_exclusion=opt_edge_exclusion
    )
else:
    d_hist_array = None
    mask_array = None
    crack_mask = None
    pore_mask = None
    overlay_combined = None
    struct_stats = None

# 处理自动调参请求
if auto_optimize and d_hist_array is not None:
    with st.spinner("正在自动寻优..."):
        progress_bar = st.progress(0)
        status_text = st.empty()

        result = auto_optimize_params(
            d_hist_array,
            min_crack_length=30.0
        )

        progress_bar.progress(100)

        if result['success']:
            st.session_state.auto_optimized = True
            st.session_state.optimized_params = result['best_params']
            st.session_state.optimized_stats = result['best_stats']

            st.success(result['message'])
            st.info(f"""
            **最优参数组合**:
            - 裂纹阈值: {result['best_params']['crack_threshold']:.2f}
            - 合并距离: {result['best_params']['merge_distance']}
            - 最小面积: {result['best_params']['crack_min_area']}
            - 边界排除: {result['best_params']['edge_exclusion']} px
            - 最长裂纹: {result['best_stats'].crack_max_length:.1f} px
            """)
        else:
            st.warning(result['message'])
elif auto_optimize and d_hist_array is None:
    st.warning("请先选择有 d_hist 数据的样本")

# 显示图像（使用 matplotlib 配合 st.pyplot）
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

def draw_crop_box(img_rgb, crop_ratio, auto_detect=True):
    """在图像上绘制裁剪框

    Args:
        img_rgb: RGB图像
        crop_ratio: 裁剪比例
        auto_detect: 是否自动检测基板中心
    """
    h, w = img_rgb.shape[:2]
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    if auto_detect:
        # 自动检测基板边界
        bbox = detect_substrate_bbox(img_gray)
        if bbox is not None:
            bx, by, bw, bh = bbox
            # 基板中心
            center_x = bx + bw // 2
            center_y = by + bh // 2
        else:
            # 回退到图像中心
            center_x = w // 2
            center_y = h // 2
    else:
        center_x = w // 2
        center_y = h // 2

    # 计算裁剪区域（以基板中心为基准，保持正方形）
    crop_size = int(min(h, w) * crop_ratio)
    offset_h = center_y - crop_size // 2
    offset_w = center_x - crop_size // 2

    # 绘制裁剪框（绿色）
    cv2.rectangle(img_rgb, (offset_w, offset_h),
                  (offset_w + crop_size, offset_h + crop_size),
                  (0, 255, 0), 3)

    # 标注
    mode_text = "Auto" if auto_detect else "Center"
    cv2.putText(img_rgb, f"Crop {crop_ratio*100:.0f}% [{mode_text}]",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 添加边角标记
    corner_size = 20
    cv2.line(img_rgb, (offset_w, offset_h), (offset_w + corner_size, offset_h), (255, 0, 0), 2)
    cv2.line(img_rgb, (offset_w, offset_h), (offset_w, offset_h + corner_size), (255, 0, 0), 2)
    cv2.line(img_rgb, (offset_w + crop_size, offset_h), (offset_w + crop_size - corner_size, offset_h), (255, 0, 0), 2)
    cv2.line(img_rgb, (offset_w + crop_size, offset_h), (offset_w + crop_size, offset_h + corner_size), (255, 0, 0), 2)
    cv2.line(img_rgb, (offset_w, offset_h + crop_size), (offset_w + corner_size, offset_h + crop_size), (255, 0, 0), 2)
    cv2.line(img_rgb, (offset_w, offset_h + crop_size), (offset_w, offset_h + crop_size - corner_size), (255, 0, 0), 2)
    cv2.line(img_rgb, (offset_w + crop_size, offset_h + crop_size), (offset_w + crop_size - corner_size, offset_h + crop_size), (255, 0, 0), 2)
    cv2.line(img_rgb, (offset_w + crop_size, offset_h + crop_size), (offset_w + crop_size, offset_h + crop_size - corner_size), (255, 0, 0), 2)

    return img_rgb

# 主图像：去掉列间距，填充整个宽度
with col1:
    st.markdown("### 🌡️ 温度场")
    if show_crop_box and temp_img is not None:
        # 显示原始图像 + 裁剪框
        img_temp_orig = np.array(temp_img.convert('L'))
        img_temp_display = cv2.cvtColor(img_temp_orig, cv2.COLOR_GRAY2RGB)
        img_temp_display = draw_crop_box(img_temp_display, crop_ratio)
        st.image(img_temp_display, caption="温度场（原始尺寸）", use_container_width=True)
    else:
        # 显示处理后的图像
        img_temp_display = (temp_array * 255).astype(np.uint8)
        if len(img_temp_display.shape) == 2:
            img_temp_display = cv2.cvtColor(img_temp_display, cv2.COLOR_GRAY2RGB)
        st.image(img_temp_display, caption="温度场", use_container_width=True)

with col2:
    st.markdown("### 💪 应力场")
    if show_crop_box and stress_img is not None:
        img_stress_orig = np.array(stress_img.convert('L'))
        img_stress_display = cv2.cvtColor(img_stress_orig, cv2.COLOR_GRAY2RGB)
        img_stress_display = draw_crop_box(img_stress_display, crop_ratio)
        st.image(img_stress_display, caption="应力场（原始尺寸）", use_container_width=True)
    else:
        img_stress_display = (stress_array * 255).astype(np.uint8)
        if len(img_stress_display.shape) == 2:
            img_stress_display = cv2.cvtColor(img_stress_display, cv2.COLOR_GRAY2RGB)
        st.image(img_stress_display, caption="应力场", use_container_width=True)

with col3:
    if d_hist_array is not None:
        st.markdown("### 📊 累积损伤场 (d_hist)")
        if show_crop_box and d_hist_img is not None:
            img_dhist_orig = np.array(d_hist_img.convert('L'))
            img_dhist_display = cv2.cvtColor(img_dhist_orig, cv2.COLOR_GRAY2RGB)
            img_dhist_display = draw_crop_box(img_dhist_display, crop_ratio)
            st.image(img_dhist_display, caption="d_hist（原始尺寸）", use_container_width=True)
        else:
            img_dhist_display = (d_hist_array * 255).astype(np.uint8)
            if len(img_dhist_display.shape) == 2:
                img_dhist_display = cv2.cvtColor(img_dhist_display, cv2.COLOR_GRAY2RGB)
            st.image(img_dhist_display, caption="累积损伤场", use_container_width=True)
    else:
        st.markdown("### 📊 累积损伤场 (d_hist)")
        st.info("无 d_hist 数据")

# 第二行：根据视图模式显示不同内容
st.markdown("---")

if view_mode == "二值掩膜":
    # 传统二值掩膜视图
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("### 🎯 二值掩膜")
        if mask_array is not None:
            fig, ax = plt.subplots(figsize=(5, 5))
            display_image_mpl(ax, mask_array, cmap='gray', title=f'Mask (T=0.1)')
            st.pyplot(fig, use_container_width=True)

            crack_ratio = mask_array.mean() * 100
            st.metric("损伤比例", f"{crack_ratio:.2f}%")

            if save_image:
                mask_uint8 = (mask_array * 255).astype(np.uint8)
                cv2.imwrite(f"output/mask_{batch_name}_{sample_index:04d}.png", mask_uint8)
                st.success(f"已保存")
        else:
            st.info("无数据")

    with col5:
        st.markdown("### 🔗 温度+掩膜叠层")
        if mask_array is not None:
            overlay = np.zeros((*temp_array.shape, 3), dtype=np.float32)
            gray = temp_array / (temp_array.max() + 1e-8)
            overlay[..., 0] = gray
            overlay[..., 1] = gray * 0.8
            overlay[..., 2] = gray * 0.6
            mask_bool = mask_array > 0.5
            overlay[mask_bool] = [1.0, 0.3, 0.2]

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(overlay, interpolation='bilinear')
            ax.set_title('Temp + Mask', fontsize=11, fontweight='bold')
            ax.axis('off')
            ax.set_aspect('equal')
            st.pyplot(fig, use_container_width=True)
        else:
            st.info("无数据")

    with col6:
        st.markdown("### 📈 灰度直方图")
        if d_hist_array is not None:
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.hist(d_hist_array.flatten(), bins=50, color='steelblue', edgecolor='white')
            ax.axvline(x=crack_threshold, color='red', linestyle='--', label=f'裂纹T={crack_threshold}')
            ax.axvline(x=pore_threshold, color='blue', linestyle='--', label=f'气孔T={pore_threshold}')
            ax.set_title('灰度分布', fontsize=11)
            ax.set_xlabel('灰度值')
            ax.set_ylabel('像素数量')
            ax.legend(fontsize=8)
            st.pyplot(fig, use_container_width=True)

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("最小值", f"{d_hist_array.min():.4f}")
                st.metric("最大值", f"{d_hist_array.max():.4f}")
            with col_b:
                st.metric("平均值", f"{d_hist_array.mean():.4f}")
                st.metric("标准差", f"{d_hist_array.std():.4f}")
        else:
            st.info("无数据")

elif view_mode == "结构分解视图":
    # 气孔与裂纹分离视图
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("### 🔴 裂纹掩膜")
        if crack_mask is not None:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(crack_mask, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f'裂纹 (T<{crack_threshold})', fontsize=11, fontweight='bold')
            ax.axis('off')
            ax.set_aspect('equal')
            st.pyplot(fig, use_container_width=True)

            if struct_stats:
                st.metric("裂纹数量", struct_stats.crack_count)
                st.metric("裂纹总面积", f"{struct_stats.crack_total_area:.0f} px")
                st.metric("最大长度", f"{struct_stats.crack_max_length:.1f} px")
        else:
            st.info("无数据")

    with col5:
        st.markdown("### 🔵 气孔掩膜")
        if pore_mask is not None:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(pore_mask, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
            ax.set_title(f'气孔 (T>{pore_threshold})', fontsize=11, fontweight='bold')
            ax.axis('off')
            ax.set_aspect('equal')
            st.pyplot(fig, use_container_width=True)

            if struct_stats:
                st.metric("气孔数量", struct_stats.pore_count)
                st.metric("气孔总面积", f"{struct_stats.pore_total_area:.0f} px")
                st.metric("平均直径", f"{struct_stats.pore_avg_diameter:.1f} px")
        else:
            st.info("无数据")

    with col6:
        st.markdown("### 🟡 结构分解叠加图")
        if overlay_combined is not None:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(overlay_combined, interpolation='bilinear')
            ax.set_title('裂纹(红) + 气孔(蓝)', fontsize=11, fontweight='bold')
            ax.axis('off')
            ax.set_aspect('equal')

            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', label=f'裂纹 ({struct_stats.crack_count})'),
                Patch(facecolor='blue', label=f'气孔 ({struct_stats.pore_count})'),
                Patch(facecolor='yellow', label='重叠区域')
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=8)

            st.pyplot(fig, use_container_width=True)

            if save_image:
                overlay_uint8 = (overlay_combined * 255).astype(np.uint8)
                cv2.imwrite(f"output/overlay_{batch_name}_{sample_index:04d}.png", overlay_uint8)
                st.success(f"已保存")
        else:
            st.info("无数据")

    # 第三行：裂纹与气孔的起源分析
    st.markdown("---")
    st.markdown("### 🔬 物理起源分析")

    col7, col8 = st.columns(2)

    with col7:
        st.markdown("#### 裂纹与温度场叠层")
        if crack_mask is not None:
            overlay = np.zeros((*temp_array.shape, 3), dtype=np.float32)
            gray = temp_array / (temp_array.max() + 1e-8)
            overlay[..., 0] = gray
            overlay[..., 1] = gray * 0.8
            overlay[..., 2] = gray * 0.6

            crack_bool = crack_mask > 0.5
            overlay[crack_bool] = [1.0, 0.2, 0.2]

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(overlay, interpolation='bilinear')
            ax.set_title('温度场 + 裂纹分布', fontsize=11, fontweight='bold')
            ax.axis('off')
            ax.set_aspect('equal')
            st.pyplot(fig, use_container_width=True)

            st.caption("红色区域：裂纹暗带 | 灰色背景：温度梯度")
        else:
            st.info("无数据")

    with col8:
        st.markdown("#### 气孔与温度场叠层")
        if pore_mask is not None:
            overlay = np.zeros((*temp_array.shape, 3), dtype=np.float32)
            gray = temp_array / (temp_array.max() + 1e-8)
            overlay[..., 0] = gray
            overlay[..., 1] = gray * 0.8
            overlay[..., 2] = gray * 0.6

            pore_bool = pore_mask > 0.5
            overlay[pore_bool] = [0.2, 0.2, 1.0]

            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(overlay, interpolation='bilinear')
            ax.set_title('温度场 + 气孔分布', fontsize=11, fontweight='bold')
            ax.axis('off')
            ax.set_aspect('equal')
            st.pyplot(fig, use_container_width=True)

            st.caption("蓝色区域：气孔亮点 | 灰色背景：温度梯度")
        else:
            st.info("无数据")

else:  # 对比视图
    col4, col5, col6 = st.columns(3)

    with col4:
        st.markdown("### 🎯 传统二值掩膜")
        if mask_array is not None:
            fig, ax = plt.subplots(figsize=(5, 5))
            display_image_mpl(ax, mask_array, cmap='gray', title='Binary Mask')
            st.pyplot(fig, use_container_width=True)
            st.metric("损伤比例", f"{mask_array.mean()*100:.2f}%")
        else:
            st.info("无数据")

    with col5:
        st.markdown("### ⚡ 结构分解")
        if overlay_combined is not None:
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(overlay_combined, interpolation='bilinear')
            ax.set_title('Crack(Red) + Pore(Blue)', fontsize=11, fontweight='bold')
            ax.axis('off')
            ax.set_aspect('equal')
            st.pyplot(fig, use_container_width=True)

            if struct_stats:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("裂纹", struct_stats.crack_count)
                with col_b:
                    st.metric("气孔", struct_stats.pore_count)
        else:
            st.info("无数据")

    with col6:
        st.markdown("### 📊 结构统计")
        if struct_stats:
            st.markdown("**裂纹特征**")
            st.write(f"- 数量: {struct_stats.crack_count}")
            st.write(f"- 总面积: {struct_stats.crack_total_area:.0f} px")
            st.write(f"- 平均面积: {struct_stats.crack_avg_area:.1f} px")
            st.write(f"- 最大长度: {struct_stats.crack_max_length:.1f} px")

            st.markdown("**气孔特征**")
            st.write(f"- 数量: {struct_stats.pore_count}")
            st.write(f"- 总面积: {struct_stats.pore_total_area:.0f} px")
            st.write(f"- 平均面积: {struct_stats.pore_avg_area:.1f} px")
            st.write(f"- 平均直径: {struct_stats.pore_avg_diameter:.1f} px")

            # 物理分析提示
            if struct_stats.crack_count > 0 and struct_stats.pore_count > 0:
                st.markdown("**💡 物理提示**")
                st.info("观察裂纹是否从气孔附近萌生，或气孔是否成为裂纹扩展的终止点")
        else:
            st.info("无统计数据")

# 等值线去除（可选）
if show_contour and d_hist_img is not None:
    st.markdown("---")
    st.markdown("### 🔧 等值线去除对比")

    col7, col8 = st.columns(2)

    # 原始图像
    with col7:
        st.markdown("**原始图像**")
        original = np.array(d_hist_img)
        if len(original.shape) == 3:
            original = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.imshow(original, cmap='gray', interpolation='bilinear')
        ax.set_title('原始 d_hist', fontsize=11, fontweight='bold')
        ax.axis('off')
        ax.set_aspect('equal')
        st.pyplot(fig, use_container_width=True)

    # 等值线去除
    with col8:
        st.markdown("**等值线去除后**")
        try:
            contour_remover = ContourRemover.create_for_batch4()
            cleaned = contour_remover(original)
            if len(cleaned.shape) == 3:
                cleaned = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.imshow(cleaned, cmap='gray', interpolation='bilinear')
            ax.set_title('等值线去除后', fontsize=11, fontweight='bold')
            ax.axis('off')
            ax.set_aspect('equal')
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.error(f"等值线去除失败: {e}")

# 页脚信息
st.markdown("---")

# ============================================================
# 交互式诊断面板（点击图像查看像素详情）
# ============================================================
if enable_click_diagnose and d_hist_array is not None:
    st.markdown("### 🔬 像素级诊断面板")

    # 初始化点击坐标
    if 'click_x' not in st.session_state:
        st.session_state.click_x = None
    if 'click_y' not in st.session_state:
        st.session_state.click_y = None

    # 布局：左侧大图 + 右侧输入
    col_diag1, col_diag2 = st.columns([3, 1])

    # 初始化 session state
    if 'click_x' not in st.session_state:
        st.session_state.click_x = None
    if 'click_y' not in st.session_state:
        st.session_state.click_y = None

    h, w = d_hist_array.shape

    with col_diag1:
        st.markdown("**📊 d_hist 图像（悬停查看坐标）**")

        # 只用一个大型 Plotly 图像（悬停 + 十字标记）
        cmap = 'Viridis' if selected_cmap == 'viridis' else 'Greys'
        fig = px.imshow(d_hist_array, color_continuous_scale=cmap, origin='upper')

        # 如果有选中点，添加十字标记
        if st.session_state.click_x is not None and st.session_state.click_y is not None:
            cx, cy = st.session_state.click_x, st.session_state.click_y
            fig.add_trace(go.Scatter(
                x=[cx], y=[cy], mode='markers',
                marker=dict(color='cyan', size=15, symbol='x-thin'),
                showlegend=False
            ))
            fig.add_hline(y=cy, line_color='cyan', line_dash='dash', opacity=0.7)
            fig.add_vline(x=cx, line_color='cyan', line_dash='dash', opacity=0.7)

        fig.update_layout(
            height=600,
            margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       fixedrange=True, scaleanchor='y', scaleratio=1),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                       fixedrange=True),
        )
        fig.update_traces(hovertemplate='x: %{x}<br>y: %{y}<br>灰度: %{z:.4f}<extra></extra>')

        st.plotly_chart(fig, use_container_width=True, key="dhist_interactive")

    with col_diag2:
        st.markdown("**📍 输入像素坐标**")
        # 坐标输入
        h, w = d_hist_array.shape
        col_x, col_y = st.columns(2)
        with col_x:
            input_x = st.number_input("X 坐标", 0, w-1, 100, 1, key="input_x")
        with col_y:
            input_y = st.number_input("Y 坐标", 0, h-1, 100, 1, key="input_y")

        # 点击诊断按钮
        if st.button("🔍 执行诊断", use_container_width=True):
            st.session_state.click_x = int(input_x)
            st.session_state.click_y = int(input_y)

        # 重置按钮
        if st.button("🔄 重置", use_container_width=True):
            st.session_state.click_x = None
            st.session_state.click_y = None

        st.markdown("---")

        # 显示诊断结果
        if st.session_state.click_x is not None and st.session_state.click_y is not None:
            x, y = st.session_state.click_x, st.session_state.click_y

            # 执行诊断
            diag = diagnose_pixel(
                x, y, d_hist_array, crack_mask, pore_mask,
                opt_crack_threshold, pore_threshold, opt_edge_exclusion, opt_merge_distance
            )

            # ========== 最终判定结果 ==========
            if diag['is_crack']:
                st.success("🔴 **该像素被识别为：裂纹**")
            elif diag['is_pore']:
                st.success("🔵 **该像素被识别为：气孔**")
            elif diag.get('in_boundary', False):
                st.warning("⚠️ **该像素位于边界排除区，未参与检测**")
            else:
                st.info("⚪ **该像素未被识别为裂纹或气孔（中性区域）**")

            st.markdown("---")

            # 详细信息
            st.write(f"📍 **坐标**: ({x}, {y})")
            st.write(f"📊 **灰度值**: {diag['gray_value']:.4f}")

            # 简单说明
            if not diag['is_crack'] and not diag['is_pore']:
                gray = diag.get('gray_value', 0)
                if not diag.get('in_boundary', False):
                    if gray < opt_crack_threshold:
                        st.caption(f"💡 灰度 {gray:.3f} < 裂纹阈值 {opt_crack_threshold}，但可能被形态学处理过滤")
                    elif gray > pore_threshold:
                        st.caption(f"💡 灰度 {gray:.3f} > 气孔阈值 {pore_threshold}，但可能被形态学处理过滤")
                    else:
                        st.caption(f"💡 灰度在裂纹阈值和气孔阈值之间，为中性区域")
        else:
            st.info("输入坐标后点击「执行诊断」")
else:
    st.markdown("### 🔬 像素级诊断面板")
    st.info("在侧边栏勾选「启用点击诊断模式」后使用")


# 物理时间信息（醒目显示）
if time_display:
    st.markdown(f"""
    <div style="background-color: #f0f0f0; padding: 10px; border-radius: 5px; text-align: center;">
        <h3 style="margin: 0; color: #333;">⏱️ 物理时间信息</h3>
        <p style="margin: 5px 0 0 0; font-size: 18px; font-weight: bold; color: #0066cc;">{time_display}</p>
    </div>
    """, unsafe_allow_html=True)

st.caption(f"批次: {batch_name} | 样本索引: {sample_index:04d} | 文件: {filename}")
