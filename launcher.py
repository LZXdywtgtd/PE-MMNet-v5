"""
PE-MMNet v4 交互式训练启动器

功能：
1. 预设模板快速启动
2. 交互式自定义配置，逐步了解每个参数的作用
3. 动态架构建议（根据选择自动推荐学习率/Dropout）
4. 配置保存/加载
5. 消融实验预设

用法：
    python train_launcher.py               # 交互式（统一入口）
    python train_launcher.py --quick ...   # 快捷批量训练
"""

import os
import sys
import json

# Windows 控制台编码修复
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass  # 忽略编码修复失败

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


# =============================================================================
# 预设模板
# =============================================================================

PRESETS = {
    "1": {
        "name": "CNN 基线 (ResNet18 + CNN-Attn + Cross-Attn)",
        "description": "默认配置，稳定可靠，训练快",
        "args": "--backbone_2d resnet18 --backbone_1d cnn_attn --fusion cross_attn",
    },
    "2": {
        "name": "ViT + Transformer + Adaptive",
        "description": "纯 Transformer 架构，表达能力强，训练较慢",
        "args": "--backbone_2d vit_small --backbone_1d transformer --fusion adaptive",
    },
    "3": {
        "name": "ResNet + DLinear + Concat",
        "description": "轻量级基线，简单有效",
        "args": "--backbone_2d resnet18 --backbone_1d dlinear --fusion concat",
    },
    "4": {
        "name": "ViT + CNN-Attn",
        "description": "混合架构：ViT 视觉 + CNN 时序",
        "args": "--backbone_2d vit_small --backbone_1d cnn_attn --fusion cross_attn",
    },
}

# 分割任务预设
SEGMENTATION_PRESETS = {
    "s1": {
        "name": "分割 CNN 基线",
        "description": "ResNet18 + CNN-Attn，掩膜分割任务",
        "args": "--backbone_2d resnet18 --backbone_1d cnn_attn --fusion cross_attn --task segmentation",
    },
    "s2": {
        "name": "分割 ViT 基线",
        "description": "ViT + Transformer，掩膜分割任务",
        "args": "--backbone_2d vit_small --backbone_1d transformer --fusion adaptive --task segmentation",
    },
}

# 多任务预设
MULTITASK_PRESETS = {
    "m1": {
        "name": "多任务 CNN 基线",
        "description": "同时输出检测框 + 掩膜",
        "args": "--backbone_2d resnet18 --backbone_1d cnn_attn --fusion cross_attn --task multitask",
    },
}

# 消融实验：使用 --variant（已在 run_train.py 完整实现）
ABLATION_PRESETS = {
    # 基础变体
    "a1": {"name": "仅2D分支 (ResNet18)", "args": "--variant 2d_only"},
    "a2": {"name": "仅1D分支 (CNN-Attn)", "args": "--variant 1d_only"},
    "a3": {"name": "双分支拼接", "args": "--variant concat"},
    "a4": {"name": "双分支加法", "args": "--variant add"},
    "a5": {"name": "Cross-Attention", "args": "--variant cross_attn"},
    "a6": {"name": "ResNet18基线", "args": "--variant resnet18"},

    # YOLO-FPN 变体
    "b1": {"name": "Swin-YOLO", "args": "--variant swin_yolo --lr 1e-4"},
    "b2": {"name": "ViT-YOLO", "args": "--variant vit_yolo --lr 1e-4"},
    "b3": {"name": "DETR", "args": "--variant detr --lr 1e-4"},

    # PatchTST 变体
    "c1": {"name": "Swin-PatchTST", "args": "--variant swin_patchtst --lr 1e-4"},
    "c2": {"name": "ViT-PatchTST", "args": "--variant vit_patchtst --lr 1e-4"},

    # 高级优化组合
    "d1": {"name": "ResNet18 + 门控融合", "args": "--variant resnet18 --fusion gated"},
    "d2": {"name": "ResNet18 + 坐标注意力", "args": "--variant resnet18 --use_coord_attn"},
    "d3": {"name": "ResNet18 + 三通道输入", "args": "--variant resnet18 --triple_channel"},
    "d4": {"name": "分阶段训练", "args": "--variant resnet18 --staged_train"},

    # 团队协作快捷预设
    "team_base": {"name": "【团队】基线模型", "args": "--variant resnet18 --epochs 100"},
    "team_yolo": {"name": "【团队】YOLO系列", "args": "--variant swin_yolo --epochs 100 --lr 1e-4"},
    "team_opt": {"name": "【团队】优化组合", "args": "--variant resnet18 --fusion gated --use_coord_attn --epochs 100"},
}

# 团队协作配置模板
TEAM_TEMPLATES = {
    "quick_baseline": {
        "name": "快速基线",
        "args": "--variant resnet18 --epochs 50 --batch_size 8",
        "time_estimate": "~30分钟"
    },
    "quick_yolo": {
        "name": "快速YOLO",
        "args": "--variant swin_yolo --epochs 50 --lr 1e-4 --batch_size 4",
        "time_estimate": "~40分钟"
    },
    "full_train": {
        "name": "完整训练",
        "args": "--variant resnet18 --epochs 200 --batch_size 16",
        "time_estimate": "~2小时"
    },
    "ablation_small": {
        "name": "消融实验(小)",
        "args": "--variant resnet18 --epochs 30 --batch_size 8",
        "time_estimate": "~15分钟"
    },
}


# =============================================================================
# 工具函数
# =============================================================================

def get_arch_suggestions(backbone_2d='resnet18', backbone_1d='cnn_attn'):
    """根据架构组合返回推荐的超参数"""
    suggestions = {
        'lr': 0.001,
        'dropout': 0.2,
        'feature_len': 300,
        'seq_interp_mode': 'interpolate',  # 修复：seq_mode -> seq_interp_mode
    }
    if backbone_2d == 'vit_small':
        suggestions['lr'] = 0.0001
        suggestions['dropout'] = max(suggestions['dropout'], 0.2)
    if backbone_1d in ['transformer', 'dlinear']:
        suggestions['lr'] = min(suggestions['lr'], 0.0001)
        suggestions['dropout'] = max(suggestions['dropout'], 0.2)
    if backbone_2d == 'vit_small' and backbone_1d == 'transformer':
        suggestions['dropout'] = max(suggestions['dropout'], 0.3)
    return suggestions


def validate_args(args_list):
    """校验参数互斥性"""
    has_variant = any('--variant' in arg for arg in args_list)
    has_backbone = any('--backbone' in arg for arg in args_list)
    if has_variant and has_backbone:
        raise ValueError("消融实验 (--variant) 与架构配置 (--backbone_*) 互斥")


def export_team_configs(filepath='team_configs.txt'):
    """导出团队协作配置文件

    生成一个简单的配置文件，团队成员可以直接复制使用。

    Args:
        filepath: 输出文件路径
    """
    lines = [
        "=" * 60,
        "PE-MMNet v4 团队协作配置",
        "=" * 60,
        "",
        "# 使用方法：将下面的命令复制到命令行执行",
        "# 或使用: python run_train.py --mode train <粘贴参数>",
        "",
        "# 快速基线训练 (~30分钟)",
        "# 适用场景：快速验证代码能跑通",
        "python run_train.py --mode train --variant resnet18 --epochs 50 --batch_size 8",
        "",
        "# YOLO系列训练 (~40分钟/模型)",
        "# 适用场景：需要更好的空间定位能力",
        "python run_train.py --mode train --variant swin_yolo --epochs 100 --lr 1e-4",
        "python run_train.py --mode train --variant vit_yolo --epochs 100 --lr 1e-4",
        "python run_train.py --mode train --variant detr --epochs 100 --lr 1e-4",
        "",
        "# PatchTST系列训练 (~40分钟/模型)",
        "# 适用场景：需要更好的时序建模能力",
        "python run_train.py --mode train --variant swin_patchtst --epochs 100 --lr 1e-4",
        "",
        "# 优化组合训练 (~1小时)",
        "# 适用场景：追求最佳性能",
        "python run_train.py --mode train --variant resnet18 --fusion gated --use_coord_attn --epochs 100",
        "",
        "# 分阶段训练 (~1.5小时)",
        "# 适用场景：长序列数据训练",
        "python run_train.py --mode train --variant resnet18 --staged_train --epochs 100",
        "",
        "# 三通道输入训练 (~1小时)",
        "# 适用场景：需要温度变化率信息",
        "python run_train.py --mode train --variant resnet18 --triple_channel --epochs 100",
        "",
        "# 完整训练 (~2小时)",
        "# 适用场景：最终交付模型",
        "python run_train.py --mode train --variant resnet18 --epochs 200 --batch_size 16",
        "",
        "# 消融实验快捷键（运行train_launcher.py后选择）",
        "# a6: 基线 | b1-b3: YOLO系列 | c1-c2: PatchTST系列 | d1-d4: 优化组合",
        "",
        "=" * 60,
    ]

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"团队配置已导出到: {filepath}")
    print("团队成员可以直接复制使用!")


def build_command(args_list, is_ablation=False):
    """构建完整命令

    Args:
        args_list: 参数列表
        is_ablation: 是否为消融实验

    Returns:
        完整的命令字符串，自动选择正确的 Python 解释器
    """
    import subprocess
    validate_args(args_list)
    mode = "ablation" if is_ablation else "train"

    # 检测正确的 Python 解释器
    # 优先使用 py Launcher（Windows 官方 Python Launcher）
    python_cmd = "py"
    try:
        result = subprocess.run(
            ["py", "-3", "--version"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and "Python" in result.stdout:
            python_cmd = "py"
        else:
            # 降级为 python
            python_cmd = "python"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # 尝试使用 python
        python_cmd = "python"

    return f"{python_cmd} run_train.py --mode {mode} {' '.join(args_list)}"


def save_config(config_list, filepath='config_launcher.json'):
    """保存配置到 JSON"""
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config_list, f, indent=2, ensure_ascii=False)


def load_command_config(filepath='config_launcher.json'):
    """从 JSON 加载命令配置"""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None


# =============================================================================
# 参数说明字典
# =============================================================================

PARAM_HELP = {
    'backbone_2d': """
  ┌─────────────────────────────────────────────────────────────────┐
  │ 2D视觉骨干网络 - 负责从温度/应力图像中提取空间特征               │
  │                                                                 │
  │ resnet18 (CNN):                                                │
  │   - 优点: 11M参数，训练快，显存占用低，稳定性好                  │
  │   - 缺点: 表达能力有限，可能错过细节                             │
  │   - 适用: 快速实验、显存受限、追求稳定性                         │
  │                                                                 │
  │ vit_small (Transformer):                                       │
  │   - 优点: 21M参数，表达能力强，能捕捉全局特征                    │
  │   - 缺点: 训练慢，显存占用高，需要更多数据                       │
  │   - 适用: 数据充足、追求最佳精度、长时训练                       │
  └─────────────────────────────────────────────────────────────────┘
""",
    'backbone_1d': """
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1D时序骨干网络 - 负责从温度时序数据中提取时间特征               │
  │                                                                 │
  │ cnn_attn (CNN+注意力):                                         │
  │   - 优点: CNN捕捉局部模式 + 注意力捕捉长程依赖，当前最佳         │
  │   - 缺点: 略复杂                                                │
  │   - 适用: 通用场景                                               │
  │                                                                 │
  │ transformer (纯注意力):                                        │
  │   - 优点: 强大的序列建模能力                                    │
  │   - 缺点: 需要更多数据，容易过拟合                              │
  │   - 适用: 长序列、数据充足                                       │
  │                                                                 │
  │ dlinear (轻量线性):                                            │
  │   - 优点: 极简、极快、显存占用低                                │
  │   - 缺点: 表达能力有限                                          │
  │   - 适用: 快速基线、显存极度受限                                 │
  └─────────────────────────────────────────────────────────────────┘
""",
    'fusion': """
  ┌─────────────────────────────────────────────────────────────────┐
  │ 多模态融合策略 - 决定2D图像特征和1D时序特征如何结合             │
  │                                                                 │
  │ cross_attn (交叉注意力) [推荐]:                                 │
  │   - 优点: 图像和时序互相"看"对方，学习它们之间的关联             │
  │   - 缺点: 计算量稍大                                            │
  │   - 适用: 追求最佳精度                                           │
  │                                                                 │
  │ concat (直接拼接):                                             │
  │   - 优点: 简单直接，计算快                                      │
  │   - 缺点: 融合能力有限                                          │
  │   - 适用: 快速实验                                               │
  │                                                                 │
  │ adaptive (自适应加权):                                         │
  │   - 优点: 网络自动学习图像和时序的权重                           │
  │   - 缺点: 可能需要更多训练数据                                   │
  │   - 适用: 中等规模数据                                          │
  └─────────────────────────────────────────────────────────────────┘
""",
    'feature_len': """
  ┌─────────────────────────────────────────────────────────────────┐
  │ 1D序列特征长度 - 温度时序数据的采样点数                          │
  │                                                                 │
  │ 100: 粗采样，速度快，可能丢失细节                                │
  │ 300: 平衡选择，保留足够细节                                      │
  │ 500: 精细采样，保留更多细节，训练较慢                            │
  │ 1000: 极高精度，仅在数据充足时使用                               │
  │                                                                 │
  │ 建议: 默认300即可，序列越长信息越丰富但训练越慢                  │
  └─────────────────────────────────────────────────────────────────┘
""",
    'seq_interp_mode': """
  ┌─────────────────────────────────────────────────────────────────┐
  │ 长序列处理模式 - 如何处理从CSV到指定长度的插值                   │
  │                                                                 │
  │ interpolate (线性插值) [推荐]:                                  │
  │   - 优点: 保留原始信号的微小变化和突变                           │
  │   - 缺点: 可能放大噪声                                          │
  │   - 适用: 关注细节变化的场景                                     │
  │                                                                 │
  │ pool (自适应池化):                                             │
  │   - 优点: 平滑噪声，训练更稳定                                  │
  │   - 缺点: 可能丢失重要细节                                      │
  │   - 适用: 噪声较多、追求训练稳定性                               │
  └─────────────────────────────────────────────────────────────────┘
""",
    'remove_contours': """
  ┌─────────────────────────────────────────────────────────────────┐
  │ 等值线去除 - 移除图像中的等值线干扰                              │
  │                                                                 │
  │ 等值线是温度场可视化时的线条，可能干扰模型识别                   │
  │                                                                 │
  │ 启用: 自动检测并移除等值线                                       │
  │ 禁用: 保留原始图像                                               │
  │                                                                 │
  │ 建议: 参数化扫描4建议启用，其他数据集通常不需要                   │
  └─────────────────────────────────────────────────────────────────┘
""",
    'epochs': """
  ┌─────────────────────────────────────────────────────────────────┐
  │ 训练轮数 - 完整遍历训练集的次数                                   │
  │                                                                 │
  │ 较少的轮数可能欠拟合（没学到足够知识）                           │
  │ 过多的轮数可能过拟合（死记硬背，泛化差）                         │
  │                                                                 │
  │ 建议:                                                          │
  │   - 100: 快速实验                                               │
  │   - 150: 标准训练                                               │
  │   - 200+: 充分训练（有早停机制，不会无限训练）                   │
  │                                                                 │
  │ 提示: 系统有早停机制(patience=30)，验证损失不降会自动停止        │
  └─────────────────────────────────────────────────────────────────┘
""",
    'lr': """
  ┌─────────────────────────────────────────────────────────────────┐
  │ 学习率 - 控制模型参数更新的步幅                                  │
  │                                                                 │
  │ 学习率太高: 训练震荡，可能不收敛                                  │
  │ 学习率太低: 训练太慢，可能陷入局部最优                            │
  │                                                                 │
  │ 架构推荐学习率:                                                  │
  │   - CNN架构 (resnet18): 0.001 (1e-3)                            │
  │   - Transformer架构: 0.0001 (1e-4)，更小的学习率                │
  │                                                                 │
  │ 提示: 交互式配置会根据您选择的架构自动推荐学习率                 │
  └─────────────────────────────────────────────────────────────────┘
""",
    'dropout': """
  ┌─────────────────────────────────────────────────────────────────┐
  │ Dropout - 训练时随机丢弃部分神经元，防止过拟合                   │
  │                                                                 │
  │ Dropout太高 (如0.5+): 欠拟合风险，训练变慢                       │
  │ Dropout太低 (如0.1-): 过拟合风险，泛化差                         │
  │                                                                 │
  │ 架构推荐Dropout:                                                │
  │   - CNN架构: 0.1~0.3                                            │
  │   - Transformer架构: 0.2~0.4（更容易过拟合）                    │
  │                                                                 │
  │ 建议: 默认0.2，数据少时可提高到0.3                               │
  └─────────────────────────────────────────────────────────────────┘
""",
    'predict_offset': """
  ┌─────────────────────────────────────────────────────────────────┐
  │ 预测偏移步数 - 预测未来时刻的裂纹状态                             │
  │                                                                 │
  │ offset=0: 预测当前时刻的裂纹位置                                  │
  │ offset=1: 预测 0.05秒 后（1步）的裂纹位置                        │
  │ offset=2: 预测 0.1秒 后（2步）的裂纹位置                         │
  │ ...以此类推                                                     │
  │                                                                 │
  │ 用途:                                                          │
  │   - offset=0: 标准训练/评估                                      │
  │   - offset>0: 预测裂纹扩展趋势                                   │
  │                                                                 │
  │ 建议: 默认0，有预测需求时可尝试1或2                              │
  └─────────────────────────────────────────────────────────────────┘
""",
    'task': """
  ┌─────────────────────────────────────────────────────────────────┐
  │ 任务模式 - 决定模型的输出形式                                    │
  │                                                                 │
  │ detection (检测) [默认]:                                        │
  │   - 输出: [x, y, l, w, confidence, density]                     │
  │   - x,y: 裂纹中心坐标  l,w: 长宽  confidence: 存在置信度        │
  │   - density: 裂纹密度（物理量）                                  │
  │                                                                 │
  │ segmentation (分割):                                           │
  │   - 输出: 像素级掩膜，标识每个像素是否属于裂纹                   │
  │   - 适用: 需要精确定位裂纹边界时                                 │
  │                                                                 │
  │ multitask (多任务):                                            │
  │   - 输出: 同时输出检测框+分割掩膜                                │
  │   - 适用: 既要知道位置，又要知道形状                             │
  └─────────────────────────────────────────────────────────────────┘
""",
}


def show_param_help(param_name):
    """显示参数说明"""
    if param_name in PARAM_HELP:
        print(PARAM_HELP[param_name])


def interactive_mode():
    """交互式自定义配置向导（带详细说明）"""
    print("\n" + "=" * 70)
    print("  自定义配置向导 - 按 ? 查看参数说明")
    print("=" * 70 + "\n")

    args_list = []
    backbone_2d = 'resnet18'
    backbone_1d = 'cnn_attn'

    # ===== 1. 任务类型 =====
    print("【1/9】任务模式")
    print("  [1] detection  - 检测裂纹位置和属性（默认）")
    print("  [2] segmentation - 像素级裂纹分割")
    print("  [3] multitask  - 检测+分割同时进行\n")
    task_choice = input("  选择 (默认1/?查看): ").strip() or "1"
    if task_choice == '?':
        show_param_help('task')
        task_choice = input("  选择 (默认1): ").strip() or "1"
    task_map = {"2": "segmentation", "3": "multitask"}
    task = task_map.get(task_choice, "detection")
    args_list.append(f"--task {task}")

    # ===== 2. 2D 骨干网络 =====
    print("\n【2/9】2D视觉骨干网络")
    print("  [1] resnet18   - CNN架构，11M参数，稳定可靠（默认）")
    print("  [2] vit_small  - Transformer，21M参数，表达能力强\n")
    choice = input("  选择 (默认1/?查看): ").strip() or "1"
    if choice == '?':
        show_param_help('backbone_2d')
        choice = input("  选择 (默认1): ").strip() or "1"
    backbone_2d = 'vit_small' if choice == "2" else 'resnet18'
    args_list.append(f"--backbone_2d {backbone_2d}")

    # ===== 3. 1D 骨干网络 =====
    print("\n【3/9】1D时序骨干网络")
    print("  [1] cnn_attn    - CNN+注意力，当前最佳（默认）")
    print("  [2] transformer - 纯Transformer，适合长序列")
    print("  [3] dlinear     - 轻量级线性模型\n")
    choice = input("  选择 (默认1/?查看): ").strip() or "1"
    if choice == '?':
        show_param_help('backbone_1d')
        choice = input("  选择 (默认1): ").strip() or "1"
    backbone_1d = {'2': 'transformer', '3': 'dlinear'}.get(choice, 'cnn_attn')
    args_list.append(f"--backbone_1d {backbone_1d}")

    # ===== 4. 融合策略 =====
    print("\n【4/9】多模态融合策略")
    print("  [1] cross_attn - 交叉注意力，特征交互最充分（默认）")
    print("  [2] concat     - 直接拼接，简单有效")
    print("  [3] adaptive   - 自适应加权\n")
    choice = input("  选择 (默认1/?查看): ").strip() or "1"
    if choice == '?':
        show_param_help('fusion')
        choice = input("  选择 (默认1): ").strip() or "1"
    fusion = {'2': 'concat', '3': 'adaptive'}.get(choice, 'cross_attn')
    args_list.append(f"--fusion {fusion}")

    suggestions = get_arch_suggestions(backbone_2d, backbone_1d)

    # ===== 5. 序列长度 =====
    print("\n【5/9】1D序列特征长度")
    print(f"  推荐: {suggestions['feature_len']}")
    print("  [1] 100  [2] 300 (默认)  [3] 500  [4] 1000\n")
    choice = input("  选择 (默认2/?查看): ").strip() or "2"
    if choice == '?':
        show_param_help('feature_len')
        choice = input("  选择 (默认2): ").strip() or "2"
    feature_len = {'1': '100', '3': '500', '4': '1000'}.get(choice, '300')
    args_list.append(f"--feature_len {feature_len}")

    # ===== 6. 序列处理模式 =====
    print("\n【6/9】长序列处理模式")
    print("  [1] interpolate - 线性插值，保留变化细节（默认）")
    print("  [2] pool        - 自适应池化，平滑噪声\n")
    choice = input("  选择 (默认1/?查看): ").strip() or "1"
    if choice == '?':
        show_param_help('seq_interp_mode')
        choice = input("  选择 (默认1): ").strip() or "1"
    if choice == "2":
        args_list.append("--seq_interp_mode pool")

    # ===== 7. 数据处理 =====
    print("\n【7/9】数据处理选项")
    choice = input("  启用等值线去除? (y/N/?): ").strip().lower()
    if choice == '?':
        show_param_help('remove_contours')
        choice = input("  启用等值线去除? (y/N): ").strip().lower()
    if choice == 'y':
        args_list.append("--remove_contours")

    # ===== 8. 训练参数 =====
    print("\n【8/9】训练参数")
    print(f"  架构推荐: 学习率={suggestions['lr']}, Dropout={suggestions['dropout']}\n")

    choice = input("  训练轮数 (默认150/?查看): ").strip() or "150"
    if choice == '?':
        show_param_help('epochs')
        choice = input("  训练轮数 (默认150): ").strip() or "150"
    epochs = choice
    args_list.append(f"--epochs {epochs}")

    choice = input(f"  学习率 (建议{suggestions['lr']}/?:查看): ").strip() or str(suggestions['lr'])
    if choice == '?':
        show_param_help('lr')
        choice = input(f"  学习率 (建议{suggestions['lr']}): ").strip() or str(suggestions['lr'])
    lr = choice
    args_list.append(f"--lr {lr}")

    choice = input(f"  Dropout (建议{suggestions['dropout']}/?:查看): ").strip() or str(suggestions['dropout'])
    if choice == '?':
        show_param_help('dropout')
        choice = input(f"  Dropout (建议{suggestions['dropout']}): ").strip() or str(suggestions['dropout'])
    dropout = choice
    args_list.append(f"--dropout {dropout}")

    # ===== 9. 高级选项 =====
    print("\n【9/9】高级选项")
    choice = input("  预测偏移步数 (默认0/?查看): ").strip() or "0"
    if choice == '?':
        show_param_help('predict_offset')
        choice = input("  预测偏移步数 (默认0): ").strip() or "0"
    if choice != "0":
        args_list.append(f"--predict_offset {choice}")

    return args_list


def run_ablation_menu():
    """消融实验子菜单"""
    print("\n" + "=" * 60)
    print("  消融实验预设")
    print("=" * 60 + "\n")

    # 分组显示
    print("【基础变体】")
    for key in ['a1', 'a2', 'a3', 'a4', 'a5', 'a6']:
        preset = ABLATION_PRESETS.get(key, {})
        print(f"  [{key}] {preset.get('name', '')}")

    print("\n【YOLO-FPN变体】")
    for key in ['b1', 'b2', 'b3']:
        preset = ABLATION_PRESETS.get(key, {})
        print(f"  [{key}] {preset.get('name', '')}")

    print("\n【PatchTST变体】")
    for key in ['c1', 'c2']:
        preset = ABLATION_PRESETS.get(key, {})
        print(f"  [{key}] {preset.get('name', '')}")

    print("\n【高级优化】")
    for key in ['d1', 'd2', 'd3', 'd4']:
        preset = ABLATION_PRESETS.get(key, {})
        print(f"  [{key}] {preset.get('name', '')}")

    print("\n【团队协作】")
    for key in ['team_base', 'team_yolo', 'team_opt']:
        preset = ABLATION_PRESETS.get(key, {})
        print(f"  [{key}] {preset.get('name', '')}")

    print("\n  [export] 导出团队配置文件  [all] 运行全部  [q] 返回")

    choice = input("选择: ").strip().lower()
    if choice == 'q':
        return
    if choice == 'export':
        export_team_configs()
        input("\n按回车继续...")
        return run_ablation_menu()
    if choice == 'all':
        for key, preset in ABLATION_PRESETS.items():
            if key != 'all' and not key.startswith('team_'):
                cmd = build_command(preset['args'].split(), is_ablation=True)
                print(f"\n执行: {preset['name']}")
                os.system(cmd)
    elif choice in ABLATION_PRESETS:
        preset = ABLATION_PRESETS[choice]
        cmd = build_command(preset['args'].split(), is_ablation=True)
        print(f"\n执行: {preset['name']}")
        os.system(cmd)


def queue_config_wizard(launcher_callback):
    """
    队列配置向导 (B+C 流程)

    配置完一条命令后自动添加到队列，可以继续配置或开始训练

    Args:
        launcher_callback: 回调对象，需要有 trainer 属性
    """
    while True:
        print("\n" + "=" * 70)
        print("  队列配置向导 - 逐步配置每条训练命令")
        print("=" * 70)
        print()
        print(f"  当前队列: {len(launcher_callback.trainer.commands)} 条命令")
        print()
        print("  [1] 使用预设模板")
        print("  [2] 自定义配置（详细参数说明）")
        print("  [3] 快捷配置（基础参数）")
        print()
        print("  [s] ▶ 开始训练（开始执行队列中的所有命令）")
        print("  [v] 查看队列")
        print("  [q] 返回主菜单")
        print()

        choice = input("请选择: ").strip().lower()

        if choice == "q":
            return

        elif choice == "v":
            # 显示队列
            if not launcher_callback.trainer.commands:
                print("\n  队列为空")
            else:
                print("\n" + "-" * 60)
                print(f"  {'序号':^4} | {'模型':^20} | {'任务':^10} | {'Epochs':^6} | {'Offset':^6}")
                print("-" * 60)
                for i, cmd in enumerate(launcher_callback.trainer.commands, 1):
                    print(f"  {i:^4} | {cmd.get_display_name():^20} | {cmd.task:^10} | {cmd.epochs:^6} | {cmd.predict_offset:^6}")
                print("-" * 60)
            input("\n按回车继续...")
            continue

        elif choice == "s":
            # 开始训练
            if not launcher_callback.trainer.commands:
                print("\n  队列为空，请先添加训练命令")
                input("\n按回车继续...")
                continue
            return "start_training"

        elif choice == "1":
            # 预设模板
            print("\n【预设模板】\n")
            for key, preset in PRESETS.items():
                print(f"  [{key}] {preset['name']}")
            print("\n  [s] 分割任务  [m] 多任务  [q] 返回")
            print()

            preset_choice = input("请选择: ").strip().lower()

            if preset_choice == "q":
                continue
            elif preset_choice == "s":
                print("\n【分割任务预设】")
                for key, preset in SEGMENTATION_PRESETS.items():
                    print(f"  [{key}] {preset['name']}")
                seg_choice = input("选择: ").strip().lower()
                if seg_choice in SEGMENTATION_PRESETS:
                    preset = SEGMENTATION_PRESETS[seg_choice]
                    args_list = preset['args'].split()
                else:
                    continue
            elif preset_choice == "m":
                print("\n【多任务预设】")
                for key, preset in MULTITASK_PRESETS.items():
                    print(f"  [{key}] {preset['name']}")
                mt_choice = input("选择: ").strip().lower()
                if mt_choice in MULTITASK_PRESETS:
                    preset = MULTITASK_PRESETS[mt_choice]
                    args_list = preset['args'].split()
                else:
                    continue
            elif preset_choice in PRESETS:
                preset = PRESETS[preset_choice]
                args_list = preset['args'].split()
            else:
                continue

            # 解析并添加到队列
            cmd_str = build_command(args_list)
            from tools.batch_train_gui import TrainingCommand
            tc = TrainingCommand.from_string(cmd_str)
            success, msg, _ = launcher_callback.trainer.add_command(tc)
            if success:
                print(f"\n  ✅ 已添加到队列")
                print(f"     {tc.get_display_name()} | {tc.task} | {tc.epochs} epochs | offset={tc.predict_offset}")
            else:
                print(f"\n  ⚠️ {msg}")

        elif choice == "2":
            # 自定义配置（详细参数）
            args_list = interactive_mode()
            cmd_str = build_command(args_list)

            from tools.batch_train_gui import TrainingCommand
            tc = TrainingCommand.from_string(cmd_str)
            success, msg, _ = launcher_callback.trainer.add_command(tc)
            if success:
                print(f"\n  ✅ 已添加到队列")
                print(f"     {tc.get_display_name()} | {tc.task} | {tc.epochs} epochs | offset={tc.predict_offset}")
            else:
                print(f"\n  ⚠️ {msg}")

        elif choice == "3":
            # 快捷配置
            print("\n【快捷配置】")
            print()

            # 任务类型
            print("  任务类型:")
            print("    [1] detection  - 矩形框检测 (默认)")
            print("    [2] segmentation - 像素级掩膜")
            print("    [3] multitask  - 检测+分割")
            task_choice = input("    选择 [1]: ").strip() or "1"
            task_map = {"2": "segmentation", "3": "multitask"}
            task = task_map.get(task_choice, "detection")

            # 模型选择
            print("\n  模型配置:")
            print("    [1] resnet18 + cnn_attn + cross_attn (CNN基线，默认)")
            print("    [2] vit_small + transformer + cross_attn (Transformer)")
            print("    [3] resnet18 + cnn_attn + concat (拼接融合)")
            model_choice = input("    选择 [1]: ").strip() or "1"

            if model_choice == "2":
                b2d, b1d, fusion = "vit_small", "transformer", "cross_attn"
            elif model_choice == "3":
                b2d, b1d, fusion = "resnet18", "cnn_attn", "concat"
            else:
                b2d, b1d, fusion = "resnet18", "cnn_attn", "cross_attn"

            # 训练参数
            print("\n  训练参数:")
            epochs = input("    训练轮数 [150]: ").strip() or "150"
            offset = input("    预测偏移 [0]: ").strip() or "0"

            # 构建命令
            args_list = [
                "--backbone_2d", b2d,
                "--backbone_1d", b1d,
                "--fusion", fusion,
                "--epochs", epochs,
                "--predict_offset", offset,
                "--task", task
            ]
            cmd_str = build_command(args_list)

            from tools.batch_train_gui import TrainingCommand
            tc = TrainingCommand.from_string(cmd_str)
            success, msg, _ = launcher_callback.trainer.add_command(tc)
            if success:
                print(f"\n  ✅ 已添加到队列")
                print(f"     {tc.get_display_name()} | {tc.task} | {tc.epochs} epochs | offset={tc.predict_offset}")
            else:
                print(f"\n  ⚠️ {msg}")

        else:
            print("\n  无效选择")

        # 配置完成后继续循环
        print()
        input("按回车继续配置下一条命令...")


def main():
    while True:
        print("=" * 60)
        print("  PE-MMNet v4 交互式训练启动器")
        print("=" * 60 + "\n")

        print("【预设模板】\n")
        for key, preset in PRESETS.items():
            print(f"  [{key}] {preset['name']}")
            print(f"       {preset['description']}")
        print("\n  [s] 分割任务  [m] 多任务  [c] 自定义  [a] 消融  [q] 退出")

        choice = input("\n请选择: ").strip().lower()

        if choice == "q":
            return

        elif choice == "s":
            print("\n【分割任务预设】")
            for key, preset in SEGMENTATION_PRESETS.items():
                print(f"  [{key}] {preset['name']}")
            seg_choice = input("选择: ").strip().lower()
            if seg_choice in SEGMENTATION_PRESETS:
                preset = SEGMENTATION_PRESETS[seg_choice]
                os.system(build_command(preset['args'].split()))
            continue

        elif choice == "m":
            print("\n【多任务预设】")
            for key, preset in MULTITASK_PRESETS.items():
                print(f"  [{key}] {preset['name']}")
            mt_choice = input("选择: ").strip().lower()
            if mt_choice in MULTITASK_PRESETS:
                preset = MULTITASK_PRESETS[mt_choice]
                os.system(build_command(preset['args'].split()))
            continue

        elif choice == "a":
            run_ablation_menu()
            continue

        elif choice == "c":
            args_list = interactive_mode()

        elif choice in PRESETS:
            preset = PRESETS[choice]
            args_list = preset['args'].split()

        else:
            print("无效选择\n")
            continue

        cmd = build_command(args_list)
        print(f"\n命令: {cmd}\n")
        action = input("[回车]执行  [s]保存  [q]返回: ").strip().lower()

        if action == "q":
            continue
        elif action == "s":
            save_config(args_list)
            print("已保存\n")
        else:
            os.system(cmd)
            return


if __name__ == "__main__":
    main()
