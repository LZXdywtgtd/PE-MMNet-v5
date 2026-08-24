"""
PE-MMNet v4 统一控制台输出工具

标准化打印格式：
- 大段章节标题：上下 --- 分隔线，白色加粗
- 重要摘要/最终结果：上下 === 分隔线，青色高亮
- 普通信息：无分隔线，灰色

使用方法：
    from utils.console import print_title, print_result, print_info, print_warning

    print_title("加载数据集...")
    print_info(f"训练样本数: {len(train_loader)}")
    print_result("R2", 0.9399)      # 青色高亮
    print_result("mIoU", 0.6103)    # 青色高亮
"""

import sys
import os

# =============================================================================
# ANSI 转义码
# =============================================================================

# 颜色
COLOR_RESET = "\033[0m"
COLOR_RED = "\033[91m"
COLOR_GREEN = "\033[92m"
COLOR_YELLOW = "\033[93m"
COLOR_BLUE = "\033[94m"
COLOR_CYAN = "\033[96m"
COLOR_WHITE = "\033[97m"
COLOR_GRAY = "\033[90m"

# 样式
STYLE_BOLD = "\033[1m"
STYLE_DIM = "\033[2m"

# Windows 兼容性：启用 ANSI 颜色支持
if sys.platform == "win32":
    os.system("")  # 启用 Windows ANSI 支持


# =============================================================================
# 工具函数
# =============================================================================

def _colorize(text, color, bold=False):
    """给文本添加颜色和样式"""
    style = STYLE_BOLD if bold else ""
    return f"{style}{color}{text}{COLOR_RESET}"


def _format_value(value, fmt=".4f"):
    """格式化数值"""
    if isinstance(value, float):
        return f"{value:{fmt}}"
    return str(value)


# =============================================================================
# 核心打印函数
# =============================================================================

def print_title(text: str, width: int = 60):
    """
    打印章节标题（上下 --- 分隔线，白色加粗）

    用于：大段章节标题，如 "加载数据集...", "开始训练..."
    """
    line = "-" * width
    print(_colorize(line, COLOR_WHITE))
    print(_colorize(text, COLOR_WHITE, bold=True))
    print(_colorize(line, COLOR_WHITE))


def print_section(text: str, width: int = 60):
    """
    打印小节标题（无分隔线，白色加粗）

    用于：子标题，如 "[4/5] 开始训练"
    """
    print(_colorize(text, COLOR_WHITE, bold=True))


def print_result(key: str, value, fmt=".4f", width: int = 60, unit: str = ""):
    """
    打印关键指标（青色高亮值）

    用于：评估结果、训练指标等

    示例：
        print_result("R2", 0.9399)        # R2: 0.9399
        print_result("mIoU", 0.6103)       # mIoU: 0.6103
        print_result("违反率", 51.5, fmt=".1f", unit="%")  # 违反率: 51.5%
    """
    formatted_value = _format_value(value, fmt)
    if unit:
        text = f"  {key}: {formatted_value}{unit}"
    else:
        text = f"  {key}: {formatted_value}"
    print(_colorize(text, COLOR_CYAN))


def print_results_table(metrics: dict, width: int = 60):
    """
    打印指标表格（青色高亮值）

    用于：评估结果汇总

    示例：
        print_results_table({
            "R2": 0.9399,
            "RMSE": 0.0522,
            "MAE": 0.0461,
            "mIoU": 0.6103,
            "违反率": (51.5, ".1f", "%"),
        })
    """
    # 打印表头
    print()
    print(_colorize("=" * width, COLOR_CYAN))
    print(_colorize("  评估结果", COLOR_CYAN, bold=True))
    print(_colorize("=" * width, COLOR_CYAN))
    print()

    for key, value in metrics.items():
        if isinstance(value, tuple):
            val, fmt, unit = value
            print_result(key, val, fmt, unit=unit)
        else:
            print_result(key, value)

    print()
    print(_colorize("=" * width, COLOR_CYAN))


def print_info(text: str, indent: int = 0):
    """
    打印普通信息（灰色）

    用于：普通日志、详细信息
    """
    prefix = "  " * indent
    print(_colorize(f"{prefix}{text}", COLOR_GRAY))


def print_warning(text: str):
    """
    打印警告信息（黄色）

    用于：警告提示
    """
    print(_colorize(f"[警告] {text}", COLOR_YELLOW))


def print_error(text: str):
    """
    打印错误信息（红色）

    用于：错误提示
    """
    print(_colorize(f"[错误] {text}", COLOR_RED))


def print_success(text: str):
    """
    打印成功信息（绿色）

    用于：成功提示
    """
    print(_colorize(f"[成功] {text}", COLOR_GREEN))


def print_progress(current: int, total: int, text: str = ""):
    """
    打印进度信息

    示例：
        print_progress(4, 5, "开始训练")  # [4/5] 开始训练
    """
    progress_text = f"[{current}/{total}] {text}"
    print(_colorize(progress_text, COLOR_WHITE, bold=True))


def print_metric_row(key: str, value, fmt=".4f", width: int = 24):
    """
    打印指标行（对齐格式）

    用于：表格中的指标行

    示例：
        print_metric_row("R2", 0.9399)    # R2               0.9399
        print_metric_row("mIoU", 0.6103)  # mIoU             0.6103
    """
    formatted_value = _format_value(value, fmt)
    text = f"  {key:<12} {formatted_value:>10}"
    print(_colorize(text, COLOR_CYAN))


def print_divider(char: str = "-", width: int = 60, color=COLOR_GRAY):
    """
    打印分隔线

    用于：自定义分隔线
    """
    print(_colorize(char * width, color))


def print_header(text: str, width: int = 60):
    """
    打印主标题（上下 === 分隔线，青色）

    用于：主标题、重要输出块
    """
    line = "=" * width
    print(_colorize(line, COLOR_CYAN))
    print(_colorize(f"  {text}", COLOR_CYAN, bold=True))
    print(_colorize(line, COLOR_CYAN))


# =============================================================================
# 便捷别名
# =============================================================================

info = print_info
warn = print_warning
error = print_error
success = print_success
title = print_title
section = print_section
result = print_result
results = print_results_table
progress = print_progress
metric = print_metric_row
header = print_header
divider = print_divider
