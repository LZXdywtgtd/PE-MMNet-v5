"""
PE-MMNet v4 配置管理模块

统一管理项目路径配置，支持从 config.json 读取或自动创建

用法:
    from utils.config import get_data_root, get_output_dir, ensure_config

    # 确保配置存在（首次运行会提示输入）
    ensure_config()

    # 获取数据根目录
    data_root = get_data_root()

    # 获取输出目录
    output_dir = get_output_dir()
"""

import os
import json
from pathlib import Path

# 配置文件路径
CONFIG_FILE = "config.json"
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)  # 回到项目根目录


def load_config():
    """加载配置文件"""
    config_path = os.path.join(PROJECT_ROOT, CONFIG_FILE)

    if os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError) as e:
            print(f"警告: 配置文件格式错误: {e}")
            print(f"  路径: {config_path}")
            print(f"  将重新创建配置文件")
            # 备份损坏的配置
            backup_path = config_path + '.bak'
            try:
                import shutil
                shutil.copy(config_path, backup_path)
                print(f"  已备份到: {backup_path}")
            except (OSError, shutil.Error) as e:
                print(f"  备份失败: {e}")
    return None


def save_config(config):
    """保存配置文件"""
    config_path = os.path.join(PROJECT_ROOT, CONFIG_FILE)

    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def get_data_root():
    """获取数据根目录"""
    config = load_config()
    if config and config.get('data_root'):
        return config['data_root']
    return None


def get_output_dir():
    """获取输出目录"""
    config = load_config()
    if config and config.get('output_dir'):
        return config['output_dir']
    return "./output"


def get_checkpoints_dir():
    """获取检查点目录"""
    config = load_config()
    if config and config.get('checkpoints_dir'):
        return config['checkpoints_dir']
    return "./checkpoints"


def get_results_dir():
    """获取结果目录"""
    config = load_config()
    if config and config.get('results_dir'):
        return config['results_dir']
    return "./benchmark_results"


def ensure_config(interactive=True):
    """
    确保配置文件存在且有效

    Args:
        interactive: 是否交互式提示用户输入

    Returns:
        bool: 配置是否有效
    """
    config = load_config()

    # 检查是否需要配置
    needs_setup = (
        config is None or
        not config.get('data_root') or
        not os.path.exists(config['data_root'])
    )

    if not needs_setup:
        return True

    if not interactive:
        return False

    # 交互式配置
    print("\n" + "=" * 60)
    print("  首次运行配置")
    print("=" * 60)
    print("\n请输入数据根目录路径（系统会自动扫描所有子目录作为数据批次）")
    print(f"\n默认路径示例: D:\\Desktop\\team_project\\simulation\\参考输入")
    print()
    print("提示: 每个子目录需要包含「表格」文件夹和 CSV 标签文件")
    print()

    while True:
        data_root = input("数据根目录: ").strip().strip('"').strip("'")

        if not data_root:
            # 尝试使用默认路径
            default_path = r"D:\Desktop\team_project\simulation\参考输入"
            if os.path.exists(default_path):
                data_root = default_path
                print(f"  使用默认路径: {data_root}")
            else:
                print("  错误: 请输入有效路径")
                continue

        if os.path.exists(data_root):
            break
        else:
            print(f"  错误: 路径不存在: {data_root}")
            # 尝试创建
            try:
                os.makedirs(data_root, exist_ok=True)
                print(f"  已创建目录: {data_root}")
                break
            except OSError as e:
                print(f"  创建目录失败: {e}")
                print("  请手动创建目录或检查路径权限")

    # 保存配置
    new_config = config or {}
    new_config['data_root'] = data_root

    # 确保输出目录存在
    output_dir = new_config.get('output_dir', './output')
    checkpoints_dir = new_config.get('checkpoints_dir', './checkpoints')
    results_dir = new_config.get('results_dir', './benchmark_results')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    save_config(new_config)

    print("\n" + "=" * 60)
    print(f"  配置已保存到: {os.path.join(PROJECT_ROOT, CONFIG_FILE)}")
    print("=" * 60)

    return True


def get_data_batches():
    """
    获取所有数据批次路径（自动扫描）

    自动扫描数据根目录，找到包含正确结构的子目录作为批次

    Returns:
        list: 数据批次路径列表
    """
    data_root = get_data_root()
    if not data_root:
        return []

    batches = []
    if not os.path.exists(data_root):
        return []

    # 自动扫描数据根目录下的所有子目录
    for entry in os.scandir(data_root):
        if entry.is_dir():
            batch_path = entry.path

            # 检查是否是有效的数据批次（必须有表格目录和CSV文件）
            table_dir = os.path.join(batch_path, '表格')
            if os.path.isdir(table_dir):
                # 检查是否包含标签CSV文件
                csv_files = [
                    '参数化扫描-表面最大值.csv',
                    'Table_表面最大值.csv',
                    'Table_Crack.csv',
                    'Table_Hist.csv',
                ]
                has_csv = any(os.path.exists(os.path.join(table_dir, f)) for f in csv_files)
                if has_csv:
                    batches.append(batch_path)

    return batches


# 便捷别名
get = load_config
set = save_config
batches = get_data_batches
