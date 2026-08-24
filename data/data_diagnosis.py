"""
PE-MMNet v4 数据完整性诊断脚本

检查所有数据批次的完整性：
- CSV 文件行数
- 图像文件数量
- 序列长度一致性
- 时间戳均匀性（时间对齐）
- 图像编号连续性

用法：
    python data_diagnosis.py
"""

import os
import re
import numpy as np
from pathlib import Path

# 数据根目录
DATA_ROOT = r"D:\Desktop\team_project\simulation\参考输入"

# 批次列表
BATCHES = [
    "单次扫描",
    "参数化扫描1",
    "参数化扫描2",
    "参数化扫描3",
    "参数化扫描4",
]


# =============================================================================
# 工具函数
# =============================================================================

def get_time_column_index(csv_path):
    """
    根据列数自动判断时间列索引

    CSV 格式：
    - 单次扫描：10列，时间在第0列
    - 参数化扫描：11列，时间在第1列
    """
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 跳过注释行，找第一行数据
        data_lines = [l for l in lines if l.strip() and not l.strip().startswith('%')]
        if not data_lines:
            return None

        # 检查第一行数据（跳过表头）
        for line in data_lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 10:
                # 10列 = 单次扫描，时间在第0列
                # 11列 = 参数化扫描，时间在第1列
                return 0 if len(parts) == 10 else 1
        return None
    except Exception:
        return None


def check_time_alignment(csv_path):
    """
    检查时间戳是否均匀分布

    Returns:
        tuple: (是否正常, 消息)
    """
    time_idx = get_time_column_index(csv_path)
    if time_idx is None:
        return False, "无法确定时间列"

    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 跳过注释行和空行
        data_lines = [l for l in lines if l.strip() and not l.strip().startswith('%')]
        if len(data_lines) < 2:
            return False, "数据行不足"

        # 提取时间列
        time_col = []
        for line in data_lines[1:]:  # 跳过表头
            parts = line.strip().split(',')
            if len(parts) > time_idx:
                try:
                    time_col.append(float(parts[time_idx]))
                except (ValueError, IndexError):
                    continue

        if len(time_col) < 2:
            return False, "时间列无效"

        time_arr = np.array(time_col)
        diffs = np.diff(time_arr)
        median_diff = np.median(diffs)
        expected = 0.02  # COMSOL 默认步长

        # 检查均匀性（容差 50%）
        if abs(median_diff - expected) > expected * 0.5:
            return False, f"步长异常: {median_diff:.4f}s vs 预期{expected}s"

        # 检查跳变（50倍阈值）
        max_diff = np.max(diffs)
        if max_diff > expected * 50:
            return False, f"时间跳变: {max_diff:.2f}s"

        return True, f"正常 (步长≈{median_diff:.4f}s)"
    except Exception as e:
        return False, f"检查失败: {e}"


def check_image_numbering(img_dir, expected_count=None):
    """
    检查 PNG 文件编号是否连续

    Args:
        img_dir: 图像目录路径
        expected_count: 预期图像数量（可选）

    Returns:
        tuple: (是否正常, 消息)
    """
    try:
        images = [f for f in os.listdir(img_dir) if f.endswith('.png')]

        if not images:
            return False, "无图像文件"

        # 提取编号
        numbers = []
        for img in images:
            match = re.search(r'(\d+)', img)
            if match:
                numbers.append(int(match.group(1)))

        if not numbers:
            return True, f"{len(images)} 张图（无法解析编号）"

        numbers.sort()
        actual_count = len(numbers)

        # 检查缺失编号（如果有预期数量）
        if expected_count is not None and expected_count > 0:
            expected_range = set(range(1, expected_count + 1))
            actual_range = set(numbers)
            missing = expected_range - actual_range

            if missing and len(missing) > expected_count * 0.1:  # 缺失超过10%
                return False, f"编号不连续: 缺失 {len(missing)} 个编号 (如 {list(missing)[:5]}...)"

            # 检查编号范围
            if numbers[0] != 1 or numbers[-1] != expected_count:
                return False, f"编号范围异常: {numbers[0]}~{numbers[-1]}, 预期 1~{expected_count}"

        # 检查连续性（简单检查）
        if len(numbers) >= 2:
            gaps = [numbers[i+1] - numbers[i] for i in range(len(numbers)-1)]
            if max(gaps) > 1:
                return False, f"编号存在间隔: 最大间隔 {max(gaps)}"

        return True, f"编号连续: 1~{actual_count}"
    except Exception as e:
        return False, f"检查失败: {e}"


# =============================================================================
# 批次诊断
# =============================================================================

def diagnose_batch(batch_path):
    """
    诊断单个批次的数据完整性

    Args:
        batch_path: 批次目录路径

    Returns:
        dict: 诊断结果
    """
    batch_name = os.path.basename(batch_path)
    issues = []
    info = {}

    # 检查目录存在
    if not os.path.exists(batch_path):
        return {
            'exists': False,
            'issues': [f"路径不存在: {batch_path}"]
        }

    # 检查表格目录
    csv_dir = os.path.join(batch_path, '表格')
    if os.path.exists(csv_dir):
        csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
        if csv_files:
            # 读取主要标签文件
            label_files = [f for f in csv_files if '表面最大值' in f or 'Crack' in f or 'Hist' in f]
            if label_files:
                csv_path = os.path.join(csv_dir, label_files[0])
                with open(csv_path, 'r', encoding='utf-8') as f:
                    lines = len(f.readlines())
                info['csv_rows'] = lines
                print(f"  CSV ({label_files[0]}): {lines} 行")

                # 检查 CSV 完整性
                if batch_name == "单次扫描":
                    expected = 21
                elif batch_name in ["参数化扫描1", "参数化扫描2"]:
                    expected = 155
                elif batch_name == "参数化扫描3":
                    expected = 1000  # 0~20s, 步长0.02s
                elif batch_name == "参数化扫描4":
                    expected = 1500  # 0~30s, 步长0.02s
                else:
                    expected = None

                if expected and lines < expected:
                    issues.append(f"CSV 仅 {lines} 行，预期 {expected} 行（可能未导出完整数据）")

                # 检查时间对齐
                ok, msg = check_time_alignment(csv_path)
                if ok:
                    print(f"  时间对齐: {msg}")
                else:
                    issues.append(f"时间对齐问题: {msg}")
                    print(f"  时间对齐: [WARN] {msg}")

            else:
                issues.append("未找到标签 CSV 文件")
        else:
            issues.append("表格目录为空")
    else:
        issues.append("未找到表格目录")

    # 检查图像数量
    for img_type in ['温度', '应力']:
        img_dir = os.path.join(batch_path, img_type)
        if os.path.exists(img_dir):
            images = [f for f in os.listdir(img_dir) if f.endswith('.png')]
            count = len(images)
            info[f'{img_type}_images'] = count
            print(f"  {img_type}图像: {count} 张")

            # 检查编号连续性
            ok, msg = check_image_numbering(img_dir, expected_count=info.get('csv_rows'))
            if ok:
                print(f"  {img_type}编号: {msg}")
            else:
                issues.append(f"{img_type}编号问题: {msg}")
                print(f"  {img_type}编号: [WARN] {msg}")

            # 检查与 CSV 一致性
            if 'csv_rows' in info and count != info['csv_rows']:
                issues.append(f"{img_type}图像 ({count}) 与 CSV ({info['csv_rows']}) 数量不一致")
        else:
            issues.append(f"未找到{img_type}目录")

    # 检查探针数据
    probe_dir = os.path.join(batch_path, '表格')
    if os.path.exists(probe_dir):
        probe_files = [f for f in os.listdir(probe_dir) if '探针' in f or 'Temp' in f]
        if probe_files:
            print(f"  探针文件: {probe_files[0]}")

    return {
        'exists': True,
        'info': info,
        'issues': issues
    }


# =============================================================================
# 汇总与建议
# =============================================================================

def print_summary(results):
    """打印汇总结果"""
    print("\n" + "=" * 60)
    print("  诊断汇总")
    print("=" * 60)

    total_issues = 0
    for batch_name, result in results.items():
        if not result['exists']:
            print(f"\n{batch_name}: [X] 路径不存在")
            total_issues += 1
        elif result['issues']:
            print(f"\n{batch_name}: [WARN] {len(result['issues'])} 个问题")
            for issue in result['issues']:
                print(f"     - {issue}")
            total_issues += len(result['issues'])
        else:
            print(f"\n{batch_name}: [OK] 完整")

    print("\n" + "=" * 60)
    if total_issues == 0:
        print("  所有批次数据完整!")
    else:
        print(f"  共发现 {total_issues} 个问题，请检查数据导出")
    print("=" * 60)


def print_recommendations(results):
    """打印建议"""
    print("\n" + "=" * 60)
    print("  处理建议")
    print("=" * 60)

    for batch_name, result in results.items():
        if not result['exists']:
            continue

        issues = result['issues']
        if not issues:
            continue

        print(f"\n{batch_name}:")

        # CSV 行数不足
        csv_issues = [i for i in issues if 'CSV' in i]
        if csv_issues:
            print("  1. CSV 数据不完整")
            print("     -> 需要重新从 COMSOL 导出完整的时间步数据")
            print("     -> 确保导出所有时间点的标签和探针数据")

        # 时间对齐问题
        time_issues = [i for i in issues if '时间对齐' in i or '时间跳变' in i]
        if time_issues:
            print("  2. 时间对齐问题")
            print("     -> 检查 COMSOL 导出设置，确保时间步均匀")
            print("     -> 可能是部分时间步被过滤或合并")

        # 图像数量不足
        img_issues = [i for i in issues if '图像' in i and '数量' in i]
        if img_issues:
            print("  3. 图像数据不完整")
            print("     -> 需要重新导出图像序列")
            print("     -> 确保每个时间步都有温度场和应力场图像")

        # 图像编号问题
        num_issues = [i for i in issues if '编号' in i]
        if num_issues:
            print("  4. 图像编号不连续")
            print("     -> 检查导出过程中是否有文件丢失")
            print("     -> 确认文件名格式统一（统一使用前导零或不适用）")

        # 数量不一致
        mismatch_issues = [i for i in issues if '不一致' in i]
        if mismatch_issues:
            print("  5. 图像与标签数量不匹配")
            print("     -> 检查数据导出设置，确保时间步对齐")

    print()


# =============================================================================
# 主函数
# =============================================================================

def main():
    print("=" * 60)
    print("  PE-MMNet v4 数据完整性诊断")
    print("=" * 60)
    print(f"\n数据目录: {DATA_ROOT}\n")

    results = {}

    for batch in BATCHES:
        batch_path = os.path.join(DATA_ROOT, batch)
        print(f"\n{batch}:")
        print("-" * 40)

        result = diagnose_batch(batch_path)
        results[batch] = result

        if result['issues']:
            print("  [WARN] 问题:")
            for issue in result['issues']:
                print(f"     - {issue}")
        else:
            print("  [OK] 数据完整")

    print_summary(results)
    print_recommendations(results)

    # 导出诊断报告
    report_path = os.path.join(os.path.dirname(__file__), 'diagnosis_report.json')
    import json
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"诊断报告已保存: {report_path}")


if __name__ == "__main__":
    main()
