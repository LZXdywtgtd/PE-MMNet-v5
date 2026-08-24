"""
数据批次验证工具
自动检测每个批次的问题，生成报告并可选择清理
"""

import os
import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Optional
import pandas as pd

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class BatchIssue:
    """批次问题"""
    issue_type: str  # missing_dir, missing_file, file_count_mismatch, etc.
    description: str
    severity: str  # critical, warning, info


@dataclass
class BatchReport:
    """批次验证报告"""
    batch_name: str
    batch_path: str
    is_valid: bool
    has_temp_images: bool
    has_stress_images: bool
    has_label_csv: bool
    has_probe_csv: bool
    has_d_hist: bool
    temp_count: int
    stress_count: int
    label_csv_count: int
    d_hist_count: int
    issues: List[BatchIssue]
    missing_dirs: List[str]
    extra_dirs: List[str]


def natural_sort_key(s):
    """自然排序 key"""
    return [int(c) if c.isdigit() else c.lower() for c in s.split('.')[0].split('_')]


def validate_batch(batch_path: str) -> BatchReport:
    """验证单个批次"""
    batch_name = os.path.basename(os.path.normpath(batch_path))
    issues = []
    missing_dirs = []
    extra_dirs = []

    # 必须存在的目录
    required_dirs = {
        '温度': False,
        '应力': False,
        '表格': False,
    }

    # 可选目录
    optional_dirs = {
        'd_hist': False,
        '等值线': False,
        '等值线去除': False,
    }

    # 检查目录
    for item in os.listdir(batch_path):
        item_path = os.path.join(batch_path, item)
        if os.path.isdir(item_path):
            if item in required_dirs:
                required_dirs[item] = True
            elif item in optional_dirs:
                optional_dirs[item] = True
            else:
                extra_dirs.append(item)

    # 检查缺失的必须目录
    for dir_name, exists in required_dirs.items():
        if not exists:
            issues.append(BatchIssue(
                issue_type='missing_dir',
                description=f"缺少必须目录: {dir_name}",
                severity='critical'
            ))
            missing_dirs.append(dir_name)

    # 检查图像文件
    temp_count = 0
    stress_count = 0
    d_hist_count = 0

    if required_dirs['温度']:
        temp_dir = os.path.join(batch_path, '温度')
        temp_files = [f for f in os.listdir(temp_dir) if f.endswith('.png')]
        temp_count = len(temp_files)

    if required_dirs['应力']:
        stress_dir = os.path.join(batch_path, '应力')
        stress_files = [f for f in os.listdir(stress_dir) if f.endswith('.png')]
        stress_count = len(stress_files)

    if optional_dirs.get('d_hist', False):
        d_hist_dir = os.path.join(batch_path, 'd_hist')
        d_hist_files = [f for f in os.listdir(d_hist_dir) if f.endswith('.png')]
        d_hist_count = len(d_hist_files)

    # 检查图像数量一致性
    if required_dirs['温度'] and required_dirs['应力']:
        if temp_count != stress_count:
            issues.append(BatchIssue(
                issue_type='file_count_mismatch',
                description=f"温度({temp_count})和应力({stress_count})图像数量不一致",
                severity='critical'
            ))

    # 检查表格文件（探针CSV不再是必须的）
    label_csv_count = 0
    has_probe_csv = False  # 不再检查

    if required_dirs['表格']:
        table_dir = os.path.join(batch_path, '表格')

        # 标签 CSV
        label_csv_names = [
            '参数化扫描-表面最大值.csv',
            'Table_表面最大值.csv',
            'Table_Crack.csv',
            'Table_Hist.csv',
        ]
        for name in label_csv_names:
            path = os.path.join(table_dir, name)
            if os.path.exists(path):
                label_csv_count = 1
                break

        if label_csv_count == 0:
            issues.append(BatchIssue(
                issue_type='missing_file',
                description="缺少标签CSV文件",
                severity='critical'
            ))

        # 探针 CSV（可选，不再是必须的）
        probe_csv_names = [
            '参数化扫描-所有探针.csv',
            'Table_Temp.csv',
        ]
        for name in probe_csv_names:
            path = os.path.join(table_dir, name)
            if os.path.exists(path):
                has_probe_csv = True
                break

        # 探针CSV不再作为验证条件

    # 检查 d_hist 数量一致性（如果有）
    if optional_dirs.get('d_hist', False) and required_dirs['温度']:
        if d_hist_count != temp_count:
            issues.append(BatchIssue(
                issue_type='file_count_mismatch',
                description=f"d_hist({d_hist_count})和温度({temp_count})图像数量不一致",
                severity='warning'
            ))

    # 警告多余目录
    for dir_name in extra_dirs:
        issues.append(BatchIssue(
            issue_type='extra_dir',
            description=f"发现多余目录: {dir_name}",
            severity='info'
        ))

    # 判断是否有效（探针CSV不再是必须的）
    is_valid = (
        required_dirs['温度'] and
        required_dirs['应力'] and
        required_dirs['表格'] and
        label_csv_count > 0 and
        temp_count == stress_count and
        temp_count > 0
    )

    return BatchReport(
        batch_name=batch_name,
        batch_path=batch_path,
        is_valid=is_valid,
        has_temp_images=required_dirs['温度'],
        has_stress_images=required_dirs['应力'],
        has_label_csv=label_csv_count > 0,
        has_probe_csv=has_probe_csv,
        has_d_hist=optional_dirs.get('d_hist', False),
        temp_count=temp_count,
        stress_count=stress_count,
        label_csv_count=label_csv_count,
        d_hist_count=d_hist_count,
        issues=issues,
        missing_dirs=missing_dirs,
        extra_dirs=extra_dirs
    )


def validate_all_batches(data_root: str) -> dict:
    """验证所有批次"""
    results = {}

    for batch_name in os.listdir(data_root):
        batch_path = os.path.join(data_root, batch_name)
        if os.path.isdir(batch_path):
            report = validate_batch(batch_path)
            results[batch_name] = report

    return results


def print_report(report: BatchReport):
    """打印验证报告"""
    status = "✅ 有效" if report.is_valid else "❌ 无效"
    print(f"\n{'='*60}")
    print(f"批次: {report.batch_name} {status}")
    print(f"{'='*60}")

    print(f"\n📁 目录检查:")
    print(f"  温度图像: {'✅' if report.has_temp_images else '❌'} ({report.temp_count}个)")
    print(f"  应力图像: {'✅' if report.has_stress_images else '❌'} ({report.stress_count}个)")
    print(f"  标签CSV: {'✅' if report.has_label_csv else '❌'}")
    print(f"  探针CSV: {'⚪ 无(可选)' if not report.has_probe_csv else '✅'}")
    print(f"  d_hist: {'✅' if report.has_d_hist else '⚪ 无(可选)'} ({report.d_hist_count}个)")

    if report.missing_dirs:
        print(f"\n⚠️ 缺失目录: {', '.join(report.missing_dirs)}")

    if report.extra_dirs:
        print(f"\n💡 多余目录: {', '.join(report.extra_dirs)}")

    if report.issues:
        print(f"\n📋 问题列表:")
        for issue in report.issues:
            icon = "🔴" if issue.severity == 'critical' else "🟡" if issue.severity == 'warning' else "🔵"
            print(f"  {icon} [{issue.severity}] {issue.description}")


def generate_summary(results: dict) -> dict:
    """生成汇总"""
    valid_batches = [r for r in results.values() if r.is_valid]
    invalid_batches = [r for r in results.values() if not r.is_valid]

    summary = {
        'total_batches': len(results),
        'valid_batches': len(valid_batches),
        'invalid_batches': len(invalid_batches),
        'valid_names': [r.batch_name for r in valid_batches],
        'invalid_names': [r.batch_name for r in invalid_batches],
    }

    return summary


def main():
    import argparse
    from utils.config import get_data_root, ensure_config

    parser = argparse.ArgumentParser(description='数据批次验证工具')
    parser.add_argument('--root', type=str, help='数据根目录')
    parser.add_argument('--clean', action='store_true', help='删除无效批次')
    parser.add_argument('--keep-valid-only', action='store_true', help='只保留有效批次')
    args = parser.parse_args()

    # 获取数据根目录
    data_root = args.root or get_data_root()

    if not os.path.exists(data_root):
        print(f"数据目录不存在: {data_root}")
        # 如果没有配置，尝试创建配置
        if not os.path.exists('config.json'):
            ensure_config()
            data_root = get_data_root()

    print(f"\n🔍 验证数据目录: {data_root}")

    # 验证所有批次
    results = validate_all_batches(data_root)

    # 打印报告
    for report in results.values():
        print_report(report)

    # 汇总
    summary = generate_summary(results)

    print(f"\n{'='*60}")
    print("📊 汇总")
    print(f"{'='*60}")
    print(f"总批次数: {summary['total_batches']}")
    print(f"有效批次: {summary['valid_batches']}")
    print(f"无效批次: {summary['invalid_batches']}")

    if summary['valid_names']:
        print(f"\n✅ 有效批次: {', '.join(summary['valid_names'])}")

    if summary['invalid_names']:
        print(f"\n❌ 无效批次: {', '.join(summary['invalid_names'])}")

    # 保存报告
    report_data = {
        'summary': summary,
        'batches': {name: asdict(r) for name, r in results.items()}
    }

    report_path = os.path.join(os.path.dirname(data_root.rstrip('/\\')), 'batch_validation_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)
    print(f"\n💾 报告已保存: {report_path}")

    # 清理选项
    if args.clean and summary['invalid_names']:
        print(f"\n⚠️ 即将删除无效批次: {', '.join(summary['invalid_names'])}")
        confirm = input("确认删除? (y/n): ")
        if confirm.lower() == 'y':
            for batch_name in summary['invalid_names']:
                batch_path = os.path.join(data_root, batch_name)
                import shutil
                shutil.rmtree(batch_path)
                print(f"  已删除: {batch_name}")
            print("清理完成!")


if __name__ == '__main__':
    main()
