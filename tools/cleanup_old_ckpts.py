"""
PE-MMNet v4 检查点清理工具

清理策略（与 run_train.py 当前实现一致）：
    每个 task 目录里只保留：
    - 1 个 _e{N}_best.pt（val_loss 最优的那个 epoch）
    - 1 个 _e{N}_last.pt（最新的那个 epoch）
    - 旧格式 _best.pt / _last.pt（无 _e{N} 后缀）保留，标记为"legacy"

被清理的对象：
    - 中间 epoch 的 _e{N}_best.pt（如 e001, e002 在 e003 是最优时该被删）
    - 旧 epoch 的 _e{N}_last.pt（每个 epoch 跑完会覆盖）

使用方法：
    # 干跑：只扫描
    & "$env:USERPROFILE\.conda\envs\pe_mmnet\python.exe" tools/cleanup_old_ckpts.py --dry-run

    # 实际清理（备份后删）
    & "$env:USERPROFILE\.conda\envs\pe_mmnet\python.exe" tools/cleanup_old_ckpts.py --yes
"""

import os
import sys
import shutil
import argparse
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Windows 控制台编码修复
if sys.platform == 'win32':
    try:
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
BACKUP_DIR = CHECKPOINT_DIR / 'backup' / f'_ckpt_cleanup_{datetime.now().strftime("%Y%m%d")}'

# 匹配 _e{N}_best.pt / _e{N}_last.pt
EPOCH_PATTERN = re.compile(r'_e(\d+)_(best|last)\.pt$')


def scan_checkpoints():
    """扫描所有 task 目录，按目录聚合文件。"""
    # variant/subdir/file.pt
    by_dir = defaultdict(lambda: {'best': [], 'last': [], 'legacy': []})
    for p in sorted(CHECKPOINT_DIR.rglob('*.pt')):
        if 'backup' in p.parts:
            continue
        rel = p.relative_to(CHECKPOINT_DIR)
        if len(rel.parts) < 2:
            continue
        # 跳过 .pre_taskid_*.pt 备份
        if '.pre_taskid_' in p.name:
            continue
        m = EPOCH_PATTERN.search(p.name)
        if m:
            epoch = int(m.group(1))
            kind = m.group(2)
            by_dir[str(rel.parent)]['best' if kind == 'best' else 'last'].append((p, epoch))
        else:
            # 旧格式 _best.pt / _last.pt
            if p.name.endswith('_best.pt'):
                by_dir[str(rel.parent)]['legacy'].append((p, 'best'))
            elif p.name.endswith('_last.pt'):
                by_dir[str(rel.parent)]['legacy'].append((p, 'last'))
    return by_dir


def decide_cleanup(by_dir):
    """决定每个目录要删哪些文件。返回 {dir: [(path, reason), ...]}。"""
    cleanup_plan = defaultdict(list)
    keep_summary = {}
    for d, files in by_dir.items():
        # 找最大 epoch
        best_files = files['best']
        last_files = files['last']

        best_max = max((ep for _, ep in best_files), default=None)
        last_max = max((ep for _, ep in last_files), default=None)

        # 删非最大 epoch 的 best
        for p, ep in best_files:
            if best_max is not None and ep != best_max:
                cleanup_plan[d].append((p, f'best epoch={ep} < max={best_max}'))
        # 删非最大 epoch 的 last
        for p, ep in last_files:
            if last_max is not None and ep != last_max:
                cleanup_plan[d].append((p, f'last epoch={ep} < max={last_max}'))

        keep_summary[d] = {
            'best_keep': f'e{best_max:03d}_best.pt' if best_max else None,
            'last_keep': f'e{last_max:03d}_last.pt' if last_max else None,
            'legacy': [p.name for p, _ in files['legacy']],
            'best_max': best_max,
            'last_max': last_max,
        }
    return cleanup_plan, keep_summary


def execute_cleanup(plan, dry_run=False):
    """执行清理：备份 + 删除。"""
    if not plan:
        return 0
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for d, items in plan.items():
        for p, reason in items:
            # 备份
            rel = p.relative_to(CHECKPOINT_DIR)
            backup_path = BACKUP_DIR / rel
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            if not dry_run and not backup_path.exists():
                shutil.copy2(p, backup_path)
            if not dry_run:
                p.unlink()
            n += 1
    return n


def main():
    parser = argparse.ArgumentParser(description='PE-MMNet v4 检查点清理工具')
    parser.add_argument('--dry-run', action='store_true', help='只扫描不删除')
    parser.add_argument('--yes', '-y', action='store_true', help='跳过交互确认')
    args = parser.parse_args()

    by_dir = scan_checkpoints()
    plan, keep_summary = decide_cleanup(by_dir)

    total_to_clean = sum(len(items) for items in plan.values())
    total_dirs = len(by_dir)
    dirs_with_cleanup = sum(1 for items in plan.values() if items)

    print(f'\n=== 扫描 {total_dirs} 个 task 目录，{dirs_with_cleanup} 个需要清理 ===\n')

    if not plan:
        print('没有需要清理的中间检查点。')
        return

    for d, items in plan.items():
        summary = keep_summary[d]
        keep_best = summary['best_keep'] or '-'
        keep_last = summary['last_keep'] or '-'
        legacy_str = f" (legacy: {', '.join(summary['legacy'])})" if summary['legacy'] else ''
        print(f'  {d}/')
        print(f'    保留: {keep_best}  +  {keep_last}{legacy_str}')
        for p, reason in items:
            print(f'    [DEL] {p.name}  ({reason})')

    print(f'\n=== 总计需删除 {total_to_clean} 个文件 ===')
    if args.dry_run:
        print('（dry-run 模式，未实际删除）')
        print(f'原文件备份到: {BACKUP_DIR.relative_to(PROJECT_ROOT)}/')
        return

    if not args.yes:
        confirm = input(f'确认删除 {total_to_clean} 个文件？(yes/no): ').strip().lower()
        if confirm != 'yes':
            print('已取消')
            return

    n = execute_cleanup(plan, dry_run=False)
    print(f'\n=== 已删除 {n} 个文件，备份在 {BACKUP_DIR.relative_to(PROJECT_ROOT)}/ ===')


if __name__ == '__main__':
    main()
