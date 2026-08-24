"""
PE-MMNet v4 检查点 save_reason 误标修复工具

背景：
    run_train.py 的 _mark_checkpoint_complete() 旧实现根据 best.pt 的原始
    save_reason 推断最终 save_reason（'improvement' → 'early_stop'），
    与训练是否真的早停无关，导致 best.pt 永远被标 'early_stop'。
    本工具批量把"实际正常完成"的检查点 best.pt.save_reason 改成 'completed'。

修复规则：
    1. is_complete=True 的 best/last
    2. 实际完成（last_epoch == target_epochs，从 task JSON 读）
    3. 排除 A1（旧格式检查点 + last_epoch=1，日志已清理无法回溯）

保留策略：
    - 不删除任何 epoch 中间检查点（e001/e002），用于蓝屏前回溯训练位置
    - 原文件备份到 backup/_save_reason_fix_20260810/，再写回

使用方法：
    # 干跑（只看不改）
    & "$env:USERPROFILE\.conda\envs\pe_mmnet\python.exe" tools/fix_save_reason.py --dry-run

    # 实际修复
    & "$env:USERPROFILE\.conda\envs\pe_mmnet\python.exe" tools/fix_save_reason.py

    # 看修复后状态
    & "$env:USERPROFILE\.conda\envs\pe_mmnet\python.exe" tools/fix_save_reason.py --verify
"""

import os
import sys
import json
import shutil
import argparse
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

from utils.config import ensure_config  # noqa: E402

CHECKPOINT_DIR = PROJECT_ROOT / 'checkpoints'
TASKS_DIR = PROJECT_ROOT / 'tasks'
BACKUP_DIR = CHECKPOINT_DIR / 'backup' / f'_save_reason_fix_{datetime.now().strftime("%Y%m%d")}'

# 已知特殊情况：旧格式检查点 + 日志已清理，无法可靠判断最终 save_reason
SKIP_TASK_IDS = {'A1'}


def load_task_targets() -> dict:
    """从 tasks/*.json 读每个 task_id 的 --epochs 值。"""
    targets = {}
    for json_file in TASKS_DIR.glob('team_*.json'):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                tasks = json.load(f)
        except Exception:
            continue
        for task in tasks:
            if 'id' not in task:
                continue
            args = task.get('args', '')
            # 解析 --epochs N
            import re
            m = re.search(r'--epochs\s+(\d+)', args)
            if m:
                targets[task['id']] = int(m.group(1))
    return targets


def _load_checkpoint_meta(p):
    """加载检查点元数据，丢弃大张量（model_state_dict 等），只保留小字段。"""
    ck = __import__('torch').load(p, map_location='cpu', weights_only=False)
    if isinstance(ck, dict):
        # 释放大对象（模型/优化器权重），只保留决策需要的元数据
        for heavy in ('model_state_dict', 'optimizer_state_dict', 'scheduler_state_dict'):
            ck.pop(heavy, None)
    return ck


def collect_checkpoints() -> dict:
    """扫描 checkpoints/，按 task_id 聚合所有 best/last 文件。

    单次 rglob + 按后缀分发，避免两次全目录扫描。返回的 dict 中每个 value
    是 [(path, loaded_meta), ...]——meta 已丢弃大张量，可直接用于决策。
    """
    by_task = defaultdict(lambda: {'best': [], 'last': []})
    for p in sorted(CHECKPOINT_DIR.rglob('*.pt')):
        if 'backup' in p.parts:
            continue
        # 按后缀分发：避免两次 rglob
        if p.name.endswith('_best.pt'):
            kind = 'best'
        elif p.name.endswith('_last.pt'):
            kind = 'last'
        else:
            continue
        try:
            ck = _load_checkpoint_meta(p)
            if isinstance(ck, dict) and ck.get('task_id'):
                by_task[ck['task_id']][kind].append((p, ck))
        except Exception:
            continue
    return by_task


def decide_fix(task_id, best_files, last_files, target_epochs):
    """判断是否需要修复，并给出新 save_reason。返回 (should_fix, plan_list)。"""
    plan = []
    # 找最新 epoch 的 best 和 last
    best_latest = max(best_files, key=lambda x: x[1].get('epoch', 0)) if best_files else None
    last_latest = max(last_files, key=lambda x: x[1].get('epoch', 0)) if last_files else None

    if not best_latest or not last_latest:
        return False, plan

    best_path, best_ck = best_latest
    last_path, last_ck = last_latest

    # 跳过未完成
    if not (best_ck.get('is_complete') or best_ck.get('is_retrain_done')):
        return False, plan
    if not (last_ck.get('is_complete') or last_ck.get('is_retrain_done')):
        return False, plan

    last_epoch = max(best_ck.get('epoch', 0), last_ck.get('epoch', 0))

    # 跳过特殊情况
    if task_id in SKIP_TASK_IDS:
        plan.append({
            'path': best_path, 'file': 'best',
            'old': best_ck.get('save_reason'),
            'new': None,
            'skip_reason': f'task_id={task_id} 在 SKIP 列表（旧格式 / 无法回溯）',
        })
        return False, plan

    # 规则 1：best='early_stop' + last='completed' + last_epoch==target → 实际是正常完成
    if (best_ck.get('save_reason') == 'early_stop'
            and last_ck.get('save_reason') == 'completed'
            and last_epoch == target_epochs):
        plan.append({
            'path': best_path, 'file': 'best',
            'old': 'early_stop', 'new': 'completed',
            'reason': f'last_epoch({last_epoch})==target({target_epochs})，实际是跑完而非早停',
        })

    # 规则 2：best='improvement' + last='epoch_end' + is_complete=True → _mark 没跑完，改成 completed
    # 不要求 last_epoch==target（适用于"目标 epoch=150 但只跑 2 epoch 后中断 + 重训完成"的情况，如 A4）
    elif (best_ck.get('save_reason') == 'improvement'
            and last_ck.get('save_reason') == 'epoch_end'
            and (best_ck.get('is_complete') or best_ck.get('is_retrain_done'))):
        plan.append({
            'path': best_path, 'file': 'best',
            'old': 'improvement', 'new': 'completed',
            'reason': 'is_complete=True 但 save_reason 仍为 improvement（_mark_checkpoint_complete 似乎没执行到底）',
        })
        plan.append({
            'path': last_path, 'file': 'last',
            'old': 'epoch_end', 'new': 'completed',
            'reason': 'is_complete=True 但 save_reason 仍为 epoch_end',
        })

    return any(p.get('new') for p in plan), plan


def execute_fix(plan, dry_run=False):
    """执行修复：备份 + 改字段 + 写回。"""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    actions = []
    for item in plan:
        if not item.get('new'):
            continue
        p = item['path']
        # 备份
        rel = p.relative_to(CHECKPOINT_DIR)
        backup_path = BACKUP_DIR / rel
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        if not backup_path.exists():
            if not dry_run:
                shutil.copy2(p, backup_path)
        # 改字段（必须重新加载完整 dict 以保留 model_state_dict 等大对象；
        # 之前 collect_checkpoints 阶段已丢弃大对象以省内存）
        if not dry_run:
            import torch
            ck = torch.load(p, map_location='cpu', weights_only=False)
            ck['save_reason'] = item['new']
            ck['save_reason_fixed_at'] = datetime.now().isoformat()
            torch.save(ck, p)
        actions.append((p, item['old'], item['new']))
    return actions


def cmd_dry_run(args, targets, by_task):
    print(f'\n=== Dry run：扫描 {len(by_task)} 个 task 的检查点 ===\n')
    total_changes = 0
    for tid in sorted(by_task.keys()):
        target = targets.get(tid)
        best_files = by_task[tid]['best']
        last_files = by_task[tid]['last']
        if target is None:
            print(f'  [skip-no-target] {tid}: 任务 JSON 中找不到 --epochs')
            continue
        should_fix, plan = decide_fix(tid, best_files, last_files, target)
        if not plan:
            print(f'  [no-change] {tid}: target={target}, 无需修改')
            continue
        print(f'\n  === {tid} (target={target}) ===')
        for item in plan:
            if item.get('new'):
                rel = item['path'].relative_to(CHECKPOINT_DIR)
                print(f'    [{item["file"]}] {rel}')
                print(f'      {item["old"]!r} → {item["new"]!r}')
                print(f'      原因: {item["reason"]}')
                total_changes += 1
            elif item.get('skip_reason'):
                rel = item['path'].relative_to(CHECKPOINT_DIR)
                print(f'    [{item["file"]}] {rel}')
                print(f'      SKIP: {item["skip_reason"]}')
    print(f'\n=== 总计需修改 {total_changes} 个文件 ===')
    if total_changes and not args.dry_run:
        print('即将执行修复…')
    elif total_changes:
        print('（dry-run 模式，未实际修改）')
    return total_changes


def cmd_verify(args, targets, by_task):
    print(f'\n=== Verify：检查 save_reason 修复结果 ===\n')
    ok = 0
    suspect = 0
    for tid in sorted(by_task.keys()):
        best_files = by_task[tid]['best']
        last_files = by_task[tid]['last']
        if not best_files or not last_files:
            continue
        best_ck = max(best_files, key=lambda x: x[1].get('epoch', 0))[1]
        last_ck = max(last_files, key=lambda x: x[1].get('epoch', 0))[1]
        is_complete = best_ck.get('is_complete') or last_ck.get('is_complete')
        if not is_complete:
            continue
        best_reason = best_ck.get('save_reason')
        last_reason = last_ck.get('save_reason')
        target = targets.get(tid, 0)
        last_epoch = max(best_ck.get('epoch', 0), last_ck.get('epoch', 0))
        consistent = (best_reason == last_reason)
        is_legit_early_stop = (best_reason == 'early_stop' and last_epoch < target)
        if consistent:
            ok += 1
            print(f'  [OK] {tid}: best={best_reason}, last={last_reason}（一致）')
        elif is_legit_early_stop:
            suspect += 0
            print(f'  [OK] {tid}: best={best_reason}, last={last_reason}（早停合法，last_epoch={last_epoch}<target={target}）')
        else:
            suspect += 1
            print(f'  [!!] {tid}: best={best_reason}, last={last_reason}（不一致，target={target}）')
    print(f'\n=== 总结：{ok} OK, {suspect} 仍异常 ===')


def main():
    parser = argparse.ArgumentParser(description='PE-MMNet v4 检查点 save_reason 误标修复')
    parser.add_argument('--dry-run', action='store_true', help='只扫描不修改')
    parser.add_argument('--verify', action='store_true', help='验证修复结果')
    parser.add_argument('--yes', '-y', action='store_true', help='跳过交互确认（脚本/CI 用）')
    args = parser.parse_args()

    ensure_config()  # 确保 config.json 存在
    targets = load_task_targets()
    by_task = collect_checkpoints()

    if args.verify:
        cmd_verify(args, targets, by_task)
        return

    total = cmd_dry_run(args, targets, by_task)
    if total == 0 or args.dry_run:
        return

    # 二次确认（避免误操作）
    if not args.yes:
        print(f'\n即将修改 {total} 个文件。原文件备份到: {BACKUP_DIR.relative_to(PROJECT_ROOT)}')
        confirm = input('确认执行？(yes/no): ').strip().lower()
        if confirm != 'yes':
            print('已取消')
            return
    else:
        print(f'\n(--yes 模式，直接执行，将修改 {total} 个文件)')

    # 重新扫描（dry-run 中没保留 plan），再执行
    n_done = 0
    for tid in sorted(by_task.keys()):
        target = targets.get(tid)
        if target is None:
            continue
        _, plan = decide_fix(tid, by_task[tid]['best'], by_task[tid]['last'], target)
        actions = execute_fix(plan, dry_run=False)
        n_done += len(actions)
        for p, old, new in actions:
            rel = p.relative_to(CHECKPOINT_DIR)
            print(f'  [FIXED] {rel}: {old!r} → {new!r}')

    print(f'\n=== 实际修改 {n_done} 个文件，备份在 {BACKUP_DIR.relative_to(PROJECT_ROOT)}/ ===')
    print('建议运行 --verify 确认修复结果')


if __name__ == '__main__':
    main()
