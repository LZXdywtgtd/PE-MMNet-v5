#!/usr/bin/env python3
"""
PE-MMNet v5-α 团队协作训练系统（V5B7）

> 继承关系：本模块继承自 v4 team_train.py（结构基本一致 + 命令名换为 v5）。
> v5-α 当前内置任务对应 tasks/team_v5_alpha.json 中的 V5A1–V5A6。

功能：
1. 配置驱动：任务由 JSON 配置文件定义（tasks/team_v5_alpha.json）
2. 依赖管理：任务间可定义前置依赖
3. 硬件感知：根据显存推荐任务等级
4. 状态可视化：已完成/可执行/警告/锁定 状态显示
5. 检查点导入：支持导入队友的 .pt 文件
6. 自动批量执行：按拓扑排序执行所有可执行任务
7. 任务日志：记录每个任务的执行情况（logs/team_training.log）

使用方法：
    python team_train_v5.py              # 交互式菜单
    python team_train_v5.py --auto     # 自动执行所有可执行任务
    python team_train_v5.py --auto --force  # 自动执行（包括硬件警告任务）
    python team_train_v5.py --import   # 导入队友检查点

与 v4 team_train.py 的差异：
- 入口脚本：run_train.py → run_train_v5.py
- 任务文件：tasks/*.json → tasks/team_v5_alpha.json
- 内置任务：A1-A5 → V5A1-V5A6（架构在 v5-α 已分层 + V5-β 钩子预留）
- 默认参数：v4 (patch=64, K=4-8) → v5-α (patch=256, K=8-16, 3 通道)
"""

import os
import sys
import json
import shutil
import socket
import re
from pathlib import Path
from typing import Dict, Set
from datetime import datetime
import torch

# =============================================================================
# 颜色支持（继承自 v4）
# =============================================================================

def get_color_support():
    """检测终端颜色支持"""
    if os.environ.get('NO_COLOR'):
        return 'none'

    try:
        import io
        if sys.platform == 'win32':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        from colorama import init, Fore, Style
        init(autoreset=True, strip=False)
        return 'colorama'
    except ImportError:
        pass

    if sys.platform == 'win32':
        try:
            version = sys.getwindowsversion()
            if version.major >= 10:
                return 'ansi'
        except Exception:
            pass

    term = os.environ.get('TERM', '')
    if 'xterm' in term or 'screen' in term or term == 'ANSI':
        return 'ansi'

    return 'none'


color_mode = get_color_support()
if color_mode == 'colorama':
    from colorama import Fore, Style
    COLORS = {
        'green': Fore.GREEN,
        'blue': Fore.BLUE,
        'yellow': Fore.YELLOW,
        'gray': Fore.LIGHTBLACK_EX,
        'red': Fore.RED,
        'bold': Style.BRIGHT,
        'reset': Style.RESET_ALL,
    }
elif color_mode == 'ansi':
    COLORS = {
        'green': '\033[92m',
        'blue': '\033[94m',
        'yellow': '\033[93m',
        'gray': '\033[90m',
        'red': '\033[91m',
        'bold': '\033[1m',
        'reset': '\033[0m',
    }
else:
    COLORS = {k: '' for k in ['green', 'blue', 'yellow', 'gray', 'red', 'bold', 'reset']}


# 路径配置
SCRIPT_DIR = Path(__file__).parent
TASKS_DIR = SCRIPT_DIR / 'tasks'
CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
LOG_DIR = SCRIPT_DIR / 'logs'
RUN_TRAIN = SCRIPT_DIR / 'run_train_v5.py'


def _resolve_python_interpreter() -> tuple:
    """选择可用的 Python 解释器（继承自 v4：优先 conda/venv 内 sys.executable）"""
    import shutil as _shutil
    exe = sys.executable
    if exe and os.path.isfile(exe):
        return f'"{exe}"', f'当前环境 ({exe})'

    for name in ('python', 'python3', 'py'):
        found = _shutil.which(name)
        if found:
            if sys.platform == 'win32' and name == 'py':
                try:
                    import subprocess
                    r = subprocess.run(
                        ['py', '-c', 'import sys; print(sys.executable)'],
                        capture_output=True, text=True, timeout=5,
                    )
                    real = r.stdout.strip()
                    if 'WindowsApps' in real or 'Microsoft' in real:
                        continue
                except Exception:
                    continue
            return f'"{found}"' if ' ' in found else found, f'PATH 找到 {name} ({found})'

    return 'python', '默认（PATH 兜底，可能未安装）'


# 全局任务字典
all_tasks: Dict = {}


# =============================================================================
# 任务执行日志（继承自 v4）
# =============================================================================

def log_task_execution(task_id, status, duration_seconds=None, error=None):
    """记录任务执行日志"""
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / 'team_training_v5.log'
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'task_id': task_id,
        'status': status,
        'duration_seconds': duration_seconds,
        'error': str(error) if error else None,
        'hostname': socket.gethostname(),
    }
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        print(f"[警告] 写入日志失败: {e}")


# =============================================================================
# 内置默认任务（v5-α：V5A1-V5A6）
# =============================================================================

DEFAULT_V5_TRAIN_TASKS = {
    # ========== V5A 组（v5-α 必做） ==========
    "V5A1": {
        "id": "V5A1",
        "name": "patch 仿真器",
        "desc": "256×256 patch 仿真器 + 物理约束（已 V5A1 完成 ✅）",
        "args": "--module simulator",
        "level": "L1",
        "time": "~3 天",
        "gpu_mem": "~0GB（纯 CPU 仿真）",
        "deps": [],
    },
    "V5A2": {
        "id": "V5A2",
        "name": "Spline 输出头 + GP 模块",
        "desc": "SplineHead + GPModule（已 V5A2 完成 ✅）",
        "args": "--module heads",
        "level": "L1",
        "time": "~1 周",
        "gpu_mem": "~0GB（独立测试）",
        "deps": [],
    },
    "V5A3": {
        "id": "V5A3",
        "name": "覆盖距离 loss",
        "desc": "Catmull-Rom 样条 + Chamfer/Hausdorff + bbox + GP + 泊松（已 V5A3 完成 ✅）",
        "args": "--module loss",
        "level": "L1",
        "time": "~3 天",
        "gpu_mem": "~0GB",
        "deps": [],
    },
    "V5A4": {
        "id": "V5A4",
        "name": "5 变体输出头迁移",
        "desc": "5 个模型变体的 output_head 替换为 SplineHead（已 V5A4 完成 ✅）",
        "args": "--module backbones",
        "level": "L2",
        "time": "~2 周",
        "gpu_mem": "~2-5GB（按变体）",
        "deps": ["V5A2", "V5A3"],
    },
    "V5A5": {
        "id": "V5A5",
        "name": "变长训练循环",
        "desc": "TrainerV5 + collate_v5 + 断点续训（已 V5A5 完成 ✅）",
        "args": "--module trainer",
        "level": "L2",
        "time": "~1 周",
        "gpu_mem": "~0GB（代码层）",
        "deps": ["V5A4"],
    },
    "V5A6": {
        "id": "V5A6",
        "name": "5 变体训练 + 评估",
        "desc": "5 变体 × 150 epoch 训练 + 评估报告",
        "args": "--variant all --epochs 150 --lr 1e-4",
        "level": "L3",
        "time": "~3 周",
        "gpu_mem": "~2-5GB（按变体）",
        "deps": ["V5A1", "V5A5"],
    },
    # ========== V5B 组（v5-β 占位任务 — 已填实） ==========
    "V5B3": {
        "id": "V5B3",
        "name": "裂纹标注工具（V5B3）",
        "desc": "crack_annotator.py（已填实，纯 Python CLI + matplotlib GUI）",
        "args": "--module annotator",
        "level": "L1",
        "time": "~3 天",
        "gpu_mem": "~0GB",
        "deps": [],
    },
    "V5B4": {
        "id": "V5B4",
        "name": "位置图像提取（V5B4）",
        "desc": "position_extractor.py（已填实，5 级降级 L0-L4）",
        "args": "--module position",
        "level": "L1",
        "time": "~3 天",
        "gpu_mem": "~0GB",
        "deps": [],
    },
    "V5B6": {
        "id": "V5B6",
        "name": "npz 磁盘缓存（V5B6）",
        "desc": "PatchDatasetV5 加 cache_dir 参数（已填实）",
        "args": "--module cache",
        "level": "L1",
        "time": "~2 天",
        "gpu_mem": "~0GB",
        "deps": [],
    },
    "V5B7": {
        "id": "V5B7",
        "name": "团队训练脚本（V5B7）",
        "desc": "team_train_v5.py（已填实，本文件）",
        "args": "--module team",
        "level": "L1",
        "time": "~2 天",
        "gpu_mem": "~0GB",
        "deps": [],
    },
}


# =============================================================================
# 核心函数（继承自 v4 + v5 适配）
# =============================================================================

def get_hardware_level():
    """检测 GPU 显存，返回硬件等级"""
    if not torch.cuda.is_available():
        return 'L1', 0.0

    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_mem = total_mem - torch.cuda.memory_allocated() / 1e9

    if total_mem >= 8:
        return 'L3', total_mem
    elif total_mem >= 6:
        return 'L2', total_mem
    elif total_mem >= 4:
        return 'L1+', total_mem
    else:
        return 'L1', total_mem


def _build_task_meta():
    """预提取每个任务的 (variant, subdir, predict_offset)"""
    out = {}
    for tid, t in all_tasks.items():
        args = t.get('args', '')
        out[tid] = (
            _extract_arg(args, '--variant'),
            t.get('subdir', ''),
            int(_extract_arg(args, '--predict_offset') or '0'),
        )
    return out


_TASK_META = {}


def _resolve_task_id(ckpt: dict, task_meta=None) -> str | None:
    """把检查点归到某个 task_id（task_id 严格匹配 → fuzzy）"""
    ck_tid = ckpt.get('task_id')
    if ck_tid in all_tasks:
        return ck_tid
    cfg = ckpt.get('config', {})
    ck_identity = (cfg.get('variant'), cfg.get('subdir', ''), cfg.get('predict_offset', 0))
    meta = task_meta if task_meta is not None else _TASK_META
    if not meta and all_tasks:
        meta = _TASK_META = _build_task_meta()
    for tid, identity in meta.items():
        if identity == ck_identity:
            return tid
    return None


def get_completed_tasks():
    """扫描 checkpoints/ 目录，识别已完成的任务"""
    global _TASK_META
    completed = set()
    if not CHECKPOINT_DIR.exists():
        return completed
    if not _TASK_META and all_tasks:
        _TASK_META = _build_task_meta()

    for ckpt_file in CHECKPOINT_DIR.rglob('*_best.pt'):
        if 'backup' in ckpt_file.parts:
            continue
        try:
            checkpoint = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            if not isinstance(checkpoint, dict):
                continue
            target_tid = _resolve_task_id(checkpoint, _TASK_META)
            if target_tid and (checkpoint.get('is_complete', False)
                               or checkpoint.get('is_retrain_done', False)):
                completed.add(target_tid)
        except Exception:
            pass

    return completed


def parse_task_id_from_filename(filename, all_tasks):
    """从文件名中解析 task_id（V5A*/V5B* 兼容）"""
    stem = Path(filename).stem
    patterns = [
        r'(V\d[A-Z]+_\d+)',    # V5A1_V5A6, V5B3
        r'(V\d[A-Z]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            candidate = match.group(1)
            if candidate in all_tasks:
                return candidate
    for task_id in all_tasks:
        if task_id in stem:
            return task_id
    return None


def check_deps_satisfied(task_id: str, completed: Set[str]) -> bool:
    """检查任务依赖是否满足"""
    task = all_tasks.get(task_id, {})
    deps = task.get('deps', [])
    return all(dep in completed for dep in deps)


def get_task_status(task_id: str, completed: Set[str], hardware_level: str) -> str:
    """获取任务状态"""
    if task_id in completed:
        return 'completed'
    if not check_deps_satisfied(task_id, completed):
        return 'locked'
    task = all_tasks.get(task_id, {})
    task_level = task.get('level', 'L1')
    levels = {'L1': 1, 'L1+': 2, 'L2': 3, 'L2+': 3, 'L3': 4}
    if levels.get(task_level, 0) > levels.get(hardware_level, 0):
        return 'warning'
    return 'executable'


def load_tasks_from_files():
    """从 tasks/ 目录加载 JSON 配置文件（v5-α：只加载 V5 前缀任务）

    注：v5 fork 自 v4，tasks/ 目录可能残留 v4 任务文件
    （team_optimization.json, team_baseline.json）。本函数只加载 V5* 前缀任务，
    忽略 v4 的 A/B/C 任务。
    """
    tasks = {}
    if not TASKS_DIR.exists():
        return tasks
    for json_file in TASKS_DIR.glob('*.json'):
        if json_file.name in ('examples.json', 'example.json', 'template.json'):
            continue
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, list):
                for task in data:
                    if 'id' in task:
                        # v5-α 过滤：只接受 V5* 前缀
                        if str(task['id']).startswith('V5'):
                            tasks[task['id']] = task
            elif isinstance(data, dict) and 'tasks' in data:
                if 'id' in data and str(data['id']).startswith('V5'):
                    tasks[data['id']] = data
        except Exception as e:
            print(f"{COLORS['yellow']}[警告]{COLORS['reset']} 加载 {json_file.name} 失败: {e}")
    return tasks


def merge_tasks(external: Dict, defaults: Dict) -> Dict:
    """合并外部配置和内置默认任务，外部优先"""
    merged = defaults.copy()
    merged.update(external)
    return merged


def topological_sort(tasks: Dict, completed: Set[str]) -> list:
    """按依赖拓扑排序"""
    result = []
    remaining = {k: v for k, v in tasks.items() if k not in completed}
    while remaining:
        executable = [
            k for k, v in remaining.items()
            if all(dep in completed or dep not in tasks for dep in v.get('deps', []))
        ]
        if not executable:
            break
        for task_id in executable:
            result.append(task_id)
            completed.add(task_id)
            del remaining[task_id]
    return result


def run_training_task(task_id: str) -> bool:
    """执行单个训练任务"""
    task = all_tasks.get(task_id)
    if not task:
        print(f"{COLORS['red']}[错误]{COLORS['reset']} 未知任务: {task_id}")
        return False

    print(f"\n{'=' * 60}")
    print(f"开始训练: {COLORS['bold']}{task['name']}{COLORS['reset']}")
    print(f"描述: {task['desc']}")
    print(f"预计时间: {task['time']}")
    print(f"任务ID: {task_id}")
    print(f"{'=' * 60}\n")

    # 特殊处理：V5A* 模块级任务（无 --module 参数直接执行测试）
    args = task.get('args', '')
    if '--module' in args:
        # 模块任务：直接跑对应测试
        module_name = _extract_arg(args, '--module')
        if module_name:
            return _run_module_test(module_name, task_id)

    # 正常训练任务：调用 run_train_v5.py
    subdir = task.get('subdir', '')
    py_cmd, py_desc = _resolve_python_interpreter()
    py_exe = py_cmd.strip('"')
    args_list = [py_exe, str(RUN_TRAIN)] + args.split() + \
                ['--task_id', task_id] + (['--subdir', subdir] if subdir else [])
    print(f"执行命令: {' '.join(args_list)}")
    print(f"解释器:   {py_desc}\n")

    import subprocess
    result = subprocess.run(args_list, shell=False).returncode

    if result == 0:
        print(f"\n{COLORS['green']}[完成]{COLORS['reset']} {task['name']} 训练完成!")
        return True
    else:
        print(f"\n{COLORS['red']}[失败]{COLORS['reset']} {task['name']} 训练失败 (错误码: {result})")
        return False


# V5-α 模块任务映射（无 GPU 训练，直接跑对应单元测试）
_V5_MODULE_TEST_MAP = {
    "simulator": ("tests/test_patch_simulator_v5.py",),
    "heads": ("tests/test_spline_head.py", "tests/test_gp_module.py"),
    "loss": ("tests/test_ordered_kp_loss.py",),
    "backbones": ("tests/test_pe_tsnet_v5.py",),
    "trainer": ("tests/test_trainer_v5_smoke.py",),
    "annotator": ("tools/crack_annotator.py",),  # 跑冒烟测试
    "position": ("data/position_extractor.py",),  # 跑冒烟测试
    "cache": ("data/patch_dataset_v5.py",),      # 跑冒烟测试
    "team": (None,),  # team_train_v5.py 自身（无需测）
}


def _run_module_test(module_name: str, task_id: str) -> bool:
    """运行 V5-α 模块的对应单元测试"""
    if module_name not in _V5_MODULE_TEST_MAP:
        print(f"  [跳过] 未知模块 {module_name!r}")
        return True
    test_files = _V5_MODULE_TEST_MAP[module_name]
    py_cmd, py_desc = _resolve_python_interpreter()
    py_exe = py_cmd.strip('"')

    all_passed = True
    for tf in test_files:
        if tf is None:
            continue
        test_path = SCRIPT_DIR / tf
        if not test_path.exists():
            print(f"  [警告] 测试文件不存在: {tf}")
            continue
        print(f"  [测试] {tf}")
        result = subprocess.run(  # noqa: F821
            [py_exe, str(test_path)], shell=False,
            capture_output=True, text=True, cwd=str(SCRIPT_DIR),
        ).returncode
        if result == 0:
            print(f"  {COLORS['green']}[PASS]{COLORS['reset']} {tf}")
        else:
            print(f"  {COLORS['red']}[FAIL]{COLORS['reset']} {tf} (exit={result})")
            all_passed = False

    if all_passed:
        print(f"\n{COLORS['green']}[完成]{COLORS['reset']} {task_id} 模块测试全过 ✅")
        return True
    else:
        print(f"\n{COLORS['red']}[失败]{COLORS['reset']} {task_id} 部分模块测试失败")
        return False


def _get_task_checkpoint_info(task_id: str) -> dict:
    """获取任务的检查点信息"""
    if not CHECKPOINT_DIR.exists():
        return {'epoch': 0, 'is_complete': False, 'best_epoch': 0,
                'last_epoch': 0, 'path': None}

    global _TASK_META
    best_epoch = 0
    last_epoch = 0
    best_path = None
    last_path = None
    is_complete = False

    if not _TASK_META and all_tasks:
        _TASK_META = _build_task_meta()

    for ckpt_file in CHECKPOINT_DIR.rglob('*_best.pt'):
        if 'backup' in ckpt_file.parts:
            continue
        try:
            ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and _resolve_task_id(ckpt, _TASK_META) == task_id:
                ep = ckpt.get('epoch', 0)
                if ep > best_epoch:
                    best_epoch = ep
                    best_path = str(ckpt_file)
                if ckpt.get('is_complete', False) or ckpt.get('is_retrain_done', False):
                    is_complete = True
        except Exception:
            pass

    for ckpt_file in CHECKPOINT_DIR.rglob('*_last.pt'):
        if 'backup' in ckpt_file.parts:
            continue
        try:
            ckpt = torch.load(ckpt_file, map_location='cpu', weights_only=False)
            if isinstance(ckpt, dict) and _resolve_task_id(ckpt, _TASK_META) == task_id:
                ep = ckpt.get('epoch', 0)
                if ep > last_epoch:
                    last_epoch = ep
                    last_path = str(ckpt_file)
                if ckpt.get('is_complete', False) or ckpt.get('is_retrain_done', False):
                    is_complete = True
        except Exception:
            pass

    return {
        'epoch': max(best_epoch, last_epoch),
        'best_epoch': best_epoch,
        'last_epoch': last_epoch,
        'is_complete': is_complete,
        'path': last_path or best_path,
    }


def _extract_arg(args_str: str, arg: str) -> str | None:
    """从 args 字符串中提取指定参数的值"""
    pattern = re.compile(re.escape(arg) + r'\s+(\S+)')
    match = pattern.search(args_str)
    return match.group(1) if match else None


def _get_task_depth(task_id, visited=None) -> int:
    """计算任务依赖深度"""
    if visited is None:
        visited = set()
    if task_id in visited:
        return 0
    visited.add(task_id)
    task = all_tasks.get(task_id, {})
    deps = task.get('deps', [])
    if not deps:
        return 0
    return 1 + max((_get_task_depth(d, visited) for d in deps), default=0)


def _print_progress_table(title=None, last_done=None, newly_unlocked=None):
    """打印实时任务进度表"""
    completed = get_completed_tasks()
    hardware_level, _ = get_hardware_level()

    task_info = {}
    for task_id in all_tasks:
        status = get_task_status(task_id, completed, hardware_level)
        ckpt_info = _get_task_checkpoint_info(task_id)
        depth = _get_task_depth(task_id)
        task_info[task_id] = {
            'status': status,
            'depth': depth,
            'epoch': ckpt_info['epoch'],
            'best_epoch': ckpt_info.get('best_epoch', 0),
            'last_epoch': ckpt_info.get('last_epoch', 0),
            'is_complete': ckpt_info['is_complete'],
        }

    stat_done = 0
    stat_exec = 0
    stat_lock = 0
    stat_interrupted = 0
    for tid, i in task_info.items():
        if i['is_complete']:
            stat_done += 1
        elif i['status'] == 'locked':
            stat_lock += 1
        elif i['last_epoch'] == 0 and i['status'] in ('executable', 'warning'):
            stat_exec += 1
        elif i['last_epoch'] > 0 and i['status'] != 'locked':
            stat_interrupted += 1

    sep = f"{'─' * 100}"
    print(f"\n{sep}")
    if title:
        print(f"  {title}")
    else:
        print(f"  团队训练进度 ({stat_done}/{len(all_tasks)} 任务)")
    print(sep)

    if title:
        print(f"{COLORS['green']}✓已完成{COLORS['reset']}  "
              f"{COLORS['yellow']}⏸中断{COLORS['reset']}  "
              f"{COLORS['blue']}▶待执行{COLORS['reset']}  "
              f"{COLORS['gray']}🔒等待依赖{COLORS['reset']}")

    max_depth = max(t['depth'] for t in task_info.values()) if task_info else 0
    for depth in range(max_depth + 1):
        tasks_at_depth = sorted(
            [(tid, info) for tid, info in task_info.items() if info['depth'] == depth],
            key=lambda x: x[0]
        )
        for task_id, info in tasks_at_depth:
            task = all_tasks[task_id]
            deps = task.get('deps', [])
            deps_str = ', '.join(deps) if deps else '无'

            target_epochs = int(_extract_arg(task.get('args', ''), '--epochs') or 0)
            is_done = info['is_complete']
            is_newly = task_id in (newly_unlocked or [])
            is_last = task_id == last_done

            if info['best_epoch'] > 0 or info['last_epoch'] > 0:
                t = target_epochs
                ckpt_txt = f"best:{info['best_epoch']:3d} last:{info['last_epoch']:3d} target:{t:3d}"
            else:
                ckpt_txt = ""

            if is_done:
                icon = f"{COLORS['green']}✓{COLORS['reset']}"
                state_txt = f"{COLORS['green']}已完成{COLORS['reset']}"
            elif info['status'] == 'locked':
                icon = f"{COLORS['gray']}🔒{COLORS['reset']}"
                state_txt = f"{COLORS['gray']}等待依赖{COLORS['reset']}"
                ckpt_txt = ""
            elif info['last_epoch'] > 0:
                icon = f"{COLORS['yellow']}⏸{COLORS['reset']}"
                state_txt = f"{COLORS['yellow']}中断{COLORS['reset']}"
            elif info['status'] == 'warning':
                icon = f"{COLORS['yellow']}!{COLORS['reset']}"
                state_txt = f"{COLORS['blue']}待执行{COLORS['reset']}"
                ckpt_txt = "新任务"
            elif info['status'] == 'executable':
                icon = f"{COLORS['blue']}▶{COLORS['reset']}"
                state_txt = f"{COLORS['blue']}待执行{COLORS['reset']}"
                ckpt_txt = "新任务"
            else:
                icon = f"{COLORS['gray']}?{COLORS['reset']}"
                state_txt = f"{COLORS['gray']}未知{COLORS['reset']}"
                ckpt_txt = ""

            marker = ""
            if is_last:
                marker = f" {COLORS['green']}[刚完成]{COLORS['reset']}"
            elif is_newly:
                marker = f" {COLORS['blue']}[刚解锁]{COLORS['reset']}"

            def display_len(s):
                plain = re.sub(r'\x1b\[[0-9;]*m', '', s)
                return len(plain) + len(re.findall(r'[一-鿿]', plain))

            def pad(s, width):
                dl = display_len(s)
                return s + ' ' * max(0, width - dl)

            def truncate_by_display(text, max_w):
                result = []
                cur = 0
                for ch in text:
                    if '一' <= ch <= '鿿':
                        cur += 2
                    else:
                        cur += 1
                    if cur > max_w:
                        break
                    result.append(ch)
                return ''.join(result)

            name = truncate_by_display(task['name'], 26)
            state_col = 12
            ckpt_col = 22
            print(f"  {icon} [{task_id}] {pad(name, 26)} {pad(state_txt, state_col)} {pad(ckpt_txt, ckpt_col)}  依赖: [{deps_str}]{marker}")

    print(sep)
    stats_parts = []
    if stat_done > 0:
        stats_parts.append(f"{COLORS['green']}✓{stat_done}{COLORS['reset']} 已完成")
    if stat_interrupted > 0:
        stats_parts.append(f"{COLORS['yellow']}⏸{stat_interrupted}{COLORS['reset']} 中断")
    if stat_exec > 0:
        stats_parts.append(f"{COLORS['blue']}▶{stat_exec}{COLORS['reset']} 待执行")
    if stat_lock > 0:
        stats_parts.append(f"{COLORS['gray']}🔒{stat_lock}{COLORS['reset']} 等待")
    print(f"  统计: {'  '.join(stats_parts)}")
    print(sep)


def auto_run_executable(force_warnings=False):
    """自动执行所有可执行任务（动态拓扑重排）"""
    hardware_level, _ = get_hardware_level()
    total_tasks = len(all_tasks)

    def get_next_task():
        completed = get_completed_tasks()
        interrupted, executable = [], []
        for task_id in all_tasks:
            status = get_task_status(task_id, completed, hardware_level)
            if status not in ('executable', 'warning'):
                continue
            ckpt_info = _get_task_checkpoint_info(task_id)
            if ckpt_info['epoch'] > 0 and not ckpt_info['is_complete']:
                interrupted.append(task_id)
            else:
                executable.append(task_id)
        if interrupted:
            return sorted(interrupted)[0], 'executable'
        if executable:
            return sorted(executable)[0], 'executable'
        return None, None

    first_task, _ = get_next_task()
    if first_task is None:
        completed = get_completed_tasks()
        print(f"\n{COLORS['yellow']}[信息]{COLORS['reset']} 没有可执行的任务 (已完成: {len(completed)}/{total_tasks})")
        return

    mode_str = "强制" if force_warnings else "标准"
    print(f"\n{COLORS['bold']}自动执行模式 ({mode_str}){COLORS['reset']} | "
          f"总任务: {total_tasks} | 每次任务完成后刷新进度\n")

    _print_progress_table(title=f"v5-α 团队训练进度 ({total_tasks} 任务)",
                          last_done=None, newly_unlocked=None)

    success_count, fail_count, skip_count = 0, 0, 0
    started_tasks = set()
    task_idx = 0
    last_done = None

    while True:
        task_id, status = get_next_task()
        if task_id is None:
            break

        task = all_tasks[task_id]
        task_idx += 1
        started_tasks.add(task_id)

        if status == 'warning':
            if not force_warnings:
                print(f"\n{COLORS['yellow']}[警告]{COLORS['reset']} 任务 {task['name']} 需要更多显存")
                confirm = input("是否继续? (y/n): ").strip().lower()
                if confirm != 'y':
                    print(f"跳过 {task['name']}")
                    log_task_execution(task_id, 'skipped', error='用户取消')
                    skip_count += 1
                    continue
            else:
                print(f"\n{COLORS['yellow']}[强制执行]{COLORS['reset']} {task['name']} (显存可能不足)")

        print(f"\n{'=' * 60}")
        print(f"[{task_idx}/{total_tasks}] 执行: {task['name']}")
        print(f"{'=' * 60}")

        log_task_execution(task_id, 'started')
        task_start = datetime.now()

        if run_training_task(task_id):
            success_count += 1
            duration = (datetime.now() - task_start).total_seconds()
            log_task_execution(task_id, 'completed', duration_seconds=duration)
            last_done = task_id
        else:
            fail_count += 1
            duration = (datetime.now() - task_start).total_seconds()
            log_task_execution(task_id, 'failed', duration_seconds=duration)
            last_done = None
            if not force_warnings:
                retry = input("\n训练失败，是否继续下一个任务? (y/n): ").strip().lower()
                if retry != 'y':
                    break
            continue

        completed = get_completed_tasks()
        newly_unlocked = []
        for tid in all_tasks:
            old_status = get_task_status(tid, completed - {task_id}, hardware_level)
            new_status = get_task_status(tid, completed, hardware_level)
            if old_status == 'locked' and new_status in ('executable', 'warning'):
                newly_unlocked.append(tid)

        _print_progress_table(title=f"任务完成，刷新进度", last_done=last_done, newly_unlocked=newly_unlocked)
        last_done = None

    print(f"\n{'=' * 60}")
    print(f"自动执行完成: {COLORS['green']}{success_count} 成功{COLORS['reset']}, "
          f"{COLORS['red']}{fail_count} 失败{COLORS['reset']}, "
          f"{COLORS['gray']}{skip_count} 跳过{COLORS['reset']}")
    print(f"{'=' * 60}")


def import_checkpoint():
    """导入队友的检查点"""
    print(f"\n{COLORS['bold']}导入检查点 (v5-α){COLORS['reset']}")
    print(f"检查点目录: {CHECKPOINT_DIR}")

    source_dir = input("\n请输入队友检查点所在的文件夹路径 (回车取消): ").strip()
    if not source_dir:
        return

    source_path = Path(source_dir)
    if not source_path.exists():
        print(f"{COLORS['red']}[错误]{COLORS['reset']} 目录不存在")
        return

    imported = 0
    skipped = 0
    for pt_file in source_path.glob('*.pt'):
        dest_file = CHECKPOINT_DIR / pt_file.name
        if dest_file.exists():
            print(f"跳过 (已存在): {pt_file.name}")
            skipped += 1
            continue

        shutil.copy2(pt_file, dest_file)
        print(f"{COLORS['green']}[导入]{COLORS['reset']} {pt_file.name}")
        imported += 1

    print(f"\n{COLORS['green']}[完成]{COLORS['reset']} 导入了 {imported} 个检查点"
          + (f"，跳过 {skipped} 个已存在" if skipped > 0 else ""))


def print_menu(completed: Set[str], hardware_level: str):
    """显示任务菜单"""
    print(f"\n{COLORS['bold']}{'=' * 60}")
    print("  PE-MMNet v5-α 团队协作训练")
    print(f"{'=' * 60}{COLORS['reset']}")

    _, gpu_mem = get_hardware_level()
    print(f"\n当前硬件等级: {COLORS['blue']}{hardware_level}{COLORS['reset']} (显存: {gpu_mem:.1f}GB)")
    print(f"已完成任务: {len(completed)}/{len(all_tasks)}")

    print(f"\n状态标记: {COLORS['green']}[OK] 已完成{COLORS['reset']} | "
          f"{COLORS['blue']}[>] 可执行{COLORS['reset']} | "
          f"{COLORS['yellow']}[!] 硬件警告{COLORS['reset']} | "
          f"{COLORS['gray']}[X] 依赖未完成{COLORS['reset']}")

    print()

    levels = {'L1': [], 'L1+': [], 'L2': [], 'L2+': [], 'L3': []}
    for task_id in all_tasks:
        task = all_tasks[task_id]
        level = task.get('level', 'L1')
        if level in levels:
            levels[level].append(task_id)

    idx = 1
    task_index = {}
    for level in ['L1', 'L1+', 'L2', 'L2+', 'L3']:
        if not levels[level]:
            continue
        print(f"  【{level}级别】")

        for task_id in levels[level]:
            task = all_tasks[task_id]
            status = get_task_status(task_id, completed, hardware_level)

            status_symbols = {
                'completed': (f"{COLORS['green']}OK{COLORS['reset']}", COLORS['green']),
                'executable': (f"{COLORS['blue']}>{COLORS['reset']}", COLORS['blue']),
                'warning': (f"{COLORS['yellow']}!{COLORS['reset']}", COLORS['yellow']),
                'locked': (f"{COLORS['gray']}X{COLORS['reset']}", COLORS['gray']),
            }
            prefix, color = status_symbols[status]

            print(f"  [{idx:>2}] {prefix} {color}{task['name']}{COLORS['reset']}")
            print(f"       {task['desc']}")
            level_str = task.get('level', 'N/A')
            time_str = task.get('time', 'N/A')
            gpu_mem_str = task.get('gpu_mem', 'N/A')
            print(f"       [L] {level_str} | [T] {time_str} | [G] {gpu_mem_str}")

            if status == 'locked':
                deps = task.get('deps', [])
                print(f"       {COLORS['gray']}依赖: {', '.join(deps)}{COLORS['reset']}")

            if status == 'warning':
                print(f"       {COLORS['yellow']}⚠️ 建议显存 ≥ {task['gpu_mem']}{COLORS['reset']}")

            task_index[idx] = task_id
            idx += 1

        print()

    print("-" * 60)
    print(f"  [a]   自动执行所有可执行任务")
    print(f"  [i]   导入队友检查点")
    print(f"  [q]   退出")
    print("-" * 60)

    return task_index


def main():
    global all_tasks

    import argparse

    parser = argparse.ArgumentParser(description='PE-MMNet v5-α 团队协作训练系统')
    parser.add_argument('--auto', action='store_true', help='自动执行所有可执行任务')
    parser.add_argument('--force', action='store_true', help='强制执行硬件警告任务（auto模式）')
    parser.add_argument('--import', dest='import_mode', action='store_true', help='导入队友检查点')
    parser.add_argument('--list-tasks', action='store_true', help='只列出任务清单')
    args = parser.parse_args()

    external_tasks = load_tasks_from_files()
    all_tasks = merge_tasks(external_tasks, DEFAULT_V5_TRAIN_TASKS)

    hardware_level, _ = get_hardware_level()

    if external_tasks:
        print(f"{COLORS['blue']}[信息]{COLORS['reset']} 已加载 {len(external_tasks)} 个外部任务")
        print(f"{COLORS['blue']}[信息]{COLORS['reset']} 配置来源: {TASKS_DIR}")
    else:
        print(f"{COLORS['yellow']}[信息]{COLORS['reset']} 使用内置默认任务列表 (V5A1-V5B7)")

    if args.list_tasks:
        print(f"\n  任务清单 ({len(all_tasks)} 个):")
        for tid in sorted(all_tasks.keys()):
            t = all_tasks[tid]
            print(f"    {tid:6s} | L{t.get('level', '?')} | {t.get('name')} | deps={t.get('deps', [])}")
        return

    if args.auto:
        auto_run_executable(force_warnings=args.force)
        return

    if args.import_mode:
        import_checkpoint()
        return

    # 交互式菜单模式
    while True:
        completed = get_completed_tasks()
        task_index = print_menu(completed, hardware_level)

        choice = input("\n请输入选项: ").strip().lower()

        if choice == 'q':
            print("\n已退出")
            break

        if choice == 'a':
            auto_run_executable()
            continue

        if choice == 'i':
            import_checkpoint()
            continue

        if choice.isdigit():
            idx = int(choice)
            if idx in task_index:
                task_id = task_index[idx]
                status = get_task_status(task_id, completed, hardware_level)

                if status == 'completed':
                    print(f"\n{COLORS['yellow']}[跳过]{COLORS['reset']} "
                          f"{all_tasks[task_id]['name']} 已完成")
                    continue

                if status == 'locked':
                    deps = all_tasks[task_id].get('deps', [])
                    print(f"\n{COLORS['red']}[锁定]{COLORS['reset']} "
                          f"请先完成以下依赖任务: {', '.join(deps)}")
                    continue

                if status == 'warning':
                    print(f"\n{COLORS['yellow']}[警告]{COLORS['reset']} "
                          f"{all_tasks[task_id]['name']} 可能需要更多显存")
                    confirm = input("是否继续? (y/n): ").strip().lower()
                    if confirm != 'y':
                        continue

                run_training_task(task_id)
            else:
                print(f"\n{COLORS['red']}[错误]{COLORS['reset']} 无效选项")
        else:
            print(f"\n{COLORS['red']}[错误]{COLORS['reset']} 无效选项")


if __name__ == '__main__':
    main()