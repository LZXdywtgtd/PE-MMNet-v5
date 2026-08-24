#!/usr/bin/env python3
"""
PE-MMNet v4 团队协作训练系统

功能：
1. 配置驱动：任务由 JSON 配置文件定义
2. 依赖管理：任务间可定义前置依赖
3. 硬件感知：根据显存推荐任务等级
4. 状态可视化：已完成/可执行/警告/锁定 状态显示
5. 检查点导入：支持导入队友的 .pt 文件
6. 自动批量执行：按拓扑排序执行所有可执行任务
7. 任务日志：记录每个任务的执行情况

使用方法：
    python team_train.py              # 交互式菜单
    python team_train.py --auto      # 自动执行所有可执行任务
    python team_train.py --auto --force  # 自动执行（包括硬件警告任务）
    python team_train.py --import   # 导入队友检查点
"""

import os
import sys
import json
import shutil
import socket
import re
from pathlib import Path
from typing import Dict, Set, Optional
from datetime import datetime
import torch

# =============================================================================
# 颜色支持（增强版）
# =============================================================================

def get_color_support():
    """检测终端颜色支持"""
    # 1. 检查 NO_COLOR 环境变量
    if os.environ.get('NO_COLOR'):
        return 'none'

    # 2. 检查 colorama
    try:
        import io
        if sys.platform == 'win32':
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        from colorama import init, Fore, Style
        init(autoreset=True, strip=False)
        return 'colorama'
    except ImportError:
        pass

    # 3. 检测 ANSI 支持（Windows 10+）
    if sys.platform == 'win32':
        try:
            version = sys.getwindowsversion()
            if version.major >= 10:
                return 'ansi'
        except:
            pass

    # 4. 检测 TERM
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
    # 无颜色模式
    COLORS = {k: '' for k in ['green', 'blue', 'yellow', 'gray', 'red', 'bold', 'reset']}

# 路径配置
SCRIPT_DIR = Path(__file__).parent
TASKS_DIR = SCRIPT_DIR / 'tasks'
CHECKPOINT_DIR = SCRIPT_DIR / 'checkpoints'
LOG_DIR = SCRIPT_DIR / 'logs'
RUN_TRAIN = SCRIPT_DIR / 'run_train.py'


def _resolve_python_interpreter() -> tuple:
    """选择可用的 Python 解释器，优先 conda 内的 sys.executable，回退到 PATH 上的 python/py。

    Returns:
        (cmd_token, description)：
        - cmd_token 是用于 shell 命令的字符串：绝对路径会用引号包裹以兼容空格路径
          （如 C:\\Program Files\\...），可执行名（python/py）直接原样
        - description 是给人看的描述
    """
    import shutil as _shutil

    # 1. 优先用当前进程的解释器（conda/venv 内必中）
    exe = sys.executable
    if exe and os.path.isfile(exe):
        return f'"{exe}"', f'当前环境 ({exe})'

    # 2. PATH 上找 python（标准 CPython 装在系统里的常见情况）
    for name in ('python', 'python3', 'py'):
        found = _shutil.which(name)
        if found:
            # py 启动器在 Windows 上是 Microsoft Store 别名，要避免
            if sys.platform == 'win32' and name == 'py':
                try:
                    import subprocess
                    r = subprocess.run(
                        ['py', '-c', 'import sys; print(sys.executable)'],
                        capture_output=True, text=True, timeout=5,
                    )
                    real = r.stdout.strip()
                    # Store 别名会返回 Microsoft Store 的路径
                    if 'WindowsApps' in real or 'Microsoft' in real:
                        continue
                except Exception:
                    continue
            return f'"{found}"' if ' ' in found else found, f'PATH 找到 {name} ({found})'

    # 3. 实在没有，至少给个能跑出错的命令
    return 'python', '默认（PATH 兜底，可能未安装）'

# 全局任务字典
all_tasks: Dict = {}

# =============================================================================
# 任务执行日志
# =============================================================================

def log_task_execution(task_id, status, duration_seconds=None, error=None):
    """记录任务执行日志

    Args:
        task_id: 任务ID
        status: 'started', 'completed', 'failed', 'skipped'
        duration_seconds: 执行时长（秒）
        error: 错误信息
    """
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / 'team_training.log'

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
# 内置默认任务
# =============================================================================

DEFAULT_TRAIN_TASKS = {
    # ========== A组：基线变体 ==========
    "A1": {
        "id": "A1",
        "name": "ResNet18基线",
        "desc": "轻量级CNN骨干，所有优化的基准线",
        "args": "--variant resnet18 --epochs 150 --patience 30",
        "level": "L1",
        "time": "~12小时",
        "gpu_mem": "~2GB",
        "deps": [],
    },
    "A2": {
        "id": "A2",
        "name": "Swin-YOLO基线",
        "desc": "推荐！Swin-Tiny骨干+YOLO网格回归，大幅提升空间定位",
        "args": "--variant swin_yolo --epochs 150 --patience 30 --lr 1e-4",
        "level": "L1+",
        "time": "~18小时",
        "gpu_mem": "~4GB",
        "deps": [],
    },
    "A3": {
        "id": "A3",
        "name": "ViT-YOLO基线",
        "desc": "基于ViT-Small的YOLO变体，验证Transformer做骨干的效果",
        "args": "--variant vit_yolo --epochs 150 --patience 30 --lr 1e-4",
        "level": "L1",
        "time": "~12小时",
        "gpu_mem": "~3GB",
        "deps": [],
    },
    "A4": {
        "id": "A4",
        "name": "DETR基线",
        "desc": "纯Transformer的DETR端到端检测，验证全局上下文感知",
        "args": "--variant detr --epochs 150 --patience 30 --lr 1e-4",
        "level": "L1+",
        "time": "~18小时",
        "gpu_mem": "~5GB",
        "deps": [],
    },
    "A5": {
        "id": "A5",
        "name": "Swin-PatchTST",
        "desc": "升级1D时序骨干为PatchTST，提升热应力时序建模能力",
        "args": "--variant swin_yolo_patchtst --epochs 150 --patience 30 --lr 1e-4",
        "level": "L2",
        "time": "~20小时",
        "gpu_mem": "~4GB",
        "deps": [],
    },

    # ========== B组：单项优化 ==========
    "B1": {
        "id": "B1",
        "name": "门控融合优化",
        "desc": "基于ResNet18：温度/应力通道分治门控融合",
        "args": "--variant resnet18 --fusion gated --epochs 150 --patience 30",
        "level": "L2",
        "time": "~15小时",
        "gpu_mem": "~2GB",
        "deps": ["A1"],
    },
    "B2": {
        "id": "B2",
        "name": "坐标注意力优化",
        "desc": "基于ResNet18：引入空间坐标注意力模块(CoordAtt)",
        "args": "--variant resnet18 --use_coord_attn --epochs 150 --patience 30",
        "level": "L2",
        "time": "~15小时",
        "gpu_mem": "~2GB",
        "deps": ["A1"],
    },
    "B3": {
        "id": "B3",
        "name": "分阶段训练优化",
        "desc": "基于ResNet18：先短序列预训练，后长序列微调",
        "args": "--variant resnet18 --staged_train --epochs 150 --patience 30",
        "level": "L2+",
        "time": "~20小时",
        "gpu_mem": "~2GB",
        "deps": ["A1"],
    },
    "B4": {
        "id": "B4",
        "name": "三通道输入优化",
        "desc": "基于ResNet18：增加初始温度、温度变化率通道",
        "args": "--variant resnet18 --triple_channel --epochs 150 --patience 30",
        "level": "L2",
        "time": "~15小时",
        "gpu_mem": "~2GB",
        "deps": ["A1"],
    },
    "B5": {
        "id": "B5",
        "name": "ThermalCutMix增强",
        "desc": "基于ResNet18：引入物理安全的温度场CutMix数据增强",
        "args": "--variant resnet18 --aug_cutmix_prob 0.3 --epochs 150 --patience 30",
        "level": "L2",
        "time": "~15小时",
        "gpu_mem": "~2GB",
        "deps": ["A1"],
    },

    # ========== C组：组合优化 ==========
    "C1": {
        "id": "C1",
        "name": "门控+坐标注意力",
        "desc": "组合优化：门控融合+坐标注意力",
        "args": "--variant resnet18 --fusion gated --use_coord_attn --epochs 150 --patience 30",
        "level": "L2",
        "time": "~18小时",
        "gpu_mem": "~2GB",
        "deps": ["B1", "B2"],
    },
    "C2": {
        "id": "C2",
        "name": "分阶段+坐标注意力",
        "desc": "组合优化：分阶段训练+坐标注意力",
        "args": "--variant resnet18 --staged_train --use_coord_attn --epochs 150 --patience 30",
        "level": "L2+",
        "time": "~24小时",
        "gpu_mem": "~2GB",
        "deps": ["B2", "B3"],
    },
    "C3": {
        "id": "C3",
        "name": "全组合优化",
        "desc": "终极组合：门控+坐标注意力+分阶段训练，最强性能验证",
        "args": "--variant resnet18 --fusion gated --use_coord_attn --staged_train --epochs 150 --patience 30",
        "level": "L3",
        "time": "~24小时",
        "gpu_mem": "~2GB",
        "deps": ["C1", "C2"],
    },
}

# =============================================================================
# 核心函数
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
    """预提取每个任务的 (variant, subdir, predict_offset)，供 fuzzy match 使用。

    all_tasks 字典在 main() 中填充后调用本函数刷新缓存。O(K) 一次性成本，
    替代 get_completed_tasks / _get_task_checkpoint_info 内部的 O(K×M) 重复抽取。
    """
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
    """把检查点归到某个 task_id：先按 task_id 严格匹配，失败则按 (variant, subdir, predict_offset) 模糊匹配。

    Args:
        ckpt: 检查点 dict
        task_meta: 预提取的任务元数据（_TASK_META），传 None 时内部临时构建
    Returns:
        匹配的 task_id；都不匹配则返回 None
    """
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
    """扫描 checkpoints/ 目录（含子目录），识别已完成的任务。

    跳过 backup/ 目录。
    完成判定：is_complete=True 或 is_retrain_done=True。

    匹配策略（按优先级）：
    1. 元数据 task_id 严格匹配
    2. （variant + subdir + predict_offset）模糊匹配
       —— 用于任务合并/重命名场景（如 A4_long → A4）
    """
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
            if target_tid and (checkpoint.get('is_complete', False) or checkpoint.get('is_retrain_done', False)):
                completed.add(target_tid)
        except Exception:
            pass

    return completed


def parse_task_id_from_filename(filename, all_tasks):
    """从文件名中解析 task_id（增强版）

    支持格式：
    - checkpoint_A1_best.pt           → A1
    - resnet18_A1_best.pt             → A1
    - resnet18_detection_off0_best.pt → 尝试匹配 A1 等任务ID
    - A1_best.pt                      → A1

    Args:
        filename: 检查点文件名
        all_tasks: 所有任务的字典

    Returns:
        task_id 或 None
    """
    stem = Path(filename).stem  # 去掉扩展名

    # 方法1: 使用正则表达式精确提取
    # 查找 TASK_ID 格式：字母+下划线+数字/字母组合
    patterns = [
        r'([A-Z][A-Z0-9]*_\d+)',      # BASELINE_1, OPT_GATED_1
        r'([A-Z]+)_\d+',               # BASELINE, OPT (后面跟数字)
    ]

    for pattern in patterns:
        match = re.search(pattern, stem)
        if match:
            candidate = match.group(1)
            if candidate in all_tasks:
                return candidate

    # 方法2: 子字符串匹配（作为备用）
    for task_id in all_tasks:
        # 使用单词边界匹配，避免误匹配
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
    """从 tasks/ 目录加载 JSON 配置文件"""
    tasks = {}
    if not TASKS_DIR.exists():
        return tasks

    for json_file in TASKS_DIR.glob('*.json'):
        # 跳过模板参考文件
        if json_file.name in ('examples.json', 'example.json', 'template.json'):
            continue
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            if isinstance(data, list):
                for task in data:
                    if 'id' in task:
                        tasks[task['id']] = task
            elif isinstance(data, dict) and 'tasks' in data:
                if 'id' in data:
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
    """按依赖拓扑排序返回可执行任务列表"""
    result = []
    remaining = {k: v for k, v in tasks.items() if k not in completed}

    while remaining:
        # 找出入度为0的任务
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

    # 添加 --task_id 和 --subdir 参数以便检查点记录任务ID和路径
    subdir = task.get('subdir', '')
    # 解释器选择：自动适配 conda / 系统 python / py 启动器（含空格路径安全）
    py_cmd, py_desc = _resolve_python_interpreter()
    # 拼装为参数列表（避免 os.system + cmd.exe 在 Windows 上把路径里的 \ 误作转义）
    # py_cmd 可能是带引号的字符串（"---\conda\python.exe"），先去引号得到裸命令
    py_exe = py_cmd.strip('"')
    args = [py_exe, str(RUN_TRAIN), '--mode', 'train'] + task['args'].split() + \
           ['--task_id', task_id] + (['--subdir', subdir] if subdir else [])
    print(f"执行命令: {' '.join(args)}")
    print(f"解释器:   {py_desc}\n")

    import subprocess
    result = subprocess.run(args, shell=False).returncode

    if result == 0:
        print(f"\n{COLORS['green']}[完成]{COLORS['reset']} {task['name']} 训练完成!")
        return True
    else:
        print(f"\n{COLORS['red']}[失败]{COLORS['reset']} {task['name']} 训练失败 (错误码: {result})")
        return False


def _get_task_checkpoint_info(task_id: str) -> dict:
    """获取任务的检查点信息（epoch、is_complete）

    扫描所有 checkpoint 文件，通过元数据 task_id 匹配。
    is_complete 判定：is_complete=True 或 is_retrain_done=True（有 epoch 记录）。

    匹配策略（按优先级）：
    1. 元数据 task_id 完全匹配
    2. （variant + subdir + predict_offset）一致
       —— 用于任务合并场景（如 A4_long → A4 合并后，老检查点认主）
    """
    if not CHECKPOINT_DIR.exists():
        return {'epoch': 0, 'is_complete': False, 'best_epoch': 0, 'last_epoch': 0, 'path': None}

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
        'path': last_path or best_path
    }


def _extract_arg(args_str: str, arg: str) -> str:
    """从 args 字符串中提取指定参数的值"""
    import re
    pattern = re.compile(re.escape(arg) + r'\s+(\S+)')
    match = pattern.search(args_str)
    return match.group(1) if match else None


def _get_task_depth(task_id, visited=None):
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
    """打印实时任务进度表

    Args:
        title: 表格标题（首次显示时用）
        last_done: 刚完成的任务ID（显示结果）
        newly_unlocked: 刚解锁的任务ID列表（高亮显示）
    """
    completed = get_completed_tasks()
    hardware_level, _ = get_hardware_level()

    # 收集所有任务状态
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

    def _target_epochs(tid):
        return int(_extract_arg(all_tasks[tid].get('args', ''), '--epochs') or 0)

    # 统计各状态数量（用 is_complete 判断是否真正完成，与 get_completed_tasks() 一致）
    # 注意：is_complete=True 表示训练正常结束（early stop 也算完成），由 save_reason 区分
    # 早停与全跑完的差异
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

    # 打印标题行
    sep = f"{'─' * 100}"
    print(f"\n{sep}")
    if title:
        print(f"  {title}")
    else:
        print(f"  团队训练进度 ({stat_done}/{len(all_tasks)} 任务)")
    print(sep)

    # 打印图例（仅首次显示）
    if title:
        print(f"{COLORS['green']}✓已完成{COLORS['reset']}  "
              f"{COLORS['yellow']}⏸中断{COLORS['reset']}  "
              f"{COLORS['blue']}▶待执行{COLORS['reset']}  "
              f"{COLORS['gray']}🔒等待依赖{COLORS['reset']}")

    # 按深度分层打印
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

            # 状态图标 + 文字
            # 真正完成：与 get_completed_tasks() 一致，只看 is_complete
            # （is_complete=True 表示训练正常结束，early stop 也算完成）
            target_epochs = int(_extract_arg(task.get('args', ''), '--epochs') or 0)
            is_done = info['is_complete']
            is_newly = task_id in (newly_unlocked or [])
            is_last = task_id == last_done

            # 检查点信息：best: N last: Y target: T
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
                # 有检查点但未真正完成：中断（按 last_epoch 判断）
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

            # 高亮刚完成/刚解锁的任务
            marker = ""
            if is_last:
                marker = f" {COLORS['green']}[刚完成]{COLORS['reset']}"
            elif is_newly:
                marker = f" {COLORS['blue']}[刚解锁]{COLORS['reset']}"

            # 计算显示宽度（中文占2格）+ 按显示宽度截断/填充
            def display_len(s):
                import re
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

    # 底部统计
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
    """自动执行所有可执行任务（动态拓扑重排）

    每次任务完成后重新扫描依赖状态，新解锁的任务会立即加入执行队列。
    优先恢复被中断的任务（is_complete=False）。"""
    hardware_level, _ = get_hardware_level()
    total_tasks = len(all_tasks)

    # 动态扫描
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

    # 首次扫描
    first_task, _ = get_next_task()
    if first_task is None:
        completed = get_completed_tasks()
        print(f"\n{COLORS['yellow']}[信息]{COLORS['reset']} 没有可执行的任务 (已完成: {len(completed)}/{total_tasks})")
        return

    mode_str = "强制" if force_warnings else "标准"
    print(f"\n{COLORS['bold']}自动执行模式 ({mode_str}){COLORS['reset']} | "
          f"总任务: {total_tasks} | 每次任务完成后刷新进度\n")

    # 首次打印进度表（带图例）
    _print_progress_table(title=f"团队训练进度 ({total_tasks} 任务)", last_done=None, newly_unlocked=None)

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

        # 任务完成后：刷新进度表
        # 判断哪些任务被解锁了（之前 locked，现在 executable）
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
    print(f"\n{COLORS['bold']}导入检查点{COLORS['reset']}")
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
    print("  PE-MMNet v4 团队协作训练")
    print(f"{'=' * 60}{COLORS['reset']}")

    _, gpu_mem = get_hardware_level()
    print(f"\n当前硬件等级: {COLORS['blue']}{hardware_level}{COLORS['reset']} (显存: {gpu_mem:.1f}GB)")
    print(f"已完成任务: {len(completed)}/{len(all_tasks)}")

    print(f"\n状态标记: {COLORS['green']}[OK] 已完成{COLORS['reset']} | "
          f"{COLORS['blue']}[>] 可执行{COLORS['reset']} | "
          f"{COLORS['yellow']}[!] 硬件警告{COLORS['reset']} | "
          f"{COLORS['gray']}[X] 依赖未完成{COLORS['reset']}")

    print()

    # 按等级分组显示
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
            # 显示等级、时间和显存（如果有）
            level = task.get('level', 'N/A')
            time_str = task.get('time', 'N/A')
            gpu_mem = task.get('gpu_mem', 'N/A')
            print(f"       [L] {level} | [T] {time_str} | [G] {gpu_mem}")

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

    # 命令行参数解析
    parser = argparse.ArgumentParser(description='PE-MMNet v4 团队协作训练系统')
    parser.add_argument('--auto', action='store_true', help='自动执行所有可执行任务')
    parser.add_argument('--force', action='store_true', help='强制执行硬件警告任务（auto模式）')
    parser.add_argument('--import', dest='import_mode', action='store_true', help='导入队友检查点')
    args = parser.parse_args()

    # 加载任务配置
    external_tasks = load_tasks_from_files()
    all_tasks = merge_tasks(external_tasks, DEFAULT_TRAIN_TASKS)

    # 检测硬件
    hardware_level, _ = get_hardware_level()

    # 显示配置来源
    if external_tasks:
        print(f"{COLORS['blue']}[信息]{COLORS['reset']} 已加载 {len(external_tasks)} 个外部任务")
        print(f"{COLORS['blue']}[信息]{COLORS['reset']} 配置来源: {TASKS_DIR}")
    else:
        print(f"{COLORS['yellow']}[信息]{COLORS['reset']} 使用内置默认任务列表")

    # 命令行模式处理
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

        # 数字选择
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
