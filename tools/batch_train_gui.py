"""
PE-MMNet v4 交互式批量训练与对比工具

功能：
1. 批量输入多条训练命令
2. 执行前预览和调整每条命令
3. 顺序执行，实时显示进度
4. 执行完成后自动评估所有模型
5. 生成对比表格（按R²排序）
6. 单个命令失败不影响其他任务

使用方法：
    python tools/batch_train_gui.py

快捷命令：
    python tools/batch_train_gui.py --quick "resnet18,100,0" "vit_small,100,0"
"""

import os
import sys
import json
import time
import argparse
import subprocess
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Callable
from enum import Enum
from datetime import datetime

# Windows 控制台编码修复（使用安全方式）
if sys.platform == 'win32':
    try:
        import io
        if sys.stdout is not None and hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if sys.stderr is not None and hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# 添加项目路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# 导入项目模块
from utils.config import get_checkpoints_dir, ensure_config
from utils.console import (
    print_title, print_section, print_result, print_results_table,
    print_info, print_warning, print_error, print_success,
    print_progress, print_header, print_divider
)


# =============================================================================
# 常量定义
# =============================================================================

CHECKPOINT_DIR = get_checkpoints_dir()
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 日志目录
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 默认参数
DEFAULTS = {
    'variant': 'resnet18',
    'backbone_2d': 'resnet18',
    'backbone_1d': 'cnn_attn',
    'fusion': 'cross_attn',
    'predict_offset': 0,
    'epochs': 100,
    'task': 'detection',
    'image_size': 512,  # 默认高分辨率
    'batch_size': None,  # 自动根据显存选择
    'fp16': True,  # 默认启用 FP16
    'learning_rate': 3e-4,
    'dropout': 0.2,
}


# =============================================================================
# 枚举和状态
# =============================================================================

class TaskStatus(Enum):
    """任务状态"""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


# =============================================================================
# 命令封装类
# =============================================================================

@dataclass
class TrainingCommand:
    """训练命令封装"""
    id: str
    variant: str = 'resnet18'
    backbone_2d: str = 'resnet18'
    backbone_1d: str = 'cnn_attn'
    fusion: str = 'cross_attn'
    predict_offset: int = 0
    epochs: int = 150
    task: str = 'detection'
    image_size: int = 512  # 图像分辨率
    batch_size: Optional[int] = None  # None 表示自动
    fp16: bool = True  # FP16 混合精度
    lr: float = 3e-4  # 学习率
    dropout: float = 0.2  # Dropout
    seq_interp_mode: str = 'interpolate'  # 序列插值模式
    force_retrain: bool = False
    status: TaskStatus = TaskStatus.PENDING
    metrics: Optional[Dict] = None
    error: Optional[str] = None
    actual_checkpoint_path: Optional[str] = None  # 记录实际检查点路径

    def to_args(self) -> List[str]:
        """转换为命令行参数列表"""
        args = [
            'python', 'run_train.py', '--mode', 'train',
            '--variant', self.variant,
            '--backbone_2d', self.backbone_2d,
            '--backbone_1d', self.backbone_1d,
            '--fusion', self.fusion,
            '--epochs', str(self.epochs),
            '--predict_offset', str(self.predict_offset),
            '--task', self.task,
            '--image_size', str(self.image_size),
            '--lr', str(self.lr),
            '--dropout', str(self.dropout),
        ]
        # batch_size 为 None 时不指定，让 run_train.py 自动选择
        if self.batch_size is not None:
            args.extend(['--batch_size', str(self.batch_size)])
        # 序列插值模式
        if self.seq_interp_mode != 'interpolate':
            args.extend(['--seq_interp_mode', self.seq_interp_mode])
        # FP16 控制
        if not self.fp16:
            args.append('--no_fp16')
        if self.force_retrain:
            args.append('--force_retrain')
        return args

    def to_command_string(self) -> str:
        """转换为完整命令字符串"""
        return ' '.join(self.to_args())

    def get_checkpoint_name(self) -> str:
        """获取检查点文件名（与run_train.py保持一致）"""
        return f"{self.variant}_{self.backbone_2d}_{self.backbone_1d}_{self.fusion}_task{self.task}_offset{self.predict_offset}_best.pt"

    def get_checkpoint_path(self) -> str:
        """获取检查点完整路径"""
        return os.path.join(CHECKPOINT_DIR, self.get_checkpoint_name())

    def get_display_name(self) -> str:
        """获取显示名称（简短）"""
        # 简化的显示名称
        if self.variant == 'resnet18':
            name = f"{self.backbone_2d}_{self.backbone_1d}"
        else:
            name = self.variant
        return f"{name}_offset{self.predict_offset}"

    def get_log_file(self) -> str:
        """获取日志文件路径"""
        log_name = f"{self.get_display_name()}_{int(time.time())}.log"
        return os.path.join(LOG_DIR, log_name)

    @classmethod
    def from_string(cls, cmd_str: str, task_id: str = None) -> 'TrainingCommand':
        """从命令字符串解析"""
        if task_id is None:
            task_id = generate_id()

        # 移除 "python run_train.py" 前缀和注释
        cmd_str = cmd_str.strip()
        cmd_str = cmd_str.split('#')[0]  # 移除注释
        cmd_str = cmd_str.replace('python run_train.py', '').strip()

        # 使用 argparse 解析
        parser = argparse.ArgumentParser(allow_abbrev=False, add_help=False)
        parser.add_argument('--variant', default=DEFAULTS['variant'])
        parser.add_argument('--backbone_2d', default=DEFAULTS['backbone_2d'])
        parser.add_argument('--backbone_1d', default=DEFAULTS['backbone_1d'])
        parser.add_argument('--fusion', default=DEFAULTS['fusion'])
        parser.add_argument('--predict_offset', type=int, default=DEFAULTS['predict_offset'])
        parser.add_argument('--epochs', type=int, default=DEFAULTS['epochs'])
        parser.add_argument('--task', default=DEFAULTS['task'])
        parser.add_argument('--image_size', type=int, default=DEFAULTS.get('image_size', 512))
        parser.add_argument('--batch_size', type=int, default=None)
        parser.add_argument('--fp16', action='store_true', default=True)
        parser.add_argument('--no_fp16', action='store_true')
        parser.add_argument('--lr', type=float, default=None)
        parser.add_argument('--dropout', type=float, default=None)
        parser.add_argument('--seq_interp_mode', default=None)
        parser.add_argument('--force_retrain', action='store_true')

        try:
            args = parser.parse_args(cmd_str.split() if cmd_str else [])
        except SystemExit:
            # 解析失败，使用默认值
            args = None

        if args:
            return cls(
                id=task_id,
                variant=args.variant,
                backbone_2d=args.backbone_2d,
                backbone_1d=args.backbone_1d,
                fusion=args.fusion,
                predict_offset=args.predict_offset,
                epochs=args.epochs,
                task=args.task,
                image_size=args.image_size,
                batch_size=args.batch_size,
                fp16=not args.no_fp16,
                lr=args.lr if args.lr is not None else DEFAULTS['learning_rate'],
                dropout=args.dropout if args.dropout is not None else DEFAULTS['dropout'],
                seq_interp_mode=args.seq_interp_mode if args.seq_interp_mode else 'interpolate',
                force_retrain=args.force_retrain if hasattr(args, 'force_retrain') else False
            )
        else:
            return cls(id=task_id)

    @classmethod
    def from_quick_format(cls, quick_str: str, task_id: str = None) -> 'TrainingCommand':
        """
        从快捷格式解析: 'backbone_2d,epochs,offset,task'
        示例: 'resnet18,100,0,detection'
        """
        if task_id is None:
            task_id = generate_id()

        parts = [p.strip() for p in quick_str.split(',')]

        # 解析快捷格式
        backbone_2d = parts[0] if len(parts) > 0 else DEFAULTS['backbone_2d']
        epochs = int(parts[1]) if len(parts) > 1 else DEFAULTS['epochs']
        predict_offset = int(parts[2]) if len(parts) > 2 else DEFAULTS['predict_offset']
        task = parts[3] if len(parts) > 3 else DEFAULTS['task']

        # 根据 backbone_2d 确定 1D 骨干和融合方式
        if backbone_2d == 'vit_small':
            backbone_1d = 'transformer'
            fusion = 'cross_attn'
        else:
            backbone_1d = 'cnn_attn'
            fusion = 'cross_attn'

        return cls(
            id=task_id,
            variant='resnet18',
            backbone_2d=backbone_2d,
            backbone_1d=backbone_1d,
            fusion=fusion,
            epochs=epochs,
            predict_offset=predict_offset,
            task=task,
            image_size=DEFAULTS['image_size'],
            batch_size=DEFAULTS['batch_size'],
            fp16=DEFAULTS['fp16'],
            lr=DEFAULTS['learning_rate'],
            dropout=DEFAULTS['dropout'],
            seq_interp_mode='interpolate',
        )

    def to_dict(self) -> dict:
        """转换为字典（用于保存）"""
        return {
            'id': self.id,
            'variant': self.variant,
            'backbone_2d': self.backbone_2d,
            'backbone_1d': self.backbone_1d,
            'fusion': self.fusion,
            'predict_offset': self.predict_offset,
            'epochs': self.epochs,
            'task': self.task,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'fp16': self.fp16,
            'lr': self.lr,
            'dropout': self.dropout,
            'seq_interp_mode': self.seq_interp_mode,
            'force_retrain': self.force_retrain,
            'status': self.status.value,
            'error': self.error,
            'actual_checkpoint_path': self.actual_checkpoint_path,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TrainingCommand':
        """从字典恢复"""
        cmd = cls(
            id=data['id'],
            variant=data.get('variant', DEFAULTS['variant']),
            backbone_2d=data.get('backbone_2d', DEFAULTS['backbone_2d']),
            backbone_1d=data.get('backbone_1d', DEFAULTS['backbone_1d']),
            fusion=data.get('fusion', DEFAULTS['fusion']),
            predict_offset=data.get('predict_offset', DEFAULTS['predict_offset']),
            epochs=data.get('epochs', DEFAULTS['epochs']),
            task=data.get('task', DEFAULTS['task']),
            image_size=data.get('image_size', DEFAULTS['image_size']),
            batch_size=data.get('batch_size', DEFAULTS['batch_size']),
            fp16=data.get('fp16', DEFAULTS['fp16']),
            lr=data.get('lr', DEFAULTS['learning_rate']),
            dropout=data.get('dropout', DEFAULTS['dropout']),
            seq_interp_mode=data.get('seq_interp_mode', 'interpolate'),
            force_retrain=data.get('force_retrain', False),
        )
        cmd.status = TaskStatus(data.get('status', 'pending'))
        cmd.error = data.get('error')
        cmd.actual_checkpoint_path = data.get('actual_checkpoint_path')
        return cmd


# =============================================================================
# 工具函数
# =============================================================================

_id_counter = 0

def generate_id() -> str:
    """生成唯一ID（简洁格式：T序号_随机后缀）"""
    global _id_counter
    _id_counter += 1
    import random
    suffix = random.randint(10, 99)
    return f"T{_id_counter:03d}_{suffix}"


def get_status_icon(status: TaskStatus) -> str:
    """获取状态图标"""
    icons = {
        TaskStatus.PENDING: "⏳",
        TaskStatus.RUNNING: "🔄",
        TaskStatus.SUCCESS: "✅",
        TaskStatus.FAILED: "❌",
        TaskStatus.SKIPPED: "⏭️",
    }
    return icons.get(status, "?")


# =============================================================================
# 批量训练控制器
# =============================================================================

class BatchTrainer:
    """批量训练管理器"""

    def __init__(self):
        self.commands: List[TrainingCommand] = []
        self.session_file = os.path.join(LOG_DIR, 'session_commands.json')
        self._load_session()

    def _load_session(self):
        """加载会话"""
        if os.path.exists(self.session_file):
            try:
                with open(self.session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.commands = [TrainingCommand.from_dict(d) for d in data]
            except Exception:
                pass

    def _save_session(self):
        """保存会话"""
        with open(self.session_file, 'w', encoding='utf-8') as f:
            json.dump([cmd.to_dict() for cmd in self.commands], f, ensure_ascii=False, indent=2)

    def add_command(self, cmd: TrainingCommand, allow_duplicate: bool = False) -> tuple:
        """
        添加命令

        Returns:
            tuple: (success: bool, message: str, existing_id: str or None)
        """
        # 检查是否已存在相同的命令
        if not allow_duplicate:
            for existing in self.commands:
                if (existing.backbone_2d == cmd.backbone_2d and
                    existing.backbone_1d == cmd.backbone_1d and
                    existing.fusion == cmd.fusion and
                    existing.predict_offset == cmd.predict_offset and
                    existing.task == cmd.task):
                    return (False, "命令已存在", existing.id)

        self.commands.append(cmd)
        self._save_session()
        return (True, "添加成功", None)

    def find_command(self, backbone_2d: str, backbone_1d: str, fusion: str,
                     predict_offset: int, task: str) -> Optional[TrainingCommand]:
        """查找是否存在相同配置的命令"""
        for cmd in self.commands:
            if (cmd.backbone_2d == backbone_2d and
                cmd.backbone_1d == backbone_1d and
                cmd.fusion == fusion and
                cmd.predict_offset == predict_offset and
                cmd.task == task):
                return cmd
        return None

    def remove_command(self, cmd_id: str) -> bool:
        """删除命令"""
        for i, cmd in enumerate(self.commands):
            if cmd.id == cmd_id:
                self.commands.pop(i)
                self._save_session()
                return True
        return False

    def edit_command(self, cmd_id: str, **kwargs) -> bool:
        """编辑命令"""
        for cmd in self.commands:
            if cmd.id == cmd_id:
                for key, value in kwargs.items():
                    if hasattr(cmd, key):
                        setattr(cmd, key, value)
                self._save_session()
                return True
        return False

    def clear_commands(self) -> None:
        """清空所有命令"""
        self.commands.clear()
        self._save_session()

    def get_command(self, cmd_id: str) -> Optional[TrainingCommand]:
        """获取命令"""
        for cmd in self.commands:
            if cmd.id == cmd_id:
                return cmd
        return None

    def execute_all(self, on_progress: Callable = None) -> Dict:
        """顺序执行所有训练任务"""
        if not self.commands:
            return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0, 'details': []}

        results = {
            'total': len(self.commands),
            'success': 0,
            'failed': 0,
            'skipped': 0,
            'details': []
        }

        for i, cmd in enumerate(self.commands):
            print()
            print_section(f"[{i+1}/{len(self.commands)}] 执行: {cmd.get_display_name()}")

            # 更新状态
            cmd.status = TaskStatus.RUNNING
            self._save_session()

            # 检查是否跳过（已存在检查点且不强制重训练）
            ckpt_path = cmd.get_checkpoint_path()

            if os.path.exists(ckpt_path) and not cmd.force_retrain:
                print_info(f"检查点已存在，跳过训练: {cmd.get_checkpoint_name()}")
                cmd.status = TaskStatus.SKIPPED
                cmd.actual_checkpoint_path = ckpt_path
                results['skipped'] += 1

                # 修复：跳过训练后自动评估，获取指标
                try:
                    metrics = self._evaluate_checkpoint(ckpt_path, image_size=cmd.image_size)
                    cmd.metrics = metrics
                    print_success(f"评估完成! R²={metrics.get('r2', 0):.4f}, RMSE={metrics.get('rmse', 0):.4f}")
                except Exception as e:
                    print_warning(f"评估失败: {e}")
            else:
                try:
                    # 执行训练
                    ckpt_path = self._execute_single(cmd)
                    cmd.actual_checkpoint_path = ckpt_path

                    # 评估模型
                    metrics = self._evaluate_checkpoint(ckpt_path, image_size=cmd.image_size)
                    cmd.metrics = metrics
                    cmd.status = TaskStatus.SUCCESS
                    results['success'] += 1

                    print_success(f"训练完成! R²={metrics.get('r2', 0):.4f}, RMSE={metrics.get('rmse', 0):.4f}")

                except KeyboardInterrupt:
                    print_warning("用户中断执行")
                    cmd.status = TaskStatus.PENDING
                    raise
                except Exception as e:
                    cmd.status = TaskStatus.FAILED
                    cmd.error = str(e)
                    results['failed'] += 1
                    print_error(f"任务失败: {e}")

            self._save_session()

            if on_progress:
                on_progress(i + 1, len(self.commands), cmd)

        return results

    def _execute_single(self, cmd: TrainingCommand) -> str:
        """执行单个训练任务，返回检查点路径"""
        log_file = cmd.get_log_file()
        ckpt_path = cmd.get_checkpoint_path()

        print_info(f"日志文件: {log_file}")
        print_info(f"检查点: {cmd.get_checkpoint_name()}")

        # 构建命令
        args = cmd.to_args()

        # 使用 subprocess 执行，实时显示输出
        # Windows 上需要禁用缓冲
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'

        with open(log_file, 'w', encoding='utf-8', buffering=1) as log:
            log.write(f"命令: {' '.join(args)}\n")
            log.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.write("=" * 60 + "\n\n")
            log.flush()

            process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,  # 无缓冲
                cwd=PROJECT_ROOT,
                env=env
            )

            # 实时读取输出
            epoch_count = 0
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    decoded_line = line.decode('utf-8', errors='replace')
                    log.write(decoded_line)
                    log.flush()
                    sys.stdout.flush()

                    # 检测 epoch 行，实时显示进度
                    if 'Epoch ' in decoded_line and '/' in decoded_line:
                        epoch_count += 1
                        if epoch_count % 5 == 0 or 'Best:' in decoded_line or 'val' in decoded_line.lower():
                            print_info(f"  {decoded_line.strip()[:80]}")
                            sys.stdout.flush()

            process.wait()

            log.write("\n" + "=" * 60 + "\n")
            log.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log.flush()

        if process.returncode != 0:
            raise RuntimeError(f"训练进程返回错误码: {process.returncode}")

        return ckpt_path

    def _evaluate_checkpoint(self, ckpt_path: str, image_size: int = None) -> Dict:
        """评估检查点"""
        print_info("评估模型...")

        # 使用 run_train.py 的评估模式
        args = [
            'python', 'run_train.py',
            '--mode', 'eval',
            '--checkpoint', ckpt_path,
        ]
        # 只有指定了 image_size 才传递（让 run_train.py 自动检测）
        if image_size is not None:
            args.extend(['--image_size', str(image_size)])

        # Windows 上使用 UTF-8 编码读取输出
        import locale
        encoding = 'utf-8' if sys.platform == 'win32' else locale.getpreferredencoding()

        result = subprocess.run(
            args,
            capture_output=True,
            encoding=encoding,
            errors='replace',  # 替换无法解码的字符
            cwd=PROJECT_ROOT
        )

        # 检查子进程返回码
        if result.returncode != 0:
            print_error(f"评估进程失败 (返回码: {result.returncode})")
            print_info(f"stderr: {result.stderr[:500] if result.stderr else 'N/A'}")
            return {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0, 'mIoU': 0.0, 'violation_rate': 0.0, 'dice': 0.0}

        # 解析输出获取指标
        output = result.stdout if result else ""

        # 处理 None 类型
        if output is None:
            output = ""

        # 优先从 JSON 行解析
        import re
        import json

        json_match = re.search(r'__EVAL_JSON__(.+?)__EVAL_JSON__', output)
        if json_match:
            try:
                metrics = json.loads(json_match.group(1))
                print_info(f"评估完成: R²={metrics.get('r2', 0):.4f}, RMSE={metrics.get('rmse', 0):.4f}")
                return metrics
            except json.JSONDecodeError as e:
                print_warning(f"JSON解析失败: {e}，使用正则解析")

        # 回退到正则解析
        ansi_escape = re.compile(r'\x1b\[[0-9;]*m')
        clean_output = ansi_escape.sub('', output)

        metrics = {'r2': 0.0, 'rmse': 0.0, 'mae': 0.0, 'mIoU': 0.0, 'violation_rate': 0.0, 'dice': 0.0}

        # 解析 R2
        r2_match = re.search(r'R2[=:\s]+([0-9.]+)', clean_output, re.IGNORECASE)
        if r2_match:
            metrics['r2'] = float(r2_match.group(1))

        # 解析 RMSE
        rmse_match = re.search(r'RMSE[=:\s]+([0-9.]+)', clean_output, re.IGNORECASE)
        if rmse_match:
            metrics['rmse'] = float(rmse_match.group(1))

        # 解析 MAE
        mae_match = re.search(r'MAE[=:\s]+([0-9.]+)', clean_output, re.IGNORECASE)
        if mae_match:
            metrics['mae'] = float(mae_match.group(1))

        # 解析 mIoU
        miou_match = re.search(r'mIoU[=:\s]+([0-9.]+)', clean_output, re.IGNORECASE)
        if miou_match:
            metrics['mIoU'] = float(miou_match.group(1))

        # 解析违反率
        vio_match = re.search(r'违反率[=:\s]+([0-9.]+)%', clean_output, re.IGNORECASE)
        if vio_match:
            metrics['violation_rate'] = float(vio_match.group(1)) / 100.0

        # 解析 Dice
        dice_match = re.search(r'Dice[=:\s]+([0-9.]+)', clean_output, re.IGNORECASE)
        if dice_match:
            metrics['dice'] = float(dice_match.group(1))

        print_info(f"解析结果: R²={metrics['r2']:.4f}, RMSE={metrics['rmse']:.4f}, mIoU={metrics['mIoU']:.4f}")

        return metrics


# =============================================================================
# 结果比较器
# =============================================================================

class ResultsComparator:
    """结果比较器"""

    def __init__(self, commands: List[TrainingCommand]):
        self.commands = [cmd for cmd in commands if cmd.status == TaskStatus.SUCCESS]
        self.failed_commands = [cmd for cmd in commands if cmd.status == TaskStatus.FAILED]

    def collect_results(self) -> List[Dict]:
        """收集所有成功任务的结果"""
        results = []
        for cmd in self.commands:
            if cmd.actual_checkpoint_path and os.path.exists(cmd.actual_checkpoint_path):
                metrics = self._eval_checkpoint(
                    cmd.actual_checkpoint_path, cmd.task, image_size=cmd.image_size
                )
                results.append({
                    'id': cmd.id,
                    'display_name': cmd.get_display_name(),
                    'checkpoint': cmd.get_checkpoint_name(),
                    'task': cmd.task,
                    'predict_offset': cmd.predict_offset,
                    'metrics': metrics,
                    'config': {
                        'variant': cmd.variant,
                        'backbone_2d': cmd.backbone_2d,
                        'backbone_1d': cmd.backbone_1d,
                        'fusion': cmd.fusion,
                    }
                })
        return results

    def _eval_checkpoint(self, ckpt_path: str, task: str, image_size: int = None) -> Dict:
        """评估单个检查点"""
        # 动态导入
        import torch

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 调用 run_train.py 的评估函数
        from run_train import eval_checkpoint
        return eval_checkpoint(ckpt_path, device, image_size=image_size)

    def generate_table(self, results: List[Dict], sort_by: str = 'r2') -> str:
        """生成对比表格"""
        if not results:
            return "暂无成功的结果"

        # 按指标排序
        results.sort(key=lambda x: x['metrics'].get(sort_by, 0), reverse=True)

        # 生成表格
        lines = []
        lines.append("")
        lines.append("=" * 100)
        lines.append("  模型性能对比表")
        lines.append("=" * 100)
        lines.append("")

        # 表头
        if results and results[0]['task'] == 'segmentation':
            lines.append("|  #  | 模型名称              | 任务        | Dice     |")
            lines.append("|----|----------------------|-------------|-----------|")
        else:
            lines.append("|  #  | 模型名称              | 任务        | R²      | RMSE    | mIoU    | 违反率   |")
            lines.append("|----|----------------------|-------------|----------|---------|---------|----------|")

        # 数据行
        for i, r in enumerate(results, 1):
            m = r['metrics']
            if r['task'] == 'segmentation':
                line = f"| {i:2} | {r['display_name']:20} | {r['task']:11} | {m.get('dice', 0)*100:7.2f}% |"
            else:
                line = f"| {i:2} | {r['display_name']:20} | {r['task']:11} | {m.get('r2', 0):7.4f} | {m.get('rmse', 0):6.4f} | {m.get('mIoU', 0):6.4f} | {m.get('violation_rate', 0)*100:7.1f}% |"
            lines.append(line)

        # 失败任务
        if self.failed_commands:
            lines.append("")
            lines.append("-" * 100)
            lines.append("  失败任务:")
            for cmd in self.failed_commands:
                lines.append(f"  ❌ {cmd.get_display_name()}: {cmd.error}")

        lines.append("")
        lines.append("=" * 100)
        lines.append(f"按 {sort_by} 降序排列")
        lines.append("=" * 100)

        return '\n'.join(lines)

    def print_table(self, results: List[Dict], sort_by: str = 'r2'):
        """打印对比表格"""
        print(self.generate_table(results, sort_by))


# =============================================================================
# 交互式界面
# =============================================================================

class InteractiveCLI:
    """交互式命令行界面"""

    def __init__(self):
        self.trainer = BatchTrainer()
        self.running = True

    def clear_screen(self):
        """清屏"""
        os.system('cls' if os.name == 'nt' else 'clear')

    def show_main_menu(self):
        """显示主菜单"""
        self.clear_screen()
        print_header("PE-MMNet v4 批量训练工具")
        print()
        print(f"  当前任务数: {len(self.trainer.commands)}")
        print()

        # 统计状态
        pending = sum(1 for c in self.trainer.commands if c.status == TaskStatus.PENDING)
        success = sum(1 for c in self.trainer.commands if c.status == TaskStatus.SUCCESS)
        failed = sum(1 for c in self.trainer.commands if c.status == TaskStatus.FAILED)

        if pending + success + failed > 0:
            print(f"  状态: ⏳待执行={pending} ✅成功={success} ❌失败={failed}")
        print()

        print_divider()
        print_section("命令管理")
        print("  [1] 查看命令列表")
        print("  [2] 添加训练命令")
        print("  [3] 编辑命令")
        print("  [4] 删除命令")
        print("  [5] 清空所有命令")
        print("  [6] 批量导入命令")
        print()
        print_section("执行操作")
        print("  [7] ▶ 开始批量训练")
        print("  [8] 评估已有检查点")
        print("  [9] 📊 生成对比表格")
        print("  [a] ⚡ 添加消融实验预设")
        print()
        print("  [0] 退出")
        print()

    def run(self):
        """运行主循环"""
        while self.running:
            self.show_main_menu()
            choice = input("选择操作: ").strip().lower()

            handlers = {
                '1': self._show_commands,
                '2': self._add_command,
                '3': self._edit_command,
                '4': self._delete_command,
                '5': self._clear_commands,
                '6': self._batch_import,
                '7': self._start_training,
                '8': self._evaluate_all,
                '9': self._generate_comparison,
                'a': self._add_ablation_preset,
                '0': self._exit,
            }

            handler = handlers.get(choice)
            if handler:
                try:
                    handler()
                except KeyboardInterrupt:
                    print_warning("\n操作已取消")
                    input("\n按回车继续...")
            else:
                print_warning("无效选择")
                input("\n按回车继续...")

    def _show_commands(self):
        """显示命令列表"""
        self.clear_screen()
        print_section("命令列表")

        if not self.trainer.commands:
            print_info("暂无命令，请先添加训练命令")
        else:
            for i, cmd in enumerate(self.trainer.commands, 1):
                status_icon = get_status_icon(cmd.status)
                metrics_str = ""
                if cmd.metrics:
                    metrics_str = f" R²={cmd.metrics.get('r2', 0):.4f}"
                # 显示完整ID
                print(f"  {i}. {status_icon} [{cmd.id}] {cmd.get_display_name()}")
                print(f"      {cmd.task} | {cmd.epochs} epochs | offset={cmd.predict_offset} | {cmd.image_size}x{cmd.image_size} | batch={cmd.batch_size} | FP16={'Y' if cmd.fp16 else 'N'}{metrics_str}")
                if cmd.error:
                    print(f"      ❌ {cmd.error[:50]}")

        input("\n按回车继续...")

    def _add_command(self):
        """添加训练命令（丰富配置模式）"""
        self.clear_screen()
        print_section("添加训练命令")
        print("(直接回车使用默认值)")
        print()

        # ========== 1. 任务类型 ==========
        print("  【任务类型】")
        print("    [1] detection    - 矩形框检测 (默认)")
        print("    [2] segmentation - 像素级掩膜")
        print("    [3] multitask    - 检测+分割")
        task_choice = input("    选择 [1]: ").strip() or "1"
        task_map = {"1": "detection", "2": "segmentation", "3": "multitask"}
        task = task_map.get(task_choice, "detection")
        print()

        # ========== 2. 模型架构 ==========
        print("  【2D视觉骨干】")
        print("    [1] resnet18  - CNN架构，11M参数，稳定可靠 (默认)")
        print("    [2] vit_small - Transformer，21M参数，表达能力强")
        b2d_choice = input("    选择 [1]: ").strip() or "1"
        backbone_2d = "vit_small" if b2d_choice == "2" else "resnet18"
        print()

        print("  【1D时序骨干】")
        print("    [1] cnn_attn    - CNN+自注意力，当前最佳 (默认)")
        print("    [2] transformer - 纯Transformer，适合长序列")
        print("    [3] dlinear     - 轻量级线性模型")
        b1d_choice = input("    选择 [1]: ").strip() or "1"
        backbone_1d_map = {"1": "cnn_attn", "2": "transformer", "3": "dlinear"}
        backbone_1d = backbone_1d_map.get(b1d_choice, "cnn_attn")
        print()

        print("  【融合方式】")
        print("    [1] cross_attn - 交叉注意力，特征交互最充分 (默认)")
        print("    [2] concat     - 直接拼接，简单有效")
        print("    [3] adaptive   - 自适应加权，可学习最优融合")
        fusion_choice = input("    选择 [1]: ").strip() or "1"
        fusion_map = {"1": "cross_attn", "2": "concat", "3": "adaptive"}
        fusion = fusion_map.get(fusion_choice, "cross_attn")
        print()

        # ========== 3. 训练参数 ==========
        print("  【训练参数】")
        epochs = input("    训练轮数 [150]: ").strip() or "150"
        print()

        # ========== 4. 高级参数 ==========
        print("  【高级参数】(直接回车使用默认值)")
        offset = input("    时间偏移 (0=当前, 1=0.05s后) [0]: ").strip() or "0"
        image_size = input("    图像尺寸 (256/384/512/768/1024) [512]: ").strip() or "512"

        # 学习率
        lr_default = "0.0001" if backbone_2d == "vit_small" or backbone_1d == "transformer" else "0.001"
        lr = input(f"    学习率 (根据架构推荐 {lr_default}): ").strip() or lr_default
        print()

        # Dropout
        dropout_default = "0.2"
        if backbone_2d == "vit_small" and backbone_1d == "transformer":
            dropout_default = "0.3"
        dropout = input(f"    Dropout (推荐 {dropout_default}): ").strip() or dropout_default
        print()

        # 序列参数
        print("  【序列参数】")
        print("    [1] interpolate - 线性插值，保留变化细节 (默认)")
        print("    [2] pool        - 自适应池化，减少高频噪声")
        seq_mode = input("    选择 [1]: ").strip() or "1"
        seq_interp_mode = "pool" if seq_mode == "2" else "interpolate"
        print()

        # 显存优化
        print("  【显存优化】")
        batch_size = input("    批次大小 (留空则自动): ").strip()
        fp16 = input("    FP16混合精度? (Y/n): ").strip().lower()
        print()

        # ========== 6. 其他 ==========
        print("  【其他】")
        force = input("    强制重训练? (y/N): ").strip().lower() == 'y'

        cmd = TrainingCommand(
            id=generate_id(),
            task=task,
            backbone_2d=backbone_2d,
            backbone_1d=backbone_1d,
            fusion=fusion,
            epochs=int(epochs),
            predict_offset=int(offset),
            image_size=int(image_size),
            batch_size=int(batch_size) if batch_size else None,
            fp16=(fp16 != 'n'),
            lr=float(lr),
            dropout=float(dropout),
            seq_interp_mode=seq_interp_mode,
            force_retrain=force
        )

        print()
        print_divider()
        print_info("生成的命令:")
        print(f"  python run_train.py {cmd.to_command_string().replace('python run_train.py ', '')}")
        print()
        print(f"  检查点: {cmd.get_checkpoint_name()}")
        print_divider()
        print()

        confirm = input("确认添加? [Y/n]: ").strip().lower()
        if confirm != 'n':
            success, msg, existing_id = self.trainer.add_command(cmd)
            if success:
                print_success("命令已添加!")
            else:
                print_warning(f"{msg} (ID: {existing_id})")
                overwrite = input("是否替换原有命令? (y/N): ").strip().lower()
                if overwrite == 'y':
                    self.trainer.remove_command(existing_id)
                    self.trainer.add_command(cmd, allow_duplicate=True)
                    print_success("命令已替换!")
                else:
                    print_info("已取消")
        else:
            print_info("已取消")

        input("\n按回车继续...")

    def _edit_command(self):
        """编辑命令"""
        self.clear_screen()
        print_section("编辑命令")
        self._show_commands()

        if not self.trainer.commands:
            input("\n按回车继续...")
            return

        cmd_id = input("\n输入要编辑的命令ID (如 T001_51): ").strip()
        cmd = None
        for c in self.trainer.commands:
            if c.id.startswith(cmd_id):
                cmd = c
                break

        if not cmd:
            print_error("未找到命令")
            input("\n按回车继续...")
            return

        print_section(f"编辑: {cmd.get_display_name()}")
        print("  (直接回车保持不变)")
        print()

        epochs = input(f"  训练轮数 [{cmd.epochs}]: ").strip()
        offset = input(f"  时间偏移 [{cmd.predict_offset}]: ").strip()
        image_size = input(f"  图像尺寸 [{cmd.image_size}]: ").strip()
        batch_size = input(f"  批次大小 [auto/{cmd.batch_size}]: ").strip()
        fp16 = input(f"  FP16启用? (Y/n): ").strip().lower()
        force = input(f"  强制重训练? (y/N): ").strip().lower()

        updates = {}
        if epochs:
            updates['epochs'] = int(epochs)
        if offset:
            updates['predict_offset'] = int(offset)
        if image_size:
            updates['image_size'] = int(image_size)
        if batch_size:
            updates['batch_size'] = int(batch_size) if batch_size.lower() != 'auto' else None
        if fp16 == 'n':
            updates['fp16'] = False
        elif fp16 == 'y':
            updates['fp16'] = True
        if force == 'y':
            updates['force_retrain'] = True

        if updates:
            self.trainer.edit_command(cmd.id, **updates)
            print_success("命令已更新!")
        else:
            print_info("未做任何更改")

        input("\n按回车继续...")

    def _delete_command(self):
        """删除命令"""
        self.clear_screen()
        print_section("删除命令")
        self._show_commands()

        if not self.trainer.commands:
            input("\n按回车继续...")
            return

        cmd_id = input("\n输入要删除的命令ID (如 T001_51): ").strip()

        if self.trainer.remove_command(cmd_id):
            print_success("命令已删除!")
        else:
            print_error("未找到命令")

        input("\n按回车继续...")

    def _clear_commands(self):
        """清空所有命令"""
        if not self.trainer.commands:
            print_info("命令列表已是空的")
            input("\n按回车继续...")
            return

        confirm = input("确认清空所有命令? (y/N): ").strip().lower()
        if confirm == 'y':
            self.trainer.clear_commands()
            print_success("已清空所有命令")

        input("\n按回车继续...")

    def _batch_import(self):
        """批量导入命令"""
        self.clear_screen()
        print_section("批量导入命令")
        print("  支持格式:")
        print("    1. 快捷格式: resnet18,100,0,detection")
        print("       格式: backbone_2d,epochs,offset,task")
        print()
        print("  输入命令，每行一条，输入空行结束:")
        print("  (以 # 或 // 开头的行视为注释，会自动忽略)")
        print()
        print("  示例输入:")
        print("    resnet18,100,0,detection")
        print("    vit_small,100,0,detection")
        print("    resnet18,100,1,detection")
        print("    vit_small,100,1,detection")
        print()
        print("-" * 50)
        print()

        lines = []
        while True:
            try:
                line = input()
                if not line.strip():
                    break
                lines.append(line)
            except EOFError:
                break

        if not lines:
            print_info("未输入任何命令")
            input("\n按回车继续...")
            return

        added = 0
        skipped = 0
        duplicates = 0
        for line in lines:
            line = line.strip()
            # 跳过空行和注释
            if not line or line.startswith('#') or line.startswith('//'):
                continue
            # 跳过已经是完整命令行的（用户可能粘贴错了）
            if line.startswith('python '):
                skipped += 1
                continue

            try:
                cmd = TrainingCommand.from_quick_format(line)
                success, msg, existing_id = self.trainer.add_command(cmd)
                if success:
                    added += 1
                else:
                    duplicates += 1
            except Exception as e:
                print_warning(f"解析失败: {line[:50]}")
                skipped += 1

        if added > 0:
            print_success(f"已添加 {added} 条命令")
        if duplicates > 0:
            print_warning(f"跳过 {duplicates} 条（已存在）")
        if skipped > 0:
            print_warning(f"跳过 {skipped} 条（格式错误或示例命令）")

        input("\n按回车继续...")

    def _add_ablation_preset(self):
        """添加消融实验预设"""
        self.clear_screen()
        print_section("添加消融实验预设")
        print()
        print("  预设方案:")
        print("    [1] CNN vs Transformer 对比 (6条)")
        print("        - resnet18_cnn_attn_offset0")
        print("        - vit_small_transformer_offset0")
        print("        - resnet18_cnn_attn_offset1")
        print("        - vit_small_transformer_offset1")
        print("        - resnet18_cnn_attn_offset2")
        print("        - vit_small_transformer_offset2")
        print()
        print("    [2] 简化对比 (2条)")
        print("        - resnet18_cnn_attn_offset0")
        print("        - vit_small_transformer_offset0")
        print()
        print("    [3] 自定义")
        print()

        choice = input("选择预设 [1]: ").strip() or "1"

        presets = {
            "1": [
                ("resnet18", "cnn_attn", "cross_attn", 0),
                ("vit_small", "transformer", "cross_attn", 0),
                ("resnet18", "cnn_attn", "cross_attn", 1),
                ("vit_small", "transformer", "cross_attn", 1),
                ("resnet18", "cnn_attn", "cross_attn", 2),
                ("vit_small", "transformer", "cross_attn", 2),
            ],
            "2": [
                ("resnet18", "cnn_attn", "cross_attn", 0),
                ("vit_small", "transformer", "cross_attn", 0),
            ],
        }

        if choice in presets:
            added_count = 0
            skip_count = 0
            for b2d, b1d, fusion, offset in presets[choice]:
                cmd = TrainingCommand(
                    id=generate_id(),
                    variant='resnet18',
                    backbone_2d=b2d,
                    backbone_1d=b1d,
                    fusion=fusion,
                    predict_offset=offset,
                    epochs=150,
                    task='detection'
                )
                success, msg, existing_id = self.trainer.add_command(cmd)
                if success:
                    added_count += 1
                else:
                    skip_count += 1

            if added_count > 0:
                print_success(f"已添加 {added_count} 条命令")
            if skip_count > 0:
                print_warning(f"跳过 {skip_count} 条（已存在）")

        input("\n按回车继续...")

    def _start_training(self):
        """开始批量训练"""
        if not self.trainer.commands:
            print_error("请先添加训练命令")
            input("\n按回车继续...")
            return

        self.clear_screen()
        print_section("开始批量训练")
        print()

        # 统计
        pending = [c for c in self.trainer.commands if c.status == TaskStatus.PENDING]
        print_info(f"待执行: {len(pending)}/{len(self.trainer.commands)}")

        if not pending:
            print_warning("所有任务已完成或失败")
            input("\n按回车继续...")
            return

        confirm = input("确认开始执行? [Y/n]: ").strip().lower()
        if confirm == 'n':
            print_info("已取消")
            input("\n按回车继续...")
            return

        try:
            results = self.trainer.execute_all()
            print()
            print_section("批量训练完成")
            print_info(f"总计: {results['total']} | 成功: {results['success']} | 失败: {results['failed']} | 跳过: {results['skipped']}")

            if results['failed'] > 0:
                print_warning("失败任务:")
                for cmd in self.trainer.commands:
                    if cmd.status == TaskStatus.FAILED:
                        print(f"  - {cmd.get_display_name()}: {cmd.error}")

        except KeyboardInterrupt:
            print_warning("\n执行被中断")

        input("\n按回车继续...")

    def _evaluate_all(self):
        """评估所有检查点"""
        self.clear_screen()
        print_section("评估已有检查点")

        # 收集需要评估的命令
        to_eval = [c for c in self.trainer.commands if c.status in [TaskStatus.SUCCESS, TaskStatus.SKIPPED]]

        if not to_eval:
            print_info("没有需要评估的检查点")
            input("\n按回车继续...")
            return

        print_info(f"将评估 {len(to_eval)} 个检查点...")

        for cmd in to_eval:
            if cmd.actual_checkpoint_path and os.path.exists(cmd.actual_checkpoint_path):
                try:
                    print_info(f"评估: {cmd.get_display_name()}")
                    metrics = self.trainer._evaluate_checkpoint(cmd.actual_checkpoint_path, image_size=cmd.image_size)
                    cmd.metrics = metrics
                    print_success(f"  R²={metrics.get('r2', 0):.4f}")
                except Exception as e:
                    print_error(f"  评估失败: {e}")

        self.trainer._save_session()
        print_success("评估完成!")

        input("\n按回车继续...")

    def _generate_comparison(self):
        """生成对比表格"""
        self.clear_screen()

        # 评估未评估的检查点
        to_eval = [c for c in self.trainer.commands if c.status == TaskStatus.SUCCESS and not c.metrics]
        if to_eval:
            print_section("评估检查点...")
            for cmd in to_eval:
                if cmd.actual_checkpoint_path and os.path.exists(cmd.actual_checkpoint_path):
                    try:
                        metrics = self.trainer._evaluate_checkpoint(cmd.actual_checkpoint_path, image_size=cmd.image_size)
                        cmd.metrics = metrics
                    except:
                        pass
            self.trainer._save_session()

        # 生成对比表格
        comparator = ResultsComparator(self.trainer.commands)
        results = comparator.collect_results()

        if results:
            print_section("模型性能对比表 (按 R² 降序)")
            comparator.print_table(results, sort_by='r2')
        else:
            print_info("暂无成功的结果可比较")

            # 显示失败任务
            failed = [c for c in self.trainer.commands if c.status == TaskStatus.FAILED]
            if failed:
                print()
                print_warning("失败任务:")
                for cmd in failed:
                    print(f"  ❌ {cmd.get_display_name()}: {cmd.error}")

        input("\n按回车继续...")

    def _exit(self):
        """退出"""
        self.running = False
        print_info("再见!")


# =============================================================================
# 主函数
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description='PE-MMNet v4 批量训练工具')
    parser.add_argument('--quick', nargs='+', help='快捷格式: "variant,epochs,offset" ...')
    args = parser.parse_args()

    # 确保配置有效
    ensure_config(interactive=False)

    cli = InteractiveCLI()

    # 处理快捷命令
    if args.quick:
        for quick_str in args.quick:
            cmd = TrainingCommand.from_quick_format(quick_str)
            cli.trainer.add_command(cmd)
        print_success(f"已添加 {len(args.quick)} 条命令")

        # 直接开始训练（跳过交互确认）
        try:
            results = cli.trainer.execute_all()
            print()
            print_section("批量训练完成")
            print_info(f"总计: {results['total']} | 成功: {results['success']} | 失败: {results['failed']} | 跳过: {results['skipped']}")
        except Exception as e:
            print_error(f"执行失败: {e}")
    else:
        # 启动交互式界面
        cli.run()


if __name__ == "__main__":
    main()
