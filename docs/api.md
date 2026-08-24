# API 参考文档

> PE-MMNet v5-α 公开 API | 版本 5.0.5-alpha
>
> 注：本文件 fork 自 v4.6.11 的同名文档；v5-α 新增 API 见 §0 顶部。
>
> 所有签名通过 `inspect.signature()` 在 commit `<latest>` 验证。

---

## 0. v5-α 新增 API（V5A1-V5A5）

### 0.1 模型 (`models`)

```python
from models import (
    SplineHead,                        # V5A2
    GPModule,                          # V5A2
    PETSNetMultimodalV5,               # V5A4
    SwinYOLOFPNV5,                     # V5A4
    ViTYOLOFPNV5,                      # V5A4
    DETRStyleV5,                       # V5A4
    SwinYOLOFPNWithPatchTSTV5,         # V5A4
    V5_MODEL_REGISTRY,                 # 5 变体注册表
    create_v5_model,                   # 工厂函数
)

# 用法
model = create_v5_model("resnet18", image_channels=3, use_gp=True,
                        max_kpts=16, min_kpts=8)
out = model(x_1d, x_2d)
# out = {"bbox": (B,4), "keypoints": (B,K,2),
#        "validity": (B,K), "K": (B,), "gp_lml": scalar}
```

### 0.2 损失 (`training.ordered_kp_loss`)

```python
from training.ordered_kp_loss import (
    catmull_rom_spline_torch,  # Catmull-Rom 三次样条（纯 PyTorch）
    coverage_loss,             # 覆盖距离 loss（chamfer / hausdorff）
    poisson_prior,             # 泊松先验（段长方差）
    OrderedKeypointLoss,       # 组合 loss
)

# 用法
loss_fn = OrderedKeypointLoss(
    lambda_coverage=1.0, lambda_bbox=1.0,
    lambda_gp=0.1, lambda_poisson=0.05,
    coverage_mode="chamfer", M=200,
)
losses = loss_fn(pred_bbox, pred_kpts, true_bbox, true_crack_pixels,
                 gp_module=model.gp_module)
# losses = {"total", "coverage", "bbox", "gp", "poisson"}
```

### 0.3 数据 (`data.patch_dataset_v5`)

```python
from data.patch_dataset_v5 import PatchDatasetV5, collate_v5

ds = PatchDatasetV5(
    patch_size=256, n_samples=100,
    min_kpts=8, max_kpts=16,
    thermal_profile=None,  # 默认 11.7h 周期
    crack_stress_threshold_MPa=50.0,
)
loader = DataLoader(ds, batch_size=4, collate_fn=collate_v5)
# batch = {x_1d, x_2d, true_bbox, true_keypoints, keypoint_mask,
#         true_crack_pixels, pixel_mask, metadata}
```

### 0.4 训练 (`training.trainer_v5`)

```python
from training.trainer_v5 import (
    TrainerV5, CSVHistory,
    save_checkpoint, load_checkpoint,
    set_seed, get_device,
)

trainer = TrainerV5(
    variant="resnet18", n_samples=100, patch_size=256,
    batch_size=4, epochs=10, lr=1e-4,
    min_kpts=8, max_kpts=16,
    use_gp=True, use_aug=True,
    log_dir="logs/training_history/run_train_v5",
)
final = trainer.fit(verbose=True, resume_from=None)
# final = {"last_train_loss", "last_val_loss", "best_val_loss"}
```

---

## 1. 数据模块 (`data.dataset_multimodal`)

---

## 1. 数据模块 (`data.dataset_multimodal`)

### `create_multibatch_dataloaders`

```python
def create_multibatch_dataloaders(
    data_roots=None,
    batch_size=16,
    seq_len=300,
    image_size=256,
    num_workers=0,
    train_ratio=0.8,
    augment=True,
    predict_offset=0,
    seq_interp_mode='interpolate',
    remove_contours=False,
    disabled_batches=None,
    task='detection',
    triple_channel=False,
    cutmix_prob=0.0,
) -> (train_loader, test_loader)
```

主入口。创建多批次数据加载器。

**Args**:

- `data_roots`: 数据批次列表（None → 调 `get_all_data_batches()`）
- `batch_size`: 批次大小
- `seq_len`: 1D 序列长度
- `image_size`: 2D 图像尺寸（256/384/512/768/1024）
- `num_workers`: DataLoader workers（Windows 兼容 0）
- `train_ratio`: train/test 划分比例
- `augment`: 是否启用数据增强
- `predict_offset`: 时间偏移量（0=当前，1=0.05s 后，2=0.1s 后）
- `seq_interp_mode`: 'interpolate' 或 'pool'
- `remove_contours`: 对参数化扫描4启用等值线去除
- `disabled_batches`: 禁用的批次名列表
- `task`: 'detection' / 'segmentation' / 'multitask'
- `triple_channel`: 三通道时序输入
- `cutmix_prob`: ThermalCutMix 概率

### 其他数据类

| 类/函数 | 签名 | 说明 |
|---------|------|------|
| `get_all_data_batches()` | () | 自动扫描所有数据批次 |
| `SingleBatchDataset(data_root, seq_len=300, image_size=256, augment=True, sample_indices=None, predict_offset=0, seq_interp_mode='interpolate', remove_contours=False, task='detection', triple_channel=False)` | 单批次数据集 |
| `MultiBatchDataset(data_roots, train_ratio=0.8, seq_len=300, image_size=256, augment=True, seed=42)` | 多批次合并数据集 |
| `MultiBatchCollateDataset(data_roots, split='train', train_ratio=0.8, seq_len=300, image_size=256, augment=True, seed=42, predict_offset=0, seq_interp_mode='interpolate', remove_contours=False, disabled_batches=None, task='detection', triple_channel=False)` | 多批次 + 自定义 collate |
| `ImagePreprocessor(image_size=256, crop_ratio=0.7)` | 图像预处理（中心裁剪 + resize） |
| `MaskLabelProcessor(image_size=256, crop_ratio=0.7, binary_threshold=0.1, invert=True)` | 分割掩膜处理 |
| `interpolate_seq(time_points, values, target_len=300)` | 线性插值到目标长度 |
| `process_seq(time_points, values, target_len=300, mode='interpolate')` | 'interpolate' / 'pool' 模式 |
| `create_triple_channel_seq(seq_1d, seq_len)` | 生成 [初始, 当前, 变化率] 三通道 |
| `compute_lw_from_density(density)` | 从密度推 l, w |
| `imread_unicode(filepath)` | 支持中文路径的图像读取 |

---

## 2. 模型模块

### 2.1 基础模型 (`models.pe_tsnet_multimodal`)

```python
class PETSNetMultimodal(nn.Module):
    def __init__(self, seq_len=300, image_channels=2, image_size=256,
                 pretrained_2d=True, dropout=0.2, fusion='cross_attn',
                 backbone_2d='resnet18', backbone_1d='cnn_attn'):
        ...

    def forward(self, seq_1d, img_2d):
        """Returns: (output, global_density) tuple."""
```

### 2.2 YOLO-FPN 变体 (`models.pe_tsnet_yolo`)

```python
class SwinYOLOFPN(nn.Module):
    """Swin-Tiny + YOLO-FPN, 16×16 固定网格."""
    actual_grid_size = 16  # 硬编码
    def forward(self, seq_1d, img_2d):
        if self.training:
            return raw_pred, global_density  # (B, 256, 6)
        else:
            best_pred = raw_pred.argmax(dim=1)  # 取最高 conf 网格
            return self.output_head(self.fusion(best_pred, feat_1d)), global_density

class ViTYOLOFPN(nn.Module):
    """ViT-Small + YOLO-FPN, dynamic grid >=16."""
    actual_grid_size  # 动态计算

class SwinYOLOFPNWithPatchTST(nn.Module):
    """Swin + PatchTST 1D 骨干 + YOLO-FPN."""
    actual_grid_size  # 动态计算
```

### 2.3 DETR 变体 (`models.pe_tsnet_detr`)

```python
class DETRStyle(nn.Module):
    """ResNet-18 + 6+6 Transformer + 100 queries."""
    d_model = 512
    num_queries = 100
    def forward(self, seq_1d, img_2d):
        if self.training:
            return raw_pred, global_density  # (B, 100, 6)
        else:
            best_pred = raw_pred.argmax(dim=1)
            ...
```

### 2.4 骨干与融合

| 类 | 来源 | 作用 |
|----|------|------|
| `ResNet18Backbone2D` | `pe_tsnet_multimodal` | 2D 空间特征 |
| `TemporalFeatureExtractor` | `pe_tsnet_multimodal` | 1D 时序特征（Micro+Macro+SelfAttn） |
| `CrossAttentionFusion` | `pe_tsnet_multimodal` | 双向交叉注意力 |
| `GatedMultimodalFusion` | `pe_tsnet_multimodal` | 门控融合（温度/应力分治） |
| `SEBlock` | `pe_tsnet_multimodal` | SE 通道注意力 |
| `CoordAtt` | `pe_tsnet_multimodal` | 坐标注意力（保留位置信息） |
| `BackboneWithAttention` | `pe_tsnet_multimodal` | 包装器：在 2D 骨干后注入注意力 |
| `PatchTST1D` | `pe_tsnet_patchtst` | 1D PatchTST 骨干（patch_len=10, d_model=64, 2 层 4 头） |
| `MaskDecoder` | `pe_tsnet_multimodal` | 分割解码器 |

### 2.5 消融变体

| 变体 key | 类 | 位置 |
|----------|-----|------|
| `1d_only` | `Model1DOnly` | `run_train.py` |
| `2d_only` | `Model2DOnly` | `run_train.py` |
| `concat` | `ModelConcat` | `run_train.py` |
| `add` | `ModelAdd` | `run_train.py` |
| `cross_attn` | `ModelCrossAttn` | `run_train.py` |

### 2.6 工厂

```python
def create_model(variant_key, config, device) -> nn.Module:
    """根据 variant_key + config 创建模型。"""
```

---

## 3. 损失函数 (`training.mono_loss`)

| 类 | 用途 | 默认参数 |
|----|------|----------|
| `MultimodalCrackLoss` | ResNet18 + 消融 | `lambda_mse_density=1.0, lambda_mono=0.1, lambda_loc=1.0, lambda_conf=1.0` |
| `YOLOLoss` | swin_yolo / vit_yolo / swin_yolo_patchtst | `lambda_box=1.0, lambda_conf=1.0, lambda_mono=0.1`（硬编码） |
| `DETRLoss` | detr | `matcher=HungarianMatcher(), lambda_bbox=1.0, lambda_conf=1.0, lambda_mono=0.1`（硬编码） |
| `SegmentationLoss` | segmentation 任务 | `lambda_dice=1.0, lambda_bce=0.5` |
| `MultimodalSegmentationLoss` | multitask 任务 | `lambda_seg=1.0, lambda_det=0.5` |
| `YOLOTargetAssigner` | YOLO 目标分配 | `grid_size=16, nearby_range=2` |
| `HungarianMatcher` | DETR 最优匹配 | (无参) |
| `MonotonicityLossV3` | 单调性约束（可复用） | (无参) |
| `SequentialMonotonicityLoss` | 序列级单调性 | (无参) |
| `LocalizationLoss` | 定位损失 | `lambda_diou=1.0, lambda_mse=0.5, use_ciou=False` |
| `ConfidenceLoss` | 置信度损失（BCE） | (无参) |
| `DiceLoss` | 分割 Dice | (无参) |
| `DensityConsistencyLoss` | 密度一致性 | `grid_size=16, neighbor_range=1, lambda_consistency=0.5` |
| `CombinedDensityLoss` | MSE + 一致性 | `lambda_mse=1.0, lambda_consistency=0.5` |
| `verify_monotonicity(pred)` | 函数 | 验证物理单调性 |

---

## 4. 数据增强 (`training.augmentation`)

| 类/函数 | 签名 | 说明 |
|---------|------|------|
| `ThermalCutMix(alpha=1.0, prob=0.0)` | 物理安全 CutMix（仅温度通道） |
| `RandomNoise(std=0.01, prob=0.5)` | 随机噪声 |
| `RandomFlip(p=0.5, vertical_prob=0.0)` | 随机翻转 |
| `Compose(transforms)` | 组合多个 transform |

---

## 5. 训练脚本 (`run_train.py`)

### 公开函数

```python
def train_model(
    model,
    train_loader,
    test_loader,
    config,
    device,
    checkpoint_path=None,
    task_id=None,
    start_epoch=0,        # v4.6.10+：续训起始 epoch（0-indexed）
    best_loss=None,        # v4.6.10+：续训时已知的最佳 loss（None → inf）
) -> (model, metrics)
```

主训练函数。`start_epoch` 和 `best_loss` 是 v4.6.10 加的关键续训参数（之前 bug：续训时未透传导致从头开始）。

```python
def evaluate_model(
    model,
    device,
    data_roots=None,
    predict_offset=0,
    seq_len=300,
    seq_interp_mode='interpolate',
    remove_contours=False,
    disabled_batches=None,
    task='detection',
    image_size=256,
    variant_key=None,
    triple_channel=False,
) -> metrics
```

评估模型。

```python
def estimate_training_time(
    model, train_loader, test_loader, criterion, optimizer, device, config, scheduler=None,
) -> dict
```

估算训练时间。

```python
def staged_training(
    variant_key, config, device, data_roots=None, task_id=None,
) -> model
```

分阶段训练（先短序列预训练，再长序列微调）。

```python
def eval_checkpoint(checkpoint_path, device, image_size=None) -> metrics
```

从 checkpoint 加载并评估。

```python
def freeze_model_backbone(
    model, freeze_2d=True, freeze_1d=False, freeze_names=None,
) -> None
```

冻结骨干（v4.6.10 后只有 `freeze_1d` 真正生效，详见 [CODE_BUGS.md B002](v5_宸茬煡闂.md)）。

### 内部工具（用户不应直接调用，但可读源码理解）

```python
class ETAEstimator:
    def __init__(self, total_epochs, alpha=0.3):
        """EMA 平滑 ETA 估算。alpha=0.3 给新数据 30% 权重。"""

def _get_2d_backbone_name(model) -> str | None:
    """从 model 提取 2D 骨干名（'branch_2d' 等）。"""

def _mark_checkpoint_complete(best_path, last_path, task_id=None, terminated_by_early_stop=False):
    """写 is_complete=True + save_reason。v4.6.11 加幂等性 guard。"""

def _build_checkpoint_data(model, config, epoch, best_loss, save_reason) -> dict:
    """构造 checkpoint dict。"""

def _delete_other_epoch_ckpts(ckpt_base: str, kind: str, keep_epoch: int, _known_files: set = None) -> int:
    """按 ckpt_base 删同 base 下其他 epoch 的 best/last。"""

def _check_signature_mismatch(saved_config: dict, current_config: dict) -> list:
    """返回签名不匹配的字段名列表。"""

def auto_select_config(args) -> dict:
    """按空闲显存自动选 image_size / batch_size / fp16。"""

def get_arch_specific_config(backbone_2d, backbone_1d, args_lr=None, args_dropout=None) -> dict:
    """按骨干微调 lr 和 dropout。"""
```

### 工厂与常量

```python
NEW_VARIANTS = frozenset({'swin_yolo', 'vit_yolo', 'detr', 'swin_yolo_patchtst'})

VARIANT_MODELS = {
    '1d_only': Model1DOnly,
    '2d_only': Model2DOnly,
    'concat': ModelConcat,
    'add': ModelAdd,
    'cross_attn': ModelCrossAttn,
    'resnet18': PETSNetMultimodal,
    'swin_yolo': SwinYOLOFPN,
    'vit_yolo': ViTYOLOFPN,
    'detr': DETRStyle,
    'swin_yolo_patchtst': SwinYOLOFPNWithPatchTST,
}

SIGNATURE_KEYS = [
    'variant', 'backbone_2d', 'backbone_1d', 'fusion', 'task',
    'predict_offset', 'triple_channel', 'use_coord_attn',
    'feature_len', 'seq_interp_mode', 'remove_contours', 'staged_train',
]
_DETR_SIGNATURE_KEYS = SIGNATURE_KEYS + ['image_size']  # 13 字段
```

---

## 6. 启动器 (`train_launcher.py` + `launcher.py`)

| 函数 | 位置 | 说明 |
|------|------|------|
| `build_command(args)` | `train_launcher.py` | 构造 shell 命令 |
| `export_team_configs()` | `train_launcher.py` | 导出当前 args 为 JSON 任务配置 |
| `validate_args(args)` | `train_launcher.py` | 校验参数合法性 |
| `main()` | `train_launcher.py` | 交互式菜单入口 |

> ⚠️ `launcher.py`（889 行）和 `train_launcher.py`（286 行）功能重叠；**以 `train_launcher.py` 为准**（[CODE_BUGS.md B011](v5_宸茬煡闂.md)）。

---

## 7. 团队协作 (`team_train.py`)

```python
def load_tasks_from_files() -> dict:
    """从 tasks/*.json 加载所有任务（顶层数组）。
    跳过 examples.json/example.json/template.json。"""

def merge_tasks(external: Dict, defaults: Dict) -> Dict:
    """合并外部 JSON 任务与内置 DEFAULT_TRAIN_TASKS。外部优先。"""

def topological_sort(tasks: Dict, completed: Set[str]) -> list:
    """拓扑排序，优先返回可执行任务。"""

def run_training_task(task_id: str) -> bool:
    """执行单个训练任务（subprocess 调 run_train.py）。"""

def get_completed_tasks() -> set:
    """返回已完成任务的 ID 集合。"""

def get_hardware_level() -> (str, float):
    """返回 ('L1'/'L1+'/'L2'/'L2+'/'L3', total_mem_gb)。"""

def log_task_execution(task_id, status, duration_seconds=None, error=None) -> None:
    """追加 JSON Lines 到 logs/team_training.log。"""

def _resolve_task_id(ckpt: dict, task_meta=None) -> str | None:
    """ckpt → task_id：先按 task_id 严格匹配，失败按 (variant, subdir, predict_offset) 模糊匹配。"""

def _build_task_meta() -> dict:
    """预提取每个任务的 (variant, subdir, predict_offset)，供 fuzzy match 使用。"""

def _get_task_checkpoint_info(task_id: str) -> dict:
    """返回 {epoch, best_epoch, last_epoch, is_complete, path}。"""

def _resolve_python_interpreter() -> (str, str):
    """自动选 conda / 系统 python / py 启动器。返回 (cmd, desc)。"""

def import_checkpoint() -> None:
    """交互式导入队友的 .pt 检查点。"""
```

### 状态机常量

```python
STATUS_DONE = 'done'
STATUS_INTERRUPTED = 'interrupted'
STATUS_EXECUTABLE = 'executable'
STATUS_WARNING = 'warning'  # 硬件警告
STATUS_LOCKED = 'locked'    # 依赖未满足
```

---

## 8. 配置模块 (`utils.config`)

```python
def load_config() -> dict:
    """加载 config.json。缺失字段填默认值。"""

def save_config(config: dict) -> None:
    """保存 config.json。"""

def ensure_config() -> dict:
    """首次运行交互式设置数据路径。"""

def get_data_root() -> str:
    """获取 data_root（带交互式回退）。"""
```

`config.json` schema：

```json
{
  "data_root": "D:\\path\\to\\data",       // 必填
  "output_dir": "./output",                // 默认 "./output"
  "checkpoints_dir": "./checkpoints",      // 默认 "./checkpoints"
  "results_dir": "./benchmark_results"     // 默认 "./benchmark_results"
}
```

---

## 9. 控制台输出 (`utils.console`)

| 函数 | 用途 |
|------|------|
| `print_title(text)` | 标题（粗体+大写） |
| `print_section(text)` | 章节标题 |
| `print_result(label, value)` | 结果行（label: value） |
| `print_results_table(rows, headers)` | 结果表格 |
| `print_info(text)` | 普通信息（蓝色） |
| `print_warning(text)` | 警告（黄色） |
| `print_error(text)` | 错误（红色） |
| `print_success(text)` | 成功（绿色） |
| `print_progress(epoch, total, metrics)` | 训练进度行 |
| `print_metric_row(name, value, delta)` | 指标行（含 delta） |
| `print_divider()` | 分隔线 |
| `print_header(text)` | 顶部标题（带边框） |

`COLORS` 字典：标准 ANSI 颜色名 + bold + reset。

---

## 10. 工具脚本 (`tools/`)

| 脚本 | 用途 |
|------|------|
| `batch_train_gui.py` | 队列训练 + 预览 + R² 排序对比 |
| `cleanup_old_ckpts.py` | 清理中间 epoch 检查点（保留 1 best + 1 last） |
| `fix_save_reason.py` | 修复 `save_reason='early_stop'` 误标 |
| `streamlit_app.py` | Streamlit 可视化（threshold / colormap / pore-crack） |
| `structure_analyzer.py` | 多阈值 + connected components 分离孔隙和裂纹 |
| `test_contour_remover.py` | 等值线去除的 before/after |
| `visualization.py` | `--check-preprocess` / `--show-contour-removal` / `--compare` / `--batch-compare` |
| `verify_checkpoints.py`（项目根） | 检查点健康验证（`--variant` / `--detail`） |

---

## 11. 验证签名

```bash
/c/Users/LZXdywtgtd/.conda/envs/pe_mmnet/python.exe -c "
import inspect, run_train
print(inspect.signature(run_train.train_model))
"
```

输出：

```
(model, train_loader, test_loader, config, device, checkpoint_path=None, task_id=None, start_epoch=0, best_loss=None)
```

确认 `start_epoch` 和 `best_loss` 是 v4.6.10+ 的新增参数。
