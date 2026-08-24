# API 参考文档

> PE-MMNet v4 公开接口

---

## 1. 数据模块 (data.dataset_multimodal)

### create_multibatch_dataloaders

创建多批次数据加载器。

```python
from data.dataset_multimodal import create_multibatch_dataloaders

train_loader, test_loader = create_multibatch_dataloaders(
    data_roots=None,              # list[str]: 数据目录（None自动扫描config.json）
    batch_size=16,                # int: 批次大小
    image_size=256,               # int: 图像尺寸（默认256）
    seq_len=300,                  # int: 序列长度
    num_workers=0,                # int: 数据加载线程数
    train_ratio=0.8,              # float: 训练集比例
    augment=True,                 # bool: 数据增强
    predict_offset=0,             # int: 时间偏移（预测未来时刻）
    seq_interp_mode='interpolate',# str: 序列插值模式
    remove_contours=False,        # bool: 是否去除轮廓干扰
    disabled_batches=None,        # list[str]: 禁用批次名列表
    task='detection',             # str: 'detection' | 'segmentation' | 'multitask'
    triple_channel=False,         # bool: 三通道时序输入（初始+当前+变化率）
    cutmix_prob=0.0               # float: ThermalCutMix启用概率（默认关闭）
)
```

**返回**：
- `train_loader`: 训练数据加载器
- `test_loader`: 测试数据加载器

### create_triple_channel_seq

生成三通道时序数据。

```python
from data.dataset_multimodal import create_triple_channel_seq

result = create_triple_channel_seq(seq_1d, seq_len=300)
# result: (3, seq_len) - [初始温度, 当前温度, 温度变化率]
```

---

## 2. 模型模块

### 2.1 基础模型 (models.pe_tsnet_multimodal)

```python
from models.pe_tsnet_multimodal import PETSNetMultimodal

model = PETSNetMultimodal(
    seq_len=300,
    image_channels=2,
    image_size=256,
    pretrained_2d=True,
    dropout=0.2,
    task='detection'
)
```

**输出**：所有模式下均返回单一 `(B, 6)` 张量（检测任务）。

### 2.2 YOLO-FPN 变体 (models.pe_tsnet_yolo)

```python
from models.pe_tsnet_yolo import SwinYOLOFPN, ViTYOLOFPN, SwinYOLOFPNWithPatchTST

# Swin-YOLO-FPN
model = SwinYOLOFPN(
    seq_len=300,
    image_channels=2,
    image_size=256,
    pretrained_2d=True,
    dropout=0.2
)

# ViT-YOLO-FPN（grid_size 动态计算，至少 16）
model = ViTYOLOFPN(
    seq_len=300,
    image_channels=2,
    image_size=256,
    pretrained_2d=True,
    dropout=0.2
)

# Swin-YOLO-FPN with PatchTST（1D骨干使用PatchTST替代TemporalFeatureExtractor）
model = SwinYOLOFPNWithPatchTST(
    seq_len=300,
    image_channels=2,
    image_size=256,
    pretrained_2d=True,
    dropout=0.2
)
```

**输出**：
- 训练模式：`(B, G², 6)`, `(B, 1)` （Swin/PatchTST 动态G²，ViT固定256网格）
- 推理模式：`(B, 6)`, `(B, 1)`

### 2.3 DETR 风格变体 (models.pe_tsnet_detr)

```python
from models.pe_tsnet_detr import DETRStyle

model = DETRStyle(
    seq_len=300,
    image_channels=2,
    image_size=256,
    pretrained_2d=True,
    dropout=0.2,
    d_model=512,
    num_queries=100,
    encoder_layers=6,
    decoder_layers=6
)
```

**输出**：
- 训练模式：`(B, 100, 6)`, `(B, 1)`
- 推理模式：`(B, 6)`, `(B, 1)` （经 Hungarian匹配→query_proj→fusion→output_head）

### 2.4 骨干网络 (models.pe_tsnet_multimodal)

```python
from models.pe_tsnet_multimodal import (
    ResNet18Backbone2D,
    TemporalFeatureExtractor
)

# 2D 骨干
backbone_2d = ResNet18Backbone2D(
    in_channels=2,
    pretrained=True
)
backbone_2d.set_spatial_output(True)  # 启用空间特征输出（DETR必须）

# 1D 特征提取器
backbone_1d = TemporalFeatureExtractor(
    input_dim=1,
    hidden_dim=32,
    num_heads=4,
    dropout=0.2
)
```

### 2.5 融合模块 (models.pe_tsnet_multimodal)

```python
from models.pe_tsnet_multimodal import CrossAttentionFusion, GatedMultimodalFusion

# 交叉注意力融合
# dim_2d 取决于使用场景：
#   - resnet18 变体：dim_2d=512
#   - YOLO 变体（推理分支）：dim_2d=6
#   - DETR 变体（推理分支）：dim_2d=128（经 query_proj 投影后）
fusion = CrossAttentionFusion(
    dim_2d=6,
    dim_1d=64,
    num_heads=4
)

out = fusion(query_feat, feat_1d)
# query_feat: (B, dim_2d)
# feat_1d: (B, dim_1d)
# out: (B, dim_2d + dim_1d)
```

### 2.6 注意力模块 (models.pe_tsnet_multimodal)

```python
from models.pe_tsnet_multimodal import SEBlock, CoordAtt, BackboneWithAttention

# SE注意力
se = SEBlock(channels=512, reduction=16)
out = se(x)  # (B, C, H, W)

# 坐标注意力
coord = CoordAtt(inp=512, oup=512, reduction=32)
out = coord(x)

# 骨干包装器
model = BackboneWithAttention(
    backbone=resnet18,
    attention_type='coord',
    reduction=32
)
# 支持多尺度骨干（返回列表时自动取最后一层）
```

### 2.7 PatchTST 时序骨干 (models.pe_tsnet_patchtst)

```python
from models.pe_tsnet_patchtst import PatchTST1D, PatchTSTWithRate

# 标准 PatchTST
model = PatchTST1D(
    seq_len=300,
    patch_size=10,    # seq_len/patch_size=30 patches
    d_model=64,
    nhead=4,
    num_layers=2,
    output_dim=64
)

# 增强版（处理三通道：初始温度+变化率）
model = PatchTSTWithRate(
    seq_len=300,
    d_model=64,
    output_dim=64
)
```

> PatchTST1D 输出维度与 TemporalFeatureExtractor 完全一致：(B, 64)，可无缝替换。

### 2.8 掩膜解码器 (models.pe_tsnet_multimodal)

```python
from models.pe_tsnet_multimodal import MaskDecoder

decoder = MaskDecoder(
    in_channels=576,   # 融合特征维度
    hidden_dim=128,
    target_size=256
)
mask = decoder(fused_feat)
# fused_feat: (B, 576) 融合特征
# mask: (B, 1, 256, 256) 二值掩膜
```

---

## 3. 损失函数 (training.mono_loss)

### MultimodalCrackLoss

检测任务损失函数（用于 resnet18）。

```python
from training.mono_loss import MultimodalCrackLoss

criterion = MultimodalCrackLoss(
    lambda_mse_density=1.0,
    lambda_mono=0.1,
    lambda_loc=1.0,
    lambda_conf=1.0,
    use_ciou=False     # bool: 是否使用CIoU替代DIoU（默认False）
)
```

### YOLOLoss

YOLO检测损失（用于 swin_yolo, vit_yolo, swin_yolo_patchtst）。

```python
from training.mono_loss import YOLOLoss, YOLOTargetAssigner

assigner = YOLOTargetAssigner(grid_size=16, nearby_range=2)
criterion = YOLOLoss(
    lambda_box=1.0,
    lambda_conf=1.0,
    lambda_mono=0.1
)
```

### DETRLoss

DETR检测损失（用于 detr）。包含 Hungarian Loss + 可选 Supervise Loss（0.5×权重）。

```python
from training.mono_loss import DETRLoss, HungarianMatcher

matcher = HungarianMatcher()
criterion = DETRLoss(
    matcher=matcher,       # HungarianMatcher 实例（必传）
    lambda_bbox=1.0,
    lambda_conf=1.0,
    lambda_mono=0.1
)
```

### SegmentationLoss

分割任务损失函数。

```python
from training.mono_loss import SegmentationLoss

criterion = SegmentationLoss(
    lambda_dice=1.0,
    lambda_bce=1.0
)
```

### YOLOTargetAssigner

YOLO目标分配器（将标签分配到网格）。

```python
from training.mono_loss import YOLOTargetAssigner

assigner = YOLOTargetAssigner(grid_size=16, nearby_range=2)
assigned_target, pos_mask = assigner(labels)
```

### HungarianMatcher

DETR 匈牙利匹配器。

```python
from training.mono_loss import HungarianMatcher

matcher = HungarianMatcher()
indices = matcher(pred, {'labels': labels})
```

### MultimodalSegmentationLoss

检测+分割联合损失（multitask模式）。

```python
from training.mono_loss import MultimodalSegmentationLoss

criterion = MultimodalSegmentationLoss(
    lambda_seg=1.0,
    lambda_det=0.5
)
```

---

## 4. 数据增强 (training.augmentation)

### ThermalCutMix

物理安全版CutMix增强（仅混合温度通道）。

```python
from training.augmentation import ThermalCutMix, RandomNoise, RandomFlip, CompositeTransform

cutmix = ThermalCutMix(alpha=1.0, prob=0.0)  # 默认关闭（prob=0.0）
img_aug, labels_aug = cutmix(img1, img2, labels1, labels2)

# img: (2, H, W) - 通道0=温度，通道1=应力
# 仅混合温度通道，应力通道保持不变
```

### RandomNoise

随机噪声增强。

```python
noise = RandomNoise(noise_level=0.01, prob=0.5)
img_aug = noise(img)
# noise_level: 噪声水平（相对于[0,1]范围的百分比）
```

### RandomFlip

随机翻转增强。

```python
flip = RandomFlip(h_prob=0.5, v_prob=0.0)
img_aug = flip(img)
# h_prob: 水平翻转概率
# v_prob: 垂直翻转概率
```

### CompositeTransform

组合多个增强变换。

```python
transform = CompositeTransform([
    ThermalCutMix(alpha=1.0, prob=0.2),
    RandomNoise(noise_level=0.01, prob=0.3),
    RandomFlip(h_prob=0.5),
])
img_aug, labels_aug = transform(img, labels)
```

---

## 5. 训练脚本 (run_train.py)

### train_model

训练模型。

```python
from run_train import train_model

model, metrics = train_model(
    model,
    train_loader,
    test_loader,
    config,
    device,
    checkpoint_path=None,  # 可选：检查点保存路径
    task_id=None          # 可选：任务ID（团队协作）
)

# 返回:
# - model: 训练后的模型
# - metrics: 最终评估指标 dict
```

### evaluate_model

评估模型。

```python
from run_train import evaluate_model

metrics = evaluate_model(
    model,
    device,
    data_roots=None,               # 可选：数据目录列表
    predict_offset=0,              # 时间偏移
    seq_len=300,                   # 序列长度
    seq_interp_mode='interpolate', # 插值模式
    remove_contours=False,         # 去除轮廓
    disabled_batches=None,         # 禁用批次
    task='detection',              # 任务类型
    image_size=256,                # 图像尺寸
    variant_key=None,              # 模型变体标识
    triple_channel=False           # 三通道输入
)

# 返回:
# - metrics: dict，包含 r2, rmse, mae, 违反率, mIoU 等
```

### estimate_training_time

动态训练时间估算（训练前测量2个epoch）。

```python
from run_train import estimate_training_time

avg_time, total_minutes = estimate_training_time(
    model, train_loader, test_loader,
    criterion, optimizer, device, config,
    scheduler=None   # 可选：学习率调度器（用于估算后恢复）
)

# 返回:
# - avg_time: float, 平均每epoch秒数
# - total_minutes: float, 预估总时间（分钟）
# 自动保存/恢复模型状态，不影响训练
```

### staged_training

分阶段训练（先短序列预训练，再长序列微调）。

```python
from run_train import staged_training

staged_training(
    variant_key='resnet18',
    config={},
    device='cuda',
    data_roots=None,
    task_id=None
)
```

### eval_checkpoint

评估检查点文件。

```python
from run_train import eval_checkpoint

metrics = eval_checkpoint(
    checkpoint_path='checkpoints/xxx_best.pt',
    device='cuda',
    image_size=None   # 可选：指定图像尺寸
)
```

### freeze_model_backbone

冻结骨干网络。

```python
from run_train import freeze_model_backbone

model = freeze_model_backbone(
    model,
    freeze_2d=True,    # 冻结2D骨干
    freeze_1d=False,   # 保持1D骨干可训练
    freeze_names=None  # 可选：额外要冻结的参数名列表
)
```

### NEW_VARIANTS

模块级常量，标识所有新变体（返回 frozenset）。

```python
from run_train import NEW_VARIANTS
# frozenset({'swin_yolo', 'vit_yolo', 'detr', 'swin_yolo_patchtst'})
is_new = 'swin_yolo' in NEW_VARIANTS
```

### _get_2d_backbone_name

获取模型中2D骨干网络的属性名（兼容多种命名）。

```python
from run_train import _get_2d_backbone_name

backbone_name = _get_2d_backbone_name(model)
# 返回 'branch_2d'（resnet18）或 'backbone_2d'（YOLO/DETR）
```

### ETAEstimator

指数移动平均 ETA 估算器。

```python
from run_train import ETAEstimator

# 初始化
eta = ETAEstimator(total_epochs=150, alpha=0.3)

# 每轮更新
eta.update(epoch_time)

# 获取 ETA 信息
info = eta.get_eta(current_epoch)
# 返回:
# {
#     'ema': 1.3,              # 指数移动平均
#     'eta_seconds': 175,      # 剩余秒数
#     'eta_str': '2m55s',     # 格式化字符串
#     'finish_time': '14:32', # 预计完成时间
#     'confidence': '±0.2s',  # 置信区间
#     'remaining_epochs': 135 # 剩余轮次
# }
```

---

## 6. 启动器 (launcher.py)

### build_command

构建训练命令。

```python
from launcher import build_command

cmd = build_command([
    '--variant', 'resnet18',
    '--epochs', '100',
    '--lr', '1e-4'
])
# 返回: 'py "run_train.py" --variant resnet18 --epochs 100 --lr 1e-4'
```

### export_team_configs

导出团队配置文件。

```python
from launcher import export_team_configs

export_team_configs(filepath='team_configs.txt')  # 默认值
```

### validate_args

验证命令行参数。

```python
from launcher import validate_args

valid, errors = validate_args(['--variant', 'resnet18', '--epochs', '100'])
# valid: bool, errors: list[str]
```

---

## 7. 团队协作 (team_train.py)

### log_task_execution

记录任务执行日志。

```python
from team_train import log_task_execution

log_task_execution(
    task_id='BASELINE_1',
    status='completed',           # 'started' | 'completed' | 'failed' | 'skipped'
    duration_seconds=43200,       # 可选：执行时长
    error=None                   # 可选：错误信息
)
# 写入 logs/team_training.log
```

### parse_task_id_from_filename

从文件名解析任务ID。

```python
from team_train import parse_task_id_from_filename

all_tasks = {'BASELINE_1': {}, 'OPT_GATED': {}}
task_id = parse_task_id_from_filename('checkpoint_BASELINE_1_best.pt', all_tasks)
# 返回: 'BASELINE_1' 或 None
```

### get_hardware_level

获取当前硬件等级。

```python
from team_train import get_hardware_level

level, gpu_mem = get_hardware_level()
# level: 'L1' | 'L1+' | 'L2' | 'L2+' | 'L3'
# gpu_mem: float, 显存大小(GB)
```

### load_tasks_from_files

从 tasks/ 目录加载 JSON 配置。

```python
from team_train import load_tasks_from_files

tasks = load_tasks_from_files()
# 返回: dict {task_id: task_config}
```

### topological_sort

按依赖拓扑排序。

```python
from team_train import topological_sort

tasks = {'T1': {'deps': []}, 'T2': {'deps': ['T1']}}
completed = set()
sorted_tasks = topological_sort(tasks, completed)
# 返回: ['T1', 'T2']
```

---

## 8. 配置模块 (utils.config)

```python
from utils.config import (
    load_config,
    save_config,
    get_data_root,
    get_checkpoints_dir,
    get_results_dir,
    get_data_batches,
    ensure_config
)

# 加载配置
config = load_config()

# 确保配置存在（不存在则创建）
ensure_config(interactive=True)

# 获取数据批次列表
batches = get_data_batches()
```

---

## 9. 控制台输出 (utils.console)

| 函数 | 颜色 | 用途 |
|------|------|------|
| `print_title(text, width)` | 白色加粗+上下分隔线 | 大段章节标题 |
| `print_section(text, width)` | 白色加粗无分隔线 | 子标题 |
| `print_result(key, value, fmt, width, unit)` | 青色高亮值 | 关键指标、评估结果 |
| `print_results_table(metrics, width)` | 青色高亮值 | 指标表格 |
| `print_info(text, indent)` | 灰色 | 普通日志 |
| `print_warning(text)` | 黄色 | 警告提示 |
| `print_error(text)` | 红色 | 错误信息 |
| `print_success(text)` | 绿色 | 成功提示 |
| `print_progress(current, total, text)` | 灰色 | 进度信息 `[4/5] 开始训练` |
| `print_metric_row(key, value, fmt, width)` | 灰色 | 表格指标行 |
| `print_divider(char, width, color)` | 可选颜色 | 自定义分隔线 |
| `print_header(text, width)` | 青色+上下 === 分隔线 | 主标题 |

```python
from utils.console import (
    print_title, print_section, print_result, print_results_table,
    print_info, print_warning, print_error, print_success,
    print_progress, print_metric_row, print_divider, print_header
)

print_title("开始训练...")
print_result("R2", 0.9399)
print_warning("显存不足")
print_error("训练失败")
print_success("训练完成")
print_divider(char='=', width=60)
print_header("PE-MMNet v4")
```

---

## 10. 模型工厂 (models.pe_tsnet_multimodal)

### create_model

统一模型工厂函数（支持 ablation 变体）。

```python
from models.pe_tsnet_multimodal import create_model

model = create_model(
    backbone_2d='resnet18',    # 'resnet18' | 'vit_small'
    backbone_1d='cnn_attn',    # 'cnn_attn' | 'transformer' | 'dlinear'
    fusion='cross_attn',       # 'cross_attn' | 'concat' | 'adaptive'
    seq_len=300,
    dropout=0.2,
    pretrained_2d=True
)
```

> 注意：`create_model` 是 ablation 实验用工厂函数，不对应任何 `run_train.py` 中的 `variant_key`。

### get_arch_specific_config

根据架构自动配置学习率和 dropout。

```python
from models.pe_tsnet_multimodal import get_arch_specific_config

config = get_arch_specific_config(
    backbone_2d='resnet18',
    backbone_1d='cnn_attn',
    args_lr=None,     # 可选：命令行传入学习率
    args_dropout=None # 可选：命令行传入dropout
)
# 返回: dict {'lr': float, 'dropout': float}
# 默认 lr=1e-3，ViT/Swin 架构自动降为 1e-4
```
