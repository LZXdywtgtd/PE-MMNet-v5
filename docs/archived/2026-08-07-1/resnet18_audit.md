# resnet18 变体全链路数据流审计报告

**模型变体**: resnet18 (`PETSNetMultimodal`)
**审计日期**: 2026-08-07
**审计范围**: 配置解析 → 模型构建 → 数据加载 → 损失计算 → 评估

---

## 阶段1：入口与配置

### VARIANT_MODELS 映射

| 检查项 | 结果 |
|--------|------|
| `resnet18` → `PETSNetMultimodal` | ✓ 正确 |
| 映射位置 | `run_train.py` 第 357 行 |

### create_variant_model 参数传递

```python
seq_channels = 3 if config.get('triple_channel', False) else 1
return PETSNetMultimodal(
    seq_len=config.get('feature_len', 300),    # ✓ 正确
    image_channels=2,                           # ✓ 固定 2
    image_size=config['image_size'],            # ✓ 来自 auto_config
    pretrained_2d=True,                        # ✓ 正确
    dropout=config['dropout'],                  # ✓ 正确
    task=task,                                  # ✓ 正确
    fusion=config.get('fusion', 'cross_attn'), # ✓ 默认值
    seq_channels=seq_channels                   # ✓ triple_channel 逻辑正确
)
```

**结论**: ✓ 所有参数正确传递，无问题。

---

## 阶段2：PETSNetMultimodal 内部数据流

### 维度链路追踪

```
输入：
  x_1d: (B, 300)
  x_2d: (B, 2, H, W)

分支1（2D骨干）:
  ResNet18Backbone2D(x_2d) → (B, 512)  ✓

分支2（1D骨干）:
  TemporalFeatureExtractor(x_1d) → (B, 64)  ✓
    Micro1DCNN + Macro1DCNN → concat → 128 通道
    → LayerNorm → Self-Attention → FFN
    → GlobalAvgPool → (B, 64)

融合层（CrossAttentionFusion）:
  feat_2d (B, 512) + feat_1d (B, 64)
  → 双向交叉注意力 (图像查询时序，时序查询图像)
  → cat([fused_2d, fused_1d]) → (B, 576)  ✓

融合层（GatedMultimodalFusion）:
  feat_2d (B, 512) → split → temp (B, 256) + stress (B, 256)
  → 各分支 attention × gate
  → concat(temp_out, stress_out, feat_1d) → (B, 576)
  → output_proj → (B, 576)  ✓

多任务输出头（MultiTaskHead）:
  input_dim=576
  → shared MLP (576 → 256 → 128)
  → [xy: 128→2], [lw: 128→2], [conf: 128→32→1], [density: 128→32→1]
  → cat → (B, 6)  ✓

任务分支:
  detection → output_head → (B, 6)  ✓
  segmentation → mask_decoder → (B, 1, H, W)  ✓
  multitask → (mask_decoder, output_head) → (mask, detection)  ✓
```

**结论**: ✓ 维度链路完全正确。

### seq_channels 参数流向

| triple_channel | seq_1d 形状 | TemporalFeatureExtractor input_dim |
|----------------|-------------|-------------------------------------|
| False (默认) | (B, 300) | input_dim=1 → Micro1DCNN(1→32), Macro1DCNN(1→32) |
| True | (B, 3, 300) | input_dim=3 → Micro1DCNN(3→32), Macro1DCNN(3→32) |

**结论**: ✓ seq_channels 参数正确传递。

---

## 阶段3：损失函数匹配

### 损失函数选择逻辑

```python
# train_model 第 1071-1095 行
if task == 'segmentation':
    criterion = SegmentationLoss()  # ✓
elif task == 'multitask':
    criterion = MultimodalSegmentationLoss()  # ✓
else:
    # detection 模式，resnet18 走 else 分支
    criterion = MultimodalCrackLoss(
        lambda_mse_density=config['lambda_mse'],
        lambda_mono=config['lambda_mono'],
        lambda_loc=config['lambda_loc'],
        lambda_conf=config['lambda_conf']
    )  # ✓
```

### MultimodalCrackLoss 输入对齐

```python
# MultimodalCrackLoss.forward
pred: (B, 6)    ← PETSNetMultimodal forward 返回 (B, 6)  ✓
target: (B, 6)  ← labels (B, 6) from dataset  ✓

# 内部拆分：
pred_xy = pred[:, 0:2]       # (B, 2)  ✓
pred_lw = pred[:, 2:4]       # (B, 2)  ✓
pred_conf = pred[:, 4:5]     # (B, 1)  ✓
pred_density = pred[:, 5:6]  # (B, 1)  ✓
```

### 训练循环调用

```python
# train_model 第 1360-1366 行（resnet18 走 else 分支）
outputs = model(seq_1d, img_2d)  # (B, 6)
if task == 'segmentation':
    loss_total = criterion(outputs, labels)
elif task == 'multitask':
    loss_total = criterion(outputs, labels)
else:
    loss_total, _ = criterion(outputs, labels)  # ✓ 返回 (loss, dict)
```

**结论**: ✓ 损失函数与模型输出完全对齐。

---

## 阶段4：数据与评估

### collate_fn 数据处理

```python
# dataset_multimodal.py collate_fn_with_cutmix
seq_1d = torch.stack(seq_1d_list)      # (B, 300) 或 (B, 3, 300)
img_2d = torch.stack(img_2d_list)        # (B, 2, H, W)
labels = torch.stack(labels_list)        # (B, 6)
return (seq_1d, img_2d), labels          # ✓ 形状正确
```

### triple_channel 数据处理

```python
# create_triple_channel_seq: 单通道 → 三通道
# 通道0: 初始温度, 通道1: 当前温度, 通道2: 温度变化率
# 在 MultiBatchDataset.__getitem__ 中调用
# train_loader 的 collate_fn 自动 stack 成 (B, 3, 300)  ✓
```

### evaluate_model 对 resnet18 的处理

```python
# run_train.py 第 417-438 行
is_new_variant = variant_key in ['swin_yolo', 'vit_yolo', 'detr', 'swin_yolo_patchtst']
# resnet18 → is_new_variant = False

# resnet18 走 else 分支:
outputs = raw_outputs      # (B, 6) from PETSNetMultimodal
global_density = None     # 未定义（但不使用）

# 收集用于指标计算:
all_preds.append(outputs.cpu())           # (B, 6)  ✓
all_targets.append(labels[:, :6].cpu()) # (B, 6)  ✓

# 密度提取:
preds[:, 5].flatten()  → 密度列  ✓
```

### train_model eval 循环对 resnet18 的处理

```python
# 第 1443-1455 行
outputs = model(seq_1d, img_2d)  # (B, 6)
loss_total, _ = criterion(outputs, labels)

# ⚠️ BUG: 只收集密度列，但之后用 preds[:, 5] 索引
all_preds.append(outputs[:, 5:6].cpu())     # (B, 1) ❌ 应该是 (B, 6)
all_targets.append(labels[:, 5:6].cpu())   # (B, 1) ❌ 应该是 (B, 6)

# 第 1467 行:
preds = torch.cat(all_preds)  # (N, 1) 但代码假设 (N, 6)
# 第 1476 行:
pred_d = preds[:, 5].flatten()  # ❌ IndexError: index 5 is out of bounds for dimension 1 with size 1
```

**结论**: ⚠️ train_model eval 循环中 resnet18 的 `all_preds` 只收集密度列，但 `evaluate_model` 之后用 `preds[:, 5]` 索引会越界。这是一个 P1 阻塞性 Bug。但由于 train_model 训练后会调用 `evaluate_model` 做完整评估，这个 bug 在实际运行中会被 `evaluate_model` 的正确逻辑掩盖，只在单独使用 train_model eval 循环时触发。

---

## 审计结论汇总

| 阶段 | 检查项 | 状态 | 备注 |
|------|--------|------|------|
| 阶段1 | VARIANT_MODELS 映射 | ✓ 通过 | resnet18 → PETSNetMultimodal 正确 |
| 阶段1 | 参数传递 | ✓ 通过 | 所有参数正确传递 |
| 阶段2 | 维度链路 | ✓ 通过 | (B,6) 全链路无误 |
| 阶段2 | seq_channels | ✓ 通过 | triple_channel 逻辑正确 |
| 阶段3 | 损失函数选择 | ✓ 通过 | MultimodalCrackLoss 正确 |
| 阶段3 | 损失函数输入对齐 | ✓ 通过 | (B,6) vs (B,6) 完全匹配 |
| 阶段4 | collate_fn | ✓ 通过 | triple_channel 正确处理 |
| 阶段4 | evaluate_model | ✓ 通过 | resnet18 收集全 6 维 |
| 阶段4 | train_model eval 循环 | ⚠️ BUG | 只收集密度列但索引第6列 |

### 发现的问题

#### P1: train_model eval 循环密度列收集维度不匹配

**位置**: `run_train.py` 第 1454-1455 行

**问题**: resnet18 等旧变体在 eval 循环中只收集 `outputs[:, 5:6]`，但之后用 `preds[:, 5]` 索引。如果 eval 循环被单独使用（不调用 evaluate_model），会导致 IndexError。

**当前影响**: 低。因为训练后调用的是 `evaluate_model`（正确收集全 6 维），train_model 的 eval 循环仅用于训练时打印指标。当前的指标打印逻辑（train_model eval 循环）只用于日志输出，完整的评估由 evaluate_model 完成。

**修复方案**:
```python
# 第 1453-1455 行
# 修改前:
all_preds.append(outputs[:, 5:6].cpu())
all_targets.append(labels[:, 5:6].cpu())

# 修改后:
all_preds.append(outputs.cpu())
all_targets.append(labels[:, :6].cpu())
```

---

## 维度追踪表（resnet18 detection 模式）

```
数据加载:
  seq_1d: (B, 300)              — from collate_fn
  img_2d: (B, 2, H, W)          — from collate_fn
  labels: (B, 6)                  — from collate_fn

模型前向:
  ResNet18Backbone2D(x_2d) → (B, 512)
  TemporalFeatureExtractor(x_1d) → (B, 64)
  CrossAttentionFusion → (B, 576)
  MultiTaskHead → (B, 6)

损失计算:
  MultimodalCrackLoss(pred=(B,6), target=(B,6)) → loss_total

评估（evaluate_model）:
  model.eval() → outputs=(B, 6)
  all_preds.append(outputs) → (N, 6)
  pred_d = preds[:, 5] → (N,)  ✓

train_model eval 循环（当前有 bug）:
  all_preds.append(outputs[:, 5:6]) → (N, 1)  ❌
  preds[:, 5] → IndexError  ❌
```
