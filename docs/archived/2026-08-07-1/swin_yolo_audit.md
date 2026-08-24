# swin_yolo 变体全链路数据流审计报告

**模型变体**: swin_yolo (`SwinYOLOFPN`)
**审计日期**: 2026-08-07
**审计范围**: 配置解析 → 模型构建 → 数据加载 → 损失计算 → 评估

---

## 阶段1：入口与配置

### VARIANT_MODELS 映射

| 检查项 | 结果 |
|--------|------|
| `swin_yolo` → `SwinYOLOFPN` | ✓ 正确 |
| 映射位置 | `run_train.py` 第 359 行 |

### create_variant_model 参数传递

```python
# run_train.py 第 1717-1725 行
elif variant_key in ['swin_yolo', 'vit_yolo', 'detr', 'swin_yolo_patchtst']:
    return ModelClass(
        seq_len=config.get('feature_len', 300),  # ✓
        image_channels=2,                        # ✓
        image_size=config['image_size'],          # ✓ 来自 auto_config
        pretrained_2d=True,                      # ✓
        dropout=config['dropout()                # ✓
    )
```

**结论**: ✓ 所有参数正确传递，无问题。

---

## 阶段2：SwinYOLOFPN 内部数据流

### 维度链路追踪

```
输入：
  x_1d: (B, 300)
  x_2d: (B, 2, H, W)

分支1（2D骨干）:
  SwinBackbone2D(x_2d)
    → input_proj: 2ch → 3ch
    → Swin-Tiny (img_size=H)
    → 返回多尺度特征列表: [C1, C2, C3, C4]  # NHWC→NCHW 转换
  取最后一层: multi_scale_features[-1]  # (B, 768, H/32, W/32)

  yolo_proj: 768 → 256  ✓
  YOLOFPNHead([feat])
    → feat (B, 256, G, G) 其中 G = max(H/32, W/32)
    → pred (B, 6, G, G)
    → reshape → (B, G*G, 6)

  输出 grid_pred: (B, num_grids, 6)
    num_grids = G*G = (image_size/32)^2
    image_size=256 → 8×8=64 grids
    image_size=512 → 16×16=256 grids  ✓

分支2（1D骨干）:
  TemporalFeatureExtractor(x_1d) → (B, 64)  ✓

推理分支（model.eval()）:
  grid_feat = grid_pred[best_idx] → (B, 6)
  CrossAttentionFusion(dim_2d=6, dim_1d=64) → (B, 70)
  MultiTaskHead(input_dim=70) → (B, 6)  ✓

训练分支（model.train()）:
  return (grid_pred, global_density)
    grid_pred: (B, num_grids, 6)
    global_density: (B, 1)  ✓
```

### ⚠️ Grid Size 不一致风险分析

**SwinYOLOFPN 硬编码 `grid_size=16` 到 YOLOFPNHead**，但 YOLOFPNHead 实际输出由特征图大小决定：

| image_size | actual_grid_size | 实际 num_grids | grid_size 参数 | assigner num_grids | 是否一致 |
|------------|-----------------|----------------|---------------|-------------------|---------|
| 256 | 8 | 64 | 16 | 64 | ✓ actual_grid_size |
| 512 | 16 | 256 | 16 | 256 | ✓ actual_grid_size |

`YOLOTargetAssigner` 使用 `actual_grid_size`（第 1085 行）：
```python
assigner = YOLOTargetAssigner(grid_size=actual_grid_size, nearby_range=2)
```
→ assigner 始终与模型真实输出匹配 ✓

**结论**: ✓ 虽然 `grid_size=16` 硬编码，但 `actual_grid_size` / `assigner` 使用动态值，维度对齐正确。

### CrossAttentionFusion dim_2d 问题

```python
# SwinYOLOFPN.__init__ 第 369 行
self.fusion = CrossAttentionFusion(dim_2d=6, dim_1d=64)
# 输入: feat_2d=(B, 6) from grid_feat, feat_1d=(B, 64)
# 输出: (B, 6+64) = (B, 70)
```

但 `MultiTaskHead` 的 `input_dim=70`：
```python
# SwinYOLOFPN.__init__ 第 373-376 行
self.output_head = MultiTaskHead(
    input_dim=fused_dim,  # 6 + 64 = 70 ✓
    hidden_dim=128,
    dropout=dropout
)
```

**结论**: ✓ 维度对齐正确。

---

## 阶段3：损失函数匹配

### 损失函数 + assigner 选择逻辑

```python
# train_model 第 1074-1085 行
if variant_key in ['swin_yolo', 'vit_yolo', 'swin_yolo_patchtst']:
    criterion = YOLOLoss(  # ✓
        lambda_box=1.0,
        lambda_conf=1.0,
        lambda_mono=0.1
    )
    if variant_key in ['swin_yolo', 'swin_yolo_patchtst']:
        actual_grid_size = model.actual_grid_size  # ✓ 动态
    else:
        actual_grid_size = 16  # vit_yolo
    assigner = YOLOTargetAssigner(grid_size=actual_grid_size, nearby_range=2)
```

### 训练循环调用

```python
# 训练循环（非FP16分支，第 1344-1348 行）
if variant_key in ['swin_yolo', 'vit_yolo', 'swin_yolo_patchtst']:
    grid_pred, global_density = model(seq_1d, img_2d)  # (B, num_grids, 6), (B, 1)
    assigned_target, pos_mask = assigner(labels)  # (B, num_grids, 6), (B, num_grids)
    loss_total = criterion(grid_pred, assigned_target, global_density, labels[:, 5:6], pos_mask)
    # ✓ grid_pred (B, num_grids, 6) vs assigned_target (B, num_grids, 6)
```

### YOLOLoss 输入对齐

```python
# YOLOLoss.forward
pred: (B, num_grids, 6)        ← grid_pred from model ✓
target: (B, num_grids, 6)      ← assigned_target from assigner ✓
global_density: (B, 1)         ← model global_density ✓
target_density: (B, 1)          ← labels[:, 5:6] ✓
positive_mask: (B, num_grids)  ← assigner pos_mask ✓
```

**结论**: ✓ 损失函数与模型输出完全对齐。

---

## 阶段4：数据与评估

### collate_fn 对 swin_yolo 的处理

```python
# dataset_multimodal.py collate_fn_with_cutmix
seq_1d = torch.stack(seq_1d_list)  # (B, 300)  ✓
img_2d = torch.stack(img_2d_list)  # (B, 2, H, W)  ✓
labels = torch.stack(labels_list)    # (B, 6)  ✓
```

**swin_yolo 不支持 triple_channel**（模型中 hardcode `input_dim=1`）：
```python
# SwinYOLOFPN.__init__ 第 360-364 行
self.backbone_1d = TemporalFeatureExtractor(
    input_dim=1,  # ❌ 硬编码，不支持 triple_channel
    ...
)
```
但 `create_variant_model` 对 swin_yolo 不传递 `seq_channels`，所以这个参数不起作用。实际 `input_dim=1`。

### evaluate_model 对 swin_yolo 的处理

```python
# run_train.py 第 417-443 行
is_new_variant = True  # swin_yolo in new_variants ✓

model.eval() → outputs=(B, 6), global_density=(B, 1)
outputs, global_density = raw_outputs
outputs.dim() == 3? → False (inference returns (B, 6))
→ outputs unchanged

# task == 'detection':
all_preds.append(outputs.cpu())          # (B, 6) ✓
all_targets.append(labels[:, :6].cpu()) # (B, 6) ✓

# 密度提取:
preds = np.vstack → (N, 6)
pred_d = preds[:, 5] → (N,)  ✓
```

### train_model eval 循环对 swin_yolo 的处理

```python
# 第 1410-1430 行
model_output, global_density = model(seq_1d, img_2d)
# model.eval() → returns (B, 6), (B, 1)

model_output.ndim == 3? → False
→ expand 路径: grid_pred = model_output.unsqueeze(1).expand(-1, actual_num_grids, -1)
# (B, 6) → (B, 1, 6) → (B, num_grids, 6) ← 全是重复的第一个网格！
→ 这是推理模式的简化处理，虽然不是最优，但不会崩溃

# 但更关键的是：swim_yolo eval 循环只用于打印日志
# 完整评估走 evaluate_model（逻辑正确）
```

**结论**: ⚠️ eval 循环的 expand 逻辑不优雅（所有网格都是重复的），但不影响实际评估结果（由 evaluate_model 覆盖）。

---

## 审计结论汇总

| 阶段 | 检查项 | 状态 | 备注 |
|------|--------|------|------|
| 阶段1 | VARIANT_MODELS 映射 | ✓ 通过 | swin_yolo → SwinYOLOFPN 正确 |
| 阶段1 | 参数传递 | ✓ 通过 | 所有参数正确传递 |
| 阶段2 | 维度链路 | ✓ 通过 | num_grids 动态计算，与 assigner 一致 |
| 阶段2 | CrossAttentionFusion dim | ✓ 通过 | dim_2d=6, dim_1d=64 → 70D → 6D |
| 阶段3 | YOLOLoss 选择 | ✓ 通过 | 正确 |
| 阶段3 | YOLOTargetAssigner | ✓ 通过 | 使用 actual_grid_size |
| 阶段3 | 损失输入对齐 | ✓ 通过 | (B, num_grids, 6) vs (B, num_grids, 6) |
| 阶段4 | collate_fn | ✓ 通过 | 不支持 triple_channel（硬编码 input_dim=1） |
| 阶段4 | evaluate_model | ✓ 通过 | swin_yolo 正确收集全 6 维 |
| 阶段4 | train_model eval 循环 | ⚠️ 注意 | expand 逻辑不优雅但不影响结果 |

### 发现的问题

#### P2: train_model eval 循环 swin_yolo expand 逻辑不优雅

**位置**: `run_train.py` 第 1418-1427 行

**问题**: 推理模式下，所有网格被填充为相同的 `model_output`（第一个网格的复制），然后用所有网格标记为正样本的 pos_mask 计算损失。这是简化的推理模式处理，不是真实的多网格预测。

**当前影响**: 低。eval 循环仅用于打印训练时指标，完整评估由 `evaluate_model` 完成。

#### 设计建议: temporal backbone 不支持 triple_channel

SwinYOLOFPN 的 1D 分支硬编码 `input_dim=1`，如果用户传 `--triple_channel`，1D 分支仍使用单通道。应在文档中说明 `--triple_channel` 仅支持 resnet18 变体。

---

## 维度追踪表（swin_yolo detection 模式）

```
数据加载:
  seq_1d: (B, 300)              — from collate_fn
  img_2d: (B, 2, H, W)         — from collate_fn
  labels: (B, 6)                — from collate_fn

模型前向（训练模式 model.train()）:
  SwinBackbone → multi_scale_features[-1] → (B, 768, G, G)
  yolo_proj → (B, 256, G, G)
  YOLOFPNHead → (B, G*G, 6)  # G = image_size/32
  TemporalFeatureExtractor → (B, 64)
  return (grid_pred=(B, G*G, 6), global_density=(B, 1))

模型前向（推理模式 model.eval()）:
  ...同上...
  best_idx = argmax(conf) → (B,)
  grid_feat = grid_pred[best_idx] → (B, 6)
  CrossAttentionFusion(grid_feat, feat_1d) → (B, 70)
  MultiTaskHead → (B, 6)

损失计算:
  YOLOTargetAssigner(labels) → (B, G*G, 6), (B, G*G) pos_mask
  YOLOLoss(grid_pred, assigned_target, global_density, labels[:, 5:6], pos_mask)
    → grid_pred (B, G*G, 6) vs assigned_target (B, G*G, 6)  ✓

评估（evaluate_model）:
  model.eval() → (B, 6), (B, 1)
  all_preds.append(outputs) → (N, 6)  ✓
  pred_d = preds[:, 5] → (N,)  ✓
```
