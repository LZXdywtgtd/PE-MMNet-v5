# vit_yolo 变体全链路数据流审计报告

**模型变体**: vit_yolo (`ViTYOLOFPN`)
**审计日期**: 2026-08-07
**审计范围**: 配置解析 → 模型构建 → 数据加载 → 损失计算 → 评估

---

## 阶段1：入口与配置

### VARIANT_MODELS 映射

| 检查项 | 结果 |
|--------|------|
| `vit_yolo` → `ViTYOLOFPN` | ✓ 正确 |
| 映射位置 | `run_train.py` 第 360 行 |

### create_variant_model 参数传递

```python
# run_train.py 第 1717-1725 行
elif variant_key in ['swin_yolo', 'vit_yolo', 'detr', 'swin_yolo_patchtst']:
    return ModelClass(
        seq_len=config.get('feature_len', 300),  # ✓
        image_channels=2,                        # ✓
        image_size=config['image_size'],          # ✓
        pretrained_2d=True,                      # ✓
        dropout=config['dropout']                 # ✓
    )
```

**结论**: ✓ 所有参数正确传递。

---

## 阶段2：ViTYOLOFPN 内部数据流

### 维度链路追踪

```
输入：
  x_1d: (B, 300)
  x_2d: (B, 2, H, W)  # H, W 可为 256/384/512

分支1（2D骨干）:
  ViTYOLOBackbone2D(x_2d)
    → input_resize: 2ch → 3ch
    → ViT-Small (img_size=224 硬编码)
      输入图像 resize 到 224×224（如果有差异）
    → features[-1] (B, 384, 224/32=7, 224/32=7)
    → F.interpolate → (B, 384, 16, 16)
    → proj → (B, 256, 16, 16)

  fpn: (B, 256, 16, 16) → (B, 256, 16, 16)  ✓
  YOLOFPNHead([feat]) → (B, 256, 6)  ✓

分支2（1D骨干）:
  TemporalFeatureExtractor(x_1d) → (B, 64)  ✓

推理分支:
  grid_feat = grid_pred[best_idx] → (B, 6)
  CrossAttentionFusion(dim_2d=6, dim_1d=64) → (B, 70)
  MultiTaskHead(input_dim=70) → (B, 6)  ✓

训练分支:
  return (grid_pred, global_density)
    grid_pred: (B, 256, 6)  # 固定16×16=256网格
    global_density: (B, 1)  ✓
```

### actual_grid_size / actual_num_grids

```python
# ViTYOLOFPN.__init__
self.actual_grid_size = 16    # 固定
self.actual_num_grids = 256   # 固定
```

**⚠️ 注意**: ViT 的 grid_size 是固定的（16），与输入图像尺寸无关。即使 `--image_size 512`，YOLO 仍输出 16×16=256 网格。相比之下，`swin_yolo` 的 grid_size 是动态的（`image_size // 32`）。

### ⚠️ image_size 参数被忽略

```python
# ViTYOLOFPN.__init__ 第 464 行
self.grid_size = grid_size  # 16 但 image_size 参数未使用！

# ViTYOLOBackbone2D.__init__ 第 119 行
def __init__(self, pretrained=True, output_size=16, input_size=224):
    # image_size 参数未传递进来！ViTYOLOFPN 没有将 image_size 传给 backbone
```

`ViTYOLOFPN` 接收 `image_size` 参数但未传递给 `ViTYOLOBackbone2D`，骨干内部硬编码 `input_size=224`。

这意味着无论用户传 `--image_size 256/384/512`，图像都会被 resize 回 224 处理。

**影响评估**: 中等。ViT 预训练权重基于 224×224 输入，在非 224 分辨率下性能可能下降。设计决策而非 bug。

### CrossAttentionFusion dim_2d 问题

与 swin_yolo 完全一致：
```python
self.fusion = CrossAttentionFusion(dim_2d=6, dim_1d=64)  # ✓
self.output_head = MultiTaskHead(input_dim=70)            # ✓
```

**结论**: ✓ 维度链路正确。

---

## 阶段3：损失函数匹配

### 损失函数 + assigner 选择逻辑

```python
# train_model 第 1074-1085 行
if variant_key in ['swin_yolo', 'vit_yolo', 'swin_yolo_patchtst']:
    criterion = YOLOLoss(...)  # ✓
    if variant_key in ['swin_yolo', 'swin_yolo_patchtst']:
        actual_grid_size = model.actual_grid_size
    else:
        actual_grid_size = 16  # ✓ vit_yolo 固定 16
    assigner = YOLOTargetAssigner(grid_size=actual_grid_size, nearby_range=2)
```

### YOLOLoss 输入对齐

```python
# 模型输出: grid_pred (B, 256, 6), global_density (B, 1)
# assigner 输出: assigned_target (B, 256, 6), pos_mask (B, 256)
# 损失计算: criterion(grid_pred, assigned_target, global_density, labels[:, 5:6], pos_mask)
#   ✓ 所有维度匹配
```

**结论**: ✓ 损失函数与模型输出完全对齐。

---

## 阶段4：数据与评估

### collate_fn 对 vit_yolo 的处理

```python
# 与 swin_yolo 相同
seq_1d = torch.stack(seq_1d_list)  # (B, 300)  ✓
img_2d = torch.stack(img_2d_list)  # (B, 2, H, W)  ✓
labels = torch.stack(labels_list)  # (B, 6)  ✓
```

**vit_yolo 不支持 triple_channel**（硬编码 `input_dim=1`）。

### evaluate_model 对 vit_yolo 的处理

```python
# 与 swin_yolo 完全一致
is_new_variant = True  # vit_yolo in new_variants ✓
model.eval() → (B, 6), (B, 1)
outputs, global_density = raw_outputs

# task == 'detection':
all_preds.append(outputs.cpu())          # (B, 6) ✓
all_targets.append(labels[:, :6].cpu()) # (B, 6) ✓
preds = np.vstack → (N, 6)
pred_d = preds[:, 5] → (N,)  ✓
```

**结论**: ✓ evaluate_model 正确处理。

---

## 与 swin_yolo 的差异

| 特性 | swin_yolo | vit_yolo |
|------|-----------|----------|
| 2D 骨干 | Swin-Tiny | ViT-Small |
| 实际 grid_size | 动态 (image_size/32) | 固定 16 |
| image_size 影响 | 改变特征图尺寸 | 被 resize 回 224 |
| num_grids (512) | 256 | 256 |
| num_grids (256) | 64 | 256 |

---

## 审计结论汇总

| 阶段 | 检查项 | 状态 | 备注 |
|------|--------|------|------|
| 阶段1 | VARIANT_MODELS 映射 | ✓ 通过 | vit_yolo → ViTYOLOFPN 正确 |
| 阶段1 | 参数传递 | ✓ 通过 | 所有参数正确传递 |
| 阶段2 | 维度链路 | ✓ 通过 | 固定 16×16=256 网格 |
| 阶段2 | CrossAttentionFusion dim | ✓ 通过 | dim_2d=6 → 70D → 6D |
| 阶段3 | YOLOLoss 选择 | ✓ 通过 | actual_grid_size=16 |
| 阶段3 | 损失输入对齐 | ✓ 通过 | (B, 256, 6) vs (B, 256, 6) |
| 阶段4 | evaluate_model | ✓ 通过 | 正确收集全 6 维 |
| 阶段4 | collate_fn | ✓ 通过 | 不支持 triple_channel |

### 发现的问题

#### P2: image_size 参数在 vit_yolo 中被忽略

**位置**: `ViTYOLOFPN.__init__` + `ViTYOLOBackbone2D.__init__`

**问题**: `ViTYOLOFPN` 接收 `image_size` 参数但未传递给骨干网络。`ViTYOLOBackbone2D` 硬编码 `input_size=224`，图像会被 resize 回 224 处理，与用户指定的 `image_size` 无关。

**影响**: 中等。ViT 预训练权重基于 224×224，这是设计决策但可能导致非 224 分辨率下性能下降。

**建议**: 要么移除 `--image_size` 对 vit_yolo 的支持，要么将 image_size 传给骨干网络的 input_resize 阶段。

#### P2: num_grids 不随 image_size 变化（与 swin_yolo 行为不一致）

**问题**: swin_yolo 的 num_grids 随 image_size 变化（256→64, 512→256），但 vit_yolo 始终为 256。

**影响**: 低。这是架构差异（ViT 固定输出 16×16，Swin 动态），但可能让用户困惑。

---

## 维度追踪表（vit_yolo detection 模式）

```
数据加载:
  seq_1d: (B, 300)
  img_2d: (B, 2, H, W)  # H,W 可为 256/384/512

模型前向（训练模式）:
  ViTYOLOBackbone2D → resize to 224 → ViT-Small → (B, 384, 7, 7)
  → interpolate → (B, 384, 16, 16)
  → proj → (B, 256, 16, 16)
  → YOLOFPNHead → (B, 256, 6)  # 固定 16×16=256 网格
  TemporalFeatureExtractor → (B, 64)
  return (grid_pred=(B, 256, 6), global_density=(B, 1))

模型前向（推理模式）:
  grid_feat = grid_pred[best_idx] → (B, 6)
  CrossAttentionFusion → (B, 70)
  MultiTaskHead → (B, 6)  ✓

损失计算:
  YOLOTargetAssigner(grid_size=16) → (B, 256, 6), (B, 256) pos_mask
  YOLOLoss(grid_pred, assigned_target, ...)  ✓

评估:
  model.eval() → (B, 6), (B, 1)
  pred_d = preds[:, 5] → (N,)  ✓
```
