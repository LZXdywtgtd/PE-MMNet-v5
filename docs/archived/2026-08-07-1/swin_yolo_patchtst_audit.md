# swin_yolo_patchtst 变体全链路数据流审计报告

**模型变体**: swin_yolo_patchtst (`SwinYOLOFPNWithPatchTST`)
**审计日期**: 2026-08-07
**审计范围**: 配置解析 → 模型构建 → 数据加载 → 损失计算 → 评估

---

## 与 swin_yolo 的差异概览

| 组件 | swin_yolo | swin_yolo_patchtst |
|------|-----------|-------------------|
| 2D 骨干 | Swin-Tiny | Swin-Tiny（相同） |
| YOLO Head | YOLOFPNHead | YOLOFPNHead（相同） |
| 1D 骨干 | `TemporalFeatureExtractor` | `PatchTST1D`（核心差异） |
| 融合层 | `CrossAttentionFusion(dim_2d=6, dim_1d=64)` | 同左 |
| 输出头 | `MultiTaskHead(input_dim=70)` | 同左 |
| 损失函数 | YOLOLoss | YOLOLoss（相同） |
| image_size 默认值 | 256 | 256（已修复，原为 512） |

---

## 阶段1：入口与配置

### VARIANT_MODELS 映射

| 检查项 | 结果 |
|--------|------|
| `swin_yolo_patchtst` → `SwinYOLOFPNWithPatchTST` | ✓ 正确 |
| 映射位置 | `run_train.py` 第 364 行 |

### create_variant_model 参数传递

```python
# run_train.py 第 1717-1725 行
elif variant_key in ['swin_yolo', 'vit_yolo', 'detr', 'swin_yolo_patchtst']:
    return ModelClass(
        seq_len=config.get('feature_len', 300),  # ✓
        image_channels=2,                        # ✓
        image_size=config['image_size'],          # ✓
        pretrained_2d=True,                      # ✓
        dropout=config['dropout']               # ✓
    )
```

**结论**: ✓ 所有参数正确传递。

---

## 阶段2：SwinYOLOFPNWithPatchTST 内部数据流

### 维度链路追踪

```
输入：
  x_1d: (B, 300)
  x_2d: (B, 2, H, W)

分支1（2D骨干）:
  # 与 swin_yolo 完全相同
  SwinBackbone2D → multi_scale_features[-1] → (B, 768, G, G)
  yolo_proj: 768 → 256  ✓
  YOLOFPNHead([feat]) → (B, G*G, 6)
  其中 G = image_size/32
  image_size=256 → 8×8=64 grids
  image_size=512 → 16×16=256 grids  ✓

分支2（1D骨干）:
  PatchTST1D(x_1d)
    → (B, num_patches, d_model) = (B, 30, 64)
    → mean pooling → (B, 64)  ✓

  # 注意：PatchTST1D 也输出 (B, 64)，与 TemporalFeatureExtractor 维度一致 ✓

推理分支:
  grid_feat = grid_pred[best_idx] → (B, 6)
  CrossAttentionFusion(dim_2d=6, dim_1d=64) → (B, 70)
  MultiTaskHead(input_dim=70) → (B, 6)  ✓

训练分支:
  return (grid_pred, global_density)
    grid_pred: (B, num_grids, 6)
    global_density: (B, 1)  ✓
```

### PatchTST1D 输出维度验证

```python
# pe_tsnet_patchtst.py PatchTST1D.forward
# seq_len=300, patch_size=10 → num_patches=30
# d_model=64, output_dim=64
x = self.patch_embed(x)          # (B, 30, 64)
x = self.transformer(x)          # (B, 30, 64)
x = x.mean(dim=1)              # (B, 64)
x = self.output_proj(x)         # (B, 64)
→ 输出 (B, 64) ✓ 与 TemporalFeatureExtractor 一致
```

### PatchTST fallback 机制

```python
# SwinYOLOFPNWithPatchTST.__init__ 第 594-608 行
if PATCHTST_AVAILABLE:
    self.backbone_1d = PatchTST1D(...)  # ✓
else:
    self.backbone_1d = TemporalFeatureExtractor(...)  # fallback ✓
```

如果 `PatchTST1D` 导入失败，回退到 `TemporalFeatureExtractor`，两者输出维度一致 `(B, 64)`，不会破坏后续融合链路。

### actual_grid_size / actual_num_grids

```python
# 与 swin_yolo 完全一致
self.actual_grid_size = image_size // 32
self.actual_num_grids = self.actual_grid_size ** 2
```

**结论**: ✓ 维度链路正确，与 swin_yolo 完全一致。

---

## 阶段3：损失函数匹配

### 损失函数 + assigner 选择逻辑

```python
# train_model 第 1074-1085 行
if variant_key in ['swin_yolo', 'vit_yolo', 'swin_yolo_patchtst']:
    criterion = YOLOLoss(...)  # ✓ 与 swin_yolo 相同
    if variant_key in ['swin_yolo', 'swin_yolo_patchtst']:
        actual_grid_size = model.actual_grid_size  # ✓ 动态
    else:
        actual_grid_size = 16  # vit_yolo
    assigner = YOLOTargetAssigner(grid_size=actual_grid_size, nearby_range=2)
```

### YOLOLoss 输入对齐

```python
# 模型输出: grid_pred (B, num_grids, 6), global_density (B, 1)
# assigner 输出: assigned_target (B, num_grids, 6), pos_mask (B, num_grids)
# 损失计算: criterion(grid_pred, assigned_target, global_density, labels[:, 5:6], pos_mask)
#   ✓ 所有维度匹配
```

**结论**: ✓ 与 swin_yolo 完全一致，损失函数完全对齐。

---

## 阶段4：数据与评估

### collate_fn 对 swin_yolo_patchtst 的处理

```python
# 与 swin_yolo 完全相同
seq_1d = torch.stack(seq_1d_list)  # (B, 300) ✓
img_2d = torch.stack(img_2d_list)  # (B, 2, H, W) ✓
labels = torch.stack(labels_list)    # (B, 6) ✓
```

**swin_yolo_patchtst 不支持 triple_channel**（PatchTST1D 或 TemporalFeatureExtractor fallback 均使用单通道序列）。

### evaluate_model 对 swin_yolo_patchtst 的处理

```python
# 与 swin_yolo 完全相同
is_new_variant = True  # swin_yolo_patchtst in new_variants ✓
model.eval() → (B, 6), (B, 1)
# task == 'detection':
all_preds.append(outputs.cpu())          # (B, 6) ✓
all_targets.append(labels[:, :6].cpu()) # (B, 6) ✓
pred_d = preds[:, 5] → (N,) ✓
```

**结论**: ✓ 与 swin_yolo 完全一致。

---

## 审计结论汇总

| 阶段 | 检查项 | 状态 | 备注 |
|------|--------|------|------|
| 阶段1 | VARIANT_MODELS 映射 | ✓ 通过 | swin_yolo_patchtst → SwinYOLOFPNWithPatchTST |
| 阶段1 | 参数传递 | ✓ 通过 | seq_len/image_size/dropout 正确 |
| 阶段2 | 维度链路 | ✓ 通过 | PatchTST 输出 (B,64) 与 TemporalFeatureExtractor 一致 |
| 阶段2 | PatchTST fallback | ✓ 通过 | 导入失败时回退到 TemporalFeatureExtractor |
| 阶段2 | CrossAttentionFusion dim | ✓ 通过 | dim_2d=6, dim_1d=64 → 70D → 6D |
| 阶段3 | YOLOLoss + assigner | ✓ 通过 | 与 swin_yolo 完全一致 |
| 阶段4 | evaluate_model | ✓ 通过 | 正确收集全 6 维 |
| 阶段4 | collate_fn | ✓ 通过 | 不支持 triple_channel |

### 与 swin_yolo 的差异总结

swin_yolo_patchtst 与 swin_yolo 在**架构和数据流上完全一致**，唯一的区别是：

1. **1D 骨干**: `PatchTST1D` 替代 `TemporalFeatureExtractor`
   - 两者输出维度完全相同：`(B, 64)`
   - PatchTST 使用 patch + Transformer，理论上捕获更长程的时序依赖
   - 有 fallback 机制（`PATCHTST_AVAILABLE`），导入失败时使用 `TemporalFeatureExtractor`

2. **image_size 默认值**: 均为 256（已修复）

**结论**: swin_yolo_patchtst 是 swin_yolo 的"即插即用"变体，不破坏任何现有链路。

---

## 维度追踪表（swin_yolo_patchtst detection 模式）

```
数据加载:
  seq_1d: (B, 300)
  img_2d: (B, 2, H, W)
  labels: (B, 6)

模型前向（训练模式）:
  SwinBackbone → multi_scale_features[-1] → (B, 768, G, G)
  yolo_proj → (B, 256, G, G)
  YOLOFPNHead → (B, G*G, 6)  # G = image_size/32
  PatchTST1D(x_1d) → (B, 64)  # 核心差异：Patch embedding + Transformer + pooling
  return (grid_pred=(B, G*G, 6), global_density=(B, 1))

模型前向（推理模式）:
  ...同上...
  grid_feat = grid_pred[best_idx] → (B, 6)
  CrossAttentionFusion → (B, 70)
  MultiTaskHead → (B, 6)  ✓

损失计算:
  YOLOTargetAssigner(grid_size=actual_grid_size) → (B, G*G, 6), (B, G*G) pos_mask
  YOLOLoss(grid_pred, assigned_target, ...)  ✓

评估:
  model.eval() → (B, 6), (B, 1)
  pred_d = preds[:, 5] → (N,) ✓
```
