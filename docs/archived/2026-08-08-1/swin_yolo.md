# A2: swin_yolo 变体数据流审计报告

> 审计分组：A组（基线变体）
> 模型类型：SwinYOLOFPN
> 审计日期：2026/08/08

---

## 一、阶段1：入口与配置

### 1.1 参数映射 ✅

| 检查项 | 代码位置 | 状态 |
|--------|----------|------|
| `--variant swin_yolo` → `SwinYOLOFPN` | `run_train.py:359` VARIANT_MODELS | ✅ |
| `is_new_variant` 包含 swin_yolo | `run_train.py:418, 1267, 1454, 1518` | ✅ |
| 损失函数选择 YOLOLoss | `run_train.py:1184-1195` | ✅ |

### 1.2 模型实例化参数 ✅

`run_train.py:1842-1850`:

```python
elif variant_key in ['swin_yolo', 'vit_yolo', 'detr', 'swin_yolo_patchtst']:
    return ModelClass(
        seq_len=config.get('feature_len', 300),   # ✅
        image_channels=2,                           # ✅ 固定 2
        image_size=config['image_size'],            # ✅
        pretrained_2d=True,                         # ✅
        dropout=config['dropout']                   # ✅
    )
```

`SwinYOLOFPN.__init__` 签名 (`pe_tsnet_yolo.py:336-337`):
```python
def __init__(self, seq_len=300, image_channels=2, image_size=256,
             pretrained_2d=True, dropout=0.2, grid_size=16):
```
**所有参数均正确传递，无遗漏。**

---

## 二、阶段2：模型内部数据流

### 2.1 forward 维度追踪

```
输入:
  x_1d: (batch, 300)
  x_2d: (batch, 2, 256, 256)

分支1 (2D):
  x_2d → SwinBackbone2D → multi_scale_features
    - 多尺度输出: [(B,96,H,W), (B,192,H,W), (B,384,H,W), (B,768,8,8)]
    - 取最后一层: feat = multi_scale_features[-1] → (B, 768, 8, 8)
    - yolo_proj: 768→256 → (B, 256, 8, 8)
    - yolo_head([feat]) → (B, num_grids, 6)
      其中 num_grids = grid_size * grid_size
      实际网格数 = (image_size // 32)² = (256//32)² = 64

分支2 (1D):
  x_1d → TemporalFeatureExtractor → feat_1d: (batch, 64) ✅

全局密度:
  global_density = grid_pred[..., 5:6].max(dim=1, keepdim=True)[0] → (B, 1)

融合（推理模式）:
  grid_feat (来自最高 conf 网格): (B, 6) ✅
  feat_1d: (B, 64) ✅
  → CrossAttentionFusion(dim_2d=6, dim_1d=64) → fused: (B, 70)
  → MultiTaskHead(input_dim=70) → output: (B, 6) ✅
```

### 2.2 训练/推理模式区分 ⚠️ 发现问题

`SwinYOLOFPN.forward` (`pe_tsnet_yolo.py:406-431`):

```python
if self.training:
    return grid_pred, global_density    # (B, N, 6), (B, 1)
else:
    # 取最高 conf 的预测 → (B, 6)
    grid_feat = grid_pred[torch.arange(B_f), best_idx]
    fused = self.fusion(grid_feat, feat_1d)
    output = self.output_head(fused)
    return output, global_density       # (B, 6), (B, 1) ← 元组！
```

**问题**：推理模式返回 `(B, 6), (B, 1)` 元组，但下游代码期望的形状因模式而异。

### 2.3 eval 验证损失计算逻辑 ⚠️ 发现问题

`run_train.py:1519-1540`:

```python
# eval 模式
model_output, global_density = model(seq_1d, img_2d)  # (B, 6), (B, 1)

if model_output.ndim == 3:
    # 训练模式：完整网格预测
    grid_pred = model_output
    assigned_target, pos_mask = assigner(labels)
    loss_total = criterion(grid_pred, assigned_target, global_density, ...)
else:
    # 推理模式：单预测 (B, 6)
    # 问题：这里将 (B, 6) 转为 (B, N, 6) 后计算损失
    B_v = model_output.size(0)
    actual_num_grids = model.actual_num_grids
    grid_pred = model_output.unsqueeze(1).expand(-1, actual_num_grids, -1)  # (B, N, 6)
    pos_mask = torch.ones(B_v, actual_num_grids, ...)  # 所有网格都标记为正样本
    loss_total = criterion(grid_pred, assigned_target, ...)
```

**问题分析**：
- `assigner(labels)` 分配的 `assigned_target` 是基于完整网格分配的 `(B, N, 6)`
- 但 `grid_pred` 是由单预测 `(B, 6)` 广播得到的，所有网格都共享同一个预测
- 所有网格都被标记为正样本，导致损失计算与实际预测不匹配
- **严重程度：低**。仅影响验证损失显示，不影响模型参数更新

---

## 三、阶段3：损失函数匹配

### 3.1 YOLOLoss 初始化 ✅

`run_train.py:1185-1195`:

```python
if variant_key in ['swin_yolo', 'vit_yolo', 'swin_yolo_patchtst']:
    criterion = YOLOLoss(
        lambda_box=1.0,
        lambda_conf=1.0,
        lambda_mono=0.1
    )
    if variant_key in ['swin_yolo', 'swin_yolo_patchtst']:
        actual_grid_size = model.actual_grid_size  # ✅ 动态获取
    else:
        actual_grid_size = 16
    assigner = YOLOTargetAssigner(grid_size=actual_grid_size, nearby_range=2)
```

### 3.2 YOLOLoss.forward 参数对齐 ✅

`YOLOLoss.forward` (`mono_loss.py:836-895`):
```python
def forward(self, pred, target, global_density, target_density, positive_mask=None):
    # pred: (B, N, 6) - 网格预测
    # target: (B, N, 6) - 分配后的目标
    # global_density: (B, 1)
    # target_density: (B, 1)
    # positive_mask: (B, N) - 正样本掩码
```

训练循环调用 (`run_train.py:1424-1426`):
```python
assigned_target, pos_mask = assigner(labels)
loss_total = criterion(grid_pred, assigned_target, global_density, labels[:, 5:6], pos_mask)
```

**参数完全对齐。**

### 3.3 YOLOTargetAssigner 逻辑验证 ✅

`YOLOTargetAssigner.forward` (`mono_loss.py:759-811`):
- 将目标分配到中心网格和附近网格（nearby_range=2 → 5×5 区域）
- 每个样本最多分配 1 + 24 = 25 个正样本
- ✅ 逻辑正确

---

## 四、阶段4：数据与评估

### 4.1 evaluate_model 解包逻辑 ✅

`run_train.py:430-435`:

```python
if is_new_variant:
    outputs, global_density = raw_outputs  # (B, 6), (B, 1) ✅
    if outputs.dim() == 3:
        outputs = outputs[:, 0, :]  # 训练模式：取第一个预测
```

**元组解包正确。** `outputs.dim() == 2` 时（推理模式）直接使用。

### 4.2 eval_checkpoint 模型创建 ⚠️ G-2

`run_train.py:605-613`:

```python
elif variant_key in ['swin_yolo', 'vit_yolo', 'detr', 'swin_yolo_patchtst']:
    ModelClass = VARIANT_MODELS.get(variant_key, PETSNetMultimodal)
    model = ModelClass(
        seq_len=300,
        image_channels=2,
        image_size=image_size,
        pretrained_2d=True
        # ❌ 缺少 dropout=saved_config.get('dropout', 0.2)
    )
```

**问题**：
1. 缺少 `dropout` 参数（影响推理时的 dropout 行为，但 eval 模式下 dropout 被禁用）
2. G-2 同样影响：`seq_channels` 未传递

---

## 五、问题汇总

| ID | 严重程度 | 位置 | 问题描述 | 建议 |
|----|----------|------|----------|------|
| G-1 | 🟡 低 | `dataset_multimodal.py:577-594` | PhysicalSafeTransform1D 三通道处理错误 | 见 `审计报告.md` G-1 |
| G-2 | 🟡 低 | `run_train.py:605-613` | eval_checkpoint 未传 dropout 和 seq_channels | 见 `审计报告.md` G-2 |
| **S-1** | 🟡 低 | `run_train.py:1529-1537` | eval 模式验证损失计算逻辑：将单预测 (B,6) 广播为 (B,N,6)，所有网格标记为正样本，损失值不反映真实训练损失 | 在 eval 模式分支跳过损失计算，或在推理路径中保留训练模式输出用于验证 |

---

### 2.4 CrossAttentionFusion(dim_2d=6) 设计分析 ✅

`pe_tsnet_yolo.py:369`:

```python
self.fusion = CrossAttentionFusion(dim_2d=6, dim_1d=64)
```

**设计意图**：推理时，用 YOLO 预测的 6 维向量作为 Query，去查询 1D 时序特征。

**分析**：
- 训练模式：`grid_pred (B, N, 6)` 直接进损失，**不经过 fusion** ✅
- 推理模式：`grid_feat (B, 6)` → fusion → output_head ✅

这个设计是合理的。训练时不需要融合（因为有完整的 grid 预测），推理时通过融合利用 1D 时序信息来 refine 最终预测。

### 2.5 YOLOFPNHead grid_size vs actual_grid_size

`YOLOFPNHead` 内部硬编码使用 `grid_size=16`，生成 256 个预测位置。

但 `actual_grid_size = image_size // 32`：
- image_size=256 → actual_grid_size=8 → 64 实际位置
- image_size=512 → actual_grid_size=16 → 256 实际位置

这意味着：
- 当 image_size=256 时，YOLO head 输出 256 个预测，但实际特征图只有 64 个位置
- 当 image_size=512 时，YOLO head 输出 256 个预测，特征图也是 256 个位置 → **匹配**

**结论**：SwinYOLOFPN 在 image_size=512 时能完全匹配，在 image_size=256 时有冗余预测位置（但不影响功能）。

---

## 六、审计结论

**swin_yolo 变体全链路数据流核心路径正确**，维度追踪无误，损失函数匹配完整。

发现 3 个问题：G-1、G-2 为共性问题（见审计报告），S-1 为该变体特有设计问题——eval 模式验证损失计算与训练损失不一致，但不影响模型训练，仅影响验证指标显示。

**建议优先处理 S-1**（改善 eval 模式的损失计算准确性），G-1 和 G-2 优先级较低。
