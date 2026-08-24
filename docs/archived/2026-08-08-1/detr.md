# A4: detr 变体数据流审计报告

> 审计分组：A组（基线变体）
> 模型类型：DETRStyle
> 审计日期：2026/08/08

---

## 一、阶段1：入口与配置

### 1.1 参数映射 ✅

| 检查项 | 代码位置 | 状态 |
|--------|----------|------|
| `--variant detr` → `DETRStyle` | `run_train.py:362` VARIANT_MODELS | ✅ |
| `is_new_variant` 包含 detr | `run_train.py:418, 1267, 1294, 1337, 1518` | ✅ |
| 损失函数选择 DETRLoss + HungarianMatcher | `run_train.py:1196-1198` | ✅ |

### 1.2 模型实例化参数 ⚠️ G-2

`run_train.py:605-613`:
```python
model = ModelClass(
    seq_len=300,
    image_channels=2,
    image_size=image_size,
    pretrained_2d=True
    # ❌ 缺少 dropout=saved_config.get('dropout', 0.2)
    # ❌ 缺少 seq_channels（虽然 DETR 目前不支持 triple_channel，但应预留）
)
```

---

## 二、阶段2：模型内部数据流

### 2.1 forward 维度追踪 ✅

```
输入:
  x_1d: (batch, 300)
  x_2d: (batch, 2, 256, 256)

2D分支:
  x_2d → ResNet18Backbone2D
    → set_spatial_output(True)
    → feat_2d: (B, 512, 8, 8) ✅
  → input_proj: 512→512 → (B, 512, 8, 8)
  → pos_encoder: 2D Sinusoidal PE
  → flatten: (B, 64, 512)
  → transformer_encoder: (B, 64, 512)
  → decoder: (B, num_queries=100, 512)
  → detr_head: (B, 100, 6)

1D分支:
  x_1d → TemporalFeatureExtractor → feat_1d: (B, 64) ✅

融合（推理模式）:
  best_query: (B, 6)
  → query_proj: 6→128
  → CrossAttentionFusion(128, 64) → fused: (B, 192)
  → MultiTaskHead(192) → output: (B, 6) ✅

维度追踪无问题。
```

### 2.2 训练/推理模式区分 ✅

`DETRStyle.forward` (`pe_tsnet_detr.py:381-405`):

```python
if self.training:
    return detr_pred, global_density  # (B, 100, 6), (B, 1) ✅
else:
    # 推理：取最高 conf → query_proj → fusion → output_head
    output, global_density
```

干跑验证正确设置 `model.train()`（`run_train.py:1248`），DETR 在干跑中返回训练模式输出。

### 2.3 CrossAttentionFusion(dim_2d=128) ✅

`DETRStyle.__init__`:
```python
self.query_proj = nn.Linear(6, 128)      # 6 → 128
self.fusion = CrossAttentionFusion(dim_2d=128, dim_1d=64)  # ✅
fused_dim = 128 + 64 = 192
```

设计正确：先用 `query_proj` 将 6D 预测投影到 128 维，再与 1D 特征融合。

---

## 三、阶段3：损失函数匹配

### 3.1 DETRLoss 初始化 ✅

`run_train.py:1196-1198`:
```python
matcher = HungarianMatcher()
criterion = DETRLoss(matcher=matcher)
```

### 3.2 DETRLoss.forward 参数对齐 ✅

`DETRLoss.forward` (`mono_loss.py:1009-1085`):
```python
def forward(self, pred, target, global_density, target_density,
            indices=None, final_output=None):
```

训练循环调用 (`run_train.py:1434-1435`):
```python
indices = matcher(grid_pred, {'labels': labels})
loss_total = criterion(grid_pred, {'labels': labels}, global_density,
                        labels[:, 5:6], indices, final_output)
```

**参数完全对齐。**

### 3.3 eval 模式处理 ⚠️ DETR eval 分支修复

`run_train.py:1541-1552`:

```python
# DETR eval: 临时切回训练模式获取全部 query 预测 ✅
model.train()
detr_full_pred, global_density = model(seq_1d, img_2d)  # (B, 100, 6)
indices = matcher(detr_full_pred, {'labels': labels})
loss_total = criterion(detr_full_pred, {'labels': labels}, ...)

model.eval()  # 恢复 eval 模式
model_output, _ = model(seq_1d, img_2d)  # (B, 6)
```

**设计正确**。DETR eval 分支通过 `model.train()` 获取完整 query 预测，避免了 swin_yolo 的 eval 损失不一致问题。

---

## 四、阶段4：数据与评估

### 4.1 evaluate_model 解包逻辑 ✅

`run_train.py:430-435`:
```python
outputs, global_density = raw_outputs  # (B, 6), (B, 1) ✅
if outputs.dim() == 3:
    outputs = outputs[:, 0, :]  # 不执行（推理模式返回 (B, 6)）
```

### 4.2 eval_checkpoint 模型创建 ⚠️ G-2（同上）

---

## 五、问题汇总

| ID | 严重程度 | 位置 | 问题描述 | 建议 |
|----|----------|------|----------|------|
| G-1 | 🟡 低 | `dataset_multimodal.py:577-594` | PhysicalSafeTransform1D 三通道处理错误 | 见 `审计报告.md` G-1 |
| G-2 | 🟡 低 | `run_train.py:605-613` | eval_checkpoint 未传 dropout | 见 `审计报告.md` G-2 |
| D-1 | 🟡 低 | `pe_tsnet_detr.py:276-290` | PositionalEncoding2D 固定初始化 feat_size=8，但 ResNet18 实际输出与 image_size 相关 | 动态初始化 PE，或移除固定值 |

---

## 六、审计结论

**DETR 变体全链路数据流正确**，核心问题最少。eval 模式的处理设计合理（通过 `model.train()` 获取完整 query 预测），避免了 swin_yolo 的 eval 损失不一致问题。

发现 1 个该变体特有低优先级问题（D-1）。
