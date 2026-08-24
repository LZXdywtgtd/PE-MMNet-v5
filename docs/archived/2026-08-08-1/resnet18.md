# A1: resnet18 变体数据流审计报告

> 审计分组：A组（基线变体）
> 模型类型：PETSNetMultimodal
> 审计日期：2026/08/08

---

## 一、阶段1：入口与配置

### 1.1 参数映射 ✅

| 检查项 | 代码位置 | 状态 | 说明 |
|--------|----------|------|------|
| `--variant resnet18` → `PETSNetMultimodal` | `run_train.py:351-357` VARIANT_MODELS | ✅ | `'resnet18': PETSNetMultimodal` |
| `--task detection` → config['task'] | `run_train.py:2212` | ✅ | config 正确传递到 train_variant |
| `--fusion cross_attn` → PETSNetMultimodal | `run_train.py:1839` | ✅ | `fusion=config.get('fusion', 'cross_attn')` |
| `--triple_channel` → seq_channels | `run_train.py:1831-1840` | ✅ | `seq_channels = 3 if config.get('triple_channel', False) else 1` |
| `--predict_offset` → 检查点路径 | `run_train.py:1797-1823` | ✅ | 影响检查点文件名含 offset |
| `--feature_len` → seq_len | `run_train.py:1833` | ✅ | `seq_len=config.get('feature_len', 300)` |
| `--dropout` → arch_config | `run_train.py:2194` | ✅ | `dropout: arch_config['dropout']` |
| `--image_size` → auto_select_config | `run_train.py:2208` | ✅ | `image_size: auto_config['image_size']` |

### 1.2 模型实例化参数 ✅

`run_train.py:1829-1841`:

```python
if variant_key == 'resnet18':
    seq_channels = 3 if config.get('triple_channel', False) else 1
    return PETSNetMultimodal(
        seq_len=config.get('feature_len', 300),   # ✅
        image_channels=2,                           # ✅ 固定
        image_size=config['image_size'],            # ✅
        pretrained_2d=True,                         # ✅ 固定
        dropout=config['dropout'],                  # ✅
        task=task,                                  # ✅
        fusion=config.get('fusion', 'cross_attn'), # ✅
        seq_channels=seq_channels                   # ✅
    )
```

所有参数均正确传递，**无遗漏**。

---

## 二、阶段2：模型内部数据流

### 2.1 `__init__` 签名 ✅

`models/pe_tsnet_multimodal.py:1185-1193`:

```python
def __init__(self,
             seq_len=300,
             image_channels=2,
             image_size=256,
             pretrained_2d=True,
             dropout=0.2,
             task='detection',
             fusion='cross_attn',
             seq_channels=1):
```

所有从 run_train.py 传入的参数均有对应接收，**无缺失参数**。

### 2.2 forward 维度追踪 ✅

```
输入:
  x_1d: (batch, 300)  或 (batch, 3, 300) triple_channel
  x_2d: (batch, 2, H, W)

分支1 (2D):
  x_2d → ResNet18Backbone2D
    - conv1 修改: 适配 2/1/多通道输入，权重自动扩展 ✅
    - forward → feat_2d: (batch, 512) ✅

分支2 (1D):
  x_1d → TemporalFeatureExtractor
    - MicroCNN (3,32) + MacroCNN (7,32 dilation=3)
    - concat → (batch, 64, seq_len)
    - LayerNorm → MultiHead Self-Attn (4 heads)
    - FFN → GlobalPool
    - feat_1d: (batch, 64) ✅
    - 支持 seq_channels=1 或 3 ✅

融合:
  CrossAttentionFusion(512, 64):
    - 2D→1D 交叉注意力: Q=feat_2d, K/V=feat_1d → (batch, 512)
    - 1D→2D 交叉注意力: Q=feat_1d, K/V=feat_2d → (batch, 64)
    - 拼接: (batch, 576) ✅
  GatedMultimodalFusion(512, 64):
    - 温度/应力拆分 → 注意力加权 → 门控融合
    - 输出: (batch, 576) ✅

输出头:
  fused (576) → MultiTaskHead
    - shared: 576→256→128
    - head_xy: 128→2 (ReLU)
    - head_lw: 128→2 (ReLU)
    - head_confidence: 128→1 (Sigmoid)
    - head_density: 128→1 (Sigmoid)
    - output: (batch, 6): [x, y, l, w, conf, density] ✅
```

**维度追踪无问题，融合层输出 576 与输出头输入 576 完全匹配。**

### 2.3 BackboneWithAttention 包装器 ✅

`run_train.py:2002-2008`:

```python
if use_coord_attn and hasattr(model, 'branch_2d'):
    model.branch_2d = BackboneWithAttention(
        model.branch_2d,
        attention_type='coord',
        reduction=16
    )
```

`BackboneWithAttention.forward` (`pe_tsnet_multimodal.py:193-212`):
- 调用原始 backbone → 获取 feat
- 若 feat 为列表，取最后一层
- 若 feat 为 4D → 应用注意力 → 全局池化 → (batch, dim)
- ✅ 正确处理

---

## 三、阶段3：损失函数匹配

### 3.1 任务 → 损失函数选择 ✅

`run_train.py:1165-1205`:

| task | 损失函数 | 代码位置 |
|------|---------|---------|
| `detection` | `MultimodalCrackLoss` | run_train.py:1200 |
| `segmentation` | `SegmentationLoss` | run_train.py:1175 |
| `multitask` | `MultimodalSegmentationLoss` | run_train.py:1177 |

### 3.2 MultimodalCrackLoss 输入对齐 ✅

调用处 (`run_train.py:1445`):
```python
loss_total, _ = criterion(outputs, labels)
# outputs: (batch, 6) → [x, y, l, w, conf, density]
# labels:  (batch, 6) → [x, y, l, w, conf, density]
```

损失函数拆分 (`mono_loss.py:483-559`):
```python
pred_xy = pred[:, 0:2]      # (batch, 2) ✅
pred_lw = pred[:, 2:4]      # (batch, 2) ✅
pred_conf = pred[:, 4:5]    # (batch, 1) ✅
pred_density = pred[:, 5:6] # (batch, 1) ✅
```

**损失函数与 forward 输出完全对齐，无错配。**

### 3.3 单调性损失验证 ✅

`MonotonicityLossV3.forward` (`mono_loss.py:234-269`):
- 输入: `pred_density: (batch, 1)` → squeeze 到 `(batch,)`
- 计算相邻差分: `diff = pred_density[1:] - pred_density[:-1]`
- 惩罚负差分: `violation = clamp(-diff, min=0)`
- ✅ 物理原理：裂纹密度随热应力增加单调递增，不可逆

---

## 四、阶段4：数据与评估

### 4.1 collate_fn 处理 ✅

`dataset_multimodal.py:1291-1336` `collate_fn_with_cutmix`:

```python
seq_1d = torch.stack(seq_1d_list)   # 普通: (B, 300) ✅
                                   # triple_channel: (B, 3, 300) ✅
img_2d = torch.stack(img_2d_list)   # (B, 2, H, W) ✅

# multitask:
if isinstance(labels_list[0], tuple):
    mask_list, detection_list = zip(*labels_list)
    labels = (torch.stack(mask_list), torch.stack(detection_list))
else:
    labels = torch.stack(labels_list)
```

**堆叠逻辑正确，无遗漏。**

### 4.2 evaluate_model 返回值处理 ✅

`run_train.py:436-438`:
```python
is_new_variant = variant_key in ['swin_yolo', 'vit_yolo', 'detr', 'swin_yolo_patchtst']
# resnet18 不在列表中 → 不解包
outputs = raw_outputs      # (batch, 6)
global_density = None
```

`run_train.py:440-443`:
```python
if task == 'detection':
    all_preds.append(outputs.cpu().numpy())           # ✅ (batch, 6)
    all_targets.append(labels[:, :6].cpu().numpy())   # ✅ (batch, 6)
```

**检测任务正确收集 6 维输出用于指标计算。**

### 4.3 eval_checkpoint 模型创建 ⚠️ G-2

`run_train.py:596-604`:

```python
model = PETSNetMultimodal(
    seq_len=300,
    image_channels=2,
    image_size=image_size,
    pretrained_2d=True,
    task=task,
    fusion=saved_config.get('fusion', 'cross_attn')
    # ❌ 缺少 seq_channels=saved_config.get('seq_channels', 1)
)
```

**影响**：如果原始训练使用了 `--triple_channel`（`seq_channels=3`），评估时模型会创建为 `seq_channels=1`，权重加载时维度不匹配。

---

## 五、问题汇总

| ID | 严重程度 | 位置 | 问题描述 | 建议 |
|----|----------|------|----------|------|
| G-1 | 🟡 低 | `dataset_multimodal.py:577-594` | `PhysicalSafeTransform1D` 对三通道输入处理错误：`len(seq)` 返回通道数 3 而非序列长度 300，导致噪声被加到通道维度 | 见 `审计报告.md` G-1 |
| G-2 | 🟡 低 | `run_train.py:596-604` | `eval_checkpoint` 创建 resnet18 模型时未传递 `seq_channels`，无法正确还原 triple_channel 训练的模型 | 见 `审计报告.md` G-2 |

---

## 六、审计结论

**resnet18 变体全链路数据流核心路径正确**，维度追踪无误，损失函数匹配完整。

发现 2 个低优先级问题（G-1, G-2），均与 `--triple_channel` 配置相关，不影响默认配置（`seq_channels=1`）下的训练和评估。
