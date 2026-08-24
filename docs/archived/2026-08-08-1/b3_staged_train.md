# B3: staged_train 审计报告

> 审计分组：B组（单项优化）
> 触发方式：`--variant resnet18 --staged_train`
> 审计日期：2026/08/08

---

## 一、Flag 行为追踪

### 1.1 配置传递链

```
run_train.py 解析 args.staged_train
  ↓ config['staged_train'] = True
  ↓ train_variant() 调用 staged_training()
  ↓ train_variant() 直接返回 staged_training() 结果
```

### 1.2 阶段流程

`run_train.py:679-786` `staged_training()`:

```
阶段1 (短序列预训练):
  数据: short_batches (默认 ['单次扫描'])
  Epochs: min(50, config['epochs'])
  模型: 初始化 → train_model()

阶段2 (长序列微调):
  数据: 全部批次
  Epochs: config['epochs'] - phase1_epochs
  模型: freeze_backbone → train_model()
```

---

## 二、发现严重问题

### ⚠️ **B3-1 CRITICAL: staged_training() 模型创建缺少多个关键参数**

**位置**：`run_train.py:733-746`

```python
ModelClass = VARIANT_MODELS.get(variant_key, PETSNetMultimodal)
model_kwargs = dict(
    seq_len=config.get('feature_len', 300),
    image_channels=2,
    image_size=config['image_size'],
    pretrained_2d=True,
    dropout=config['dropout'],
)
# ❌ 缺少以下参数：
# - fusion=config.get('fusion', 'cross_attn')
# - seq_channels=3 if config.get('triple_channel') else 1
# - task=config.get('task', 'detection') ← 只对 resnet18 条件添加

if variant_key == 'resnet18':
    model_kwargs['task'] = config.get('task', 'detection')
model = ModelClass(**model_kwargs)
```

**缺少的参数及其后果**：

| 参数 | 缺失后果 | 触发条件 |
|------|---------|---------|
| `fusion` | 模型使用默认 `cross_attn` 而非 `gated` | `--fusion gated --staged_train` |
| `seq_channels` | triple_channel 模型崩溃（维度不匹配） | `--triple_channel --staged_train` |
| `task` | multitask 模型缺少 mask_decoder | `--task multitask --staged_train` |

---

## 三、冻结逻辑

### 3.1 freeze_model_backbone 实现 ✅

`run_train.py:631-675`:

```python
def freeze_model_backbone(model, freeze_2d=True, freeze_1d=False):
    for name, param in model.named_parameters():
        if freeze_2d and is_2d_backbone(name):
            param.requires_grad = False
        if freeze_1d and is_1d_backbone(name):
            param.requires_grad = False
```

**兼容命名**：
- PETSNetMultimodal: `branch_2d`, `branch_1d`
- SwinYOLOFPN/DETR: `backbone_2d`, `backbone_1d`

### 3.2 冻结时机

`run_train.py:755-760`:

```python
if config.get('freeze_backbone', True):
    freeze_2d = config.get('freeze_2d', True)
    freeze_1d = config.get('freeze_1d', False)
    print_info(f"冻结策略: freeze_2d={freeze_2d}, freeze_1d={freeze_1d}")
    model = freeze_model_backbone(model, freeze_2d=freeze_2d, freeze_1d=freeze_1d)
```

**设计合理**：默认冻结 2D 骨干（保留微调能力），1D 继续训练。

---

## 四、数据加载器差异

### 4.1 阶段1 vs 阶段2

| 属性 | 阶段1 | 阶段2 |
|------|-------|-------|
| 数据根目录 | `short_batches`（仅 `['单次扫描']`） | 全部批次 |
| seq_len | `config.get('feature_len', 300)` | 同上 |
| triple_channel | `config.get('triple_channel', False)` | 同上 ✅ |
| cutmix_prob | ❌ 未传递（阶段1数据加载器无此参数） | ❌ 未传递 |

### 4.2 ⚠️ 发现问题：cutmix_prob 未传递

**位置**：`run_train.py:720-731, 763-774`

阶段1和阶段2的 `create_multibatch_dataloaders` 调用都**缺少 `cutmix_prob` 参数**：

```python
train_loader_1, test_loader_1 = create_multibatch_dataloaders(
    data_roots=phase1_roots,
    batch_size=config['batch_size'],
    ...
    triple_channel=config.get('triple_channel', False)
    # ❌ 缺少 cutmix_prob=config.get('aug_cutmix_prob', 0.0)
)
```

**后果**：`--aug_cutmix_prob --staged_train` 组合中，cutmix 在 staged_train 中被禁用。

---

## 五、与其他 flag 的交互

### B3 + B1 (staged + gated)

- **B1-2**：`staged_training()` 未传递 `fusion`，导致 B3+B1 实际使用 `cross_attn`
- 修复 B3-1 后自动解决

### B3 + B2 (staged + coord_attn)

- **B2-2**：`staged_training()` 未注入 CoordAtt，导致 B2+B3 CoordAtt 不生效
- 修复 B3-1 后需单独处理 B2-2

### B3 + B4 (staged + triple_channel)

- `create_multibatch_dataloaders` 中 `triple_channel` 参数正确传递 ✅
- 但 `staged_training()` 模型创建缺少 `seq_channels`（B3-1）
- 修复 B3-1 后自动解决

### B3 + B5 (staged + aug_cutmix)

- `staged_training()` 中 cutmix_prob 未传递给数据加载器（B3-2）
- **B3+B5 组合 cutmix 完全不生效**

---

## 六、问题汇总

| ID | 严重程度 | 位置 | 问题描述 | 建议 |
|----|----------|------|----------|------|
| **B3-1** | 🔴 **严重** | `run_train.py:733-746` | staged_training() 模型创建缺少 fusion、seq_channels、task 参数 | 补充所有参数，与 create_variant_model() 对齐 |
| **B3-2** | 🟡 低 | `run_train.py:720-731, 763-774` | 阶段1/阶段2数据加载器未传递 cutmix_prob | 添加 cutmix_prob 参数 |
| **B3-3** | 🟡 低 | `run_train.py:733-760` | staged_training() 不处理 coord_attn | 添加 CoordAtt 注入逻辑 |

---

## 七、审计结论

**B3 staged_train 是所有 B组问题中受影响最广的**：

1. **B3-1**（CRITICAL）导致 B1/B4 在 staged_train 下实际行为与预期不符
2. **B3-2** 导致 cutmix 在 staged_train 下不生效
3. **B3-3** 导致 coord_attn 在 staged_train 下不生效

建议修复优先级：**P0（B3-1 立即修复）> P1（B3-2, B3-3）**
