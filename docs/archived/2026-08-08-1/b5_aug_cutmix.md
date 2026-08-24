# B5: aug_cutmix_prob 审计报告

> 审计分组：B组（单项优化）
> 触发方式：`--variant resnet18 --aug_cutmix_prob 0.3`
> 审计日期：2026/08/08

---

## 一、Flag 行为追踪

### 1.1 配置传递链

```
run_train.py 解析 args.aug_cutmix_prob
  ↓ config['aug_cutmix_prob']
  ↓ create_multibatch_dataloaders(cutmix_prob=config['aug_cutmix_prob'])
  ↓ collate_fn_with_cutmix(batch, cutmix_prob)
```

### 1.2 cutmix 触发逻辑

`data/dataset_multimodal.py:1291-1336` `collate_fn_with_cutmix`:

```python
def collate_fn_with_cutmix(batch):
    features_and_labels = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    seq_1d_list, img_2d_list = zip(*features_and_labels)
    seq_1d_list, img_2d_list = list(seq_1d_list), list(img_2d_list)

    cutmix_applied = [False] * len(batch)
    # 触发条件：概率 > 0 + augment=True + batch >= 2 + 随机命中
    if cutmix_prob > 0 and augment and len(batch) >= 2 and random.random() < cutmix_prob:
        idx1, idx2 = random.sample(range(len(batch)), 2)
        lambda_ = random.betavariate(1.0, 1.0)

        img1 = img_2d_list[idx1]
        img2 = img_2d_list[idx2]

        mixed_img = img1.clone()
        mixed_img[0] = img1[0] * lambda_ + img2[0] * (1 - lambda_)  # 只混合温度通道
        # 应力通道保持不变

        mixed_labels = lambda_ * labels_list[idx1] + (1 - lambda_) * labels_list[idx2]

        img_2d_list[idx1] = mixed_img
        labels_list[idx1] = mixed_labels
        cutmix_applied[idx1] = True

    # 堆叠
    seq_1d = torch.stack(seq_1d_list)
    img_2d = torch.stack(img_2d_list)

    if isinstance(labels_list[0], tuple):
        mask_list, detection_list = zip(*labels_list)
        labels = (torch.stack(mask_list), torch.stack(detection_list))
    else:
        labels = torch.stack(labels_list)
```

**设计分析**：
- 批量级别混合：每批次最多一次
- 仅混合温度通道（`img[0]`），保持应力通道（`img[1]`）不变 ✅ 物理安全
- 标签线性插值：`mixed_labels = lambda_ * l1 + (1 - lambda_) * l2` ✅

---

## 二、潜在问题分析

### 2.1 批量大小为 1 时 cutmix 永不触发

```python
if len(batch) >= 2 and random.random() < cutmix_prob:
```

当 `batch_size=1` 时，`len(batch) >= 2` 为 False，cutmix 不触发。

**这不是 bug**，而是合理的保守设计（需要两个样本才能混合）。

### 2.2 lambda 采样

```python
lambda_ = random.betavariate(1.0, 1.0)
```

Beta(1,1) 是均匀分布，结果在 [0,1] 上均匀。

**设计合理** ✅

### 2.3 标签插值 vs 标签选择

```python
mixed_labels = lambda_ * labels_list[idx1] + (1 - lambda_) * labels_list[idx2]
```

标签是**连续插值**而非**离散选择**。对于检测任务（6 维连续值），这是合理的。

**但注意**：对于分类任务（如裂纹存在/不存在），线性插值标签可能不合理。当前任务使用连续值，无此问题。

---

## 三、与其他 flag 的交互

### B5 + B1 (aug_cutmix + gated)

- cutmix 在 collate_fn 中处理，不影响模型结构
- **无冲突** ✅

### B5 + B2 (aug_cutmix + coord_attn)

- cutmix 在 collate_fn 中处理，不影响模型结构
- **无冲突** ✅

### B5 + B3 (aug_cutmix + staged_train)

- **B3-2**：`staged_training()` 的数据加载器未传递 `cutmix_prob`
- B5+B3 组合中，阶段1和阶段2的 cutmix 都不生效 ⚠️
- 修复 B3-2 后自动解决

### B5 + B4 (aug_cutmix + triple_channel)

- cutmix 混合 2D 图像（`img_2d_list[idx1]`），不影响 1D 序列
- **无冲突** ✅
- B5+B4+B3 组合：cutmix 不生效（受 B3-2 影响）

---

## 四、eval 模式处理

### 4.1 eval 时 cutmix 被禁用

```python
if cutmix_prob > 0 and augment and len(batch) >= 2 and random.random() < cutmix_prob:
```

`create_multibatch_dataloaders` 创建 test_loader 时：

```python
test_dataset = MultiBatchCollateDataset(
    ...,
    augment=False,  # eval 时 augment=False
    ...
)
```

`collate_fn_with_cutmix` 使用外层 `augment` 参数，eval 模式下为 False。

**结论**：eval 模式正确禁用 cutmix ✅

### 4.2 eval_checkpoint 不涉及 cutmix

`eval_checkpoint` 直接加载模型推理，不使用 DataLoader 的 collate_fn。

**无相关问题** ✅

---

## 五、问题汇总

| ID | 严重程度 | 位置 | 问题描述 | 建议 |
|----|----------|------|----------|------|
| B3-2 | 🟡 低 | `run_train.py:720-731, 763-774` | staged_training() 未传递 cutmix_prob，staged_train 组合中 cutmix 不生效 | 添加 cutmix_prob 参数 |

---

## 六、审计结论

**B5 aug_cutmix_prob 设计正确**：

1. cutmix 在 collate_fn 批量级别混合，不影响模型结构
2. 仅混合温度通道（物理安全），保持应力通道不变
3. eval 模式正确禁用 cutmix
4. 与其他 flag 无直接冲突

**唯一问题**是 B3-2（staged_train 中 cutmix 不生效），这是 staged_train 的问题而非 cutmix 本身。
