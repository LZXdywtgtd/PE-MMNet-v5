# B2: coord_attn 审计报告

> 审计分组：B组（单项优化）
> 触发方式：`--variant resnet18 --use_coord_attn`
> 审计日期：2026/08/08

---

## 一、Flag 行为追踪

### 1.1 配置传递链

```
run_train.py 解析 args.use_coord_attn
  ↓ config['use_coord_attn']
  ↓ train_variant() 中 post-model creation 注入
  ↓ BackboneWithAttention(model.branch_2d, attention_type='coord')
```

### 1.2 注入时机

`run_train.py:1999-2009`:

```python
model = create_variant_model()
model = model.to(device)

# 坐标注意力集成
if use_coord_attn and hasattr(model, 'branch_2d'):
    from models.pe_tsnet_multimodal import BackboneWithAttention
    model.branch_2d = BackboneWithAttention(
        model.branch_2d,
        attention_type='coord',
        reduction=16
    )
    print_info(f"[OK] 坐标注意力已启用")
```

**注入位置**：模型创建后、训练前，以包装器方式注入，不修改原始模型类。

---

## 二、BackboneWithAttention 行为

### 2.1 CoordAtt 实现

`pe_tsnet_multimodal.py:73-138`:

```python
class CoordAtt(nn.Module):
    def forward(self, x):
        # x: (B, C, H, W)
        identity = x
        n, c, h, w = x.size()

        # 高度方向编码
        x_h = self.pool_h(x)  # (B, C, H, 1)
        # 宽度方向编码
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # (B, C, 1, W) → (B, C, W, 1)

        y = torch.cat([x_h, x_w], dim=2)  # (B, C, H+1, 1)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()  # (B, C, H, 1)
        a_w = self.conv_w(x_w).sigmoid()  # (B, C, 1, W)

        return identity * a_w * a_h
```

### 2.2 BackboneWithAttention.forward

`pe_tsnet_multimodal.py:193-212`:

```python
def forward(self, x):
    feat = self.backbone(x)

    # 处理多尺度特征
    if isinstance(feat, (list, tuple)):
        feat = feat[-1]

    # 应用注意力
    if feat.dim() == 4:
        feat = self.attention(feat)  # CoordAtt
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
    return feat
```

**维度追踪**：
```
输入: (B, 2, H, W)
  ↓ backbone (ResNet18Backbone2D)
feat: (B, 512) ← 全局平均池化已在 backbone 内部完成
  ↓ attention 检查
feat.dim() == 2（不是4），**不进入 CoordAtt 分支！**
  ↓ 直接返回 (B, 512)
```

### 2.3 ⚠️ 发现问题：CoordAtt 被跳过

**位置**：`pe_tsnet_multimodal.py:193-212`

`ResNet18Backbone2D` 内部已完成全局平均池化，输出 `(B, 512)` 而非 `(B, 512, H, W)`。

`BackboneWithAttention.forward` 中：
```python
if feat.dim() == 4:
    feat = self.attention(feat)  # 只有 4D 才进入
    feat = F.adaptive_avg_pool2d(feat, (1, 1)).view(feat.size(0), -1)
```

**结果**：CoordAtt 根本不会被调用！包装器注入成功但无效。

---

## 三、与 staged_train 的交互

### 3.1 问题：B2 + B3 组合

`staged_training()` 函数：
1. 创建模型时**不使用** `use_coord_attn`
2. 训练完阶段1后直接 `train_model(phase2)`

```python
# run_train.py:744 - staged_training() 中没有 coord_attn 逻辑！
if variant_key == 'resnet18':
    model_kwargs['task'] = config.get('task', 'detection')
model = ModelClass(**model_kwargs)
```

**结果**：
- `--variant resnet18 --use_coord_attn --staged_train` 不会应用 CoordAtt
- 需要在 `staged_training()` 中添加与 `train_variant()` 相同的 CoordAtt 注入逻辑

### 3.2 问题：B2 + B3 + B1 组合

`staged_training()` 还缺少 `fusion` 参数（B1-2）。

---

## 四、问题汇总

| ID | 严重程度 | 位置 | 问题描述 | 建议 |
|----|----------|------|----------|------|
| **B2-1** | 🟡 低 | `pe_tsnet_multimodal.py:209` | `BackboneWithAttention` 对已池化的特征无效：ResNet18Backbone2D 输出 `(B, 512)`，CoordAtt 需要 `(B, C, H, W)` | 修改 `ResNet18Backbone2D.set_spatial_output(True)` 输出特征图，或修改包装器逻辑 |
| **B2-2** | 🟡 低 | `run_train.py:733-746` | `staged_training()` 不处理 `use_coord_attn`，B2+B3 组合 CoordAtt 不生效 | 在 `staged_training()` 的模型创建后添加 CoordAtt 注入逻辑 |

---

## 五、审计结论

**B2 coord_attn 存在设计问题**：

1. **B2-1**：CoordAtt 包装器对 resnet18 无效（backbone 已池化）。代码不会报错，但注意力机制完全没被调用——等于白跑。

2. **B2-2**：staged_train 不支持 coord_attn，B2+B3/C2 组合 CoordAtt 不生效。

**注意**：B2-1 不是"崩溃"而是"静默无效"。如果团队成员用 `--use_coord_attn` 训练，模型看起来正常但实际上没有使用坐标注意力。

建议修复优先级：**P1（B2-2）> P2（B2-1 需重新设计）**
