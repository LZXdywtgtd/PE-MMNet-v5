# B1: gated fusion 审计报告

> 审计分组：B组（单项优化）
> 触发方式：`--variant resnet18 --fusion gated`
> 审计日期：2026/08/08

---

## 一、Flag 行为追踪

### 1.1 配置传递链

```
run_train.py 解析 args.fusion
  ↓ config['fusion']
  ↓ create_variant_model()
  ↓ PETSNetMultimodal(fusion=config.get('fusion', 'cross_attn'))
  ↓ PETSNetMultimodal.__init__
  ↓ if fusion == 'gated': GatedMultimodalFusion()
```

### 1.2 模型内部行为

`pe_tsnet_multimodal.py:1236-1241`：

```python
if fusion == 'gated':
    self.fusion = GatedMultimodalFusion(
        dim_2d=feat_dim_2d,
        dim_1d=feat_dim_1d,
        split_ratio=0.5
    )
else:
    self.fusion = CrossAttentionFusion(...)
fused_dim = feat_dim_2d + feat_dim_1d  # 512 + 64 = 576
```

**参数全部正确传递。**

---

## 二、发现严重问题

### ⚠️ **B1-1 CRITICAL: GatedMultimodalFusion 输出维度与 MultiTaskHead 输入不匹配**

**位置**：`models/pe_tsnet_fusion.py:160-187` + `pe_tsnet_multimodal.py:1250, 1255`

**问题分析**：

| 组件 | 输出维度 | 期望输入维度 | 是否匹配 |
|------|---------|-------------|---------|
| `GatedMultimodalFusion` | `(B, 64)` | - | - |
| `MultiTaskHead` | - | `input_dim=576` | ❌ |
| `CrossAttentionFusion` | `(B, 576)` | `input_dim=576` | ✅ |

**根本原因**：

`GatedMultimodalFusion.forward`:
```python
def forward(self, feat_2d, feat_1d):
    # ...
    fused = gate_temp * temp_out + gate_stress * stress_out  # (B, dim_1d) = (B, 64)
    return fused  # ❌ 只返回 64 维！
```

而 `PETSNetMultimodal.__init__`:
```python
fused_dim = feat_dim_2d + feat_dim_1d  # 512 + 64 = 576  ← 硬编码为 576
self.output_head = MultiTaskHead(input_dim=fused_dim)  # 创建 576 维输入头
```

**后果**：`model(x_1d, x_2d)` 会抛出：

```
RuntimeError: mat1 and mat2 shape mismatch: mat1 (B, 64), mat2 (576, 256)
```

训练会直接崩溃。

**修复建议**：

方案A（推荐）：修改 `GatedMultimodalFusion` 输出维度：
```python
def forward(self, feat_2d, feat_1d):
    temp_out = self.temp_attn(temp_feat, feat_1d)  # (B, dim_1d)
    stress_out = self.stress_attn(stress_feat, feat_1d)  # (B, dim_1d)
    gate_weight = self.gate(feat_1d)
    gated = gate_temp * temp_out + gate_stress * stress_out  # (B, dim_1d)
    # 将原始 2D 特征与门控结果拼接
    return torch.cat([feat_2d, gated], dim=-1)  # (B, 512 + 64) = (B, 576)
```

方案B：修改 `PETSNetMultimodal.__init__` 根据 fusion 类型设置不同 `fused_dim`：
```python
if fusion == 'gated':
    fused_dim = feat_dim_1d  # 64
else:
    fused_dim = feat_dim_2d + feat_dim_1d  # 576
```

---

## 三、eval_checkpoint 兼容性

`run_train.py:596-604`:

```python
model = PETSNetMultimodal(
    seq_len=300,
    image_channels=2,
    image_size=image_size,
    pretrained_2d=True,
    task=task,
    fusion=saved_config.get('fusion', 'cross_attn')  # ✅ 正确读取
    # ❌ 缺少 seq_channels=saved_config.get('seq_channels', 1)
    # ❌ 缺少 dropout=saved_config.get('dropout', 0.2)
)
```

**eval_checkpoint 对 `fusion` 参数处理正确**（G-2 中的 dropout/seq_channels 缺失不影响 gated 逻辑）。

---

## 四、与其他 flag 的交互

### B1 + B2 (gated + coord_attn)

- `coord_attn` 包装 `model.branch_2d`
- `gated` 使用 `self.fusion = GatedMultimodalFusion`
- **两者独立，无冲突**

### B1 + B3 (gated + staged_train)

- `staged_train` 的 `staged_training()` 函数创建模型时**未传递 `fusion` 参数**：
```python
model_kwargs = dict(
    seq_len=..., image_channels=2, image_size=...,
    pretrained_2d=True, dropout=config['dropout'],
)
# ❌ 缺少 fusion=config.get('fusion', 'cross_attn')
if variant_key == 'resnet18':
    model_kwargs['task'] = config.get('task', 'detection')
```
- **问题**：staged_train 使用 gated fusion 时会崩溃！
- 同样缺少 `seq_channels` 参数

### B1 + B4 (gated + triple_channel)

- `triple_channel` 影响 `TemporalFeatureExtractor(input_dim=seq_channels)`
- `GatedMultimodalFusion` 不涉及序列通道数，无直接冲突
- 但 `staged_training()` 中的 B1+B3 问题会传递到 B1+B4

### B1 + B5 (gated + aug_cutmix)

- `aug_cutmix` 在 `collate_fn_with_cutmix` 中处理，不影响模型结构
- **无冲突**

---

## 五、问题汇总

| ID | 严重程度 | 位置 | 问题描述 | 建议 |
|----|----------|------|----------|------|
| **B1-1** | 🔴 **严重** | `pe_tsnet_fusion.py:185` | `GatedMultimodalFusion` 输出 `(B, 64)` 但 `MultiTaskHead` 期望 576 维 | 立即修复：拼接 `feat_2d + gated` 输出 576 维 |
| G-2 | 🟡 低 | `run_train.py:596-604` | eval_checkpoint 缺少 seq_channels 和 dropout | 见审计报告 G-2 |
| **B1-2** | 🟡 低 | `run_train.py:744` | staged_train 未传递 fusion 参数给模型 | 修复 staged_training() 中的 model_kwargs |

---

## 六、审计结论

**B1 gated fusion 存在致命 bug（B1-1）**：直接运行 `--fusion gated` 会因维度不匹配崩溃。

**B1-2**（staged_train 不传 fusion）在 B1+B3 组合时会连锁触发。

建议修复优先级：**P0（B1-1 立即修复）> P1（B1-2）> P2（G-2）**
