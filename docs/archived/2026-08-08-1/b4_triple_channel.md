# B4: triple_channel 审计报告

> 审计分组：B组（单项优化）
> 触发方式：`--variant resnet18 --triple_channel`
> 审计日期：2026/08/08

---

## 一、Flag 行为追踪

### 1.1 配置传递链

```
run_train.py 解析 args.triple_channel
  ↓ config['triple_channel'] = True
  ↓ seq_channels = 3 if config.get('triple_channel') else 1
  ↓ PETSNetMultimodal(seq_channels=seq_channels)
  ↓ TemporalFeatureExtractor(input_dim=seq_channels)
  ↓ collate_fn: torch.stack 生成 (B, 3, 300)
```

### 1.2 数据流追踪

```
数据加载 (dataset_multimodal.py):
  seq_1d: (B, 3, 300)  ← 三通道时序
    ↓ torch.stack
  DataLoader batch

collate_fn:
  seq_1d = torch.stack(seq_1d_list)  # (B, 3, 300) ✅

模型输入:
  PETSNetMultimodal.forward(x_1d=(B,3,300), x_2d=(B,2,H,W))
    ↓ branch_1d(x_1d)
  TemporalFeatureExtractor(x_1d=(B,3,300))
```

---

## 二、TemporalFeatureExtractor 对三通道输入的处理

`pe_tsnet_multimodal.py:714-725`:

```python
def forward(self, x):
    # 输入调整：(batch, seq_len) → (batch, 1, seq_len)
    if x.dim() == 2:
        x = x.unsqueeze(1)  # (B, 300) → (B, 1, 300)

    # 双分支并行卷积
    micro_feat = self.micro_branch(x)    # Conv1d(in=1, out=32)
    macro_feat = self.macro_branch(x)    # Conv1d(in=1, out=32)
```

**问题**：`TemporalFeatureExtractor` 只处理 `(B, 1, seq_len)` 输入！

- 当 `x.dim() == 2`（即 `(B, 300)`）时，`unsqueeze(1)` 变成 `(B, 1, 300)` ✅
- 但当 `x.dim() == 3`（即 `(B, 3, 300)`）时，直接用 `micro_branch` 处理 3 通道输入

`Micro1DCNN` 定义（`pe_tsnet_multimodal.py:581`）:
```python
class Micro1DCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=32):
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
```

`TemporalFeatureExtractor.__init__`:
```python
self.micro_branch = Micro1DCNN(in_channels=input_dim, ...)  # input_dim=3
self.macro_branch = Macro1DCNN(in_channels=input_dim, ...)  # input_dim=3
```

**结论**：`TemporalFeatureExtractor` 正确处理三通道输入 ✅

- `input_dim=3` → Micro/Macro CNN 期望 3 通道
- `x.dim() == 3` 时不触发 `unsqueeze`，直接用 `(B, 3, 300)` 处理

---

## 三、发现的问题

### 3.1 G-1: PhysicalSafeTransform1D 三通道处理错误

**位置**：`data/dataset_multimodal.py:577-594`

```python
class PhysicalSafeTransform1D:
    def __call__(self, seq):
        seq = seq.copy()
        if np.random.random() < 0.3:
            seq_len = len(seq)   # 对 (3, 300) 返回 3，而非 300 ❌
            mask_start = np.random.randint(0, seq_len - mask_len)  # 报错或静默错误
            seq[mask_start:mask_start + mask_len] = 0
        seq += np.random.randn(len(seq)).astype(np.float32) * self.noise_std  # 形状错误 ❌
        return seq
```

**触发条件**：`--triple_channel` + `augment=True`

**后果**：
- `len(seq)` 返回 3（通道数）而非 300（序列长度）
- `mask_start = np.random.randint(0, 3 - mask_len)` 大概率为 0
- `seq += np.random.randn(3)` 与 `(3, 300)` 形状不匹配
- 可能触发广播错误或静默产生错误数据

### 3.2 G-2: eval_checkpoint 缺少 seq_channels

见审计报告 G-2。

### 3.3 B3-1: staged_train 缺少 seq_channels

见 B3-1。

---

## 四、与其他 flag 的交互

### B4 + B1 (triple_channel + gated)

- `GatedMultimodalFusion` 不涉及序列通道数，无直接冲突 ✅
- 但 B1-1（维度不匹配）会导致 triple_channel + gated 直接崩溃

### B4 + B2 (triple_channel + coord_attn)

- `coord_attn` 只影响 2D 分支，不影响 1D 序列处理 ✅
- 但 B2-1（CoordAtt 无效）可能掩盖 triple_channel 的效果

### B4 + B3 (triple_channel + staged_train)

- B3-1（staged_training 缺少 seq_channels）导致 B4+B3 组合崩溃 ⚠️
- 修复 B3-1 后自动解决

### B4 + B5 (triple_channel + aug_cutmix)

- cutmix 只混合 2D 图像，不影响 1D 序列 ✅
- B3-2（staged 中 cutmix 未传递）影响 B4+B5+B3 组合

---

## 五、问题汇总

| ID | 严重程度 | 位置 | 问题描述 | 建议 |
|----|----------|------|----------|------|
| G-1 | 🟡 低 | `dataset_multimodal.py:577-594` | PhysicalSafeTransform1D 对 (3, 300) 处理错误：len(seq)=3，噪声加到通道维度 | 重写 `__call__` 支持多通道输入，对 `seq[..., -mask_len:]` 加噪 |
| G-2 | 🟡 低 | `run_train.py:596-604` | eval_checkpoint 缺少 seq_channels | 将 seq_channels 写入检查点，eval 时读取传递 |
| B3-1 | 🔴 严重 | `run_train.py:733-746` | staged_training() 缺少 seq_channels | 补充参数 |

---

## 六、审计结论

**B4 triple_channel 本身的数据流正确**（TemporalFeatureExtractor 正确处理三通道），但存在两个依赖问题：

1. **G-1**：数据增强在 triple_channel 下产生错误数据，影响训练质量
2. **B3-1**：staged_train 组合会崩溃（独立于 triple_channel）

triple_channel + cross_attn 组合是**可用的**（无 G-1 时），但 G-1 会悄悄破坏训练。
