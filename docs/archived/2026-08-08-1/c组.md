# C组：组合优化审计报告

> 审计分组：C组（组合优化）
> 审计日期：2026/08/08

---

## C组概览

| 组合 | flags | 训练路径 | 继承问题 |
|------|-------|---------|---------|
| C1 | gated + coord_attn | `train_variant()` | B1-1, B2-1 |
| C2 | staged_train + coord_attn | `staged_training()` | B3-1, B2-2, B2-1 |
| C3 | 全组合 | `staged_training()` | 所有 B组问题 |

---

## C1: gated + coord_attn

### 配置传递链

```
--variant resnet18 --fusion gated --use_coord_attn
  ↓ train_variant(variant_key='resnet18', config, ..., use_coord_attn=True)
  ↓ create_variant_model() → PETSNetMultimodal(fusion='gated')
  ↓ CoordAtt 包装 model.branch_2d（注入到已创建的模型）
```

### 继承问题

| 来源 | 问题 | 严重程度 | 是否继承 |
|------|------|---------|---------|
| B1-1 | GatedMultimodalFusion 输出 64 维 vs MultiTaskHead 期望 576 维 | 🔴 崩溃 | ✅ **直接崩溃** |
| B2-1 | CoordAtt 对 resnet18 无效（backbone 已池化） | ⚠️ 静默 | ✅ 但无额外影响 |

### 实际运行行为

```
C1 运行流程：
  model = PETSNetMultimodal(fusion='gated')
    → self.fusion = GatedMultimodalFusion(dim_2d=512, dim_1d=64)
    → fused_dim = 512 + 64 = 576  ← 错误！
    → self.output_head = MultiTaskHead(input_dim=576)

  model.branch_2d = BackboneWithAttention(model.branch_2d, 'coord')
    → 包装器创建成功（但 CoordAtt 不生效）

  第一次 forward:
    feat_2d = (B, 512)
    feat_1d = (B, 64)
    fused = self.fusion(feat_2d, feat_1d)  → (B, 64)  ← 输出 64 维！
    output = self.output_head(fused)        → RuntimeError: mat1 (B, 64), mat2 (576, 256)
```

**结论**：C1 直接崩溃于 B1-1，修复 B1-1 后 CoordAtt 仍不生效。

### 问题汇总

| ID | 严重程度 | 来源 | 问题描述 |
|----|----------|------|----------|
| **C1-1** | 🔴 崩溃 | B1-1 | GatedMultimodalFusion 维度不匹配，直接崩溃 |
| C1-2 | ⚠️ 静默 | B2-1 | CoordAtt 不生效（同 B2-1） |

---

## C2: staged_train + coord_attn

### 配置传递链

```
--variant resnet18 --staged_train --use_coord_attn
  ↓ main() 检测 args.staged_train=True
  ↓ staged_training('resnet18', config, device, data_roots)
  ↓ create_variant_model() → PETSNetMultimodal() ← 无 fusion 参数！
  ↓ 训练阶段1 → freeze_backbone → 训练阶段2
  ↓ use_coord_attn 在 staged_training 中未处理
```

### 继承问题

| 来源 | 问题 | 严重程度 | 是否继承 |
|------|------|---------|---------|
| B3-1 | staged_training() 缺少 fusion、seq_channels、task 参数 | 🔴 崩溃 | ✅ **阶段1崩溃** |
| B2-2 | staged_training() 不处理 coord_attn | 🟡 失效 | ✅ |
| B2-1 | CoordAtt 对 resnet18 无效 | ⚠️ 静默 | ✅ 但无额外影响 |
| B3-2 | staged 数据加载器缺少 cutmix_prob | 🟡 失效 | ✅ |

### 实际运行行为

```
C2 运行流程：
  model = PETSNetMultimodal(
    seq_len=300, image_channels=2, image_size=256,
    pretrained_2d=True, dropout=0.2
    # ❌ 缺少 fusion='cross_attn'（默认）、seq_channels=1、task='detection'
  )

  阶段1训练：
    → 正常（fusion 默认 cross_attn，不崩溃）
    → CoordAtt 未注入（B2-2）

  阶段2冻结：
    freeze_model_backbone(model, freeze_2d=True)
    → 冻结 branch_2d.* 和 backbone_2d.* 参数
    → 如果 CoordAtt 存在，其参数也会被冻结（因为 name 包含 'branch_2d'）
    → 但由于 B2-2，CoordAtt 根本没注入，不影响

  阶段2训练：
    → 正常（fusion 默认 cross_attn）
```

**结论**：C2 **不崩溃**（因为 fusion 默认 cross_attn），但 coord_attn 完全不生效。

### 问题汇总

| ID | 严重程度 | 来源 | 问题描述 |
|----|----------|------|----------|
| C2-1 | ⚠️ 失效 | B2-2 | staged_train 不处理 coord_attn，C2 中 CoordAtt 不生效 |
| C2-2 | ⚠️ 静默 | B2-1 | 即使注入 CoordAtt 也无效（同 B2-1） |
| C2-3 | ⚠️ 失效 | B3-2 | cutmix 在 staged 模式不生效（同 B3-2） |

---

## C3: 全组合（gated + coord_attn + staged_train）

### 配置传递链

```
--variant resnet18 --fusion gated --use_coord_attn --staged_train
  ↓ main() 检测 args.staged_train=True
  ↓ staged_training('resnet18', config, device, data_roots)
  ↓ create_variant_model() → PETSNetMultimodal() ← 无 fusion 参数！
```

### 继承问题

| 来源 | 问题 | 严重程度 | 是否继承 |
|------|------|---------|---------|
| B3-1 | staged_training() 缺少 fusion 参数 | 🔴 崩溃 | ✅ **阶段1用 cross_attn 不崩溃** |
| B1-1 | GatedMultimodalFusion 维度不匹配 | 🔴 崩溃 | ✅ **但 staged 不走 gated** |

### 实际运行行为

```
C3 运行流程：
  model = PETSNetMultimodal(
    seq_len=300, image_channels=2, image_size=256,
    pretrained_2d=True, dropout=0.2
    # ❌ 缺少 fusion='cross_attn'（默认）、seq_channels=1、task='detection'
  )
  # 使用默认 fusion='cross_attn'，不崩溃！

  阶段1训练：cross_attn，正常
  阶段2冻结：正常
  阶段2训练：cross_attn，正常

  CoordAtt 未注入（B2-2）
```

### ⚠️ 发现新问题：C3 实际使用 cross_attn 而非 gated

**问题**：C3 配置中指定了 `--fusion gated`，但 `staged_training()` 不传递 `fusion` 参数，导致模型使用默认的 `cross_attn`。

**结论**：C3 名为"全组合"，实际运行的是 `--staged_train`（cross_attn + coord_attn 不生效）。

### 问题汇总

| ID | 严重程度 | 来源 | 问题描述 |
|----|----------|------|----------|
| C3-1 | 🔴 配置不符 | B3-1 | C3 名为 gated，实际用 cross_attn |
| C3-2 | ⚠️ 失效 | B2-2 | coord_attn 不生效（同 B2-2） |
| C3-3 | ⚠️ 静默 | B2-1 | 即使注入 CoordAtt 也无效（同 B2-1） |
| C3-4 | ⚠️ 失效 | B3-2 | cutmix 不生效（同 B3-2） |

---

## C组问题索引

| 组合 | 文档 | 严重问题 | 低优先级问题 |
|------|------|----------|-------------|------|
| C1 (gated+coord_attn) | 本文件 C1 节 | **C1-1** 🔴 崩溃 | C1-2 |
| C2 (staged+coord_attn) | 本文件 C2 节 | 0（不崩溃） | C2-1, C2-2, C2-3 |
| C3 (全组合) | 本文件 C3 节 | **C3-1** 🔴 配置不符 | C3-2, C3-3, C3-4 |

---

## C组综合结论

### C组整体问题图谱

```
C1: gated + coord_attn
    → B1-1 直接崩溃
    → B2-1 CoordAtt 不生效

C2: staged + coord_attn
    → B3-1 不崩溃（fusion 默认 cross_attn）
    → B2-2 CoordAtt 不注入
    → B2-1 即使注入也不生效

C3: 全组合（gated + staged + coord_attn）
    → B3-1 覆盖 B1-1（staged 不走 gated）
    → C3 实际运行 cross_attn + staged + coord_attn 不生效
    → 最接近"全组合"实际效果的是 C2
```

### 关键发现

1. **C1 是唯一一个真正会崩溃的 C组组合**（直接走 gated fusion）
2. **C2 和 C3 都不崩溃**（staged 默认用 cross_attn）
3. **所有 C组都存在 coord_attn 不生效的问题**（B2-1 + B2-2 叠加）
4. **C3 名为"全组合"，实际运行内容与 C2 几乎相同**

### C组修复优先级

| 优先级 | 问题 | 修复方案 |
|--------|------|----------|
| **P0** | B1-1 | 修复 GatedMultimodalFusion 维度（影响 C1） |
| **P0** | B3-1 | staged_training 补充所有参数（影响 C2, C3） |
| **P1** | B2-2 | staged_training 添加 coord_attn 处理（影响 C2, C3） |
| **P2** | B2-1 | CoordAtt 对 resnet18 无效问题（影响所有含 coord_attn 的组合） |

---

## 附录：C组与 B组问题继承关系图

```
                B1-1 (gated 崩溃)
                      ↓
          ┌───────────┴───────────┐
          ↓                       ↓
       C1 崩溃                C3 用 cross_attn（不崩溃但非预期）
     (直接 gated)           (staged 覆盖 gated)

                B2-1 (CoordAtt 无效)
                      ↓
          ┌───────────┴───────────┐
          ↓                       ↓
       C1 CoordAtt             C2/C3 CoordAtt
       不生效                  不生效（B2-2 叠加）
```

---

## 审计结论

**C组是 B组问题叠加的放大器**：

1. **C1** 直接暴露 B1-1（崩溃）和 B2-1（无效）
2. **C2** 中 B3-1 的"保守设计"反而防止了崩溃，但完全丢失了 gated 和 coord_attn 的效果
3. **C3** 的实际运行内容与 team_train.py 中描述的 C3 任务目标严重不符——用户看到 `--fusion gated --use_coord_attn --staged_train`，但实际只有 staged_train 在生效

**建议**：
- 修复 B1-1 和 B3-1 后再启用 C组任务
- C3 需在 team_train.py 中说明其实际行为（staged + coord_attn 不生效）
