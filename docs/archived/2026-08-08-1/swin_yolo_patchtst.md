# A5: swin_yolo_patchtst 变体数据流审计报告

> 审计分组：A组（基线变体）
> 模型类型：SwinYOLOFPNWithPatchTST
> 审计日期：2026/08/08

---

## 一、与 swin_yolo 的差异点

swin_yolo_patchtst 与 swin_yolo 的架构几乎完全相同，唯一差异：

| 组件 | swin_yolo | swin_yolo_patchtst |
|------|-----------|-------------------|
| 1D 骨干 | `TemporalFeatureExtractor` (CNN+SelfAttn) | `PatchTST1D` (Transformer) |
| 其他组件 | 完全相同 | 完全相同 |

### 1.1 PatchTST1D 输入格式 ⚠️ 发现问题

`PatchTST1D.forward` (`pe_tsnet_patchtst.py:108-146`):
```python
def forward(self, x):
    # x 期望: (B, seq_len)
    batch_size = x.size(0)
    x = x.view(batch_size, self.num_patches, self.patch_size)  # 假设 x 是 1D
```

**潜在问题**：`PatchTST1D` 只接受 `(B, seq_len)` 格式，不支持 `(B, 3, seq_len)` 三通道输入。

在 `SwinYOLOFPNWithPatchTST` 中：
```python
feat_1d = self.backbone_1d(x_1d)  # x_1d 可能是 (B, 3, 300)
```

`TemporalFeatureExtractor` 支持 `(B, 3, 300)`（会 `unsqueeze` 到 `(B, 1, seq_len)`），但 `PatchTST1D` 不支持。

**触发条件**：在 swin_yolo_patchtst 上使用 `--triple_channel`
**后果**：`view(batch_size, self.num_patches, self.patch_size)` 会失败（维度不匹配）

**严重程度：低**（目前未使用 triple_channel + swin_yolo_patchtst 组合）。

---

## 二、继承自 swin_yolo 的问题

| ID | 严重程度 | 问题 | 来源 |
|----|----------|------|------|
| S-1 | ⚠️ 严重 | assigner 使用 `actual_grid_size` 而非 `grid_size` | swin_yolo |
| S-2 | 🟡 低 | eval 模式验证损失计算语义不一致 | swin_yolo |
| G-1 | 🟡 低 | PhysicalSafeTransform1D 三通道处理错误 | 共性 |
| G-2 | 🟡 低 | eval_checkpoint 未传 dropout/seq_channels | 共性 |
| P-1 | 🟡 低 | PatchTST1D 不支持三通道输入 | 特有 |

---

## 三、审计结论

**swin_yolo_patchtst 继承 swin_yolo 的所有问题**，额外发现 1 个 PatchTST1D 三通道不支持的问题。

**建议优先修复 S-1**（严重影响损失计算正确性），P-1 在启用 triple_channel 训练前需修复。
