# A3: vit_yolo 变体数据流审计报告

> 审计分组：A组（基线变体）
> 模型类型：ViTYOLOFPN
> 审计日期：2026/08/08

---

## 一、与 swin_yolo 的差异点

ViTYOLOFPN 与 SwinYOLOFPN 的架构差异：

| 组件 | swin_yolo | vit_yolo |
|------|-----------|----------|
| 2D 骨干 | Swin-Backbone2D (多尺度) | ViTYOLOBackbone2D (固定 16×16 输出) |
| FPN | 跳过（直接用 C4） | 简化为单层 3×3 卷积 |
| grid_size | 固定 16 | 固定 16 |
| 实际网格 | `image_size // 32` | 固定 16 |
| 其他 | 相同 | 相同 |

### 1.1 2D 骨干输出固定 16×16 ✅

`ViTYOLOBackbone2D` (`pe_tsnet_yolo.py:119-188`):
```python
self.output_size = output_size  # 默认 16
self.out_size = 256  # 投影后固定 256 通道

def forward(self, x):
    # ViT 输出 → resize 到 output_size × output_size
    feat = F.interpolate(feat, size=(self.output_size, self.output_size), ...)
    feat = self.proj(feat)  # → (B, 256, 16, 16)
    return feat
```

无论 `image_size` 是多少，骨干网络都输出固定的 16×16 特征图。

### 1.2 grid_size 与实际网格一致 ✅

`ViTYOLOFPN.__init__`:
```python
self.actual_grid_size = 16        # 固定 ✅
self.actual_num_grids = 256       # 固定 ✅
```

assigner 初始化（`run_train.py:1194`）:
```python
actual_grid_size = 16  # vit_yolo 走 else 分支 ✅
assigner = YOLOTargetAssigner(grid_size=16, nearby_range=2)
```

**结论**：vit_yolo 不存在 swin_yolo 的 S-1 问题（assigner 网格尺寸与模型输出一致）。

---

## 二、阶段3：损失函数匹配

### 2.1 YOLOLoss 调用 ✅

vit_yolo 与 swin_yolo 使用相同的 YOLOLoss，但 assigner 用 `actual_grid_size=16`：

```python
if variant_key in ['swin_yolo', 'swin_yolo_patchtst']:
    actual_grid_size = model.actual_grid_size  # swin: image_size // 32
else:
    actual_grid_size = 16  # vit_yolo: 固定 16 ✅
```

**vit_yolo 的 assigner 网格尺寸与模型输出一致**，S-1 问题不存在。

---

## 三、继承/特有新问题

| ID | 严重程度 | 问题 | 说明 |
|----|----------|------|------|
| V-1 | 🟡 低 | ViTYOLOBackbone2D 固定输出 16×16，与 image_size 无关 | 语义上不直观，但不影响功能 |
| S-2 | 🟡 低 | eval 模式验证损失计算语义不一致 | 继承自 swin_yolo |
| G-1 | 🟡 低 | PhysicalSafeTransform1D 三通道处理错误 | 共性 |
| G-2 | 🟡 低 | eval_checkpoint 未传 dropout | 共性 |

---

## 四、审计结论

**vit_yolo 是所有 YOLO 变体中问题最少的**，不存在 swin_yolo 的 S-1（网格尺寸不匹配）问题。

仅发现 V-1（设计上的不直观）和继承的 S-2 问题。
