# 更新日志 (CHANGELOG) — v5

> v5 独立版本线。从 v4.6.12 fork。
>
> 格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)。

---

## [v5.0.4-alpha] - 2026-08-12

### 新增（V5A4）

5 个 v5-α 模型变体（输出头已替换为 SplineHead + GP 嵌入）：

- `models/pe_tsnet_multimodal_v5.py`：`PETSNetMultimodalV5`
  - ResNet-18 + CrossAttentionFusion + SplineHead + GP（默认启用）
  - 输入：3 通道（温度场+应力场+热力图），符合 v5-α 设计
- `models/pe_tsnet_yolo_v5.py`：
  - `SwinYOLOFPNV5` — Swin-Tiny + YOLO-FPN
  - `ViTYOLOFPNV5` — ViT-Small + YOLO-FPN
  - `ViTYOLOBackbone2DV5` — 自适应输入通道版（v4 硬编码 2 通道 → v5-α 1/2/3/4 通道）
- `models/pe_tsnet_detr_v5.py`：`DETRStyleV5` — ResNet-18 + Transformer Encoder-Decoder
- `models/pe_tsnet_patchtst_v5.py`：`SwinYOLOFPNWithPatchTSTV5` — Swin + YOLO + PatchTST 1D
- `models/__init__.py`：注册 `V5_MODEL_REGISTRY` + 工厂函数 `create_v5_model(variant, **kwargs)`
- `tests/test_pe_tsnet_v5.py`：18 项单元测试

### 关键改动

| 维度 | v4 | v5-α |
|---|---|---|
| 输出 | 6 维向量 / YOLO 网格 6 维 / DETR queries 6 维 | bbox + 8-16 动态关键点（dict） |
| 输出头 | MultiTaskHead / YOLOFPNHead / DETRHead | SplineHead（统一） |
| 物理约束 | 单调性损失 | GP 嵌入模块（嵌入特征流）+ GP LML 正则 |
| 输入通道 | 2（温度+应力） | 3（温度场+应力场+热力图） |
| 训练接口 | `model(x_1d, x_2d) → Tensor` | `model(x_1d, x_2d) → dict` |

### 设计决策

**复用 v4 骨干，不重写**：
- 5 个变体的 2D 骨干（ResNet-18 / Swin-Tiny / ViT-Small）和 1D 骨干（TemporalFeatureExtractor / PatchTST1D）全部沿用 v4
- 仅替换输出头 + 可选 GP 模块插入融合后特征
- 总代码量减半，单测覆盖完整数据流

**DETR 的 top-1 query 选择**：
- 沿用 v4 推理链：`DETRHead → (B, num_queries, 6) → 取最高 conf → query_proj → fusion`
- v5-α 训练时也用同一 query（保持 train/eval 一致性，避免 query 选择 抖动）

**YOLO 变体的全局池化**：
- v4 输出 `(B, num_grids, 6)`（网格级预测）
- v5-α 用 `mean(dim=1)` 池化到 `(B, 6)`（全局特征），再接 1D 拼接
- 简化版聚合策略，避免在 v5-α 阶段处理 grid-to-kpts 的复杂映射

### 测试

- 单元测试合计：**72/72 PASSED**
  * V5A1 patch_simulator: 12 项
  * V5A2 SplineHead: 8 项 + GPModule: 10 项
  * V5A3 ordered_kp_loss: 24 项
  * V5A4 pe_tsnet_v5: **18 项**（新增：5 变体 forward + backward + 端到端 loss 反向）
- 端到端验证：所有 5 个变体都能跑 `model → OrderedKeypointLoss → backward` 全链路

### 已知问题新增

- V5-015：GP 模块 batch 内归一化（V5-014）已在 v5-α 模型集成中保留，**未升级到 running stats**
  → 训练/推理 stats 不一致风险持续存在
  → 解决路径：V5A5 trainer 集成时改用 `(running_mean, running_std)`

---

## [v5.0.3-alpha] - 2026-08-12

### 新增（V5A3）

v5-α 主损失函数（覆盖距离 + 组合 loss）：

- `training/ordered_kp_loss.py`：v5-α 核心损失
  * `catmull_rom_spline_torch(points, num_samples)`：纯 PyTorch 三次样条拟合
    - 端点镜像延拓 ghost points（保持 C¹ 连续）
    - K=2 退化为线性插值
    - 完全在 PyTorch 内实现，可对控制点求梯度（autograd 通路畅通）
  * `coverage_loss(pred_kpts, true_pixels, M, mode)`：
    - "chamfer"（默认，mean over true pixels of min distance）— 平滑，适合训练
    - "hausdorff"（max over true pixels of min distance）— 严格，评估用
  * `poisson_prior(pred_kpts)`：段长方差最小化（鼓励等弧长分布）
  * `OrderedKeypointLoss` 组合 loss：
    ```
    total = 1.0  * coverage
          + 1.0  * bbox (Smooth L1)
          + 0.1  * gp (-log_marginal_likelihood)
          + 0.05 * poisson
    ```
  * 返回 dict：`{total, coverage, bbox, gp, poisson}`（含 detached 监控值）
- `tests/test_ordered_kp_loss.py`：24 项单元测试（全部通过）

### 关键设计决策

**为何选 Catmull-Rom 而非 scipy CubicSpline？**

- `scipy.interpolate.CubicSpline` 不可微（基于 numpy，无 autograd 通路）
- Catmull-Rom 是 Hermite 三次样条（C¹ 连续），完全在 PyTorch 内实现
- 对每个 segment (P_{i-1}, P_i, P_{i+1}, P_{i+2}) 做三次插值：用户文档中"三次 B 样条"即指此
- 端点用镜像延拓 ghost points（保持端点切线连续）

### 测试

- 单元测试合计：54/54 PASSED
  * V5A1 patch_simulator: 12 项
  * V5A2 SplineHead: 8 项
  * V5A2 GPModule: 10 项
  * V5A3 ordered_kp_loss: **24 项**（新增）
- 关键验证点：
  * 完美匹配：真值像素 = 样条采样点 → coverage loss ≈ 0（实测 0.00e+00）
  * chamfer ≤ hausdorff（同数据）实测：0.044 ≤ 0.270 ✓
  * 反向传播：pred_kpts / pred_bbox / GP 超参数梯度全部正常

### 已知问题新增

（无新增 V5-α 阻塞问题；V5-014 仍为待 V5A4 集成处理）

---

## [v5.0.2-alpha] - 2026-08-12

### 新增（V5A2）

模型输出头与物理约束模块：

- `models/spline_head.py`：Spline 输出头
  * 输入：(B, in_dim) 融合特征
  * 输出：bbox (B, 4) + 关键点 (B, max_kpts, 2) + validity (B, max_kpts) + K (B,)
  * K 动态范围 [min_kpts, max_kpts] = [8, 16]
  * bbox 自动满足 x1≤x2, y1≤y2（重新参数化）
  * `predict_kpts_only(sort_by="x"|"validity")` 推理便捷接口
- `models/gp_module.py`：GP 嵌入模块
  * 可微 RBF 核平滑（ℓ、σ² 可学习）
  * log_marginal_likelihood 正则项
  * LayerNorm 稳定输出
  * 非可学习模式（buffer）支持
- `tests/test_spline_head.py`：8 项单元测试（值域、bbox 顺序、反向传播、推理接口）
- `tests/test_gp_module.py`：10 项单元测试（核对称、LML、超参数梯度、batch=1 边界）

### 关键修复

**GP 模块 K 矩阵退化为单位矩阵**：

- 现象：默认 in_dim=128 时，||x_i - x_j||² ~ 128，远超默认 ℓ=1.0，
  导致 K ≈ I（单位矩阵），GP 平滑失效
- 修复：`_normalize_for_kernel` 改为按 L2 范数归一化（除以 sqrt(in_dim)），
  使 ||x_i - x_j||² ~ O(1)，ℓ=1.0 物理意义清晰
- 影响：修复后 GP 模块对超参数 ℓ/σ² 都有非零梯度（之前 ≈ 0）

### 测试

- 单元测试合计：30/30 PASSED
  * V5A1 patch_simulator: 12 项
  * V5A2 SplineHead: 8 项
  * V5A2 GPModule: 10 项

### 已知问题新增

- V5-014：GP 模块依赖 batch 内 L2 归一化（无 running stats）
  → 训练/推理 stats 不一致风险（v5-α MVP 可接受）

---

## [v5.0.1-alpha] - 2026-08-12

### 新增（V5A1）

patch 仿真器升级（S1–S4 + S7）：

- `data/thermal_profile.py`（S4）：三段升降温曲线生成器
- `data/surface_radiation.py`（S2）：Stefan-Boltzmann 表面辐射
- `data/viscoelastic.py`（S3）：Maxwell 粘弹性应力模型
- `data/patch_simulator_v5.py`（S1 + S7 + 集成）：主仿真器
- `tools/patch_simulator_visualizer.py`：4 联可视化
- `tests/test_patch_simulator_v5.py`：12 项单元测试（全过）

### 物理修复

Maxwell update_step 公式多了一个 `(dt/τ)` 因子，导致累计应力被压缩4 个数量级（应 360 MPa，实际 0.02 MPa）。修正为标准离散化形式：

```
旧：σ_next = σ + dt·[(E/τ)·α·ΔT − σ/τ]
新：σ_next = σ + E·α·ΔT − (dt/τ)·σ
```

### 验证

- 单元测试：12/12 PASSED
- 端到端冒烟：64×64 patch, peak_σ = 59.2 MPa, K = 8 关键点
- 可视化：output/v5_sim_sample.png + v5_stress_evolution.png

### 已知问题新增

- V5-011：低阈值下 crack_mask 占满 patch（88% 覆盖率）
- V5-012：v5-α 裂纹萌生判据是简化版（未用 Griffith S5 + LEFM S6）
- V5-013：Maxwell 显式 Euler 在 dt > τ/2 时 clip 到 0

---

## [v5.0.0-alpha] - 2026-08-12

### Fork

- **基线版本**：v4.6.12
- **Fork 范围**：完整复制 `project_v4/` 到 `project_v5/`（跳过 checkpoints/logs/__pycache__/.git/benchmark_results）
- **目录关系**：v5 与 v4 并列（同级），v4 不动

### 范围声明

**v5-α 严格实现范围**：
- ✅ 256×256 虚拟 patch 输入（形状无关）
- ✅ 3 模态融合（温度场 + 应力场 + 热力图）
- ✅ 5 模型变体保留（resnet18 / swin_yolo / vit_yolo / detr / swin_yolo_patchtst）
- ✅ 8–16 动态有序关键点 + bbox 输出
- ✅ B 样条 + Hausdorff 覆盖距离损失
- ✅ 高斯过程嵌入模块（GP module）
- ✅ 泊松过程先验
- ✅ patch 仿真器升级（S1–S4：多方向热交换 + 辐射 + 应力松弛 + 三段曲线）
- ✅ 像素级裂纹 mask 输出（GT 配套）

**v5-α 严格不实现范围**（v5-β 文档占位）：
- ❌ 真实 3D 物体（杯/碗/盘等）
- ❌ UV 展开 + 多角度拍摄管线
- ❌ 真实烧制数据采集流程
- ❌ 位置图像 L0/L1 完整角点检测管线
- ❌ crack_annotator 标注工具
- ❌ 3D FEM 仿真

详细见 [docs/v5_范围说明.md](docs/v5_范围说明.md)。

### 任务清单（v5-α）

| 任务 | 名称 | 状态 |
|---|---|---|
| V5A1 | patch 仿真器升级（S1–S4） | 待启动 |
| V5A2 | Spline 输出头 + GP 模块 | 待启动 |
| V5A3 | 覆盖距离 loss | 待启动 |
| V5A4 | 5 变体输出头迁移 | 待 V5A2, V5A3 |
| V5A5 | 变长训练循环 | 待 V5A4 |
| V5A6 | 5 变体训练 + 评估 | 待 V5A1, V5A5 |

详细见 [tasks/team_v5_alpha.json](tasks/team_v5_alpha.json)。

### 已知问题

见 [docs/v5_已知问题.md](docs/v5_已知问题.md)。

---

## 迁移指南

### 从 v4 升级到 v5-α

**目录关系**：`project_v5/` 与 `project_v4/` 并列，不互相依赖。

**配置**：v5 使用独立 `config.json`，新增 `patch_size`、`kiln_dims`、`min_keypoints`、`max_keypoints` 等字段。

**数据**：v5-α 使用仿真生成的 patch 数据，无需 v4 真实数据。

**训练命令**（规划）：
```bash
cd D:\team_project\projects\pe_mmnet\project_v5
python run_train_v5.py --variant resnet18 --epochs 150 --min_kpts 8 --max_kpts 16
```

---

## 未来版本

### [v5.1.0] - v5-β（未规划时间）

- 真实 3D 物体接入
- UV 展开管线
- 多角度拍摄与 crack_annotator 工具
- 位置图像完整管线（L0/L1）
- 3D FEM 仿真
- 真实烧制数据采集 + GT 标注